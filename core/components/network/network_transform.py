# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
import math
from collections import deque
from enum import Enum
from core.ecs.ecs import Component, ComponentRegistry
from core.maths.math3d import Vec3, Quat
from core.components.inspector_meta import FieldType, InspectorField


class TransformAuthority(Enum):
    SERVER = "server"
    OWNER = "owner"


@ComponentRegistry.register
class NetworkTransform(Component):
    _icon = "NetworkTransform.png"
    _gizmo_icon_color = (80, 220, 120)
    _gizmo_icon_label = "T"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("sync_position", "Sync Position", FieldType.BOOL),
            InspectorField("sync_rotation", "Sync Rotation", FieldType.BOOL),
            InspectorField("sync_scale", "Sync Scale", FieldType.BOOL),
            InspectorField("authority", "Authority", FieldType.ENUM, enum_class=TransformAuthority),
            InspectorField("send_rate", "Send Rate", FieldType.FLOAT, min_val=1.0, max_val=60.0),
            InspectorField("pos_threshold", "Pos Threshold", FieldType.FLOAT, min_val=0.0, max_val=5.0),
            InspectorField("rot_threshold", "Rot Threshold", FieldType.FLOAT, min_val=0.0, max_val=30.0),
            InspectorField("interpolate", "Interpolate", FieldType.BOOL),
            InspectorField("interp_delay", "Interp Delay", FieldType.FLOAT, min_val=0.0, max_val=0.5),
            InspectorField("teleport_threshold", "Teleport Threshold", FieldType.FLOAT, min_val=0.1, max_val=100.0),
        ]

    def __init__(self):
        super().__init__()
        self.sync_position: bool = True
        self.sync_rotation: bool = True
        self.sync_scale: bool = False
        self.authority: TransformAuthority = TransformAuthority.SERVER
        self.send_rate: float = 20.0
        self.pos_threshold: float = 0.001
        self.rot_threshold: float = 0.5
        self.scale_threshold: float = 0.001
        self.interpolate: bool = True
        self.interp_delay: float = 0.1
        self.teleport_threshold: float = 5.0
        self._buffer: deque = deque(maxlen=24)
        self._last_sent_pos: Vec3 = Vec3.zero()
        self._last_sent_rot: Quat = Quat.identity()
        self._last_sent_scale: Vec3 = Vec3.one()
        self._send_accum: float = 0.0
        self._has_sent_initial: bool = False

    def _get_identity(self):
        ent = self._entity
        if ent is None:
            return None
        return ent.get_component_by_name("NetworkIdentity")

    def _can_send(self) -> bool:
        ident = self._get_identity()
        if ident is None or ident.net_id < 0:
            return False
        if self.authority == TransformAuthority.SERVER:
            try:
                from core.network.transport import get_transport
                return get_transport().is_server
            except Exception:
                return False
        try:
            from core.network.transport import get_transport
            return ident.is_owner or get_transport().is_server
        except Exception:
            return False

    def _is_owner(self) -> bool:
        ident = self._get_identity()
        if ident is None:
            return False
        return ident.is_owner

    def _should_send(self, pos: Vec3, rot: Quat, scale: Vec3) -> bool:
        if not self._has_sent_initial:
            return True
        if self.sync_position:
            dx = pos.x - self._last_sent_pos.x
            dy = pos.y - self._last_sent_pos.y
            dz = pos.z - self._last_sent_pos.z
            if dx * dx + dy * dy + dz * dz > self.pos_threshold * self.pos_threshold:
                return True
        if self.sync_rotation and self.rot_threshold > 0:
            dot = rot.x * self._last_sent_rot.x + rot.y * self._last_sent_rot.y + rot.z * self._last_sent_rot.z + rot.w * self._last_sent_rot.w
            dot = max(-1.0, min(1.0, dot))
            angle = math.degrees(2.0 * math.acos(abs(dot)))
            if angle > self.rot_threshold:
                return True
        if self.sync_scale:
            dx = scale.x - self._last_sent_scale.x
            dy = scale.y - self._last_sent_scale.y
            dz = scale.z - self._last_sent_scale.z
            if dx * dx + dy * dy + dz * dz > self.scale_threshold * self.scale_threshold:
                return True
        return False

    def _send_state(self, pos: Vec3, rot: Quat, scale: Vec3):
        ident = self._get_identity()
        if ident is None:
            return
        payload = {
            "net_id": ident.net_id,
            "t": time.time(),
        }
        if self.sync_position:
            payload["p"] = pos.to_list()
        if self.sync_rotation:
            payload["r"] = rot.to_list()
        if self.sync_scale:
            payload["s"] = scale.to_list()
        try:
            from core.network.transport import get_transport
            from core.network.protocol import MessageType
            get_transport().broadcast(MessageType.NET_TRANSFORM, payload)
        except Exception:
            pass
        self._last_sent_pos = Vec3(pos.x, pos.y, pos.z)
        self._last_sent_rot = Quat(rot.x, rot.y, rot.z, rot.w)
        self._last_sent_scale = Vec3(scale.x, scale.y, scale.z)
        self._has_sent_initial = True

    def apply_snapshot(self, data: dict):
        t = float(data.get("t", time.time()))
        p = data.get("p")
        r = data.get("r")
        s = data.get("s")
        ident = self._get_identity()
        if ident is not None and ident.is_owner and self.authority == TransformAuthority.OWNER:
            return
        if ident is not None and self.authority == TransformAuthority.SERVER:
            try:
                from core.network.transport import get_transport
                if get_transport().is_server:
                    return
            except Exception:
                pass
        if p is not None and s is None and r is None:
            pass
        if self.interpolate and self.teleport_threshold > 0 and p is not None:
            tr = self.transform
            if tr is not None:
                cur = tr.local_position
                dx = float(p[0]) - cur.x
                dy = float(p[1]) - cur.y
                dz = float(p[2]) - cur.z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist > self.teleport_threshold:
                    self._buffer.clear()
                    if p is not None:
                        tr.local_position = Vec3(float(p[0]), float(p[1]), float(p[2]))
                    if r is not None:
                        tr.local_rotation = Quat(float(r[0]), float(r[1]), float(r[2]), float(r[3]))
                    if s is not None:
                        tr.local_scale = Vec3(float(s[0]), float(s[1]), float(s[2]))
                    return
        entry = {"t": t}
        if p is not None:
            entry["p"] = [float(p[0]), float(p[1]), float(p[2])]
        if r is not None:
            entry["r"] = [float(r[0]), float(r[1]), float(r[2]), float(r[3])]
        if s is not None:
            entry["s"] = [float(s[0]), float(s[1]), float(s[2])]
        self._buffer.append(entry)
        if not self.interpolate:
            tr = self.transform
            if tr is None:
                return
            if p is not None:
                tr.local_position = Vec3(float(p[0]), float(p[1]), float(p[2]))
            if r is not None:
                tr.local_rotation = Quat(float(r[0]), float(r[1]), float(r[2]), float(r[3]))
            if s is not None:
                tr.local_scale = Vec3(float(s[0]), float(s[1]), float(s[2]))

    def teleport(self, pos: Vec3, rot: Quat | None = None, scale: Vec3 | None = None):
        tr = self.transform
        if tr is None:
            return
        tr.local_position = pos
        if rot is not None:
            tr.local_rotation = rot
        if scale is not None:
            tr.local_scale = scale
        self._buffer.clear()
        self._send_state(tr.local_position, tr.local_rotation, tr.local_scale)

    def on_update(self, dt: float):
        tr = self.transform
        if tr is None:
            return
        if self._can_send():
            self._send_accum += dt
            interval = 1.0 / max(1.0, self.send_rate)
            if self._send_accum >= interval:
                self._send_accum = 0.0
                pos = tr.local_position
                rot = tr.local_rotation
                scale = tr.local_scale
                if self._should_send(pos, rot, scale):
                    self._send_state(pos, rot, scale)
            return
        if not self.interpolate or len(self._buffer) == 0:
            return
        now = time.time()
        target = now - self.interp_delay
        buf = self._buffer
        if len(buf) == 1:
            entry = buf[0]
            if entry["t"] <= target:
                if "p" in entry and self.sync_position:
                    p = entry["p"]
                    tr.local_position = Vec3(float(p[0]), float(p[1]), float(p[2]))
                if "r" in entry and self.sync_rotation:
                    r = entry["r"]
                    tr.local_rotation = Quat(float(r[0]), float(r[1]), float(r[2]), float(r[3]))
                if "s" in entry and self.sync_scale:
                    s = entry["s"]
                    tr.local_scale = Vec3(float(s[0]), float(s[1]), float(s[2]))
            return
        prev = None
        nxt = None
        for e in buf:
            if e["t"] <= target:
                prev = e
            else:
                nxt = e
                break
        if prev is None:
            prev = buf[0]
            nxt = buf[1] if len(buf) > 1 else None
        if nxt is None:
            if "p" in prev and self.sync_position:
                p = prev["p"]
                tr.local_position = Vec3(float(p[0]), float(p[1]), float(p[2]))
            if "r" in prev and self.sync_rotation:
                r = prev["r"]
                tr.local_rotation = Quat(float(r[0]), float(r[1]), float(r[2]), float(r[3]))
            if "s" in prev and self.sync_scale:
                s = prev["s"]
                tr.local_scale = Vec3(float(s[0]), float(s[1]), float(s[2]))
            return
        dt_span = nxt["t"] - prev["t"]
        if dt_span < 0.0001:
            alpha = 1.0
        else:
            alpha = (target - prev["t"]) / dt_span
            alpha = max(0.0, min(1.0, alpha))
        if "p" in prev and "p" in nxt and self.sync_position:
            pa = prev["p"]; pb = nxt["p"]
            x = pa[0] + (pb[0] - pa[0]) * alpha
            y = pa[1] + (pb[1] - pa[1]) * alpha
            z = pa[2] + (pb[2] - pa[2]) * alpha
            tr.local_position = Vec3(float(x), float(y), float(z))
        if "r" in prev and "r" in nxt and self.sync_rotation:
            ra = prev["r"]; rb = nxt["r"]
            qa = Quat(float(ra[0]), float(ra[1]), float(ra[2]), float(ra[3]))
            qb = Quat(float(rb[0]), float(rb[1]), float(rb[2]), float(rb[3]))
            try:
                q = qa.slerp(qb, float(alpha))
            except Exception:
                q = Quat(
                    qa.x + (qb.x - qa.x) * alpha,
                    qa.y + (qb.y - qa.y) * alpha,
                    qa.z + (qb.z - qa.z) * alpha,
                    qa.w + (qb.w - qa.w) * alpha,
                ).normalized()
            tr.local_rotation = q
        if "s" in prev and "s" in nxt and self.sync_scale:
            sa = prev["s"]; sb = nxt["s"]
            x = sa[0] + (sb[0] - sa[0]) * alpha
            y = sa[1] + (sb[1] - sa[1]) * alpha
            z = sa[2] + (sb[2] - sa[2]) * alpha
            tr.local_scale = Vec3(float(x), float(y), float(z))
        while len(buf) > 2 and buf[1]["t"] <= target:
            buf.popleft()

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "sync_position": self.sync_position,
            "sync_rotation": self.sync_rotation,
            "sync_scale": self.sync_scale,
            "authority": self.authority.value,
            "send_rate": self.send_rate,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "scale_threshold": self.scale_threshold,
            "interpolate": self.interpolate,
            "interp_delay": self.interp_delay,
            "teleport_threshold": self.teleport_threshold,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> NetworkTransform:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.sync_position = bool(data.get("sync_position", True))
        inst.sync_rotation = bool(data.get("sync_rotation", True))
        inst.sync_scale = bool(data.get("sync_scale", False))
        raw = data.get("authority", "server")
        try:
            inst.authority = TransformAuthority(raw)
        except Exception:
            inst.authority = TransformAuthority.SERVER
        inst.send_rate = float(data.get("send_rate", 20.0))
        inst.pos_threshold = float(data.get("pos_threshold", 0.001))
        inst.rot_threshold = float(data.get("rot_threshold", 0.5))
        inst.scale_threshold = float(data.get("scale_threshold", 0.001))
        inst.interpolate = bool(data.get("interpolate", True))
        inst.interp_delay = float(data.get("interp_delay", 0.1))
        inst.teleport_threshold = float(data.get("teleport_threshold", 5.0))
        return inst
