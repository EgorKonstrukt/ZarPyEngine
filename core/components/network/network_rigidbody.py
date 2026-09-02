# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
from collections import deque
from enum import Enum
from core.ecs.ecs import Component, ComponentRegistry
from core.maths.math3d import Vec3
from core.components.inspector_meta import FieldType, InspectorField


class RigidbodyAuthority(Enum):
    SERVER = "server"
    OWNER = "owner"


@ComponentRegistry.register
class NetworkRigidbody(Component):
    _icon = "NetworkRigidbody.png"
    _gizmo_icon_color = (80, 200, 220)
    _gizmo_icon_label = "R"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("sync_velocity", "Sync Velocity", FieldType.BOOL),
            InspectorField("sync_angular", "Sync Angular", FieldType.BOOL),
            InspectorField("authority", "Authority", FieldType.ENUM, enum_class=RigidbodyAuthority),
            InspectorField("send_rate", "Send Rate", FieldType.FLOAT, min_val=1.0, max_val=60.0),
            InspectorField("velocity_threshold", "Vel Threshold", FieldType.FLOAT, min_val=0.0, max_val=10.0),
            InspectorField("interpolate", "Interpolate", FieldType.BOOL),
            InspectorField("interp_delay", "Interp Delay", FieldType.FLOAT, min_val=0.0, max_val=0.5),
        ]

    def __init__(self):
        super().__init__()
        self.sync_velocity: bool = True
        self.sync_angular: bool = True
        self.authority: RigidbodyAuthority = RigidbodyAuthority.SERVER
        self.send_rate: float = 20.0
        self.velocity_threshold: float = 0.01
        self.interpolate: bool = True
        self.interp_delay: float = 0.08
        self._buffer: deque = deque(maxlen=20)
        self._last_sent_vel: Vec3 = Vec3.zero()
        self._last_sent_avel: Vec3 = Vec3.zero()
        self._send_accum: float = 0.0
        self._has_sent: bool = False
        self._seq: int = 0

    def _get_identity(self):
        ent = self._entity
        if ent is None:
            return None
        return ent.get_component_by_name("NetworkIdentity")

    def _get_rigidbody(self):
        ent = self._entity
        if ent is None:
            return None
        return ent.get_component_by_name("Rigidbody")

    def _can_send(self) -> bool:
        ident = self._get_identity()
        if ident is None or ident.net_id < 0:
            return False
        if self.authority == RigidbodyAuthority.SERVER:
            try:
                from core.network.transport import get_transport
                return get_transport().is_server
            except Exception:
                return False
        if self.authority == RigidbodyAuthority.OWNER:
            try:
                return ident.is_owner
            except Exception:
                return False
        return False

    def _should_send(self, vel: Vec3, avel: Vec3) -> bool:
        if not self._has_sent:
            return True
        if self.sync_velocity:
            dx = vel.x - self._last_sent_vel.x
            dy = vel.y - self._last_sent_vel.y
            dz = vel.z - self._last_sent_vel.z
            if dx * dx + dy * dy + dz * dz > self.velocity_threshold * self.velocity_threshold:
                return True
        if self.sync_angular:
            dx = avel.x - self._last_sent_avel.x
            dy = avel.y - self._last_sent_avel.y
            dz = avel.z - self._last_sent_avel.z
            if dx * dx + dy * dy + dz * dz > self.velocity_threshold * self.velocity_threshold:
                return True
        return False

    def _send(self, vel: Vec3, avel: Vec3):
        ident = self._get_identity()
        if ident is None:
            return
        if not self._can_send():
            return
        self._seq += 1
        payload: dict = {"net_id": ident.net_id, "t": time.time(), "seq": self._seq}
        if self.sync_velocity:
            payload["v"] = vel.to_list()
        if self.sync_angular:
            payload["w"] = avel.to_list()
        try:
            from core.network.transport import get_transport
            from core.network.protocol import MessageType
            get_transport().broadcast(MessageType.NET_RIGIDBODY, payload)
        except Exception:
            pass
        self._last_sent_vel = Vec3(vel.x, vel.y, vel.z)
        self._last_sent_avel = Vec3(avel.x, avel.y, avel.z)
        self._has_sent = True

    def apply_snapshot(self, data: dict):
        ident = self._get_identity()
        if ident is not None and ident.is_owner and self.authority == RigidbodyAuthority.OWNER:
            return
        if self.authority == RigidbodyAuthority.SERVER:
            try:
                from core.network.transport import get_transport
                if get_transport().is_server:
                    return
            except Exception:
                pass
        seq = int(data.get("seq", 0))
        if seq and self._buffer and seq <= int(self._buffer[-1].get("seq", 0)):
            return
        entry: dict = {"t": float(data.get("t", time.time())), "seq": seq}
        v = data.get("v")
        w = data.get("w")
        if v is not None:
            entry["v"] = [float(v[0]), float(v[1]), float(v[2])]
        if w is not None:
            entry["w"] = [float(w[0]), float(w[1]), float(w[2])]
        self._buffer.append(entry)
        if not self.interpolate:
            rb = self._get_rigidbody()
            if rb is None:
                return
            if v is not None:
                rb.velocity = Vec3(float(v[0]), float(v[1]), float(v[2]))
            if w is not None:
                rb.angular_velocity = Vec3(float(w[0]), float(w[1]), float(w[2]))

    def on_fixed_update(self, dt: float):
        rb = self._get_rigidbody()
        if rb is None:
            return
        if self._can_send():
            self._send_accum += dt
            interval = 1.0 / max(1.0, self.send_rate)
            if self._send_accum >= interval:
                self._send_accum = 0.0
                vel = rb.velocity
                avel = rb.angular_velocity
                if self._should_send(vel, avel):
                    self._send(vel, avel)

    def on_update(self, dt: float):
        if self._can_send():
            return
        if not self.interpolate or len(self._buffer) == 0:
            return
        rb = self._get_rigidbody()
        if rb is None:
            return
        now = time.time()
        target = now - self.interp_delay
        buf = self._buffer
        if len(buf) == 1:
            e = buf[0]
            if e["t"] <= target:
                if "v" in e and self.sync_velocity:
                    v = e["v"]
                    rb.velocity = Vec3(float(v[0]), float(v[1]), float(v[2]))
                if "w" in e and self.sync_angular:
                    w = e["w"]
                    rb.angular_velocity = Vec3(float(w[0]), float(w[1]), float(w[2]))
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
            if "v" in prev and self.sync_velocity:
                v = prev["v"]
                rb.velocity = Vec3(float(v[0]), float(v[1]), float(v[2]))
            if "w" in prev and self.sync_angular:
                w = prev["w"]
                rb.angular_velocity = Vec3(float(w[0]), float(w[1]), float(w[2]))
            return
        dt_span = nxt["t"] - prev["t"]
        if dt_span < 0.0001:
            alpha = 1.0
        else:
            alpha = (target - prev["t"]) / dt_span
            alpha = max(0.0, min(1.0, alpha))
        cur_v = rb.velocity
        cur_w = rb.angular_velocity
        if "v" in prev and "v" in nxt and self.sync_velocity:
            va = prev["v"]; vb = nxt["v"]
            tx = va[0] + (vb[0] - va[0]) * alpha
            ty = va[1] + (vb[1] - va[1]) * alpha
            tz = va[2] + (vb[2] - va[2]) * alpha
            tgt = Vec3(float(tx), float(ty), float(tz))
            rb.velocity = Vec3(
                cur_v.x + (tgt.x - cur_v.x) * min(1.0, dt * 12.0),
                cur_v.y + (tgt.y - cur_v.y) * min(1.0, dt * 12.0),
                cur_v.z + (tgt.z - cur_v.z) * min(1.0, dt * 12.0),
            )
        if "w" in prev and "w" in nxt and self.sync_angular:
            wa = prev["w"]; wb = nxt["w"]
            tx = wa[0] + (wb[0] - wa[0]) * alpha
            ty = wa[1] + (wb[1] - wa[1]) * alpha
            tz = wa[2] + (wb[2] - wa[2]) * alpha
            tgt = Vec3(float(tx), float(ty), float(tz))
            rb.angular_velocity = Vec3(
                cur_w.x + (tgt.x - cur_w.x) * min(1.0, dt * 12.0),
                cur_w.y + (tgt.y - cur_w.y) * min(1.0, dt * 12.0),
                cur_w.z + (tgt.z - cur_w.z) * min(1.0, dt * 12.0),
            )
        while len(buf) > 2 and buf[1]["t"] <= target:
            buf.popleft()

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "sync_velocity": self.sync_velocity,
            "sync_angular": self.sync_angular,
            "authority": self.authority.value,
            "send_rate": self.send_rate,
            "velocity_threshold": self.velocity_threshold,
            "interpolate": self.interpolate,
            "interp_delay": self.interp_delay,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> NetworkRigidbody:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.sync_velocity = bool(data.get("sync_velocity", True))
        inst.sync_angular = bool(data.get("sync_angular", True))
        raw = data.get("authority", "server")
        try:
            inst.authority = RigidbodyAuthority(raw)
        except Exception:
            inst.authority = RigidbodyAuthority.SERVER
        inst.send_rate = float(data.get("send_rate", 20.0))
        inst.velocity_threshold = float(data.get("velocity_threshold", 0.01))
        inst.interpolate = bool(data.get("interpolate", True))
        inst.interp_delay = float(data.get("interp_delay", 0.08))
        return inst
