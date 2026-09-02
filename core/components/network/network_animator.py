# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
from enum import Enum
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField


class AnimatorAuthority(Enum):
    SERVER = "server"
    OWNER = "owner"


@ComponentRegistry.register
class NetworkAnimator(Component):
    _icon = "NetworkAnimator.png"
    _gizmo_icon_color = (180, 120, 255)
    _gizmo_icon_label = "A"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("sync_clip", "Sync Clip", FieldType.BOOL),
            InspectorField("sync_time", "Sync Time", FieldType.BOOL),
            InspectorField("sync_speed", "Sync Speed", FieldType.BOOL),
            InspectorField("sync_playing", "Sync Playing", FieldType.BOOL),
            InspectorField("authority", "Authority", FieldType.ENUM, enum_class=AnimatorAuthority),
            InspectorField("send_rate", "Send Rate", FieldType.FLOAT, min_val=1.0, max_val=30.0),
            InspectorField("param_sync", "Sync Params", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.sync_clip: bool = True
        self.sync_time: bool = True
        self.sync_speed: bool = True
        self.sync_playing: bool = True
        self.authority: AnimatorAuthority = AnimatorAuthority.OWNER
        self.send_rate: float = 10.0
        self.param_sync: bool = True
        self._last_clip: str = ""
        self._last_time: float = -1.0
        self._last_speed: float = -1.0
        self._last_playing: bool | None = None
        self._last_params: dict = {}
        self._send_accum: float = 0.0

    def _get_identity(self):
        ent = self._entity
        if ent is None:
            return None
        return ent.get_component_by_name("NetworkIdentity")

    def _get_animation(self):
        ent = self._entity
        if ent is None:
            return None
        a = ent.get_component_by_name("Animation")
        if a is not None:
            return a
        return ent.get_component_by_name("Animator")

    def _can_send(self) -> bool:
        ident = self._get_identity()
        if ident is None or ident.net_id < 0:
            return False
        if self.authority == AnimatorAuthority.SERVER:
            try:
                from core.network.transport import get_transport
                return get_transport().is_server
            except Exception:
                return False
        try:
            return ident.is_owner
        except Exception:
            return False

    def _collect_state(self) -> dict | None:
        anim = self._get_animation()
        if anim is None:
            return None
        state: dict = {}
        if self.sync_clip:
            clip = getattr(anim, "clip", "")
            if hasattr(anim, "_current_clip"):
                clip = getattr(anim, "_current_clip", clip)
            state["clip"] = str(clip) if clip is not None else ""
        if self.sync_time:
            t = getattr(anim, "_time", None)
            if t is None:
                t = getattr(anim, "time", 0.0)
            state["time"] = float(t) if t is not None else 0.0
        if self.sync_speed:
            sp = getattr(anim, "speed", 1.0)
            state["speed"] = float(sp)
        if self.sync_playing:
            pl = getattr(anim, "_is_playing", None)
            if pl is None:
                pl = getattr(anim, "is_playing", True)
            state["playing"] = bool(pl)
        if self.param_sync:
            params = {}
            src = getattr(anim, "parameters", None)
            if src is None:
                src = getattr(anim, "_parameters", None)
            if isinstance(src, dict):
                for k, v in src.items():
                    if isinstance(v, bool):
                        params[k] = bool(v)
                    elif isinstance(v, int):
                        params[k] = int(v)
                    elif isinstance(v, float):
                        params[k] = float(v)
            if params:
                state["params"] = params
            trigger = getattr(anim, "_trigger", None)
            if trigger:
                state["trigger"] = str(trigger)
        return state

    def _has_changed(self, state: dict) -> bool:
        if state.get("clip", self._last_clip) != self._last_clip:
            return True
        if self.sync_time and abs(state.get("time", 0.0) - self._last_time) > 0.02:
            return True
        if self.sync_speed and abs(state.get("speed", 1.0) - self._last_speed) > 0.001:
            return True
        if self.sync_playing and state.get("playing") != self._last_playing:
            return True
        if self.param_sync and state.get("params") != self._last_params:
            return True
        if "trigger" in state:
            return True
        return self._last_time < 0

    def _send_state(self, state: dict):
        ident = self._get_identity()
        if ident is None:
            return
        payload = {"net_id": ident.net_id, "t": time.time()}
        payload.update(state)
        try:
            from core.network.transport import get_transport
            from core.network.protocol import MessageType
            get_transport().broadcast(MessageType.NET_ANIMATOR, payload)
        except Exception:
            pass
        self._last_clip = state.get("clip", self._last_clip)
        self._last_time = state.get("time", self._last_time)
        self._last_speed = state.get("speed", self._last_speed)
        self._last_playing = state.get("playing", self._last_playing)
        self._last_params = dict(state.get("params", {}))

    def apply_snapshot(self, data: dict):
        ident = self._get_identity()
        if ident is not None and ident.is_owner and self.authority == AnimatorAuthority.OWNER:
            return
        if self.authority == AnimatorAuthority.SERVER:
            try:
                from core.network.transport import get_transport
                if get_transport().is_server:
                    return
            except Exception:
                pass
        anim = self._get_animation()
        if anim is None:
            return
        if "clip" in data and self.sync_clip:
            clip = str(data["clip"])
            if hasattr(anim, "clip"):
                try:
                    anim.clip = clip
                except Exception:
                    pass
            if hasattr(anim, "_current_clip"):
                try:
                    anim._current_clip = clip
                except Exception:
                    pass
        if "time" in data and self.sync_time:
            try:
                anim._time = float(data["time"])
            except Exception:
                try:
                    anim.time = float(data["time"])
                except Exception:
                    pass
        if "speed" in data and self.sync_speed:
            try:
                anim.speed = float(data["speed"])
            except Exception:
                pass
        if "playing" in data and self.sync_playing:
            try:
                anim._is_playing = bool(data["playing"])
            except Exception:
                try:
                    anim.is_playing = bool(data["playing"])
                except Exception:
                    pass
        if "params" in data and self.param_sync:
            params = data["params"]
            if isinstance(params, dict):
                dst = getattr(anim, "parameters", None)
                if dst is None:
                    dst = getattr(anim, "_parameters", None)
                if isinstance(dst, dict):
                    for k, v in params.items():
                        dst[k] = v
                else:
                    for k, v in params.items():
                        try:
                            setattr(anim, k, v)
                        except Exception:
                            pass
        if "trigger" in data:
            trig = str(data["trigger"])
            if hasattr(anim, "set_trigger"):
                try:
                    anim.set_trigger(trig)
                except Exception:
                    pass
            elif hasattr(anim, "trigger"):
                try:
                    anim.trigger(trig)
                except Exception:
                    pass

    def trigger(self, name: str):
        anim = self._get_animation()
        if anim is not None:
            if hasattr(anim, "set_trigger"):
                try:
                    anim.set_trigger(name)
                except Exception:
                    pass
        if self._can_send():
            ident = self._get_identity()
            if ident is None:
                return
            try:
                from core.network.transport import get_transport
                from core.network.protocol import MessageType
                get_transport().broadcast(MessageType.NET_ANIMATOR, {"net_id": ident.net_id, "trigger": str(name), "t": time.time()})
            except Exception:
                pass

    def set_float(self, name: str, value: float):
        anim = self._get_animation()
        if anim is not None:
            dst = getattr(anim, "parameters", None)
            if isinstance(dst, dict):
                dst[name] = float(value)
            else:
                try:
                    setattr(anim, name, float(value))
                except Exception:
                    pass

    def set_bool(self, name: str, value: bool):
        anim = self._get_animation()
        if anim is not None:
            dst = getattr(anim, "parameters", None)
            if isinstance(dst, dict):
                dst[name] = bool(value)
            else:
                try:
                    setattr(anim, name, bool(value))
                except Exception:
                    pass

    def on_update(self, dt: float):
        if not self._can_send():
            return
        self._send_accum += dt
        interval = 1.0 / max(1.0, self.send_rate)
        if self._send_accum < interval:
            return
        self._send_accum = 0.0
        state = self._collect_state()
        if state is None:
            return
        if self._has_changed(state):
            self._send_state(state)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "sync_clip": self.sync_clip,
            "sync_time": self.sync_time,
            "sync_speed": self.sync_speed,
            "sync_playing": self.sync_playing,
            "authority": self.authority.value,
            "send_rate": self.send_rate,
            "param_sync": self.param_sync,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> NetworkAnimator:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.sync_clip = bool(data.get("sync_clip", True))
        inst.sync_time = bool(data.get("sync_time", True))
        inst.sync_speed = bool(data.get("sync_speed", True))
        inst.sync_playing = bool(data.get("sync_playing", True))
        raw = data.get("authority", "owner")
        try:
            inst.authority = AnimatorAuthority(raw)
        except Exception:
            inst.authority = AnimatorAuthority.OWNER
        inst.send_rate = float(data.get("send_rate", 10.0))
        inst.param_sync = bool(data.get("param_sync", True))
        return inst
