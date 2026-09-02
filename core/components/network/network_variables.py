# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
from enum import Enum
from typing import Any
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField


class VariableAuthority(Enum):
    SERVER = "server"
    OWNER = "owner"


@ComponentRegistry.register
class NetworkVariables(Component):
    _icon = "NetworkVariables.png"
    _gizmo_icon_color = (255, 180, 60)
    _gizmo_icon_label = "V"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("authority", "Authority", FieldType.ENUM, enum_class=VariableAuthority),
            InspectorField("send_rate", "Send Rate", FieldType.FLOAT, min_val=1.0, max_val=30.0),
            InspectorField("reliable", "Reliable", FieldType.BOOL),
            InspectorField("sync_on_change", "Sync On Change", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.authority: VariableAuthority = VariableAuthority.SERVER
        self.send_rate: float = 10.0
        self.reliable: bool = True
        self.sync_on_change: bool = True
        self._vars: dict[str, Any] = {}
        self._last_vars: dict[str, Any] = {}
        self._send_accum: float = 0.0
        self._dirty: bool = False

    def _get_identity(self):
        ent = self._entity
        if ent is None:
            return None
        return ent.get_component_by_name("NetworkIdentity")

    def _can_write(self) -> bool:
        ident = self._get_identity()
        if ident is None or ident.net_id < 0:
            return False
        if self.authority == VariableAuthority.SERVER:
            try:
                from core.network.transport import get_transport
                return get_transport().is_server
            except Exception:
                return False
        try:
            return ident.is_owner
        except Exception:
            return False

    def set_var(self, name: str, value: Any):
        self._vars[str(name)] = value
        self._dirty = True
        if self.sync_on_change:
            self._send_if_needed(force=False)

    def get_var(self, name: str, default: Any = None) -> Any:
        return self._vars.get(str(name), default)

    def set_vars(self, data: dict):
        changed = False
        for k, v in data.items():
            if self._vars.get(k) != v:
                changed = True
            self._vars[k] = v
        if changed:
            self._dirty = True

    def has_var(self, name: str) -> bool:
        return str(name) in self._vars

    def remove_var(self, name: str):
        if str(name) in self._vars:
            del self._vars[str(name)]
            self._dirty = True

    def all_vars(self) -> dict:
        return dict(self._vars)

    def _collapse(self, v: Any) -> Any:
        if hasattr(v, "to_list"):
            try:
                return v.to_list()
            except Exception:
                pass
        if isinstance(v, (list, tuple)):
            return list(v)
        if isinstance(v, dict):
            return {str(k): self._collapse(x) for k, x in v.items()}
        return v

    def _send_if_needed(self, force: bool = False):
        if not self._can_write() or not self._dirty:
            return
        if not force and self.sync_on_change is False:
            return
        ident = self._get_identity()
        if ident is None:
            return
        collapsed = {k: self._collapse(v) for k, v in self._vars.items()}
        payload = {"net_id": ident.net_id, "vars": collapsed, "t": time.time()}
        try:
            from core.network.transport import get_transport
            from core.network.protocol import MessageType
            get_transport().broadcast(MessageType.NET_VARIABLES, payload)
        except Exception:
            pass
        self._last_vars = dict(collapsed)
        self._dirty = False

    def apply_snapshot(self, data: dict):
        ident = self._get_identity()
        if ident is not None and ident.is_owner and self.authority == VariableAuthority.OWNER:
            return
        if self.authority == VariableAuthority.SERVER:
            try:
                from core.network.transport import get_transport
                if get_transport().is_server:
                    return
            except Exception:
                pass
        vars_data = data.get("vars", data.get("data", {}))
        if not isinstance(vars_data, dict):
            return
        for k, v in vars_data.items():
            self._vars[str(k)] = v
        self._last_vars = dict(vars_data)
        self._dirty = False

    def on_update(self, dt: float):
        if not self._can_write():
            return
        self._send_accum += dt
        interval = 1.0 / max(1.0, self.send_rate)
        if self._send_accum >= interval:
            self._send_accum = 0.0
            if self._dirty:
                self._send_if_needed(force=True)
        elif self._dirty and self.sync_on_change:
            cur = {k: self._collapse(v) for k, v in self._vars.items()}
            if cur != self._last_vars:
                self._send_if_needed(force=True)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "authority": self.authority.value,
            "send_rate": self.send_rate,
            "reliable": self.reliable,
            "sync_on_change": self.sync_on_change,
            "vars": dict(self._vars),
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> NetworkVariables:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        raw = data.get("authority", "server")
        try:
            inst.authority = VariableAuthority(raw)
        except Exception:
            inst.authority = VariableAuthority.SERVER
        inst.send_rate = float(data.get("send_rate", 10.0))
        inst.reliable = bool(data.get("reliable", True))
        inst.sync_on_change = bool(data.get("sync_on_change", True))
        v = data.get("vars", {})
        if isinstance(v, dict):
            inst._vars = dict(v)
            inst._last_vars = dict(v)
        return inst
