# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
@ComponentRegistry.register
class NetworkIdentity(Component):
    _icon = "NetworkIdentity.png"
    _gizmo_icon_color = (255, 100, 180)
    _gizmo_icon_label = "N"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("network_id", "Network ID", FieldType.INT, min_val=-1, max_val=999999),
            InspectorField("is_server", "Is Server", FieldType.BOOL),
            InspectorField("is_local_player", "Is Local Player", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.network_id: int = -1
        self.is_server: bool = False
        self.is_local_player: bool = False
    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"network_id": self.network_id, "is_server": self.is_server, "is_local_player": self.is_local_player})
        return d
    @classmethod
    def deserialize(cls, data: dict) -> NetworkIdentity:
        ni = cls()
        ni.enabled = data.get("enabled", True)
        ni.network_id = data.get("network_id", -1)
        ni.is_server = data.get("is_server", False)
        ni.is_local_player = data.get("is_local_player", False)
        return ni
