# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional, Any
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
@ComponentRegistry.register
class MeshFilter(Component):
    _icon = "MeshFilter.png"
    _gizmo_icon_color = (180, 180, 180)
    _gizmo_icon_label = "M"
    _show_gizmo_icon: bool = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("mesh_name", "Mesh", FieldType.STRING),
            InspectorField("mesh_path", "Source", FieldType.RESOURCE_PATH, file_filter="Models (*.obj *.fbx)"),
        ]

    def __init__(self):
        super().__init__()
        self.mesh_path: str = ""
        self.mesh_name: str = ""
        self._mesh_data: Optional[Any] = None
    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"mesh_path": self.mesh_path, "mesh_name": self.mesh_name})
        return d
    @classmethod
    def deserialize(cls, data: dict) -> MeshFilter:
        mf = cls()
        mf.enabled = data.get("enabled", True)
        mf.mesh_path = data.get("mesh_path", "") or ""
        mf.mesh_name = data.get("mesh_name", "") or ""
        return mf
