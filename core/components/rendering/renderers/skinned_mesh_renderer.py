# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField, ListElementField


@ComponentRegistry.register
class SkinnedMeshRenderer(Component):
    _icon = "SkinnedMeshRenderer.png"
    _show_gizmo_icon: bool = False
    _gizmo_icon_label = "S"
    _category = "Skinned Mesh"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("mesh_name", "Mesh", FieldType.STRING),
            InspectorField("mesh_path", "Source", FieldType.RESOURCE_PATH, file_filter="Models (*.obj *.fbx *.glb *.gltf)"),
            InspectorField("materials", "Materials", FieldType.LIST, element_fields=[
                ListElementField("path", "Material", FieldType.RESOURCE_PATH, file_filter="Material (*.mat)"),
            ]),
            InspectorField("cast_shadows", "Cast Shadows", FieldType.BOOL),
            InspectorField("receive_shadows", "Receive Shadows", FieldType.BOOL),
            InspectorField("update_when_offscreen", "Update When Offscreen", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.mesh_path: str = ""
        self.mesh_name: str = ""
        self.materials: list[dict] = [{"path": ""}]
        self.cast_shadows: bool = True
        self.receive_shadows: bool = True
        self.update_when_offscreen: bool = True
        self._mesh_data: Optional[object] = None

    def get_material_path(self, sub_mesh_index: int = 0) -> str:
        if sub_mesh_index < len(self.materials):
            return self.materials[sub_mesh_index].get("path", "")
        if self.materials:
            return self.materials[-1].get("path", "")
        return ""

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "mesh_path": self.mesh_path,
            "mesh_name": self.mesh_name,
            "materials": self.materials,
            "cast_shadows": self.cast_shadows,
            "receive_shadows": self.receive_shadows,
            "update_when_offscreen": self.update_when_offscreen,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> SkinnedMeshRenderer:
        smr = cls()
        smr.enabled = data.get("enabled", True)
        smr.mesh_path = data.get("mesh_path", "") or ""
        smr.mesh_name = data.get("mesh_name", "") or ""
        raw = data.get("materials")
        if raw:
            smr.materials = raw
        elif "material_path" in data:
            smr.materials = [{"path": data.get("material_path", "")}]
        else:
            smr.materials = [{"path": ""}]
        smr.cast_shadows = data.get("cast_shadows", True)
        smr.receive_shadows = data.get("receive_shadows", True)
        smr.update_when_offscreen = data.get("update_when_offscreen", True)
        return smr
