# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField, ListElementField
@ComponentRegistry.register
class MeshRenderer(Component):
    _icon = "MeshRenderer.png"
    _gizmo_icon_color = (160, 160, 160)
    _gizmo_icon_label = "M"
    _show_gizmo_icon: bool = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("materials", "Materials", FieldType.LIST, element_fields=[
                ListElementField("path", "Material", FieldType.RESOURCE_PATH, file_filter="Material (*.mat)"),
            ]),
            InspectorField("cast_shadows", "Cast Shadows", FieldType.BOOL),
            InspectorField("receive_shadows", "Receive Shadows", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.materials: list[dict] = [{"path": ""}]
        self.cast_shadows: bool = True
        self.receive_shadows: bool = True

    def get_material_path(self, sub_mesh_index: int = 0) -> str:
        if sub_mesh_index < len(self.materials):
            return self.materials[sub_mesh_index].get("path", "")
        if self.materials:
            return self.materials[-1].get("path", "")
        return ""

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"materials": self.materials, "cast_shadows": self.cast_shadows, "receive_shadows": self.receive_shadows})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> MeshRenderer:
        mr = cls()
        mr.enabled = data.get("enabled", True)
        raw = data.get("materials")
        if raw:
            mr.materials = raw
        elif "material_path" in data:
            mr.materials = [{"path": data.get("material_path", "")}]
        else:
            mr.materials = [{"path": ""}]
        mr.cast_shadows = data.get("cast_shadows", True)
        mr.receive_shadows = data.get("receive_shadows", True)
        return mr
