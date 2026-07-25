# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField


@ComponentRegistry.register
class GaussianSplatRenderer(Component):
    _icon = "MeshRenderer.png"
    _gizmo_icon_color = (200, 120, 255)
    _gizmo_icon_label = "GS"
    _show_gizmo_icon: bool = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("ply_path", "PLY Path", FieldType.RESOURCE_PATH,
                           file_filter="PLY (*.ply)"),
            InspectorField("sh_degree", "SH Degree", FieldType.INT_SLIDER,
                           min_val=0, max_val=3, step=1),
            InspectorField("opacity_threshold", "Opacity Cutoff", FieldType.FLOAT,
                           min_val=0.0, max_val=1.0, step=0.01),
            InspectorField("max_screen_size", "Max Screen Size", FieldType.FLOAT,
                           min_val=0.0, max_val=64.0, step=1.0),
        ]

    def __init__(self):
        super().__init__()
        self.ply_path: str = ""
        self.sh_degree: int = 3
        self.opacity_threshold: float = 0.005
        self.max_screen_size: float = 32.0

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "ply_path": self.ply_path,
            "sh_degree": self.sh_degree,
            "opacity_threshold": self.opacity_threshold,
            "max_screen_size": self.max_screen_size,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> GaussianSplatRenderer:
        g = cls()
        g.enabled = data.get("enabled", True)
        g.ply_path = data.get("ply_path", "")
        g.sh_degree = data.get("sh_degree", 3)
        g.opacity_threshold = data.get("opacity_threshold", 0.005)
        g.max_screen_size = data.get("max_screen_size", 32.0)
        return g
