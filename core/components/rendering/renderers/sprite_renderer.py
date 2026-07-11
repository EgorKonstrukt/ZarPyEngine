# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
@ComponentRegistry.register
class SpriteRenderer(Component):
    _icon = "SpriteRenderer.png"
    _gizmo_icon_color = (120, 200, 80)
    _gizmo_icon_label = "S"
    _show_gizmo_icon: bool = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("texture_path", "Texture", FieldType.RESOURCE_PATH, file_filter="Textures (*.png *.jpg *.jpeg)"),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("flip_x", "Flip X", FieldType.BOOL),
            InspectorField("flip_y", "Flip Y", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.texture_path: str = ""
        self._color: list[float] = [1, 1, 1, 1]
        self.flip_x: bool = False
        self.flip_y: bool = False

    @property
    def color(self) -> list[float]:
        return self._color

    @color.setter
    def color(self, val):
        if val is None:
            self._color = [1, 1, 1, 1]
        elif len(val) == 3:
            self._color = [val[0], val[1], val[2], 1.0]
        elif len(val) >= 4:
            self._color = [val[0], val[1], val[2], val[3]]
        else:
            self._color = [1, 1, 1, 1]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "texture_path": self.texture_path,
            "color": self.color,
            "flip_x": self.flip_x,
            "flip_y": self.flip_y,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> SpriteRenderer:
        sr = cls()
        sr.enabled = data.get("enabled", True)
        sr.texture_path = data.get("texture_path", "") or ""
        sr.color = data.get("color", [1, 1, 1, 1])
        sr.flip_x = data.get("flip_x", False)
        sr.flip_y = data.get("flip_y", False)
        return sr
