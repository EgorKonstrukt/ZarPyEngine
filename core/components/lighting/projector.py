# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from core.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
@ComponentRegistry.register
class Projector(Component):
    _icon = "Projector.png"
    _gizmo_icon_color = (255, 180, 60)
    _gizmo_icon_label = "P"
    _gizmo_pass = "light"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("texture_path", "Texture", FieldType.RESOURCE_PATH, file_filter="Textures (*.png *.jpg *.jpeg)"),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=1000.0, step=0.1, decimals=3),
            InspectorField("range", "Range", FieldType.FLOAT, min_val=0.0, max_val=10000.0, step=0.5, decimals=2),
            InspectorField("spot_angle", "Spot Angle", FieldType.FLOAT, min_val=1.0, max_val=179.0, step=1.0, decimals=1),
            InspectorField("aspect_ratio", "Aspect", FieldType.FLOAT, min_val=0.1, max_val=10.0, step=0.1, decimals=2),
            InspectorField("near_plane", "Near Plane", FieldType.FLOAT, min_val=0.01, max_val=10.0, step=0.1, decimals=2),
            InspectorField("far_plane", "Far Plane", FieldType.FLOAT, min_val=1.0, max_val=10000.0, step=1.0, decimals=1),
        ]

    def __init__(self):
        super().__init__()
        self.texture_path: str = ""
        self.color: list[float] = [1.0, 1.0, 1.0]
        self.intensity: float = 1.0
        self.range: float = 10.0
        self.spot_angle: float = 30.0
        self.aspect_ratio: float = 1.0
        self.near_plane: float = 0.1
        self.far_plane: float = 100.0

    def gizmo(self):
        tr = self.transform
        if not tr:
            return []
        from core.ecs import GizmoPrimitive
        pos = tr.position
        fwd = tr.forward
        up = tr.up
        right = tr.right
        c = self.color or [1.0, 1.0, 1.0]
        brightness = max(c[0], c[1], c[2])
        if brightness < 0.01:
            col = [1.0, 1.0, 1.0, 0.8]
        else:
            col = [c[0] / brightness, c[1] / brightness, c[2] / brightness, 0.8]
        lines = []
        segments = 32
        rng = max(self.range, 0.1)
        half_angle = self.spot_angle * 0.5 * math.pi / 180.0
        aspect = max(self.aspect_ratio, 0.1)
        cone_r_x = rng * math.tan(half_angle)
        cone_r_y = cone_r_x / aspect
        base_center = pos + fwd * rng
        base_pts = []
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            rx = math.cos(a) * cone_r_x
            ry = math.sin(a) * cone_r_y
            base_pts.append(base_center + right * rx + up * ry)
        for i in range(segments):
            lines.append((base_pts[i], base_pts[(i + 1) % segments], col))
        for i in range(8):
            a = 2.0 * math.pi * i / 8
            rx = math.cos(a) * cone_r_x
            ry = math.sin(a) * cone_r_y
            bp = base_center + right * rx + up * ry
            lines.append((pos, bp, col))
        lines.append((pos, base_center, col))
        sprite_hw = cone_r_x * 0.5
        sprite_hh = cone_r_y * 0.5
        sprite_center = pos + fwd * rng * 0.5
        corners = [
            sprite_center - right * sprite_hw - up * sprite_hh,
            sprite_center + right * sprite_hw - up * sprite_hh,
            sprite_center + right * sprite_hw + up * sprite_hh,
            sprite_center - right * sprite_hw + up * sprite_hh,
        ]
        sprite_col = [col[0], col[1], col[2], 0.3]
        for i in range(4):
            lines.append((corners[i], corners[(i + 1) % 4], sprite_col))
        lines.append((corners[0], corners[2], sprite_col))
        lines.append((corners[1], corners[3], sprite_col))
        if not lines:
            return []
        return [GizmoPrimitive.from_lines(lines)]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "texture_path": self.texture_path,
            "color": self.color,
            "intensity": self.intensity,
            "range": self.range,
            "spot_angle": self.spot_angle,
            "aspect_ratio": self.aspect_ratio,
            "near_plane": self.near_plane,
            "far_plane": self.far_plane,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Projector:
        p = cls()
        p.enabled = data.get("enabled", True)
        p.texture_path = data.get("texture_path", "") or ""
        p.color = data.get("color", [1.0, 1.0, 1.0])
        p.intensity = data.get("intensity", 1.0)
        p.range = data.get("range", 10.0)
        p.spot_angle = data.get("spot_angle", 30.0)
        p.aspect_ratio = data.get("aspect_ratio", 1.0)
        p.near_plane = data.get("near_plane", 0.1)
        p.far_plane = data.get("far_plane", 100.0)
        return p
