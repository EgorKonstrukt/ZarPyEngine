# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import numpy as np
from enum import Enum
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry
from core.math.math3d import Vec3
from core.components.inspector_meta import FieldType, InspectorField


MAX_FORCE_FIELDS = 16

FORCE_FIELD_DTYPE = np.dtype([
    ('position', np.float32, 4),
    ('force', np.float32, 4),
    ('box_half', np.float32, 4),
    ('extras', np.float32, 4),
])

FORCE_FIELD_SSBO_SIZE = MAX_FORCE_FIELDS * FORCE_FIELD_DTYPE.itemsize


class ForceFieldShape(Enum):
    SPHERE = "sphere"
    BOX = "box"


@ComponentRegistry.register
class ParticleForceField(Component):
    _allow_multiple = False
    _gizmo_icon_color = (100, 200, 255)
    _gizmo_icon_label = "FF"
    _gizmo_pass = "force_field"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("shape", "Shape", FieldType.ENUM, enum_class=ForceFieldShape),
            InspectorField("radius", "Radius", FieldType.FLOAT, min_val=0.01, max_val=500.0, step=0.1, decimals=2),
            InspectorField("box_size", "Box Size", FieldType.VEC3, min_val=0.01, max_val=500.0, step=0.1, decimals=2),
            InspectorField("start_range", "Start Range", FieldType.FLOAT, min_val=0.0, max_val=500.0, step=0.1, decimals=2),
            InspectorField("force_x", "Force X", FieldType.FLOAT, min_val=-1000.0, max_val=1000.0, step=0.1, decimals=2),
            InspectorField("force_y", "Force Y", FieldType.FLOAT, min_val=-1000.0, max_val=1000.0, step=0.1, decimals=2),
            InspectorField("force_z", "Force Z", FieldType.FLOAT, min_val=-1000.0, max_val=1000.0, step=0.1, decimals=2),
            InspectorField("intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=100.0, step=0.01, decimals=2),
            InspectorField("multiply_by_distance", "Multiply by Distance", FieldType.BOOL),
            InspectorField("drag", "Drag", FieldType.FLOAT, min_val=0.0, max_val=100.0, step=0.01, decimals=3),
            InspectorField("gravity", "Gravity", FieldType.FLOAT, min_val=-100.0, max_val=100.0, step=0.1, decimals=2),
        ]

    def __init__(self):
        super().__init__()
        self.shape: ForceFieldShape = ForceFieldShape.SPHERE
        self.radius: float = 5.0
        self.box_size: list[float] = [5.0, 5.0, 5.0]
        self.start_range: float = 0.0
        self.force_x: float = 0.0
        self.force_y: float = 1.0
        self.force_z: float = 0.0
        self.intensity: float = 1.0
        self.multiply_by_distance: bool = True
        self.drag: float = 0.0
        self.gravity: float = 0.0
        self._gpu_cache: Optional[np.ndarray] = None

    def to_gpu_data(self) -> np.ndarray:
        if self._gpu_cache is None:
            self._gpu_cache = np.zeros(1, dtype=FORCE_FIELD_DTYPE)
        arr = self._gpu_cache
        t = self.transform
        if t is not None:
            pos = t.position
            arr['position'][0, 0] = pos.x
            arr['position'][0, 1] = pos.y
            arr['position'][0, 2] = pos.z
        else:
            arr['position'][0, 0] = 0.0
            arr['position'][0, 1] = 0.0
            arr['position'][0, 2] = 0.0
        arr['position'][0, 3] = self.radius
        arr['force'][0, 0] = self.force_x
        arr['force'][0, 1] = self.force_y
        arr['force'][0, 2] = self.force_z
        arr['force'][0, 3] = self.intensity
        bx, by, bz = self.box_size
        arr['box_half'][0, 0] = bx * 0.5
        arr['box_half'][0, 1] = by * 0.5
        arr['box_half'][0, 2] = bz * 0.5
        arr['box_half'][0, 3] = 0.0 if self.shape == ForceFieldShape.SPHERE else 1.0
        arr['extras'][0, 0] = self.start_range
        arr['extras'][0, 1] = self.drag
        arr['extras'][0, 2] = self.gravity
        arr['extras'][0, 3] = 1.0 if self.multiply_by_distance else 0.0
        return arr

    def gizmo_lines(self, color: list[float] | None = None) -> list[tuple[Vec3, Vec3, list[float]]]:
        if color is None:
            color = [0.4, 0.8, 1.0, 0.6]
        tr = self.transform
        if not tr:
            return []
        world_pos = tr.position
        lines: list[tuple[Vec3, Vec3, list[float]]] = []
        segments = 24
        if self.shape == ForceFieldShape.SPHERE:
            for axis in range(3):
                pts = []
                for i in range(segments + 1):
                    theta = 2.0 * math.pi * i / segments
                    if axis == 0:
                        p = world_pos + Vec3(0, math.cos(theta) * self.radius, math.sin(theta) * self.radius)
                    elif axis == 1:
                        p = world_pos + Vec3(math.cos(theta) * self.radius, 0, math.sin(theta) * self.radius)
                    else:
                        p = world_pos + Vec3(math.cos(theta) * self.radius, math.sin(theta) * self.radius, 0)
                    pts.append(p)
                for i in range(segments):
                    lines.append((pts[i], pts[i + 1], color))
            if self.start_range > 0.01:
                inner_color = [0.2, 0.5, 0.8, 0.4]
                for axis in range(3):
                    pts = []
                    for i in range(segments + 1):
                        theta = 2.0 * math.pi * i / segments
                        r = self.start_range
                        if axis == 0:
                            p = world_pos + Vec3(0, math.cos(theta) * r, math.sin(theta) * r)
                        elif axis == 1:
                            p = world_pos + Vec3(math.cos(theta) * r, 0, math.sin(theta) * r)
                        else:
                            p = world_pos + Vec3(math.cos(theta) * r, math.sin(theta) * r, 0)
                        pts.append(p)
                    for i in range(segments):
                        lines.append((pts[i], pts[i + 1], inner_color))
        else:
            hx, hy, hz = [s * 0.5 for s in self.box_size]
            corners_local = [
                Vec3(-hx, -hy, -hz), Vec3(hx, -hy, -hz),
                Vec3(hx, hy, -hz), Vec3(-hx, hy, -hz),
                Vec3(-hx, -hy, hz), Vec3(hx, -hy, hz),
                Vec3(hx, hy, hz), Vec3(-hx, hy, hz),
            ]
            corners = [world_pos + v for v in corners_local]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for a, b in edges:
                lines.append((corners[a], corners[b], color))
            if self.start_range > 0.01:
                sr = self.start_range
                inner_color = [0.2, 0.5, 0.8, 0.4]
                ic = [
                    Vec3(-sr, -sr, -sr), Vec3(sr, -sr, -sr),
                    Vec3(sr, sr, -sr), Vec3(-sr, sr, -sr),
                    Vec3(-sr, -sr, sr), Vec3(sr, -sr, sr),
                    Vec3(sr, sr, sr), Vec3(-sr, sr, sr),
                ]
                icorners = [world_pos + v for v in ic]
                for a, b in edges:
                    lines.append((icorners[a], icorners[b], inner_color))
        return lines

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "shape": self.shape.value,
            "radius": self.radius,
            "box_size": self.box_size,
            "start_range": self.start_range,
            "force_x": self.force_x,
            "force_y": self.force_y,
            "force_z": self.force_z,
            "intensity": self.intensity,
            "multiply_by_distance": self.multiply_by_distance,
            "drag": self.drag,
            "gravity": self.gravity,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ParticleForceField:
        ff = cls()
        ff.enabled = data.get("enabled", True)
        ff.shape = ForceFieldShape(data.get("shape", "sphere"))
        ff.radius = data.get("radius", 5.0)
        ff.box_size = data.get("box_size", [5.0, 5.0, 5.0])
        ff.start_range = data.get("start_range", 0.0)
        ff.force_x = data.get("force_x", 0.0)
        ff.force_y = data.get("force_y", 1.0)
        ff.force_z = data.get("force_z", 0.0)
        ff.intensity = data.get("intensity", 1.0)
        ff.multiply_by_distance = data.get("multiply_by_distance", True)
        ff.drag = data.get("drag", 0.0)
        ff.gravity = data.get("gravity", 0.0)
        return ff

    def on_destroy(self):
        pass
