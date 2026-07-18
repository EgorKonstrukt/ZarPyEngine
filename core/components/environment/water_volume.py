# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import time
from core.ecs.ecs import Component, ComponentRegistry
from core.math.math3d import Vec3
from core.components.inspector_meta import FieldType, InspectorField


@ComponentRegistry.register
class WaterVolume(Component):
    _icon = "WaterVolume.png"
    _gizmo_icon_color = (40, 130, 220)
    _gizmo_icon_label = "W"
    _show_gizmo_icon: bool = True
    _gizmo_pass = "force_field"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("mode", "Mode", FieldType.ENUM, enum_options=["Infinite", "Box"]),
            InspectorField("water_level", "Water Level", FieldType.FLOAT, min_val=-10000.0, max_val=10000.0, step=0.01, decimals=3),
            InspectorField("size", "Box Size", FieldType.VEC3, min_val=0.01, max_val=100000.0, step=0.1, decimals=2),
            InspectorField("density", "Water Density", FieldType.FLOAT, min_val=1.0, max_val=2000.0, step=0.1, decimals=2),
            InspectorField("flow_strength", "Flow Strength", FieldType.FLOAT, min_val=0.0, max_val=50.0, step=0.01, decimals=3),
            InspectorField("flow_direction", "Flow Dir (deg)", FieldType.FLOAT, min_val=0.0, max_val=360.0, step=1.0, decimals=1),
            InspectorField("wave_amplitude", "Wave Amplitude", FieldType.FLOAT, min_val=0.0, max_val=50.0, step=0.01, decimals=3),
            InspectorField("wave_frequency", "Wave Frequency", FieldType.FLOAT, min_val=0.0, max_val=10.0, step=0.01, decimals=3),
            InspectorField("wave_speed", "Wave Speed", FieldType.FLOAT, min_val=0.0, max_val=10.0, step=0.01, decimals=3),
            InspectorField("wave_choppiness", "Wave Choppiness", FieldType.FLOAT, min_val=0.0, max_val=4.0, step=0.01, decimals=3),
        ]

    def __init__(self):
        super().__init__()
        self.mode: str = "Infinite"
        self.water_level: float = 0.0
        self.size: Vec3 = Vec3(100.0, 100.0, 100.0)
        self.density: float = 1000.0
        self.flow_strength: float = 0.0
        self.flow_direction: float = 0.0
        self.wave_amplitude: float = 0.0
        self.wave_frequency: float = 0.3
        self.wave_speed: float = 1.0
        self.wave_choppiness: float = 1.0

    def _flow_vec(self) -> tuple[float, float]:
        ang = math.radians(self.flow_direction)
        return (math.cos(ang), math.sin(ang))

    def contains(self, x: float, y: float, z: float) -> bool:
        if self.mode != "Box":
            return True
        tr = self.transform
        if tr is None:
            cx, cy, cz = 0.0, 0.0, 0.0
        else:
            p = tr.position
            cx, cy, cz = p.x, p.y, p.z
        hx, hy, hz = self.size.x * 0.5, self.size.y * 0.5, self.size.z * 0.5
        return (abs(x - cx) <= hx and abs(y - cy) <= hy and abs(z - cz) <= hz)

    def height_at(self, x: float, z: float, t: float | None = None) -> float:
        if t is None:
            t = time.time()
        y = self.water_level
        if self.wave_amplitude > 0.0 and self.wave_frequency > 0.0:
            k = 2.0 * math.pi * self.wave_frequency
            w = self.wave_speed * k
            chop = max(0.01, self.wave_choppiness)
            y += self.wave_amplitude * (
                0.6 * math.sin(k * x * chop + w * t)
                + 0.4 * math.sin(k * z * chop + w * 0.8 * t + 1.3)
            )
        return y

    def flow_at(self, x: float, y: float, z: float, t: float | None = None) -> Vec3:
        if self.flow_strength <= 0.0:
            return Vec3.zero()
        if not self.contains(x, y, z):
            return Vec3.zero()
        dx, dz = self._flow_vec()
        s = self.flow_strength
        if self.wave_amplitude > 0.0:
            s *= (1.0 + 0.15 * math.sin(t * 1.7 if t is not None else time.time() * 1.7))
        return Vec3(dx * s, 0.0, dz * s)

    def gizmo_lines(self) -> list[tuple[Vec3, Vec3, list[float]]]:
        if self.mode != "Box":
            return []
        tr = self.transform
        if tr is None:
            return []
        pos = tr.position
        hx, hy, hz = self.size.x * 0.5, self.size.y * 0.5, self.size.z * 0.5
        c = [0.25, 0.55, 0.9, 0.4]
        corners = [
            Vec3(pos.x - hx, pos.y - hy, pos.z - hz),
            Vec3(pos.x + hx, pos.y - hy, pos.z - hz),
            Vec3(pos.x + hx, pos.y - hy, pos.z + hz),
            Vec3(pos.x - hx, pos.y - hy, pos.z + hz),
            Vec3(pos.x - hx, pos.y + hy, pos.z - hz),
            Vec3(pos.x + hx, pos.y + hy, pos.z - hz),
            Vec3(pos.x + hx, pos.y + hy, pos.z + hz),
            Vec3(pos.x - hx, pos.y + hy, pos.z + hz),
        ]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        return [(corners[a], corners[b], c) for a, b in edges]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "mode": self.mode,
            "water_level": self.water_level,
            "size": self.size.to_list(),
            "density": self.density,
            "flow_strength": self.flow_strength,
            "flow_direction": self.flow_direction,
            "wave_amplitude": self.wave_amplitude,
            "wave_frequency": self.wave_frequency,
            "wave_speed": self.wave_speed,
            "wave_choppiness": self.wave_choppiness,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> WaterVolume:
        v = cls()
        v.enabled = data.get("enabled", True)
        v.mode = data.get("mode", "Infinite")
        v.water_level = data.get("water_level", 0.0)
        s = data.get("size", [100.0, 100.0, 100.0])
        v.size = Vec3(*s)
        v.density = data.get("density", 1000.0)
        v.flow_strength = data.get("flow_strength", 0.0)
        v.flow_direction = data.get("flow_direction", 0.0)
        v.wave_amplitude = data.get("wave_amplitude", 0.0)
        v.wave_frequency = data.get("wave_frequency", 0.3)
        v.wave_speed = data.get("wave_speed", 1.0)
        v.wave_choppiness = data.get("wave_choppiness", 1.0)
        return v
