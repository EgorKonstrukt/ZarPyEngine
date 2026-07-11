# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import time
from core.ecs.ecs import Component, ComponentRegistry
from core.math.math3d import Vec3
from core.components.inspector_meta import FieldType, InspectorField


@ComponentRegistry.register
class WindZone(Component):
    _icon = "WindZone.png"
    _gizmo_icon_color = (150, 200, 255)
    _gizmo_icon_label = "W"
    _show_gizmo_icon: bool = True
    _gizmo_pass = "force_field"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("mode", "Mode", FieldType.ENUM, enum_options=["Global", "Local"]),
            InspectorField("direction", "Direction (deg)", FieldType.FLOAT, min_val=0.0, max_val=360.0, step=1.0, decimals=1),
            InspectorField("speed", "Wind Speed (m/s)", FieldType.FLOAT, min_val=0.0, max_val=60.0, step=0.1, decimals=2),
            InspectorField("turbulence", "Turbulence", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("gust_strength", "Gust Strength", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("gust_frequency", "Gust Frequency", FieldType.FLOAT, min_val=0.0, max_val=2.0, step=0.01, decimals=3),
            InspectorField("alignment", "Wave Alignment", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("radius", "Radius", FieldType.FLOAT, min_val=0.0, max_val=5000.0, step=1.0, decimals=1),
            InspectorField("falloff", "Edge Falloff", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
        ]

    def __init__(self):
        super().__init__()
        self.mode: str = "Global"
        self.direction: float = 45.0
        self.speed: float = 8.0
        self.turbulence: float = 0.4
        self.gust_strength: float = 0.4
        self.gust_frequency: float = 0.15
        self.alignment: float = 0.7
        self.radius: float = 100.0
        self.falloff: float = 0.5

    def _dir_vec(self) -> tuple[float, float]:
        ang = math.radians(self.direction)
        return (math.cos(ang), math.sin(ang))

    def _gust_at(self, t: float) -> float:
        if self.gust_strength <= 0.0:
            return 0.0
        base = 0.5 + 0.5 * math.sin(2.0 * math.pi * self.gust_frequency * t)
        wobble = 0.5 + 0.5 * math.sin(2.0 * math.pi * self.gust_frequency * 2.37 * t + 1.7)
        g = self.gust_strength * (0.6 * base + 0.4 * wobble)
        return max(0.0, g)

    def sample(self, x: float = 0.0, z: float = 0.0, t: float | None = None) -> dict:
        if t is None:
            t = time.time()
        dx, dz = self._dir_vec()
        strength = 1.0
        if self.mode == "Local" and self.radius > 0.0:
            tr = self.transform
            cx, cz = (tr.position.x, tr.position.z) if tr else (0.0, 0.0)
            d = math.hypot(x - cx, z - cz)
            if d >= self.radius:
                return {"dir": (dx, dz), "speed": 0.0, "turbulence": 0.0,
                        "gust": 0.0, "strength": 0.0, "alignment": self.alignment}
            edge = self.radius * max(0.0, 1.0 - self.falloff)
            if d > edge:
                k = (self.radius - d) / max(1e-4, self.radius - edge)
                strength = min(1.0, max(0.0, k))
        return {
            "dir": (dx, dz),
            "speed": self.speed * strength,
            "turbulence": self.turbulence,
            "gust": self._gust_at(t) * strength,
            "strength": strength,
            "alignment": self.alignment,
        }

    def gizmo_lines(self) -> list[tuple[Vec3, Vec3, list[float]]]:
        tr = self.transform
        if not tr:
            return []
        pos = tr.position
        dx, dz = self._dir_vec()
        lines: list[tuple[Vec3, Vec3, list[float]]] = []
        arrow_color = [0.6, 0.78, 1.0, 0.9]
        len0 = max(self.radius * 0.5, 4.0) if self.mode == "Local" and self.radius > 0.0 else 6.0
        tip = pos + Vec3(dx * len0, 0.0, dz * len0)
        lines.append((pos, tip, arrow_color))
        perp = Vec3(-dz, 0.0, dx)
        back = Vec3(-dx, 0.0, -dz)
        a = 0.22 * len0
        b = 0.45 * len0
        for s in (1.0, -1.0):
            lines.append((tip, tip + back * b + perp * (a * s), arrow_color))
        if self.mode == "Local" and self.radius > 0.0 and self.radius > 0.01:
            segments = 40
            ring_color = [0.55, 0.75, 1.0, 0.35]
            pts = []
            for i in range(segments + 1):
                th = 2.0 * math.pi * i / segments
                pts.append(pos + Vec3(math.cos(th) * self.radius, 0.0, math.sin(th) * self.radius))
            for i in range(segments):
                lines.append((pts[i], pts[i + 1], ring_color))
        return lines

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "mode": self.mode,
            "direction": self.direction,
            "speed": self.speed,
            "turbulence": self.turbulence,
            "gust_strength": self.gust_strength,
            "gust_frequency": self.gust_frequency,
            "alignment": self.alignment,
            "radius": self.radius,
            "falloff": self.falloff,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> WindZone:
        z = cls()
        z.enabled = data.get("enabled", True)
        z.mode = data.get("mode", "Global")
        z.direction = data.get("direction", 45.0)
        z.speed = data.get("speed", 8.0)
        z.turbulence = data.get("turbulence", 0.4)
        z.gust_strength = data.get("gust_strength", 0.4)
        z.gust_frequency = data.get("gust_frequency", 0.15)
        z.alignment = data.get("alignment", 0.7)
        z.radius = data.get("radius", 100.0)
        z.falloff = data.get("falloff", 0.5)
        return z
