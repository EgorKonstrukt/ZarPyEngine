# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import time
import numpy as np
from core.ecs.ecs import Component, ComponentRegistry
from core.maths.math3d import Vec3
from core.components.inspector_meta import FieldType, InspectorField


def _fbm1d(t: float, octaves: int = 4) -> float:
    v = 0.0
    a = 1.0
    f = 1.0
    for _ in range(octaves):
        v += a * math.sin(2.0 * math.pi * f * t + 1.37 * f)
        a *= 0.5
        f *= 2.17
    return v


def _smooth_noise(t: float) -> float:
    return math.sin(t * 2.094 + 0.3) * 0.3 + math.sin(t * 3.771 + 1.7) * 0.2 + math.sin(t * 5.113 + 2.3) * 0.1


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
            InspectorField("turbulence_scale", "Turbulence Scale", FieldType.FLOAT, min_val=0.1, max_val=10.0, step=0.1, decimals=2),
            InspectorField("gust_strength", "Gust Strength", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("gust_frequency", "Gust Frequency (Hz)", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.01, decimals=3),
            InspectorField("gust_octaves", "Gust Octaves", FieldType.INT, min_val=1, max_val=6, step=1),
            InspectorField("gust_sharpness", "Gust Sharpness", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("vertical_boost", "Vertical Boost", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("alignment", "Wave Alignment", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("micro_turbulence", "Micro Turbulence", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("micro_frequency", "Micro Frequency", FieldType.FLOAT, min_val=0.0, max_val=30.0, step=0.1, decimals=2),
            InspectorField("radius", "Radius", FieldType.FLOAT, min_val=0.0, max_val=5000.0, step=1.0, decimals=1),
            InspectorField("falloff", "Edge Falloff", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
        ]

    def __init__(self):
        super().__init__()
        self.mode: str = "Global"
        self.direction: float = 45.0
        self.speed: float = 8.0
        self.turbulence: float = 0.4
        self.turbulence_scale: float = 1.5
        self.gust_strength: float = 0.5
        self.gust_frequency: float = 0.25
        self.gust_octaves: int = 4
        self.gust_sharpness: float = 0.3
        self.vertical_boost: float = 0.2
        self.alignment: float = 0.7
        self.micro_turbulence: float = 0.15
        self.micro_frequency: float = 8.0
        self.radius: float = 100.0
        self.falloff: float = 0.5

    def _dir_vec(self) -> tuple[float, float]:
        ang = math.radians(self.direction)
        return (math.cos(ang), math.sin(ang))

    def _base_gust(self, t: float) -> float:
        if self.gust_strength <= 0.0:
            return 0.0
        raw = _fbm1d(t * self.gust_frequency, self.gust_octaves)
        g = 0.5 + 0.5 * raw
        if self.gust_sharpness > 0.0:
            g = math.pow(g, 1.0 + self.gust_sharpness * 4.0)
        return self.gust_strength * max(0.0, min(1.0, g))

    def _turbulence_at(self, x: float, z: float, t: float) -> tuple[float, float]:
        if self.turbulence <= 0.0:
            return (0.0, 0.0)
        s = self.turbulence_scale
        tx = _smooth_noise(x * s * 0.01 + t * 0.15)
        tz = _smooth_noise(z * s * 0.01 + t * 0.17 + 100.0)
        return (tx * self.turbulence, tz * self.turbulence)

    def _micro_at(self, t: float) -> tuple[float, float]:
        if self.micro_turbulence <= 0.0:
            return (0.0, 0.0)
        mf = self.micro_frequency
        mx = math.sin(t * mf * 1.13 + 0.7) * 0.5 + math.sin(t * mf * 2.47 + 1.3) * 0.3
        mz = math.sin(t * mf * 1.07 + 2.1) * 0.5 + math.sin(t * mf * 2.13 + 0.3) * 0.3
        return (mx * self.micro_turbulence, mz * self.micro_turbulence)

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
                        "turbulence_scale": 0.0, "gust": 0.0, "strength": 0.0,
                        "alignment": self.alignment, "vertical_boost": 0.0,
                        "micro_turbulence": 0.0}
            edge = self.radius * max(0.0, 1.0 - self.falloff)
            if d > edge:
                k = (self.radius - d) / max(1e-4, self.radius - edge)
                strength = min(1.0, max(0.0, k))

        gust = self._base_gust(t) * strength
        turb_x, turb_z = self._turbulence_at(x, z, t)
        micro_x, micro_z = self._micro_at(t)

        combined_dir_x = dx + turb_x + micro_x
        combined_dir_z = dz + turb_z + micro_z
        dir_len = math.hypot(combined_dir_x, combined_dir_z)
        if dir_len > 1e-6:
            combined_dir_x /= dir_len
            combined_dir_z /= dir_len
        else:
            combined_dir_x = dx
            combined_dir_z = dz

        base_speed = self.speed * strength
        gust_speed_factor = 1.0 + gust * 1.5
        final_speed = base_speed * gust_speed_factor

        return {
            "dir": (combined_dir_x, combined_dir_z),
            "speed": final_speed,
            "turbulence": self.turbulence * strength,
            "turbulence_scale": self.turbulence_scale,
            "gust": gust,
            "gust_strength": self.gust_strength * strength,
            "strength": strength,
            "alignment": self.alignment,
            "vertical_boost": self.vertical_boost,
            "micro_turbulence": self.micro_turbulence * strength,
        }

    def sample_dir_vec3(self, x: float = 0.0, z: float = 0.0, t: float | None = None) -> Vec3:
        s = self.sample(x, z, t)
        d = s["dir"]
        vb = s.get("vertical_boost", 0.0)
        return Vec3(d[0], vb * 0.3, d[1])

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
        gust_line = tip + Vec3(0.0, 2.0, 0.0)
        lines.append((tip, gust_line, [0.8, 0.9, 1.0, 0.5]))
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
            "mode": self.mode, "direction": self.direction, "speed": self.speed,
            "turbulence": self.turbulence, "turbulence_scale": self.turbulence_scale,
            "gust_strength": self.gust_strength, "gust_frequency": self.gust_frequency,
            "gust_octaves": self.gust_octaves, "gust_sharpness": self.gust_sharpness,
            "vertical_boost": self.vertical_boost, "alignment": self.alignment,
            "micro_turbulence": self.micro_turbulence, "micro_frequency": self.micro_frequency,
            "radius": self.radius, "falloff": self.falloff,
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
        z.turbulence_scale = data.get("turbulence_scale", 1.5)
        z.gust_strength = data.get("gust_strength", 0.5)
        z.gust_frequency = data.get("gust_frequency", 0.25)
        z.gust_octaves = data.get("gust_octaves", 4)
        z.gust_sharpness = data.get("gust_sharpness", 0.3)
        z.vertical_boost = data.get("vertical_boost", 0.2)
        z.alignment = data.get("alignment", 0.7)
        z.micro_turbulence = data.get("micro_turbulence", 0.15)
        z.micro_frequency = data.get("micro_frequency", 8.0)
        z.radius = data.get("radius", 100.0)
        z.falloff = data.get("falloff", 0.5)
        return z
