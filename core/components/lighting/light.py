# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from enum import Enum
import math
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Vec3
class LightType(Enum):
    DIRECTIONAL = "directional"
    POINT = "point"
    SPOT = "spot"
    AREA = "area"
class LightAreaType(Enum):
    RECT = "rect"
    DISK = "disk"
@ComponentRegistry.register
class Light(Component):
    _icon = "Light.png"
    _gizmo_icon_color = (255, 220, 50)
    _gizmo_icon_label = "L"
    _gizmo_pass = "light"
    _gizmo_cache_attrs = ("light_type", "range", "spot_angle", "area_width", "area_height",
                          "area_type", "area_double_sided", "color")

    LUX_TO_RADIANCE = 1e-5
    CANDELA_TO_RADIANCE = 1e-2
    AREA_TO_RADIANCE = 1e-2
    LEGACY_DIRECTIONAL_MULT = 100000.0
    LEGACY_POINT_MULT = 2000.0
    LEGACY_AREA_MULT = 100.0
    _LIGHT_SCALE = 1.0

    @classmethod
    def set_light_scale(cls, scale: float) -> None:
        cls._LIGHT_SCALE = max(float(scale), 0.0)

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("light_type", "Type", FieldType.ENUM, enum_class=LightType),
            InspectorField("procedural_sky_lighting", "Procedural Sky Lighting", FieldType.BOOL),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("intensity", "Intensity (lux / lumens / nits)", FieldType.FLOAT, min_val=0.0, max_val=200000.0, step=10.0, decimals=1),
            InspectorField("range", "Range", FieldType.FLOAT, min_val=0.0, max_val=10000.0, step=0.5, decimals=2),
            InspectorField("spot_angle", "Spot Angle", FieldType.FLOAT, min_val=1.0, max_val=179.0, step=1.0, decimals=1),
            InspectorField("cast_shadows", "Cast Shadows", FieldType.BOOL),
            InspectorField("area_type", "Area Type", FieldType.ENUM, enum_class=LightAreaType),
            InspectorField("area_width", "Area Width", FieldType.FLOAT, min_val=0.01, max_val=100.0, step=0.1, decimals=2),
            InspectorField("area_height", "Area Height", FieldType.FLOAT, min_val=0.01, max_val=100.0, step=0.1, decimals=2),
            InspectorField("area_double_sided", "Double Sided", FieldType.BOOL),
            InspectorField("area_samples", "Samples", FieldType.INT, min_val=1, max_val=64, step=1),
            InspectorField("area_shadow_bias", "Shadow Bias", FieldType.FLOAT, min_val=0.0, max_val=0.1, step=0.001, decimals=4),
        ]

    def __init__(self):
        super().__init__()
        self.light_type: LightType = LightType.DIRECTIONAL
        self.color: list[float] = [1.0, 1.0, 1.0]
        self.intensity: float = 100000.0
        self.procedural_sky_lighting: bool = False
        self.range: float = 10.0
        self.spot_angle: float = 30.0
        self.spot_inner_angle: float = 20.0
        self.cast_shadows: bool = True
        self.area_type: LightAreaType = LightAreaType.RECT
        self.area_width: float = 1.0
        self.area_height: float = 1.0
        self.area_double_sided: bool = False
        self.area_samples: int = 6
        self.area_shadow_bias: float = 0.005

    def gizmo(self):
        tr = self.transform
        if not tr:
            return []
        from core.ecs.ecs import GizmoPrimitive
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
        if self.light_type == LightType.DIRECTIONAL:
            r = 0.3
            ray_len = 1.5
            pts = []
            for i in range(segments):
                a = 2.0 * math.pi * i / segments
                pts.append(pos + (right * math.cos(a) + up * math.sin(a)) * r)
            for i in range(segments):
                lines.append((pts[i], pts[(i + 1) % segments], col))
            spacing = 0.2
            for i in (-1, 0, 1):
                dx = right * (i * spacing)
                for j in (-1, 0, 1):
                    origin = pos + dx + up * (j * spacing)
                    tip = origin + fwd * ray_len
                    lines.append((origin, tip, col))
        elif self.light_type == LightType.POINT:
            rng = max(self.range, 0.1)
            for axis_idx in range(3):
                pts = []
                for i in range(segments):
                    a = 2.0 * math.pi * i / segments
                    if axis_idx == 0:
                        pt = right * 0.0 + up * (math.cos(a) * rng) + fwd * (math.sin(a) * rng)
                    elif axis_idx == 1:
                        pt = right * (math.cos(a) * rng) + up * 0.0 + fwd * (math.sin(a) * rng)
                    else:
                        pt = right * (math.cos(a) * rng) + up * (math.sin(a) * rng) + fwd * 0.0
                    pts.append(pos + pt)
                for i in range(segments):
                    lines.append((pts[i], pts[(i + 1) % segments], col))
            half = rng * 0.15
            for d in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                end = pos + right * d[0] * half + up * d[1] * half + fwd * d[2] * half
                lines.append((pos, end, col))
        elif self.light_type == LightType.SPOT:
            rng = max(self.range, 0.1)
            half_angle = self.spot_angle * 0.5 * math.pi / 180.0
            cone_r = rng * math.tan(half_angle)
            base_center = pos + fwd * rng
            base_pts = []
            for i in range(segments):
                a = 2.0 * math.pi * i / segments
                base_pts.append(base_center + (right * math.cos(a) + up * math.sin(a)) * cone_r)
            for i in range(segments):
                lines.append((base_pts[i], base_pts[(i + 1) % segments], col))
            for i in range(8):
                a = 2.0 * math.pi * i / 8
                bp = base_center + (right * math.cos(a) + up * math.sin(a)) * cone_r
                lines.append((pos, bp, col))
            lines.append((pos, base_center, col))
        elif self.light_type == LightType.AREA:
            hw = self.area_width * 0.5
            hh = self.area_height * 0.5
            corners = [
                pos - right * hw - up * hh,
                pos + right * hw - up * hh,
                pos + right * hw + up * hh,
                pos - right * hw + up * hh,
            ]
            for i in range(4):
                lines.append((corners[i], corners[(i + 1) % 4], col))
            mid_r = pos + fwd * 0.1
            lines.append((pos, mid_r, col))
            if self.area_type == LightAreaType.DISK:
                disk_pts = []
                for i in range(segments):
                    a = 2.0 * math.pi * i / segments
                    disk_pts.append(pos + (right * math.cos(a) + up * math.sin(a)) * hw)
                for i in range(segments):
                    lines.append((disk_pts[i], disk_pts[(i + 1) % segments], [col[0], col[1], col[2], 0.3]))
            if self.area_double_sided:
                back_corners = [
                    pos - fwd * 0.02 - right * hw - up * hh,
                    pos - fwd * 0.02 + right * hw - up * hh,
                    pos - fwd * 0.02 + right * hw + up * hh,
                    pos - fwd * 0.02 - right * hw + up * hh,
                ]
                dim_col = [col[0] * 0.4, col[1] * 0.4, col[2] * 0.4, col[3] * 0.5]
                for i in range(4):
                    lines.append((back_corners[i], back_corners[(i + 1) % 4], dim_col))
        if not lines:
            return []
        return [GizmoPrimitive.from_lines(lines)]

    @staticmethod
    def _planck_white(color_temp: float) -> list[float]:
        """Approximate black-body white point (Tanner Helland formula)."""
        t = max(float(color_temp), 1000.0) / 100.0
        if t <= 66.0:
            r = 255.0
            g = 99.4708025861 * math.log(t) - 161.1195681661
            b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307 if t > 19.0 else 0.0
        else:
            r = 329.698727446 * math.pow(t - 60.0, -0.1332047592)
            g = 288.1221695283 * math.pow(t - 60.0, -0.0755148492)
            b = 255.0
        r = max(0.0, min(255.0, r))
        g = max(0.0, min(255.0, g))
        b = max(0.0, min(255.0, b))
        m = max(r, g, b)
        if m <= 0.0:
            return [1.0, 1.0, 1.0]
        return [r / m, g / m, b / m]

    @staticmethod
    def compute_sun_light(sun_dir: Vec3, color_temp: float = 5778.0,
                          aerosol_scale: float = 1.0,
                          use_atmosphere: bool = True) -> tuple[list[float], float]:
        elevation = float(sun_dir.y)
        if elevation <= 0.0:
            night = max(0.0, min(1.0, -elevation * 2.0))
            moonlight = 0.02 * (1.0 - night * 0.75)
            return [0.3, 0.35, 0.55], moonlight
        # Kasten-Young air mass: how much atmosphere the sunlight crosses.
        e = max(elevation, 1e-4)
        e_deg = math.degrees(e)
        airmass = 1.0 / (e + 0.50572 * math.pow(e_deg + 6.07995, -1.6364))
        if not use_atmosphere:
            airmass = 1.0
        # Vertical optical depths at sea level, using the engine's atmosphere
        # constants (Hr = 8 km, Hm = 1.2 km). Aerosol scales the Mie term.
        wp = Light._planck_white(color_temp)
        tau_r = [5.802e-6 * 8000.0 * airmass,
                 13.558e-6 * 8000.0 * airmass,
                 33.1e-6 * 8000.0 * airmass]
        tau_m = 3.996e-6 * 1200.0 * max(float(aerosol_scale), 0.0) * airmass
        color = [wp[i] * math.exp(-(tau_r[i] + tau_m)) for i in range(3)]
        luma = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        if luma > 1e-6:
            color = [c / luma for c in color]
        else:
            color = [1.0, 0.7, 0.4]
        intensity = 1.2 * (1.0 - math.exp(-elevation * 4.0))
        tau_luma = 0.2126 * tau_r[0] + 0.7152 * tau_r[1] + 0.0722 * tau_r[2] + tau_m
        intensity *= (0.5 + 0.5 * math.exp(-tau_luma))
        return color, intensity

    @staticmethod
    def shader_radiance(light: Light, transform) -> tuple[list[float], float]:
        sc = Light._LIGHT_SCALE
        if light.light_type == LightType.DIRECTIONAL:
            if light.procedural_sky_lighting:
                color_temp = 5778.0
                aerosol = 1.0
                try:
                    from core.components.rendering.environment.atmosphere import Atmosphere
                    atmos = next((a for a in Atmosphere._registry
                                  if a.enabled and a.entity and a.entity.active), None)
                    if atmos is not None:
                        color_temp = float(getattr(atmos, "_color_temperature", 5778.0))
                        aerosol = float(getattr(atmos, "_aerosol_scale", 1.0))
                except Exception:
                    pass
                c, i = Light.compute_sun_light(-transform.forward, color_temp, aerosol)
                return [c[0], c[1], c[2]], i * sc
            return list(light.color), light.intensity * Light.LUX_TO_RADIANCE * sc
        if light.light_type == LightType.AREA:
            return list(light.color), light.intensity * Light.AREA_TO_RADIANCE * sc
        candela = max(float(light.intensity), 0.0) / (4.0 * math.pi)
        return list(light.color), candela * Light.CANDELA_TO_RADIANCE * sc

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "light_version": 2, "light_type": self.light_type.value, "color": self.color,
            "intensity": self.intensity, "range": self.range,
            "spot_angle": self.spot_angle, "spot_inner_angle": self.spot_inner_angle,
            "cast_shadows": self.cast_shadows,
            "procedural_sky_lighting": self.procedural_sky_lighting,
            "area_type": self.area_type.value, "area_width": self.area_width,
            "area_height": self.area_height, "area_double_sided": self.area_double_sided,
            "area_samples": self.area_samples, "area_shadow_bias": self.area_shadow_bias,
        })
        return d
    @classmethod
    def deserialize(cls, data: dict) -> Light:
        l = cls()
        l.enabled = data.get("enabled", True)
        l.light_type = LightType(data.get("light_type", "directional"))
        l.color = data.get("color", [1.0,1.0,1.0])
        l.intensity = data.get("intensity", 1.0)
        if data.get("light_version", 1) < 2:
            if l.light_type == LightType.DIRECTIONAL:
                l.intensity = l.intensity * Light.LEGACY_DIRECTIONAL_MULT
            elif l.light_type in (LightType.POINT, LightType.SPOT):
                l.intensity = l.intensity * Light.LEGACY_POINT_MULT
            else:
                l.intensity = l.intensity * Light.LEGACY_AREA_MULT
        l.procedural_sky_lighting = data.get("procedural_sky_lighting", False)
        l.range = data.get("range", 10.0)
        l.spot_angle = data.get("spot_angle", 30.0)
        l.spot_inner_angle = data.get("spot_inner_angle", 20.0)
        l.cast_shadows = data.get("cast_shadows", True)
        l.area_type = LightAreaType(data.get("area_type", "rect"))
        l.area_width = data.get("area_width", 1.0)
        l.area_height = data.get("area_height", 1.0)
        l.area_double_sided = data.get("area_double_sided", False)
        l.area_samples = data.get("area_samples", 6)
        l.area_shadow_bias = data.get("area_shadow_bias", 0.005)
        return l
