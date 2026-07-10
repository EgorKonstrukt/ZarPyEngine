# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
import math
import numpy as np
import moderngl
from core.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField, ListElementField
from core.math3d import Mat4, Vec3
from core.components.lighting.light import Light


_DEFAULT_WAVES = [
    {"direction": 12.0,  "amplitude": 0.30, "wavelength": 31.0, "speed": 0.9, "steepness": 0.42},
    {"direction": 57.0,  "amplitude": 0.21, "wavelength": 19.0, "speed": 1.0, "steepness": 0.48},
    {"direction": 103.0, "amplitude": 0.14, "wavelength": 12.0, "speed": 1.1, "steepness": 0.55},
    {"direction": 149.0, "amplitude": 0.10, "wavelength": 8.0,  "speed": 1.2, "steepness": 0.60},
    {"direction": 201.0, "amplitude": 0.07, "wavelength": 5.4,  "speed": 1.3, "steepness": 0.62},
    {"direction": 256.0, "amplitude": 0.045, "wavelength": 3.6, "speed": 1.4, "steepness": 0.68},
    {"direction": 309.0, "amplitude": 0.028, "wavelength": 2.4, "speed": 1.5, "steepness": 0.72},
    {"direction": 338.0, "amplitude": 0.018, "wavelength": 1.5, "speed": 1.6, "steepness": 0.78},
]


def _set_vec3(prog, name, value):
    if name not in prog:
        return
    if isinstance(value, (list, tuple)):
        arr = np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
    else:
        arr = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    prog[name].write(arr.tobytes())


def _set_float(prog, name, value):
    if name in prog:
        prog[name].value = float(value)


@ComponentRegistry.register
class Water(Component):
    _icon = "Water.png"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("material_path", "Water Shader", FieldType.RESOURCE_PATH, file_filter="Shader (*.shader)"),
            InspectorField("surface_type", "Surface Type", FieldType.ENUM, enum_options=["Ocean", "Pond"]),
            InspectorField("infinite_ocean", "Infinite Ocean", FieldType.BOOL),
            InspectorField("ocean_size", "Ocean Size", FieldType.FLOAT, min_val=10.0, max_val=5000.0, step=10.0, decimals=1),
            InspectorField("waves", "Waves", FieldType.LIST, element_fields=[
                ListElementField("direction", "Dir (deg)", FieldType.FLOAT, min_val=0.0, max_val=360.0, step=1.0, decimals=1),
                ListElementField("amplitude", "Amp", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.01, decimals=3),
                ListElementField("wavelength", "Length", FieldType.FLOAT, min_val=0.1, max_val=200.0, step=0.1, decimals=2),
                ListElementField("speed", "Speed", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.01, decimals=3),
                ListElementField("steepness", "Steep", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            ]),
            InspectorField("deep_color", "Deep Color", FieldType.COLOR),
            InspectorField("shallow_color", "Shallow Color", FieldType.COLOR),
            InspectorField("foam_color", "Foam Color", FieldType.COLOR),
            InspectorField("sss_color", "SSS Color", FieldType.COLOR),
            InspectorField("horizon_color", "Horizon Color", FieldType.COLOR),
            InspectorField("smoothness", "Smoothness", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("distortion", "Distortion", FieldType.SLIDER, min_val=0.0, max_val=0.3, step=0.005, decimals=3),
            InspectorField("normal_strength", "Detail Normal", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("wave_tiling", "Wave Tiling", FieldType.SLIDER, min_val=0.01, max_val=2.0, step=0.01, decimals=3),
            InspectorField("warp_amount", "Domain Warp", FieldType.SLIDER, min_val=0.0, max_val=4.0, step=0.05, decimals=3),
            InspectorField("detail_speed", "Detail Speed", FieldType.SLIDER, min_val=0.0, max_val=3.0, step=0.01, decimals=3),
            InspectorField("refract_strength", "Refraction Tint", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("fresnel_power", "Fresnel Power", FieldType.SLIDER, min_val=0.5, max_val=8.0, step=0.1, decimals=2),
            InspectorField("foam_strength", "Foam Strength", FieldType.SLIDER, min_val=0.0, max_val=3.0, step=0.05, decimals=2),
            InspectorField("specular", "Specular", FieldType.SLIDER, min_val=0.0, max_val=4.0, step=0.05, decimals=2),
            InspectorField("shore_fade", "Shore Fade", FieldType.FLOAT, min_val=0.1, max_val=20.0, step=0.1, decimals=2),
            InspectorField("choppiness", "Choppiness", FieldType.SLIDER, min_val=0.0, max_val=2.0, step=0.01, decimals=3),
            InspectorField("caustics", "Caustics", FieldType.SLIDER, min_val=0.0, max_val=2.0, step=0.01, decimals=3),
            InspectorField("wind_direction", "Wind Direction (deg)", FieldType.FLOAT, min_val=0.0, max_val=360.0, step=1.0, decimals=1),
            InspectorField("wind_speed", "Wind Speed (m/s)", FieldType.FLOAT, min_val=0.0, max_val=60.0, step=0.1, decimals=2),
            InspectorField("wind_turbulence", "Wind Turbulence", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("wind_influence", "Wind Influence", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("wind_gust", "Wind Gust", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("mesh_resolution", "Mesh Resolution", FieldType.SLIDER, min_val=2.0, max_val=400.0, step=1.0, decimals=0),
            InspectorField("chaos", "Chaos", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("macro_wave", "Macro Waves", FieldType.SLIDER, min_val=0.0, max_val=3.0, step=0.01, decimals=3),
            InspectorField("detail_scale", "Detail Scale", FieldType.SLIDER, min_val=0.1, max_val=4.0, step=0.01, decimals=3),
            InspectorField("detail_octaves", "Detail Octaves", FieldType.SLIDER, min_val=1.0, max_val=12.0, step=1.0, decimals=0),
            InspectorField("detail_fade", "Detail Fade Dist", FieldType.FLOAT, min_val=20.0, max_val=2000.0, step=10.0, decimals=0),
        ]

    def __init__(self):
        super().__init__()
        self._time_origin: float = time.time()
        self.material_path: str = "core/shaders/Water.shader"
        self.surface_type: str = "Ocean"
        self.infinite_ocean: bool = True
        self.ocean_size: float = 2000.0
        self.waves: list[dict] = [dict(w) for w in _DEFAULT_WAVES]
        self.deep_color: list[float] = [0.02, 0.18, 0.28]
        self.shallow_color: list[float] = [0.10, 0.42, 0.52]
        self.foam_color: list[float] = [0.92, 0.97, 1.0]
        self.sss_color: list[float] = [0.0, 0.55, 0.45]
        self.horizon_color: list[float] = [0.62, 0.78, 0.86]
        self.smoothness: float = 0.96
        self.distortion: float = 0.04
        self.normal_strength: float = 0.35
        self.wave_tiling: float = 0.25
        self.warp_amount: float = 1.6
        self.detail_speed: float = 1.0
        self.refract_strength: float = 0.55
        self.fresnel_power: float = 4.0
        self.foam_strength: float = 1.0
        self.specular: float = 1.0
        self.shore_fade: float = 3.0
        self.choppiness: float = 1.0
        self.caustics: float = 0.6
        self.wind_direction: float = 45.0
        self.wind_speed: float = 8.0
        self.wind_turbulence: float = 0.4
        self.wind_influence: float = 0.0
        self.wind_gust: float = 0.4
        self.mesh_resolution: float = 128.0
        self.chaos: float = 0.5
        self.macro_wave: float = 0.5
        self.detail_scale: float = 1.0
        self.detail_octaves: float = 6.0
        self.detail_fade: float = 350.0

    def _resolve_wind(self, wind_zones, wx, wz, t):
        best = None
        if wind_zones:
            for wz_comp in wind_zones:
                try:
                    s = wz_comp.sample(wx, wz, t)
                except Exception:
                    s = None
                if not s or s.get("strength", 0.0) <= 0.0:
                    continue
                strength = s.get("strength", 1.0)
                if best is None or strength > best[0]:
                    best = (strength, s)
        if best is not None:
            s = best[1]
            return {
                "dir": s.get("dir", (1.0, 0.0)),
                "speed": s.get("speed", 0.0),
                "turbulence": s.get("turbulence", 0.0),
                "gust": s.get("gust", 0.0),
                "influence": s.get("alignment", self.wind_influence),
            }
        ang = math.radians(self.wind_direction)
        dx, dz = math.cos(ang), math.sin(ang)
        gust = self.wind_gust * (0.5 + 0.5 * math.sin(2.0 * math.pi * 0.15 * t))
        return {
            "dir": (dx, dz),
            "speed": self.wind_speed,
            "turbulence": self.wind_turbulence,
            "gust": max(0.0, gust),
            "influence": self.wind_influence,
        }

    def render_water(self, ctx, shaders, view_mat, proj_mat, dir_light, cam_pos,
                      water_mesh, scene_color_tex, scene_depth_tex, viewport_size, cam_near, cam_far,
                      wind_zones=None, lights=None, chunk_models=None, is_box=False):
        prog = shaders.get_or_compile(self.material_path) if shaders else None
        if not prog:
            return

        if dir_light:
            dl, dt = dir_light
            sun_dir = -dt.forward
            if "_SunDirection" in prog:
                prog["_SunDirection"].write(np.array([sun_dir.x, sun_dir.y, sun_dir.z], dtype=np.float32).tobytes())
            if "_SunColor" in prog:
                if dl.procedural_sky_lighting:
                    sc, si = Light.compute_sun_light(-dt.forward)
                    prog["_SunColor"].write(np.array(sc, dtype=np.float32).tobytes())
                    if "_SunIntensity" in prog:
                        prog["_SunIntensity"].value = si
                else:
                    prog["_SunColor"].write(np.array(dl.color, dtype=np.float32).tobytes())
                    if "_SunIntensity" in prog:
                        prog["_SunIntensity"].value = dl.intensity

        t = time.time() - self._time_origin
        if "u_time" in prog:
            prog["u_time"].value = t
        if "_Time" in prog:
            prog["_Time"].value = t

        tr = self.transform
        water_y = tr.position.y if tr else 0.0
        if chunk_models:
            models = list(chunk_models)
            water_wx, water_wz = cam_pos.x, cam_pos.z
        elif self.infinite_ocean and self.surface_type == "Ocean":
            models = [Mat4.scale(Vec3(self.ocean_size, 1.0, self.ocean_size)) * \
                      Mat4.translation(Vec3(cam_pos.x, water_y, cam_pos.z))]
            water_wx, water_wz = cam_pos.x, cam_pos.z
        else:
            models = [tr.world_matrix if tr else Mat4.identity()]
            water_wx = tr.position.x if tr else 0.0
            water_wz = tr.position.z if tr else 0.0
        if "u_view" in prog:
            prog["u_view"].write(view_mat.to_f32().tobytes())
        if "u_proj" in prog:
            prog["u_proj"].write(proj_mat.to_f32().tobytes())
        if "u_camera_pos" in prog:
            prog["u_camera_pos"].write(np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32).tobytes())

        wind = self._resolve_wind(wind_zones, water_wx, water_wz, t)
        wdir = wind["dir"]
        wind_speed = wind["speed"]
        wind_infl = wind["influence"]
        # Coherent wind alignment: even without manually setting Wind
        # Influence, stronger wind steers the waves toward the wind heading.
        wind_norm = min(max(wind_speed / 60.0, 0.0), 1.0)
        wind_align = min(1.0, wind_infl + wind_norm * 0.8)

        if "_WaveCount" in prog:
            n = min(len(self.waves), 8)
            prog["_WaveCount"].value = n
            dirs = np.zeros((8, 2), dtype=np.float32)
            params = np.zeros((8, 4), dtype=np.float32)
            for i in range(n):
                w = self.waves[i]
                ang = math.radians(float(w.get("direction", 0.0)))
                dirs[i] = [math.cos(ang), math.sin(ang)]
                params[i] = [
                    float(w.get("amplitude", 0.0)),
                    float(w.get("wavelength", 1.0)),
                    float(w.get("speed", 1.0)),
                    float(w.get("steepness", 0.0)),
                ]
            prog["_WaveDirection"].write(dirs.tobytes())
            prog["_WaveParams"].write(params.tobytes())

        if "_WindDir" in prog:
            prog["_WindDir"].write(np.array([float(wdir[0]), float(wdir[1])], dtype=np.float32).tobytes())
        _set_float(prog, "_WindSpeed", wind_speed)
        _set_float(prog, "_WindGust", wind["gust"])
        _set_float(prog, "_WindTurbulence", wind["turbulence"])
        _set_float(prog, "_WindAlign", wind_align)
        _set_float(prog, "_Choppiness", self.choppiness)
        _set_float(prog, "_Caustics", self.caustics)
        _set_float(prog, "_MacroWave", self.macro_wave)
        _set_float(prog, "_Chaos", self.chaos)
        _set_float(prog, "_DetailScale", self.detail_scale)
        _set_float(prog, "_DetailOctaves", self.detail_octaves)
        _set_float(prog, "_DetailFade", self.detail_fade)
        if "_IsBox" in prog:
            prog["_IsBox"].value = 1 if is_box else 0

        if "_LightCount" in prog:
            from core.components.lighting import LightType
            lpos = np.zeros((16, 3), dtype=np.float32)
            lcol = np.zeros((16, 3), dtype=np.float32)
            lint = np.zeros(16, dtype=np.float32)
            lrange = np.zeros(16, dtype=np.float32)
            ldir = np.zeros((16, 3), dtype=np.float32)
            lspot = np.full(16, -1.0, dtype=np.float32)
            count = 0
            if lights:
                for l, lt in lights:
                    if not l or not l.enabled or lt is None:
                        continue
                    if l.light_type == LightType.DIRECTIONAL:
                        continue
                    if count >= 16:
                        break
                    pos = lt.position
                    lpos[count] = [pos.x, pos.y, pos.z]
                    col = l.color if isinstance(l.color, (list, tuple)) else [1.0, 1.0, 1.0]
                    lcol[count] = [float(col[0]), float(col[1]), float(col[2])]
                    lint[count] = float(l.intensity)
                    lrange[count] = max(float(l.range), 0.001)
                    if l.light_type == LightType.SPOT:
                        fwd = lt.forward
                        ldir[count] = [fwd.x, fwd.y, fwd.z]
                        half = math.radians(max(float(l.spot_angle), 1.0)) * 0.5
                        lspot[count] = math.cos(half)
                    count += 1
            prog["_LightCount"].value = count
            prog["_LightPos"].write(lpos.tobytes())
            prog["_LightColor"].write(lcol.tobytes())
            prog["_LightIntensity"].write(lint.tobytes())
            prog["_LightRange"].write(lrange.tobytes())
            prog["_LightDir"].write(ldir.tobytes())
            prog["_LightSpotCos"].write(lspot.tobytes())

        _set_vec3(prog, "_DeepColor", self.deep_color)
        _set_vec3(prog, "_ShallowColor", self.shallow_color)
        _set_vec3(prog, "_FoamColor", self.foam_color)
        _set_vec3(prog, "_SSSColor", self.sss_color)
        _set_vec3(prog, "_HorizonColor", self.horizon_color)
        _set_float(prog, "_Smoothness", self.smoothness)
        _set_float(prog, "_Distortion", self.distortion)
        _set_float(prog, "_NormalStrength", self.normal_strength)
        _set_float(prog, "_WaveTiling", self.wave_tiling)
        _set_float(prog, "_WarpAmount", self.warp_amount)
        _set_float(prog, "_DetailSpeed", self.detail_speed)
        _set_float(prog, "_RefractStrength", self.refract_strength)
        _set_float(prog, "_FresnelPower", self.fresnel_power)
        _set_float(prog, "_FoamStrength", self.foam_strength)
        _set_float(prog, "_Specular", self.specular)
        _set_float(prog, "_ShoreFade", self.shore_fade)

        if scene_color_tex is not None and "_SceneColor" in prog:
            scene_color_tex.use(15)
            prog["_SceneColor"] = 15
            if "_HasScene" in prog:
                prog["_HasScene"].value = 1
        elif "_HasScene" in prog:
            prog["_HasScene"].value = 0
        if scene_depth_tex is not None and "_SceneDepth" in prog:
            scene_depth_tex.use(16)
            prog["_SceneDepth"] = 16
        if "_ViewportSize" in prog:
            prog["_ViewportSize"].value = (float(viewport_size[0]), float(viewport_size[1]))
        if "_CamNear" in prog:
            prog["_CamNear"].value = float(cam_near)
        if "_CamFar" in prog:
            prog["_CamFar"].value = float(cam_far)

        old_depth_mask = ctx.depth_mask
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.disable(moderngl.CULL_FACE)
        try:
            for model in models:
                if "u_model" in prog:
                    prog["u_model"].write(model.to_f32().tobytes())
                water_mesh.render(prog)
        finally:
            ctx.enable(moderngl.CULL_FACE)
            ctx.disable(moderngl.BLEND)
            ctx.depth_mask = old_depth_mask

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "material_path": self.material_path,
            "surface_type": self.surface_type,
            "infinite_ocean": self.infinite_ocean,
            "ocean_size": self.ocean_size,
            "waves": [dict(w) for w in self.waves],
            "deep_color": self.deep_color,
            "shallow_color": self.shallow_color,
            "foam_color": self.foam_color,
            "sss_color": self.sss_color,
            "horizon_color": self.horizon_color,
            "smoothness": self.smoothness,
            "distortion": self.distortion,
            "normal_strength": self.normal_strength,
            "wave_tiling": self.wave_tiling,
            "warp_amount": self.warp_amount,
            "detail_speed": self.detail_speed,
            "refract_strength": self.refract_strength,
            "fresnel_power": self.fresnel_power,
            "foam_strength": self.foam_strength,
            "specular": self.specular,
            "shore_fade": self.shore_fade,
            "choppiness": self.choppiness,
            "caustics": self.caustics,
            "wind_direction": self.wind_direction,
            "wind_speed": self.wind_speed,
            "wind_turbulence": self.wind_turbulence,
            "wind_influence": self.wind_influence,
            "wind_gust": self.wind_gust,
            "mesh_resolution": self.mesh_resolution,
            "chaos": self.chaos,
            "macro_wave": self.macro_wave,
            "detail_scale": self.detail_scale,
            "detail_octaves": self.detail_octaves,
            "detail_fade": self.detail_fade,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Water:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.material_path = data.get("material_path", "core/shaders/Water.shader")
        c.surface_type = data.get("surface_type", "Ocean")
        c.infinite_ocean = data.get("infinite_ocean", True)
        c.ocean_size = data.get("ocean_size", 2000.0)
        raw_waves = data.get("waves")
        if raw_waves:
            c.waves = [dict(w) for w in raw_waves]
        else:
            c.waves = [dict(w) for w in _DEFAULT_WAVES]
        c.deep_color = data.get("deep_color", [0.02, 0.18, 0.28])
        c.shallow_color = data.get("shallow_color", [0.10, 0.42, 0.52])
        c.foam_color = data.get("foam_color", [0.92, 0.97, 1.0])
        c.sss_color = data.get("sss_color", [0.0, 0.55, 0.45])
        c.horizon_color = data.get("horizon_color", [0.62, 0.78, 0.86])
        c.smoothness = data.get("smoothness", 0.96)
        c.distortion = data.get("distortion", 0.04)
        c.normal_strength = data.get("normal_strength", 0.35)
        c.wave_tiling = data.get("wave_tiling", 0.25)
        c.warp_amount = data.get("warp_amount", 1.6)
        c.detail_speed = data.get("detail_speed", 1.0)
        c.refract_strength = data.get("refract_strength", 0.55)
        c.fresnel_power = data.get("fresnel_power", 4.0)
        c.foam_strength = data.get("foam_strength", 1.0)
        c.specular = data.get("specular", 1.0)
        c.shore_fade = data.get("shore_fade", 3.0)
        c.choppiness = data.get("choppiness", 1.0)
        c.caustics = data.get("caustics", 0.6)
        c.wind_direction = data.get("wind_direction", 45.0)
        c.wind_speed = data.get("wind_speed", 8.0)
        c.wind_turbulence = data.get("wind_turbulence", 0.4)
        c.wind_influence = data.get("wind_influence", 0.0)
        c.wind_gust = data.get("wind_gust", 0.4)
        c.mesh_resolution = data.get("mesh_resolution", 128.0)
        c.chaos = data.get("chaos", 0.5)
        c.macro_wave = data.get("macro_wave", 0.5)
        c.detail_scale = data.get("detail_scale", 1.0)
        c.detail_octaves = data.get("detail_octaves", 6.0)
        c.detail_fade = data.get("detail_fade", 350.0)
        return c