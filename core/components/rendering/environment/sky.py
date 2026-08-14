# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import time
import numpy as np
import moderngl
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Mat4, Vec3
from core.components.lighting.light import Light
from core.components.rendering.environment.sky_ibl import get_sky_ibl, release_sky_ibl_cache, get_procedural_sky_ibl
from core.components.rendering.environment.atmosphere import Atmosphere

_ENV_TEX_CACHE: dict[str, tuple[float, object]] = {}


def _decode_rgbe_scanline_rle(raw: bytes, pos: int, width: int):
    channels = [np.empty(width, dtype=np.uint8) for _ in range(4)]
    for ch in channels:
        filled = 0
        while filled < width:
            t = raw[pos]
            pos += 1
            if t > 128:
                cnt = t - 128
                val = raw[pos]
                pos += 1
                ch[filled:filled + cnt] = val
            else:
                cnt = t
                ch[filled:filled + cnt] = np.frombuffer(raw[pos:pos + cnt], dtype=np.uint8)
                pos += cnt
            filled += cnt
    return channels, pos


def _decode_rgbe(data: bytes):
    line_end = data.index(b"\n")
    if not data[:line_end].strip().startswith(b"#?"):
        raise ValueError("not radiance hdr")
    pos = line_end + 1
    while True:
        line_end = data.index(b"\n", pos)
        line = data[pos:line_end].strip()
        pos = line_end + 1
        if not line:
            break
        if line.lower().startswith(b"format=") and b"32-bit_rle_rgbe" not in line.lower():
            raise ValueError("unsupported hdr format")
    res_end = data.index(b"\n", pos)
    parts = data[pos:res_end].strip().split()
    pos = res_end + 1
    if len(parts) != 4:
        raise ValueError("bad resolution line")
    if parts[0] == b"-Y":
        height = int(parts[1])
        width = int(parts[3])
        flip = False
    elif parts[0] == b"+Y":
        height = int(parts[1])
        width = int(parts[3])
        flip = True
    else:
        raise ValueError("bad resolution line")
    out = np.empty((height, width, 3), dtype=np.float32)
    for y in range(height):
        a = data[pos]
        b = data[pos + 1]
        if a == 2 and b == 2:
            w = (data[pos + 2] << 8) | data[pos + 3]
            if w != width:
                raise ValueError("scanline width mismatch")
            pos += 4
            chans, pos = _decode_rgbe_scanline_rle(data, pos, width)
            rr, gg, bb, ee = chans
        elif a == 2 and b < 128:
            w = b
            if w != width:
                raise ValueError("scanline width mismatch")
            pos += 2
            rr = np.empty(width, dtype=np.uint8)
            gg = np.empty(width, dtype=np.uint8)
            bb = np.empty(width, dtype=np.uint8)
            ee = np.empty(width, dtype=np.uint8)
            filled = 0
            while filled < width:
                t = data[pos]
                pos += 1
                if t > 128:
                    cnt = t - 128
                    val = data[pos]
                    pos += 1
                    rr[filled:filled + cnt] = val
                    gg[filled:filled + cnt] = val
                    bb[filled:filled + cnt] = val
                    ee[filled:filled + cnt] = val
                else:
                    cnt = t
                    seg = np.frombuffer(data[pos:pos + cnt * 4], dtype=np.uint8).reshape(cnt, 4)
                    pos += cnt * 4
                    rr[filled:filled + cnt] = seg[:, 0]
                    gg[filled:filled + cnt] = seg[:, 1]
                    bb[filled:filled + cnt] = seg[:, 2]
                    ee[filled:filled + cnt] = seg[:, 3]
                filled += cnt
        else:
            pixels = np.frombuffer(
                bytes([a, b]) + data[pos:pos + width * 4 - 2], dtype=np.uint8
            ).reshape(width, 4)
            pos += width * 4 - 2
            rr, gg, bb, ee = pixels[:, 0], pixels[:, 1], pixels[:, 2], pixels[:, 3]
        scale = np.exp2((ee.astype(np.int32) - 136).astype(np.float32))
        row = np.empty((width, 3), dtype=np.float32)
        row[:, 0] = rr.astype(np.float32) * scale
        row[:, 1] = gg.astype(np.float32) * scale
        row[:, 2] = bb.astype(np.float32) * scale
        out[y if not flip else height - 1 - y] = row
    return out


def _load_env_float(path: str):
    ext = os.path.splitext(path)[1].lower()
    img = None
    try:
        import imageio.v2 as iio
        img = iio.imread(path)
    except Exception:
        img = None
    if img is None and ext == ".hdr":
        try:
            with open(path, "rb") as f:
                img = _decode_rgbe(f.read())
        except Exception:
            img = None
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    arr = arr[:, :, :3]
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return np.ascontiguousarray(arr)


def _get_env_texture(ctx: moderngl.Context, path: str):
    if not path:
        return None
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return None
    mtime = os.path.getmtime(abs_path)
    cached = _ENV_TEX_CACHE.get(abs_path)
    if cached is not None:
        cm, tex = cached
        if abs(mtime - cm) < 0.001:
            return tex
        if tex is not None:
            try:
                tex.release()
            except Exception:
                pass
    arr = _load_env_float(abs_path)
    if arr is None:
        _ENV_TEX_CACHE[abs_path] = (mtime, None)
        return None
    h, w = arr.shape[:2]
    tex = ctx.texture((w, h), 3, arr.tobytes(), dtype="f4")
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = True
    tex.repeat_y = True
    _ENV_TEX_CACHE[abs_path] = (mtime, tex)
    return tex


_MOON_TEX_CACHE: dict[str, tuple[float, object]] = {}
_WHITE_TEX: object = None


def _load_moon_float(path: str):
    try:
        import imageio.v2 as iio
        img = iio.imread(path)
    except Exception:
        return None
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return np.ascontiguousarray(np.clip(arr, 0.0, 1.0))


def _get_moon_texture(ctx: moderngl.Context, path: str):
    if not path:
        return None
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return None
    mtime = os.path.getmtime(abs_path)
    cached = _MOON_TEX_CACHE.get(abs_path)
    if cached is not None:
        cm, tex = cached
        if abs(mtime - cm) < 0.001:
            return tex
        if tex is not None:
            try:
                tex.release()
            except Exception:
                pass
    arr = _load_moon_float(abs_path)
    if arr is None:
        _MOON_TEX_CACHE[abs_path] = (mtime, None)
        return None
    h, w = arr.shape[:2]
    tex = ctx.texture((w, h), 3, arr.tobytes(), dtype="f4")
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = False
    tex.repeat_y = False
    _MOON_TEX_CACHE[abs_path] = (mtime, tex)
    return tex


def _get_white_tex(ctx: moderngl.Context):
    global _WHITE_TEX
    if _WHITE_TEX is None:
        _WHITE_TEX = ctx.texture((1, 1), 3, b"\xff\xff\xff")
    return _WHITE_TEX


def release_env_cache():
    for _m, tex in _ENV_TEX_CACHE.values():
        if tex is not None:
            try:
                tex.release()
            except Exception:
                pass
    _ENV_TEX_CACHE.clear()
    for _m, tex in _MOON_TEX_CACHE.values():
        if tex is not None:
            try:
                tex.release()
            except Exception:
                pass
    _MOON_TEX_CACHE.clear()
    global _WHITE_TEX
    if _WHITE_TEX is not None:
        try:
            _WHITE_TEX.release()
        except Exception:
            pass
        _WHITE_TEX = None
    release_sky_ibl_cache()


@ComponentRegistry.register
class Sky(Component):
    _icon = "Sky.png"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("", "Procedural Night Sky", FieldType.HEADER),
            InspectorField("night_sky_enabled", "Night Sky", FieldType.BOOL),
            InspectorField("night_exposure", "Night Exposure", FieldType.SLIDER, min_val=0.0, max_val=3.0, step=0.1, decimals=1),
            InspectorField("", "Stars", FieldType.HEADER),
            InspectorField("star_enabled", "Stars", FieldType.BOOL),
            InspectorField("star_density", "Density", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
            InspectorField("star_intensity", "Intensity", FieldType.SLIDER, min_val=0.0, max_val=5.0, step=0.1, decimals=1),
            InspectorField("star_scale", "Scale", FieldType.SLIDER, min_val=20.0, max_val=200.0, step=1.0, decimals=0),
            InspectorField("star_twinkle", "Twinkle", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.05, decimals=2),
            InspectorField("star_seed", "Seed", FieldType.SLIDER, min_val=0.0, max_val=100.0, step=0.1, decimals=1),
            InspectorField("star_color", "Tint", FieldType.COLOR),
            InspectorField("", "Milky Way", FieldType.HEADER),
            InspectorField("milky_way_enabled", "Milky Way", FieldType.BOOL),
            InspectorField("milky_way_intensity", "Intensity", FieldType.SLIDER, min_val=0.0, max_val=3.0, step=0.1, decimals=1),
            InspectorField("milky_way_pole", "Band Pole", FieldType.VEC3),
            InspectorField("", "Moon", FieldType.HEADER),
            InspectorField("moon_enabled", "Moon", FieldType.BOOL),
            InspectorField("moon_direction", "Direction", FieldType.VEC3),
            InspectorField("moon_size", "Angular Radius (deg)", FieldType.SLIDER, min_val=0.05, max_val=1.5, step=0.01, decimals=2),
            InspectorField("moon_intensity", "Intensity", FieldType.SLIDER, min_val=0.0, max_val=5.0, step=0.1, decimals=1),
            InspectorField("moon_phase", "Phase", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
            InspectorField("moon_orbit_speed", "Orbit Speed (deg/s)", FieldType.SLIDER, min_val=0.0, max_val=20.0, step=0.1, decimals=1),
            InspectorField("moon_texture_path", "Texture", FieldType.RESOURCE_PATH, file_filter="Images (*.png *.jpg *.jpeg *.tga *.bmp)"),
        ]

    def __init__(self):
        super().__init__()
        self.material_path: str = "core/shaders/Sky.shader"
        self.environment_path: str = ""
        self.night_sky_enabled: bool = True
        self.night_exposure: float = 1.0
        self.star_enabled: bool = True
        self.star_density: float = 0.45
        self.star_intensity: float = 1.0
        self.star_scale: float = 80.0
        self.star_twinkle: float = 0.5
        self.star_seed: float = 1.0
        self.star_color: list[float] = [0.9, 0.93, 1.0]
        self.milky_way_enabled: bool = True
        self.milky_way_intensity: float = 0.6
        self.milky_way_pole: Vec3 = Vec3(0.4, 0.3, 0.85)
        self.moon_enabled: bool = True
        self.moon_direction: Vec3 = Vec3(0.25, 0.6, 0.75)
        self.moon_size: float = 0.27
        self.moon_intensity: float = 1.0
        self.moon_phase: float = 1.0
        self.moon_orbit_speed: float = 2.0
        self.moon_texture_path: str = "core/textures/moon.tga"
        self._sky_ibl = None

    def _night_settings_key(self) -> tuple:
        return (
            self.night_sky_enabled, self.night_exposure,
            self.star_enabled, self.star_density, self.star_intensity,
            self.star_scale, self.star_twinkle, self.star_seed,
            tuple(self.star_color),
            self.milky_way_enabled, self.milky_way_intensity,
            tuple(self.milky_way_pole),
            self.moon_enabled, tuple(self.moon_direction),
            self.moon_size, self.moon_intensity,
            self.moon_phase, self.moon_orbit_speed, self.moon_texture_path,
        )

    def _apply_night_sky(self, prog, ctx=None):
        def f(name, val):
            if name in prog:
                prog[name].value = float(val)

        def v3(name, val):
            if name in prog:
                if isinstance(val, Vec3):
                    val = [val.x, val.y, val.z]
                prog[name].write(np.array(val, dtype=np.float32).tobytes())

        f("_NightSkyEnabled", 1.0 if self.night_sky_enabled else 0.0)
        f("_NightExposure", self.night_exposure)
        f("_StarEnabled", 1.0 if self.star_enabled else 0.0)
        f("_StarDensity", self.star_density)
        f("_StarIntensity", self.star_intensity)
        f("_StarScale", self.star_scale)
        f("_StarTwinkle", self.star_twinkle)
        f("_StarSeed", self.star_seed)
        v3("_StarColor", self.star_color)
        f("_MilkyWayEnabled", 1.0 if self.milky_way_enabled else 0.0)
        f("_MilkyWayIntensity", self.milky_way_intensity)
        v3("_MilkyWayPole", self.milky_way_pole)
        f("_MoonEnabled", 1.0 if self.moon_enabled else 0.0)
        v3("_MoonDirection", self.moon_direction)
        f("_MoonSize", self.moon_size)
        f("_MoonIntensity", self.moon_intensity)
        f("_MoonPhase", self.moon_phase)
        f("_MoonOrbitSpeed", self.moon_orbit_speed)
        if ctx is not None and "u_moon_tex" in prog:
            moon_tex = _get_moon_texture(ctx, self.moon_texture_path)
            if moon_tex is None:
                moon_tex = _get_white_tex(ctx)
            moon_tex.use(4)
            prog["u_moon_tex"].value = 4
            prog["u_use_moon_tex"].value = 1.0 if self.moon_texture_path else 0.0
        if "u_time" in prog:
            prog["u_time"].value = time.time()

    def render_sky(self, ctx, shaders, view_mat, proj_mat, dir_light, cube_mesh):
        prog = shaders.get_or_compile(self.material_path) if shaders else None
        if not prog:
            return
        self._apply_night_sky(prog, ctx)

        def set_sun_disk_defaults(p):
            if "_SunAngularRadius" in p:
                p["_SunAngularRadius"].value = 0.27
            if "_SunLimbDarkening" in p:
                p["_SunLimbDarkening"].value = 0.7
            if "_SunConvergence" in p:
                p["_SunConvergence"].value = 0.5

        sun_world = None
        sun_color = None
        sun_intensity = None
        if dir_light:
            dl, dt = dir_light
            sky_color, sky_intensity = Light.shader_radiance(dl, dt)
            if "_SunDirection" in prog:
                sun_dir = -dt.forward
                prog["_SunDirection"].write(np.array([sun_dir.x, sun_dir.y, sun_dir.z], dtype=np.float32).tobytes())
            if "_SunColor" in prog:
                prog["_SunColor"].write(np.array(sky_color, dtype=np.float32).tobytes())
            if "_SunIntensity" in prog:
                prog["_SunIntensity"].value = sky_intensity
            set_sun_disk_defaults(prog)
            sun_world = (-dt.forward.x, -dt.forward.y, -dt.forward.z)
            sun_color = sky_color
            sun_intensity = sky_intensity
        else:
            if "_SunDirection" in prog:
                prog["_SunDirection"].write(np.array([0.0, -0.3, -1.0], dtype=np.float32).tobytes())
            if "_SunColor" in prog:
                prog["_SunColor"].write(np.array([1.0, 0.95, 0.85], dtype=np.float32).tobytes())
            if "_SunIntensity" in prog:
                prog["_SunIntensity"].value = 1.0
            set_sun_disk_defaults(prog)
            sun_world = (0.0, -0.3, -1.0)
            sun_color = [1.0, 0.95, 0.85]
            sun_intensity = 1.0
        atmos = next((a for a in Atmosphere._registry
                      if a.enabled and a.entity and a.entity.active), None)
        if atmos is not None and "u_use_atmosphere" in prog:
            if atmos.ensure_luts(ctx, sun_world, sun_color, sun_intensity):
                atmos.bind_sky(prog)
            else:
                prog["u_use_atmosphere"].value = 0
        elif "u_use_atmosphere" in prog:
            prog["u_use_atmosphere"].value = 0
        env_tex = _get_env_texture(ctx, self.environment_path) if self.environment_path else None
        if self.environment_path:
            self._sky_ibl = get_sky_ibl(ctx, self.environment_path, env_tex)
        else:
            sun_dir = None
            sun_color = None
            sun_intensity = None
            if dir_light:
                dl, dt = dir_light
                sky_c, sky_i = Light.shader_radiance(dl, dt)
                sun_dir = (-dt.forward.x, -dt.forward.y, -dt.forward.z)
                sun_color = sky_c
                sun_intensity = sky_i
            self._sky_ibl = get_procedural_sky_ibl(ctx, prog, self.material_path,
                                                   sun_dir, sun_color, sun_intensity,
                                                   settings_key=self._night_settings_key())
        if "u_env_tex" in prog:
            if env_tex is not None:
                env_tex.use(0)
                prog["u_env_tex"].value = 0
                if "u_use_env" in prog:
                    prog["u_use_env"].value = 1
            elif "u_use_env" in prog:
                prog["u_use_env"].value = 0
        sky_view = np.eye(4, dtype=np.float64)
        sky_view[:3, :3] = view_mat._d[:3, :3].copy()
        sky_mat4 = Mat4(sky_view)
        mvp = sky_mat4 * proj_mat
        if "u_mvp" in prog:
            prog["u_mvp"].write(mvp.to_f32().tobytes())
        ctx.disable(moderngl.CULL_FACE)
        ctx.disable(moderngl.DEPTH_TEST)
        cube_mesh.render(prog)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)

    def serialize(self) -> dict:
        d = super().serialize()
        d["material_path"] = self.material_path
        d["environment_path"] = self.environment_path
        d["night_sky_enabled"] = self.night_sky_enabled
        d["night_exposure"] = self.night_exposure
        d["star_enabled"] = self.star_enabled
        d["star_density"] = self.star_density
        d["star_intensity"] = self.star_intensity
        d["star_scale"] = self.star_scale
        d["star_twinkle"] = self.star_twinkle
        d["star_seed"] = self.star_seed
        d["star_color"] = self.star_color
        d["milky_way_enabled"] = self.milky_way_enabled
        d["milky_way_intensity"] = self.milky_way_intensity
        d["milky_way_pole"] = self.milky_way_pole.to_list()
        d["moon_enabled"] = self.moon_enabled
        d["moon_direction"] = self.moon_direction.to_list()
        d["moon_size"] = self.moon_size
        d["moon_intensity"] = self.moon_intensity
        d["moon_phase"] = self.moon_phase
        d["moon_orbit_speed"] = self.moon_orbit_speed
        d["moon_texture_path"] = self.moon_texture_path
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Sky:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.material_path = data.get("material_path", "core/shaders/Sky.shader")
        c.environment_path = data.get("environment_path", "")
        c.night_sky_enabled = data.get("night_sky_enabled", True)
        c.night_exposure = data.get("night_exposure", 1.0)
        c.star_enabled = data.get("star_enabled", True)
        c.star_density = data.get("star_density", 0.45)
        c.star_intensity = data.get("star_intensity", 1.0)
        c.star_scale = data.get("star_scale", 80.0)
        c.star_twinkle = data.get("star_twinkle", 0.5)
        c.star_seed = data.get("star_seed", 1.0)
        c.star_color = data.get("star_color", [0.9, 0.93, 1.0])
        c.milky_way_enabled = data.get("milky_way_enabled", True)
        c.milky_way_intensity = data.get("milky_way_intensity", 0.6)
        c.milky_way_pole = Vec3(*data.get("milky_way_pole", [0.4, 0.3, 0.85]))
        c.moon_enabled = data.get("moon_enabled", True)
        c.moon_direction = Vec3(*data.get("moon_direction", [0.25, 0.6, 0.75]))
        c.moon_size = data.get("moon_size", 0.27)
        c.moon_intensity = data.get("moon_intensity", 1.0)
        c.moon_phase = data.get("moon_phase", 1.0)
        c.moon_orbit_speed = data.get("moon_orbit_speed", 2.0)
        c.moon_texture_path = data.get("moon_texture_path", "core/textures/moon.tga")
        return c
