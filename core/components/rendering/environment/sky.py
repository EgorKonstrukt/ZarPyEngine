# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import numpy as np
import moderngl
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Mat4
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


def release_env_cache():
    for _m, tex in _ENV_TEX_CACHE.values():
        if tex is not None:
            try:
                tex.release()
            except Exception:
                pass
    _ENV_TEX_CACHE.clear()
    release_sky_ibl_cache()


@ComponentRegistry.register
class Sky(Component):
    _icon = "Sky.png"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("material_path", "Sky Shader", FieldType.RESOURCE_PATH, file_filter="Shader (*.shader)"),
            InspectorField("environment_path", "Environment Map", FieldType.RESOURCE_PATH, file_filter="HDR/EXR (*.hdr *.exr)"),
        ]

    def __init__(self):
        super().__init__()
        self.material_path: str = "core/shaders/Sky.shader"
        self.environment_path: str = ""
        self._sky_ibl = None

    def render_sky(self, ctx, shaders, view_mat, proj_mat, dir_light, cube_mesh):
        prog = shaders.get_or_compile(self.material_path) if shaders else None
        if not prog:
            return
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
            if "_SunSize" in prog:
                prog["_SunSize"].value = 0.0008
            if "_SunConvergence" in prog:
                prog["_SunConvergence"].value = 0.5
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
            if "_SunSize" in prog:
                prog["_SunSize"].value = 0.0008
            if "_SunConvergence" in prog:
                prog["_SunConvergence"].value = 0.5
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
                                                   sun_dir, sun_color, sun_intensity)
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
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Sky:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.material_path = data.get("material_path", "core/shaders/Sky.shader")
        c.environment_path = data.get("environment_path", "")
        return c
