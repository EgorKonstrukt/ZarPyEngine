# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import os
import time
from collections import OrderedDict
import numpy as np
import moderngl
from typing import Optional
from core.components.rendering.environment.dynamic_cubemap import (
    _allocate_cube_mip_levels,
    _BRDF_LUT_FRAG,
    _FACE_BASIS,
    _FACE_DIRS,
    _FULLSCREEN_QUAD_VERT,
    _IRRADIANCE_FRAG,
    _PREFILTER_FRAG,
    _PREFILTER_MAX_LOD,
    _restore_framebuffer,
    _write_cube_face_mip,
)

_EQUIRECT_TO_CUBE_FRAG = """
#version 460 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_equirect;
uniform vec3 u_face_x;
uniform vec3 u_face_y;
uniform vec3 u_face_z;
const float PI = 3.14159265359;
void main() {
    vec2 tc = v_uv * 2.0 - 1.0;
    vec3 dir = normalize(u_face_x * tc.x + u_face_y * tc.y + u_face_z);
    vec2 uv = vec2(0.5 + atan(dir.z, dir.x) / 6.28318530718, acos(clamp(dir.y, -1.0, 1.0)) / PI);
    frag_color = vec4(texture(u_equirect, uv).rgb, 1.0);
}
"""

_SKY_IBL_CACHE: dict[str, tuple[int, float, Optional["SkyIbl"]]] = {}


class SkyIbl:
    def __init__(self):
        self._irradiance_tex: Optional[moderngl.TextureCube] = None
        self._prefilter_tex: Optional[moderngl.TextureCube] = None
        self._brdf_lut_tex: Optional[moderngl.Texture] = None
        self.ready = False

    def bind(self, prog: moderngl.Program, start_unit: int = 14):
        unit = start_unit
        if self._irradiance_tex is not None:
            self._irradiance_tex.use(unit)
            try:
                prog["u_irradiance_map"].value = unit
                prog["u_irradiance_map_Active"].value = 1
            except Exception:
                pass
            unit += 1
        if self._prefilter_tex is not None:
            self._prefilter_tex.use(unit)
            try:
                prog["u_prefilter_map"].value = unit
                prog["u_prefilter_map_Active"].value = 1
            except Exception:
                pass
            unit += 1
        if self._brdf_lut_tex is not None:
            self._brdf_lut_tex.use(unit)
            try:
                prog["u_brdf_lut"].value = unit
                prog["u_brdf_lut_Active"].value = 1
            except Exception:
                pass
            unit += 1
        try:
            if "u_env_map_rotation" in prog:
                prog["u_env_map_rotation"].value = 0.0
        except Exception:
            pass
        return unit

    def release(self):
        seen = set()
        for tex in (self._irradiance_tex, self._prefilter_tex, self._brdf_lut_tex):
            if tex is not None and id(tex) not in seen:
                seen.add(id(tex))
                try:
                    tex.release()
                except Exception:
                    pass
        self._irradiance_tex = None
        self._prefilter_tex = None
        self._brdf_lut_tex = None
        self.ready = False


def _make_face_vao(ctx: moderngl.Context, prog: moderngl.Program):
    verts = np.array([
        -1, -1, 0, 0,
         1, -1, 1, 0,
         1,  1, 1, 1,
        -1, -1, 0, 0,
         1,  1, 1, 1,
        -1,  1, 0, 1,
    ], dtype=np.float32)
    vbo = ctx.buffer(verts.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "2f 2f", "in_position", "in_uv")])
    return vao, vbo


def _render_cubemap_from_equirect(ctx: moderngl.Context, env_tex: moderngl.Texture, res: int):
    cube_tex = ctx.texture_cube((res, res), 4, dtype="f4")
    cube_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    cube_tex.repeat_x = False
    cube_tex.repeat_y = False
    prog = ctx.program(vertex_shader=_FULLSCREEN_QUAD_VERT, fragment_shader=_EQUIRECT_TO_CUBE_FRAG)
    vao, vbo = _make_face_vao(ctx, prog)
    env_tex.use(0)
    prog["u_equirect"].value = 0
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
    prev_fbo = ctx.fbo
    try:
        for face in range(6):
            fx, fy, fz = _FACE_BASIS[face]
            face_tex = ctx.texture((res, res), 4, dtype="f4")
            fbo = ctx.framebuffer(color_attachments=[face_tex])
            fbo.use()
            fbo.viewport = (0, 0, res, res)
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            prog["u_face_x"].value = fx
            prog["u_face_y"].value = fy
            prog["u_face_z"].value = fz
            vao.render(moderngl.TRIANGLES)
            cube_tex.write(face, face_tex.read())
            fbo.release()
            face_tex.release()
    finally:
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        _restore_framebuffer(ctx, prev_fbo)
        vao.release()
        vbo.release()
    return cube_tex


def _render_irradiance(ctx: moderngl.Context, src_tex: moderngl.TextureCube, res: int):
    irr_res = max(32, res // 4)
    irr_tex = ctx.texture_cube((irr_res, irr_res), 4, dtype="f4")
    irr_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    irr_tex.repeat_x = False
    irr_tex.repeat_y = False
    prog = ctx.program(vertex_shader=_FULLSCREEN_QUAD_VERT, fragment_shader=_IRRADIANCE_FRAG)
    vao, vbo = _make_face_vao(ctx, prog)
    src_tex.use(0)
    prog["u_cubemap"].value = 0
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
    prev_fbo = ctx.fbo
    try:
        for face in range(6):
            fx, fy, fz = _FACE_BASIS[face]
            face_tex = ctx.texture((irr_res, irr_res), 4, dtype="f4")
            fbo = ctx.framebuffer(color_attachments=[face_tex])
            fbo.use()
            fbo.viewport = (0, 0, irr_res, irr_res)
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            prog["u_face_x"].value = fx
            prog["u_face_y"].value = fy
            prog["u_face_z"].value = fz
            src_tex.use(0)
            vao.render(moderngl.TRIANGLES)
            irr_tex.write(face, face_tex.read())
            fbo.release()
            face_tex.release()
    finally:
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        _restore_framebuffer(ctx, prev_fbo)
        vao.release()
        vbo.release()
    return irr_tex


def _render_brdf_lut(ctx: moderngl.Context):
    res = 256
    brdf_tex = ctx.texture((res, res), 2, dtype="f4")
    brdf_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    brdf_tex.repeat_x = False
    brdf_tex.repeat_y = False
    prog = ctx.program(vertex_shader=_FULLSCREEN_QUAD_VERT, fragment_shader=_BRDF_LUT_FRAG)
    vao, vbo = _make_face_vao(ctx, prog)
    prev_fbo = ctx.fbo
    fbo = ctx.framebuffer(color_attachments=[brdf_tex])
    fbo.use()
    fbo.viewport = (0, 0, res, res)
    ctx.clear(0.0, 0.0, 0.0, 1.0)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
    try:
        vao.render(moderngl.TRIANGLES)
    finally:
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        _restore_framebuffer(ctx, prev_fbo)
        fbo.release()
        vao.release()
        vbo.release()
    return brdf_tex


def _render_prefilter(ctx: moderngl.Context, src_cube: moderngl.TextureCube, res: int):
    pref = ctx.texture_cube((res, res), 4, dtype="f4")
    pref.filter = (moderngl.LINEAR, moderngl.LINEAR)
    pref.repeat_x = False
    pref.repeat_y = False
    _allocate_cube_mip_levels(pref, res, _PREFILTER_MAX_LOD)
    prog = ctx.program(vertex_shader=_FULLSCREEN_QUAD_VERT, fragment_shader=_PREFILTER_FRAG)
    vao, vbo = _make_face_vao(ctx, prog)
    src_cube.use(0)
    prog["u_cubemap"].value = 0
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
    prev_fbo = ctx.fbo
    try:
        for level in range(_PREFILTER_MAX_LOD + 1):
            s = max(1, res >> level)
            roughness = level / float(_PREFILTER_MAX_LOD)
            for face in range(6):
                fx, fy, fz = _FACE_BASIS[face]
                face_tex = ctx.texture((s, s), 4, dtype="f4")
                fbo = ctx.framebuffer(color_attachments=[face_tex])
                fbo.use()
                fbo.viewport = (0, 0, s, s)
                ctx.clear(0.0, 0.0, 0.0, 1.0)
                prog["u_roughness"].value = roughness
                prog["u_face_x"].value = fx
                prog["u_face_y"].value = fy
                prog["u_face_z"].value = fz
                src_cube.use(0)
                vao.render(moderngl.TRIANGLES)
                _write_cube_face_mip(pref, face, level, s, face_tex.read())
                fbo.release()
                face_tex.release()
    finally:
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        _restore_framebuffer(ctx, prev_fbo)
        vao.release()
        vbo.release()
    return pref


def _generate_ibl_from_cube(ctx: moderngl.Context, src_cube: moderngl.TextureCube, res: int = 128) -> Optional[SkyIbl]:
    ibl = SkyIbl()
    try:
        ibl._prefilter_tex = _render_prefilter(ctx, src_cube, res)
        ibl._irradiance_tex = _render_irradiance(ctx, src_cube, res)
        ibl._brdf_lut_tex = _render_brdf_lut(ctx)
        ibl.ready = True
    except Exception:
        ibl.release()
        return None
    return ibl


def _generate_ibl(ctx: moderngl.Context, env_tex: moderngl.Texture, res: int = 128) -> Optional[SkyIbl]:
    src_cube = None
    try:
        src_cube = _render_cubemap_from_equirect(ctx, env_tex, res)
        src_cube.build_mipmaps()
        return _generate_ibl_from_cube(ctx, src_cube, res)
    except Exception:
        return None
    finally:
        if src_cube is not None:
            try:
                src_cube.release()
            except Exception:
                pass


def _render_cubemap_from_sky(ctx: moderngl.Context, sky_prog: moderngl.Program,
                             sun_dir, sun_color, sun_intensity, res: int = 128) -> moderngl.TextureCube:
    from core.maths.math3d import Mat4, Vec3
    from core.renderer.meshes import make_cube_mesh

    cube_tex = ctx.texture_cube((res, res), 4, dtype="f4")
    cube_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    cube_tex.repeat_x = False
    cube_tex.repeat_y = False
    mesh = make_cube_mesh()
    vbo = ctx.buffer(mesh.vertices.tobytes())
    ibo = ctx.buffer(mesh.indices.astype(np.uint32).tobytes())
    vao = ctx.vertex_array(sky_prog, [(vbo, "3f", "in_position")], ibo)
    sun_dir = np.asarray(sun_dir, dtype=np.float32).reshape(3)
    sun_color = np.asarray(sun_color, dtype=np.float32).reshape(-1)[:3]
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
    prev_fbo = ctx.fbo
    try:
        for face in range(6):
            fwd, up = _FACE_DIRS[face]
            face_view = Mat4.look_at(Vec3(0.0, 0.0, 0.0),
                                     Vec3(fwd.x, fwd.y, fwd.z),
                                     Vec3(up.x, up.y, up.z))
            face_proj = Mat4.perspective(90.0, 1.0, 0.1, 100.0)
            mvp = face_view * face_proj
            face_tex = ctx.texture((res, res), 4, dtype="f4")
            fbo = ctx.framebuffer(color_attachments=[face_tex])
            fbo.use()
            fbo.viewport = (0, 0, res, res)
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            sky_prog["u_mvp"].write(mvp.to_f32().tobytes())
            try:
                sky_prog["u_use_env"].value = 0.0
            except Exception:
                pass
            try:
                sky_prog["_SunDirection"].write(sun_dir.tobytes())
                sky_prog["_SunColor"].write(sun_color.tobytes())
                sky_prog["_SunIntensity"].value = float(sun_intensity)
            except Exception:
                pass
            vao.render(moderngl.TRIANGLES, vertices=mesh.indices.size)
            cube_tex.write(face, face_tex.read())
            fbo.release()
            face_tex.release()
    finally:
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        _restore_framebuffer(ctx, prev_fbo)
        vao.release()
        vbo.release()
        ibo.release()
    return cube_tex


_PROC_SKY_IBL_CACHE: "OrderedDict[str, SkyIbl]" = OrderedDict()
_PROC_SKY_IBL_CACHE_MAX = 8

_PROC_SUN_DEFAULT_DIR = np.array([0.0, -0.3, -1.0], dtype=np.float32)
_PROC_SUN_DEFAULT_COLOR = np.array([1.0, 0.95, 0.85], dtype=np.float32)
_PROC_SUN_DEFAULT_INTENSITY = 1.0

_SUN_DIR_GRID = 1024.0

_PROC_REGEN_COOLDOWN = 0.15
_PROC_REGEN_CAP_DEG = 2.0
_PROC_LAST_GEN: dict[int, tuple] = {}


def _snap_sun_dir(v):
    a = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(a)
    if n <= 1e-9:
        return _PROC_SUN_DEFAULT_DIR.copy()
    a = a / n
    a = np.round(a * _SUN_DIR_GRID) / _SUN_DIR_GRID
    n2 = np.linalg.norm(a)
    if n2 <= 1e-9:
        return _PROC_SUN_DEFAULT_DIR.copy()
    return (a / n2).astype(np.float32)


def _ang_deg(a, b):
    d = float(np.clip(np.dot(np.asarray(a, dtype=np.float64).ravel(),
                              np.asarray(b, dtype=np.float64).ravel()), -1.0, 1.0))
    return math.degrees(math.acos(d))


def _quant(v, ndig: int):
    return tuple(round(float(x), ndig) for x in np.asarray(v).ravel())


def get_procedural_sky_ibl(ctx: moderngl.Context, sky_prog: moderngl.Program,
                           material_path: str, sun_dir=None, sun_color=None,
                           sun_intensity=None, res: int = 128,
                           settings_key=None) -> Optional[SkyIbl]:
    if sun_dir is None:
        sun_dir = _PROC_SUN_DEFAULT_DIR
    if sun_color is None:
        sun_color = _PROC_SUN_DEFAULT_COLOR
    if sun_intensity is None:
        sun_intensity = _PROC_SUN_DEFAULT_INTENSITY
    true_dir = np.asarray(sun_dir, dtype=np.float64).reshape(3)
    sun_dir = _snap_sun_dir(sun_dir)
    key = "|".join([
        material_path or "",
        repr(_quant(sun_dir, 3)),
        repr(_quant(sun_color, 2)),
        str(round(float(sun_intensity), 2)),
    ])
    if settings_key is not None:
        key += "|" + repr(settings_key)
    key = f"{id(ctx)}|{key}"
    if key in _PROC_SKY_IBL_CACHE:
        ibl = _PROC_SKY_IBL_CACHE.pop(key)
        _PROC_SKY_IBL_CACHE[key] = ibl
        return ibl
    cid = id(ctx)
    settings_ident = (material_path or "", settings_key,
                      repr(_quant(sun_color, 2)), str(round(float(sun_intensity), 2)))
    now = time.perf_counter()
    last = _PROC_LAST_GEN.get(cid)
    if last is not None:
        last_t, last_dir, last_ibl, last_settings = last
        if (last_ibl is not None and last_ibl.ready
                and last_settings == settings_ident
                and (now - last_t) < _PROC_REGEN_COOLDOWN):
            if _ang_deg(last_dir, true_dir) < _PROC_REGEN_CAP_DEG:
                return last_ibl
    src_cube = None
    ibl = None
    try:
        src_cube = _render_cubemap_from_sky(ctx, sky_prog, sun_dir, sun_color, sun_intensity, res)
        src_cube.build_mipmaps()
        ibl = _generate_ibl_from_cube(ctx, src_cube, res)
    except Exception:
        ibl = None
    finally:
        if src_cube is not None:
            try:
                src_cube.release()
            except Exception:
                pass
    if ibl is not None:
        _PROC_SKY_IBL_CACHE[key] = ibl
        _PROC_LAST_GEN[cid] = (now, true_dir, ibl, settings_ident)
        while len(_PROC_SKY_IBL_CACHE) > _PROC_SKY_IBL_CACHE_MAX:
            _old_key, _old = _PROC_SKY_IBL_CACHE.popitem(last=False)
            try:
                _old.release()
            except Exception:
                pass
    return ibl


def get_sky_ibl(ctx: moderngl.Context, env_path: str, env_tex: Optional[moderngl.Texture] = None) -> Optional[SkyIbl]:
    if not env_path:
        return None
    abs_path = os.path.abspath(env_path)
    if not os.path.exists(abs_path):
        return None
    mtime = os.path.getmtime(abs_path)
    cached = _SKY_IBL_CACHE.get(abs_path)
    if cached is not None:
        cctx, cm, ibl = cached
        if cctx == id(ctx) and abs(mtime - cm) < 0.001:
            return ibl
        if ibl is not None:
            try:
                ibl.release()
            except Exception:
                pass
    ibl = _generate_ibl(ctx, env_tex, 128) if env_tex is not None else None
    _SKY_IBL_CACHE[abs_path] = (id(ctx), mtime, ibl)
    return ibl


def release_sky_ibl_cache():
    for _c, _m, ibl in _SKY_IBL_CACHE.values():
        if ibl is not None:
            try:
                ibl.release()
            except Exception:
                pass
    _SKY_IBL_CACHE.clear()
    for ibl in _PROC_SKY_IBL_CACHE.values():
        if ibl is not None:
            try:
                ibl.release()
            except Exception:
                pass
    _PROC_SKY_IBL_CACHE.clear()
    _PROC_LAST_GEN.clear()
