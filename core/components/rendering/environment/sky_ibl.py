# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import ctypes
import os
import numpy as np
import moderngl
from ctypes import c_void_p
from typing import Optional
from core.components.rendering.environment.dynamic_cubemap import (
    _BRDF_LUT_FRAG,
    _FACE_BASIS,
    _FULLSCREEN_QUAD_VERT,
    _IRRADIANCE_FRAG,
    _PREFILTER_FRAG,
)

_GL_TEXTURE_CUBE_MAP = 0x8513
_GL_TEXTURE_CUBE_MAP_POSITIVE_X = 0x8515
_GL_RGBA16F = 0x881A
_GL_RGBA = 0x1908
_GL_FLOAT = 0x1406
_GL_TEXTURE_MAX_LEVEL = 0x813D
_GL_TEXTURE_MIN_FILTER = 0x2801
_GL_TEXTURE_MAG_FILTER = 0x2800
_GL_TEXTURE_WRAP_S = 0x2802
_GL_TEXTURE_WRAP_T = 0x2803
_GL_TEXTURE_WRAP_R = 0x8072
_GL_LINEAR_MIPMAP_LINEAR = 0x2703
_GL_LINEAR = 0x2601
_GL_CLAMP_TO_EDGE = 0x812F

_PREFILTER_MAX_LOD = 4

_opengl32 = ctypes.windll.opengl32
_opengl32.glGetError.restype = ctypes.c_uint
_opengl32.glBindTexture.restype = None
_opengl32.glBindTexture.argtypes = (ctypes.c_uint, ctypes.c_uint)
_opengl32.glTexImage2D.restype = None
_opengl32.glTexImage2D.argtypes = (
    ctypes.c_uint, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_uint, ctypes.c_uint, c_void_p,
)
_opengl32.glTexSubImage2D.restype = None
_opengl32.glTexSubImage2D.argtypes = (
    ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, c_void_p,
)
_opengl32.glTexParameteri.restype = None
_opengl32.glTexParameteri.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_int)


def _allocate_cube_mip_levels(tex: moderngl.TextureCube, res: int, max_level: int):
    _opengl32.glBindTexture(_GL_TEXTURE_CUBE_MAP, tex.glo)
    for level in range(max_level + 1):
        s = max(1, res >> level)
        for face in range(6):
            _opengl32.glTexImage2D(
                _GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, level, _GL_RGBA16F,
                s, s, 0, _GL_RGBA, _GL_FLOAT, None,
            )
    _opengl32.glTexParameteri(_GL_TEXTURE_CUBE_MAP, _GL_TEXTURE_MAX_LEVEL, max_level)
    _opengl32.glTexParameteri(_GL_TEXTURE_CUBE_MAP, _GL_TEXTURE_MIN_FILTER, _GL_LINEAR_MIPMAP_LINEAR)
    _opengl32.glTexParameteri(_GL_TEXTURE_CUBE_MAP, _GL_TEXTURE_MAG_FILTER, _GL_LINEAR)
    _opengl32.glTexParameteri(_GL_TEXTURE_CUBE_MAP, _GL_TEXTURE_WRAP_S, _GL_CLAMP_TO_EDGE)
    _opengl32.glTexParameteri(_GL_TEXTURE_CUBE_MAP, _GL_TEXTURE_WRAP_T, _GL_CLAMP_TO_EDGE)
    _opengl32.glTexParameteri(_GL_TEXTURE_CUBE_MAP, _GL_TEXTURE_WRAP_R, _GL_CLAMP_TO_EDGE)
    _opengl32.glBindTexture(_GL_TEXTURE_CUBE_MAP, 0)


def _write_cube_face_mip(tex: moderngl.TextureCube, face: int, level: int, size: int, data: bytes):
    buf = np.frombuffer(data, np.float32)
    _opengl32.glBindTexture(_GL_TEXTURE_CUBE_MAP, tex.glo)
    _opengl32.glTexSubImage2D(
        _GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, level, 0, 0,
        size, size, _GL_RGBA, _GL_FLOAT, buf.ctypes.data_as(c_void_p),
    )
    _opengl32.glBindTexture(_GL_TEXTURE_CUBE_MAP, 0)

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
            vao.render(moderngl.TRIANGLES)
            irr_tex.write(face, face_tex.read())
            fbo.release()
            face_tex.release()
    finally:
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
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
    try:
        prog["u_resolution"].value = float(res)
    except Exception:
        pass
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
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
                vao.render(moderngl.TRIANGLES)
                _write_cube_face_mip(pref, face, level, s, face_tex.read())
                fbo.release()
                face_tex.release()
    finally:
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        vao.release()
        vbo.release()
    return pref


def _generate_ibl(ctx: moderngl.Context, env_tex: moderngl.Texture, res: int = 128) -> Optional[SkyIbl]:
    ibl = SkyIbl()
    src_cube = None
    try:
        src_cube = _render_cubemap_from_equirect(ctx, env_tex, res)
        src_cube.build_mipmaps()
        ibl._prefilter_tex = _render_prefilter(ctx, src_cube, res)
        ibl._irradiance_tex = _render_irradiance(ctx, src_cube, res)
        ibl._brdf_lut_tex = _render_brdf_lut(ctx)
        ibl.ready = True
    except Exception:
        ibl.release()
        return None
    finally:
        if src_cube is not None:
            try:
                src_cube.release()
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
