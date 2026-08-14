# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import ctypes
import time
import numpy as np
import moderngl
from ctypes import c_void_p
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.foundation.logger import Logger
from core.maths.math3d import Mat4, Vec3


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


def _restore_framebuffer(ctx: moderngl.Context, prev_fbo):
    if prev_fbo is not None:
        try:
            prev_fbo.use()
        except Exception:
            pass


_FACE_DIRS = [
    (Vec3(1, 0, 0), Vec3(0, -1, 0)),
    (Vec3(-1, 0, 0), Vec3(0, -1, 0)),
    (Vec3(0, 1, 0), Vec3(0, 0, 1)),
    (Vec3(0, -1, 0), Vec3(0, 0, -1)),
    (Vec3(0, 0, 1), Vec3(0, -1, 0)),
    (Vec3(0, 0, -1), Vec3(0, -1, 0)),
]

_FACE_BASIS = [
    ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
]

_FULLSCREEN_QUAD_VERT = """
#version 460 core
in vec2 in_position;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_IRRADIANCE_FRAG = """
#version 460 core
in vec2 v_uv;
out vec4 frag_color;
uniform samplerCube u_cubemap;
uniform vec3 u_face_x;
uniform vec3 u_face_y;
uniform vec3 u_face_z;
const float PI = 3.14159265359;
void main() {
    vec2 tc = v_uv * 2.0 - 1.0;
    vec3 dir = normalize(u_face_x * tc.x + u_face_y * tc.y + u_face_z);
    vec3 N = dir;
    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 right = normalize(cross(up, N));
    up = cross(N, right);
    vec3 irradiance = vec3(0.0);
    float nrSamples = 0.0;
    for (float phi = 0.0; phi < 2.0 * PI; phi += 0.1) {
        for (float theta = 0.0; theta < 0.5 * PI; theta += 0.1) {
            vec3 tangent = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            vec3 sampleVec = tangent.x * right + tangent.y * up + tangent.z * N;
            irradiance += texture(u_cubemap, sampleVec).rgb * cos(theta) * sin(theta);
            nrSamples += 1.0;
        }
    }
    irradiance = PI * irradiance * (1.0 / nrSamples);
    frag_color = vec4(irradiance, 1.0);
}
"""

_PREFILTER_FRAG = """
#version 460 core
in vec2 v_uv;
out vec4 frag_color;
uniform samplerCube u_cubemap;
uniform vec3 u_face_x;
uniform vec3 u_face_y;
uniform vec3 u_face_z;
uniform float u_roughness;
const float PI = 3.14159265359;
float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}
vec2 hammersley(int i, int N) {
    uint bits = uint(i);
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    float rdi = float(bits) * 2.3283064365386963e-10;
    return vec2(float(i) / float(N), rdi);
}
vec3 importance_sample_ggx(vec2 Xi, vec3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    vec3 H = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}
void main() {
    vec2 tc = v_uv * 2.0 - 1.0;
    vec3 N = normalize(u_face_x * tc.x + u_face_y * tc.y + u_face_z);
    vec3 R = N;
    vec3 V = R;
    const int SAMPLE_COUNT = 1024;
    vec3 prefilteredColor = vec3(0.0);
    float totalWeight = 0.0;
    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        vec2 Xi = hammersley(i, SAMPLE_COUNT);
        vec3 H = importance_sample_ggx(Xi, N, u_roughness);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            prefilteredColor += texture(u_cubemap, L).rgb * NdotL;
            totalWeight += NdotL;
        }
    }
    prefilteredColor = prefilteredColor / max(totalWeight, 0.001);
    frag_color = vec4(prefilteredColor, 1.0);
}
"""

_BRDF_LUT_FRAG = """
#version 460 core
in vec2 v_uv;
out vec4 frag_color;
const float PI = 3.14159265359;
float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}
float geometry_schlick_ggx(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}
float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}
vec2 hammersley(int i, int N) {
    uint bits = uint(i);
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    float rdi = float(bits) * 2.3283064365386963e-10;
    return vec2(float(i) / float(N), rdi);
}
vec3 importance_sample_ggx(vec2 Xi, vec3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    vec3 H = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}
void main() {
    float NdotV = v_uv.x;
    float roughness = v_uv.y;
    NdotV = max(NdotV, 0.0001);
    vec3 V = vec3(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);
    vec3 N = vec3(0.0, 0.0, 1.0);
    float A = 0.0;
    float B = 0.0;
    const int SAMPLE_COUNT = 1024;
    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        vec2 Xi = hammersley(i, SAMPLE_COUNT);
        vec3 H = importance_sample_ggx(Xi, N, roughness);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);
        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);
        if (NdotL > 0.0) {
            float G = geometry_smith(N, V, L, roughness);
            float Gvis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0 - VdotH, 5.0);
            A += (1.0 - Fc) * Gvis;
            B += Fc * Gvis;
        }
    }
    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);
    frag_color = vec4(A, B, 0.0, 1.0);
}
"""


@ComponentRegistry.register
class DynamicCubemaps(Component):
    _allow_multiple = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("follow_camera", "Follow Camera", FieldType.BOOL),
            InspectorField("resolution", "Resolution", FieldType.INT_SLIDER, 32, 512),
            InspectorField("update_interval", "Update Interval (s)", FieldType.FLOAT, 0.05, 2.0),
            InspectorField("near_plane", "Near Plane", FieldType.FLOAT, 0.01, 10.0),
            InspectorField("far_plane", "Far Plane", FieldType.FLOAT, 1.0, 500.0),
            InspectorField("intensity", "Intensity", FieldType.FLOAT, 0.0, 5.0),
        ]

    def __init__(self):
        super().__init__()
        self.follow_camera: bool = True
        self.resolution: int = 128
        self.update_interval: float = 0.33
        self.near_plane: float = 0.1
        self.far_plane: float = 100.0
        self.intensity: float = 1.0

        self._ctx_id: int = 0
        self._face_fbos: list[Optional[moderngl.Framebuffer]] = [None] * 6
        self._face_depth: list[Optional[moderngl.Renderbuffer]] = [None] * 6
        self._face_color_texs: list[Optional[moderngl.Texture]] = [None] * 6
        self._cubemap_tex: Optional[moderngl.Texture] = None
        self._irradiance_tex: Optional[moderngl.Texture] = None
        self._irradiance_face_texs: list[Optional[moderngl.Texture]] = [None] * 6
        self._irradiance_face_fbos: list[Optional[moderngl.Framebuffer]] = [None] * 6
        self._prefilter_tex: Optional[moderngl.Texture] = None
        self._prefilter_fbos: list[Optional[moderngl.Framebuffer]] = []
        self._brdf_lut_tex: Optional[moderngl.Texture] = None
        self._brdf_lut_fbo: Optional[moderngl.Framebuffer] = None
        self._irradiance_prog: Optional[moderngl.Program] = None
        self._prefilter_prog: Optional[moderngl.Program] = None
        self._brdf_prog: Optional[moderngl.Program] = None
        self._current_face: int = 0
        self._last_update_time: float = 0.0
        self._all_faces_updated: bool = False

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "follow_camera": self.follow_camera,
            "resolution": self.resolution,
            "update_interval": self.update_interval,
            "near_plane": self.near_plane,
            "far_plane": self.far_plane,
            "intensity": self.intensity,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> DynamicCubemaps:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.follow_camera = bool(data.get("follow_camera", True))
        c.resolution = int(data.get("resolution", 128))
        c.update_interval = float(data.get("update_interval", 0.33))
        c.near_plane = float(data.get("near_plane", 0.1))
        c.far_plane = float(data.get("far_plane", 100.0))
        c.intensity = float(data.get("intensity", 1.0))
        return c

    def _ensure_resources(self, ctx: moderngl.Context) -> bool:
        ctx_id = id(ctx)
        if self._ctx_id == ctx_id and self._cubemap_tex is not None:
            return True
        self._release_gl()
        self._ctx_id = ctx_id
        res = self.resolution

        try:
            self._cubemap_tex = ctx.texture_cube((res, res), 4, dtype="f4")
            self._cubemap_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._cubemap_tex.repeat_x = False
            self._cubemap_tex.repeat_y = False
        except Exception as e:
            Logger.error(f"DynamicCubemaps: failed to create cubemap: {e}")
            return False

        for i in range(6):
            try:
                self._face_color_texs[i] = ctx.texture((res, res), 4, dtype="f4")
                self._face_color_texs[i].filter = (moderngl.LINEAR, moderngl.LINEAR)
                self._face_depth[i] = ctx.depth_renderbuffer((res, res))
                self._face_fbos[i] = ctx.framebuffer(
                    color_attachments=[self._face_color_texs[i]],
                    depth_attachment=self._face_depth[i],
                )
            except Exception as e:
                Logger.error(f"DynamicCubemaps: failed to create face FBO {i}: {e}")
                return False

        irr_res = max(32, res // 4)
        try:
            self._irradiance_tex = ctx.texture_cube((irr_res, irr_res), 4, dtype="f4")
            self._irradiance_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._irradiance_tex.repeat_x = False
            self._irradiance_tex.repeat_y = False
            for i in range(6):
                self._irradiance_face_texs[i] = ctx.texture((irr_res, irr_res), 4, dtype="f4")
                self._irradiance_face_texs[i].filter = (moderngl.LINEAR, moderngl.LINEAR)
                self._irradiance_face_fbos[i] = ctx.framebuffer(
                    color_attachments=[self._irradiance_face_texs[i]]
                )
        except Exception as e:
            Logger.error(f"DynamicCubemaps: failed to create irradiance: {e}")
            return False

        prefilter_mip_count = 5
        self._prefilter_fbos = []
        try:
            self._prefilter_tex = ctx.texture_cube((res, res), 4, dtype="f4")
            self._prefilter_tex.repeat_x = False
            self._prefilter_tex.repeat_y = False
            _allocate_cube_mip_levels(self._prefilter_tex, res, _PREFILTER_MAX_LOD)
            for mip in range(prefilter_mip_count):
                mip_res = max(1, res // (2 ** mip))
                mip_tex = ctx.texture((mip_res, mip_res), 4, dtype="f4")
                mip_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                mip_fbo = ctx.framebuffer(color_attachments=[mip_tex])
                self._prefilter_fbos.append((mip_fbo, mip_tex, mip_res, mip))
        except Exception as e:
            Logger.error(f"DynamicCubemaps: failed to create prefilter: {e}")
            return False

        brdf_res = 256
        try:
            self._brdf_lut_tex = ctx.texture((brdf_res, brdf_res), 2, dtype="f4")
            self._brdf_lut_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._brdf_lut_fbo = ctx.framebuffer(color_attachments=[self._brdf_lut_tex])
        except Exception as e:
            Logger.error(f"DynamicCubemaps: failed to create BRDF LUT: {e}")
            return False

        self._ensure_shaders(ctx)

        return True

    def _ensure_shaders(self, ctx: moderngl.Context):
        if self._irradiance_prog is None:
            try:
                self._irradiance_prog = ctx.program(
                    vertex_shader=_FULLSCREEN_QUAD_VERT,
                    fragment_shader=_IRRADIANCE_FRAG,
                )
            except Exception as e:
                Logger.error(f"DynamicCubemaps: irradiance shader compile error: {e}")

        if self._prefilter_prog is None:
            try:
                self._prefilter_prog = ctx.program(
                    vertex_shader=_FULLSCREEN_QUAD_VERT,
                    fragment_shader=_PREFILTER_FRAG,
                )
            except Exception as e:
                Logger.error(f"DynamicCubemaps: prefilter shader compile error: {e}")

        if self._brdf_prog is None:
            try:
                self._brdf_prog = ctx.program(
                    vertex_shader=_FULLSCREEN_QUAD_VERT,
                    fragment_shader=_BRDF_LUT_FRAG,
                )
            except Exception as e:
                Logger.error(f"DynamicCubemaps: BRDF LUT shader compile error: {e}")

    def _make_face_vao(self, ctx: moderngl.Context, prog: moderngl.Program) -> Optional[moderngl.VertexArray]:
        verts = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
             1,  1, 1, 1,
            -1, -1, 0, 0,
             1,  1, 1, 1,
            -1,  1, 0, 1,
        ], dtype=np.float32)
        vbo = ctx.buffer(verts.tobytes())
        return ctx.vertex_array(prog, [(vbo, "2f 2f", "in_position", "in_uv")])

    def render_face(self, ctx: moderngl.Context, face: int,
                    view_mat: Mat4, proj_mat: Mat4, cam_pos: Vec3,
                    scene, renderer, main_snap=None, skip_entity=None) -> bool:
        if not self._ensure_resources(ctx):
            return False
        if face < 0 or face > 5:
            return False

        fbo = self._face_fbos[face]
        if fbo is None:
            return False

        res = self.resolution
        fwd, up = _FACE_DIRS[face]
        target = cam_pos + fwd
        face_view = Mat4.look_at(cam_pos, target, up)
        face_proj = Mat4.perspective(90.0, 1.0, self.near_plane, self.far_plane)
        view_f32 = face_view.to_f32()
        proj_f32 = face_proj.to_f32()

        prev_fbo = None
        try:
            prev_fbo = ctx.fbo
        except Exception:
            prev_fbo = None

        if main_snap is not None:
            snap = main_snap
        else:
            snap = renderer._collect_snapshot(scene, self.near_plane, self.far_plane, 90.0, face_view, face_proj, cam_pos)

        renderer._rendering_cubemap_face = True
        try:
            renderer.render_cubemap_face(snap, fbo, res, view_f32, proj_f32, cam_pos, snap.lights, skip_entity=skip_entity)
        finally:
            renderer._rendering_cubemap_face = False

        _restore_framebuffer(ctx, prev_fbo)

        face_tex = self._face_color_texs[face]
        if face_tex is not None and self._cubemap_tex is not None:
            data = face_tex.read()
            self._cubemap_tex.write(face, data)
        return True

    def generate_irradiance(self, ctx: moderngl.Context):
        if self._irradiance_prog is None or self._cubemap_tex is None:
            return
        if self._irradiance_tex is None:
            return

        prev_fbo = None
        try:
            prev_fbo = ctx.fbo
        except Exception:
            prev_fbo = None

        prog = self._irradiance_prog
        irr_res = max(32, self.resolution // 4)

        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)

        vao = self._make_face_vao(ctx, prog)
        try:
            self._cubemap_tex.use(0)
            prog["u_cubemap"].value = 0
        except Exception:
            vao.release()
            _restore_framebuffer(ctx, prev_fbo)
            return

        for face in range(6):
            fx, fy, fz = _FACE_BASIS[face]
            fbo = self._irradiance_face_fbos[face]
            if fbo is None:
                continue
            fbo.use()
            fbo.viewport = (0, 0, irr_res, irr_res)
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            try:
                prog["u_face_x"].value = fx
                prog["u_face_y"].value = fy
                prog["u_face_z"].value = fz
            except Exception:
                continue
            self._cubemap_tex.use(0)
            vao.render(moderngl.TRIANGLES)
            try:
                data = self._irradiance_face_texs[face].read()
                self._irradiance_tex.write(face, data)
            except Exception:
                pass

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        vao.release()
        _restore_framebuffer(ctx, prev_fbo)

    def generate_prefilter(self, ctx: moderngl.Context):
        if self._prefilter_prog is None or self._cubemap_tex is None:
            return

        prev_fbo = None
        try:
            prev_fbo = ctx.fbo
        except Exception:
            prev_fbo = None

        prog = self._prefilter_prog
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)

        vao = self._make_face_vao(ctx, prog)
        try:
            self._cubemap_tex.use(0)
            prog["u_cubemap"].value = 0
        except Exception:
            vao.release()
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.CULL_FACE)
            _restore_framebuffer(ctx, prev_fbo)
            return

        for mip_idx, (mip_fbo, mip_tex, mip_res, mip_level) in enumerate(self._prefilter_fbos):
            roughness = mip_level / max(1, len(self._prefilter_fbos) - 1)
            for face in range(6):
                fx, fy, fz = _FACE_BASIS[face]
                mip_fbo.use()
                mip_fbo.viewport = (0, 0, mip_res, mip_res)
                ctx.clear(0.0, 0.0, 0.0, 1.0)
                try:
                    prog["u_roughness"].value = roughness
                    prog["u_face_x"].value = fx
                    prog["u_face_y"].value = fy
                    prog["u_face_z"].value = fz
                except Exception:
                    continue
                self._cubemap_tex.use(0)
                vao.render(moderngl.TRIANGLES)

                try:
                    mip_data = mip_tex.read()
                    _write_cube_face_mip(self._prefilter_tex, face, mip_level, mip_res, mip_data)
                except Exception:
                    pass

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        vao.release()
        _restore_framebuffer(ctx, prev_fbo)

    def generate_brdf_lut(self, ctx: moderngl.Context):
        if self._brdf_prog is None or self._brdf_lut_fbo is None:
            return

        prev_fbo = None
        try:
            prev_fbo = ctx.fbo
        except Exception:
            prev_fbo = None

        brdf_res = 256
        self._brdf_lut_fbo.use()
        self._brdf_lut_fbo.viewport = (0, 0, brdf_res, brdf_res)
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)

        vao = self._make_face_vao(ctx, self._brdf_prog)
        vao.render(moderngl.TRIANGLES)
        vao.release()

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        _restore_framebuffer(ctx, prev_fbo)

    def update(self, ctx: moderngl.Context, view_mat: Mat4, proj_mat: Mat4,
               cam_pos: Vec3, scene, renderer, main_snap=None, skip_entity=None) -> bool:
        if not self._ensure_resources(ctx):
            return False

        now = time.perf_counter()
        needs_update = False

        if not self._all_faces_updated:
            needs_update = True
        elif self.update_interval <= 0.0:
            needs_update = True
        elif (now - self._last_update_time) >= self.update_interval:
            needs_update = True

        if not needs_update:
            return self._all_faces_updated

        if not self._all_faces_updated:
            face = self._current_face
            self.render_face(ctx, face, view_mat, proj_mat, cam_pos, scene, renderer, main_snap=main_snap, skip_entity=skip_entity)
            self._current_face = (self._current_face + 1) % 6
            if self._current_face == 0:
                self._all_faces_updated = True
                self._last_update_time = now
                self.generate_irradiance(ctx)
                self.generate_prefilter(ctx)
                self.generate_brdf_lut(ctx)
        else:
            for face in range(6):
                self.render_face(ctx, face, view_mat, proj_mat, cam_pos, scene, renderer, main_snap=main_snap, skip_entity=skip_entity)
            self._all_faces_updated = True
            self._last_update_time = now
            self.generate_irradiance(ctx)
            self.generate_prefilter(ctx)
            self.generate_brdf_lut(ctx)
            self._current_face = 0

        return True

    def bind_ibl(self, prog: moderngl.Program, start_unit: int = 14):
        unit = start_unit
        if self._irradiance_tex is not None:
            self._irradiance_tex.use(unit)
            try:
                prog["u_irradiance_map"].value = unit
                prog["u_irradiance_map_Active"].value = 1
            except KeyError:
                pass
            unit += 1
        if self._prefilter_tex is not None:
            self._prefilter_tex.use(unit)
            try:
                prog["u_prefilter_map"].value = unit
                prog["u_prefilter_map_Active"].value = 1
            except KeyError:
                pass
            unit += 1
        if self._brdf_lut_tex is not None:
            self._brdf_lut_tex.use(unit)
            try:
                prog["u_brdf_lut"].value = unit
                prog["u_brdf_lut_Active"].value = 1
            except KeyError:
                pass
            unit += 1
        if "u_env_map_rotation" in prog:
            try:
                prog["u_env_map_rotation"].value = 0.0
            except Exception:
                pass
        return unit

    def on_destroy(self):
        self._release_gl()

    def on_disable(self):
        self._release_gl()

    def _release_gl(self):
        for fbo in self._face_fbos:
            if fbo:
                try:
                    fbo.release()
                except Exception:
                    pass
        self._face_fbos = [None] * 6

        for tex in self._face_color_texs:
            if tex:
                try:
                    tex.release()
                except Exception:
                    pass
        self._face_color_texs = [None] * 6

        for d in self._face_depth:
            if d:
                try:
                    d.release()
                except Exception:
                    pass
        self._face_depth = [None] * 6

        for tex in [self._cubemap_tex, self._irradiance_tex, self._brdf_lut_tex]:
            if tex:
                try:
                    tex.release()
                except Exception:
                    pass
        self._cubemap_tex = None
        self._irradiance_tex = None
        self._brdf_lut_tex = None

        for tex in self._irradiance_face_texs:
            if tex:
                try:
                    tex.release()
                except Exception:
                    pass
        self._irradiance_face_texs = [None] * 6

        for fbo in self._irradiance_face_fbos:
            if fbo:
                try:
                    fbo.release()
                except Exception:
                    pass
        self._irradiance_face_fbos = [None] * 6

        for item in getattr(self, '_prefilter_fbos', []):
            if item:
                try:
                    item[0].release()
                    item[1].release()
                except Exception:
                    pass
        self._prefilter_fbos = []

        for fbo in [self._brdf_lut_fbo]:
            if fbo:
                try:
                    fbo.release()
                except Exception:
                    pass
        self._brdf_lut_fbo = None

        for prog in [self._irradiance_prog, self._prefilter_prog, self._brdf_prog]:
            if prog:
                try:
                    prog.release()
                except Exception:
                    pass
        self._irradiance_prog = None
        self._prefilter_prog = None
        self._brdf_prog = None

        self._ctx_id = 0
        self._current_face = 0
        self._all_faces_updated = False
        self._last_update_time = 0.0