# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
import moderngl
from typing import Optional
from core.ecs.ecs import ComponentRegistry
from core.components.rendering.postfx.graphics_effect import GraphicsEffect
from core.components.inspector_meta import FieldType, InspectorField


FLARE_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

EXTRACT_FRAG = """
#version 460 core
uniform sampler2D u_scene_color;
uniform float u_threshold;
uniform float u_soft_threshold;
uniform vec2 u_texel_size;
in vec2 v_uv;
out vec4 frag_color;

vec3 lum_max(vec2 uv, vec2 h) {
    vec3 c0 = texture(u_scene_color, uv + vec2(-h.x, -h.y)).rgb;
    vec3 c1 = texture(u_scene_color, uv + vec2( h.x, -h.y)).rgb;
    vec3 c2 = texture(u_scene_color, uv + vec2(-h.x,  h.y)).rgb;
    vec3 c3 = texture(u_scene_color, uv + vec2( h.x,  h.y)).rgb;
    float l0 = dot(c0, vec3(0.299, 0.587, 0.114));
    float l1 = dot(c1, vec3(0.299, 0.587, 0.114));
    float l2 = dot(c2, vec3(0.299, 0.587, 0.114));
    float l3 = dot(c3, vec3(0.299, 0.587, 0.114));
    if (l1 > l0 && l1 > l2 && l1 > l3) return c1;
    if (l2 > l0 && l2 > l1 && l2 > l3) return c2;
    if (l3 > l0 && l3 > l1 && l3 > l2) return c3;
    return c0;
}

void main() {
    vec3 c = lum_max(v_uv, u_texel_size * 0.5);
    float lum = dot(c, vec3(0.299, 0.587, 0.114));
    float knee = u_threshold * u_soft_threshold;
    float soft = lum - u_threshold + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 0.0001);
    float bright = max(lum - u_threshold, soft);
    frag_color = vec4(c * (bright / max(lum, 0.0001)), 1.0);
}
"""

FLARE_FRAG = """
#version 460 core
const int MAX_LIGHTS = 16;
const int MAX_GHOSTS = 9;

uniform int u_light_count;
uniform vec2 u_light_pos[MAX_LIGHTS];
uniform vec4 u_light_col[MAX_LIGHTS];
uniform float u_aspect;
uniform float u_intensity;
uniform float u_scale;
uniform float u_glow;
uniform float u_ghost_intensity;
uniform float u_ghost_spacing;
uniform int u_ghost_count;
uniform float u_anamorphic;
uniform float u_chromatic;
uniform float u_motes;

in vec2 v_uv;
out vec4 frag_color;

// Zeiss-style ghost train: offset multiplier along the light-to-center axis,
// size scale, tint and weight for each internal lens reflection.
const float G_A[MAX_GHOSTS] = float[](
    2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5
);
const float G_S[MAX_GHOSTS] = float[](
    0.06, 0.09, 0.12, 0.16, 0.22, 0.32, 0.20, 0.14, 0.10
);
const float G_W[MAX_GHOSTS] = float[](
    0.10, 0.16, 0.24, 0.34, 0.42, 0.55, 0.38, 0.28, 0.20
);
const vec3 G_T[MAX_GHOSTS] = vec3[](
    vec3(1.00, 0.75, 0.45),
    vec3(0.75, 0.90, 1.00),
    vec3(1.00, 0.80, 0.55),
    vec3(0.85, 0.90, 1.00),
    vec3(1.00, 0.70, 0.50),
    vec3(0.60, 1.00, 0.80),
    vec3(1.00, 0.55, 0.65),
    vec3(0.65, 0.80, 1.00),
    vec3(0.85, 0.70, 1.00)
);

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float soft_poly(vec2 p, float r, float blur, int n) {
    float an = 6.28318530718 / float(n);
    float a = atan(p.x, p.y);
    float b = a - an * floor((a + 3.14159265359 / float(n)) / an);
    vec2 proj = vec2(cos(b), sin(b)) * r;
    return 1.0 - smoothstep(r - blur, r + blur, length(p - proj));
}

vec3 glow_pass(vec2 sp, vec2 center, float s, vec3 col) {
    vec2 d = sp - center;
    float r2 = dot(d, d);
    float g = exp(-r2 / (2.0 * s * s));
    g += 0.65 * exp(-r2 / (2.0 * s * s * 0.09));
    return col * g;
}

vec3 ghost_pass(vec2 sp, vec2 center, float size, vec3 tint, vec2 axis) {
    float r = max(size, 0.004);
    float blur = max(r * 0.22, 0.003);
    vec2 stretch = vec2(sp.x, sp.y * 1.7);
    vec2 cstr = vec2(center.x, center.y * 1.7);
    vec2 off = axis * (u_chromatic * r * 0.5);
    float fr = soft_poly(stretch - (cstr + off), r, blur, 6);
    float fg = soft_poly(stretch - cstr, r, blur, 6);
    float fb = soft_poly(stretch - (cstr - off), r, blur, 6);
    return vec3(fr, fg, fb) * tint;
}

vec3 streak_pass(vec2 sp, vec2 center, vec3 col) {
    vec2 d = sp - center;
    float sy = 0.0038;
    float sx = 0.65;
    float fr = exp(-pow(d.y - 0.0035, 2.0) / (2.0 * sy * sy)) * exp(-abs(d.x) / sx);
    float fg = exp(-pow(d.y, 2.0) / (2.0 * sy * sy)) * exp(-abs(d.x) / sx);
    float fb = exp(-pow(d.y + 0.0035, 2.0) / (2.0 * sy * sy)) * exp(-abs(d.x) / sx);
    return vec3(fr, fg, fb) * col;
}

float ring_pass(vec2 p, float r) {
    return 1.0 - smoothstep(0.0, 0.012, abs(length(p) - r));
}

vec3 motes_pass(vec2 sp, vec2 center, vec2 axis, float d, vec3 col) {
    vec3 acc = vec3(0.0);
    for (int j = 0; j < 7; j++) {
        float h1 = hash12(vec2(float(j) * 1.37 + 0.13, d));
        float h2 = hash12(vec2(float(j) * 0.73 + 0.71, d));
        float a = -0.45 + (float(j) + h1 * 0.5) * 0.42;
        vec2 mp = axis * (a * d);
        vec2 perp = vec2(-axis.y, axis.x) * ((h2 - 0.5) * d * 1.1);
        float s = 0.0025 + h1 * 0.0055;
        float f = soft_poly(sp - (mp + perp), s, s * 0.7, 9);
        float fall = max(0.65 - 0.35 * abs(a), 0.15);
        acc += col * f * (0.10 + h1 * 0.30) * fall;
    }
    return acc;
}

void main() {
    vec2 sp = vec2((v_uv.x - 0.5) * u_aspect, v_uv.y - 0.5);
    vec3 acc = vec3(0.0);
    for (int i = 0; i < MAX_LIGHTS; i++) {
        if (i >= u_light_count) break;
        vec2 lp = vec2((u_light_pos[i].x - 0.5) * u_aspect, u_light_pos[i].y - 0.5);
        float d = max(length(lp), 1e-5);
        vec2 axis = lp / d;
        float lum = clamp(u_light_col[i].w, 0.0, 1.0);
        if (lum < 0.004) continue;
        vec3 col = u_light_col[i].rgb;
        float vis = clamp(d / 0.22, 0.0, 1.0);
        float base = u_intensity * lum;
        float gs = 0.05 * u_scale * (0.55 + 0.65 * d);
        acc += glow_pass(sp, lp, gs, col) * base * u_glow * (0.55 + 0.45 * vis);
        acc += ring_pass(sp - lp, gs * 1.55) * base * u_glow * vis * 0.28
             * vec3(0.85, 0.90, 1.0);
        acc += streak_pass(sp, lp, col) * base * u_anamorphic;
        int ng = min(max(u_ghost_count, 0), MAX_GHOSTS);
        for (int g = 0; g < MAX_GHOSTS; g++) {
            if (g >= ng) break;
            float a = G_A[g] * u_ghost_spacing;
            vec2 gc = axis * (a * d);
            float size = G_S[g] * u_scale * (0.10 + d);
            acc += ghost_pass(sp, gc, size, G_T[g], axis)
                 * col * base * u_ghost_intensity * G_W[g] * vis;
        }
        if (u_motes > 0.001) {
            acc += motes_pass(sp, lp, axis, d, col) * base * u_motes;
        }
    }
    frag_color = vec4(max(acc, vec3(0.0)), 1.0);
}
"""


@ComponentRegistry.register
class LensFlares(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "\u2726"
    render_type = "additive"
    _intensity_prop = "_intensity"
    MAX_LIGHTS = 16
    MAX_GHOSTS = 9

    def __init__(self):
        super().__init__()
        self._intensity: float = 1.0
        self._scale: float = 1.0
        self._glow: float = 1.0
        self._threshold: float = 0.9
        self._soft_threshold: float = 0.5
        self._ghost_intensity: float = 1.0
        self._ghost_spacing: float = 1.0
        self._ghost_count: int = 9
        self._anamorphic: float = 0.8
        self._chromatic: float = 1.0
        self._motes: float = 0.6
        self._max_lights: int = 8
        self._det_width: int = 64
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None
        self._extract_prog: Optional[moderngl.Program] = None
        self._extract_vao: Optional[moderngl.VertexArray] = None
        self._det_tex: Optional[moderngl.Texture] = None
        self._det_fbo: Optional[moderngl.Framebuffer] = None
        self._det_size: tuple[int, int] = (0, 0)
        self._det_ctx: Optional[moderngl.Context] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.05, decimals=2),
            InspectorField("_scale", "Scale", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.05, decimals=2),
            InspectorField("_glow", "Glow", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.05, decimals=2),
            InspectorField("_threshold", "Threshold", FieldType.FLOAT, min_val=0.0, max_val=2.0, step=0.05, decimals=3),
            InspectorField("_soft_threshold", "Soft Knee", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.05, decimals=2),
            InspectorField("_ghost_intensity", "Ghost Intensity", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.05, decimals=2),
            InspectorField("_ghost_spacing", "Ghost Spacing", FieldType.FLOAT, min_val=0.0, max_val=3.0, step=0.05, decimals=2),
            InspectorField("_ghost_count", "Ghost Count", FieldType.INT, min_val=0, max_val=9, step=1),
            InspectorField("_anamorphic", "Anamorphic Streak", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.05, decimals=2),
            InspectorField("_chromatic", "Chromatic", FieldType.FLOAT, min_val=0.0, max_val=3.0, step=0.05, decimals=2),
            InspectorField("_motes", "Dust Motes", FieldType.FLOAT, min_val=0.0, max_val=3.0, step=0.05, decimals=2),
            InspectorField("_max_lights", "Max Lights", FieldType.INT, min_val=1, max_val=16, step=1),
            InspectorField("_det_width", "Detection Width", FieldType.INT, min_val=16, max_val=128, step=8),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_intensity": self._intensity,
            "_scale": self._scale,
            "_glow": self._glow,
            "_threshold": self._threshold,
            "_soft_threshold": self._soft_threshold,
            "_ghost_intensity": self._ghost_intensity,
            "_ghost_spacing": self._ghost_spacing,
            "_ghost_count": self._ghost_count,
            "_anamorphic": self._anamorphic,
            "_chromatic": self._chromatic,
            "_motes": self._motes,
            "_max_lights": self._max_lights,
            "_det_width": self._det_width,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> LensFlares:
        inst = super().deserialize(data)
        inst._intensity = float(data.get("_intensity", 1.0))
        inst._scale = float(data.get("_scale", 1.0))
        inst._glow = float(data.get("_glow", 1.0))
        inst._threshold = float(data.get("_threshold", 0.9))
        inst._soft_threshold = float(data.get("_soft_threshold", 0.5))
        inst._ghost_intensity = float(data.get("_ghost_intensity", 1.0))
        inst._ghost_spacing = float(data.get("_ghost_spacing", 1.0))
        inst._ghost_count = int(data.get("_ghost_count", 9))
        inst._anamorphic = float(data.get("_anamorphic", 0.8))
        inst._chromatic = float(data.get("_chromatic", 1.0))
        inst._motes = float(data.get("_motes", 0.6))
        inst._max_lights = int(data.get("_max_lights", 8))
        inst._det_width = int(data.get("_det_width", 64))
        inst._prog = None
        inst._vao = None
        inst._vbo = None
        inst._ibo = None
        inst._extract_prog = None
        inst._extract_vao = None
        inst._det_tex = None
        inst._det_fbo = None
        inst._det_size = (0, 0)
        inst._det_ctx = None
        return inst

    _res_prog_cache: dict[int, dict] = {}

    def _ensure_resources(self, ctx: moderngl.Context):
        ctx_id = id(ctx)
        cached = self._res_prog_cache.get(ctx_id)
        if cached is not None:
            self._ctx = ctx
            self._prog = cached['_prog']
            self._vao = cached['_vao']
            self._vbo = cached['_vbo']
            self._ibo = cached['_ibo']
            self._extract_prog = cached['_extract_prog']
            self._extract_vao = cached['_extract_vao']
            return
        self._ctx = ctx
        verts = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        self._vbo = ctx.buffer(verts.tobytes())
        self._ibo = ctx.buffer(indices.tobytes())
        self._prog = ctx.program(vertex_shader=FLARE_VERT, fragment_shader=FLARE_FRAG)
        self._extract_prog = ctx.program(vertex_shader=FLARE_VERT, fragment_shader=EXTRACT_FRAG)
        self._vao = ctx.vertex_array(self._prog, [(self._vbo, '2f', 'in_position')], self._ibo)
        self._extract_vao = ctx.vertex_array(self._extract_prog, [(self._vbo, '2f', 'in_position')], self._ibo)
        self._res_prog_cache[ctx_id] = {
            '_prog': self._prog,
            '_vao': self._vao,
            '_vbo': self._vbo,
            '_ibo': self._ibo,
            '_extract_prog': self._extract_prog,
            '_extract_vao': self._extract_vao,
        }
        if len(self._res_prog_cache) > 4:
            oldest = next(iter(self._res_prog_cache))
            self._release_cache_objects({oldest: self._res_prog_cache[oldest]})
            del self._res_prog_cache[oldest]

    def _ensure_det(self, ctx: moderngl.Context, det_w: int, det_h: int):
        if self._det_tex is not None and self._det_size == (det_w, det_h) and self._det_ctx is ctx:
            return
        self._release_det()
        self._det_tex = ctx.texture((det_w, det_h), 4, dtype='f2')
        self._det_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._det_tex.repeat_x = False
        self._det_tex.repeat_y = False
        self._det_fbo = ctx.framebuffer(self._det_tex)
        self._det_size = (det_w, det_h)
        self._det_ctx = ctx

    def _release_det(self):
        for obj in (self._det_fbo, self._det_tex):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._det_fbo = None
        self._det_tex = None
        self._det_size = (0, 0)
        self._det_ctx = None

    def _detect_lights(self, det_w: int, det_h: int) -> list:
        raw = self._det_tex.read()
        arr = np.frombuffer(raw, dtype=np.float16).reshape(det_h, det_w, 4).astype(np.float32)
        lum = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        floor = max(0.05, 0.25 * self._threshold)
        cand = lum > floor
        visited = np.zeros((det_h, det_w), dtype=bool)
        lights = []
        limit = min(max(self._max_lights, 1), self.MAX_LIGHTS)
        while len(lights) < limit:
            avail = cand & ~visited
            if not avail.any():
                break
            idx = int(np.argmax(np.where(avail, lum, 0.0)))
            cy, cx = divmod(idx, det_w)
            stack = [(cy, cx)]
            visited[cy, cx] = True
            blob_py = []
            blob_px = []
            blob_w = []
            blob_c = np.zeros(3, dtype=np.float32)
            while stack:
                y, x = stack.pop()
                blob_py.append(y)
                blob_px.append(x)
                wgt = lum[y, x]
                blob_w.append(wgt)
                blob_c += arr[y, x, :3] * wgt
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < det_h and 0 <= nx < det_w and cand[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            ws = float(sum(blob_w))
            if ws <= 0.0:
                continue
            py = sum(y * w for y, w in zip(blob_py, blob_w)) / ws
            px = sum(x * w for x, w in zip(blob_px, blob_w)) / ws
            rgb = blob_c / max(ws, 1e-6)
            mx = max(float(rgb.max()), 1e-6)
            chroma = rgb / mx
            lmax = float(lum[cy, cx])
            lum_norm = float(1.0 - np.exp(-lmax * 0.5))
            lights.append((
                (px + 0.5) / det_w,
                (py + 0.5) / det_h,
                float(chroma[0]), float(chroma[1]), float(chroma[2]),
                lum_norm,
            ))
        return lights

    def render(self, ctx, scene_color_tex, scene_depth_tex,
               view_mat, proj_mat, cam_pos, viewport_w, viewport_h, **kwargs):
        if not self.enabled or not self.entity or not self.entity.active:
            return
        self._ensure_resources(ctx)
        aspect = viewport_w / max(viewport_h, 1)
        det_w = max(8, int(self._det_width))
        det_h = max(8, min(96, int(round(det_w / max(aspect, 0.1)))))
        self._ensure_det(ctx, det_w, det_h)
        prev_fbo = ctx.fbo

        self._det_fbo.use()
        self._det_fbo.viewport = (0, 0, det_w, det_h)
        self._extract_prog["u_scene_color"] = 0
        self._extract_prog["u_threshold"].value = float(self._threshold)
        self._extract_prog["u_soft_threshold"].value = float(self._soft_threshold)
        self._extract_prog["u_texel_size"].value = (1.0 / det_w, 1.0 / det_h)
        ctx.disable(moderngl.BLEND)
        scene_color_tex.use(0)
        self._extract_vao.render()

        lights = self._detect_lights(det_w, det_h)
        if not lights:
            self._restore_target(prev_fbo, viewport_w, viewport_h)
            return

        if prev_fbo is not None:
            w, h = prev_fbo.size
        else:
            w, h = viewport_w, viewport_h
        self._restore_target(prev_fbo, w, h)
        ctx.viewport = (0, 0, w, h)

        flat_pos = []
        flat_col = []
        for light in lights[: self.MAX_LIGHTS]:
            flat_pos.append((float(light[0]), float(light[1])))
            flat_col.append((float(light[2]), float(light[3]), float(light[4]), float(light[5])))
        n = len(lights[: self.MAX_LIGHTS])
        flat_pos += [(0.0, 0.0)] * (self.MAX_LIGHTS - n)
        flat_col += [(0.0, 0.0, 0.0, 0.0)] * (self.MAX_LIGHTS - n)

        prog = self._prog
        prog["u_light_count"].value = n
        prog["u_light_pos"].value = flat_pos
        prog["u_light_col"].value = flat_col
        prog["u_aspect"].value = aspect
        prog["u_intensity"].value = float(self._intensity)
        prog["u_scale"].value = float(self._scale)
        prog["u_glow"].value = float(self._glow)
        prog["u_ghost_intensity"].value = float(self._ghost_intensity)
        prog["u_ghost_spacing"].value = float(self._ghost_spacing)
        prog["u_ghost_count"].value = int(self._ghost_count)
        prog["u_anamorphic"].value = float(self._anamorphic)
        prog["u_chromatic"].value = float(self._chromatic)
        prog["u_motes"].value = float(self._motes)

        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ctx.enable(moderngl.BLEND)
        self._vao.render()
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def _restore_target(self, prev_fbo, w, h):
        if prev_fbo is not None:
            prev_fbo.use()
            prev_fbo.viewport = (0, 0, w, h)
        elif self._ctx is not None and self._ctx.screen is not None:
            self._ctx.screen.use()
            self._ctx.viewport = (0, 0, w, h)

    def _release_gl(self):
        self._release_det()
        for obj in (self._prog, self._vao, self._vbo, self._ibo,
                    self._extract_prog, self._extract_vao):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._ctx = None
        self._prog = None
        self._vao = None
        self._vbo = None
        self._ibo = None
        self._extract_prog = None
        self._extract_vao = None

    @property
    def intensity(self) -> float:
        return getattr(self, '_intensity', 1.0)

    @intensity.setter
    def intensity(self, v: float):
        self._intensity = v

    @property
    def threshold(self) -> float:
        return getattr(self, '_threshold', 0.9)

    @threshold.setter
    def threshold(self, v: float):
        self._threshold = v
