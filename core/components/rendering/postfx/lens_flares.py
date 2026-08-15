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
in vec2 v_uv;
out vec4 frag_color;
void main() {
    vec3 c = texture(u_scene_color, v_uv).rgb;
    float lum = dot(c, vec3(0.299, 0.587, 0.114));
    float knee = u_threshold * u_soft_threshold;
    float soft = lum - u_threshold + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 0.0001);
    float bright = max(lum - u_threshold, soft);
    frag_color = vec4(c * (bright / max(lum, 0.0001)), 1.0);
}
"""

DOWNSAMPLE_FRAG = """
#version 460 core
uniform sampler2D u_src;
uniform vec2 u_texel;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    vec2 h = u_texel * 0.5;
    vec3 c0 = texture(u_src, v_uv + vec2(-h.x, -h.y)).rgb;
    vec3 c1 = texture(u_src, v_uv + vec2( h.x, -h.y)).rgb;
    vec3 c2 = texture(u_src, v_uv + vec2(-h.x,  h.y)).rgb;
    vec3 c3 = texture(u_src, v_uv + vec2( h.x,  h.y)).rgb;
    frag_color = vec4((c0 + c1 + c2 + c3) * 0.25, 1.0);
}
"""

FLARE_FRAG = """
#version 460 core
const int GHOSTS = 7;

uniform sampler2D u_bright_tex;
uniform sampler2D u_glow_tex;
uniform vec2 u_texel;
uniform float u_intensity;
uniform float u_scale;
uniform float u_glow;
uniform float u_ghost_intensity;
uniform float u_anamorphic;
uniform float u_chromatic;

in vec2 v_uv;
out vec4 frag_color;

// Ghost train: each element is the bright map re-sampled around the screen
// center at scale s = 1 / a (a = position multiplier along the center-light
// axis), so bright regions are ghosted as continuous scaled copies -- no
// discrete light detection, exactly like a bloom downsample chain.
const float G_A[GHOSTS] = float[]( 1.5,  1.0,  0.5,  0.25,  0.10, -0.5, -1.0 );
const float G_W[GHOSTS] = float[]( 0.14, 0.30, 0.40, 0.25,  0.16, 0.12, 0.08 );
const float G_TEX[GHOSTS] = float[]( 0.0,  1.0,  0.0,  1.0,   1.0,  0.0,  1.0 );
const vec3 G_T[GHOSTS] = vec3[](
    vec3(1.00, 0.97, 0.90),
    vec3(1.00, 0.98, 0.92),
    vec3(1.00, 0.93, 0.80),
    vec3(0.92, 0.97, 1.00),
    vec3(0.90, 0.98, 1.00),
    vec3(1.00, 0.92, 0.80),
    vec3(0.94, 0.96, 1.00)
);

float edge_fade(vec2 coord) {
    return smoothstep(0.0, 0.03, coord.x) * (1.0 - smoothstep(0.97, 1.0, coord.x))
         * smoothstep(0.0, 0.03, coord.y) * (1.0 - smoothstep(0.97, 1.0, coord.y));
}

vec3 sample_ghost(float s, float tint_w) {
    vec2 c = vec2(0.5, 0.5);
    vec2 coord = c + (v_uv - c) * s;
    float fade = edge_fade(coord);
    vec2 pc = clamp(coord, vec2(0.001), vec2(0.999));
    return texture(u_bright_tex, pc).rgb * fade * tint_w;
}

vec3 sample_ghost_chrom(float s, float dc) {
    vec2 c = vec2(0.5, 0.5);
    vec2 cr = c + (v_uv - c) * (s + dc);
    vec2 cg = c + (v_uv - c) * s;
    vec2 cb = c + (v_uv - c) * (s - dc);
    float fr = edge_fade(cr);
    float fg = edge_fade(cg);
    float fb = edge_fade(cb);
    vec3 col;
    col.r = texture(u_bright_tex, clamp(cr, vec2(0.001), vec2(0.999))).r * fr;
    col.g = texture(u_bright_tex, clamp(cg, vec2(0.001), vec2(0.999))).g * fg;
    col.b = texture(u_bright_tex, clamp(cb, vec2(0.001), vec2(0.999))).b * fb;
    return col;
}

void main() {
    vec3 acc = vec3(0.0);

    // Glow: the bloom-style wide soft pass at identity position.
    vec3 glow_col = texture(u_glow_tex, v_uv).rgb;
    acc += glow_col * u_glow * vec3(1.0, 0.99, 0.97);

    // Ghost train with a light chromatic split on the main warm ghosts.
    float dc = u_chromatic * 0.02;
    for (int i = 0; i < GHOSTS; i++) {
        float s = 1.0 / G_A[i];
        float w = G_W[i] * u_ghost_intensity;
        vec3 col;
        if (i == 2 || i == 3 || i == 5) {
            col = sample_ghost_chrom(s, dc);
        } else {
            col = sample_ghost(s, 1.0);
        }
        acc += col * G_T[i] * w;
    }

    // Anamorphic streak: horizontal gaussian smear of the bright pass.
    if (u_anamorphic > 0.001) {
        vec3 st = vec3(0.0);
        float wsum = 0.0;
        for (int k = -6; k <= 6; k++) {
            float kk = float(k);
            vec2 p = v_uv + vec2(kk * 1.5 * u_texel.x, 0.0);
            vec3 tc = texture(u_glow_tex, clamp(p, vec2(0.0), vec2(1.0))).rgb;
            float wgt = exp(-(kk * kk) / (2.0 * 4.0));
            st += tc * wgt;
            wsum += wgt;
        }
        acc += st / wsum * u_anamorphic;
    }

    frag_color = vec4(max(acc * u_intensity * u_scale, vec3(0.0)), 1.0);
}
"""


@ComponentRegistry.register
class LensFlares(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "\u2726"
    render_type = "additive"
    _intensity_prop = "_intensity"

    def __init__(self):
        super().__init__()
        self._intensity: float = 1.0
        self._scale: float = 1.0
        self._glow: float = 1.0
        self._threshold: float = 1.0
        self._soft_threshold: float = 0.5
        self._ghost_intensity: float = 1.0
        self._anamorphic: float = 0.6
        self._chromatic: float = 0.8
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None
        self._extract_prog: Optional[moderngl.Program] = None
        self._extract_vao: Optional[moderngl.VertexArray] = None
        self._ds_prog: Optional[moderngl.Program] = None
        self._ds_vao: Optional[moderngl.VertexArray] = None
        self._bright_tex: Optional[moderngl.Texture] = None
        self._bright_fbo: Optional[moderngl.Framebuffer] = None
        self._glow_tex: Optional[moderngl.Texture] = None
        self._glow_fbo: Optional[moderngl.Framebuffer] = None
        self._chain_size: tuple[int, int] = (0, 0)

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.05, decimals=2),
            InspectorField("_scale", "Scale", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.05, decimals=2),
            InspectorField("_glow", "Glow", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.05, decimals=2),
            InspectorField("_threshold", "Bright Threshold", FieldType.FLOAT, min_val=0.0, max_val=2.0, step=0.05, decimals=3),
            InspectorField("_soft_threshold", "Soft Knee", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.05, decimals=2),
            InspectorField("_ghost_intensity", "Ghost Intensity", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.05, decimals=2),
            InspectorField("_anamorphic", "Anamorphic Streak", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.05, decimals=2),
            InspectorField("_chromatic", "Chromatic", FieldType.FLOAT, min_val=0.0, max_val=3.0, step=0.05, decimals=2),
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
            "_anamorphic": self._anamorphic,
            "_chromatic": self._chromatic,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> LensFlares:
        inst = super().deserialize(data)
        inst._intensity = float(data.get("_intensity", 1.0))
        inst._scale = float(data.get("_scale", 1.0))
        inst._glow = float(data.get("_glow", 1.0))
        inst._threshold = float(data.get("_threshold", 1.0))
        inst._soft_threshold = float(data.get("_soft_threshold", 0.5))
        inst._ghost_intensity = float(data.get("_ghost_intensity", 1.0))
        inst._anamorphic = float(data.get("_anamorphic", 0.6))
        inst._chromatic = float(data.get("_chromatic", 0.8))
        inst._prog = None
        inst._vao = None
        inst._vbo = None
        inst._ibo = None
        inst._extract_prog = None
        inst._extract_vao = None
        inst._ds_prog = None
        inst._ds_vao = None
        inst._bright_tex = None
        inst._bright_fbo = None
        inst._glow_tex = None
        inst._glow_fbo = None
        inst._chain_size = (0, 0)
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
            self._ds_prog = cached['_ds_prog']
            self._ds_vao = cached['_ds_vao']
            return
        self._ctx = ctx
        verts = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        self._vbo = ctx.buffer(verts.tobytes())
        self._ibo = ctx.buffer(indices.tobytes())
        self._prog = ctx.program(vertex_shader=FLARE_VERT, fragment_shader=FLARE_FRAG)
        self._extract_prog = ctx.program(vertex_shader=FLARE_VERT, fragment_shader=EXTRACT_FRAG)
        self._ds_prog = ctx.program(vertex_shader=FLARE_VERT, fragment_shader=DOWNSAMPLE_FRAG)
        self._vao = ctx.vertex_array(self._prog, [(self._vbo, '2f', 'in_position')], self._ibo)
        self._extract_vao = ctx.vertex_array(self._extract_prog, [(self._vbo, '2f', 'in_position')], self._ibo)
        self._ds_vao = ctx.vertex_array(self._ds_prog, [(self._vbo, '2f', 'in_position')], self._ibo)
        self._res_prog_cache[ctx_id] = {
            '_prog': self._prog,
            '_vao': self._vao,
            '_vbo': self._vbo,
            '_ibo': self._ibo,
            '_extract_prog': self._extract_prog,
            '_extract_vao': self._extract_vao,
            '_ds_prog': self._ds_prog,
            '_ds_vao': self._ds_vao,
        }
        if len(self._res_prog_cache) > 4:
            oldest = next(iter(self._res_prog_cache))
            self._release_cache_objects({oldest: self._res_prog_cache[oldest]})
            del self._res_prog_cache[oldest]

    def _ensure_chain(self, ctx: moderngl.Context, rw: int, rh: int):
        if self._bright_tex is not None and self._chain_size == (rw, rh):
            return
        self._release_chain()
        gw = max(rw // 2, 16)
        gh = max(rh // 2, 16)
        self._bright_tex = ctx.texture((rw, rh), 4, dtype='f4')
        self._bright_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._bright_fbo = ctx.framebuffer(self._bright_tex)
        self._glow_tex = ctx.texture((gw, gh), 4, dtype='f4')
        self._glow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._glow_fbo = ctx.framebuffer(self._glow_tex)
        self._chain_size = (rw, rh)

    def _release_chain(self):
        for obj in (self._bright_fbo, self._bright_tex, self._glow_fbo, self._glow_tex):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._bright_tex = None
        self._bright_fbo = None
        self._glow_tex = None
        self._glow_fbo = None
        self._chain_size = (0, 0)

    def render(self, ctx, scene_color_tex, scene_depth_tex,
               view_mat, proj_mat, cam_pos, viewport_w, viewport_h, **kwargs):
        if not self.enabled or not self.entity or not self.entity.active:
            return
        if self.should_skip():
            return
        self._ensure_resources(ctx)
        rw = max(64, int(viewport_w / 4))
        rh = max(36, int(viewport_h / 4))
        self._ensure_chain(ctx, rw, rh)
        prev_fbo = ctx.fbo

        ctx.disable(moderngl.BLEND)

        self._bright_fbo.use()
        self._bright_fbo.viewport = (0, 0, rw, rh)
        self._extract_prog["u_scene_color"] = 0
        self._extract_prog["u_threshold"].value = float(self._threshold)
        self._extract_prog["u_soft_threshold"].value = float(self._soft_threshold)
        scene_color_tex.use(0)
        self._extract_vao.render()

        self._glow_fbo.use()
        self._glow_fbo.viewport = (0, 0, rw // 2, rh // 2)
        self._ds_prog["u_src"] = 0
        self._ds_prog["u_texel"].value = (1.0 / rw, 1.0 / rh)
        self._bright_tex.use(0)
        self._ds_vao.render()

        if prev_fbo is not None:
            w, h = prev_fbo.size
        else:
            w, h = viewport_w, viewport_h
        prev_fbo.use()
        prev_fbo.viewport = (0, 0, w, h)
        ctx.viewport = (0, 0, w, h)

        prog = self._prog
        prog["u_bright_tex"] = 0
        prog["u_glow_tex"] = 1
        prog["u_texel"].value = (1.0 / rw, 1.0 / rh)
        prog["u_intensity"].value = float(self._intensity)
        prog["u_scale"].value = float(self._scale)
        prog["u_glow"].value = float(self._glow)
        prog["u_ghost_intensity"].value = float(self._ghost_intensity)
        prog["u_anamorphic"].value = float(self._anamorphic)
        prog["u_chromatic"].value = float(self._chromatic)

        self._bright_tex.use(0)
        self._glow_tex.use(1)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ctx.enable(moderngl.BLEND)
        self._vao.render()
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def _release_gl(self):
        self._release_chain()
        for obj in (self._prog, self._vao, self._vbo, self._ibo,
                    self._extract_prog, self._extract_vao,
                    self._ds_prog, self._ds_vao):
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
        self._ds_prog = None
        self._ds_vao = None

    @property
    def intensity(self) -> float:
        return getattr(self, '_intensity', 1.0)

    @intensity.setter
    def intensity(self, v: float):
        self._intensity = v

    @property
    def threshold(self) -> float:
        return getattr(self, '_threshold', 1.0)

    @threshold.setter
    def threshold(self, v: float):
        self._threshold = v
