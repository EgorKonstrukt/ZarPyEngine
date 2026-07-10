# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
import moderngl
from typing import Optional
from core.ecs import ComponentRegistry
from core.components.rendering.graphics_effect import GraphicsEffect
from core.components.inspector_meta import FieldType, InspectorField


HATCHING_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

HATCHING_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform sampler2D u_depth_tex;
uniform vec2 u_pixel_size;
uniform float u_intensity;
uniform vec3 u_edge_color;
uniform float u_edge_threshold;
uniform float u_hatch_scale;
uniform float u_noise_frequency;
uniform float u_noise_offset_intensity;
uniform float u_distort;
uniform float u_normal_strength;
uniform vec3 u_ink_color;
uniform vec3 u_paper_color;
uniform sampler2D u_hatch1;
uniform sampler2D u_hatch2;
uniform sampler2D u_hatch3;
in vec2 v_uv;
out vec4 frag_color;

// ---- Perlin noise (port of the reference) ----
vec2 hash(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

vec2 fade(vec2 t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float grad(vec2 h, vec2 dir) {
    vec2 grad_dir = vec2(h.x * 2.0 - 1.0, h.y * 2.0 - 1.0);
    return dot(grad_dir, dir);
}

float perlin(vec2 pos) {
    vec2 p = floor(pos);
    vec2 f = fract(pos);
    f = fade(f);
    vec2 top_left = p;
    vec2 top_right = p + vec2(1.0, 0.0);
    vec2 bottom_left = p + vec2(0.0, 1.0);
    vec2 bottom_right = p + vec2(1.0, 1.0);
    float tl = grad(hash(top_left), f);
    float tr = grad(hash(top_right), f - vec2(1.0, 0.0));
    float bl = grad(hash(bottom_left), f - vec2(0.0, 1.0));
    float br = grad(hash(bottom_right), f - vec2(1.0, 1.0));
    float top = mix(tl, tr, f.x);
    float bottom = mix(bl, br, f.x);
    return mix(top, bottom, f.y);
}

// View normal reconstructed from the depth buffer (no G-buffer needed).
vec3 reconstruct_normal(vec2 uv, vec2 texel) {
    float c = texture(u_depth_tex, uv).r;
    float l = texture(u_depth_tex, uv - vec2(texel.x, 0.0)).r;
    float r = texture(u_depth_tex, uv + vec2(texel.x, 0.0)).r;
    float d = texture(u_depth_tex, uv - vec2(0.0, texel.y)).r;
    float u = texture(u_depth_tex, uv + vec2(0.0, texel.y)).r;
    vec3 n = vec3((l - r), (d - u), u_normal_strength * 0.05);
    return normalize(n);
}

// Three tiled hatch textures (generated procedurally, see _build_hatch_textures).
// Sampled exactly like the Godot reference: texture(hatchN, distorted_uv * scale).

void main() {
    vec2 screen_uv = v_uv;
    vec3 origin_color = texture(u_input_tex, screen_uv).rgb;

    // Perlin-noise UV offset (hand-drawn jitter).
    vec2 noiseValue = vec2(
        perlin(screen_uv * u_noise_frequency),
        perlin((screen_uv + vec2(0.5, 0.5)) * u_noise_frequency)
    );
    vec2 uv = screen_uv + noiseValue * u_noise_offset_intensity;

    float dx = u_pixel_size.x;
    float dy = u_pixel_size.y;

    // 3x3 samples for Sobel (depth + reconstructed normal).
    float d00 = texture(u_depth_tex, uv + vec2(-dx, -dy)).r;
    float d01 = texture(u_depth_tex, uv + vec2( 0.0, -dy)).r;
    float d02 = texture(u_depth_tex, uv + vec2( dx, -dy)).r;
    float d10 = texture(u_depth_tex, uv + vec2(-dx, 0.0)).r;
    float d12 = texture(u_depth_tex, uv + vec2( dx, 0.0)).r;
    float d20 = texture(u_depth_tex, uv + vec2(-dx,  dy)).r;
    float d21 = texture(u_depth_tex, uv + vec2( 0.0,  dy)).r;
    float d22 = texture(u_depth_tex, uv + vec2( dx,  dy)).r;

    vec3 n00 = reconstruct_normal(uv + vec2(-dx, -dy), vec2(dx, dy));
    vec3 n01 = reconstruct_normal(uv + vec2( 0.0, -dy), vec2(dx, dy));
    vec3 n02 = reconstruct_normal(uv + vec2( dx, -dy), vec2(dx, dy));
    vec3 n10 = reconstruct_normal(uv + vec2(-dx, 0.0), vec2(dx, dy));
    vec3 n12 = reconstruct_normal(uv + vec2( dx, 0.0), vec2(dx, dy));
    vec3 n20 = reconstruct_normal(uv + vec2(-dx,  dy), vec2(dx, dy));
    vec3 n21 = reconstruct_normal(uv + vec2( 0.0,  dy), vec2(dx, dy));
    vec3 n22 = reconstruct_normal(uv + vec2( dx,  dy), vec2(dx, dy));

    // Depth Sobel (reference kernels).
    float depthSobelX = -d00 - 2.0 * d10 - d20 + d02 + 2.0 * d12 + d22;
    float depthSobelY = -d00 + d02 - 2.0 * d01 + 2.0 * d21 - d20 + d22;
    // Normal Sobel.
    float normalSobelX = -n00.x - 2.0 * n10.x - n20.x + n02.x + 2.0 * n12.x + n22.x;
    float normalSobelY = -n00.y + n02.y - 2.0 * n01.y + 2.0 * n21.y - n20.y + n22.y;

    float depthMagnitude = length(vec2(depthSobelX, depthSobelY));
    float normalMagnitude = length(vec2(normalSobelX, normalSobelY));
    float magnitude = depthMagnitude + normalMagnitude;

    // Distort the hatching along the surface normal.
    vec3 normal_sample = reconstruct_normal(uv, vec2(dx, dy));
    vec2 distorted_uv = uv + normal_sample.xy * u_distort;

    vec3 screen_color = texture(u_input_tex, uv).rgb;
    float luminance = dot(screen_color, vec3(0.299, 0.587, 0.114));

    vec3 texture_color;
    if (luminance < 0.6) {
        texture_color = mix(u_paper_color, u_ink_color,
                             texture(u_hatch3, distorted_uv * u_hatch_scale).r);
    } else if (luminance < 0.7) {
        texture_color = mix(u_paper_color, u_ink_color,
                             texture(u_hatch2, distorted_uv * u_hatch_scale).r);
    } else if (luminance < 0.8) {
        texture_color = mix(u_paper_color, u_ink_color,
                             texture(u_hatch1, distorted_uv * u_hatch_scale).r);
    } else {
        texture_color = origin_color;
    }

    vec3 drawn;
    if (magnitude > u_edge_threshold) {
        drawn = u_edge_color;
    } else {
        drawn = texture_color;
    }

    vec3 final_color = mix(origin_color, drawn, clamp(u_intensity, 0.0, 1.0));
    frag_color = vec4(final_color, 1.0);
}
"""


@ComponentRegistry.register
class Hatching(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "Ht"
    render_type = "screen"
    _intensity_prop = "_intensity"

    def __init__(self):
        super().__init__()
        self._intensity: float = 1.0
        self._edge_color = (0.0, 0.0, 0.0)
        self._edge_threshold: float = 0.05
        self._hatch_scale: float = 20.0
        self._noise_frequency: float = 10.0
        self._noise_offset_intensity: float = 0.002
        self._distort: float = 0.04
        self._normal_strength: float = 1.0
        self._ink_color = (0.08, 0.07, 0.09)
        self._paper_color = (0.95, 0.93, 0.86)
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None
        self._hatch_tex: list = [None, None, None]

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.05, decimals=3),
            InspectorField("_edge_color", "Edge Color", FieldType.COLOR),
            InspectorField("_edge_threshold", "Edge Threshold", FieldType.FLOAT, min_val=0.001, max_val=0.5, step=0.001, decimals=3),
            InspectorField("_hatch_scale", "Hatch Scale", FieldType.FLOAT, min_val=2.0, max_val=60.0, step=1.0, decimals=0),
            InspectorField("_noise_frequency", "Noise Frequency", FieldType.FLOAT, min_val=0.1, max_val=20.0, step=0.1, decimals=1),
            InspectorField("_noise_offset_intensity", "Noise Offset", FieldType.FLOAT, min_val=0.0, max_val=0.05, step=0.001, decimals=3),
            InspectorField("_distort", "Hatch Distort", FieldType.FLOAT, min_val=0.0, max_val=0.3, step=0.005, decimals=3),
            InspectorField("_normal_strength", "Normal Strength", FieldType.FLOAT, min_val=0.1, max_val=5.0, step=0.1, decimals=2),
            InspectorField("_ink_color", "Ink Color", FieldType.COLOR),
            InspectorField("_paper_color", "Paper Color", FieldType.COLOR),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_intensity": self._intensity,
            "_edge_color": list(self._edge_color),
            "_edge_threshold": self._edge_threshold,
            "_hatch_scale": self._hatch_scale,
            "_noise_frequency": self._noise_frequency,
            "_noise_offset_intensity": self._noise_offset_intensity,
            "_distort": self._distort,
            "_normal_strength": self._normal_strength,
            "_ink_color": list(self._ink_color),
            "_paper_color": list(self._paper_color),
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Hatching:
        inst = super().deserialize(data)
        inst._intensity = float(data.get("_intensity", 1.0))
        inst._edge_color = tuple(data.get("_edge_color", [0.0, 0.0, 0.0]))
        inst._edge_threshold = float(data.get("_edge_threshold", 0.05))
        inst._hatch_scale = float(data.get("_hatch_scale", 20.0))
        inst._noise_frequency = float(data.get("_noise_frequency", 10.0))
        inst._noise_offset_intensity = float(data.get("_noise_offset_intensity", 0.002))
        inst._distort = float(data.get("_distort", 0.04))
        inst._normal_strength = float(data.get("_normal_strength", 1.0))
        inst._ink_color = tuple(data.get("_ink_color", [0.08, 0.07, 0.09]))
        inst._paper_color = tuple(data.get("_paper_color", [0.95, 0.93, 0.86]))
        inst._ctx = None
        inst._prog = None
        inst._vao = None
        inst._vbo = None
        inst._ibo = None
        inst._hatch_tex = [None, None, None]
        return inst

    _res_cache: dict[int, dict] = {}

    @staticmethod
    def _build_hatch_textures(ctx: moderngl.Context) -> list:
        N = 64

        def smoothstep(e0, e1, x):
            t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        yy, xx = np.mgrid[0:N, 0:N]

        def tile(lines):
            mask = np.ones((N, N), dtype=np.float32)
            for ang, period, thick in lines:
                a = np.deg2rad(ang)
                px = -np.sin(a)
                py = np.cos(a)
                pc = (xx * px + yy * py)
                m = np.mod(pc + period * 0.5, period) - period * 0.5
                dist = np.abs(m)
                ink = 1.0 - smoothstep(thick * 0.5 - 0.7, thick * 0.5 + 0.7, dist)
                mask = np.minimum(mask, 1.0 - ink)
            ink_mask = 1.0 - mask
            arr = (ink_mask * 255.0).astype(np.uint8)
            return np.dstack([arr, arr, arr, np.full_like(arr, 255)])

        # Lighter -> darker (matches luminance bands 0.8 / 0.7 / 0.6).
        specs = [
            [(45.0, 32.0, 2.0)],                          # hatch1: sparse
            [(45.0, 16.0, 2.0), (135.0, 16.0, 2.0)],      # hatch2: cross
            [(45.0, 8.0, 2.0), (135.0, 8.0, 2.0)],        # hatch3: dense cross
        ]
        texs = []
        for spec in specs:
            tex = ctx.texture((N, N), 4, dtype='u1')
            tex.write(tile(spec).tobytes())
            tex.repeat_x = True
            tex.repeat_y = True
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            texs.append(tex)
        return texs

    def _ensure_resources(self, ctx: moderngl.Context):
        ctx_id = id(ctx)
        cached = self._res_cache.get(ctx_id)
        if cached is not None:
            self._ctx = ctx
            self._prog = cached['_prog']
            self._vao = cached['_vao']
            self._vbo = cached['_vbo']
            self._ibo = cached['_ibo']
            self._hatch_tex = cached['_hatch_tex']
            return
        self._ctx = ctx
        self._prog = ctx.program(
            vertex_shader=HATCHING_VERT,
            fragment_shader=HATCHING_FRAG
        )
        verts = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        self._vbo = ctx.buffer(verts.tobytes())
        self._ibo = ctx.buffer(indices.tobytes())
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._vbo, '2f', 'in_position')],
            self._ibo
        )
        self._hatch_tex = self._build_hatch_textures(ctx)
        self._res_cache[ctx_id] = {
            '_prog': self._prog,
            '_vao': self._vao,
            '_vbo': self._vbo,
            '_ibo': self._ibo,
            '_hatch_tex': self._hatch_tex,
        }
        if len(self._res_cache) > 4:
            oldest = next(iter(self._res_cache))
            for obj in self._res_cache[oldest].values():
                if obj is not None and hasattr(obj, 'release'):
                    try:
                        obj.release()
                    except Exception:
                        pass
            del self._res_cache[oldest]

    def render(self, ctx, scene_color_tex, scene_depth_tex,
               view_mat, proj_mat, cam_pos, viewport_w, viewport_h,
               input_tex=None, output_fbo=None, **kwargs):
        if not self.enabled or not self.entity or not self.entity.active:
            return
        self._ensure_resources(ctx)
        tex = input_tex if input_tex is not None else scene_color_tex
        self._prog["u_input_tex"] = 0
        tex.use(0)
        if "u_depth_tex" in self._prog:
            self._prog["u_depth_tex"] = 1
            scene_depth_tex.use(1)
        if "u_pixel_size" in self._prog:
            self._prog["u_pixel_size"].value = (1.0 / viewport_w, 1.0 / viewport_h)
        if "u_intensity" in self._prog:
            self._prog["u_intensity"].value = self._intensity
        if "u_edge_color" in self._prog:
            self._prog["u_edge_color"].value = tuple(self._edge_color)
        if "u_edge_threshold" in self._prog:
            self._prog["u_edge_threshold"].value = self._edge_threshold
        if "u_hatch_scale" in self._prog:
            self._prog["u_hatch_scale"].value = self._hatch_scale
        if "u_noise_frequency" in self._prog:
            self._prog["u_noise_frequency"].value = self._noise_frequency
        if "u_noise_offset_intensity" in self._prog:
            self._prog["u_noise_offset_intensity"].value = self._noise_offset_intensity
        if "u_distort" in self._prog:
            self._prog["u_distort"].value = self._distort
        if "u_normal_strength" in self._prog:
            self._prog["u_normal_strength"].value = self._normal_strength
        if "u_ink_color" in self._prog:
            self._prog["u_ink_color"].value = tuple(self._ink_color)
        if "u_paper_color" in self._prog:
            self._prog["u_paper_color"].value = tuple(self._paper_color)
        for i, t in enumerate(self._hatch_tex):
            if t is not None and ("u_hatch%d" % (i + 1)) in self._prog:
                self._prog["u_hatch%d" % (i + 1)] = 2 + i
                t.use(2 + i)
        ctx.disable(moderngl.BLEND)
        self._vao.render()

    def _release_gl(self):
        for obj in (self._prog, self._vao, self._vbo, self._ibo):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        for t in self._hatch_tex:
            if t is not None:
                try:
                    t.release()
                except Exception:
                    pass
        self._ctx = None
        self._prog = None
        self._vao = None
        self._vbo = None
        self._ibo = None
        self._hatch_tex = [None, None, None]