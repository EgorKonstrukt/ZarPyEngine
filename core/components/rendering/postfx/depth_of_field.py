# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
import moderngl
from typing import Optional
from enum import Enum
from core.ecs.ecs import ComponentRegistry
from core.components.rendering.postfx.graphics_effect import GraphicsEffect
from core.components.inspector_meta import FieldType, InspectorField


class DoFMode(Enum):
    GAUSSIAN = "Gaussian"
    BOKEH = "Bokeh"


DOF_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

DOF_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform sampler2D u_depth_tex;
uniform mat4 u_inv_proj;
uniform float u_focal_distance;
uniform float u_focal_range;
uniform float u_aperture;
uniform float u_max_blur_size;
uniform float u_bokeh_boost;
uniform float u_foreground_scale;
uniform int u_mode;
uniform int u_ring_count;
uniform int u_blade_count;
uniform float u_blade_curvature;
uniform float u_blade_rotation;
uniform bool u_visualize_coc;
uniform vec2 u_pixel_size;
in vec2 v_uv;
out vec4 frag_color;

float view_z(vec2 uv, float depth) {
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 v = u_inv_proj * ndc;
    return v.z / v.w;
}

float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float bokeh_radius(float angle) {
    float bc = float(u_blade_count);
    float blade_angle = 6.28318 / bc;
    float a = mod(angle + u_blade_rotation * 0.017453, blade_angle);
    float half_angle = blade_angle * 0.5;
    float poly_r = cos(a - half_angle) / cos(half_angle);
    return mix(poly_r, 1.0, u_blade_curvature);
}

float compute_coc(float z) {
    float d = z - u_focal_distance;
    float blend = smoothstep(0.0, max(u_focal_range, 0.001), abs(d));
    float coc = u_max_blur_size * u_aperture * blend * abs(d) / max(z, 0.001);
    if (d < 0.0) {
        coc = min(coc, u_max_blur_size * u_aperture * u_foreground_scale);
    } else {
        coc = min(coc, u_max_blur_size * u_aperture);
    }
    return d < 0.0 ? -coc : coc;
}

void main() {
    vec3 color = texture(u_input_tex, v_uv).rgb;
    float depth = texture(u_depth_tex, v_uv).r;
    bool is_sky = depth >= 1.0;

    float z = is_sky ? 0.0 : -view_z(v_uv, depth);
    float coc;
    float blend;
    if (is_sky) {
        coc = u_max_blur_size * u_aperture;
        blend = 1.0;
    } else {
        coc = compute_coc(z);
        blend = smoothstep(0.0, max(u_focal_range, 0.001), abs(z - u_focal_distance));
    }
    float c_abs = abs(coc);

    if (u_visualize_coc) {
        frag_color = vec4(mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), blend), 1.0);
        return;
    }

    if (c_abs < 0.5) {
        frag_color = vec4(color, 1.0);
        return;
    }

    float jit = hash2(v_uv) * 6.28318;
    int rings = is_sky ? min(u_ring_count, 2) : u_ring_count;
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    for (int r = 1; r <= 8; r++) {
        if (r > rings) break;
        float t = float(r) / float(rings);
        float radius = t * c_abs;
        if (radius < 0.5) continue;
        int samples = r * 8;
        float a0 = jit + float(r) * 2.39996;
        for (int i = 0; i < 64; i++) {
            if (i >= samples) break;
            float a = a0 + 6.28318 * float(i) / float(samples);
            float shape = 1.0;
            if (u_mode == 1) {
                shape = bokeh_radius(a);
            }
            vec2 dir = vec2(cos(a), sin(a));
            vec2 uv = v_uv + dir * radius * shape * u_pixel_size;
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) continue;
            float sd = texture(u_depth_tex, uv).r;
            float sc = abs(compute_coc(-view_z(uv, sd)));
            float spread = max(c_abs, sc);
            if (t * spread * shape < 0.5) continue;
            vec2 uv2 = v_uv + dir * t * spread * shape * u_pixel_size;
            if (uv2.x < 0.0 || uv2.x > 1.0 || uv2.y < 0.0 || uv2.y > 1.0) continue;
            vec3 s = texture(u_input_tex, uv2).rgb;
            float l = dot(s, vec3(0.299, 0.587, 0.114));
            float w = 1.0 + u_bokeh_boost * l;
            acc += s * w;
            wsum += w;
        }
    }
    vec3 blur = wsum > 0.0 ? acc / wsum : color;
    frag_color = vec4(mix(color, blur, blend), 1.0);
}
"""


@ComponentRegistry.register
class DepthOfField(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "Df"
    render_type = "screen"
    _skip_rate = 0

    def __init__(self):
        super().__init__()
        self._mode: DoFMode = DoFMode.GAUSSIAN
        self._focal_distance: float = 10.0
        self._focal_range: float = 8.0
        self._aperture: float = 1.0
        self._max_blur_size: float = 12.0
        self._bokeh_boost: float = 0.5
        self._foreground_scale: float = 2.0
        self._ring_count: int = 3
        self._blade_count: int = 6
        self._blade_curvature: float = 1.0
        self._blade_rotation: float = 0.0
        self._visualize_coc: bool = False
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_header_mode", "Mode", FieldType.HEADER),
            InspectorField("_mode", "Mode", FieldType.ENUM, enum_class=DoFMode),

            InspectorField("_header_focus", "Focus", FieldType.HEADER),
            InspectorField("_focal_distance", "Focal Distance", FieldType.SLIDER, min_val=0.1, max_val=200.0, step=0.1, decimals=2),
            InspectorField("_focal_range", "Focal Range", FieldType.SLIDER, min_val=0.001, max_val=100.0, step=0.1, decimals=2),
            InspectorField("_aperture", "Aperture", FieldType.SLIDER, min_val=0.0, max_val=5.0, step=0.05, decimals=3),
            InspectorField("_max_blur_size", "Max Blur Size", FieldType.SLIDER, min_val=1.0, max_val=50.0, step=0.5, decimals=1),
            InspectorField("_bokeh_boost", "Bokeh Highlight Boost", FieldType.SLIDER, min_val=0.0, max_val=3.0, step=0.05, decimals=2),
            InspectorField("_foreground_scale", "Foreground Blur Scale", FieldType.SLIDER, min_val=1.0, max_val=5.0, step=0.1, decimals=1),

            InspectorField("_header_quality", "Quality", FieldType.HEADER),
            InspectorField("_ring_count", "Rings", FieldType.INT_SLIDER, min_val=1, max_val=5, step=1),

            InspectorField("_header_bokeh", "Bokeh Shape", FieldType.HEADER),
            InspectorField("_blade_count", "Blade Count", FieldType.INT_SLIDER, min_val=3, max_val=9, step=1),
            InspectorField("_blade_curvature", "Curvature", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.05, decimals=3),
            InspectorField("_blade_rotation", "Rotation", FieldType.SLIDER, min_val=0.0, max_val=360.0, step=1.0, decimals=1),

            InspectorField("_header_debug", "Debug", FieldType.HEADER),
            InspectorField("_visualize_coc", "Visualize CoC", FieldType.BOOL),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_mode": self._mode.value if hasattr(self._mode, 'value') else "Gaussian",
            "_focal_distance": self._focal_distance,
            "_focal_range": self._focal_range,
            "_aperture": self._aperture,
            "_max_blur_size": self._max_blur_size,
            "_bokeh_boost": self._bokeh_boost,
            "_foreground_scale": self._foreground_scale,
            "_ring_count": self._ring_count,
            "_blade_count": self._blade_count,
            "_blade_curvature": self._blade_curvature,
            "_blade_rotation": self._blade_rotation,
            "_visualize_coc": self._visualize_coc,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> DepthOfField:
        inst = super().deserialize(data)
        mode_str = data.get("_mode", "Gaussian")
        try:
            inst._mode = DoFMode(mode_str)
        except ValueError:
            inst._mode = DoFMode.GAUSSIAN
        inst._focal_distance = float(data.get("_focal_distance", 10.0))
        inst._focal_range = float(data.get("_focal_range", 8.0))
        inst._aperture = float(data.get("_aperture", 1.0))
        inst._max_blur_size = float(data.get("_max_blur_size", 12.0))
        inst._bokeh_boost = float(data.get("_bokeh_boost", 0.5))
        inst._foreground_scale = float(data.get("_foreground_scale", 2.0))
        inst._ring_count = int(data.get("_ring_count", 3))
        inst._blade_count = int(data.get("_blade_count", 6))
        inst._blade_curvature = float(data.get("_blade_curvature", 1.0))
        inst._blade_rotation = float(data.get("_blade_rotation", 0.0))
        inst._visualize_coc = bool(data.get("_visualize_coc", False))
        inst._ctx = None
        inst._prog = None
        inst._vao = None
        inst._vbo = None
        inst._ibo = None
        return inst

    _res_cache: dict[int, dict] = {}

    def _ensure_resources(self, ctx: moderngl.Context):
        ctx_id = id(ctx)
        cached = self._res_cache.get(ctx_id)
        if cached is not None:
            self._ctx = ctx
            self._prog = cached['_prog']
            self._vao = cached['_vao']
            self._vbo = cached['_vbo']
            self._ibo = cached['_ibo']
            return
        self._ctx = ctx
        self._prog = ctx.program(
            vertex_shader=DOF_VERT,
            fragment_shader=DOF_FRAG
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
        self._res_cache[ctx_id] = {
            '_prog': self._prog,
            '_vao': self._vao,
            '_vbo': self._vbo,
            '_ibo': self._ibo,
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
        if "u_inv_proj" in self._prog:
            self._prog["u_inv_proj"].write(proj_mat.inverted().to_f32().tobytes())
        if "u_focal_distance" in self._prog:
            self._prog["u_focal_distance"].value = self._focal_distance
        if "u_focal_range" in self._prog:
            self._prog["u_focal_range"].value = self._focal_range
        if "u_aperture" in self._prog:
            self._prog["u_aperture"].value = self._aperture
        if "u_max_blur_size" in self._prog:
            self._prog["u_max_blur_size"].value = self._max_blur_size
        if "u_bokeh_boost" in self._prog:
            self._prog["u_bokeh_boost"].value = self._bokeh_boost
        if "u_foreground_scale" in self._prog:
            self._prog["u_foreground_scale"].value = self._foreground_scale
        if "u_mode" in self._prog:
            self._prog["u_mode"].value = 1 if self._mode == DoFMode.BOKEH else 0
        if "u_ring_count" in self._prog:
            self._prog["u_ring_count"].value = self._ring_count
        if "u_blade_count" in self._prog:
            self._prog["u_blade_count"].value = self._blade_count
        if "u_blade_curvature" in self._prog:
            self._prog["u_blade_curvature"].value = self._blade_curvature
        if "u_blade_rotation" in self._prog:
            self._prog["u_blade_rotation"].value = self._blade_rotation
        if "u_visualize_coc" in self._prog:
            self._prog["u_visualize_coc"].value = self._visualize_coc
        if "u_pixel_size" in self._prog:
            self._prog["u_pixel_size"].value = (1.0 / viewport_w, 1.0 / viewport_h)
        ctx.disable(moderngl.BLEND)
        self._vao.render()

    def _release_gl(self):
        for obj in (self._prog, self._vao, self._vbo, self._ibo):
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