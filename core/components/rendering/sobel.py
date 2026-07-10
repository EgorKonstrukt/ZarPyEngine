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


SOBEL_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

SOBEL_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform vec2 u_pixel_size;
uniform float u_threshold;
uniform float u_thickness;
uniform float u_intensity;
uniform vec3 u_edge_color;
uniform int u_mode;
in vec2 v_uv;
out vec4 frag_color;

vec3 sobel(vec2 uv, vec2 px) {
    float tl = dot(texture(u_input_tex, uv + vec2(-1.0, -1.0) * px * u_thickness).rgb, vec3(0.299, 0.587, 0.114));
    float t  = dot(texture(u_input_tex, uv + vec2( 0.0, -1.0) * px * u_thickness).rgb, vec3(0.299, 0.587, 0.114));
    float tr = dot(texture(u_input_tex, uv + vec2( 1.0, -1.0) * px * u_thickness).rgb, vec3(0.299, 0.587, 0.114));
    float l  = dot(texture(u_input_tex, uv + vec2(-1.0,  0.0) * px * u_thickness).rgb, vec3(0.299, 0.587, 0.114));
    float r  = dot(texture(u_input_tex, uv + vec2( 1.0,  0.0) * px * u_thickness).rgb, vec3(0.299, 0.587, 0.114));
    float bl = dot(texture(u_input_tex, uv + vec2(-1.0,  1.0) * px * u_thickness).rgb, vec3(0.299, 0.587, 0.114));
    float b  = dot(texture(u_input_tex, uv + vec2( 0.0,  1.0) * px * u_thickness).rgb, vec3(0.299, 0.587, 0.114));
    float br = dot(texture(u_input_tex, uv + vec2( 1.0,  1.0) * px * u_thickness).rgb, vec3(0.299, 0.587, 0.114));

    float gx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
    float gy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
    float mag = sqrt(gx * gx + gy * gy);
    float angle = atan(gy, gx);
    return vec3(mag, angle, 0.0);
}

void main() {
    vec2 px = u_pixel_size;
    vec3 s = sobel(v_uv, px);
    float edge = smoothstep(u_threshold, u_threshold + 0.5, s.r);

    if (u_mode == 0) {
        vec3 scene = texture(u_input_tex, v_uv).rgb;
        vec3 result = mix(scene, u_edge_color, edge * u_intensity);
        frag_color = vec4(result, 1.0);
    } else if (u_mode == 1) {
        vec3 result = mix(vec3(1.0), u_edge_color, edge * u_intensity);
        frag_color = vec4(result, 1.0);
    } else if (u_mode == 2) {
        vec3 scene = texture(u_input_tex, v_uv).rgb;
        vec3 result = mix(scene, vec3(0.0), edge * u_intensity);
        frag_color = vec4(result, 1.0);
    } else if (u_mode == 3) {
        float a = s.g + 3.14159;
        vec3 dir_color = vec3(sin(a), sin(a + 2.094), sin(a + 4.188)) * 0.5 + 0.5;
        vec3 result = mix(vec3(0.0), dir_color, edge * u_intensity);
        frag_color = vec4(result, 1.0);
    } else {
        vec3 scene = texture(u_input_tex, v_uv).rgb;
        float gray = dot(scene, vec3(0.299, 0.587, 0.114));
        vec3 glow = vec3(edge * 2.0);
        vec3 result = mix(scene, scene + glow, u_intensity * edge);
        frag_color = vec4(result, 1.0);
    }
}
"""


@ComponentRegistry.register
class Sobel(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "Sb"
    render_type = "screen"
    _intensity_prop = "_intensity"

    def __init__(self):
        super().__init__()
        self._threshold: float = 0.1
        self._thickness: float = 1.0
        self._intensity: float = 1.0
        self._edge_color = (0.0, 0.0, 0.0)
        self._mode: int = 0
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_mode", "Mode", FieldType.INT, min_val=0, max_val=4, step=1, decimals=0),
            InspectorField("_threshold", "Threshold", FieldType.FLOAT, min_val=0.0, max_val=2.0, step=0.05, decimals=3),
            InspectorField("_thickness", "Thickness", FieldType.FLOAT, min_val=0.5, max_val=8.0, step=0.5, decimals=1),
            InspectorField("_intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=3.0, step=0.05, decimals=3),
            InspectorField("_edge_color", "Edge Color", FieldType.COLOR),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_mode": self._mode,
            "_threshold": self._threshold,
            "_thickness": self._thickness,
            "_intensity": self._intensity,
            "_edge_color": list(self._edge_color),
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Sobel:
        inst = super().deserialize(data)
        inst._mode = int(data.get("_mode", 0))
        inst._threshold = float(data.get("_threshold", 0.1))
        inst._thickness = float(data.get("_thickness", 1.0))
        inst._intensity = float(data.get("_intensity", 1.0))
        inst._edge_color = tuple(data.get("_edge_color", [0.0, 0.0, 0.0]))
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
            vertex_shader=SOBEL_VERT,
            fragment_shader=SOBEL_FRAG
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
        if "u_pixel_size" in self._prog:
            self._prog["u_pixel_size"].value = (1.0 / viewport_w, 1.0 / viewport_h)
        if "u_threshold" in self._prog:
            self._prog["u_threshold"].value = self._threshold
        if "u_thickness" in self._prog:
            self._prog["u_thickness"].value = self._thickness
        if "u_intensity" in self._prog:
            self._prog["u_intensity"].value = self._intensity
        if "u_edge_color" in self._prog:
            self._prog["u_edge_color"].value = tuple(self._edge_color)
        if "u_mode" in self._prog:
            self._prog["u_mode"].value = self._mode
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