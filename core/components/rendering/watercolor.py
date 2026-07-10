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


WATERCOLOR_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

WATERCOLOR_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform vec2 u_pixel_size;
uniform float u_brush_size;
uniform float u_edge_darken;
uniform float u_wetness;
uniform float u_intensity;
in vec2 v_uv;
out vec4 frag_color;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec3 sum = vec3(0.0);
    float weights = 0.0;
    vec2 px = u_pixel_size;
    int r = int(clamp(u_brush_size, 1.0, 8.0));

    for (int y = -r; y <= r; y++) {
        for (int x = -r; x <= r; x++) {
            vec2 offset = vec2(float(x), float(y));
            float d = length(offset);
            if (d > float(r)) continue;
            float jitter = hash(v_uv + offset * 0.1) * 0.6 + 0.4;
            float w = (1.0 - d / float(r + 1)) * jitter;
            vec2 uv = v_uv + offset * px;
            vec3 c = texture(u_input_tex, clamp(uv, 0.001, 0.999)).rgb;
            sum += c * w;
            weights += w;
        }
    }

    vec3 base = sum / weights;
    float gray = dot(base, vec3(0.299, 0.587, 0.114));
    float quant = floor(gray * 6.0 + 0.5) / 6.0;
    vec3 pal = base * quant / (gray + 0.001);

    float edge = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            vec3 n = texture(u_input_tex, v_uv + vec2(float(x), float(y)) * px * 4.0).rgb;
            edge += dot(abs(base - n), vec3(0.333));
        }
    }
    edge = clamp(edge * u_edge_darken, 0.0, 1.0);

    vec3 wet = base * (1.0 - edge * 0.3 * u_wetness);
    vec3 result = mix(pal, wet, u_wetness);
    result = mix(result, result * (1.0 - edge * 0.5), u_edge_darken);

    vec3 original = texture(u_input_tex, v_uv).rgb;
    frag_color = vec4(mix(original, result, u_intensity), 1.0);
}
"""


@ComponentRegistry.register
class Watercolor(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "Wc"
    render_type = "screen"
    _intensity_prop = "_intensity"

    def __init__(self):
        super().__init__()
        self._brush_size: float = 4.0
        self._edge_darken: float = 0.8
        self._wetness: float = 0.5
        self._intensity: float = 1.0
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_brush_size", "Brush Size", FieldType.FLOAT, min_val=1.0, max_val=8.0, step=1.0, decimals=0),
            InspectorField("_edge_darken", "Edge Darken", FieldType.FLOAT, min_val=0.0, max_val=2.0, step=0.05, decimals=3),
            InspectorField("_wetness", "Wetness", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.05, decimals=3),
            InspectorField("_intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.05, decimals=3),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_brush_size": self._brush_size,
            "_edge_darken": self._edge_darken,
            "_wetness": self._wetness,
            "_intensity": self._intensity,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Watercolor:
        inst = super().deserialize(data)
        inst._brush_size = float(data.get("_brush_size", 4.0))
        inst._edge_darken = float(data.get("_edge_darken", 0.8))
        inst._wetness = float(data.get("_wetness", 0.5))
        inst._intensity = float(data.get("_intensity", 1.0))
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
            vertex_shader=WATERCOLOR_VERT,
            fragment_shader=WATERCOLOR_FRAG
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
        if "u_brush_size" in self._prog:
            self._prog["u_brush_size"].value = self._brush_size
        if "u_edge_darken" in self._prog:
            self._prog["u_edge_darken"].value = self._edge_darken
        if "u_wetness" in self._prog:
            self._prog["u_wetness"].value = self._wetness
        if "u_intensity" in self._prog:
            self._prog["u_intensity"].value = self._intensity
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