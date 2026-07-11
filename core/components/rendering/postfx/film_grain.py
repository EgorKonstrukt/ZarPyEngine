# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
import moderngl
import time
from typing import Optional
from core.ecs.ecs import ComponentRegistry
from core.components.rendering.postfx.graphics_effect import GraphicsEffect
from core.components.inspector_meta import FieldType, InspectorField


GRAIN_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

GRAIN_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform float u_intensity;
uniform float u_size;
uniform float u_luminance_influence;
uniform float u_time;
in vec2 v_uv;
out vec4 frag_color;

float grain_hash(vec2 uv, float t) {
    return fract(sin(dot(uv * 12.9898 + t, vec2(78.233, 151.7182))) * 43758.5453);
}

void main() {
    vec3 color = texture(u_input_tex, v_uv).rgb;
    vec2 grain_uv = v_uv * u_size;
    float noise = grain_hash(grain_uv, u_time);
    float lum = dot(color, vec3(0.299, 0.587, 0.114));
    float grain = (noise - 0.5) * u_intensity;
    grain *= 1.0 - lum * u_luminance_influence;
    color += grain;
    frag_color = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""


@ComponentRegistry.register
class FilmGrain(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "G"
    render_type = "screen"
    _intensity_prop = "_intensity"
    def __init__(self):
        super().__init__()
        self._intensity: float = 0.1
        self._size: float = 1.0
        self._luminance_influence: float = 0.5
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("_size", "Size", FieldType.FLOAT, min_val=0.1, max_val=10.0, step=0.1, decimals=2),
            InspectorField("_luminance_influence", "Luminance Influence", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.05, decimals=3),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_intensity": self._intensity,
            "_size": self._size,
            "_luminance_influence": self._luminance_influence,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> FilmGrain:
        inst = super().deserialize(data)
        inst._intensity = float(data.get("_intensity", 0.1))
        inst._size = float(data.get("_size", 1.0))
        inst._luminance_influence = float(data.get("_luminance_influence", 0.5))
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
            vertex_shader=GRAIN_VERT,
            fragment_shader=GRAIN_FRAG
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
        if "u_intensity" in self._prog:
            self._prog["u_intensity"].value = self._intensity
        if "u_size" in self._prog:
            self._prog["u_size"].value = self._size
        if "u_luminance_influence" in self._prog:
            self._prog["u_luminance_influence"].value = self._luminance_influence
        if "u_time" in self._prog:
            self._prog["u_time"].value = time.perf_counter() * 10.0
        ctx.disable(moderngl.BLEND)
        self._vao.render()