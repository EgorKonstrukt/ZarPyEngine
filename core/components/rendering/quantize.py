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


QUANTIZE_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

QUANTIZE_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform float u_bit_depth;
uniform float u_intensity;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 color = texture(u_input_tex, v_uv).rgb;
    float levels = exp2(u_bit_depth);
    vec3 quantized = floor(color * levels) / levels;
    frag_color = vec4(mix(color, quantized, u_intensity), 1.0);
}
"""


@ComponentRegistry.register
class Quantize(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "Qn"
    render_type = "screen"
    _intensity_prop = "_intensity"
    def __init__(self):
        super().__init__()
        self._bit_depth: float = 4.0
        self._intensity: float = 1.0
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_bit_depth", "Bit Depth", FieldType.FLOAT, min_val=1.0, max_val=8.0, step=0.5, decimals=2),
            InspectorField("_intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.05, decimals=3),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_bit_depth": self._bit_depth,
            "_intensity": self._intensity,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Quantize:
        inst = super().deserialize(data)
        inst._bit_depth = float(data.get("_bit_depth", 4.0))
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
            vertex_shader=QUANTIZE_VERT,
            fragment_shader=QUANTIZE_FRAG
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
        if "u_bit_depth" in self._prog:
            self._prog["u_bit_depth"].value = self._bit_depth
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