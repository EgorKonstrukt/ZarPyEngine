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


FXAA_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

FXAA_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform vec2 u_pixel_size;
uniform float u_subpix;
uniform float u_reduce;
uniform float u_reduce_min;
uniform float u_span;
in vec2 v_uv;
out vec4 frag_color;

const vec3 LUMA = vec3(0.299, 0.587, 0.114);

void main() {
    vec2 res = u_pixel_size;
    vec3 rgbNW = texture(u_input_tex, v_uv + vec2(-1.0, 1.0) * res).rgb;
    vec3 rgbNE = texture(u_input_tex, v_uv + vec2( 1.0, 1.0) * res).rgb;
    vec3 rgbSW = texture(u_input_tex, v_uv + vec2(-1.0, -1.0) * res).rgb;
    vec3 rgbSE = texture(u_input_tex, v_uv + vec2( 1.0, -1.0) * res).rgb;
    vec3 rgbM  = texture(u_input_tex, v_uv).rgb;

    float lumaNW = dot(rgbNW, LUMA);
    float lumaNE = dot(rgbNE, LUMA);
    float lumaSW = dot(rgbSW, LUMA);
    float lumaSE = dot(rgbSE, LUMA);
    float lumaM  = dot(rgbM, LUMA);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    vec2 dir;
    dir.x = -((lumaNW + lumaNW) - (lumaNE + lumaNE)) - ((lumaSW + lumaSW) - (lumaSE + lumaSE));
    dir.y =  ((lumaNW + lumaNW) - (lumaSW + lumaSW)) + ((lumaNE + lumaNE) - (lumaSE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * u_reduce), u_reduce_min);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(vec2(u_span), max(vec2(-u_span), dir * rcpDirMin)) * res * u_subpix;

    vec3 rgbA = 0.5 * (
        texture(u_input_tex, v_uv + dir * (1.0 / 3.0 - 0.5)).rgb +
        texture(u_input_tex, v_uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture(u_input_tex, v_uv + dir * (0.0 / 3.0 - 0.5)).rgb +
        texture(u_input_tex, v_uv + dir * (3.0 / 3.0 - 0.5)).rgb);

    float lumaB = dot(rgbB, LUMA);
    vec3 result = ((lumaB < lumaMin) || (lumaB > lumaMax)) ? rgbA : rgbB;
    frag_color = vec4(result, 1.0);
}
"""


@ComponentRegistry.register
class FXAA(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "FX"
    render_type = "screen"
    _intensity_prop = "_subpix"

    def __init__(self):
        super().__init__()
        self._subpix: float = 1.0
        self._reduce: float = 0.125
        self._reduce_min: float = 0.0078125
        self._span: float = 8.0
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_subpix", "Subpixel", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("_reduce", "Reduce", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.001, decimals=3),
            InspectorField("_reduce_min", "Reduce Min", FieldType.FLOAT, min_val=0.0, max_val=0.1, step=0.0001, decimals=4),
            InspectorField("_span", "Span", FieldType.FLOAT, min_val=1.0, max_val=16.0, step=0.1, decimals=2),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_subpix": self._subpix,
            "_reduce": self._reduce,
            "_reduce_min": self._reduce_min,
            "_span": self._span,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> FXAA:
        inst = super().deserialize(data)
        inst._subpix = float(data.get("_subpix", 1.0))
        inst._reduce = float(data.get("_reduce", 0.125))
        inst._reduce_min = float(data.get("_reduce_min", 0.0078125))
        inst._span = float(data.get("_span", 8.0))
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
            vertex_shader=FXAA_VERT,
            fragment_shader=FXAA_FRAG
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
        if output_fbo is not None:
            output_fbo.use()
            output_fbo.viewport = (0, 0, viewport_w, viewport_h)
        ctx.disable(moderngl.BLEND)
        self._prog["u_input_tex"] = 0
        tex.use(0)
        if "u_pixel_size" in self._prog:
            self._prog["u_pixel_size"].value = (1.0 / viewport_w, 1.0 / viewport_h)
        if "u_subpix" in self._prog:
            self._prog["u_subpix"].value = self._subpix
        if "u_reduce" in self._prog:
            self._prog["u_reduce"].value = self._reduce
        if "u_reduce_min" in self._prog:
            self._prog["u_reduce_min"].value = self._reduce_min
        if "u_span" in self._prog:
            self._prog["u_span"].value = self._span
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

    @property
    def subpix(self) -> float:
        return getattr(self, '_subpix', 1.0)

    @subpix.setter
    def subpix(self, v: float):
        self._subpix = v

    @property
    def reduce(self) -> float:
        return getattr(self, '_reduce', 0.125)

    @reduce.setter
    def reduce(self, v: float):
        self._reduce = v

    @property
    def reduce_min(self) -> float:
        return getattr(self, '_reduce_min', 0.0078125)

    @reduce_min.setter
    def reduce_min(self, v: float):
        self._reduce_min = v

    @property
    def span(self) -> float:
        return getattr(self, '_span', 8.0)

    @span.setter
    def span(self, v: float):
        self._span = v