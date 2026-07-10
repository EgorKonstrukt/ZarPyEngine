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


LENS_DISTORTION_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

LENS_DISTORTION_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform float u_strength;
uniform float u_chromatic;
uniform float u_scale;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec2 uv = v_uv - 0.5;
    float r = length(uv);
    float k = 1.0 + r * r * u_strength;
    float s = 1.0 / (u_scale + 1.0e-6);

    vec2 duv = uv * k;

    float r_chroma = u_chromatic * 0.02;
    float r2 = texture(u_input_tex, duv * s + 0.5 + vec2(r_chroma, 0.0)).r;
    float g2 = texture(u_input_tex, duv * s + 0.5).g;
    float b2 = texture(u_input_tex, duv * s + 0.5 - vec2(r_chroma, 0.0)).b;

    vec2 edge_uv = duv * s + 0.5;
    if (edge_uv.x < 0.0 || edge_uv.x > 1.0 ||
        edge_uv.y < 0.0 || edge_uv.y > 1.0) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    frag_color = vec4(r2, g2, b2, 1.0);
}
"""


@ComponentRegistry.register
class LensDistortion(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "Ld"
    render_type = "screen"

    def __init__(self):
        super().__init__()
        self._strength: float = 0.3
        self._chromatic: float = 0.5
        self._scale: float = 0.9
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_strength", "Strength", FieldType.FLOAT, min_val=-1.0, max_val=1.0, step=0.05, decimals=3),
            InspectorField("_chromatic", "Chromatic", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.05, decimals=3),
            InspectorField("_scale", "Scale", FieldType.FLOAT, min_val=0.5, max_val=1.5, step=0.05, decimals=3),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_strength": self._strength,
            "_chromatic": self._chromatic,
            "_scale": self._scale,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> LensDistortion:
        inst = super().deserialize(data)
        inst._strength = float(data.get("_strength", 0.3))
        inst._chromatic = float(data.get("_chromatic", 0.5))
        inst._scale = float(data.get("_scale", 0.9))
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
            vertex_shader=LENS_DISTORTION_VERT,
            fragment_shader=LENS_DISTORTION_FRAG
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
        if "u_strength" in self._prog:
            self._prog["u_strength"].value = self._strength
        if "u_chromatic" in self._prog:
            self._prog["u_chromatic"].value = self._chromatic
        if "u_scale" in self._prog:
            self._prog["u_scale"].value = self._scale
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