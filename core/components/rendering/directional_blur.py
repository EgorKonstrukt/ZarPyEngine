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
from core.math3d import Vec2


DIRECTIONAL_BLUR_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

DIRECTIONAL_BLUR_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform vec2 u_direction;
uniform float u_strength;
uniform vec2 u_pixel_size;
uniform int u_samples;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec2 dir = normalize(u_direction) * u_strength;
    vec3 sum = texture(u_input_tex, v_uv).rgb;
    float total = 1.0;
    for (int i = 1; i <= u_samples; i++) {
        float t = float(i) / float(u_samples);
        vec2 off = dir * t * u_pixel_size;
        sum += texture(u_input_tex, v_uv + off).rgb;
        sum += texture(u_input_tex, v_uv - off).rgb;
        total += 2.0;
    }
    frag_color = vec4(sum / total, 1.0);
}
"""


@ComponentRegistry.register
class DirectionalBlur(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "Db"
    render_type = "screen"
    _intensity_prop = "_strength"

    def __init__(self):
        super().__init__()
        self._direction: Vec2 = Vec2(1.0, 0.0)
        self._strength: float = 8.0
        self._samples: int = 16
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_direction", "Direction", FieldType.VEC2, min_val=-1.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("_strength", "Strength", FieldType.SLIDER, min_val=0.0, max_val=64.0, step=0.5, decimals=1),
            InspectorField("_samples", "Samples", FieldType.INT_SLIDER, min_val=4, max_val=64, step=2),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_direction": self._direction.to_list(),
            "_strength": self._strength,
            "_samples": self._samples,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> DirectionalBlur:
        inst = super().deserialize(data)
        d = data.get("_direction", [1.0, 0.0])
        inst._direction = Vec2(float(d[0]), float(d[1]))
        inst._strength = float(data.get("_strength", 8.0))
        inst._samples = int(data.get("_samples", 16))
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
            self._prog = cached["_prog"]
            self._vao = cached["_vao"]
            self._vbo = cached["_vbo"]
            self._ibo = cached["_ibo"]
            return
        self._ctx = ctx
        self._prog = ctx.program(
            vertex_shader=DIRECTIONAL_BLUR_VERT,
            fragment_shader=DIRECTIONAL_BLUR_FRAG
        )
        verts = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        self._vbo = ctx.buffer(verts.tobytes())
        self._ibo = ctx.buffer(indices.tobytes())
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._vbo, "2f", "in_position")],
            self._ibo
        )
        self._res_cache[ctx_id] = {
            "_prog": self._prog,
            "_vao": self._vao,
            "_vbo": self._vbo,
            "_ibo": self._ibo,
        }
        if len(self._res_cache) > 4:
            oldest = next(iter(self._res_cache))
            for obj in self._res_cache[oldest].values():
                if obj is not None and hasattr(obj, "release"):
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
        dir_v = self._direction
        length = dir_v.length()
        if length < 1e-6:
            n = Vec2(1.0, 0.0)
        else:
            n = dir_v * (1.0 / length)
        self._prog["u_direction"].value = (n.x, n.y)
        self._prog["u_strength"].value = self._strength
        self._prog["u_pixel_size"].value = (1.0 / viewport_w, 1.0 / viewport_h)
        self._prog["u_samples"].value = self._samples
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
