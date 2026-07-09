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


RADIAL_BLUR_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

RADIAL_BLUR_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform sampler2D u_velocity_tex;
uniform vec2 u_center;
uniform float u_strength;
uniform float u_threshold;
uniform int u_samples;
uniform bool u_use_velocity;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec2 dir = v_uv - u_center;
    float d = length(dir);
    if (d < 1e-6) {
        frag_color = texture(u_input_tex, v_uv);
        return;
    }
    if (u_use_velocity) {
        vec2 vel = texture(u_velocity_tex, v_uv).rg;
        float speed = length(vel);
        if (speed < u_threshold) {
            frag_color = texture(u_input_tex, v_uv);
            return;
        }
    } else {
        frag_color = texture(u_input_tex, v_uv);
        return;
    }
    dir /= d;
    vec3 sum = texture(u_input_tex, v_uv).rgb;
    float total = 1.0;
    for (int i = 1; i <= u_samples; i++) {
        float t = float(i) / float(u_samples) * u_strength;
        vec2 off = dir * t;
        sum += texture(u_input_tex, v_uv + off).rgb;
        sum += texture(u_input_tex, v_uv - off).rgb;
        total += 2.0;
    }
    frag_color = vec4(sum / total, 1.0);
}
"""


@ComponentRegistry.register
class RadialBlur(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "Rb"
    render_type = "screen"
    _use_velocity = True
    _intensity_prop = "_strength"

    def __init__(self):
        super().__init__()
        self._center: Vec2 = Vec2(0.5, 0.5)
        self._strength: float = 0.05
        self._samples: int = 16
        self._threshold: float = 0.001
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_center", "Center", FieldType.VEC2, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("_strength", "Strength", FieldType.SLIDER, min_val=0.0, max_val=0.5, step=0.001, decimals=4),
            InspectorField("_samples", "Samples", FieldType.INT_SLIDER, min_val=4, max_val=64, step=2),
            InspectorField("_threshold", "Threshold", FieldType.SLIDER, min_val=0.0, max_val=0.01, step=0.0001, decimals=5),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_center": self._center.to_list(),
            "_strength": self._strength,
            "_samples": self._samples,
            "_threshold": self._threshold,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> RadialBlur:
        inst = super().deserialize(data)
        c = data.get("_center", [0.5, 0.5])
        inst._center = Vec2(float(c[0]), float(c[1]))
        inst._strength = float(data.get("_strength", 0.05))
        inst._samples = int(data.get("_samples", 16))
        inst._threshold = float(data.get("_threshold", 0.001))
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
            vertex_shader=RADIAL_BLUR_VERT,
            fragment_shader=RADIAL_BLUR_FRAG
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
               input_tex=None, output_fbo=None,
               velocity_tex=None, prev_view_proj=None, **kwargs):
        if not self.enabled or not self.entity or not self.entity.active:
            return
        self._ensure_resources(ctx)
        tex = input_tex if input_tex is not None else scene_color_tex
        self._prog["u_input_tex"] = 0
        tex.use(0)
        if velocity_tex is not None:
            self._prog["u_velocity_tex"] = 1
            velocity_tex.use(1)
        self._prog["u_use_velocity"].value = velocity_tex is not None
        self._prog["u_center"].value = (self._center.x, self._center.y)
        self._prog["u_strength"].value = self._strength
        self._prog["u_samples"].value = self._samples
        self._prog["u_threshold"].value = self._threshold
        ctx.disable(moderngl.BLEND)
        if output_fbo is not None:
            output_fbo.use()
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
