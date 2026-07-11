# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import numpy as np
import moderngl
from typing import Optional
from core.ecs.ecs import ComponentRegistry
from core.components.rendering.postfx.graphics_effect import GraphicsEffect
from core.components.inspector_meta import FieldType, InspectorField


PANINI_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

PANINI_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform vec2 u_view_extents;
uniform float u_distance;
uniform float u_crop_scale;
in vec2 v_uv;
out vec4 frag_color;

vec2 panini_generic(vec2 view_pos, float d) {
    float view_dist = 1.0 + d;
    float view_hyp_sq = view_pos.x * view_pos.x + view_dist * view_dist;

    float isect_D = view_pos.x * d;
    float isect_discrim = view_hyp_sq - isect_D * isect_D;

    float cyl_dist_minus_d = (-isect_D * view_pos.x + view_dist * sqrt(isect_discrim)) / view_hyp_sq;
    float cyl_dist = cyl_dist_minus_d + d;

    vec2 cyl_pos = view_pos * (cyl_dist / view_dist);
    return cyl_pos / (cyl_dist - d);
}

void main() {
    vec2 view_pos = (v_uv * 2.0 - 1.0) * u_view_extents * u_crop_scale;

    vec2 proj_pos = panini_generic(view_pos, u_distance);

    vec2 proj_ndc = proj_pos / u_view_extents;
    vec2 uv = proj_ndc * 0.5 + 0.5;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    frag_color = vec4(texture(u_input_tex, uv).rgb, 1.0);
}
"""


@ComponentRegistry.register
class PaniniProjection(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "Pn"
    render_type = "screen"

    def __init__(self):
        super().__init__()
        self._distance: float = 0.5
        self._crop_to_fit: float = 1.0
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_distance", "Distance", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("_crop_to_fit", "Crop to Fit", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_distance": self._distance,
            "_crop_to_fit": self._crop_to_fit,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> PaniniProjection:
        inst = super().deserialize(data)
        inst._distance = float(data.get("_distance", 0.5))
        inst._crop_to_fit = float(data.get("_crop_to_fit", 1.0))
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
            vertex_shader=PANINI_VERT,
            fragment_shader=PANINI_FRAG
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

        p = proj_mat._d
        f = p[1, 1]
        view_ext_y = 2.0 / f
        aspect = viewport_w / viewport_h
        view_ext_x = view_ext_y * aspect

        d = self._distance
        view_dist = 1.0 + d
        proj_hyp = math.sqrt(view_ext_x * view_ext_x + 1.0)
        cyl_dist_minus_d = 1.0 / proj_hyp
        cyl_dist = cyl_dist_minus_d + d
        cyl_pos_x = view_ext_x * cyl_dist_minus_d
        crop_ext_x = cyl_pos_x * (view_dist / cyl_dist)
        scale_f = crop_ext_x / view_ext_x
        crop_scale = 1.0 + (min(scale_f, 1.0) - 1.0) * self._crop_to_fit

        if "u_view_extents" in self._prog:
            self._prog["u_view_extents"].value = (float(view_ext_x), float(view_ext_y))
        if "u_distance" in self._prog:
            self._prog["u_distance"].value = d
        if "u_crop_scale" in self._prog:
            self._prog["u_crop_scale"].value = crop_scale

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