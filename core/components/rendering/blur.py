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


BLUR_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

BLUR_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform float u_radius;
uniform vec2 u_pixel_size;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    int r = int(min(u_radius, 64.0));
    float sigma = max(u_radius * 0.5, 0.5);

    vec3 sum = texture(u_input_tex, v_uv).rgb;
    float total = 1.0;

    for (int i = 1; i <= r; i++) {
        vec2 off = vec2(float(i)) * u_pixel_size;
        float w = exp(-float(i * i) / (2.0 * sigma * sigma));
        sum += texture(u_input_tex, v_uv + off).rgb * w;
        sum += texture(u_input_tex, v_uv - off).rgb * w;
        total += 2.0 * w;
    }

    frag_color = vec4(sum / total, 1.0);
}
"""


@ComponentRegistry.register
class Blur(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "Bl"
    render_type = "screen"

    def __init__(self):
        super().__init__()
        self._radius: float = 2.0
        self._iterations: int = 1
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None
        self._temp_fbo: Optional[moderngl.Framebuffer] = None
        self._fbo_size: tuple[int, int] = (0, 0)

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_radius", "Radius", FieldType.FLOAT, min_val=0.0, max_val=32.0, step=0.5, decimals=2),
            InspectorField("_iterations", "Iterations", FieldType.INT, min_val=1, max_val=8, step=1),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_radius": self._radius,
            "_iterations": self._iterations,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Blur:
        inst = super().deserialize(data)
        inst._radius = float(data.get("_radius", 2.0))
        inst._iterations = int(data.get("_iterations", 1))
        inst._ctx = None
        inst._prog = None
        inst._vao = None
        inst._vbo = None
        inst._ibo = None
        inst._temp_fbo = None
        inst._fbo_size = (0, 0)
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
            vertex_shader=BLUR_VERT,
            fragment_shader=BLUR_FRAG
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

    def _ensure_temp_fbo(self, ctx: moderngl.Context, w: int, h: int):
        if self._temp_fbo is None or self._fbo_size != (w, h):
            if self._temp_fbo is not None:
                self._temp_fbo.release()
            self._temp_fbo = ctx.framebuffer(
                color_attachments=[ctx.texture((w, h), 4)]
            )
            self._fbo_size = (w, h)

    def render(self, ctx, scene_color_tex, scene_depth_tex,
               view_mat, proj_mat, cam_pos, viewport_w, viewport_h,
               input_tex=None, output_fbo=None, **kwargs):
        if not self.enabled or not self.entity or not self.entity.active:
            return
        self._ensure_resources(ctx)
        self._ensure_temp_fbo(ctx, viewport_w, viewport_h)

        tex = input_tex if input_tex is not None else scene_color_tex
        tex_slot = self._temp_fbo.color_attachments[0]

        for _ in range(self._iterations):
            # Horizontal pass
            self._temp_fbo.use()
            self._temp_fbo.clear(0.0, 0.0, 0.0, 1.0)
            self._prog["u_input_tex"] = 0
            tex.use(0)
            self._prog["u_radius"].value = self._radius
            self._prog["u_pixel_size"].value = (1.0 / viewport_w, 0.0)
            ctx.disable(moderngl.BLEND)
            self._vao.render()

            # Vertical pass
            if output_fbo is not None:
                output_fbo.use()
            self._prog["u_input_tex"] = 0
            tex_slot.use(0)
            self._prog["u_pixel_size"].value = (0.0, 1.0 / viewport_h)
            self._vao.render()

            tex = tex_slot

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
        if self._temp_fbo is not None:
            try:
                self._temp_fbo.release()
            except Exception:
                pass
            self._temp_fbo = None
        self._fbo_size = (0, 0)