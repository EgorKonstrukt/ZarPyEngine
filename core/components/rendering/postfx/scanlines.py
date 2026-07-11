# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
import moderngl
from typing import Optional
from core.ecs.ecs import ComponentRegistry
from core.components.rendering.postfx.graphics_effect import GraphicsEffect
from core.components.inspector_meta import FieldType, InspectorField


SCANLINE_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

SCANLINE_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform vec2 u_resolution;
uniform float u_intensity;
uniform float u_frequency;
uniform float u_curvature;
uniform float u_vignette;
uniform float u_rgb_offset;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec2 uv = v_uv;

    if (u_curvature > 0.0) {
        vec2 centered = uv * 2.0 - 1.0;
        float dist = dot(centered, centered);
        uv = (centered * (1.0 + dist * u_curvature * 0.15) + 1.0) * 0.5;
    }

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 color;
    if (u_rgb_offset > 0.0) {
        float r = texture(u_input_tex, uv + vec2(u_rgb_offset, 0.0)).r;
        float g = texture(u_input_tex, uv).g;
        float b = texture(u_input_tex, uv - vec2(u_rgb_offset, 0.0)).b;
        color = vec3(r, g, b);
    } else {
        color = texture(u_input_tex, uv).rgb;
    }

    float scanline = abs(sin(uv.y * u_frequency * 3.14159265));
    color *= 1.0 - scanline * u_intensity;

    vec2 vig_uv = (uv - 0.5) * 1.1;
    float vig = 1.0 - dot(vig_uv, vig_uv) * u_vignette;
    color *= clamp(vig, 0.0, 1.0);

    frag_color = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""


@ComponentRegistry.register
class Scanlines(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "CRT"
    render_type = "screen"
    _intensity_prop = "_intensity"
    def __init__(self):
        super().__init__()
        self._intensity: float = 0.3
        self._frequency: float = 200.0
        self._curvature: float = 0.1
        self._vignette: float = 0.3
        self._rgb_offset: float = 0.002
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("_frequency", "Frequency", FieldType.FLOAT, min_val=1.0, max_val=500.0, step=1.0, decimals=1),
            InspectorField("_curvature", "Curvature", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("_vignette", "Vignette", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("_rgb_offset", "RGB Offset", FieldType.FLOAT, min_val=0.0, max_val=0.05, step=0.001, decimals=4),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_intensity": self._intensity,
            "_frequency": self._frequency,
            "_curvature": self._curvature,
            "_vignette": self._vignette,
            "_rgb_offset": self._rgb_offset,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Scanlines:
        inst = super().deserialize(data)
        inst._intensity = float(data.get("_intensity", 0.3))
        inst._frequency = float(data.get("_frequency", 200.0))
        inst._curvature = float(data.get("_curvature", 0.1))
        inst._vignette = float(data.get("_vignette", 0.3))
        inst._rgb_offset = float(data.get("_rgb_offset", 0.002))
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
            vertex_shader=SCANLINE_VERT,
            fragment_shader=SCANLINE_FRAG
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
        if "u_resolution" in self._prog:
            self._prog["u_resolution"].value = (float(viewport_w), float(viewport_h))
        if "u_intensity" in self._prog:
            self._prog["u_intensity"].value = self._intensity
        if "u_frequency" in self._prog:
            self._prog["u_frequency"].value = self._frequency
        if "u_curvature" in self._prog:
            self._prog["u_curvature"].value = self._curvature
        if "u_vignette" in self._prog:
            self._prog["u_vignette"].value = self._vignette
        if "u_rgb_offset" in self._prog:
            self._prog["u_rgb_offset"].value = self._rgb_offset
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