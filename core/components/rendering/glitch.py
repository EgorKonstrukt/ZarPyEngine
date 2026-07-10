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
from core.ecs import ComponentRegistry
from core.components.rendering.graphics_effect import GraphicsEffect
from core.components.inspector_meta import FieldType, InspectorField


GLITCH_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

GLITCH_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform float u_intensity;
uniform float u_speed;
uniform float u_frequency;
uniform float u_time;
uniform vec2 u_resolution;
in vec2 v_uv;
out vec4 frag_color;

float hash(float x) {
    return fract(sin(x * 127.1 + u_time * u_speed) * 43758.5453);
}

void main() {
    vec2 uv = v_uv;

    float block = floor(uv.y * u_frequency);
    float rand_h = hash(block);
    float rand_v = hash(block + 1.7);

    float disp = (rand_h - 0.5) * u_intensity * 0.15;
    float block_strip = step(0.95, rand_v) * u_intensity;
    uv.x += disp + block_strip * rand_h * 0.08;
    uv.x = clamp(uv.x, 0.0, 1.0);

    float split = rand_v * u_intensity * 0.04;
    float r = texture(u_input_tex, uv + vec2(split, 0.0)).r;
    float g = texture(u_input_tex, uv).g;
    float b = texture(u_input_tex, uv - vec2(split, 0.0)).b;

    vec3 color = vec3(r, g, b);

    float scanline = sin(uv.y * u_resolution.y * 3.14159) * 0.5 + 0.5;
    scanline = smoothstep(0.3, 0.7, scanline);
    color *= 0.85 + 0.15 * scanline;

    frag_color = vec4(color, 1.0);
}
"""


@ComponentRegistry.register
class Glitch(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "Gl"
    render_type = "screen"
    _intensity_prop = "_intensity"
    def __init__(self):
        super().__init__()
        self._intensity: float = 0.5
        self._speed: float = 3.0
        self._frequency: float = 20.0
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=3.0, step=0.05, decimals=3),
            InspectorField("_speed", "Speed", FieldType.FLOAT, min_val=0.0, max_val=20.0, step=0.5, decimals=2),
            InspectorField("_frequency", "Frequency", FieldType.FLOAT, min_val=1.0, max_val=100.0, step=1.0, decimals=1),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_intensity": self._intensity,
            "_speed": self._speed,
            "_frequency": self._frequency,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Glitch:
        inst = super().deserialize(data)
        inst._intensity = float(data.get("_intensity", 0.5))
        inst._speed = float(data.get("_speed", 3.0))
        inst._frequency = float(data.get("_frequency", 20.0))
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
            vertex_shader=GLITCH_VERT,
            fragment_shader=GLITCH_FRAG
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
        if "u_speed" in self._prog:
            self._prog["u_speed"].value = self._speed
        if "u_frequency" in self._prog:
            self._prog["u_frequency"].value = self._frequency
        if "u_time" in self._prog:
            self._prog["u_time"].value = time.perf_counter()
        if "u_resolution" in self._prog:
            self._prog["u_resolution"].value = (float(viewport_w), float(viewport_h))
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
        ctx_id = id(self._ctx) if self._ctx else None
        if ctx_id is not None and ctx_id in self._res_cache:
            del self._res_cache[ctx_id]