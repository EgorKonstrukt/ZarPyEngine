# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from enum import Enum
import numpy as np
import moderngl
from typing import Optional
from core.ecs.ecs import ComponentRegistry
from core.components.rendering.postfx.graphics_effect import GraphicsEffect
from core.components.inspector_meta import FieldType, InspectorField


class TonemapMode(Enum):
    Off = "Off"
    ACES = "ACES"
    Reinhard = "Reinhard"
    Neutral = "Neutral"


_TONEMAP_INT_MAP = {"Off": 0, "ACES": 1, "Reinhard": 2, "Neutral": 3}


CG_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

CG_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform float u_exposure;
uniform float u_contrast;
uniform float u_saturation;
uniform float u_temperature;
uniform float u_tint;
uniform int u_tonemap_mode;
in vec2 v_uv;
out vec4 frag_color;

vec3 tonemap_aces(vec3 c) {
    float a = 2.51;
    float b = 0.03;
    float c_ = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((c * (a * c + b)) / (c * (c_ * c + d) + e), 0.0, 1.0);
}

vec3 tonemap_reinhard(vec3 c) {
    return c / (1.0 + c);
}

vec3 tonemap_neutral(vec3 c) {
    return 1.0 - exp(-c * 2.0);
}

vec3 white_balance(vec3 c, float temp, float tint) {
    float r_scale = 1.0 + temp * 0.05;
    float b_scale = 1.0 - temp * 0.05;
    float g_scale = 1.0 + tint * 0.05;
    return c * vec3(r_scale, g_scale, b_scale);
}

void main() {
    vec3 color = texture(u_input_tex, v_uv).rgb;
    color *= pow(2.0, u_exposure);
    color = white_balance(color, u_temperature, u_tint);
    if (u_tonemap_mode == 1) {
        color = tonemap_aces(color);
    } else if (u_tonemap_mode == 2) {
        color = tonemap_reinhard(color);
    } else if (u_tonemap_mode == 3) {
        color = tonemap_neutral(color);
    }
    color = (color - 0.5) * u_contrast + 0.5;
    float lum = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(lum), color, u_saturation);
    frag_color = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""


@ComponentRegistry.register
class ColorGrading(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "CG"
    render_type = "screen"

    def __init__(self):
        super().__init__()
        self._exposure: float = 0.0
        self._contrast: float = 1.0
        self._saturation: float = 1.0
        self._temperature: float = 0.0
        self._tint: float = 0.0
        self._tonemap_mode: TonemapMode = TonemapMode.Off
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_exposure", "Exposure (EV)", FieldType.FLOAT, min_val=-5.0, max_val=5.0, step=0.1, decimals=2),
            InspectorField("_contrast", "Contrast", FieldType.FLOAT, min_val=0.0, max_val=3.0, step=0.05, decimals=3),
            InspectorField("_saturation", "Saturation", FieldType.FLOAT, min_val=0.0, max_val=3.0, step=0.05, decimals=3),
            InspectorField("_temperature", "Temperature", FieldType.FLOAT, min_val=-1.0, max_val=1.0, step=0.05, decimals=3),
            InspectorField("_tint", "Tint", FieldType.FLOAT, min_val=-1.0, max_val=1.0, step=0.05, decimals=3),
            InspectorField("_tonemap_mode", "Tonemapping", FieldType.ENUM, min_val=0.0, max_val=3.0, step=1.0, decimals=0, enum_class=TonemapMode),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_exposure": self._exposure,
            "_contrast": self._contrast,
            "_saturation": self._saturation,
            "_temperature": self._temperature,
            "_tint": self._tint,
            "_tonemap_mode": self._tonemap_mode.value if isinstance(self._tonemap_mode, TonemapMode) else self._tonemap_mode,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ColorGrading:
        inst = super().deserialize(data)
        inst._exposure = float(data.get("_exposure", 0.0))
        inst._contrast = float(data.get("_contrast", 1.0))
        inst._saturation = float(data.get("_saturation", 1.0))
        inst._temperature = float(data.get("_temperature", 0.0))
        inst._tint = float(data.get("_tint", 0.0))
        raw = data.get("_tonemap_mode", 0)
        if isinstance(raw, str):
            inst._tonemap_mode = TonemapMode(raw)
        else:
            inst._tonemap_mode = TonemapMode.Off
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
            vertex_shader=CG_VERT,
            fragment_shader=CG_FRAG
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
        if "u_exposure" in self._prog:
            self._prog["u_exposure"].value = self._exposure
        if "u_contrast" in self._prog:
            self._prog["u_contrast"].value = self._contrast
        if "u_saturation" in self._prog:
            self._prog["u_saturation"].value = self._saturation
        if "u_temperature" in self._prog:
            self._prog["u_temperature"].value = self._temperature
        if "u_tint" in self._prog:
            self._prog["u_tint"].value = self._tint
        if "u_tonemap_mode" in self._prog:
            self._prog["u_tonemap_mode"].value = _TONEMAP_INT_MAP.get(self._tonemap_mode.value, 0)
        ctx.disable(moderngl.BLEND)
        self._vao.render()