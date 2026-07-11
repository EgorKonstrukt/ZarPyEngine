# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import numpy as np
import moderngl
from typing import Optional
from core.ecs.ecs import ComponentRegistry
from core.components.rendering.postfx.graphics_effect import GraphicsEffect
from core.components.inspector_meta import FieldType, InspectorField


BLOOM_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

EXTRACT_FRAG = """
#version 460 core
uniform sampler2D u_scene_color;
uniform float u_threshold;
uniform float u_soft_threshold;
uniform vec2 u_texel_size;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec2 h = u_texel_size;
    vec3 color = vec3(0.0);
    color += texture(u_scene_color, v_uv + vec2(-h.x, -h.y)).rgb;
    color += texture(u_scene_color, v_uv + vec2( h.x, -h.y)).rgb;
    color += texture(u_scene_color, v_uv + vec2(-h.x,  h.y)).rgb;
    color += texture(u_scene_color, v_uv + vec2( h.x,  h.y)).rgb;
    color *= 0.25;
    float lum = dot(color, vec3(0.299, 0.587, 0.114));
    float knee = u_threshold * u_soft_threshold;
    float soft = lum - u_threshold + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 0.0001);
    float bright = max(lum - u_threshold, soft);
    frag_color = vec4(color * (bright / max(lum, 0.0001)), 1.0);
}
"""

BLUR_H_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform vec2 u_texel_size;
uniform float u_diffusion;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec2 step = vec2(u_texel_size.x * u_diffusion, 0.0);
    float w[5] = float[](0.227027, 0.194595, 0.121621, 0.054054, 0.016216);
    vec3 result = texture(u_input_tex, v_uv).rgb * w[0];
    for (int i = 1; i <= 4; i++) {
        result += texture(u_input_tex, v_uv + step * float( i)).rgb * w[i];
        result += texture(u_input_tex, v_uv - step * float( i)).rgb * w[i];
    }
    frag_color = vec4(result, 1.0);
}
"""

BLUR_V_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform vec2 u_texel_size;
uniform float u_diffusion;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec2 step = vec2(0.0, u_texel_size.y * u_diffusion);
    float w[5] = float[](0.227027, 0.194595, 0.121621, 0.054054, 0.016216);
    vec3 result = texture(u_input_tex, v_uv).rgb * w[0];
    for (int i = 1; i <= 4; i++) {
        result += texture(u_input_tex, v_uv + step * float( i)).rgb * w[i];
        result += texture(u_input_tex, v_uv - step * float( i)).rgb * w[i];
    }
    frag_color = vec4(result, 1.0);
}
"""

OUTPUT_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform sampler2D u_dirt_tex;
uniform float u_intensity;
uniform float u_dirt_intensity;
uniform int u_has_dirt;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 bloom = texture(u_input_tex, v_uv).rgb;
    if (u_has_dirt != 0) {
        vec3 dirt = texture(u_dirt_tex, v_uv).rgb;
        bloom += dirt * u_dirt_intensity;
    }
    frag_color = vec4(bloom * u_intensity, 1.0);
}
"""


@ComponentRegistry.register
class Bloom(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "\u2606"
    render_type = "additive"
    _intensity_prop = "_intensity"

    def __init__(self):
        super().__init__()
        self._intensity: float = 1.0
        self._threshold: float = 0.8
        self._soft_threshold: float = 0.5
        self._downsample: int = 4
        self._iterations: int = 1
        self._diffusion: float = 1.0
        self._dirt_texture: str = ""
        self._dirt_intensity: float = 0.0
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._extract_prog: Optional[moderngl.Program] = None
        self._blur_h_prog: Optional[moderngl.Program] = None
        self._blur_v_prog: Optional[moderngl.Program] = None
        self._output_prog: Optional[moderngl.Program] = None
        self._extract_vao: Optional[moderngl.VertexArray] = None
        self._blur_h_vao: Optional[moderngl.VertexArray] = None
        self._blur_v_vao: Optional[moderngl.VertexArray] = None
        self._output_vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None
        self._bloom_tex: Optional[moderngl.Texture] = None
        self._bloom_fbo: Optional[moderngl.Framebuffer] = None
        self._bloom_temp_tex: Optional[moderngl.Texture] = None
        self._bloom_temp_fbo: Optional[moderngl.Framebuffer] = None
        self._bloom_size: tuple[int, int] = (0, 0)
        self._dirt_tex: Optional[moderngl.Texture] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.1, decimals=2),
            InspectorField("_threshold", "Threshold", FieldType.FLOAT, min_val=0.0, max_val=2.0, step=0.05, decimals=3),
            InspectorField("_soft_threshold", "Soft Knee", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.05, decimals=2),
            InspectorField("_downsample", "Downsample", FieldType.INT, min_val=1, max_val=8, step=1),
            InspectorField("_iterations", "Iterations", FieldType.INT, min_val=1, max_val=8, step=1),
            InspectorField("_diffusion", "Diffusion", FieldType.FLOAT, min_val=0.5, max_val=5.0, step=0.1, decimals=2),
            InspectorField("_dirt_texture", "Dirt Texture", FieldType.RESOURCE_PATH, min_val=0.0, max_val=0.0, step=0.0, decimals=0, file_filter="Images (*.png *.jpg *.jpeg *.tga *.bmp)"),
            InspectorField("_dirt_intensity", "Dirt Intensity", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.05, decimals=2),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_intensity": self._intensity,
            "_threshold": self._threshold,
            "_soft_threshold": self._soft_threshold,
            "_downsample": self._downsample,
            "_iterations": self._iterations,
            "_diffusion": self._diffusion,
            "_dirt_texture": self._dirt_texture,
            "_dirt_intensity": self._dirt_intensity,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Bloom:
        inst = super().deserialize(data)
        inst._intensity = float(data.get("_intensity", 1.0))
        inst._threshold = float(data.get("_threshold", 0.8))
        inst._soft_threshold = float(data.get("_soft_threshold", 0.5))
        inst._downsample = int(data.get("_downsample", 4))
        inst._iterations = int(data.get("_iterations", 1))
        inst._diffusion = float(data.get("_diffusion", 1.0))
        inst._dirt_texture = str(data.get("_dirt_texture", "") or "")
        inst._dirt_intensity = float(data.get("_dirt_intensity", 0.0))
        inst._prog = None
        inst._extract_prog = None
        inst._blur_h_prog = None
        inst._blur_v_prog = None
        inst._output_prog = None
        inst._extract_vao = None
        inst._blur_h_vao = None
        inst._blur_v_vao = None
        inst._output_vao = None
        inst._vbo = None
        inst._ibo = None
        inst._bloom_tex = None
        inst._bloom_fbo = None
        inst._bloom_temp_tex = None
        inst._bloom_temp_fbo = None
        inst._bloom_size = (0, 0)
        inst._dirt_tex = None
        return inst

    _res_prog_cache: dict[int, dict] = {}

    def _ensure_resources(self, ctx: moderngl.Context, viewport_w: int, viewport_h: int):
        ctx_id = id(ctx)
        cached = self._res_prog_cache.get(ctx_id)
        if cached is not None:
            self._ctx = ctx
            self._prog = cached['_prog']
            self._extract_prog = cached['_extract_prog']
            self._blur_h_prog = cached['_blur_h_prog']
            self._blur_v_prog = cached['_blur_v_prog']
            self._output_prog = cached['_output_prog']
            self._extract_vao = cached['_extract_vao']
            self._blur_h_vao = cached['_blur_h_vao']
            self._blur_v_vao = cached['_blur_v_vao']
            self._output_vao = cached['_output_vao']
            self._vbo = cached['_vbo']
            self._ibo = cached['_ibo']
            return
        else:
            self._ctx = ctx
            verts = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32)
            indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
            self._vbo = ctx.buffer(verts.tobytes())
            self._ibo = ctx.buffer(indices.tobytes())

            self._prog = ctx.program(vertex_shader=BLOOM_VERT, fragment_shader=EXTRACT_FRAG)
            self._extract_prog = ctx.program(vertex_shader=BLOOM_VERT, fragment_shader=EXTRACT_FRAG)
            self._blur_h_prog = ctx.program(vertex_shader=BLOOM_VERT, fragment_shader=BLUR_H_FRAG)
            self._blur_v_prog = ctx.program(vertex_shader=BLOOM_VERT, fragment_shader=BLUR_V_FRAG)
            self._output_prog = ctx.program(vertex_shader=BLOOM_VERT, fragment_shader=OUTPUT_FRAG)

            self._extract_vao = ctx.vertex_array(
                self._extract_prog,
                [(self._vbo, '2f', 'in_position')],
                self._ibo
            )
            self._blur_h_vao = ctx.vertex_array(
                self._blur_h_prog,
                [(self._vbo, '2f', 'in_position')],
                self._ibo
            )
            self._blur_v_vao = ctx.vertex_array(
                self._blur_v_prog,
                [(self._vbo, '2f', 'in_position')],
                self._ibo
            )
            self._output_vao = ctx.vertex_array(
                self._output_prog,
                [(self._vbo, '2f', 'in_position')],
                self._ibo
            )
            self._res_prog_cache[ctx_id] = {
                '_prog': self._prog,
                '_extract_prog': self._extract_prog,
                '_blur_h_prog': self._blur_h_prog,
                '_blur_v_prog': self._blur_v_prog,
                '_output_prog': self._output_prog,
                '_extract_vao': self._extract_vao,
                '_blur_h_vao': self._blur_h_vao,
                '_blur_v_vao': self._blur_v_vao,
                '_output_vao': self._output_vao,
                '_vbo': self._vbo,
                '_ibo': self._ibo,
            }
            if len(self._res_prog_cache) > 4:
                oldest = next(iter(self._res_prog_cache))
                self._release_cache_objects({oldest: self._res_prog_cache[oldest]})
                del self._res_prog_cache[oldest]

        ds = max(1, self._downsample)
        bw = max(1, viewport_w // ds)
        bh = max(1, viewport_h // ds)
        if self._bloom_size != (bw, bh):
            self._release_bloom_resources()
            self._bloom_tex = ctx.texture((bw, bh), 4, dtype='f1')
            self._bloom_tex.repeat_x = False
            self._bloom_tex.repeat_y = False
            self._bloom_fbo = ctx.framebuffer(self._bloom_tex)
            self._bloom_temp_tex = ctx.texture((bw, bh), 4, dtype='f1')
            self._bloom_temp_tex.repeat_x = False
            self._bloom_temp_tex.repeat_y = False
            self._bloom_temp_fbo = ctx.framebuffer(self._bloom_temp_tex)
            self._bloom_size = (bw, bh)

        if self._dirt_texture and self._dirt_intensity > 0.0:
            if self._dirt_tex is None:
                try:
                    from PIL import Image
                    img = Image.open(self._dirt_texture).convert("RGBA")
                    img = img.resize((viewport_w, viewport_h), Image.LANCZOS)
                    self._dirt_tex = ctx.texture((viewport_w, viewport_h), 4, data=img.tobytes())
                    self._dirt_tex.repeat_x = False
                    self._dirt_tex.repeat_y = False
                except Exception:
                    self._dirt_tex = None
        else:
            self._dirt_tex = None

    def _release_bloom_resources(self):
        for obj in [self._bloom_fbo, self._bloom_tex, self._bloom_temp_fbo, self._bloom_temp_tex]:
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._bloom_fbo = None
        self._bloom_tex = None
        self._bloom_temp_fbo = None
        self._bloom_temp_tex = None
        self._bloom_size = (0, 0)

    def _release_dirt(self):
        if self._dirt_tex is not None:
            try:
                self._dirt_tex.release()
            except Exception:
                pass
            self._dirt_tex = None

    def render(self, ctx, scene_color_tex, scene_depth_tex,
               view_mat, proj_mat, cam_pos, viewport_w, viewport_h,
               input_tex=None, output_fbo=None, **kwargs):
        if not self.enabled or not self.entity or not self.entity.active:
            return
        self._ensure_resources(ctx, viewport_w, viewport_h)

        bw, bh = self._bloom_size
        prev_fbo = ctx.fbo

        self._bloom_fbo.use()
        self._bloom_fbo.viewport = (0, 0, bw, bh)
        self._extract_prog["u_scene_color"] = 0
        self._extract_prog["u_threshold"].value = self._threshold
        self._extract_prog["u_soft_threshold"].value = self._soft_threshold
        self._extract_prog["u_texel_size"].value = (1.0 / viewport_w, 1.0 / viewport_h)
        ctx.disable(moderngl.BLEND)
        scene_color_tex.use(0)
        self._extract_vao.render()

        for _ in range(self._iterations):
            self._bloom_temp_fbo.use()
            self._bloom_temp_fbo.viewport = (0, 0, bw, bh)
            self._blur_h_prog["u_input_tex"] = 0
            self._blur_h_prog["u_texel_size"].value = (1.0 / bw, 1.0 / bh)
            self._blur_h_prog["u_diffusion"].value = self._diffusion
            self._bloom_tex.use(0)
            self._blur_h_vao.render()

            self._bloom_fbo.use()
            self._bloom_fbo.viewport = (0, 0, bw, bh)
            self._blur_v_prog["u_input_tex"] = 0
            self._blur_v_prog["u_texel_size"].value = (1.0 / bw, 1.0 / bh)
            self._blur_v_prog["u_diffusion"].value = self._diffusion
            self._bloom_temp_tex.use(0)
            self._blur_v_vao.render()

        if prev_fbo is not None:
            prev_fbo.use()
            prev_fbo.viewport = (0, 0, viewport_w, viewport_h)
        elif ctx.screen is not None:
            ctx.screen.use()
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ctx.enable(moderngl.BLEND)
        self._output_prog["u_input_tex"] = 0
        self._output_prog["u_intensity"].value = self._intensity
        if self._dirt_tex is not None and self._dirt_intensity > 0.0:
            self._output_prog["u_dirt_tex"] = 1
            self._dirt_tex.use(1)
            self._output_prog["u_dirt_intensity"].value = self._dirt_intensity
            self._output_prog["u_has_dirt"].value = 1
        else:
            self._output_prog["u_has_dirt"].value = 0
        self._bloom_tex.use(0)
        self._output_vao.render()
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def _release_gl(self):
        self._release_bloom_resources()
        self._release_dirt()
        for obj in (self._prog, self._extract_prog, self._blur_h_prog, self._blur_v_prog, self._output_prog,
                    self._extract_vao, self._blur_h_vao, self._blur_v_vao, self._output_vao,
                    self._vbo, self._ibo):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._prog = None
        self._extract_prog = None
        self._blur_h_prog = None
        self._blur_v_prog = None
        self._output_prog = None
        self._extract_vao = None
        self._blur_h_vao = None
        self._blur_v_vao = None
        self._output_vao = None
        self._vbo = None
        self._ibo = None
        self._bloom_tex = None
        self._bloom_fbo = None
        self._bloom_temp_tex = None
        self._bloom_temp_fbo = None
        self._bloom_size = (0, 0)

    @property
    def intensity(self) -> float:
        return getattr(self, '_intensity', 1.0)

    @intensity.setter
    def intensity(self, v: float):
        self._intensity = v

    @property
    def threshold(self) -> float:
        return getattr(self, '_threshold', 0.8)

    @threshold.setter
    def threshold(self, v: float):
        self._threshold = v