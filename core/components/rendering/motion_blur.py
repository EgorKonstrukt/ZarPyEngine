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



MOTION_BLUR_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

MOTION_BLUR_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform sampler2D u_velocity_tex;
uniform sampler2D u_depth_tex;
uniform mat4 u_inv_view_proj;
uniform mat4 u_prev_view_proj;
uniform float u_threshold;
uniform int u_samples;
uniform float u_strength;
uniform bool u_use_velocity;
uniform bool u_use_velocity_tex;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 scene = texture(u_input_tex, v_uv).rgb;
    if (!u_use_velocity) {
        frag_color = vec4(scene, 1.0);
        return;
    }

    vec2 velocity;
    if (u_use_velocity_tex) {
        velocity = texture(u_velocity_tex, v_uv).rg;
    } else {
        float depth = texture(u_depth_tex, v_uv).r;
        vec4 clip_pos = vec4(v_uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
        vec4 world_pos = u_inv_view_proj * clip_pos;
        world_pos /= world_pos.w;
        vec4 prev_clip = u_prev_view_proj * world_pos;
        vec2 prev_uv = prev_clip.xy / prev_clip.w * 0.5 + 0.5;
        velocity = v_uv - prev_uv;
    }

    float speed = length(velocity);
    if (speed < u_threshold) {
        frag_color = vec4(scene, 1.0);
        return;
    }

    vec3 acc = scene;
    int count = 1;
    for (int i = 1; i <= u_samples; i++) {
        float t = float(i) / float(u_samples) * u_strength;
        vec2 sample_uv = v_uv - velocity * t;
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0)
            break;
        acc += texture(u_input_tex, sample_uv).rgb;
        count++;
    }
    frag_color = vec4(acc / float(count), 1.0);
}
"""


@ComponentRegistry.register
class MotionBlur(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "Mb"
    render_type = "screen"
    _use_velocity = True
    _intensity_prop = "_strength"

    def __init__(self):
        super().__init__()
        self._samples: int = 16
        self._strength: float = 1.0
        self._threshold: float = 0.001
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_samples", "Samples", FieldType.INT_SLIDER, min_val=4, max_val=64, step=2),
            InspectorField("_strength", "Strength", FieldType.SLIDER, min_val=0.0, max_val=1.5, step=0.01, decimals=3),
            InspectorField("_threshold", "Threshold", FieldType.SLIDER, min_val=0.0, max_val=0.01, step=0.0001, decimals=5),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_samples": self._samples,
            "_strength": self._strength,
            "_threshold": self._threshold,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> MotionBlur:
        inst = super().deserialize(data)
        inst._samples = int(data.get("_samples", 16))
        inst._strength = float(data.get("_strength", 1.0))
        inst._threshold = float(data.get("_threshold", 0.001))
        inst._ctx = None
        inst._prog = None
        inst._vao = None
        inst._vbo = None
        inst._ibo = None
        return inst

    _shader_version: int = 9
    _res_cache: dict[int, dict] = {}

    def _ensure_resources(self, ctx: moderngl.Context):
        ctx_id = (id(ctx), self._shader_version)
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
            vertex_shader=MOTION_BLUR_VERT,
            fragment_shader=MOTION_BLUR_FRAG
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

        has_vel_tex = velocity_tex is not None
        has_fallback = prev_view_proj is not None
        use_vel = has_vel_tex or has_fallback

        self._prog["u_input_tex"] = 0
        scene_color_tex.use(0)
        if has_vel_tex:
            self._prog["u_velocity_tex"] = 1
            velocity_tex.use(1)
            self._prog["u_use_velocity_tex"].value = True
        elif has_fallback:
            self._prog["u_depth_tex"] = 1
            scene_depth_tex.use(1)
            cur_vp = proj_mat @ view_mat
            inv_vp = cur_vp.inverted()
            self._prog["u_inv_view_proj"].write(inv_vp.to_f32().tobytes())
            self._prog["u_prev_view_proj"].write(prev_view_proj.to_f32().tobytes())
            self._prog["u_use_velocity_tex"].value = False
        self._prog["u_use_velocity"].value = use_vel
        for name, val in [("u_threshold", self._threshold),
                          ("u_samples", self._samples),
                          ("u_strength", self._strength)]:
            try:
                self._prog[name].value = val
            except KeyError:
                pass
        ctx.disable(moderngl.BLEND)

        if output_fbo is not None:
            output_fbo.use()
            output_fbo.viewport = (0, 0, viewport_w, viewport_h)
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