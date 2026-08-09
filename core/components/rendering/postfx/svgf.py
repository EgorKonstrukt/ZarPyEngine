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


SVGF_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

SVGF_REPROJ_FRAG = """
#version 460 core
uniform sampler2D u_current;
uniform sampler2D u_depth;
uniform sampler2D u_velocity;
uniform sampler2D u_history;
uniform sampler2D u_moment;
uniform vec2 u_pixel;
uniform float u_has_velocity;
uniform float u_has_history;
uniform float u_disocclusion;
uniform float u_stability;
in vec2 v_uv;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_moment;

float svgf_luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 svgf_rgb_to_ycocg(vec3 c) {
    return vec3(
        dot(c, vec3(0.25, 0.5, 0.25)),
        dot(c, vec3(0.5, 0.0, -0.5)),
        dot(c, vec3(-0.25, 0.5, -0.25))
    );
}

vec3 svgf_ycocg_to_rgb(vec3 c) {
    return vec3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

vec3 svgf_clip_to_box(vec3 history, vec3 center, vec3 extents) {
    vec3 v = history - center;
    vec3 unit = v / max(extents, vec3(1e-5));
    float ma = max(unit.x, max(unit.y, unit.z));
    float mi = min(unit.x, min(unit.y, unit.z));
    float m = max(ma, -mi);
    return m > 1.0 ? center + v / m : history;
}

vec2 svgf_dilate_velocity(vec2 uv, float curD, out float nearestD) {
    vec2 bestMotion = texture(u_velocity, uv).rg;
    nearestD = curD;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            if (x == 0 && y == 0) continue;
            vec2 suv = uv + vec2(float(x), float(y)) * u_pixel;
            float d = texture(u_depth, suv).r;
            if (d < nearestD) {
                nearestD = d;
                bestMotion = texture(u_velocity, suv).rg;
            }
        }
    }
    return bestMotion;
}

void main() {
    vec3 cur = texture(u_current, v_uv).rgb;
    float curD = texture(u_depth, v_uv).r;
    float curLuma = svgf_luma(cur);

    if (u_has_history < 0.5) {
        out_color = vec4(cur, 1.0);
        out_moment = vec2(curLuma, curLuma * curLuma);
        return;
    }

    float nearestD;
    vec2 motion = vec2(0.0);
    if (u_has_velocity > 0.5) {
        motion = svgf_dilate_velocity(v_uv, curD, nearestD);
    }
    vec2 prev_uv = v_uv - motion;

    vec3 mn = cur;
    vec3 mx = cur;
    float lumaSum = curLuma;
    float lumaSqSum = curLuma * curLuma;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            if (x == 0 && y == 0) continue;
            vec2 ouv = v_uv + vec2(float(x), float(y)) * u_pixel;
            vec3 c = texture(u_current, ouv).rgb;
            mn = min(mn, c);
            mx = max(mx, c);
            float l = svgf_luma(c);
            lumaSum += l;
            lumaSqSum += l * l;
        }
    }

    vec3 centerY = svgf_rgb_to_ycocg((mn + mx) * 0.5);
    float meanLuma = lumaSum / 9.0;
    float varLuma = max(lumaSqSum / 9.0 - meanLuma * meanLuma, 0.0);
    float stdLuma = sqrt(varLuma);
    vec3 extents = max(vec3(stdLuma * 1.5), vec3(1e-4));

    vec3 hist = texture(u_history, prev_uv).rgb;
    vec3 histY = svgf_rgb_to_ycocg(hist);
    vec3 histYClipped = svgf_clip_to_box(histY, centerY, extents);
    vec3 histClipped = clamp(svgf_ycocg_to_rgb(histYClipped), mn, mx);

    float blend = u_stability;
    if (any(lessThan(prev_uv, vec2(0.0))) || any(greaterThan(prev_uv, vec2(1.0)))) {
        blend = 0.0;
    } else {
        float prevD = texture(u_depth, prev_uv).r;
        float ddepth = abs(curD - prevD);
        float diso = smoothstep(0.0, max(u_disocclusion, 1e-4), ddepth);
        blend = mix(u_stability, 0.0, diso);
    }

    vec3 result = mix(cur, histClipped, clamp(blend, 0.0, 1.0));
    if (any(isnan(result)) || any(isinf(result))) {
        result = cur;
    }

    vec2 momentPrev = texture(u_moment, prev_uv).rg;
    float resultLuma = svgf_luma(result);

    float newMomentLuma = mix(curLuma, momentPrev.r, blend);
    float newMomentLumaSq = mix(curLuma * curLuma, momentPrev.g, blend);

    out_color = vec4(result, 1.0);
    out_moment = vec2(newMomentLuma, newMomentLumaSq);
}
"""

SVGF_ATROUS_FRAG = """
#version 460 core
uniform sampler2D u_color;
uniform sampler2D u_depth;
uniform sampler2D u_moment;
uniform vec2 u_pixel;
uniform float u_step_size;
uniform float u_phi_depth;
uniform float u_phi_normal;
uniform float u_phi_luma;
in vec2 v_uv;
out vec4 frag_color;

float svgf_luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 svgf_normal_from_depth(vec2 uv, float d0) {
    vec2 texel = vec2(1.0) / max(vec2(textureSize(u_depth, 0)), vec2(1.0));
    float dx = texture(u_depth, uv + vec2(texel.x, 0.0)).r - texture(u_depth, uv - vec2(texel.x, 0.0)).r;
    float dy = texture(u_depth, uv + vec2(0.0, texel.y)).r - texture(u_depth, uv - vec2(0.0, texel.y)).r;
    vec3 n = normalize(vec3(-dx, -dy, 0.02));
    return n;
}

void main() {
    vec3 c_center = texture(u_color, v_uv).rgb;
    float d_center = texture(u_depth, v_uv).r;
    float luma_center = svgf_luma(c_center);

    vec2 moment_center = texture(u_moment, v_uv).rg;
    float variance = max(moment_center.g - moment_center.r * moment_center.r, 0.0);

    vec3 n_center = svgf_normal_from_depth(v_uv, d_center);

    float sigma_luma = u_phi_luma * sqrt(variance + 1e-6);
    float w_total = 1.0;
    vec3 result = c_center;

    float kernel[5];
    kernel[0] = 1.0 / 16.0;
    kernel[1] = 4.0 / 16.0;
    kernel[2] = 6.0 / 16.0;
    kernel[3] = 4.0 / 16.0;
    kernel[4] = 1.0 / 16.0;

    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            if (x == 0 && y == 0) continue;
            vec2 ouv = v_uv + vec2(float(x), float(y)) * u_pixel * u_step_size;
            ouv = clamp(ouv, vec2(0.0), vec2(1.0));

            vec3 c_sample = texture(u_color, ouv).rgb;
            float d_sample = texture(u_depth, ouv).r;
            float luma_sample = svgf_luma(c_sample);

            float w_depth = exp(-abs(d_center - d_sample) / (u_phi_depth * abs(d_center) + 1e-4));
            float w_luma = exp(-abs(luma_center - luma_sample) / (sigma_luma + 1e-4));

            vec3 n_sample = svgf_normal_from_depth(ouv, d_sample);
            float w_normal = pow(max(dot(n_center, n_sample), 0.0), u_phi_normal);

            int ix = x + 2;
            int iy = y + 2;
            float w_kernel = kernel[ix] * kernel[iy];
            float w = w_kernel * w_depth * w_luma * w_normal;

            result += w * c_sample;
            w_total += w;
        }
    }

    frag_color = vec4(result / w_total, luma_center);
}
"""

SVGF_FINAL_FRAG = """
#version 460 core
uniform sampler2D u_input;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    frag_color = texture(u_input, v_uv);
}
"""


@ComponentRegistry.register
class SpatiotemporalVarianceGuidedFilter(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "SVGF"
    render_type = "screen"
    _use_velocity = True

    def __init__(self):
        super().__init__()
        self._stability: float = 0.9
        self._disocclusion: float = 0.08
        self._phi_depth: float = 1.0
        self._phi_normal: float = 128.0
        self._phi_luma: float = 4.0
        self._iterations: int = 5
        self._ctx: Optional[moderngl.Context] = None
        self._reproj_prog: Optional[moderngl.Program] = None
        self._atrous_prog: Optional[moderngl.Program] = None
        self._final_prog: Optional[moderngl.Program] = None
        self._vao_reproj: Optional[moderngl.VertexArray] = None
        self._vao_atrous: Optional[moderngl.VertexArray] = None
        self._vao_final: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None
        self._history: list = [None, None]
        self._history_fbo: list = [None, None]
        self._moment: list = [None, None]
        self._moment_fbo: list = [None, None]
        self._atrous_buf: list = [None, None]
        self._atrous_fbo: list = [None, None]
        self._buf_w: int = 0
        self._buf_h: int = 0
        self._cur_index: int = 0
        self._history_valid: bool = False
        self._frame_count: int = 0

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_stability", "Temporal Stability", FieldType.FLOAT, min_val=0.0, max_val=0.99, step=0.01, decimals=3),
            InspectorField("_disocclusion", "Disocclusion Depth", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.005, decimals=3),
            InspectorField("_phi_depth", "Phi Depth", FieldType.FLOAT, min_val=0.01, max_val=10.0, step=0.05, decimals=3),
            InspectorField("_phi_normal", "Phi Normal", FieldType.FLOAT, min_val=1.0, max_val=512.0, step=1.0, decimals=1),
            InspectorField("_phi_luma", "Phi Luminance", FieldType.FLOAT, min_val=0.1, max_val=16.0, step=0.1, decimals=2),
            InspectorField("_iterations", "Iterations", FieldType.INT, min_val=1, max_val=8, step=1),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_stability": self._stability,
            "_disocclusion": self._disocclusion,
            "_phi_depth": self._phi_depth,
            "_phi_normal": self._phi_normal,
            "_phi_luma": self._phi_luma,
            "_iterations": self._iterations,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> SpatiotemporalVarianceGuidedFilter:
        inst = super().deserialize(data)
        inst._stability = float(data.get("_stability", 0.9))
        inst._disocclusion = float(data.get("_disocclusion", 0.08))
        inst._phi_depth = float(data.get("_phi_depth", 1.0))
        inst._phi_normal = float(data.get("_phi_normal", 128.0))
        inst._phi_luma = float(data.get("_phi_luma", 4.0))
        inst._iterations = int(data.get("_iterations", 5))
        inst._ctx = None
        inst._reproj_prog = None
        inst._atrous_prog = None
        inst._final_prog = None
        inst._vao_reproj = None
        inst._vao_atrous = None
        inst._vao_final = None
        inst._vbo = None
        inst._ibo = None
        inst._history = [None, None]
        inst._history_fbo = [None, None]
        inst._moment = [None, None]
        inst._moment_fbo = [None, None]
        inst._atrous_buf = [None, None]
        inst._atrous_fbo = [None, None]
        inst._buf_w = 0
        inst._buf_h = 0
        inst._cur_index = 0
        inst._history_valid = False
        inst._frame_count = 0
        return inst

    _res_cache: dict[int, dict] = {}

    def _ensure_resources(self, ctx: moderngl.Context):
        ctx_id = id(ctx)
        cached = self._res_cache.get(ctx_id)
        if cached is not None:
            self._ctx = ctx
            self._reproj_prog = cached['_reproj_prog']
            self._atrous_prog = cached['_atrous_prog']
            self._final_prog = cached['_final_prog']
            self._vao_reproj = cached['_vao_reproj']
            self._vao_atrous = cached['_vao_atrous']
            self._vao_final = cached['_vao_final']
            self._vbo = cached['_vbo']
            self._ibo = cached['_ibo']
            return
        self._ctx = ctx
        self._reproj_prog = ctx.program(vertex_shader=SVGF_VERT, fragment_shader=SVGF_REPROJ_FRAG)
        self._atrous_prog = ctx.program(vertex_shader=SVGF_VERT, fragment_shader=SVGF_ATROUS_FRAG)
        self._final_prog = ctx.program(vertex_shader=SVGF_VERT, fragment_shader=SVGF_FINAL_FRAG)
        verts = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        self._vbo = ctx.buffer(verts.tobytes())
        self._ibo = ctx.buffer(indices.tobytes())
        self._vao_reproj = ctx.vertex_array(self._reproj_prog, [(self._vbo, '2f', 'in_position')], self._ibo)
        self._vao_atrous = ctx.vertex_array(self._atrous_prog, [(self._vbo, '2f', 'in_position')], self._ibo)
        self._vao_final = ctx.vertex_array(self._final_prog, [(self._vbo, '2f', 'in_position')], self._ibo)
        self._res_cache[ctx_id] = {
            '_reproj_prog': self._reproj_prog,
            '_atrous_prog': self._atrous_prog,
            '_final_prog': self._final_prog,
            '_vao_reproj': self._vao_reproj,
            '_vao_atrous': self._vao_atrous,
            '_vao_final': self._vao_final,
            '_vbo': self._vbo,
            '_ibo': self._ibo,
        }
        if len(self._res_cache) > 4:
            oldest = next(iter(self._res_cache))
            self._release_cache_objects(self._res_cache[oldest])
            del self._res_cache[oldest]

    @staticmethod
    def _release_cache_objects(entry: dict):
        for obj in entry.values():
            if obj is None:
                continue
            try:
                obj.release()
            except Exception:
                pass

    def _ensure_buffers(self, ctx: moderngl.Context, w: int, h: int):
        if self._buf_w == w and self._buf_h == h and self._history[0] is not None:
            return
        self._release_buffers()
        self._buf_w = w
        self._buf_h = h
        for i in range(2):
            htex = ctx.texture((w, h), 4, dtype='f2')
            htex.repeat_x = False
            htex.repeat_y = False
            self._history[i] = htex
            mtex = ctx.texture((w, h), 2, dtype='f2')
            mtex.repeat_x = False
            mtex.repeat_y = False
            self._moment[i] = mtex
            self._history_fbo[i] = ctx.framebuffer(color_attachments=[htex, mtex])
            self._moment_fbo[i] = ctx.framebuffer(mtex)
            atex = ctx.texture((w, h), 4, dtype='f2')
            atex.repeat_x = False
            atex.repeat_y = False
            self._atrous_buf[i] = atex
            self._atrous_fbo[i] = ctx.framebuffer(atex)
        self._history_valid = False
        self._cur_index = 0

    def _release_buffers(self):
        for i in range(2):
            for obj in (self._history[i], self._history_fbo[i],
                        self._moment[i], self._moment_fbo[i],
                        self._atrous_buf[i], self._atrous_fbo[i]):
                if obj is not None:
                    try:
                        obj.release()
                    except Exception:
                        pass
        self._history = [None, None]
        self._history_fbo = [None, None]
        self._moment = [None, None]
        self._moment_fbo = [None, None]
        self._atrous_buf = [None, None]
        self._atrous_fbo = [None, None]
        self._buf_w = 0
        self._buf_h = 0
        self._history_valid = False

    def render(self, ctx, scene_color_tex, scene_depth_tex,
               view_mat, proj_mat, cam_pos, viewport_w, viewport_h,
               input_tex=None, output_fbo=None, velocity_tex=None,
               prev_view_proj=None, **kwargs):
        if not self.enabled or not self.entity or not self.entity.active:
            return

        self._ensure_resources(ctx)
        self._ensure_buffers(ctx, viewport_w, viewport_h)

        cur = self._cur_index
        prev = 1 - cur

        ctx.disable(moderngl.BLEND)

        current_tex = input_tex if input_tex is not None else scene_color_tex

        self._history_fbo[cur].use()
        self._history_fbo[cur].viewport = (0, 0, viewport_w, viewport_h)
        self._reproj_prog["u_current"] = 0
        current_tex.use(0)
        self._reproj_prog["u_depth"] = 1
        scene_depth_tex.use(1)
        if velocity_tex is not None and "u_velocity" in self._reproj_prog:
            self._reproj_prog["u_velocity"] = 2
            velocity_tex.use(2)
            self._reproj_prog["u_has_velocity"] = 1.0
        else:
            self._reproj_prog["u_has_velocity"] = 0.0
        self._reproj_prog["u_history"] = 4
        self._history[prev].use(4)
        self._reproj_prog["u_moment"] = 5
        self._moment[prev].use(5)
        self._reproj_prog["u_pixel"].value = (1.0 / viewport_w, 1.0 / viewport_h)
        self._reproj_prog["u_disocclusion"].value = self._disocclusion
        self._reproj_prog["u_stability"].value = self._stability
        self._reproj_prog["u_has_history"] = 1.0 if self._history_valid else 0.0
        self._vao_reproj.render()
        ctx.memory_barrier(moderngl.ALL_BARRIER_BITS)

        atrous_src = self._history[cur]
        atrous_src_fbo = self._history_fbo[cur]

        iters = max(1, min(self._iterations, 8))
        for i in range(iters):
            atrous_dst_idx = i % 2
            atrous_dst_fbo = self._atrous_fbo[atrous_dst_idx]
            atrous_dst_fbo.use()
            atrous_dst_fbo.viewport = (0, 0, viewport_w, viewport_h)
            self._atrous_prog["u_color"] = 0
            atrous_src.use(0)
            self._atrous_prog["u_depth"] = 1
            scene_depth_tex.use(1)
            self._atrous_prog["u_moment"] = 5
            self._moment[cur].use(5)
            self._atrous_prog["u_pixel"].value = (1.0 / viewport_w, 1.0 / viewport_h)
            self._atrous_prog["u_step_size"] = float(1 << i)
            self._atrous_prog["u_phi_depth"].value = self._phi_depth
            self._atrous_prog["u_phi_normal"].value = self._phi_normal
            self._atrous_prog["u_phi_luma"].value = self._phi_luma
            self._vao_atrous.render()
            atrous_src = self._atrous_buf[atrous_dst_idx]
            atrous_src_fbo = atrous_dst_fbo

        if output_fbo is not None:
            output_fbo.use()
            output_fbo.viewport = (0, 0, viewport_w, viewport_h)
        self._final_prog["u_input"] = 0
        atrous_src.use(0)
        self._vao_final.render()

        self._cur_index = prev
        self._history_valid = True
        self._frame_count += 1

    def on_disable(self):
        super().on_disable()
        self._history_valid = False

    def _release_gl(self):
        self._release_buffers()
        for obj in (self._reproj_prog, self._atrous_prog, self._final_prog,
                    self._vao_reproj, self._vao_atrous, self._vao_final,
                    self._vbo, self._ibo):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._reproj_prog = None
        self._atrous_prog = None
        self._final_prog = None
        self._vao_reproj = None
        self._vao_atrous = None
        self._vao_final = None
        self._vbo = None
        self._ibo = None

    @property
    def stability(self) -> float:
        return getattr(self, '_stability', 0.9)

    @stability.setter
    def stability(self, v: float):
        self._stability = v

    @property
    def disocclusion(self) -> float:
        return getattr(self, '_disocclusion', 0.08)

    @disocclusion.setter
    def disocclusion(self, v: float):
        self._disocclusion = v

    @property
    def phi_depth(self) -> float:
        return getattr(self, '_phi_depth', 1.0)

    @phi_depth.setter
    def phi_depth(self, v: float):
        self._phi_depth = v

    @property
    def phi_normal(self) -> float:
        return getattr(self, '_phi_normal', 128.0)

    @phi_normal.setter
    def phi_normal(self, v: float):
        self._phi_normal = v

    @property
    def phi_luma(self) -> float:
        return getattr(self, '_phi_luma', 4.0)

    @phi_luma.setter
    def phi_luma(self, v: float):
        self._phi_luma = v

    @property
    def iterations(self) -> int:
        return getattr(self, '_iterations', 5)

    @iterations.setter
    def iterations(self, v: int):
        self._iterations = v
