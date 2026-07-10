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


FSR2_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

EASU_FRAG = """
#version 460 core
uniform sampler2D u_src;
uniform vec2 u_src_size;
uniform vec2 u_src_pixel;
uniform float u_sharpness;
in vec2 v_uv;
out vec4 frag_color;

vec3 fsr2_catmull_rom(sampler2D tex, vec2 uv, vec2 texSize, vec2 texelSize) {
    vec2 samplePos = uv * texSize;
    vec2 texPos1 = floor(samplePos - 0.5) + 0.5;
    vec2 f = samplePos - texPos1;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / max(w12, vec2(1e-5));
    vec2 texPos0 = (texPos1 - 1.0) * texelSize;
    vec2 texPos3 = (texPos1 + 2.0) * texelSize;
    vec2 texPos12 = (texPos1 + offset12) * texelSize;
    vec3 result = vec3(0.0);
    result += texture(tex, vec2(texPos0.x, texPos0.y)).rgb * w0.x * w0.y;
    result += texture(tex, vec2(texPos12.x, texPos0.y)).rgb * w12.x * w0.y;
    result += texture(tex, vec2(texPos3.x, texPos0.y)).rgb * w3.x * w0.y;
    result += texture(tex, vec2(texPos0.x, texPos12.y)).rgb * w0.x * w12.y;
    result += texture(tex, vec2(texPos12.x, texPos12.y)).rgb * w12.x * w12.y;
    result += texture(tex, vec2(texPos3.x, texPos12.y)).rgb * w3.x * w12.y;
    result += texture(tex, vec2(texPos0.x, texPos3.y)).rgb * w0.x * w3.y;
    result += texture(tex, vec2(texPos12.x, texPos3.y)).rgb * w12.x * w3.y;
    result += texture(tex, vec2(texPos3.x, texPos3.y)).rgb * w3.x * w3.y;
    return max(result, vec3(0.0));
}

float fsr2_luma(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
    vec3 base = fsr2_catmull_rom(u_src, v_uv, u_src_size, u_src_pixel);

    vec3 c00 = texture(u_src, v_uv + vec2(-1.0, -1.0) * u_src_pixel).rgb;
    vec3 c10 = texture(u_src, v_uv + vec2(0.0, -1.0) * u_src_pixel).rgb;
    vec3 c20 = texture(u_src, v_uv + vec2(1.0, -1.0) * u_src_pixel).rgb;
    vec3 c01 = texture(u_src, v_uv + vec2(-1.0, 0.0) * u_src_pixel).rgb;
    vec3 c11 = texture(u_src, v_uv).rgb;
    vec3 c21 = texture(u_src, v_uv + vec2(1.0, 0.0) * u_src_pixel).rgb;
    vec3 c02 = texture(u_src, v_uv + vec2(-1.0, 1.0) * u_src_pixel).rgb;
    vec3 c12 = texture(u_src, v_uv + vec2(0.0, 1.0) * u_src_pixel).rgb;
    vec3 c22 = texture(u_src, v_uv + vec2(1.0, 1.0) * u_src_pixel).rgb;

    float l00 = fsr2_luma(c00), l10 = fsr2_luma(c10), l20 = fsr2_luma(c20);
    float l01 = fsr2_luma(c01), l11 = fsr2_luma(c11), l21 = fsr2_luma(c21);
    float l02 = fsr2_luma(c02), l12 = fsr2_luma(c12), l22 = fsr2_luma(c22);

    float gx = (l20 + 2.0 * l21 + l22) - (l00 + 2.0 * l01 + l02);
    float gy = (l02 + 2.0 * l12 + l22) - (l00 + 2.0 * l10 + l20);
    float edge = clamp(length(vec2(gx, gy)), 0.0, 1.0);

    vec3 blurAvg = (c00 + c10 + c20 + c01 + c11 + c21 + c02 + c12 + c22) / 9.0;
    vec3 mn = min(min(min(c00, c10), min(c20, c01)), min(min(c11, c21), min(c02, min(c12, c22))));
    vec3 mx = max(max(max(c00, c10), max(c20, c01)), max(max(c11, c21), max(c02, max(c12, c22))));

    vec3 sharpened = base + (base - blurAvg) * (edge * u_sharpness * 2.0);
    sharpened = clamp(sharpened, mn, mx);

    frag_color = vec4(mix(base, sharpened, edge), 1.0);
}
"""

TEMPORAL_FRAG = """
#version 460 core
uniform sampler2D u_current;
uniform sampler2D u_depth;
uniform sampler2D u_velocity;
uniform sampler2D u_history;
uniform sampler2D u_prev_depth;
uniform vec2 u_pixel;
uniform vec2 u_size;
uniform float u_stability;
uniform float u_disocclusion;
uniform float u_variance_gamma;
uniform float u_has_velocity;
uniform int u_has_history;
uniform int u_has_reproj;
uniform mat4 u_inv_view_proj;
uniform mat4 u_prev_view_proj;
in vec2 v_uv;
out vec4 frag_color;

vec3 fsr2_rgb_to_ycocg(vec3 c) {
    return vec3(
        dot(c, vec3(0.25, 0.5, 0.25)),
        dot(c, vec3(0.5, 0.0, -0.5)),
        dot(c, vec3(-0.25, 0.5, -0.25))
    );
}

vec3 fsr2_ycocg_to_rgb(vec3 c) {
    float y = c.x;
    float co = c.y;
    float cg = c.z;
    return vec3(y + co - cg, y + cg, y - co - cg);
}

vec3 fsr2_clip_to_box(vec3 history, vec3 center, vec3 extents) {
    vec3 v = history - center;
    vec3 unit = v / max(extents, vec3(1e-5));
    float ma = max(unit.x, max(unit.y, unit.z));
    float mi = min(unit.x, min(unit.y, unit.z));
    float m = max(ma, -mi);
    return m > 1.0 ? center + v / m : history;
}

vec3 fsr2_history_catmull_rom(sampler2D tex, vec2 uv, vec2 texSize) {
    vec2 samplePos = uv * texSize;
    vec2 texPos1 = floor(samplePos - 0.5) + 0.5;
    vec2 f = samplePos - texPos1;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / max(w12, vec2(1e-5));
    vec2 texPos0 = (texPos1 - 1.0) / texSize;
    vec2 texPos3 = (texPos1 + 2.0) / texSize;
    vec2 texPos12 = (texPos1 + offset12) / texSize;
    vec3 result = vec3(0.0);
    result += texture(tex, vec2(texPos0.x, texPos0.y)).rgb * w0.x * w0.y;
    result += texture(tex, vec2(texPos12.x, texPos0.y)).rgb * w12.x * w0.y;
    result += texture(tex, vec2(texPos3.x, texPos0.y)).rgb * w3.x * w0.y;
    result += texture(tex, vec2(texPos0.x, texPos12.y)).rgb * w0.x * w12.y;
    result += texture(tex, vec2(texPos12.x, texPos12.y)).rgb * w12.x * w12.y;
    result += texture(tex, vec2(texPos3.x, texPos12.y)).rgb * w3.x * w12.y;
    result += texture(tex, vec2(texPos0.x, texPos3.y)).rgb * w0.x * w3.y;
    result += texture(tex, vec2(texPos12.x, texPos3.y)).rgb * w12.x * w3.y;
    result += texture(tex, vec2(texPos3.x, texPos3.y)).rgb * w3.x * w3.y;
    return max(result, vec3(0.0));
}

void main() {
    vec3 cur = texture(u_current, v_uv).rgb;
    float curD = texture(u_depth, v_uv).r;

    if (u_has_history == 0) {
        frag_color = vec4(cur, 1.0);
        return;
    }

    vec2 motion = u_has_velocity > 0.5 ? texture(u_velocity, v_uv).rg : vec2(0.0);
    if (u_has_reproj == 1 && dot(motion, motion) < 1e-10) {
        vec4 clipPos = vec4(v_uv * 2.0 - 1.0, curD * 2.0 - 1.0, 1.0);
        vec4 worldPos = u_inv_view_proj * clipPos;
        if (abs(worldPos.w) > 1e-8) {
            worldPos /= worldPos.w;
            vec4 prevClip = u_prev_view_proj * worldPos;
            if (abs(prevClip.w) > 1e-6) {
                vec2 prevNdc = prevClip.xy / prevClip.w;
                motion = v_uv - (prevNdc * 0.5 + 0.5);
            }
        }
    }
    vec2 prev_uv = v_uv - motion;

    vec3 mn = cur;
    vec3 mx = cur;
    vec3 ySum = fsr2_rgb_to_ycocg(cur);
    vec3 ySqSum = ySum * ySum;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            if (x == 0 && y == 0) continue;
            vec3 c = texture(u_current, v_uv + vec2(float(x), float(y)) * u_pixel).rgb;
            mn = min(mn, c);
            mx = max(mx, c);
            vec3 yc = fsr2_rgb_to_ycocg(c);
            ySum += yc;
            ySqSum += yc * yc;
        }
    }
    vec3 yMean = ySum / 9.0;
    vec3 yVar = max(ySqSum / 9.0 - yMean * yMean, vec3(0.0));
    vec3 yStd = sqrt(yVar);
    vec3 yExtents = max(yStd * u_variance_gamma, vec3(1e-4));

    vec3 hist = fsr2_history_catmull_rom(u_history, prev_uv, u_size);
    vec3 histY = fsr2_rgb_to_ycocg(hist);
    vec3 histYClipped = fsr2_clip_to_box(histY, yMean, yExtents);
    vec3 histClipped = clamp(fsr2_ycocg_to_rgb(histYClipped), mn, mx);

    float clipDist = length(histY - histYClipped) / max(length(yStd), 1e-4);

    float blend = u_stability;
    if (any(lessThan(prev_uv, vec2(0.0))) || any(greaterThan(prev_uv, vec2(1.0)))) {
        blend = 0.0;
    } else {
        float prevD = texture(u_prev_depth, prev_uv).r;
        float ddepth = abs(curD - prevD);
        float diso = smoothstep(0.0, max(u_disocclusion, 1e-4), ddepth);
        blend = mix(u_stability, 0.0, diso);
        float velPx = length(motion) / max(u_pixel.x, u_pixel.y);
        blend = mix(blend, blend * 0.5, clamp(velPx / 200.0, 0.0, 1.0));
        blend = mix(blend, blend * 0.35, clamp(clipDist * 0.5, 0.0, 1.0));
    }

    vec3 result = mix(cur, histClipped, clamp(blend, 0.0, 1.0));
    frag_color = vec4(result, 1.0);
}
"""

COPY_DEPTH_FRAG = """
#version 460 core
uniform sampler2D u_depth;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    frag_color = vec4(texture(u_depth, v_uv).r, 0.0, 0.0, 1.0);
}
"""

RCAS_FRAG = """
#version 460 core
uniform sampler2D u_rcas_input;
uniform vec2 u_rcas_size;
uniform float u_rcas_con;
in vec2 v_uv;
out vec4 frag_color;

vec3 load(ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(u_rcas_size) - ivec2(1));
    return texelFetch(u_rcas_input, p, 0).rgb;
}

const float FSR_RCAS_LIMIT = (0.25 - (1.0 / 16.0));

void main() {
    ivec2 ip = ivec2(gl_FragCoord.xy);
    vec3 b = load(ip + ivec2(0, -1));
    vec3 d = load(ip + ivec2(-1, 0));
    vec3 e = load(ip);
    vec3 f = load(ip + ivec2(1, 0));
    vec3 h = load(ip + ivec2(0, 1));

    float bR = b.r, bG = b.g, bB = b.b;
    float dR = d.r, dG = d.g, dB = d.b;
    float eR = e.r, eG = e.g, eB = e.b;
    float fR = f.r, fG = f.g, fB = f.b;
    float hR = h.r, hG = h.g, hB = h.b;

    float bL = bB * 0.5 + (bR * 0.5 + bG);
    float dL = dB * 0.5 + (dR * 0.5 + dG);
    float eL = eB * 0.5 + (eR * 0.5 + eG);
    float fL = fB * 0.5 + (fR * 0.5 + fG);
    float hL = hB * 0.5 + (hR * 0.5 + hG);

    float nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
    nz = clamp(abs(nz) / max(max(max(max(bL, dL), eL), fL) - min(min(min(min(bL, dL), eL), fL), hL), 1e-6), 0.0, 1.0);
    nz = -0.5 * nz + 1.0;

    float mn4R = min(min(min(bR, dR), fR), hR);
    float mn4G = min(min(min(bG, dG), fG), hG);
    float mn4B = min(min(min(bB, dB), fB), hB);
    float mx4R = max(max(max(bR, dR), fR), hR);
    float mx4G = max(max(max(bG, dG), fG), hG);
    float mx4B = max(max(max(bB, dB), fB), hB);

    float hitMinR = mn4R / max(4.0 * mx4R, 1e-4);
    float hitMinG = mn4G / max(4.0 * mx4G, 1e-4);
    float hitMinB = mn4B / max(4.0 * mx4B, 1e-4);
    float hitMaxR = (1.0 - mx4R) / max(4.0 * mn4R - 4.0, -3.999999);
    float hitMaxG = (1.0 - mx4G) / max(4.0 * mn4G - 4.0, -3.999999);
    float hitMaxB = (1.0 - mx4B) / max(4.0 * mn4B - 4.0, -3.999999);

    float lobeR = max(-hitMinR, hitMaxR);
    float lobeG = max(-hitMinG, hitMaxG);
    float lobeB = max(-hitMinB, hitMaxB);
    float lobe = max(-FSR_RCAS_LIMIT, min(max(lobeR, max(lobeG, lobeB)), 0.0)) * u_rcas_con;
    lobe *= nz;

    float rcpL = 1.0 / (4.0 * lobe + 1.0);
    vec3 pix = (lobe * (b + d + h + f) + e) * rcpL;
    frag_color = vec4(pix, 1.0);
}
"""


@ComponentRegistry.register
class FidelityFXSuperResolution2(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "FS2"
    render_type = "screen"
    _use_velocity = True
    _is_upscaler = True

    def __init__(self):
        super().__init__()
        self._stability: float = 0.9
        self._rcas_sharpness: float = 0.2
        self._disocclusion: float = 0.15
        self._variance_gamma: float = 1.25
        self._easu_sharpness: float = 0.35
        self._ctx: Optional[moderngl.Context] = None
        self._easu_prog: Optional[moderngl.Program] = None
        self._temporal_prog: Optional[moderngl.Program] = None
        self._copy_depth_prog: Optional[moderngl.Program] = None
        self._rcas_prog: Optional[moderngl.Program] = None
        self._vao_easu: Optional[moderngl.VertexArray] = None
        self._vao_temporal: Optional[moderngl.VertexArray] = None
        self._vao_copy_depth: Optional[moderngl.VertexArray] = None
        self._vao_rcas: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None
        self._easu_tex: Optional[moderngl.Texture] = None
        self._easu_fbo: Optional[moderngl.Framebuffer] = None
        self._recon: list = [None, None]
        self._recon_fbo: list = [None, None]
        self._prev_depth: Optional[moderngl.Texture] = None
        self._prev_depth_fbo: Optional[moderngl.Framebuffer] = None
        self._buf_w: int = 0
        self._buf_h: int = 0
        self._cur_index: int = 0
        self._history_valid: bool = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_stability", "Temporal Stability", FieldType.FLOAT, min_val=0.0, max_val=0.99, step=0.01, decimals=3),
            InspectorField("_rcas_sharpness", "RCAS Attenuation", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("_disocclusion", "Disocclusion Depth", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.005, decimals=3),
            InspectorField("_variance_gamma", "Variance Clip Gamma", FieldType.FLOAT, min_val=0.5, max_val=4.0, step=0.05, decimals=2),
            InspectorField("_easu_sharpness", "EASU Edge Sharpness", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_stability": self._stability,
            "_rcas_sharpness": self._rcas_sharpness,
            "_disocclusion": self._disocclusion,
            "_variance_gamma": self._variance_gamma,
            "_easu_sharpness": self._easu_sharpness,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> FidelityFXSuperResolution2:
        inst = super().deserialize(data)
        inst._stability = float(data.get("_stability", 0.9))
        inst._rcas_sharpness = float(data.get("_rcas_sharpness", 0.2))
        inst._disocclusion = float(data.get("_disocclusion", 0.15))
        inst._variance_gamma = float(data.get("_variance_gamma", 1.25))
        inst._easu_sharpness = float(data.get("_easu_sharpness", 0.35))
        inst._ctx = None
        inst._easu_prog = None
        inst._temporal_prog = None
        inst._copy_depth_prog = None
        inst._rcas_prog = None
        inst._vao_easu = None
        inst._vao_temporal = None
        inst._vao_copy_depth = None
        inst._vao_rcas = None
        inst._vbo = None
        inst._ibo = None
        inst._easu_tex = None
        inst._easu_fbo = None
        inst._recon = [None, None]
        inst._recon_fbo = [None, None]
        inst._prev_depth = None
        inst._prev_depth_fbo = None
        inst._buf_w = 0
        inst._buf_h = 0
        inst._cur_index = 0
        inst._history_valid = False
        return inst

    _res_cache: dict[int, dict] = {}

    def _ensure_resources(self, ctx: moderngl.Context):
        ctx_id = id(ctx)
        cached = self._res_cache.get(ctx_id)
        if cached is not None:
            self._ctx = ctx
            self._easu_prog = cached['_easu_prog']
            self._temporal_prog = cached['_temporal_prog']
            self._copy_depth_prog = cached['_copy_depth_prog']
            self._rcas_prog = cached['_rcas_prog']
            self._vao_easu = cached['_vao_easu']
            self._vao_temporal = cached['_vao_temporal']
            self._vao_copy_depth = cached['_vao_copy_depth']
            self._vao_rcas = cached['_vao_rcas']
            self._vbo = cached['_vbo']
            self._ibo = cached['_ibo']
            return
        self._ctx = ctx
        self._easu_prog = ctx.program(vertex_shader=FSR2_VERT, fragment_shader=EASU_FRAG)
        self._temporal_prog = ctx.program(vertex_shader=FSR2_VERT, fragment_shader=TEMPORAL_FRAG)
        self._copy_depth_prog = ctx.program(vertex_shader=FSR2_VERT, fragment_shader=COPY_DEPTH_FRAG)
        self._rcas_prog = ctx.program(vertex_shader=FSR2_VERT, fragment_shader=RCAS_FRAG)
        verts = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        self._vbo = ctx.buffer(verts.tobytes())
        self._ibo = ctx.buffer(indices.tobytes())
        self._vao_easu = ctx.vertex_array(self._easu_prog, [(self._vbo, '2f', 'in_position')], self._ibo)
        self._vao_temporal = ctx.vertex_array(self._temporal_prog, [(self._vbo, '2f', 'in_position')], self._ibo)
        self._vao_copy_depth = ctx.vertex_array(self._copy_depth_prog, [(self._vbo, '2f', 'in_position')], self._ibo)
        self._vao_rcas = ctx.vertex_array(self._rcas_prog, [(self._vbo, '2f', 'in_position')], self._ibo)
        self._res_cache[ctx_id] = {
            '_easu_prog': self._easu_prog,
            '_temporal_prog': self._temporal_prog,
            '_copy_depth_prog': self._copy_depth_prog,
            '_rcas_prog': self._rcas_prog,
            '_vao_easu': self._vao_easu,
            '_vao_temporal': self._vao_temporal,
            '_vao_copy_depth': self._vao_copy_depth,
            '_vao_rcas': self._vao_rcas,
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
        if self._buf_w == w and self._buf_h == h and self._recon[0] is not None:
            return
        self._release_buffers()
        self._buf_w = w
        self._buf_h = h
        for i in range(2):
            tex = ctx.texture((w, h), 4, dtype='f2')
            tex.repeat_x = False
            tex.repeat_y = False
            self._recon[i] = tex
            self._recon_fbo[i] = ctx.framebuffer(tex)
        self._easu_tex = ctx.texture((w, h), 4, dtype='f2')
        self._easu_tex.repeat_x = False
        self._easu_tex.repeat_y = False
        self._easu_fbo = ctx.framebuffer(self._easu_tex)
        self._prev_depth = ctx.texture((w, h), 4, dtype='f2')
        self._prev_depth.repeat_x = False
        self._prev_depth.repeat_y = False
        self._prev_depth_fbo = ctx.framebuffer(self._prev_depth)
        self._history_valid = False
        self._cur_index = 0

    def _release_buffers(self):
        for obj in (self._recon[0], self._recon[1], self._recon_fbo[0], self._recon_fbo[1],
                    self._easu_tex, self._easu_fbo, self._prev_depth, self._prev_depth_fbo):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._recon = [None, None]
        self._recon_fbo = [None, None]
        self._easu_tex = None
        self._easu_fbo = None
        self._prev_depth = None
        self._prev_depth_fbo = None
        self._buf_w = 0
        self._buf_h = 0
        self._history_valid = False

    def render(self, ctx, scene_color_tex, scene_depth_tex,
               view_mat, proj_mat, cam_pos, viewport_w, viewport_h,
               input_tex=None, output_fbo=None, velocity_tex=None,
               prev_view_proj=None, **kwargs):
        if not self.enabled or not self.entity or not self.entity.active:
            return
        render_w = int(kwargs.get('render_w') or viewport_w)
        render_h = int(kwargs.get('render_h') or viewport_h)
        self._ensure_resources(ctx)
        self._ensure_buffers(ctx, viewport_w, viewport_h)

        upscaling = (render_w != viewport_w) or (render_h != viewport_h)
        cur = self._cur_index
        prev = 1 - cur

        ctx.disable(moderngl.BLEND)

        if upscaling:
            self._easu_fbo.use()
            self._easu_fbo.viewport = (0, 0, viewport_w, viewport_h)
            self._easu_prog["u_src"] = 0
            scene_color_tex.use(0)
            self._easu_prog["u_src_size"].value = (float(render_w), float(render_h))
            self._easu_prog["u_src_pixel"].value = (1.0 / max(1, render_w), 1.0 / max(1, render_h))
            self._easu_prog["u_sharpness"].value = float(self._easu_sharpness)
            self._vao_easu.render()
            current_tex = self._easu_tex
        else:
            current_tex = input_tex if input_tex is not None else scene_color_tex

        self._recon_fbo[cur].use()
        self._recon_fbo[cur].viewport = (0, 0, viewport_w, viewport_h)
        self._temporal_prog["u_current"] = 0
        current_tex.use(0)
        self._temporal_prog["u_depth"] = 1
        scene_depth_tex.use(1)
        if velocity_tex is not None:
            self._temporal_prog["u_velocity"] = 2
            velocity_tex.use(2)
            self._temporal_prog["u_has_velocity"] = 1.0
        else:
            self._temporal_prog["u_has_velocity"] = 0.0
        self._temporal_prog["u_history"] = 3
        self._recon[prev].use(3)
        self._temporal_prog["u_prev_depth"] = 4
        self._prev_depth.use(4)
        self._temporal_prog["u_pixel"].value = (1.0 / viewport_w, 1.0 / viewport_h)
        self._temporal_prog["u_size"].value = (float(viewport_w), float(viewport_h))
        self._temporal_prog["u_stability"].value = self._stability
        self._temporal_prog["u_disocclusion"].value = self._disocclusion
        self._temporal_prog["u_variance_gamma"].value = self._variance_gamma
        self._temporal_prog["u_has_history"] = 1 if self._history_valid else 0

        has_reproj = prev_view_proj is not None
        if has_reproj:
            try:
                cur_forward = proj_mat @ view_mat
                inv_vp = cur_forward.inverted().to_f32()
                self._temporal_prog["u_inv_view_proj"].write(inv_vp.tobytes())
                self._temporal_prog["u_prev_view_proj"].write(prev_view_proj.to_f32().tobytes())
                self._temporal_prog["u_has_reproj"] = 1
            except Exception:
                self._temporal_prog["u_has_reproj"] = 0
        else:
            self._temporal_prog["u_has_reproj"] = 0

        self._vao_temporal.render()

        self._prev_depth_fbo.use()
        self._prev_depth_fbo.viewport = (0, 0, viewport_w, viewport_h)
        self._copy_depth_prog["u_depth"] = 0
        scene_depth_tex.use(0)
        self._vao_copy_depth.render()

        if output_fbo is not None:
            output_fbo.use()
            output_fbo.viewport = (0, 0, viewport_w, viewport_h)
        ratio = max(float(viewport_w) / max(1.0, float(render_w)), float(viewport_h) / max(1.0, float(render_h)))
        taper = 1.0 / max(1.0, 1.0 + (ratio - 1.0) * 0.5)
        self._rcas_prog["u_rcas_input"] = 0
        self._recon[cur].use(0)
        self._rcas_prog["u_rcas_size"].value = (float(viewport_w), float(viewport_h))
        self._rcas_prog["u_rcas_con"].value = float(2.0 ** (-(self._rcas_sharpness * taper)))
        self._vao_rcas.render()

        self._cur_index = prev
        self._history_valid = True

    def on_disable(self):
        super().on_disable()
        self._history_valid = False

    def _release_gl(self):
        self._release_buffers()
        for obj in (self._easu_prog, self._temporal_prog, self._copy_depth_prog, self._rcas_prog,
                    self._vao_easu, self._vao_temporal, self._vao_copy_depth, self._vao_rcas,
                    self._vbo, self._ibo):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._easu_prog = None
        self._temporal_prog = None
        self._copy_depth_prog = None
        self._rcas_prog = None
        self._vao_easu = None
        self._vao_temporal = None
        self._vao_copy_depth = None
        self._vao_rcas = None
        self._vbo = None
        self._ibo = None

    @property
    def stability(self) -> float:
        return getattr(self, '_stability', 0.9)

    @stability.setter
    def stability(self, v: float):
        self._stability = v

    @property
    def rcas_sharpness(self) -> float:
        return getattr(self, '_rcas_sharpness', 0.2)

    @rcas_sharpness.setter
    def rcas_sharpness(self, v: float):
        self._rcas_sharpness = v

    @property
    def disocclusion(self) -> float:
        return getattr(self, '_disocclusion', 0.15)

    @disocclusion.setter
    def disocclusion(self, v: float):
        self._disocclusion = v

    @property
    def variance_gamma(self) -> float:
        return getattr(self, '_variance_gamma', 1.25)

    @variance_gamma.setter
    def variance_gamma(self, v: float):
        self._variance_gamma = v

    @property
    def easu_sharpness(self) -> float:
        return getattr(self, '_easu_sharpness', 0.35)

    @easu_sharpness.setter
    def easu_sharpness(self, v: float):
        self._easu_sharpness = v