# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import numpy as np
import moderngl

_VERT = """#version 460
in vec2 in_position;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

_FRAG = """#version 460
uniform sampler2D u_wave;
uniform sampler2D u_spec;
uniform sampler2D u_spec_hold;
uniform vec4 u_panel;
uniform vec4 u_radar;
uniform float u_wave_n;
uniform float u_spec_n;
uniform float u_has_signal;
uniform float u_fh;
uniform float u_flip;
uniform float u_wave_gain;
out vec4 out_color;

void main() {
    vec2 p = vec2(gl_FragCoord.x, mix(gl_FragCoord.y, u_fh - gl_FragCoord.y, u_flip));
    vec3 col = vec3(0.0);
    float alpha = 0.0;
    if (u_radar.w > 0.5) {
        vec2 c = u_radar.xy;
        float R = u_radar.z;
        vec2 dv = p - c;
        float d = length(dv);
        if (d <= R) {
            col = vec3(0.04, 0.09, 0.11);
            alpha = 0.20;
        }
        if (d <= R && abs(d - R) < 1.5) {
            col = vec3(0.30, 0.62, 0.78);
            alpha = 0.75;
        }
        if (d <= R) {
            if (abs(dv.x) < 0.7) {
                col = vec3(0.16, 0.38, 0.48);
                alpha = 0.35;
            }
            if (abs(dv.y) < 0.7) {
                col = vec3(0.16, 0.38, 0.48);
                alpha = 0.35;
            }
            if (abs(abs(dv.x) - abs(dv.y)) < 0.7) {
                col = vec3(0.12, 0.30, 0.40);
                alpha = 0.25;
            }
        }
    }
    vec4 r = u_panel;
    if (p.x >= r.x && p.x <= r.x + r.z && p.y >= r.y && p.y <= r.y + r.w) {
        float bx = min(min(p.x - r.x, r.x + r.z - p.x), min(p.y - r.y, r.y + r.w - p.y));
        if (bx < 1.5) {
            col = vec3(0.32, 0.55, 0.78);
            alpha = 0.85;
        } else {
            col = vec3(0.05, 0.06, 0.07);
            alpha = 0.62;
            float gx = mod(p.x - r.x, r.z / 8.0);
            float gy = mod(p.y - r.y, r.w / 5.0);
            if (gx < 1.0 || gy < 1.0) {
                col = vec3(0.16, 0.19, 0.23);
                alpha = 0.35;
            }
            float spec_bottom = r.y + r.w * 0.52;
            if (p.y < spec_bottom) {
                float fx = clamp((p.x - r.x) / r.z, 0.0, 0.999);
                float ti = floor(fx * u_spec_n);
                float v = texture(u_spec, vec2((ti + 0.5) / u_spec_n, 0.5)).r;
                float bh = v * (spec_bottom - r.y - 4.0);
                float hline = spec_bottom - 2.0 - bh;
                if (p.y >= hline) {
                    float hue = ti / max(u_spec_n - 1.0, 1.0);
                    vec3 c0 = vec3(0.12, 0.95, 0.45);
                    vec3 c1 = vec3(1.0, 0.85, 0.2);
                    vec3 c2 = vec3(1.0, 0.2, 0.18);
                    col = mix(mix(c0, c1, smoothstep(0.0, 0.45, hue)), c2, smoothstep(0.45, 1.0, hue));
                    alpha = 0.92;
                }
                float hv = texture(u_spec_hold, vec2((ti + 0.5) / u_spec_n, 0.5)).r;
                float hh = spec_bottom - 2.0 - hv * (spec_bottom - r.y - 4.0);
                if (hv > 0.001 && abs(p.y - hh) < 1.0) {
                    col = vec3(1.0, 0.92, 0.55);
                    alpha = 0.9;
                }
            } else {
                float wave_top = spec_bottom;
                float wave_bottom = r.y + r.w;
                float xf = clamp((p.x - r.x) / r.z, 0.0, 1.0);
                float xfn = xf * (u_wave_n - 1.0);
                float i0 = floor(xfn);
                float f = xfn - i0;
                float i1 = min(i0 + 1.0, u_wave_n - 1.0);
                float y0 = texture(u_wave, vec2((i0 + 0.5) / u_wave_n, 0.5)).r;
                float y1 = texture(u_wave, vec2((i1 + 0.5) / u_wave_n, 0.5)).r;
                float yv = mix(y0, y1, f) * u_wave_gain;
                yv = clamp(yv, -1.0, 1.0);
                float mid = (wave_top + wave_bottom) * 0.5;
                float amp = (wave_bottom - wave_top) * 0.5 - 2.0;
                float py = mid - yv * amp;
                if (yv > 0.0 && p.y >= py && p.y < mid) {
                    col = vec3(0.12, 0.4, 0.55);
                    alpha = 0.16;
                }
                if (abs(py - p.y) < 1.1) {
                    col = vec3(0.4, 0.85, 1.0);
                    alpha = 0.95;
                }
                if (abs(p.y - mid) < 0.6) {
                    col = vec3(0.28, 0.33, 0.38);
                    alpha = 0.5;
                }
            }
        }
    }
    if (alpha <= 0.001) {
        out_color = vec4(0.0);
        return;
    }
    float sf = 0.35 + 0.65 * u_has_signal;
    out_color = vec4(col, alpha * sf);
}
"""

_QUAD_VERTS = np.array([
    -1.0, -1.0,
    1.0, -1.0,
    1.0, 1.0,
    -1.0, 1.0,
], dtype=np.float32)

_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)

_SCOPE_VERT = """#version 460
in vec2 a_scope;
uniform vec4 u_radar;
uniform float u_fw;
uniform float u_fh;
uniform float u_flip;
uniform float u_scale;
void main() {
    vec2 p = u_radar.xy + a_scope * u_radar.z * u_scale;
    float fy = mix(p.y, u_fh - p.y, u_flip);
    gl_Position = vec4(p.x / u_fw * 2.0 - 1.0, fy / u_fh * 2.0 - 1.0, 0.0, 1.0);
}
"""

_SCOPE_FRAG = """#version 460
uniform float u_has_signal;
uniform float u_alpha;
out vec4 out_color;
void main() {
    float sf = 0.35 + 0.65 * u_has_signal;
    out_color = vec4(0.25, 0.95, 0.6, u_alpha * sf);
}
"""


class AudioVizGL:
    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._prog = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        self._vbo = ctx.buffer(_QUAD_VERTS.tobytes())
        self._ibo = ctx.buffer(_QUAD_INDICES.tobytes())
        self._vao = ctx.vertex_array(self._prog, [(self._vbo, "2f", "in_position")], self._ibo)
        self._wave_tex = None
        self._spec_tex = None
        self._hold_tex = None
        self._flip = None
        self._error = None
        self._radar_rect = (0.0, 0.0, 0.0, 0.0)
        self._scope_vbo = None
        self._scope_vao = None
        self._scope_prog = ctx.program(vertex_shader=_SCOPE_VERT, fragment_shader=_SCOPE_FRAG)
        self._opts = {
            "scope_gain": 0.82,
            "wave_gain": None,
        }

    def set_options(self, **opts):
        for k, v in opts.items():
            if k in self._opts:
                self._opts[k] = v

    def ready(self) -> bool:
        return self._prog is not None

    def release(self):
        for t in (self._wave_tex, self._spec_tex, self._hold_tex):
            if t is not None:
                try:
                    t.release()
                except Exception:
                    pass
        self._wave_tex = None
        self._spec_tex = None
        self._hold_tex = None
        if self._scope_vao is not None:
            try:
                self._scope_vao.release()
            except Exception:
                pass
        if self._scope_vbo is not None:
            try:
                self._scope_vbo.release()
            except Exception:
                pass
        self._scope_vao = None
        self._scope_vbo = None
        try:
            self._scope_prog.release()
        except Exception:
            pass
        try:
            self._vao.release()
            self._vbo.release()
            self._ibo.release()
            self._prog.release()
        except Exception:
            pass

    def panel_rect(self, fw: int, fh: int):
        margin = 14
        w = min(int(fw * 0.42), 640)
        h = min(int(fh * 0.26), 210)
        if w < 120:
            w = 120
        if h < 90:
            h = 90
        if w > fw - margin * 2:
            w = max(120, fw - margin * 2)
        if h > fh - margin * 2:
            h = max(90, fh - margin * 2)
        x = margin
        y = fh - margin - h
        return (x, y, w, h)

    def _set_radar_uniforms(self, analyzer, fw: int, fh: int):
        R = min(int(min(fw, fh) * 0.18), 110)
        margin = 16
        cx = fw - margin - R
        cy = margin + R
        self._radar_rect = (float(cx), float(cy), float(R), 1.0)
        self._prog["u_radar"] = self._radar_rect

    def _ensure_texture(self, size: int):
        if self._wave_tex is None or self._wave_tex.width != size:
            if self._wave_tex is not None:
                try:
                    self._wave_tex.release()
                except Exception:
                    pass
            self._wave_tex = self._ctx.texture((size, 1), 1, dtype="f4")
            self._wave_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._wave_tex.repeat_x = False
        return self._wave_tex

    def render(self, analyzer, fw: int, fh: int, fbo=None, flip_y=None, now: float = 0.0):
        wave = analyzer.wave
        spec = analyzer.spec
        if wave is None or spec is None or wave.size == 0 or spec.size == 0:
            return
        self._ensure_texture(int(wave.size))
        if self._spec_tex is None or self._spec_tex.width != int(spec.size):
            if self._spec_tex is not None:
                try:
                    self._spec_tex.release()
                except Exception:
                    pass
            self._spec_tex = self._ctx.texture((int(spec.size), 1), 1, dtype="f4")
            self._spec_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._spec_tex.repeat_x = False
        self._wave_tex.write(np.ascontiguousarray(wave, dtype=np.float32).tobytes())
        self._spec_tex.write(np.ascontiguousarray(spec, dtype=np.float32).tobytes())
        if self._hold_tex is None or self._hold_tex.width != int(spec.size):
            if self._hold_tex is not None:
                try:
                    self._hold_tex.release()
                except Exception:
                    pass
            self._hold_tex = self._ctx.texture((int(spec.size), 1), 1, dtype="f4")
            self._hold_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._hold_tex.repeat_x = False
        hold = getattr(analyzer, "spec_hold", None)
        if hold is not None and hold.size == spec.size:
            self._hold_tex.write(np.ascontiguousarray(hold, dtype=np.float32).tobytes())
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._prog["u_wave"].value = 0
        self._prog["u_spec"].value = 1
        self._prog["u_spec_hold"].value = 2
        self._wave_tex.use(0)
        self._spec_tex.use(1)
        self._hold_tex.use(2)
        self._prog["u_panel"] = tuple(float(v) for v in self.panel_rect(fw, fh))
        self._set_radar_uniforms(analyzer, fw, fh)
        self._prog["u_wave_n"] = float(wave.size)
        self._prog["u_spec_n"] = float(spec.size)
        self._prog["u_has_signal"] = 1.0 if getattr(analyzer, "has_signal", False) else 0.0
        self._prog["u_fh"] = float(fh)
        peak = float(getattr(analyzer, "level", 0.0) or 0.0)
        wg = self._opts.get("wave_gain")
        if wg is None:
            wg = min(20.0, max(1.0, 0.8 / max(peak, 0.08)))
        self._prog["u_wave_gain"] = float(min(20.0, max(0.5, wg)))
        flip = self._flip
        if flip is None:
            if flip_y is not None:
                flip = 1.0 if flip_y else 0.0
            elif fbo is not None:
                flip = self._probe_flip(fbo, fw, fh)
            else:
                flip = 0.0
            self._flip = flip
        self._prog["u_flip"] = float(flip)
        self._vao.render()
        self._draw_scope(analyzer, fw, fh, flip)

    def _draw_scope(self, analyzer, fw: int, fh: int, flip: float):
        scope = getattr(analyzer, "scope", None)
        if scope is None or scope.size == 0:
            return
        pts = np.ascontiguousarray(scope, dtype=np.float32)
        if self._scope_vbo is None or self._scope_vbo.size != pts.nbytes:
            if self._scope_vbo is not None:
                try:
                    self._scope_vbo.release()
                except Exception:
                    pass
            self._scope_vbo = self._ctx.buffer(reserve=pts.nbytes, dynamic=True)
            self._scope_vao = self._ctx.vertex_array(
                self._scope_prog, [(self._scope_vbo, "2f", "a_scope")]
            )
        self._scope_vbo.write(pts.tobytes())
        self._scope_prog["u_radar"] = self._radar_rect
        self._scope_prog["u_fw"] = float(fw)
        self._scope_prog["u_fh"] = float(fh)
        self._scope_prog["u_flip"] = float(flip)
        self._scope_prog["u_scale"] = float(self._opts.get("scope_gain", 0.82))
        self._scope_prog["u_has_signal"] = 1.0 if getattr(analyzer, "has_signal", False) else 0.0
        self._scope_prog["u_alpha"] = 0.85
        try:
            self._ctx.line_width = 1.6
        except Exception:
            pass
        self._scope_vao.render(moderngl.LINE_STRIP)

    def _probe_flip(self, fbo, fw: int, fh: int) -> float:
        x, y, w, h = self.panel_rect(fw, fh)

        def read_px(px, py):
            try:
                buf = fbo.read(viewport=(int(px), int(py), 1, 1), components=4, dtype="f1")
                return tuple(np.frombuffer(buf, dtype=np.uint8))
            except Exception:
                return (0, 0, 0, 0)

        bottom_px = (int(x) + 2, int(y) + 2)
        top_px = (int(x) + 2, int(fh) - 1 - (int(y) + 2))
        before_bottom = read_px(*bottom_px)
        before_top = read_px(*top_px)
        self._prog["u_flip"] = 0.0
        self._vao.render()
        self._ctx.finish()
        if read_px(*bottom_px) != before_bottom:
            return 0.0
        if read_px(*top_px) != before_top:
            return 1.0
        return 0.0
