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


SSR_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""


SSR_FRAG = """
#version 460 core
uniform sampler2D u_depth_tex;
uniform sampler2D u_input_tex;
uniform mat4 u_proj;
uniform mat4 u_inv_proj;
uniform vec2 u_pixel_size;
uniform int u_steps;
uniform float u_thickness;
uniform float u_max_distance;
uniform float u_edge_fade;
uniform float u_distance_fade;
uniform float u_fresnel_power;
uniform float u_fresnel_strength;

in vec2 v_uv;
out vec4 frag_ssr;

vec3 view_pos(vec2 uv, float depth) {
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 v = u_inv_proj * ndc;
    return v.xyz / v.w;
}

vec3 view_normal(vec2 uv) {
    float d = texture(u_depth_tex, uv).r;
    vec3 c = view_pos(uv, d);
    vec2 off = u_pixel_size;
    vec3 r = view_pos(uv + vec2(off.x, 0.0), texture(u_depth_tex, uv + vec2(off.x, 0.0)).r);
    vec3 u = view_pos(uv + vec2(0.0, off.y), texture(u_depth_tex, uv + vec2(0.0, off.y)).r);
    return normalize(cross(r - c, u - c));
}

void main() {
    float depth = texture(u_depth_tex, v_uv).r;
    if (depth >= 1.0) {
        frag_ssr = vec4(0.0);
        return;
    }
    vec3 pos = view_pos(v_uv, depth);
    vec3 normal = view_normal(v_uv);
    vec3 V = normalize(pos);
    if (dot(normal, V) < 0.0) normal = -normal;
    vec3 R = reflect(V, normal);

    float start_t = 0.02;
    float base_step = 0.15;
    float t = start_t;
    bool hit = false;
    float tHit = 0.0;
    float last_step = 0.0;
    for (int i = 0; i < 64; i++) {
        if (i >= u_steps) break;
        float step = base_step * pow(1.12, float(i));
        if (t + step > u_max_distance) {
            step = u_max_distance - t;
            if (step <= 0.0) break;
        }
        t += step;
        last_step = step;
        vec3 p = pos + R * t;
        vec4 clip = u_proj * vec4(p, 1.0);
        if (clip.w <= 0.001) break;
        vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
        if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) break;
        float sceneD = -view_pos(uv, texture(u_depth_tex, uv).r).z;
        float rayD = -p.z;
        if (rayD > sceneD && rayD - sceneD < u_thickness) {
            hit = true;
            tHit = t;
            break;
        }
    }
    if (!hit) {
        frag_ssr = vec4(0.0);
        return;
    }
    float lo = max(tHit - last_step, 0.01);
    float hi = tHit;
    vec2 hitUV = vec2(0.5);
    for (int j = 0; j < 8; j++) {
        float mid = (lo + hi) * 0.5;
        vec3 pm = pos + R * mid;
        vec4 cm = u_proj * vec4(pm, 1.0);
        vec2 um = cm.xy / cm.w * 0.5 + 0.5;
        float sceneD = -view_pos(um, texture(u_depth_tex, um).r).z;
        float rayD = -pm.z;
        hitUV = um;
        if (rayD > sceneD) hi = mid; else lo = mid;
    }
    vec2 edge = min(hitUV, 1.0 - hitUV);
    float edgeFade = smoothstep(0.0, u_edge_fade, min(edge.x, edge.y));
    float distFade = 1.0 - smoothstep(u_max_distance * u_distance_fade, u_max_distance, tHit);
    float fresnel = u_fresnel_strength + (1.0 - u_fresnel_strength) * pow(1.0 - max(dot(normal, V), 0.0), u_fresnel_power);
    float alpha = edgeFade * distFade * fresnel;
    vec3 refl = texture(u_input_tex, hitUV).rgb;
    frag_ssr = vec4(refl, alpha);
}
"""


COMP_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
uniform sampler2D u_ssr_tex;
uniform float u_intensity;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 scene_color = texture(u_input_tex, v_uv).rgb;
    vec4 ssr = texture(u_ssr_tex, v_uv);
    vec3 col = scene_color + ssr.rgb * ssr.a * u_intensity;
    frag_color = vec4(col, 1.0);
}
"""


@ComponentRegistry.register
class SSR(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "SSR"
    render_type = "screen"
    _skip_rate = 0
    _intensity_prop = "_intensity"

    def __init__(self):
        super().__init__()
        self._intensity: float = 0.5
        self._steps: int = 32
        self._thickness: float = 1.5
        self._max_distance: float = 60.0
        self._edge_fade: float = 0.12
        self._distance_fade: float = 0.6
        self._fresnel_power: float = 4.0
        self._fresnel_strength: float = 0.05
        self._performance_scale: float = 0.5
        self._ctx: Optional[moderngl.Context] = None
        self._prog_ssr: Optional[moderngl.Program] = None
        self._prog_comp: Optional[moderngl.Program] = None
        self._vao_ssr: Optional[moderngl.VertexArray] = None
        self._vao_comp: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None
        self._temp_fbo: Optional[moderngl.Framebuffer] = None
        self._fbo_size: tuple[int, int] = (0, 0)

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_header_main", "Screen Space Reflections", FieldType.HEADER),
            InspectorField("_intensity", "Intensity", FieldType.SLIDER, min_val=0.0, max_val=2.0, step=0.05, decimals=3),
            InspectorField("_steps", "Steps", FieldType.INT_SLIDER, min_val=8, max_val=64, step=1),
            InspectorField("_thickness", "Thickness", FieldType.SLIDER, min_val=0.05, max_val=4.0, step=0.05, decimals=2),
            InspectorField("_max_distance", "Max Distance", FieldType.SLIDER, min_val=10.0, max_val=200.0, step=1.0, decimals=0),
            InspectorField("_edge_fade", "Edge Fade", FieldType.SLIDER, min_val=0.02, max_val=0.3, step=0.01, decimals=3),
            InspectorField("_distance_fade", "Distance Fade", FieldType.SLIDER, min_val=0.3, max_val=0.95, step=0.05, decimals=2),
            InspectorField("_fresnel_power", "Fresnel Power", FieldType.SLIDER, min_val=1.0, max_val=8.0, step=0.1, decimals=1),
            InspectorField("_fresnel_strength", "Fresnel Strength", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.05, decimals=2),
            InspectorField("_header_quality", "Quality", FieldType.HEADER),
            InspectorField("_performance_scale", "Resolution Scale", FieldType.SLIDER, min_val=0.25, max_val=1.0, step=0.25, decimals=2),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_intensity": self._intensity,
            "_steps": self._steps,
            "_thickness": self._thickness,
            "_max_distance": self._max_distance,
            "_edge_fade": self._edge_fade,
            "_distance_fade": self._distance_fade,
            "_fresnel_power": self._fresnel_power,
            "_fresnel_strength": self._fresnel_strength,
            "_performance_scale": self._performance_scale,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> SSR:
        inst = super().deserialize(data)
        inst._intensity = float(data.get("_intensity", 0.5))
        inst._steps = int(data.get("_steps", 24))
        inst._thickness = float(data.get("_thickness", 1.5))
        inst._max_distance = float(data.get("_max_distance", 60.0))
        inst._edge_fade = float(data.get("_edge_fade", 0.12))
        inst._distance_fade = float(data.get("_distance_fade", 0.6))
        inst._fresnel_power = float(data.get("_fresnel_power", 4.0))
        inst._fresnel_strength = float(data.get("_fresnel_strength", 0.05))
        inst._performance_scale = float(data.get("_performance_scale", 0.5))
        inst._ctx = None
        inst._prog_ssr = None
        inst._prog_comp = None
        inst._vao_ssr = None
        inst._vao_comp = None
        inst._vbo = None
        inst._ibo = None
        inst._temp_fbo = None
        inst._fbo_size = (0, 0)
        return inst

    _res_cache: dict[int, dict] = {}

    def _ensure_resources(self, ctx: moderngl.Context):
        ctx_id = id(ctx)
        old_cache = self._res_cache.get(ctx_id)
        if old_cache is not None:
            self._ctx = ctx
            self._prog_ssr = old_cache['_prog_ssr']
            self._prog_comp = old_cache['_prog_comp']
            self._vao_ssr = old_cache['_vao_ssr']
            self._vao_comp = old_cache['_vao_comp']
            self._vbo = old_cache['_vbo']
            self._ibo = old_cache['_ibo']
            return

        self._ctx = ctx
        self._prog_ssr = ctx.program(
            vertex_shader=SSR_VERT,
            fragment_shader=SSR_FRAG
        )
        self._prog_comp = ctx.program(
            vertex_shader=SSR_VERT,
            fragment_shader=COMP_FRAG
        )
        verts = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        self._vbo = ctx.buffer(verts.tobytes())
        self._ibo = ctx.buffer(indices.tobytes())
        self._vao_ssr = ctx.vertex_array(
            self._prog_ssr,
            [(self._vbo, '2f', 'in_position')],
            self._ibo
        )
        self._vao_comp = ctx.vertex_array(
            self._prog_comp,
            [(self._vbo, '2f', 'in_position')],
            self._ibo
        )
        self._res_cache[ctx_id] = {
            '_prog_ssr': self._prog_ssr,
            '_prog_comp': self._prog_comp,
            '_vao_ssr': self._vao_ssr,
            '_vao_comp': self._vao_comp,
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
                color_attachments=[ctx.texture((w, h), 4, dtype='f4')]
            )
            self._fbo_size = (w, h)

    def render(self, ctx, scene_color_tex, scene_depth_tex,
               view_mat, proj_mat, cam_pos, viewport_w, viewport_h,
               input_tex=None, output_fbo=None, **kwargs):
        if not self.enabled or not self.entity or not self.entity.active:
            return
        if output_fbo is None:
            return
        scale = max(0.25, min(1.0, self._performance_scale))
        sw = max(1, int(viewport_w * scale))
        sh = max(1, int(viewport_h * scale))

        self._ensure_resources(ctx)
        self._ensure_temp_fbo(ctx, sw, sh)

        tex = input_tex if input_tex is not None else scene_color_tex

        # Pass 1: SSR ray march -> scaled temp FBO
        self._temp_fbo.use()
        self._temp_fbo.viewport = (0, 0, sw, sh)
        self._temp_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._prog_ssr["u_depth_tex"] = 0
        scene_depth_tex.use(0)
        self._prog_ssr["u_input_tex"] = 1
        tex.use(1)
        self._prog_ssr["u_proj"].write(proj_mat.to_f32().tobytes())
        self._prog_ssr["u_inv_proj"].write(proj_mat.inverted().to_f32().tobytes())
        self._prog_ssr["u_pixel_size"].value = (1.0 / sw, 1.0 / sh)
        self._prog_ssr["u_steps"].value = self._steps
        self._prog_ssr["u_thickness"].value = self._thickness
        self._prog_ssr["u_max_distance"].value = self._max_distance
        self._prog_ssr["u_edge_fade"].value = self._edge_fade
        self._prog_ssr["u_distance_fade"].value = self._distance_fade
        self._prog_ssr["u_fresnel_power"].value = self._fresnel_power
        self._prog_ssr["u_fresnel_strength"].value = self._fresnel_strength
        ctx.disable(moderngl.BLEND)
        self._vao_ssr.render()

        # Pass 2: Composite -> full-res output_fbo
        output_fbo.use()
        output_fbo.viewport = (0, 0, viewport_w, viewport_h)
        self._prog_comp["u_input_tex"] = 0
        tex.use(0)
        self._prog_comp["u_ssr_tex"] = 1
        self._temp_fbo.color_attachments[0].use(1)
        self._prog_comp["u_intensity"].value = self._intensity
        self._vao_comp.render()

    def _release_gl(self):
        for obj in (self._prog_ssr, self._prog_comp, self._vao_ssr, self._vao_comp, self._vbo, self._ibo):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._ctx = None
        self._prog_ssr = None
        self._prog_comp = None
        self._vao_ssr = None
        self._vao_comp = None
        self._vbo = None
        self._ibo = None
        if self._temp_fbo is not None:
            try:
                self._temp_fbo.release()
            except Exception:
                pass
            self._temp_fbo = None
        self._fbo_size = (0, 0)
