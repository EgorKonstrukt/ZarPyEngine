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
from core.components.lighting.light import Light, LightType
from core.components.lighting.projector import Projector
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Vec3


GOD_RAYS_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

GOD_RAYS_FRAG = """
#version 460 core
uniform sampler2D u_scene_color;
uniform sampler2D u_depth_tex;
uniform vec2 u_light_uv;
uniform vec3 u_light_color;
uniform float u_light_intensity;
uniform float u_intensity;
uniform float u_exposure;
uniform float u_threshold;
uniform int u_num_samples;
in vec2 v_uv;
out vec4 frag_color;

bool uv_valid(vec2 uv) {
    return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

void main() {
    if (!uv_valid(u_light_uv)) {
        frag_color = vec4(0.0);
        return;
    }

    // Adaptive threshold: keep only content clearly brighter than the sun's
    // own disc luminance, so the ordinary sky never re-adds itself (which
    // washed the whole screen white) regardless of exposure/sun intensity.
    vec3 sun_col = texture(u_scene_color, u_light_uv).rgb;
    float sun_lum = dot(sun_col, vec3(0.299, 0.587, 0.114));
    float threshold = max(sun_lum * u_threshold, 0.15);

    vec2 dir = u_light_uv - v_uv;
    float dist = length(dir);
    if (dist < 0.0005) {
        frag_color = vec4(0.0);
        return;
    }

    float d0 = texture(u_depth_tex, v_uv).r;
    vec2 delta = dir / float(max(u_num_samples, 1));

    // Per-pixel hash: shift the sample grid by up to half a step so a low
    // sample count reads as soft noise instead of square ray polygons.
    float rnd = fract(sin(dot(floor(v_uv * vec2(1024.0)), vec2(12.9898, 78.233))) * 43758.5453);
    vec2 sample_uv = v_uv + normalize(dir) * ((rnd - 0.5) * length(delta));

    float decay = 0.97;
    float weight = 1.0;
    vec3 accum = vec3(0.0);

    for (int i = 0; i < 128; i++) {
        if (i >= u_num_samples) break;
        sample_uv += delta;
        if (!uv_valid(sample_uv)) break;
        vec2 remaining = u_light_uv - sample_uv;
        if (dot(remaining, delta) < 0.0) break;

        // Geometry clearly closer to the camera than the start surface blocks
        // light: the ray stops there, so rays are occluded by buildings/clouds.
        float ds = texture(u_depth_tex, sample_uv).r;
        float closer = max(d0 - ds, 0.0);
        float occ = 1.0 - smoothstep(0.0, 0.003, closer);
        if (occ < 0.02) break;

        vec3 col = texture(u_scene_color, sample_uv).rgb;
        float lum = dot(col, vec3(0.299, 0.587, 0.114));
        float excess = max(lum - threshold, 0.0);
        if (excess > 0.0) {
            // Soft-knee the brightness so the HDR sun disc / glow stays
            // bounded (no white-out) while retaining a radial falloff.
            float bright = 3.0 * (1.0 - exp(-excess / 3.0));
            // Use the sampled hue only; col * bright would square the HDR
            // luminance and blow the screen out near the sun.
            vec3 chroma = col / max(lum, 1e-4);
            accum += chroma * bright * weight * u_exposure * occ;
        }
        weight *= decay;
    }

    vec3 tint = u_light_color * u_light_intensity;
    frag_color = vec4(max(vec3(0.0), min(accum * u_intensity * tint, vec3(1.0))), 1.0);
}
"""

GODRAYS_COPY_FRAG = """
#version 460 core
uniform sampler2D u_scene_color;
uniform sampler2D u_depth_tex;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    frag_color = vec4(texture(u_scene_color, v_uv).rgb, 1.0);
    // A hair below the source depth so every fragment passes the default
    // GL_LESS test against the cleared 1.0; gl_FragDepth is only written when
    // the depth test is enabled, so color and depth land in one pass.
    gl_FragDepth = max(texture(u_depth_tex, v_uv).r - 1e-5, 0.0);
}
"""


@ComponentRegistry.register
class GodRays(GraphicsEffect):
    _allow_multiple = False
    _gizmo_icon_label = "вЂ"
    _intensity_prop = "_intensity"

    def __init__(self):
        super().__init__()
        self._intensity: float = 0.8
        self._exposure: float = 0.01
        self._num_samples: int = 64
        self._threshold: float = 0.25
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None
        self._copy_prog: Optional[moderngl.Program] = None
        self._copy_vao: Optional[moderngl.VertexArray] = None
        self._scratch_color: Optional[moderngl.Texture] = None
        self._scratch_depth: Optional[moderngl.Texture] = None
        self._scratch_fbo: Optional[moderngl.Framebuffer] = None
        self._scratch_size: Optional[tuple[int, int]] = None
        self._scratch_ctx: Optional[moderngl.Context] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.1, decimals=2),
            InspectorField("_exposure", "Exposure", FieldType.FLOAT, min_val=0.0, max_val=0.5, step=0.001, decimals=4),
            InspectorField("_threshold", "Threshold", FieldType.SLIDER, min_val=0.05, max_val=0.5, step=0.01, decimals=2),
            InspectorField("_num_samples", "Samples", FieldType.INT, min_val=8, max_val=256, step=8),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_intensity": self._intensity,
            "_exposure": self._exposure,
            "_num_samples": self._num_samples,
            "_threshold": self._threshold,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> GodRays:
        inst = super().deserialize(data)
        inst._intensity = float(data.get("_intensity", 0.8))
        inst._exposure = float(data.get("_exposure", 0.01))
        inst._num_samples = int(data.get("_num_samples", 64))
        inst._threshold = float(data.get("_threshold", 0.25))
        inst._prog = None
        inst._vao = None
        inst._vbo = None
        inst._ibo = None
        inst._copy_prog = None
        inst._copy_vao = None
        inst._scratch_color = None
        inst._scratch_depth = None
        inst._scratch_fbo = None
        inst._scratch_size = None
        inst._scratch_ctx = None
        return inst

    _res_cache: dict[int, dict] = {}

    def _ensure_resources(self, ctx: moderngl.Context):
        ctx_id = id(ctx)
        cached = self._res_cache.get(ctx_id)
        if cached is not None and cached.get('_prog') is not None:
            self._ctx = ctx
            self._prog = cached['_prog']
            self._vao = cached['_vao']
            self._vbo = cached['_vbo']
            self._ibo = cached['_ibo']
            self._copy_prog = cached['_copy_prog']
            self._copy_vao = cached['_copy_vao']
            return
        self._ctx = ctx
        self._prog = ctx.program(
            vertex_shader=GOD_RAYS_VERT,
            fragment_shader=GOD_RAYS_FRAG
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
        self._copy_prog = ctx.program(
            vertex_shader=GOD_RAYS_VERT,
            fragment_shader=GODRAYS_COPY_FRAG
        )
        self._copy_vao = ctx.vertex_array(
            self._copy_prog,
            [(self._vbo, '2f', 'in_position')],
            self._ibo
        )
        self._res_cache[ctx_id] = {
            '_prog': self._prog,
            '_vao': self._vao,
            '_vbo': self._vbo,
            '_ibo': self._ibo,
            '_copy_prog': self._copy_prog,
            '_copy_vao': self._copy_vao,
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

    def _get_light_source(self):
        """Find the light source associated with this entity.
        Returns (component, transform) or (None, None)."""
        if not self.entity:
            return None, None
        light = self.entity.get_component(Light)
        if light:
            return light, self.transform
        proj = self.entity.get_component(Projector)
        if proj:
            return proj, self.transform
        if self.entity._scene:
            for ent in self.entity._scene.get_entities_with_component(Light):
                li = ent.get_component(Light)
                tr = ent.transform
                if li and tr:
                    return li, tr
            for ent in self.entity._scene.get_entities_with_component(Projector):
                pj = ent.get_component(Projector)
                tr = ent.transform
                if pj and tr:
                    return pj, tr
        return None, None

    def _project_pos_to_uv(self, pos: Vec3, view_mat, proj_mat) -> Optional[np.ndarray]:
        v = np.array([pos.x, pos.y, pos.z, 1.0], dtype=np.float32)
        vm = view_mat._d.astype(np.float32)
        pm = proj_mat._d.astype(np.float32)
        v_view = (vm.T) @ v
        v_clip = (pm.T) @ v_view
        if v_clip[3] <= 0:
            return None
        ndc = v_clip[:2] / v_clip[3]
        uv = ndc * 0.5 + 0.5
        if uv[0] < -0.1 or uv[0] > 1.1 or uv[1] < -0.1 or uv[1] > 1.1:
            return None
        return uv

    def _project_dir_to_uv(
        self, light_fwd: Vec3, light_pos: Vec3, view_mat, proj_mat
    ) -> Optional[np.ndarray]:
        sun_dir = Vec3(-light_fwd.x, -light_fwd.y, -light_fwd.z).normalized()
        sun_pos = Vec3(
            light_pos.x + sun_dir.x * 5000.0,
            light_pos.y + sun_dir.y * 5000.0,
            light_pos.z + sun_dir.z * 5000.0
        )
        return self._project_pos_to_uv(sun_pos, view_mat, proj_mat)

    def _compute_light_info(self, view_mat, proj_mat):
        """Returns (uv, color_r, color_g, color_b, intensity) or None."""
        source, transform = self._get_light_source()
        if source is None or transform is None:
            return None
        if isinstance(source, Light):
            if source.light_type == LightType.DIRECTIONAL:
                uv = self._project_dir_to_uv(
                    transform.forward, transform.position, view_mat, proj_mat
                )
            else:
                uv = self._project_pos_to_uv(transform.position, view_mat, proj_mat)
        elif isinstance(source, Projector):
            uv = self._project_pos_to_uv(transform.position, view_mat, proj_mat)
        else:
            return None
        if uv is None:
            return None
        if isinstance(source, Light):
            color, intensity = Light.shader_radiance(source, transform)
        elif isinstance(source, Projector):
            color = source.color
            intensity = float(source.intensity)
        else:
            color = [1.0, 1.0, 1.0]
            intensity = 1.0
        return (uv, float(color[0]), float(color[1]), float(color[2]), float(intensity))

    def _ensure_scratch(self, ctx: moderngl.Context, w: int, h: int):
        if self._scratch_fbo is not None and self._scratch_size == (w, h) and self._scratch_ctx is ctx:
            return
        self._release_scratch()
        self._scratch_color = ctx.texture((w, h), 4, dtype='f4')
        self._scratch_color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._scratch_depth = ctx.depth_texture((w, h))
        self._scratch_fbo = ctx.framebuffer(
            color_attachments=[self._scratch_color],
            depth_attachment=self._scratch_depth,
        )
        self._scratch_size = (w, h)
        self._scratch_ctx = ctx

    def render(self, ctx, scene_color_tex, scene_depth_tex,
               view_mat, proj_mat, cam_pos, viewport_w, viewport_h, **kwargs):
        if not self.enabled or not self.entity or not self.entity.active:
            return
        self._ensure_resources(ctx)
        info = self._compute_light_info(view_mat, proj_mat)
        if info is None:
            return
        light_uv, cr, cg, cb, lintensity = info
        # Sampling a texture that is also an attachment of the FBO we draw
        # into is undefined behaviour on the GPU and renders as 4x4/8x8 tile
        # artefacts (the renderer draws effects onto the scene FBO, whose
        # colour/depth are exactly the textures this effect reads). So first
        # copy the scene colour and depth into a private scratch FBO -- there
        # nothing sampled is attached -- then run the ray march against the
        # copies while drawing into the renderer's bound target.
        target = ctx.detect_framebuffer()
        self._ensure_scratch(ctx, viewport_w, viewport_h)
        self._scratch_fbo.use()
        self._scratch_fbo.viewport = (0, 0, viewport_w, viewport_h)
        ctx.viewport = (0, 0, viewport_w, viewport_h)
        ctx.clear(0.0, 0.0, 0.0, 0.0, 1.0)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)
        self._copy_prog["u_scene_color"] = 0
        scene_color_tex.use(0)
        self._copy_prog["u_depth_tex"] = 1
        scene_depth_tex.use(1)
        self._copy_vao.render()
        ctx.disable(moderngl.DEPTH_TEST)

        target.use()
        target.viewport = (0, 0, viewport_w, viewport_h)
        ctx.viewport = (0, 0, viewport_w, viewport_h)
        self._prog["u_scene_color"] = 0
        self._scratch_color.use(0)
        self._prog["u_depth_tex"] = 1
        self._scratch_depth.use(1)
        if "u_light_uv" in self._prog:
            self._prog["u_light_uv"].value = (float(light_uv[0]), float(light_uv[1]))
        if "u_light_color" in self._prog:
            self._prog["u_light_color"].value = (cr, cg, cb)
        if "u_light_intensity" in self._prog:
            self._prog["u_light_intensity"].value = lintensity
        if "u_intensity" in self._prog:
            self._prog["u_intensity"].value = self._intensity
        if "u_exposure" in self._prog:
            self._prog["u_exposure"].value = self._exposure
        if "u_threshold" in self._prog:
            self._prog["u_threshold"].value = self._threshold
        if "u_num_samples" in self._prog:
            self._prog["u_num_samples"].value = self._num_samples
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ctx.enable(moderngl.BLEND)
        self._vao.render()
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def _release_scratch(self):
        for obj in (self._scratch_color, self._scratch_depth, self._scratch_fbo):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._scratch_color = None
        self._scratch_depth = None
        self._scratch_fbo = None
        self._scratch_size = None
        self._scratch_ctx = None

    def _release_gl(self):
        for obj in (self._prog, self._vao, self._vbo, self._ibo,
                    self._copy_prog, self._copy_vao):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._release_scratch()
        self._ctx = None
        self._prog = None
        self._vao = None
        self._vbo = None
        self._ibo = None
        self._copy_prog = None
        self._copy_vao = None

    @property
    def intensity(self) -> float:
        return getattr(self, '_intensity', 0.8)

    @intensity.setter
    def intensity(self, v: float):
        self._intensity = v

    @property
    def exposure(self) -> float:
        return getattr(self, '_exposure', 0.01)

    @exposure.setter
    def exposure(self, v: float):
        self._exposure = v

    @property
    def num_samples(self) -> int:
        return getattr(self, '_num_samples', 64)

    @num_samples.setter
    def num_samples(self, v: int):
        self._num_samples = v