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
from core.components.lighting.light import Light, LightType
from core.components.lighting.projector import Projector
from core.components.inspector_meta import FieldType, InspectorField
from core.math3d import Vec3


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
uniform vec2 u_light_uv;
uniform vec3 u_light_color;
uniform float u_light_intensity;
uniform float u_intensity;
uniform float u_exposure;
uniform int u_num_samples;
uniform vec2 u_viewport_size;
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

    vec2 dir = u_light_uv - v_uv;
    float dist = length(dir);
    if (dist < 0.001) {
        frag_color = vec4(0.0);
        return;
    }
    vec2 delta = dir / float(max(u_num_samples, 1));
    vec2 sample_uv = v_uv;
    float decay = 0.97;
    float weight = 1.0;
    vec3 accum = vec3(0.0);

    for (int i = 0; i < 128; i++) {
        if (i >= u_num_samples) break;
        sample_uv += delta;
        if (!uv_valid(sample_uv)) break;
        vec2 remaining = u_light_uv - sample_uv;
        if (dot(remaining, delta) < 0.0) break;
        vec3 col = texture(u_scene_color, sample_uv).rgb;
        float lum = dot(col, vec3(0.299, 0.587, 0.114));
        float bright = max(0.0, lum - 0.2);
        accum += col * bright * weight * u_exposure;
        weight *= decay;
    }

    vec3 tint = u_light_color * u_light_intensity;
    frag_color = vec4(max(vec3(0.0), accum * u_intensity * tint), 1.0);
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
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.1, decimals=2),
            InspectorField("_exposure", "Exposure", FieldType.FLOAT, min_val=0.0, max_val=0.5, step=0.001, decimals=4),
            InspectorField("_num_samples", "Samples", FieldType.INT, min_val=8, max_val=256, step=8),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_intensity": self._intensity,
            "_exposure": self._exposure,
            "_num_samples": self._num_samples,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> GodRays:
        inst = super().deserialize(data)
        inst._intensity = float(data.get("_intensity", 0.8))
        inst._exposure = float(data.get("_exposure", 0.01))
        inst._num_samples = int(data.get("_num_samples", 64))
        inst._prog = None
        inst._vao = None
        inst._vbo = None
        inst._ibo = None
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
                tr = ent.get_component_by_name("Transform")
                if li and tr:
                    return li, tr
            for ent in self.entity._scene.get_entities_with_component(Projector):
                pj = ent.get_component(Projector)
                tr = ent.get_component_by_name("Transform")
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
            color = source.color
            intensity = source.intensity
        elif isinstance(source, Projector):
            color = source.color
            intensity = source.intensity
        else:
            color = [1.0, 1.0, 1.0]
            intensity = 1.0
        return (uv, float(color[0]), float(color[1]), float(color[2]), float(intensity))

    def render(self, ctx, scene_color_tex, scene_depth_tex,
               view_mat, proj_mat, cam_pos, viewport_w, viewport_h):
        if not self.enabled or not self.entity or not self.entity.active:
            return
        self._ensure_resources(ctx)
        info = self._compute_light_info(view_mat, proj_mat)
        if info is None:
            return
        light_uv, cr, cg, cb, lintensity = info
        self._prog["u_scene_color"] = 0
        scene_color_tex.use(0)
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
        if "u_num_samples" in self._prog:
            self._prog["u_num_samples"].value = self._num_samples
        if "u_viewport_size" in self._prog:
            self._prog["u_viewport_size"].value = (float(viewport_w), float(viewport_h))
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ctx.enable(moderngl.BLEND)
        self._vao.render()
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

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
