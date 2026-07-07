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


VOLUMETRIC_VERT = """
#version 460 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

VOLUMETRIC_FRAG = """
#version 460 core

uniform sampler2D u_depth_tex;
uniform mat4 u_inv_view_proj;
uniform vec3 u_camera_pos;
uniform float u_intensity;
uniform float u_density;
uniform float u_scattering;
uniform int u_step_count;

const int MAX_LIGHTS = 8;
const int MAX_PROJECTORS = 4;

uniform int   u_light_type[MAX_LIGHTS];
uniform vec3  u_light_pos[MAX_LIGHTS];
uniform vec3  u_light_dir[MAX_LIGHTS];
uniform vec3  u_light_col[MAX_LIGHTS];
uniform float u_light_int[MAX_LIGHTS];
uniform float u_light_range[MAX_LIGHTS];
uniform float u_light_spot_outer[MAX_LIGHTS];
uniform float u_light_spot_inner[MAX_LIGHTS];
uniform int   u_light_count;

uniform vec3  u_proj_pos[MAX_PROJECTORS];
uniform vec3  u_proj_dir[MAX_PROJECTORS];
uniform vec3  u_proj_col[MAX_PROJECTORS];
uniform float u_proj_int[MAX_PROJECTORS];
uniform float u_proj_range[MAX_PROJECTORS];
uniform float u_proj_angle[MAX_PROJECTORS];
uniform int   u_proj_count;

in vec2 v_uv;
out vec4 frag_color;

vec3 eval_light(vec3 pos, int i) {
    int tp = u_light_type[i];
    if (tp == 0) {
        return u_light_col[i] * u_light_int[i];
    }
    vec3 L = u_light_pos[i] - pos;
    float dist = length(L);
    if (dist > u_light_range[i]) return vec3(0.0);
    float atten = 1.0 / (1.0 + dist * dist * 0.01);
    if (tp == 1 || tp == 3) {
        return u_light_col[i] * u_light_int[i] * atten;
    }
    if (tp == 2) {
        vec3 dir = normalize(L);
        float half_rad = u_light_spot_outer[i] * 0.5;
        float cos_outer = cos(half_rad * 0.0174532925);
        float cos_ang = dot(dir, normalize(u_light_dir[i]));
        if (cos_ang < cos_outer) return vec3(0.0);
        float cos_inner = cos(u_light_spot_inner[i] * 0.0174532925);
        if (cos_inner > cos_outer) cos_inner = cos_outer;
        float spot = smoothstep(cos_outer, cos_inner, cos_ang);
        return u_light_col[i] * u_light_int[i] * atten * spot;
    }
    return vec3(0.0);
}

vec3 eval_projector(vec3 pos, int i) {
    vec3 L = pos - u_proj_pos[i];
    float dist = length(L);
    if (dist > u_proj_range[i]) return vec3(0.0);
    float half_rad = u_proj_angle[i] * 0.5;
    float cos_outer = cos(half_rad * 0.0174532925);
    vec3 Ln = normalize(-L);
    float cos_ang = dot(Ln, normalize(u_proj_dir[i]));
    if (cos_ang < cos_outer) return vec3(0.0);
    float atten = 1.0 / (1.0 + dist * dist * 0.01);
    float spot = max(0.0, cos_ang - cos_outer) / (1.0 - cos_outer);
    return u_proj_col[i] * u_proj_int[i] * atten * spot;
}

void main() {
    float depth = texture(u_depth_tex, v_uv).r;
    if (depth >= 1.0) {
        discard;
    }
    vec4 clip = vec4(v_uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 world = u_inv_view_proj * clip;
    vec3 surface_pos = world.xyz / world.w;
    vec3 view_dir = normalize(surface_pos - u_camera_pos);
    float march_dist = length(surface_pos - u_camera_pos);
    if (march_dist < 0.001) {
        discard;
    }
    float step_size = march_dist / float(max(u_step_count, 1));
    vec3 march_pos = u_camera_pos + view_dir * step_size * 0.5;
    vec3 accum = vec3(0.0);
    float transmittance = 1.0;

    for (int s = 0; s < u_step_count; s++) {
        float step_t = u_density * step_size;
        float step_trans = exp(-step_t);
        vec3 scatter = vec3(0.0);
        for (int li = 0; li < u_light_count; li++) {
            scatter += eval_light(march_pos, li);
        }
        for (int pj = 0; pj < u_proj_count; pj++) {
            scatter += eval_projector(march_pos, pj);
        }
        accum += scatter * u_scattering * (1.0 - step_trans) * transmittance;
        transmittance *= step_trans;
        march_pos += view_dir * step_size;
    }

    frag_color = vec4(accum * u_intensity, 1.0);
}
"""


@ComponentRegistry.register
class VolumetricLight(GraphicsEffect):
    _allow_multiple = True
    _gizmo_icon_label = "V"
    _intensity_prop = "_intensity"

    def __init__(self):
        super().__init__()
        self._intensity: float = 2.0
        self._density: float = 0.1
        self._scattering: float = 0.5
        self._step_count: int = 32
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("_intensity", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=10.0, step=0.1, decimals=2),
            InspectorField("_density", "Density", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.005, decimals=4),
            InspectorField("_scattering", "Scattering", FieldType.FLOAT, min_val=0.0, max_val=2.0, step=0.1, decimals=2),
            InspectorField("_step_count", "Steps", FieldType.INT, min_val=8, max_val=128, step=8),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_intensity": self._intensity,
            "_density": self._density,
            "_scattering": self._scattering,
            "_step_count": self._step_count,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> VolumetricLight:
        inst = super().deserialize(data)
        inst._intensity = float(data.get("_intensity", 2.0))
        inst._density = float(data.get("_density", 0.1))
        inst._scattering = float(data.get("_scattering", 0.5))
        inst._step_count = int(data.get("_step_count", 32))
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
            vertex_shader=VOLUMETRIC_VERT,
            fragment_shader=VOLUMETRIC_FRAG
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

    def _collect_lights(self) -> list:
        result = []
        if not self.entity or not self.entity._scene:
            return result
        scene = self.entity._scene
        for ent in scene.get_entities_with_component(Light):
            if not ent.active:
                continue
            lc = ent.get_component(Light)
            tr = ent.transform
            if lc and lc.enabled and tr:
                result.append((lc, tr))
        return result

    def _collect_projectors(self) -> list:
        result = []
        if not self.entity or not self.entity._scene:
            return result
        scene = self.entity._scene
        for ent in scene.get_entities_with_component(Projector):
            if not ent.active:
                continue
            pc = ent.get_component(Projector)
            tr = ent.transform
            if pc and pc.enabled and tr:
                result.append((pc, tr))
        return result

    def _set_light_uniforms(self, prog, lights):
        MAX_L = 8
        types = np.zeros(MAX_L, dtype=np.int32)
        positions = np.zeros((MAX_L, 3), dtype=np.float32)
        dirs = np.zeros((MAX_L, 3), dtype=np.float32)
        colors = np.zeros((MAX_L, 3), dtype=np.float32)
        intens = np.zeros(MAX_L, dtype=np.float32)
        ranges = np.zeros(MAX_L, dtype=np.float32)
        spot_outer = np.zeros(MAX_L, dtype=np.float32)
        spot_inner = np.zeros(MAX_L, dtype=np.float32)
        n = min(len(lights), MAX_L)
        for i in range(n):
            lc, lt = lights[i]
            pos = lt.position
            fwd = lt.forward
            if lc.light_type == LightType.DIRECTIONAL:
                types[i] = 0
            elif lc.light_type == LightType.POINT:
                types[i] = 1
            elif lc.light_type == LightType.SPOT:
                types[i] = 2
            else:
                types[i] = 3
            positions[i] = [pos.x, pos.y, pos.z]
            dirs[i] = [fwd.x, fwd.y, fwd.z]
            colors[i] = lc.color[:3]
            intens[i] = float(lc.intensity)
            ranges[i] = float(lc.range)
            spot_outer[i] = float(lc.spot_angle)
            spot_inner[i] = float(getattr(lc, 'spot_inner_angle', lc.spot_angle * 0.8))
        prog["u_light_type"].write(types.tobytes())
        prog["u_light_pos"].write(positions.tobytes())
        prog["u_light_dir"].write(dirs.tobytes())
        prog["u_light_col"].write(colors.tobytes())
        prog["u_light_int"].write(intens.tobytes())
        prog["u_light_range"].write(ranges.tobytes())
        prog["u_light_spot_outer"].write(spot_outer.tobytes())
        prog["u_light_spot_inner"].write(spot_inner.tobytes())
        prog["u_light_count"] = n

    def _set_projector_uniforms(self, prog, projectors):
        MAX_P = 4
        positions = np.zeros((MAX_P, 3), dtype=np.float32)
        dirs = np.zeros((MAX_P, 3), dtype=np.float32)
        colors = np.zeros((MAX_P, 3), dtype=np.float32)
        intens = np.zeros(MAX_P, dtype=np.float32)
        ranges = np.zeros(MAX_P, dtype=np.float32)
        angles = np.zeros(MAX_P, dtype=np.float32)
        n = min(len(projectors), MAX_P)
        for i in range(n):
            pc, pt = projectors[i]
            pos = pt.position
            fwd = pt.forward
            positions[i] = [pos.x, pos.y, pos.z]
            dirs[i] = [fwd.x, fwd.y, fwd.z]
            colors[i] = pc.color[:3]
            intens[i] = float(pc.intensity)
            ranges[i] = float(pc.range)
            angles[i] = float(pc.spot_angle)
        prog["u_proj_pos"].write(positions.tobytes())
        prog["u_proj_dir"].write(dirs.tobytes())
        prog["u_proj_col"].write(colors.tobytes())
        prog["u_proj_int"].write(intens.tobytes())
        prog["u_proj_range"].write(ranges.tobytes())
        prog["u_proj_angle"].write(angles.tobytes())
        prog["u_proj_count"] = n

    def render(self, ctx, scene_color_tex, scene_depth_tex,
               view_mat, proj_mat, cam_pos, viewport_w, viewport_h):
        if not self.enabled or not self.entity or not self.entity.active:
            return
        if self._intensity < 0.001 or self._density < 0.0001:
            return
        self._ensure_resources(ctx)
        prog = self._prog

        lights = self._collect_lights()
        projectors = self._collect_projectors()

        vp = proj_mat * view_mat
        inv_vp = vp.inverted()
        inv_vp_f32 = inv_vp.to_f32()

        try:
            prog["u_inv_view_proj"].write(inv_vp_f32.tobytes())
            prog["u_camera_pos"].write(np.array(
                [cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32
            ).tobytes())
            prog["u_intensity"] = self._intensity
            prog["u_density"] = self._density
            prog["u_scattering"] = self._scattering
            prog["u_step_count"] = self._step_count

            prog["u_depth_tex"] = 0
            scene_depth_tex.use(0)

            self._set_light_uniforms(prog, lights)
            self._set_projector_uniforms(prog, projectors)

            ctx.blend_func = moderngl.ONE, moderngl.ONE
            ctx.enable(moderngl.BLEND)
            self._vao.render()
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        except Exception as e:
            from core.logger import Logger
            Logger.error(f"VolumetricLight error: {e}")

    def _release_gl(self):
        self._res_cache.pop(id(self._ctx), None)
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
        return getattr(self, '_intensity', 1.0)

    @intensity.setter
    def intensity(self, v: float):
        self._intensity = v

    @property
    def density(self) -> float:
        return getattr(self, '_density', 0.1)

    @density.setter
    def density(self, v: float):
        self._density = v

    @property
    def scattering(self) -> float:
        return getattr(self, '_scattering', 0.5)

    @scattering.setter
    def scattering(self, v: float):
        self._scattering = v

    @property
    def step_count(self) -> int:
        return getattr(self, '_step_count', 32)

    @step_count.setter
    def step_count(self, v: int):
        self._step_count = v
