# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
import numpy as np
from core.ecs.ecs import ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.math.math3d import Vec3
from core.components.rendering.effects.object_effect import ObjectEffect


VOXELIZE_GEOM_SHADER = """#version 460 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 32) out;

in vec3 gs_world_pos[3];
in vec3 gs_normal[3];
in vec2 gs_uv[3];
in vec3 gs_view_pos[3];
in vec3 gs_local_pos[3];

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out vec3 v_view_pos;
out vec3 v_local_pos;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_matrix;
uniform vec3 u_obj_center;
uniform float u_obj_scale;

uniform float u_vox_amount;
uniform float u_vox_size;
uniform float u_vox_height;
uniform float u_vox_jitter;
uniform float u_vox_show_base;
uniform float u_vox_world_grid;

float hash31(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

void emitFW(vec3 a, vec3 b, vec3 c, vec3 d, vec3 nrm) {
    v_normal = nrm;
    v_uv = vec2(0.0);
    vec4 wa = u_view * vec4(a, 1.0);
    vec4 wb = u_view * vec4(b, 1.0);
    vec4 wc = u_view * vec4(c, 1.0);
    vec4 wd = u_view * vec4(d, 1.0);
    v_world_pos = a; v_local_pos = a; v_view_pos = wa.xyz; gl_Position = u_proj * wa; EmitVertex();
    v_world_pos = b; v_local_pos = b; v_view_pos = wb.xyz; gl_Position = u_proj * wb; EmitVertex();
    v_world_pos = c; v_local_pos = c; v_view_pos = wc.xyz; gl_Position = u_proj * wc; EmitVertex();
    v_world_pos = d; v_local_pos = d; v_view_pos = wd.xyz; gl_Position = u_proj * wd; EmitVertex();
    EndPrimitive();
}

void emitCube(vec3 C, float h, float hy) {
    emitFW(C+vec3(-h,hy,-h), C+vec3(-h,hy, h), C+vec3( h,hy,-h), C+vec3( h,hy, h), vec3( 0, 1, 0));
    emitFW(C+vec3(-h,-hy,-h), C+vec3( h,-hy,-h), C+vec3(-h,-hy, h), C+vec3( h,-hy, h), vec3( 0,-1, 0));
    emitFW(C+vec3( h,-hy,-h), C+vec3( h, hy,-h), C+vec3( h,-hy, h), C+vec3( h, hy, h), vec3( 1, 0, 0));
    emitFW(C+vec3(-h,-hy,-h), C+vec3(-h,-hy, h), C+vec3(-h, hy,-h), C+vec3(-h, hy, h), vec3(-1, 0, 0));
    emitFW(C+vec3(-h,-hy, h), C+vec3( h,-hy, h), C+vec3(-h, hy, h), C+vec3( h, hy, h), vec3( 0, 0, 1));
    emitFW(C+vec3( h,-hy,-h), C+vec3(-h,-hy,-h), C+vec3( h, hy,-h), C+vec3(-h, hy,-h), vec3( 0, 0,-1));
}

void main() {
    if (u_vox_show_base > 0.5) {
        for (int i = 0; i < 3; i++) {
            gl_Position = gl_in[i].gl_Position;
            v_world_pos = gs_world_pos[i];
            v_normal = gs_normal[i];
            v_uv = gs_uv[i];
            v_view_pos = gs_view_pos[i];
            v_local_pos = gs_local_pos[i];
            EmitVertex();
        }
        EndPrimitive();
    }

    if (u_vox_amount <= 0.0) return;

    float vs = max(0.0001, u_vox_size);
    float h = vs * 0.5;
    float hy = h * max(0.01, u_vox_height);

    vec3 wc;
    if (u_vox_world_grid > 0.5) {
        vec3 w0 = (u_model * vec4(gs_local_pos[0], 1.0)).xyz;
        vec3 w1 = (u_model * vec4(gs_local_pos[1], 1.0)).xyz;
        vec3 w2 = (u_model * vec4(gs_local_pos[2], 1.0)).xyz;
        vec3 s0 = (floor(w0 / vs) + 0.5) * vs;
        vec3 s1 = (floor(w1 / vs) + 0.5) * vs;
        vec3 s2 = (floor(w2 / vs) + 0.5) * vs;
        float d0 = length(w0 - s0);
        float d1 = length(w1 - s1);
        float d2 = length(w2 - s2);
        if (d0 <= d1 && d0 <= d2) wc = s0;
        else if (d1 <= d2) wc = s1;
        else wc = s2;
    } else {
        vec3 s0 = (floor(gs_local_pos[0] / vs) + 0.5) * vs;
        vec3 s1 = (floor(gs_local_pos[1] / vs) + 0.5) * vs;
        vec3 s2 = (floor(gs_local_pos[2] / vs) + 0.5) * vs;
        float d0 = length(gs_local_pos[0] - s0);
        float d1 = length(gs_local_pos[1] - s1);
        float d2 = length(gs_local_pos[2] - s2);
        vec3 lc;
        if (d0 <= d1 && d0 <= d2) lc = s0;
        else if (d1 <= d2) lc = s1;
        else lc = s2;
        wc = (u_model * vec4(lc, 1.0)).xyz;
    }

    vec3 cell = floor(wc / vs);
    wc = (cell + 0.5) * vs;
    if (u_vox_jitter > 0.0) {
        float r1 = hash31(cell + 1.3);
        float r2 = hash31(cell + 7.7);
        float r3 = hash31(cell + 19.1);
        wc += (vec3(r1, r2, r3) - 0.5) * u_vox_jitter * vs;
    }
    emitCube(wc, h, hy);
}
"""

VOXELIZE_FRAG_UNIFORMS = """
uniform float u_vox_amount;
uniform vec3 u_vox_color;
uniform float u_vox_emission;
uniform float u_vox_rim;
uniform float u_vox_cell;
"""

VOXELIZE_FRAG_SNIPPET = """
    if (u_vox_amount > 0.0) {
        vec3 V = normalize(u_camera_pos - v_world_pos);
        float fres = pow(1.0 - max(dot(normalize(v_normal), V), 0.0), max(0.1, u_vox_rim));
        float h = hash31(floor((v_local_pos - u_obj_center) / max(0.001, u_vox_cell)) + 5.7);
        float glow = u_vox_amount * (u_vox_emission + fres * u_vox_emission * 1.5) * (0.6 + 0.4 * h);
        vec3 voxelCol = u_vox_color * glow;
        result = result + voxelCol;
        fx_alpha = clamp(fx_alpha + fres * u_vox_amount * 0.4, 0.0, 1.0);
    }
"""


@ComponentRegistry.register
class VoxelizeEffect(ObjectEffect):
    _gizmo_icon_label = "V"
    fx_uniform_defaults = {"u_vox_amount": 0.0, "u_vox_show_base": 1.0, "u_vox_world_grid": 0.0}

    @classmethod
    def fx_geometry_shader(cls) -> "str | None":
        return VOXELIZE_GEOM_SHADER

    @classmethod
    def fx_fragment_uniforms(cls) -> str:
        return VOXELIZE_FRAG_UNIFORMS

    @classmethod
    def fx_fragment_snippet(cls) -> str:
        return VOXELIZE_FRAG_SNIPPET

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("amount", "Amount", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("show_base_mesh", "Show Base Mesh", FieldType.BOOL),
            InspectorField("world_grid", "World Grid", FieldType.BOOL),
            InspectorField("voxel_size", "Voxel Size", FieldType.FLOAT, step=0.01, decimals=3),
            InspectorField("height", "Voxel Height", FieldType.FLOAT, step=0.05, decimals=3),
            InspectorField("jitter", "Jitter", FieldType.FLOAT, step=0.01, decimals=3),
            InspectorField("emission", "Glow", FieldType.FLOAT, step=0.05, decimals=3),
            InspectorField("color", "Voxel Color", FieldType.COLOR),
            InspectorField("rim", "Rim Power", FieldType.FLOAT, step=0.05, decimals=3),
            InspectorField("cell", "Cell Size", FieldType.FLOAT, step=0.01, decimals=3),
            InspectorField("double_sided", "Double Sided", FieldType.BOOL),
            InspectorField("animate", "Animate", FieldType.BOOL),
            InspectorField("speed_anim", "Anim Speed", FieldType.FLOAT, step=0.05, decimals=3),
            InspectorField("ping_pong", "Ping Pong", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.amount: float = 0.5
        self.show_base_mesh: bool = True
        self.world_grid: bool = False
        self.voxel_size: float = 0.3
        self.height: float = 1.0
        self.jitter: float = 0.0
        self.emission: float = 2.5
        self.color: list[float] = [0.4, 1.0, 0.6]
        self.rim: float = 2.5
        self.cell: float = 0.3
        self.double_sided: bool = True
        self.animate: bool = False
        self.speed_anim: float = 0.5
        self.ping_pong: bool = False
        self._anim_active: bool = False
        self._col_buf = np.zeros(3, dtype=np.float32)

    def on_awake(self):
        super().on_awake()
        self._time_offset = time.time()

    def _apply(self, prog):
        if not self.enabled:
            self._set(prog, "u_vox_amount", 0.0)
            return
        if self.animate:
            if not self._anim_active:
                self._time_offset = time.time()
                self._anim_active = True
            t = (time.time() - self._time_offset) * self.speed_anim
            tri = abs((t % 2.0) - 1.0) if self.ping_pong else (t % 1.0)
            self.amount = max(0.0, min(1.0, tri))
        else:
            self._anim_active = False
        if self.amount <= 0.0:
            self._set(prog, "u_vox_amount", 0.0)
            self._set(prog, "u_vox_show_base", 1.0 if self.show_base_mesh else 0.0)
            self._set(prog, "u_double_sided", 1.0 if self.double_sided else 0.0)
            return
        self._col_buf[0] = self.color[0]
        self._col_buf[1] = self.color[1]
        self._col_buf[2] = self.color[2]
        self._set(prog, "u_vox_amount", float(self.amount))
        self._set(prog, "u_vox_show_base", 1.0 if self.show_base_mesh else 0.0)
        self._set(prog, "u_vox_world_grid", 1.0 if self.world_grid else 0.0)
        self._set(prog, "u_vox_size", float(self.voxel_size))
        self._set(prog, "u_vox_height", float(self.height))
        self._set(prog, "u_vox_jitter", float(self.jitter))
        self._set(prog, "u_vox_emission", float(self.emission))
        self._set_vec_bytes(prog, "u_vox_color", self._col_buf)
        self._set(prog, "u_vox_rim", float(self.rim))
        self._set(prog, "u_vox_cell", float(self.cell))
        self._set(prog, "u_double_sided", 1.0 if self.double_sided else 0.0)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "amount": self.amount,
            "show_base_mesh": self.show_base_mesh,
            "world_grid": self.world_grid,
            "voxel_size": self.voxel_size,
            "height": self.height,
            "jitter": self.jitter,
            "emission": self.emission,
            "color": list(self.color),
            "rim": self.rim,
            "cell": self.cell,
            "double_sided": self.double_sided,
            "animate": self.animate,
            "speed_anim": self.speed_anim,
            "ping_pong": self.ping_pong,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> "VoxelizeEffect":
        fx = cls()
        fx.enabled = data.get("enabled", True)
        fx.amount = data.get("amount", 0.5)
        fx.show_base_mesh = data.get("show_base_mesh", True)
        fx.world_grid = data.get("world_grid", False)
        fx.voxel_size = data.get("voxel_size", 0.3)
        fx.height = data.get("height", 1.0)
        fx.jitter = data.get("jitter", 0.0)
        fx.emission = data.get("emission", 2.5)
        fc = data.get("color", [0.4, 1.0, 0.6])
        fx.color = list(fc)
        fx.rim = data.get("rim", 2.5)
        fx.cell = data.get("cell", 0.3)
        fx.double_sided = data.get("double_sided", True)
        fx.animate = data.get("animate", False)
        fx.speed_anim = data.get("speed_anim", 0.5)
        fx.ping_pong = data.get("ping_pong", False)
        return fx
