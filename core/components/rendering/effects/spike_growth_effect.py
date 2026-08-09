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
from core.maths.math3d import Vec3
from core.components.rendering.effects.object_effect import ObjectEffect


SPIKE_GEOM_SHADER = """#version 460 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 12) out;

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

uniform float u_spike_amount;
uniform float u_spike_length;
uniform float u_spike_radius;
uniform float u_spike_noise;
uniform float u_spike_twist;

float hash31(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

void emitVert(vec3 localPos, vec3 nrm, int idx) {
    vec4 wp = u_model * vec4(localPos, 1.0);
    v_world_pos = wp.xyz;
    v_normal = normalize(u_normal_matrix * nrm);
    v_uv = gs_uv[idx];
    vec4 vp = u_view * wp;
    v_view_pos = vp.xyz;
    v_local_pos = localPos;
    gl_Position = u_proj * vp;
    EmitVertex();
}

void main() {
    if (u_spike_amount <= 0.0) {
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
        return;
    }

    vec3 c0 = gs_local_pos[0];
    vec3 c1 = gs_local_pos[1];
    vec3 c2 = gs_local_pos[2];
    vec3 triCenter = (c0 + c1 + c2) / 3.0;
    vec3 nrm = normalize(gs_normal[0] + gs_normal[1] + gs_normal[2] + vec3(1e-5));
    vec3 n0 = gs_normal[0];
    vec3 n1 = gs_normal[1];
    vec3 n2 = gs_normal[2];

    float seed = hash31(floor((triCenter - u_obj_center) / max(0.001, u_obj_scale)) + 4.7);
    float lenVar = 1.0 + (hash31(triCenter * 5.3 + u_obj_center) - 0.5) * u_spike_noise * 2.0;
    float baseLen = u_spike_amount * u_spike_length * u_obj_scale * lenVar * (0.4 + seed);
    float radius = u_spike_amount * u_spike_radius * u_obj_scale * 0.5;
    float twist = u_spike_twist * u_spike_amount * 6.2831853 * seed;

    vec3 b0 = mix(triCenter, c0, 1.0 - radius);
    vec3 b1 = mix(triCenter, c1, 1.0 - radius);
    vec3 b2 = mix(triCenter, c2, 1.0 - radius);

    vec3 apexLocal = triCenter + nrm * baseLen;
    vec3 sideDir = normalize(cross(nrm, vec3(0.0, 1.0, 0.0)) + vec3(1e-5));
    if (abs(dot(nrm, vec3(0.0, 1.0, 0.0))) > 0.99) {
        sideDir = normalize(cross(nrm, vec3(1.0, 0.0, 0.0)) + vec3(1e-5));
    }
    vec3 perp = normalize(cross(nrm, sideDir));
    apexLocal += (sideDir * cos(twist) + perp * sin(twist)) * radius * 0.5;

    emitVert(b0, n0, 0);
    emitVert(b1, n1, 1);
    emitVert(b2, n2, 2);
    EndPrimitive();

    emitVert(apexLocal, nrm, 0);
    emitVert(b0, n0, 0);
    emitVert(b1, n1, 1);
    EndPrimitive();

    emitVert(apexLocal, nrm, 1);
    emitVert(b1, n1, 1);
    emitVert(b2, n2, 2);
    EndPrimitive();

    emitVert(apexLocal, nrm, 2);
    emitVert(b2, n2, 2);
    emitVert(b0, n0, 0);
    EndPrimitive();
}
"""

SPIKE_FRAG_UNIFORMS = """
uniform float u_spike_amount;
uniform vec3 u_spike_color;
uniform float u_spike_emission;
uniform float u_spike_rim;
uniform float u_spike_cell;
"""

SPIKE_FRAG_SNIPPET = """
    if (u_spike_amount > 0.0) {
        vec3 V = normalize(u_camera_pos - v_world_pos);
        float fres = pow(1.0 - max(dot(normalize(v_normal), V), 0.0), max(0.1, u_spike_rim));
        float h = hash31(floor((v_local_pos - u_obj_center) / max(0.001, u_spike_cell)) + 3.1);
        float glow = u_spike_amount * (u_spike_emission + fres * u_spike_emission * 1.5) * (0.6 + 0.4 * h);
        vec3 spikeCol = u_spike_color * glow;
        result = result + spikeCol;
        fx_alpha = clamp(fx_alpha + fres * u_spike_amount * 0.4, 0.0, 1.0);
    }
"""


@ComponentRegistry.register
class SpikeGrowthEffect(ObjectEffect):
    _gizmo_icon_label = "S"
    fx_uniform_defaults = {"u_spike_amount": 0.0}

    @classmethod
    def fx_geometry_shader(cls) -> "str | None":
        return SPIKE_GEOM_SHADER

    @classmethod
    def fx_fragment_uniforms(cls) -> str:
        return SPIKE_FRAG_UNIFORMS

    @classmethod
    def fx_fragment_snippet(cls) -> str:
        return SPIKE_FRAG_SNIPPET

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("amount", "Amount", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("length", "Spike Length", FieldType.FLOAT, step=0.05, decimals=3),
            InspectorField("radius", "Base Shrink", FieldType.FLOAT, step=0.01, decimals=3),
            InspectorField("noise", "Length Noise", FieldType.FLOAT, step=0.05, decimals=3),
            InspectorField("twist", "Twist", FieldType.FLOAT, step=0.05, decimals=3),
            InspectorField("emission", "Glow", FieldType.FLOAT, step=0.05, decimals=3),
            InspectorField("color", "Spike Color", FieldType.COLOR),
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
        self.length: float = 1.5
        self.radius: float = 0.35
        self.noise: float = 0.6
        self.twist: float = 0.6
        self.emission: float = 3.0
        self.color: list[float] = [0.35, 0.9, 1.0]
        self.rim: float = 2.5
        self.cell: float = 0.25
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
            self._set(prog, "u_spike_amount", 0.0)
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
            self._set(prog, "u_spike_amount", 0.0)
            self._set(prog, "u_double_sided", 1.0 if self.double_sided else 0.0)
            return
        self._col_buf[0] = self.color[0]
        self._col_buf[1] = self.color[1]
        self._col_buf[2] = self.color[2]
        self._set(prog, "u_spike_amount", float(self.amount))
        self._set(prog, "u_spike_length", float(self.length))
        self._set(prog, "u_spike_radius", float(self.radius))
        self._set(prog, "u_spike_noise", float(self.noise))
        self._set(prog, "u_spike_twist", float(self.twist))
        self._set(prog, "u_spike_emission", float(self.emission))
        self._set_vec_bytes(prog, "u_spike_color", self._col_buf)
        self._set(prog, "u_spike_rim", float(self.rim))
        self._set(prog, "u_spike_cell", float(self.cell))
        self._set(prog, "u_double_sided", 1.0 if self.double_sided else 0.0)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "amount": self.amount,
            "length": self.length,
            "radius": self.radius,
            "noise": self.noise,
            "twist": self.twist,
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
    def deserialize(cls, data: dict) -> "SpikeGrowthEffect":
        fx = cls()
        fx.enabled = data.get("enabled", True)
        fx.amount = data.get("amount", 0.5)
        fx.length = data.get("length", 1.5)
        fx.radius = data.get("radius", 0.35)
        fx.noise = data.get("noise", 0.6)
        fx.twist = data.get("twist", 0.6)
        fx.emission = data.get("emission", 3.0)
        fc = data.get("color", [0.35, 0.9, 1.0])
        fx.color = list(fc)
        fx.rim = data.get("rim", 2.5)
        fx.cell = data.get("cell", 0.25)
        fx.double_sided = data.get("double_sided", True)
        fx.animate = data.get("animate", False)
        fx.speed_anim = data.get("speed_anim", 0.5)
        fx.ping_pong = data.get("ping_pong", False)
        return fx
