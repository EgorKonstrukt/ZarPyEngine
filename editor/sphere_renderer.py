# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import numpy as np
import moderngl
from typing import Optional
from PyQt6.QtGui import QImage
from core.maths.math3d import Mat4, Vec3

VSHADER = """
#version 460 core
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
out vec3 v_normal;
out vec3 v_pos;
out vec2 v_uv;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
void main() {
    vec4 world_pos = u_model * vec4(in_pos, 1.0);
    v_pos = world_pos.xyz;
    v_normal = mat3(u_model) * in_normal;
    v_uv = in_uv;
    gl_Position = u_proj * u_view * world_pos;
}
"""

FSHADER = """
#version 460 core
in vec3 v_normal;
in vec3 v_pos;
in vec2 v_uv;
out vec4 frag_color;
uniform vec3 u_albedo;
uniform float u_metallic;
uniform float u_smoothness;
uniform vec3 u_emission;
uniform vec3 u_light_dir;
uniform vec3 u_cam_pos;
uniform bool u_has_tex;
uniform sampler2D u_tex;
void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;
    vec3 albedo = u_albedo;
    if (u_has_tex) {
        albedo *= texture(u_tex, v_uv).rgb;
    }
    vec3 l = normalize(u_light_dir);
    vec3 v = normalize(u_cam_pos - v_pos);
    vec3 h = normalize(l + v);
    float diff = max(dot(n, l), 0.0);
    float spec = pow(max(dot(n, h), 0.0), mix(1.0, 256.0, u_smoothness));
    vec3 ambient = 0.25 * albedo;
    vec3 diffuse = diff * albedo;
    vec3 specular = spec * mix(vec3(0.04), albedo, u_metallic);
    vec3 color = ambient + diffuse + specular + u_emission;
    frag_color = vec4(color, 1.0);
}
"""


def make_uv_sphere(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    verts = []
    idxs = []
    for r in range(rows + 1):
        phi = math.pi * r / rows
        for c in range(cols + 1):
            theta = 2.0 * math.pi * c / cols
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            u = 1.0 - c / cols
            v = 1.0 - r / rows
            verts.extend([x, y, z, x, y, z, u, v])
    for r in range(rows):
        for c in range(cols):
            a = r * (cols + 1) + c
            b = a + cols + 1
            idxs.extend([a, b, a + 1, b, b + 1, a + 1])
    return np.array(verts, dtype=np.float32), np.array(idxs, dtype=np.int32)


def load_texture(ctx: moderngl.Context, path: str) -> Optional[moderngl.Texture]:
    try:
        img = QImage(path)
        if img.isNull():
            return None
        img = img.convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = img.width(), img.height()
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        data = np.frombuffer(ptr, dtype=np.uint8)
        tex = ctx.texture((w, h), 4, data.tobytes())
        tex.build_mipmaps()
        tex.anisotropy = 4.0
        return tex
    except Exception:
        return None


def make_widget_resources(ctx: moderngl.Context):
    prog = ctx.program(vertex_shader=VSHADER, fragment_shader=FSHADER)
    verts, idxs = make_uv_sphere(48, 32)
    idx_count = len(idxs)
    vbo = ctx.buffer(verts.tobytes())
    ibo = ctx.buffer(idxs.tobytes())
    vao = ctx.vertex_array(
        prog,
        [(vbo, "3f 3f 2f", "in_pos", "in_normal", "in_uv")],
        ibo,
    )
    return prog, vao, vbo, ibo, idx_count


def render_view(prog, vao, ctx, width, height,
                albedo, metallic, smoothness, emission, emit_intensity,
                theta, phi, dist, tex=None):
    ctx.clear(0.1, 0.1, 0.1, 1.0)
    ctx.enable(moderngl.DEPTH_TEST)
    aspect = width / height if height > 0 else 1.0
    proj = Mat4.perspective(20.0, aspect, 0.1, 100.0).to_f32()
    eye = Vec3(
        dist * math.sin(theta) * math.cos(phi),
        dist * math.sin(phi),
        dist * math.cos(theta) * math.cos(phi),
    )
    view = Mat4.look_at(eye, Vec3(0, 0, 0), Vec3(0, 1, 0)).to_f32()
    model = Mat4.identity().to_f32()
    prog["u_proj"].write(proj.tobytes())
    prog["u_view"].write(view.tobytes())
    prog["u_model"].write(model.tobytes())
    c = albedo[:3] if len(albedo) >= 3 else [*albedo, 1.0]
    prog["u_albedo"].value = tuple(c)
    prog["u_metallic"].value = metallic
    prog["u_smoothness"].value = smoothness
    if emission and len(emission) >= 3:
        prog["u_emission"].value = (
            emission[0] * emit_intensity,
            emission[1] * emit_intensity,
            emission[2] * emit_intensity,
        )
    else:
        prog["u_emission"].value = (0.0, 0.0, 0.0)
    ld = (-0.5, 0.7, 0.8)
    n = math.sqrt(sum(d * d for d in ld))
    prog["u_light_dir"].value = (ld[0] / n, ld[1] / n, ld[2] / n)
    prog["u_cam_pos"].value = (eye.x, eye.y, eye.z)
    if tex is not None:
        tex.use(0)
        prog["u_tex"].value = 0
        prog["u_has_tex"].value = True
    else:
        prog["u_has_tex"].value = False
    vao.render(moderngl.TRIANGLES)
