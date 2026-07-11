# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import os
import numpy as np
import moderngl
from typing import Optional
from PyQt6.QtGui import QImage, QPixmap
from core.math.math3d import Mat4, Vec3
from core.foundation.logger import Logger

_ctx: Optional[moderngl.Context] = None
_sph_prog: Optional[moderngl.Program] = None
_sph_vao: Optional[moderngl.VertexArray] = None
_sph_vbo: Optional[moderngl.Buffer] = None
_sph_ibo: Optional[moderngl.Buffer] = None
_sph_count: int = 0
_mdl_prog: Optional[moderngl.Program] = None
_ready: bool = False
_safe: bool = True

SPH_VSHADER = """
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

SPH_FSHADER = """
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

MDL_VSHADER = """
#version 460 core
in vec3 in_pos;
in vec3 in_nrm;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_mat;
out vec3 v_normal;
out vec3 v_pos;
void main() {
    vec4 world_pos = u_model * vec4(in_pos, 1.0);
    v_pos = world_pos.xyz;
    v_normal = u_normal_mat * in_nrm;
    gl_Position = u_proj * u_view * world_pos;
}
"""

MDL_FSHADER = """
#version 460 core
in vec3 v_normal;
in vec3 v_pos;
out vec4 frag_color;
uniform vec3 u_light_dir;
uniform vec3 u_cam_pos;
uniform vec3 u_color;
uniform float u_ambient;
void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;
    vec3 l = normalize(u_light_dir);
    vec3 v = normalize(u_cam_pos - v_pos);
    vec3 h = normalize(l + v);
    float diff = max(dot(n, l), 0.0);
    float spec = pow(max(dot(n, h), 0.0), 32.0);
    vec3 color = u_color * (u_ambient + diff * (1.0 - u_ambient)) + vec3(spec * 0.3);
    frag_color = vec4(color, 1.0);
}
"""


def _ensure() -> bool:
    global _ctx, _ready, _safe
    if _ready:
        return _safe
    if not _safe:
        return False
    try:
        _ctx = moderngl.create_standalone_context(require=460)
        _ctx.pixel_alignment = 1
        _ready = True
        return True
    except Exception as e:
        Logger.warn(f"gl_offscreen: cannot create GL context: {e}")
        _safe = False
        return False


def _ensure_sph():
    global _sph_prog, _sph_vao, _sph_vbo, _sph_ibo, _sph_count
    if _sph_prog is not None:
        return
    _sph_prog = _ctx.program(vertex_shader=SPH_VSHADER, fragment_shader=SPH_FSHADER)
    verts, idxs = _make_uv_sphere(48, 32)
    _sph_count = len(idxs)
    _sph_vbo = _ctx.buffer(verts.tobytes())
    _sph_ibo = _ctx.buffer(idxs.tobytes())
    _sph_vao = _ctx.vertex_array(
        _sph_prog,
        [(_sph_vbo, "3f 3f 2f", "in_pos", "in_normal", "in_uv")],
        _sph_ibo,
    )


def _make_uv_sphere(rows, cols):
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


def _load_tex(path: str) -> Optional[moderngl.Texture]:
    try:
        img = QImage(path)
        if img.isNull():
            return None
        img = img.convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = img.width(), img.height()
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        data = np.frombuffer(ptr, dtype=np.uint8)
        tex = _ctx.texture((w, h), 4, data.tobytes())
        tex.build_mipmaps()
        tex.anisotropy = 4.0
        return tex
    except Exception:
        return None


def _set_view(prog, theta, phi, dist, aspect):
    proj = Mat4.perspective(30.0, aspect, 0.1, 100.0).to_f32()
    eye = Vec3(
        dist * math.sin(theta) * math.cos(phi),
        dist * math.sin(phi),
        dist * math.cos(theta) * math.cos(phi),
    )
    view = Mat4.look_at(eye, Vec3(0, 0, 0), Vec3(0, 1, 0)).to_f32()
    prog["u_proj"].write(proj.tobytes())
    prog["u_view"].write(view.tobytes())
    prog["u_cam_pos"].value = (eye.x, eye.y, eye.z)


def render_sphere(w: int, h: int, albedo, metallic, smoothness,
                  emission, emit_intensity, theta, phi, dist,
                  tex_path: str = "") -> Optional[QPixmap]:
    if not _ensure() or w < 1 or h < 1:
        return None
    _ensure_sph()
    tex = None
    if tex_path and os.path.exists(tex_path):
        tex = _load_tex(tex_path)
    fbo = _ctx.framebuffer(
        color_attachments=[_ctx.texture((w, h), 4)],
        depth_attachment=_ctx.depth_texture((w, h)),
    )
    fbo.use()
    _ctx.clear(0.1, 0.1, 0.1, 1.0)
    _ctx.enable(moderngl.DEPTH_TEST)
    _set_view(_sph_prog, theta, phi, dist, w / h if h > 0 else 1.0)
    model = Mat4.identity().to_f32()
    _sph_prog["u_model"].write(model.tobytes())
    c = albedo[:3] if len(albedo) >= 3 else [*albedo, 1.0]
    _sph_prog["u_albedo"].value = tuple(c)
    _sph_prog["u_metallic"].value = metallic
    _sph_prog["u_smoothness"].value = smoothness
    if emission and len(emission) >= 3:
        _sph_prog["u_emission"].value = (
            emission[0] * emit_intensity,
            emission[1] * emit_intensity,
            emission[2] * emit_intensity,
        )
    else:
        _sph_prog["u_emission"].value = (0.0, 0.0, 0.0)
    ld = (-0.5, 0.7, 0.8)
    n = math.sqrt(sum(d * d for d in ld))
    _sph_prog["u_light_dir"].value = (ld[0] / n, ld[1] / n, ld[2] / n)
    if tex is not None:
        tex.use(0)
        _sph_prog["u_tex"].value = 0
        _sph_prog["u_has_tex"].value = True
    else:
        _sph_prog["u_has_tex"].value = False
    _sph_vao.render(moderngl.TRIANGLES)
    data = fbo.read(components=4, dtype='f1')
    color_tex = fbo.color_attachments[0]
    depth_tex = fbo.depth_attachment
    fbo.release()
    color_tex.release()
    depth_tex.release()
    if tex:
        tex.release()
    if data is None or len(data) < w * h * 4:
        return None
    img = QImage(data, w, h, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(img)


def _compute_normals(pts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    n_vert = len(pts)
    nrm = np.zeros((n_vert, 3), dtype=np.float32)
    idx_arr = indices.reshape(-1, 3)
    for tri in idx_arr:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        e1 = pts[i1] - pts[i0]
        e2 = pts[i2] - pts[i0]
        fn = np.cross(e1, e2)
        fn_len = np.linalg.norm(fn)
        if fn_len > 1e-10:
            fn /= fn_len
        nrm[i0] += fn
        nrm[i1] += fn
        nrm[i2] += fn
    for i in range(n_vert):
        nl = np.linalg.norm(nrm[i])
        if nl > 1e-10:
            nrm[i] /= nl
        else:
            nrm[i] = np.array([0.0, 1.0, 0.0])
    return nrm


def render_mesh(w: int, h: int, verts: np.ndarray, indices: np.ndarray,
                theta, phi, dist, color=(0.7, 0.7, 0.7),
                normals: Optional[np.ndarray] = None) -> Optional[QPixmap]:
    if not _ensure() or w < 4 or h < 4 or len(verts) < 3 or len(indices) < 3:
        return None
    global _mdl_prog
    if _mdl_prog is None:
        _mdl_prog = _ctx.program(vertex_shader=MDL_VSHADER, fragment_shader=MDL_FSHADER)
    pts = verts.reshape(-1, 3).astype(np.float32)
    n_vert = len(pts)
    if normals is not None and len(normals) >= n_vert * 3:
        nrm = normals.reshape(-1, 3).astype(np.float32)
    else:
        nrm = _compute_normals(pts, indices)
    data = np.zeros((n_vert, 6), dtype=np.float32)
    data[:, :3] = pts
    data[:, 3:6] = nrm
    vbo = _ctx.buffer(data.tobytes())
    ibo = _ctx.buffer(indices.astype(np.int32).tobytes())
    vao = _ctx.vertex_array(
        _mdl_prog,
        [(vbo, "3f 3f", "in_pos", "in_nrm")],
        ibo,
    )
    fbo = _ctx.framebuffer(
        color_attachments=[_ctx.texture((w, h), 4)],
        depth_attachment=_ctx.depth_texture((w, h)),
    )
    fbo.use()
    phi = max(-math.pi / 2 + 0.05, min(math.pi / 2 - 0.05, phi))
    _ctx.clear(0.1, 0.1, 0.1, 1.0)
    _ctx.enable(moderngl.DEPTH_TEST)
    _set_view(_mdl_prog, theta, phi, dist, w / h if h > 0 else 1.0)
    model = Mat4.identity().to_f32()
    _mdl_prog["u_model"].write(model.tobytes())
    _mdl_prog["u_normal_mat"].write(np.eye(3, dtype=np.float32).tobytes())
    _mdl_prog["u_color"].value = tuple(color[:3])
    _mdl_prog["u_ambient"].value = 0.3
    ld = (-0.5, 0.7, 0.8)
    n = math.sqrt(sum(d * d for d in ld))
    _mdl_prog["u_light_dir"].value = (ld[0] / n, ld[1] / n, ld[2] / n)
    vao.render(moderngl.TRIANGLES)
    data = fbo.read(components=4, dtype='f1')
    vao.release()
    vbo.release()
    ibo.release()
    color_tex = fbo.color_attachments[0]
    depth_tex = fbo.depth_attachment
    fbo.release()
    color_tex.release()
    depth_tex.release()
    if data is None or len(data) < w * h * 4:
        return None
    img = QImage(data, w, h, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(img)


_ensure()
