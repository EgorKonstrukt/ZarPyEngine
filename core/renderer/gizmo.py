# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
import moderngl
from typing import Optional
from core.maths.math3d import Vec3, Mat4
from core.renderer.gpu_primitives import (
    GpuMesh, make_cone_mesh, make_cylinder_mesh,
    make_cube_mesh, make_quad_mesh, make_circle_ring_mesh, make_instance_vao,
    make_unit_box_line_verts, make_unit_sphere_line_verts,
    make_unit_rect_line_verts, make_unit_circle_line_verts, make_unit_capsule_line_verts,
    make_instance_line_vao
)

_STRIP_T = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
_STRIP_S = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float32)

_RL_DIFF_THRESHOLD = 50000

FATLINE_VERT = """
#version 460 core
uniform mat4 u_mvp;
in vec3 a_start;
in vec3 a_end;
in float a_t;
in float a_side;
in vec4 a_color;
out vec3 v_color;
out float v_alpha;
out float v_t;
out float v_line_len;
uniform float u_thickness_ndc_x;
uniform float u_thickness_ndc_y;
void main() {
    vec4 clip_start = u_mvp * vec4(a_start, 1.0);
    vec4 clip_end   = u_mvp * vec4(a_end,   1.0);
    vec4 clipPos = mix(clip_start, clip_end, a_t);
    vec2 dxy = clip_end.xy - clip_start.xy;
    float len = length(dxy);
    vec2 perp;
    if (len > 1e-6) {
        vec2 dir = dxy / len;
        perp = vec2(-dir.y, dir.x);
    } else {
        perp = vec2(1.0, 0.0);
    }
    float thickness = max(u_thickness_ndc_x, u_thickness_ndc_y);
    float aspect = u_thickness_ndc_y / max(u_thickness_ndc_x, 1e-6);
    vec2 adj_perp = normalize(vec2(perp.x * aspect, perp.y));
    clipPos.xy += adj_perp * a_side * thickness * clipPos.w;
    gl_Position = clipPos;
    v_color = a_color.rgb;
    v_alpha = a_color.a;
    v_t = a_t;
    v_line_len = length(a_end - a_start);
}
"""

FATLINE_FRAG = """
#version 460 core
in vec3 v_color;
in float v_alpha;
in float v_t;
in float v_line_len;
uniform bool u_dash_enabled;
uniform float u_dash_length;
uniform float u_gap_length;
uniform float u_dash_time;
out vec4 fragColor;
void main() {
    if (u_dash_enabled) {
        float total = u_dash_length + u_gap_length;
        float offset = u_dash_time * 1.2;
        float pos = v_t * v_line_len + offset;
        float mod_pos = mod(pos, total);
        if (mod_pos > u_dash_length) {
            discard;
        }
    }
    fragColor = vec4(v_color, v_alpha);
}
"""

INSTANCED_VERT = """
#version 460 core
in vec3 in_position;
in vec4 in_color;
in vec4 i_row0;
in vec4 i_row1;
in vec4 i_row2;
in vec4 i_row3;
in vec4 i_color;
uniform mat4 u_mvp;
out vec4 v_color;
void main() {
    mat4 i_model = mat4(i_row0, i_row1, i_row2, i_row3);
    v_color = i_color * in_color;
    gl_Position = u_mvp * i_model * vec4(in_position, 1.0);
}
"""

INSTANCED_FRAG = """
#version 460 core
in vec4 v_color;
out vec4 fragColor;
void main() {
    fragColor = v_color;
}
"""

RAWLINES_VERT = """
#version 460 core
in vec4 i_start;
in vec4 i_end;
in vec4 i_color;
in vec2 a_strip;
uniform mat4 u_mvp;
uniform float u_thickness_ndc_x;
uniform float u_thickness_ndc_y;
uniform bool u_dash_enabled;
uniform float u_dash_length;
uniform float u_gap_length;
uniform float u_dash_time;
out vec4 v_color;
out float v_t;
out float v_line_len;
void main() {
    float a_t = a_strip.x;
    float a_side = a_strip.y;
    vec4 clip_start = u_mvp * vec4(i_start.xyz, 1.0);
    vec4 clip_end   = u_mvp * vec4(i_end.xyz, 1.0);
    vec4 clipPos = mix(clip_start, clip_end, a_t);
    vec2 dxy = clip_end.xy - clip_start.xy;
    float len = length(dxy);
    vec2 perp;
    if (len > 1e-6) {
        vec2 dir = dxy / len;
        perp = vec2(-dir.y, dir.x);
    } else {
        perp = vec2(1.0, 0.0);
    }
    clipPos.xy += perp * a_side * vec2(u_thickness_ndc_x, u_thickness_ndc_y) * clipPos.w;
    gl_Position = clipPos;
    v_color = i_color;
    v_t = a_t;
    v_line_len = length(i_end.xyz - i_start.xyz);
}
"""

RAWLINES_FRAG = """
#version 460 core
in vec4 v_color;
in float v_t;
in float v_line_len;
uniform bool u_dash_enabled;
uniform float u_dash_length;
uniform float u_gap_length;
uniform float u_dash_time;
out vec4 fragColor;
void main() {
    if (u_dash_enabled) {
        float total = u_dash_length + u_gap_length;
        float offset = u_dash_time * 1.2;
        float pos = v_t * v_line_len + offset;
        float mod_pos = mod(pos, total);
        if (mod_pos > u_dash_length) {
            discard;
        }
    }
    fragColor = v_color;
}
"""

class GizmoRenderer:
    def __init__(self, ctx: moderngl.Context, gizmo_prog: moderngl.Program,
                 fatline_prog: moderngl.Program, solid_prog: moderngl.Program):
        self._ctx = ctx
        self._gizmo_prog = gizmo_prog
        self._fatline_prog = fatline_prog
        self._solid_prog = solid_prog
        self._line_width: float = 2.0
        self._fatline_vbo_start: Optional[moderngl.Buffer] = None
        self._fatline_vbo_end: Optional[moderngl.Buffer] = None
        self._fatline_vbo_t: Optional[moderngl.Buffer] = None
        self._fatline_vbo_side: Optional[moderngl.Buffer] = None
        self._fatline_vbo_color: Optional[moderngl.Buffer] = None
        self._fatline_vao: Optional[moderngl.VertexArray] = None
        self._fatline_capacity: int = 0
        self._fs_starts = np.empty((0, 3), dtype=np.float32)
        self._fs_ends = np.empty((0, 3), dtype=np.float32)
        self._fs_t = np.empty((0,), dtype=np.float32)
        self._fs_side = np.empty((0,), dtype=np.float32)
        self._fs_colors = np.empty((0, 4), dtype=np.float32)
        self._solid_vbo: Optional[moderngl.Buffer] = None
        self._solid_ibo: Optional[moderngl.Buffer] = None
        self._solid_vao: Optional[moderngl.VertexArray] = None
        self._solid_vbo_cap: int = 0
        self._solid_ibo_cap: int = 0
        self._instanced_prog: Optional[moderngl.Program] = None
        self._cone_mesh: Optional[GpuMesh] = None
        self._cylinder_mesh: Optional[GpuMesh] = None
        self._cube_mesh: Optional[GpuMesh] = None
        self._quad_mesh: Optional[GpuMesh] = None
        self._circle_mesh: Optional[GpuMesh] = None
        self._instanced_initialized: bool = False
        self._inst_line_prog: Optional[moderngl.Program] = None
        self._box_inst_mesh: Optional[GpuMesh] = None
        self._sphere_inst_mesh: Optional[GpuMesh] = None
        self._inst_line_initialized: bool = False
        self._raw_line_prog: Optional[moderngl.Program] = None
        self._raw_line_static_vbo: Optional[moderngl.Buffer] = None
        self._raw_line_start_vbo: Optional[moderngl.Buffer] = None
        self._raw_line_end_vbo: Optional[moderngl.Buffer] = None
        self._raw_line_color_vbo: Optional[moderngl.Buffer] = None
        self._raw_line_vao: Optional[moderngl.VertexArray] = None
        self._raw_line_capacity: int = 0
        self._fs_starts: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self._fs_ends: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self._fs_colors4: np.ndarray = np.empty((0, 4), dtype=np.float32)
        self._rl_shadow_scalar: np.ndarray = np.zeros(3, dtype=np.uint32)
        self._rl_shadow_starts: np.ndarray = np.zeros(0, dtype=np.uint32)
        self._rl_shadow_ends: np.ndarray = np.zeros(0, dtype=np.uint32)
        self._rl_shadow_colors: np.ndarray = np.zeros(0, dtype=np.uint32)
        self._rl_shadow_n: int = 0
        self._build_fatline_buffers()
        self._build_solid_buffers()
        self._stat_lines: int = 0
        self._stat_instances: int = 0
        self._stat_mesh_verts: int = 0
        self._stat_draws: int = 0
        self._stat_upload_bytes: int = 0
        self._stat_upload_full: int = 0
        self._stat_upload_partial: int = 0

    def _ensure_instanced_prog(self):
        if self._instanced_prog is not None:
            return
        try:
            self._instanced_prog = self._ctx.program(
                vertex_shader=INSTANCED_VERT,
                fragment_shader=INSTANCED_FRAG
            )
        except Exception:
            self._instanced_prog = None

    def initialize_instanced_meshes(self):
        if self._instanced_initialized:
            return
        self._ensure_instanced_prog()
        if self._instanced_prog is None:
            return
        ctx = self._ctx
        prog = self._instanced_prog
        self._cone_mesh = make_instance_vao(ctx, prog, make_cone_mesh(ctx, prog))
        self._cylinder_mesh = make_instance_vao(ctx, prog, make_cylinder_mesh(ctx, prog))
        self._cube_mesh = make_instance_vao(ctx, prog, make_cube_mesh(ctx, prog))
        self._quad_mesh = make_instance_vao(ctx, prog, make_quad_mesh(ctx, prog))
        self._circle_mesh = make_instance_vao(ctx, prog, make_circle_ring_mesh(ctx, prog))
        self._instanced_initialized = True

    def _build_fatline_buffers(self, capacity: int = 32768):
        cap = max(capacity, 32768)
        dummy3 = np.zeros(cap * 3, dtype=np.float32)
        dummy1 = np.zeros(cap, dtype=np.float32)
        dummy4 = np.zeros(cap * 4, dtype=np.float32)
        for buf_name in ('_fatline_vbo_start', '_fatline_vbo_end', '_fatline_vbo_t', '_fatline_vbo_side', '_fatline_vbo_color'):
            b = getattr(self, buf_name, None)
            if b is not None:
                try: b.release()
                except: pass
        if self._fatline_vao is not None:
            try: self._fatline_vao.release()
            except: pass
        self._fatline_vbo_start = self._ctx.buffer(dummy3.tobytes(), dynamic=True)
        self._fatline_vbo_end = self._ctx.buffer(dummy3.tobytes(), dynamic=True)
        self._fatline_vbo_t = self._ctx.buffer(dummy1.tobytes(), dynamic=True)
        self._fatline_vbo_side = self._ctx.buffer(dummy1.tobytes(), dynamic=True)
        self._fatline_vbo_color = self._ctx.buffer(dummy4.tobytes(), dynamic=True)
        self._fatline_vao = self._ctx.vertex_array(
            self._fatline_prog,
            [
                (self._fatline_vbo_start, "3f", "a_start"),
                (self._fatline_vbo_end, "3f", "a_end"),
                (self._fatline_vbo_t, "1f", "a_t"),
                (self._fatline_vbo_side, "1f", "a_side"),
                (self._fatline_vbo_color, "4f", "a_color"),
            ]
        )
        self._fs_starts = np.empty((cap, 3), dtype=np.float32)
        self._fs_ends = np.empty((cap, 3), dtype=np.float32)
        self._fs_t = np.empty((cap,), dtype=np.float32)
        self._fs_side = np.empty((cap,), dtype=np.float32)
        self._fs_colors = np.empty((cap, 4), dtype=np.float32)
        self._fatline_capacity = cap

    def _ensure_fatline_capacity(self, n_verts: int):
        if n_verts <= self._fatline_capacity:
            return
        new_cap = int(n_verts * 1.5 + 4096)
        self._build_fatline_buffers(new_cap)

    def _build_solid_buffers(self, vcap: int = 4096, icap: int = 8192):
        vcap = max(vcap, 4096)
        icap = max(icap, 8192)
        for buf_name in ('_solid_vbo', '_solid_ibo'):
            b = getattr(self, buf_name, None)
            if b is not None:
                try: b.release()
                except: pass
        if self._solid_vao is not None:
            try: self._solid_vao.release()
            except: pass
        self._solid_vbo = self._ctx.buffer(np.zeros(vcap * 7, dtype=np.float32).tobytes(), dynamic=True)
        self._solid_ibo = self._ctx.buffer(np.zeros(icap, dtype=np.uint32).tobytes(), dynamic=True)
        self._solid_vao = self._ctx.vertex_array(
            self._solid_prog,
            [(self._solid_vbo, "3f 4f", "in_position", "in_color")],
            self._solid_ibo
        )
        self._solid_vbo_cap = vcap
        self._solid_ibo_cap = icap

    def render_lines(self, lines, vp_mat: Mat4, fw: int = 1920, fh: int = 1080, thickness_multiplier: float = 1.0):
        if not self._fatline_prog or not lines:
            return
        desired_pixels = max(1.0, float(self._line_width) * 1.5 * thickness_multiplier)
        if isinstance(lines, tuple) and len(lines) == 3 and all(isinstance(a, np.ndarray) for a in lines):
            starts_np, ends_np, colors_np = lines
            self._render_lines_np(starts_np, ends_np, colors_np, vp_mat, fw, fh, desired_pixels)
            return
        color_groups: dict = {}
        for start, end, color in lines:
            if isinstance(start, Vec3):
                key = (float(color[0]), float(color[1]), float(color[2]))
                color_groups.setdefault(key, []).append((start, end))
            else:
                key = (float(color[0]), float(color[1]), float(color[2]))
                color_groups.setdefault(key, []).append((start, end))
        try:
            old_cull = bool(self._ctx.cull_face)
        except:
            old_cull = True
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.disable(moderngl.DEPTH_TEST)
        prog = self._fatline_prog
        vp_f32 = vp_mat.to_f32()
        if "u_mvp" in prog:
            prog["u_mvp"].write(vp_f32.tobytes())
        ndc_x = desired_pixels / max(1.0, float(fw))
        ndc_y = desired_pixels / max(1.0, float(fh))
        if "u_thickness_ndc_x" in prog:
            prog["u_thickness_ndc_x"] = float(ndc_x)
        if "u_thickness_ndc_y" in prog:
            prog["u_thickness_ndc_y"] = float(ndc_y)
        strip_t = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        strip_s = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float32)
        try:
            for color_key, segs in color_groups.items():
                alpha_val = float(color_key[3]) if len(color_key) > 3 else 1.0
                if alpha_val <= 0.001:
                    continue
                n_segs = len(segs)
                n_verts = n_segs * 6
                pts = np.empty((n_segs, 6), dtype=np.float32)
                for i, (s, e) in enumerate(segs):
                    pts[i, 0] = s.x; pts[i, 1] = s.y; pts[i, 2] = s.z
                    pts[i, 3] = e.x; pts[i, 4] = e.y; pts[i, 5] = e.z
                starts_arr = np.repeat(pts[:, :3], 6, axis=0)
                ends_arr = np.repeat(pts[:, 3:], 6, axis=0)
                ts_arr = np.tile(strip_t, n_segs)
                side_arr = np.tile(strip_s, n_segs)
                self._ensure_fatline_capacity(n_verts)
                self._fatline_vbo_start.write(starts_arr.tobytes())
                self._fatline_vbo_end.write(ends_arr.tobytes())
                self._fatline_vbo_t.write(ts_arr.tobytes())
                self._fatline_vbo_side.write(side_arr.tobytes())
                color_arr = np.zeros((n_verts, 4), dtype=np.float32)
                color_arr[:, 0] = color_key[0]
                color_arr[:, 1] = color_key[1]
                color_arr[:, 2] = color_key[2]
                color_arr[:, 3] = alpha_val
                self._fatline_vbo_color.write(color_arr.tobytes())
                self._stat_upload_bytes += n_verts * 48
                self._stat_upload_full += 1
                self._fatline_vao.render(moderngl.TRIANGLES, vertices=n_verts)
        finally:
            if old_cull:
                self._ctx.enable(moderngl.CULL_FACE)
            else:
                self._ctx.disable(moderngl.CULL_FACE)
            self._ctx.enable(moderngl.DEPTH_TEST)

    def _render_lines_np(self, starts: np.ndarray, ends: np.ndarray, colors: np.ndarray,
                          vp_mat: Mat4, fw: int, fh: int, desired_pixels: float,
                          dash_opts: Optional[dict] = None):
        n_segs = starts.shape[0]
        if n_segs == 0:
            return
        self._stat_lines += n_segs
        self._stat_draws += 1
        n_verts = n_segs * 6
        try:
            old_cull = bool(self._ctx.cull_face)
        except:
            old_cull = True
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.disable(moderngl.DEPTH_TEST)
        prog = self._fatline_prog
        vp_f32 = vp_mat.to_f32()
        if "u_mvp" in prog:
            prog["u_mvp"].write(vp_f32.tobytes())
        ndc_x = desired_pixels / max(1.0, float(fw))
        ndc_y = desired_pixels / max(1.0, float(fh))
        if "u_thickness_ndc_x" in prog:
            prog["u_thickness_ndc_x"] = float(ndc_x)
        if "u_thickness_ndc_y" in prog:
            prog["u_thickness_ndc_y"] = float(ndc_y)
        if dash_opts and "u_dash_enabled" in prog:
            prog["u_dash_enabled"] = True
            prog["u_dash_length"] = float(dash_opts.get('dash_length', 0.3))
            prog["u_gap_length"] = float(dash_opts.get('gap_length', 0.15))
            prog["u_dash_time"] = float(dash_opts.get('time', 0.0))
        elif "u_dash_enabled" in prog:
            prog["u_dash_enabled"] = False
        self._ensure_fatline_capacity(n_verts)
        sv = self._fs_starts[:n_verts].reshape(-1, 6, 3)
        sv[:] = starts[:, None, :]
        ev = self._fs_ends[:n_verts].reshape(-1, 6, 3)
        ev[:] = ends[:, None, :]
        tv = self._fs_t[:n_verts].reshape(-1, 6)
        tv[:] = _STRIP_T[None, :]
        sidev = self._fs_side[:n_verts].reshape(-1, 6)
        sidev[:] = _STRIP_S[None, :]
        if colors.shape[1] == 3:
            cr = np.empty((n_segs, 4), dtype=np.float32)
            cr[:, :3] = colors
            cr[:, 3] = 1.0
        elif colors.shape[1] >= 4:
            cr = colors[:, :4]
        else:
            cr = np.full((n_segs, 4), 0.5, dtype=np.float32)
        cv = self._fs_colors[:n_verts].reshape(-1, 6, 4)
        cv[:] = cr[:, None, :]
        self._fatline_vbo_start.write(memoryview(self._fs_starts[:n_verts]))
        self._fatline_vbo_end.write(memoryview(self._fs_ends[:n_verts]))
        self._fatline_vbo_t.write(memoryview(self._fs_t[:n_verts]))
        self._fatline_vbo_side.write(memoryview(self._fs_side[:n_verts]))
        self._fatline_vbo_color.write(memoryview(self._fs_colors[:n_verts]))
        self._stat_upload_bytes += n_verts * 48
        self._stat_upload_full += 1
        self._fatline_vao.render(moderngl.TRIANGLES, vertices=n_verts)
        if old_cull:
            self._ctx.enable(moderngl.CULL_FACE)
        else:
            self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.DEPTH_TEST)

    def render_instanced(self, mesh: GpuMesh, instance_data: np.ndarray, vp_mat: Mat4, num_instances: int):
        if not self._instanced_initialized or mesh.instance_vbo is None:
            return
        self._stat_instances += num_instances
        self._stat_draws += 1
        prog = self._instanced_prog
        vp_f32 = vp_mat.to_f32()
        if "u_mvp" in prog:
            prog["u_mvp"].write(vp_f32.tobytes())
        data_size = num_instances * mesh.instance_stride
        if data_size > 0:
            mesh.instance_vbo.write(instance_data[:data_size].tobytes())
            self._stat_upload_bytes += data_size
            self._stat_upload_full += 1
        try:
            old_cull = bool(self._ctx.cull_face)
        except:
            old_cull = True
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.disable(moderngl.DEPTH_TEST)
        mesh.vao.render(moderngl.TRIANGLES, instances=num_instances)
        if old_cull:
            self._ctx.enable(moderngl.CULL_FACE)
        else:
            self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.DEPTH_TEST)

    def initialize_instanced_lines(self):
        if self._inst_line_initialized:
            return
        self._ensure_inst_line_prog()
        if self._inst_line_prog is None:
            return
        ctx = self._ctx
        prog = self._inst_line_prog
        self._box_inst_mesh = make_instance_line_vao(ctx, prog, make_unit_box_line_verts())
        self._sphere_inst_mesh = make_instance_line_vao(ctx, prog, make_unit_sphere_line_verts())
        self._rect_inst_mesh = make_instance_line_vao(ctx, prog, make_unit_rect_line_verts())
        self._circle_inst_mesh = make_instance_line_vao(ctx, prog, make_unit_circle_line_verts())
        self._capsule_inst_mesh = make_instance_line_vao(ctx, prog, make_unit_capsule_line_verts())
        self._inst_line_initialized = True

    def _ensure_inst_line_prog(self):
        if self._inst_line_prog is not None:
            return
        try:
            self._inst_line_prog = self._ctx.program(
                vertex_shader="""
#version 460 core
layout(location = 0) in vec3 a_unit_start;
layout(location = 1) in vec3 a_unit_end;
layout(location = 2) in float a_t;
layout(location = 3) in float a_side;
in vec4 i_row0;
in vec4 i_row1;
in vec4 i_row2;
in vec4 i_row3;
in vec4 i_color;
uniform mat4 u_mvp;
uniform float u_thickness_ndc_x;
uniform float u_thickness_ndc_y;
out vec4 v_color;
out float v_world_z;
void main() {
    mat4 model = mat4(i_row0, i_row1, i_row2, i_row3);
    vec4 ws_start = model * vec4(a_unit_start, 1.0);
    vec4 ws_end   = model * vec4(a_unit_end, 1.0);
    vec4 clip_start = u_mvp * ws_start;
    vec4 clip_end   = u_mvp * ws_end;
    vec4 clipPos = mix(clip_start, clip_end, a_t);
    vec2 dxy = clip_end.xy - clip_start.xy;
    float len = length(dxy);
    vec2 perp;
    if (len > 1e-6) {
        vec2 dir = dxy / len;
        perp = vec2(-dir.y, dir.x);
    } else {
        perp = vec2(1.0, 0.0);
    }
    clipPos.xy += perp * a_side * vec2(u_thickness_ndc_x, u_thickness_ndc_y) * clipPos.w;
    gl_Position = clipPos;
    v_color = i_color;
    v_world_z = (ws_start.z + ws_end.z) * 0.5;
}
""",
                fragment_shader="""
#version 460 core
in vec4 v_color;
in float v_world_z;
uniform vec3 u_camera_pos;
out vec4 fragColor;
void main() {
    float dist = abs(v_world_z - u_camera_pos.z);
    float fade = 1.0 - smoothstep(20.0, 80.0, dist);
    fragColor = vec4(v_color.rgb * fade, v_color.a * fade);
}
""",
            )
        except Exception:
            self._inst_line_prog = None

    def render_instanced_lines(self, shape_type: str, instance_data: np.ndarray,
                                num_instances: int, vp_mat: Mat4, fw: int, fh: int,
                                thickness_multiplier: float = 1.0,
                                cam_pos: Vec3 = Vec3(0, 0, 0)):
        if not self._inst_line_initialized or self._inst_line_prog is None:
            self.initialize_instanced_lines()
            if self._inst_line_prog is None:
                return
        mesh_map = {
            'box': self._box_inst_mesh,
            'sphere': self._sphere_inst_mesh,
            'rect': self._rect_inst_mesh,
            'circle': self._circle_inst_mesh,
            'capsule': self._capsule_inst_mesh,
        }
        mesh = mesh_map.get(shape_type)
        if mesh is None or mesh.instance_vbo is None or num_instances == 0:
            return
        self._stat_instances += num_instances
        self._stat_draws += 1
        prog = self._inst_line_prog
        vp_f32 = vp_mat.to_f32()
        if "u_mvp" in prog:
            prog["u_mvp"].write(vp_f32.tobytes())
        desired_pixels = max(1.0, float(self._line_width) * 1.5 * thickness_multiplier)
        ndc_x = desired_pixels / max(1.0, float(fw))
        ndc_y = desired_pixels / max(1.0, float(fh))
        if "u_thickness_ndc_x" in prog:
            prog["u_thickness_ndc_x"] = float(ndc_x)
        if "u_thickness_ndc_y" in prog:
            prog["u_thickness_ndc_y"] = float(ndc_y)
        if "u_camera_pos" in prog:
            prog["u_camera_pos"].write(np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32).tobytes())
        data_size = num_instances * mesh.instance_stride
        if data_size > 0:
            mesh.instance_vbo.write(instance_data[:data_size].tobytes())
            self._stat_upload_bytes += data_size
            self._stat_upload_full += 1
        try:
            old_cull = bool(self._ctx.cull_face)
        except:
            old_cull = True
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.disable(moderngl.DEPTH_TEST)
        mesh.vao.render(moderngl.TRIANGLES, vertices=mesh.vertex_count, instances=num_instances)
        if old_cull:
            self._ctx.enable(moderngl.CULL_FACE)
        else:
            self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.DEPTH_TEST)

    def _ensure_rawline_prog(self):
        if self._raw_line_prog is not None:
            return
        try:
            self._raw_line_prog = self._ctx.program(
                vertex_shader=RAWLINES_VERT,
                fragment_shader=RAWLINES_FRAG
            )
        except Exception:
            self._raw_line_prog = None

    def _build_rawline_buffers(self, capacity: int = 262144):
        cap = max(capacity, 262144)
        for buf_name in ('_raw_line_vao', '_raw_line_static_vbo',
                         '_raw_line_start_vbo', '_raw_line_end_vbo', '_raw_line_color_vbo'):
            b = getattr(self, buf_name, None)
            if b is not None:
                try:
                    b.release()
                except Exception:
                    pass
        static = np.empty((6, 2), dtype=np.float32)
        static[:, 0] = _STRIP_T
        static[:, 1] = _STRIP_S
        self._raw_line_static_vbo = self._ctx.buffer(static.tobytes())
        self._raw_line_start_vbo = self._ctx.buffer(reserve=cap * 12, dynamic=True)
        self._raw_line_end_vbo = self._ctx.buffer(reserve=cap * 12, dynamic=True)
        self._raw_line_color_vbo = self._ctx.buffer(reserve=cap * 16, dynamic=True)
        self._raw_line_vao = self._ctx.vertex_array(
            self._raw_line_prog,
            [
                (self._raw_line_static_vbo, "2f", "a_strip"),
                (self._raw_line_start_vbo, "3f /i", "i_start"),
                (self._raw_line_end_vbo, "3f /i", "i_end"),
                (self._raw_line_color_vbo, "4f /i", "i_color"),
            ]
        )
        self._fs_starts = np.empty((cap, 3), dtype=np.float32)
        self._fs_ends = np.empty((cap, 3), dtype=np.float32)
        self._fs_colors4 = np.empty((cap, 4), dtype=np.float32)
        self._rl_shadow_starts = np.zeros(cap, dtype=np.uint32)
        self._rl_shadow_ends = np.zeros(cap, dtype=np.uint32)
        self._rl_shadow_colors = np.zeros(cap, dtype=np.uint32)
        self._rl_shadow_n = 0
        self._rl_shadow_scalar[:] = 0
        self._raw_line_capacity = cap

    def _ensure_rawline_capacity(self, n: int):
        if n <= self._raw_line_capacity:
            return False
        new_cap = self._raw_line_capacity or 262144
        while new_cap < n:
            new_cap <<= 1
        self._build_rawline_buffers(new_cap)
        return True

    def render_raw_lines(self, starts: np.ndarray, ends: np.ndarray, colors: np.ndarray,
                         vp_mat: Mat4, fw: int, fh: int, desired_pixels: float,
                         dash_opts: Optional[dict] = None, dirty: bool = True):
        n = starts.shape[0]
        if n == 0:
            return
        self._ensure_rawline_prog()
        if self._raw_line_prog is None:
            self._render_lines_np(starts, ends, colors, vp_mat, fw, fh, desired_pixels, dash_opts)
            return
        grew = self._ensure_rawline_capacity(n)
        if grew:
            dirty = True
        self._stat_lines += n
        self._stat_draws += 1
        if dirty:
            if (starts.flags.c_contiguous and ends.flags.c_contiguous and colors.ndim == 2
                    and colors.shape[1] >= 4 and colors.flags.c_contiguous):
                s_arr = starts
                e_arr = ends
                c_arr = colors[:, :4]
            else:
                fs = self._fs_starts[:n]
                fs[:] = starts
                self._fs_ends[:n] = ends
                fc = self._fs_colors4[:n]
                if colors.shape[1] >= 4:
                    fc[:] = colors[:, :4]
                elif colors.shape[1] == 3:
                    fc[:, :3] = colors
                    fc[:, 3] = 1.0
                else:
                    fc[:] = 0.5
                s_arr = self._fs_starts[:n]
                e_arr = self._fs_ends[:n]
                c_arr = self._fs_colors4[:n]
            if n >= _RL_DIFF_THRESHOLD:
                self._rl_upload_diff(s_arr, e_arr, c_arr)
            else:
                self._raw_line_start_vbo.write(memoryview(s_arr))
                self._raw_line_end_vbo.write(memoryview(e_arr))
                self._raw_line_color_vbo.write(memoryview(c_arr))
                self._stat_upload_bytes += n * (12 + 12 + 16)
                self._stat_upload_full += 1
                self._rl_refresh_shadow(s_arr, e_arr, c_arr)
        self._draw_rawlines(n, vp_mat, fw, fh, desired_pixels, dash_opts)

    def _rl_refresh_shadow(self, s_arr: np.ndarray, e_arr: np.ndarray, c_arr: np.ndarray):
        n = s_arr.shape[0]
        vs = s_arr.view(np.uint32)
        ve = e_arr.view(np.uint32)
        vc = c_arr.view(np.uint32)
        self._rl_shadow_starts[:n] = np.bitwise_xor(np.bitwise_xor(vs[:, 0], vs[:, 1]), vs[:, 2])
        self._rl_shadow_ends[:n] = np.bitwise_xor(np.bitwise_xor(ve[:, 0], ve[:, 1]), ve[:, 2])
        self._rl_shadow_colors[:n] = np.bitwise_xor(np.bitwise_xor(np.bitwise_xor(vc[:, 0], vc[:, 1]), vc[:, 2]), vc[:, 3])
        self._rl_shadow_n = n
        self._rl_shadow_scalar[0] = np.bitwise_xor.reduce(vs.ravel())
        self._rl_shadow_scalar[1] = np.bitwise_xor.reduce(ve.ravel())
        self._rl_shadow_scalar[2] = np.bitwise_xor.reduce(vc.ravel())

    def _rl_upload_ranges(self, vbo, data: np.ndarray, mask: np.ndarray, stride: int):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return
        bounds = np.flatnonzero(np.diff(idx) > 1)
        rs = np.concatenate(([idx[0]], idx[bounds + 1]))
        re = np.concatenate((idx[bounds] + 1, [idx[-1] + 1]))
        if idx.size > 0:
            self._stat_upload_partial += 1
        for r0, r1 in zip(rs.tolist(), re.tolist()):
            vbo.write(memoryview(data[r0:r1]), offset=r0 * stride)
            self._stat_upload_bytes += (r1 - r0) * stride

    def _rl_upload_diff(self, s_arr: np.ndarray, e_arr: np.ndarray, c_arr: np.ndarray) -> bool:
        n = s_arr.shape[0]
        vs = s_arr.view(np.uint32)
        ve = e_arr.view(np.uint32)
        vc = c_arr.view(np.uint32)
        scalars = np.empty(3, dtype=np.uint32)
        scalars[0] = np.bitwise_xor.reduce(vs.ravel())
        scalars[1] = np.bitwise_xor.reduce(ve.ravel())
        scalars[2] = np.bitwise_xor.reduce(vc.ravel())
        if n == self._rl_shadow_n and (scalars == self._rl_shadow_scalar).all():
            return False
        grew = n != self._rl_shadow_n
        for arr, shadow_name, stride, s_idx, vbo in (
                (vs, '_rl_shadow_starts', 12, 0, self._raw_line_start_vbo),
                (ve, '_rl_shadow_ends', 12, 1, self._raw_line_end_vbo),
                (vc, '_rl_shadow_colors', 16, 2, self._raw_line_color_vbo)):
            shadow = getattr(self, shadow_name)
            if not (grew or scalars[s_idx] != self._rl_shadow_scalar[s_idx]):
                continue
            m = arr.shape[0]
            cs = np.bitwise_xor(np.bitwise_xor(arr[:, 0], arr[:, 1]), arr[:, 2])
            if arr.shape[1] > 3:
                cs = np.bitwise_xor(cs, arr[:, 3])
            mask = cs != shadow[:m]
            shadow[:m] = cs
            self._rl_upload_ranges(vbo, arr.view(np.float32), mask, stride)
        self._rl_shadow_n = n
        self._rl_shadow_scalar[:] = scalars
        return True

    def _draw_rawlines(self, n: int, vp_mat: Mat4, fw: int, fh: int, desired_pixels: float,
                       dash_opts: Optional[dict]):
        try:
            old_cull = bool(self._ctx.cull_face)
        except Exception:
            old_cull = True
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.disable(moderngl.DEPTH_TEST)
        prog = self._raw_line_prog
        vp_f32 = vp_mat.to_f32()
        if "u_mvp" in prog:
            prog["u_mvp"].write(vp_f32.tobytes())
        ndc_x = desired_pixels / max(1.0, float(fw))
        ndc_y = desired_pixels / max(1.0, float(fh))
        if "u_thickness_ndc_x" in prog:
            prog["u_thickness_ndc_x"] = float(ndc_x)
        if "u_thickness_ndc_y" in prog:
            prog["u_thickness_ndc_y"] = float(ndc_y)
        if dash_opts and "u_dash_enabled" in prog:
            prog["u_dash_enabled"] = True
            prog["u_dash_length"] = float(dash_opts.get('dash_length', 0.3))
            prog["u_gap_length"] = float(dash_opts.get('gap_length', 0.15))
            prog["u_dash_time"] = float(dash_opts.get('time', 0.0))
        elif "u_dash_enabled" in prog:
            prog["u_dash_enabled"] = False
        self._raw_line_vao.render(moderngl.TRIANGLES, vertices=6, instances=n)
        if old_cull:
            self._ctx.enable(moderngl.CULL_FACE)
        else:
            self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.DEPTH_TEST)

    def render_meshes(self, meshes: list[tuple], vp_mat: Mat4):
        if not self._solid_prog or not meshes:
            return
        for verts, indices, colors in meshes:
            if not verts or not indices or len(indices) < 3:
                continue
            self._stat_mesh_verts += len(verts)
            self._stat_draws += 1
            n = len(verts)
            v_data = np.empty((n, 7), dtype=np.float32)
            for i in range(n):
                v = verts[i]
                v_data[i, 0] = v.x; v_data[i, 1] = v.y; v_data[i, 2] = v.z
                c = colors[i] if i < len(colors) else [1, 1, 1, 1]
                v_data[i, 3] = c[0]; v_data[i, 4] = c[1]; v_data[i, 5] = c[2]
                v_data[i, 6] = c[3] if len(c) > 3 else 1.0
            idx_arr = np.array(indices, dtype=np.uint32)
            n_idx = len(idx_arr)
            if n > self._solid_vbo_cap or n_idx > self._solid_ibo_cap:
                self._build_solid_buffers(n, n_idx)
            self._solid_vbo.write(v_data.tobytes())
            self._solid_ibo.write(idx_arr.tobytes())
            self._stat_upload_bytes += len(v_data) * 28 + len(idx_arr) * 4
            self._stat_upload_full += 1
            self._solid_vao.render(moderngl.TRIANGLES, vertices=n_idx)
        self._ctx.disable(moderngl.BLEND)
        try:
            self._ctx.enable(moderngl.CULL_FACE)
        except:
            pass

    def render_mesh_np(self, v_data: np.ndarray, idx_arr: np.ndarray, vp_mat: Mat4):
        prog = self._solid_prog
        if not prog:
            return
        self._stat_mesh_verts += v_data.shape[0]
        self._stat_draws += 1
        vp_f32 = vp_mat.to_f32()
        if "u_mvp" in prog:
            prog["u_mvp"].write(vp_f32.tobytes())
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.BLEND)
        n_idx = len(idx_arr)
        n = v_data.shape[0]
        if n > self._solid_vbo_cap or n_idx > self._solid_ibo_cap:
            self._build_solid_buffers(n, n_idx)
        self._solid_vbo.write(v_data.tobytes())
        self._solid_ibo.write(idx_arr.tobytes())
        self._stat_upload_bytes += len(v_data) * 28 + len(idx_arr) * 4
        self._stat_upload_full += 1
        self._solid_vao.render(moderngl.TRIANGLES, vertices=n_idx)
        self._ctx.disable(moderngl.BLEND)
        try:
            self._ctx.enable(moderngl.CULL_FACE)
        except:
            pass

    def render_wireframe_box(self, center: Vec3, size: Vec3, color: list[float], vp_mat: Mat4):
        h = Vec3(size.x * 0.5, size.y * 0.5, size.z * 0.5)
        corners = [
            Vec3(center.x - h.x, center.y - h.y, center.z - h.z),
            Vec3(center.x + h.x, center.y - h.y, center.z - h.z),
            Vec3(center.x + h.x, center.y + h.y, center.z - h.z),
            Vec3(center.x - h.x, center.y + h.y, center.z - h.z),
            Vec3(center.x - h.x, center.y - h.y, center.z + h.z),
            Vec3(center.x + h.x, center.y - h.y, center.z + h.z),
            Vec3(center.x + h.x, center.y + h.y, center.z + h.z),
            Vec3(center.x - h.x, center.y + h.y, center.z + h.z),
        ]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        lines = [(corners[a], corners[b], color) for a, b in edges]
        self.render_lines(lines, vp_mat)

    def release(self):
        for buf_name in ('_fatline_vbo_start', '_fatline_vbo_end', '_fatline_vbo_t', '_fatline_vbo_side', '_fatline_vbo_color',
                         '_solid_vbo', '_solid_ibo'):
            b = getattr(self, buf_name, None)
            if b is not None:
                try: b.release()
                except: pass
        for vao_name in ('_fatline_vao', '_solid_vao'):
            v = getattr(self, vao_name, None)
            if v is not None:
                try: v.release()
                except: pass
        for mesh_name in ('_cone_mesh', '_cylinder_mesh', '_cube_mesh', '_quad_mesh', '_circle_mesh'):
            m = getattr(self, mesh_name, None)
            if m is not None:
                if m.vao:
                    try: m.vao.release()
                    except: pass
                if m.vbo:
                    try: m.vbo.release()
                    except: pass
                if m.ibo:
                    try: m.ibo.release()
                    except: pass
                if m.instance_vbo:
                    try: m.instance_vbo.release()
                    except: pass
        if self._instanced_prog:
            try: self._instanced_prog.release()
            except: pass
        for mesh_name in ('_box_inst_mesh', '_sphere_inst_mesh', '_rect_inst_mesh', '_circle_inst_mesh', '_capsule_inst_mesh'):
            m = getattr(self, mesh_name, None)
            if m is not None:
                if m.vao:
                    try: m.vao.release()
                    except: pass
                if m.vbo:
                    try: m.vbo.release()
                    except: pass
                if m.instance_vbo:
                    try: m.instance_vbo.release()
                    except: pass
        if self._inst_line_prog:
            try: self._inst_line_prog.release()
            except: pass
