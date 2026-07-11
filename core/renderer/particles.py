# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import time
import numpy as np
import moderngl
from typing import Optional, Any
from core.math.math3d import Vec3
from core.components.rendering.particles.particle_force_field import FORCE_FIELD_DTYPE, FORCE_FIELD_SSBO_SIZE, MAX_FORCE_FIELDS


class ParticleRenderer:
    def __init__(self, ctx: moderngl.Context, particle_prog: moderngl.Program):
        self._ctx = ctx
        self._particle_prog = particle_prog
        self._compute_prog: Optional[moderngl.ComputeShader] = None
        self._particle_ssbo: Optional[moderngl.Buffer] = None
        self._dead_ssbo: Optional[moderngl.Buffer] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._max_particles: int = 0
        self._textures: dict = {}
        self._ff_ssbo: Optional[moderngl.Buffer] = None
        self._vao = self._ctx.vertex_array(self._particle_prog, [])
        self._last_particle_time: float = time.perf_counter()
        self._render_dt: float = 0.016
        self._last_frame: int = -1

    def load_compute_shader(self, path: str) -> bool:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            return False
        try:
            with open(abs_path) as f:
                src = f.read()
            glsl_start = src.find("GLSLPROGRAM")
            glsl_end = src.find("ENDGLSL", glsl_start)
            if glsl_start < 0 or glsl_end < 0:
                return False
            source = src[glsl_start + len("GLSLPROGRAM"):glsl_end].strip()
            if self._compute_prog:
                self._compute_prog.release()
            self._compute_prog = self._ctx.compute_shader(source)
            return True
        except Exception as e:
            from core.foundation.logger import Logger
            Logger.error(f"Particle compute shader compile error: {e}")
            return False

    def _ensure_buffers(self, max_particles: int):
        needed = max_particles * 112
        if self._particle_ssbo is None or self._particle_ssbo.size < needed:
            if self._particle_ssbo:
                self._particle_ssbo.release()
            if self._dead_ssbo:
                self._dead_ssbo.release()
            self._particle_ssbo = self._ctx.buffer(data=b'\x00' * needed)
            dead_size = 4 + max_particles * 4
            self._dead_ssbo = self._ctx.buffer(data=b'\x00' * dead_size)
            self._max_particles = max_particles

    def _ensure_ff_ssbo(self):
        if self._ff_ssbo is None or self._ff_ssbo.size < FORCE_FIELD_SSBO_SIZE:
            if self._ff_ssbo:
                self._ff_ssbo.release()
            self._ff_ssbo = self._ctx.buffer(data=b'\x00' * FORCE_FIELD_SSBO_SIZE)

    def upload_force_fields(self, ff_array: np.ndarray):
        self._ensure_ff_ssbo()
        if self._ff_ssbo is None:
            return
        data = ff_array.tobytes()
        pad = FORCE_FIELD_SSBO_SIZE - len(data)
        if pad > 0:
            data += b'\x00' * pad
        self._ff_ssbo.write(data)

    def upload_all(self, particles_np: np.ndarray):
        if self._particle_ssbo is None:
            return
        self._particle_ssbo.write(particles_np.tobytes())

    def readback_all(self, particles_np: np.ndarray):
        if self._particle_ssbo is None:
            return
        data = self._particle_ssbo.read()
        arr = np.frombuffer(data, dtype=particles_np.dtype)
        particles_np[:] = arr[:len(particles_np)]

    def dispatch(self, params: dict):
        if not self._compute_prog or not self._particle_ssbo:
            return
        n = params.get('num_particles', 0)
        if n == 0:
            return
        self._ensure_buffers(n)
        self._dead_ssbo.write(np.zeros(1, dtype=np.uint32).tobytes())
        self._particle_ssbo.bind_to_storage_buffer(0)
        self._dead_ssbo.bind_to_storage_buffer(1)
        self._ensure_ff_ssbo()
        self._ff_ssbo.bind_to_storage_buffer(2)
        prog = self._compute_prog
        prog['u_dt'] = self._render_dt
        prog['u_gravity'] = params.get('gravity', 0.0)
        prog['u_simulation_space'] = 1 if params.get('simulation_space') == 'local' else 0
        if 'u_num_force_fields' in prog:
            prog['u_num_force_fields'] = params.get('num_force_fields', 0)
        if 'u_emitter_delta' in prog:
            d = params.get('emitter_delta', (0, 0, 0))
            prog['u_emitter_delta'].write(np.array(d, dtype=np.float32).tobytes())
        prog['u_num_particles'] = n
        prog['u_size_enabled'] = 1 if params.get('size_enabled') else 0
        if params.get('size_enabled') and 'u_size_curve' in prog:
            prog['u_size_curve'].write(np.array(params['size_curve'], dtype=np.float32).tobytes())
        prog['u_color_enabled'] = 1 if params.get('color_enabled') else 0
        if params.get('color_enabled'):
            for key, uname in [('color_curve_r', 'u_color_curve_r'), ('color_curve_g', 'u_color_curve_g'),
                               ('color_curve_b', 'u_color_curve_b'), ('alpha_curve', 'u_alpha_curve')]:
                if uname in prog:
                    prog[uname].write(np.array(params[key], dtype=np.float32).tobytes())
        prog['u_rotation_enabled'] = 1 if params.get('rotation_enabled') else 0
        if params.get('rotation_enabled') and 'u_angular_velocity' in prog:
            prog['u_angular_velocity'] = params.get('angular_velocity', 0.0)
        prog['u_velocity_enabled'] = 1 if params.get('velocity_enabled') else 0
        if params.get('velocity_enabled'):
            if 'u_vel_linear' in prog:
                prog['u_vel_linear'].write(np.array(params.get('vel_linear', (0, 0, 0)), dtype=np.float32).tobytes())
            if 'u_vel_orbital' in prog:
                prog['u_vel_orbital'].write(np.array(params.get('vel_orbital', (0, 0, 0)), dtype=np.float32).tobytes())
        groups = (n + 63) // 64
        prog.run(groups)

    def read_dead_list(self) -> np.ndarray:
        if self._dead_ssbo is None:
            return np.zeros(0, dtype=np.uint32)
        data = self._dead_ssbo.read()
        dead_count = int(np.frombuffer(data[:4], dtype=np.uint32)[0])
        if dead_count == 0:
            return np.zeros(0, dtype=np.uint32)
        return np.frombuffer(data[4:4 + dead_count * 4], dtype=np.uint32).copy()

    def load_texture(self, path: str) -> Optional[Any]:
        if not path:
            return None
        if path in self._textures:
            return self._textures[path]
        abs_path = path
        if not __import__('os').path.isabs(path):
            abs_path = __import__('os').path.join(__import__('os').getcwd(), path)
            if not __import__('os').path.exists(abs_path):
                alt = __import__('os').path.join("assets", path)
                if __import__('os').path.exists(alt):
                    abs_path = alt
        if not __import__('os').path.exists(abs_path):
            return None
        try:
            from PIL import Image
            img = Image.open(abs_path).convert("RGBA")
            tex = self._ctx.texture(img.size, 4, img.tobytes())
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.repeat_x = True
            tex.repeat_y = True
            self._textures[path] = tex
            return tex
        except Exception:
            return None

    def begin_frame(self, view_mat, proj_mat):
        if not self._particle_prog or not self._vao:
            return
        prog = self._particle_prog
        vp_mat = view_mat * proj_mat
        cam_right = Vec3(float(view_mat._d[0, 0]), float(view_mat._d[1, 0]), float(view_mat._d[2, 0]))
        cam_up = Vec3(float(view_mat._d[0, 1]), float(view_mat._d[1, 1]), float(view_mat._d[2, 1]))
        right_arr = np.array([cam_right.x, cam_right.y, cam_right.z], dtype=np.float32)
        up_arr = np.array([cam_up.x, cam_up.y, cam_up.z], dtype=np.float32)
        vp_f32 = vp_mat.to_f32()
        if "u_view_proj" in prog:
            prog["u_view_proj"].write(vp_f32.tobytes())
        if "u_camera_right" in prog:
            prog["u_camera_right"].write(right_arr.tobytes())
        if "u_camera_up" in prog:
            prog["u_camera_up"].write(up_arr.tobytes())
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._ctx.depth_mask = False

    def end_frame(self):
        self._ctx.enable(moderngl.CULL_FACE)
        self._ctx.depth_mask = True

    def render_single(self, ps):
        if not self._particle_prog or not self._vao:
            return
        n = len(ps._particles) if ps._particles is not None else 0
        if n == 0:
            return
        prog = self._particle_prog
        self._particle_ssbo.bind_to_storage_buffer(0)
        tex_path = ps.texture_path
        particle_tex = self.load_texture(tex_path) if tex_path else None
        if "u_use_texture" in prog:
            prog["u_use_texture"].value = 1 if particle_tex else 0
        if "u_albedo" in prog:
            prog["u_albedo"].write(np.array([1, 1, 1, 1], dtype=np.float32).tobytes())
        if particle_tex and "u_texture" in prog:
            particle_tex.use(0)
            prog["u_texture"].value = 0
        try:
            self._vao.render(moderngl.TRIANGLES, vertices=n * 6)
        except Exception as e:
            from core.foundation.logger import Logger
            Logger.error(f"Particle render error: {e}")

    def release(self):
        if self._vao:
            self._vao.release()
        if self._particle_ssbo:
            self._particle_ssbo.release()
        if self._dead_ssbo:
            self._dead_ssbo.release()
        if self._compute_prog:
            self._compute_prog.release()
        if self._ff_ssbo:
            self._ff_ssbo.release()
        for tex in self._textures.values():
            try:
                tex.release()
            except Exception:
                pass
