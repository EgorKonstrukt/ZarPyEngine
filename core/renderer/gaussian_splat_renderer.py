# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
import moderngl
from typing import Optional
from core.assets.ply_loader import load_ply_gaussian_splat, GaussianSplatData
from core.renderer.mesh_data import read_shader


_SPLAT_DTYPE = np.dtype([
    ("pos_x", np.float32), ("pos_y", np.float32), ("pos_z", np.float32),
    ("sh_dc_0", np.float32), ("sh_dc_1", np.float32), ("sh_dc_2", np.float32),
    ("sh_rest", np.float32, (45,)),
    ("opacity", np.float32),
    ("scale_0", np.float32), ("scale_1", np.float32), ("scale_2", np.float32),
    ("quat_x", np.float32), ("quat_y", np.float32), ("quat_z", np.float32), ("quat_w", np.float32),
])

_SPLAT_STRUCT_SIZE = _SPLAT_DTYPE.itemsize


class GaussianSplatRenderer:
    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._prog: Optional[moderngl.Program] = None
        self._ssbo: Optional[moderngl.Buffer] = None
        self._idx_ssbo: Optional[moderngl.Buffer] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._uploaded_path: Optional[str] = None
        self._uploaded_n: int = 0
        self._loaded: dict[str, GaussianSplatData] = {}
        self._gpu_data: dict[str, np.ndarray] = {}
        self._sort_cache: dict[str, tuple[bytes, np.ndarray]] = {}
        self._init_shaders()

    def _init_shaders(self):
        try:
            vert_src = read_shader("gaussian_splat.vert")
            frag_src = read_shader("gaussian_splat.frag")
            self._prog = self._ctx.program(
                vertex_shader=vert_src,
                fragment_shader=frag_src,
            )
            self._vao = self._ctx.vertex_array(self._prog, [])
        except Exception:
            self._prog = None
            self._vao = None

    def _ensure_buffers(self, num_splats: int):
        needed = max(1, num_splats) * _SPLAT_STRUCT_SIZE
        if self._ssbo is None or self._ssbo.size < needed:
            if self._ssbo:
                self._ssbo.release()
            self._ssbo = self._ctx.buffer(reserve=needed)
            self._uploaded_path = None

        idx_needed = max(1, num_splats) * 4
        if self._idx_ssbo is None or self._idx_ssbo.size < idx_needed:
            if self._idx_ssbo:
                self._idx_ssbo.release()
            self._idx_ssbo = self._ctx.buffer(reserve=idx_needed)

    def load_data(self, path: str) -> bool:
        if path in self._loaded:
            return True
        data = load_ply_gaussian_splat(path)
        if data is None:
            return False
        self._loaded[path] = data
        gpu = self._pack_for_gpu(data)
        self._gpu_data[path] = gpu
        self._uploaded_path = None
        self._sort_cache.pop(path, None)
        return True

    def _pack_for_gpu(self, data: GaussianSplatData) -> np.ndarray:
        n = data.num_splats
        num_rest = data.sh_coeffs.shape[1] - 3
        rest_padded = np.zeros((n, 45), dtype=np.float32)
        rest_padded[:, :num_rest] = data.sh_coeffs[:, 3:3 + num_rest]

        gpu = np.zeros(n, dtype=_SPLAT_DTYPE)
        gpu["pos_x"] = data.positions[:, 0]
        gpu["pos_y"] = data.positions[:, 1]
        gpu["pos_z"] = data.positions[:, 2]
        gpu["sh_dc_0"] = data.sh_coeffs[:, 0]
        gpu["sh_dc_1"] = data.sh_coeffs[:, 1]
        gpu["sh_dc_2"] = data.sh_coeffs[:, 2]
        gpu["sh_rest"] = rest_padded
        gpu["opacity"] = data.opacity
        gpu["scale_0"] = data.scales[:, 0]
        gpu["scale_1"] = data.scales[:, 1]
        gpu["scale_2"] = data.scales[:, 2]
        gpu["quat_x"] = data.quaternions[:, 0]
        gpu["quat_y"] = data.quaternions[:, 1]
        gpu["quat_z"] = data.quaternions[:, 2]
        gpu["quat_w"] = data.quaternions[:, 3]
        return gpu

    def _update_sort_order(self, path: str, gpu: np.ndarray, model_f32: np.ndarray, view_f32: np.ndarray) -> np.ndarray:
        key = model_f32.tobytes() + view_f32.tobytes()
        cached = self._sort_cache.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]

        n = len(gpu)
        mat_model = model_f32.reshape(4, 4)
        mat_view = view_f32.reshape(4, 4)
        pos_h = np.ones((n, 4), dtype=np.float32)
        pos_h[:, 0] = gpu["pos_x"]
        pos_h[:, 1] = gpu["pos_y"]
        pos_h[:, 2] = gpu["pos_z"]
        world = pos_h @ mat_model
        view_space = world @ mat_view
        order = np.argsort(view_space[:, 2], kind="stable").astype(np.uint32)
        self._sort_cache[path] = (key, order)
        return order

    def render(self, path: str, model_matrix, view_mat, proj_mat, cam_pos, viewport_w, viewport_h,
               opacity_threshold=0.005, sh_degree=3, max_screen_size=32.0):
        if not self._prog or not self._vao:
            return
        if path not in self._gpu_data:
            if not self.load_data(path):
                return

        gpu = self._gpu_data.get(path)
        if gpu is None or len(gpu) == 0:
            return

        n = len(gpu)
        self._ensure_buffers(n)

        if self._uploaded_path != path or self._uploaded_n != n:
            self._ssbo.write(gpu.tobytes())
            self._uploaded_path = path
            self._uploaded_n = n

        prog = self._prog
        self._ssbo.bind_to_storage_buffer(0)

        model_f32 = np.array(model_matrix.to_f32(), dtype=np.float32)
        view_f32 = np.array(view_mat.to_f32(), dtype=np.float32)
        proj_f32 = np.array(proj_mat.to_f32(), dtype=np.float32)

        order = self._update_sort_order(path, gpu, model_f32, view_f32)
        self._idx_ssbo.write(order.tobytes())
        self._idx_ssbo.bind_to_storage_buffer(1)

        if "u_model" in prog:
            prog["u_model"].write(model_f32.tobytes())
        if "u_view" in prog:
            prog["u_view"].write(view_f32.tobytes())
        if "u_proj" in prog:
            prog["u_proj"].write(proj_f32.tobytes())
        if "u_viewport" in prog:
            prog["u_viewport"].write(np.array([viewport_w, viewport_h], dtype=np.float32).tobytes())
        if "u_camera_pos" in prog:
            prog["u_camera_pos"].write(np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32).tobytes())
        if "u_sh_degree" in prog:
            prog["u_sh_degree"].value = int(sh_degree)
        if "u_opacity_threshold" in prog:
            prog["u_opacity_threshold"].value = float(opacity_threshold)
        if "u_max_screen_size" in prog:
            prog["u_max_screen_size"].value = float(max_screen_size)

        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self._ctx.depth_mask = False

        try:
            self._vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)
        except Exception as e:
            from core.foundation.logger import Logger
            Logger.error(f"Gaussian Splat render error: {e}")

        self._ctx.enable(moderngl.CULL_FACE)
        self._ctx.depth_mask = True
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def release(self):
        if self._vao:
            self._vao.release()
        if self._ssbo:
            self._ssbo.release()
        if self._idx_ssbo:
            self._idx_ssbo.release()
        if self._prog:
            self._prog.release()