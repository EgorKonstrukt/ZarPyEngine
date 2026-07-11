from __future__ import annotations
import numpy as np
import moderngl
from typing import Optional
from core.math.math3d import Mat4
from core.renderer.culling import extract_frustum_planes


WORLD_MATRIX_BINDING = 4
INDEX_BINDING = 5


def _supports_compute(ctx: moderngl.Context) -> bool:
    try:
        ctx.compute_shader('')
        return True
    except Exception:
        return False


class GpuStorage:
    """Manages persistent GPU storage for SSBO-based instancing.

    - World matrix SSBO (binding 4): ALL world matrices, updated when dirty
    - Optional compute culling resources
    """

    __slots__ = (
        '_ctx', '_world_mat_ssbo', '_culling',
        '_capacity', '_version', '_last_upload_version',
    )

    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._world_mat_ssbo: Optional[moderngl.Buffer] = None
        self._culling: Optional[GpuCulling] = None
        self._capacity: int = 0
        self._version: int = 0
        self._last_upload_version: int = -1

    def ensure_capacity(self, n: int):
        if n <= self._capacity:
            return
        self._release_ssbo()
        self._capacity = n
        if n == 0:
            return
        mat_size = n * 64
        self._world_mat_ssbo = self._ctx.buffer(reserve=mat_size)

    def upload_world_matrices(self, matrices: list[Mat4],
                              bounding_radii: np.ndarray, version: int):
        if version == self._last_upload_version or len(matrices) == 0:
            return
        self._last_upload_version = version
        self.ensure_capacity(len(matrices))
        self._world_mat_ssbo.write(Mat4.batch_to_f32(matrices).tobytes())
        if self._culling and len(bounding_radii) == len(matrices):
            self._culling.upload_bounding(matrices, bounding_radii)

    def get_world_matrix_ssbo(self) -> Optional[moderngl.Buffer]:
        return self._world_mat_ssbo

    def bind_world_matrices(self):
        if self._world_mat_ssbo:
            self._world_mat_ssbo.bind_to_storage_buffer(WORLD_MATRIX_BINDING)

    def get_or_create_culling(self) -> Optional[GpuCulling]:
        if self._culling is None and _supports_compute(self._ctx):
            self._culling = GpuCulling(self._ctx)
        return self._culling

    def _release_ssbo(self):
        if self._world_mat_ssbo:
            try:
                self._world_mat_ssbo.release()
            except Exception:
                pass
            self._world_mat_ssbo = None

    def release(self):
        self._release_ssbo()
        if self._culling:
            self._culling.release()
            self._culling = None
        self._capacity = 0


class GpuCulling:
    """Compute-shader-based frustum culling.

    Uses 4 SSBOs:
      0: WorldMatrices (readonly) - mat4[]
      1: BoundingData   (readonly) - vec4[] (xyz=center, w=radius)
      2: InstanceVBO    (writeonly) - mat4[] (output, used as instance VBO)
      3: CounterBuf     - uint count
    """

    __slots__ = (
        '_ctx', '_compute_shader', '_bounding_ssbo',
        '_instance_output_ssbo', '_counter_ssbo',
        '_capacity', '_count',
    )

    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._compute_shader: Optional[moderngl.ComputeShader] = None
        self._bounding_ssbo: Optional[moderngl.Buffer] = None
        self._instance_output_ssbo: Optional[moderngl.Buffer] = None
        self._counter_ssbo: Optional[moderngl.Buffer] = None
        self._capacity: int = 0
        self._count: int = 0

    def ensure_resources(self, max_instances: int):
        if max_instances <= self._capacity:
            return
        self.release()
        self._capacity = max_instances
        sphere_size = max_instances * 16
        output_size = max_instances * 64
        self._bounding_ssbo = self._ctx.buffer(reserve=sphere_size)
        self._instance_output_ssbo = self._ctx.buffer(reserve=output_size)
        self._counter_ssbo = self._ctx.buffer(reserve=4)
        from core.renderer.mesh_data import SHADER_DIR
        import os
        path = os.path.join(SHADER_DIR, "cull.comp")
        if os.path.exists(path):
            with open(path) as f:
                src = f.read()
            try:
                self._compute_shader = self._ctx.compute_shader(src)
            except Exception:
                self._compute_shader = None

    def _build_spheres(self, matrices: list[Mat4], bounding_radii: np.ndarray) -> np.ndarray:
        from core._render_utils import compute_bounding_spheres
        return compute_bounding_spheres(matrices, bounding_radii)

    def upload_bounding(self, matrices: list[Mat4], bounding_radii: np.ndarray):
        n = len(matrices)
        self._count = n
        self.ensure_resources(n)
        if n == 0:
            return
        spheres = self._build_spheres(matrices, bounding_radii)
        self._bounding_ssbo.write(spheres.tobytes())

    def upload_world_matrices(self, matrices: list[Mat4],
                              bounding_radii: np.ndarray):
        n = len(matrices)
        self._count = n
        self.ensure_resources(n)
        if n == 0:
            return
        spheres = self._build_spheres(matrices, bounding_radii)
        self._bounding_ssbo.write(spheres.tobytes())

    def cull(self, view_mat: Mat4, proj_mat: Mat4,
             world_mat_ssbo: moderngl.Buffer) -> bool:
        if not self._compute_shader:
            return False
        n = self._count
        if n <= 0:
            return False
        vp = proj_mat._d.T @ view_mat._d.T
        planes = extract_frustum_planes(vp)

        self._counter_ssbo.clear()
        self._instance_output_ssbo.clear()

        cs = self._compute_shader
        cs["u_frustum_planes[0]"].write(planes.astype(np.float32).tobytes())
        cs["u_total"].value = n

        world_mat_ssbo.bind_to_storage_buffer(0)
        self._bounding_ssbo.bind_to_storage_buffer(1)
        self._instance_output_ssbo.bind_to_storage_buffer(2)
        self._counter_ssbo.bind_to_storage_buffer(3)

        groups = (n + 63) // 64
        cs.run(groups, 1, 1)
        self._ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)
        return True

    def get_instance_vbo(self) -> Optional[moderngl.Buffer]:
        return self._instance_output_ssbo

    def read_visible_count(self) -> int:
        if self._counter_ssbo is None:
            return -1
        data = self._counter_ssbo.read(0, 4)
        return int(np.frombuffer(data, dtype=np.uint32)[0])

    def release(self):
        for buf in (self._bounding_ssbo, self._instance_output_ssbo,
                    self._counter_ssbo):
            if buf:
                try:
                    buf.release()
                except Exception:
                    pass
        self._bounding_ssbo = None
        self._instance_output_ssbo = None
        self._counter_ssbo = None
        if self._compute_shader:
            try:
                self._compute_shader.release()
            except Exception:
                pass
            self._compute_shader = None
        self._capacity = 0
        self._count = 0
