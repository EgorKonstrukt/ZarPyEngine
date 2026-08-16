from __future__ import annotations
import numpy as np
import moderngl
from typing import Optional
from core.maths.math3d import Mat4
from core.renderer.culling import extract_frustum_planes


WORLD_MATRIX_BINDING = 4
INDEX_BINDING = 5


def _supports_compute(ctx: moderngl.Context) -> bool:
    try:
        ctx.compute_shader(
            "#version 460 core\n"
            "layout(local_size_x = 1) in;\n"
            "void main() {}\n"
        )
        return True
    except Exception:
        return False


class GpuStorage:
    """Manages persistent GPU storage for SSBO-based instancing.

    - World matrix SSBO (binding 4): ALL world matrices, updated when dirty
    - Optional compute culling resources
    - GPU-driven rendering flag (opt-in via ``set_gpu_driven``)
    """

    __slots__ = (
        '_ctx', '_world_mat_ssbo', '_culling',
        '_capacity', '_last_upload_version',
        '_gpu_driven',
    )

    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._world_mat_ssbo: Optional[moderngl.Buffer] = None
        self._culling: Optional[GpuCulling] = None
        self._capacity: int = 0
        self._last_upload_version: int = -1
        self._gpu_driven: bool = False

    def set_gpu_driven(self, enabled: bool):
        self._gpu_driven = bool(enabled)
        if self._gpu_driven:
            self.get_or_create_culling()

    def is_gpu_driven(self) -> bool:
        return self._gpu_driven and self._culling is not None and self._culling.is_ready()

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
        if len(matrices) == 0:
            return
        self._last_upload_version = version
        self.ensure_capacity(len(matrices))
        try:
            from core._render_utils import batch_mat4_to_f32_flat
            data = batch_mat4_to_f32_flat(matrices).tobytes()
        except ImportError:
            data = Mat4.batch_to_f32(matrices).tobytes()
        self._world_mat_ssbo.write(data)
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

    def cull_and_get_indices(self, view_f32, proj_f32,
                             world_mat_ssbo: moderngl.Buffer,
                             version: int) -> Optional[moderngl.Buffer]:
        """Run the compute culler and return the compacted index SSBO.

        Returns None only when culling is unavailable. When nothing is
        visible the (empty) index SSBO is still returned so the caller can
        detect a fully-culled group via ``last_visible_count``.
        """
        culling = self.get_or_create_culling()
        if culling is None or not culling.is_ready():
            return None
        if not culling.cull_f32(view_f32, proj_f32, world_mat_ssbo):
            return None
        return culling.get_instance_indices_ssbo()

    @property
    def last_visible_count(self) -> int:
        if self._culling is None:
            return 0
        return self._culling.read_visible_count()

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
    """Compute-shader-based frustum culling + index compaction.

    Uses 4 SSBOs:
      0: WorldMatrices (readonly) - mat4[]
      1: BoundingData   (readonly) - vec4[] (xyz=center, w=radius)
      2: OutIndices     (writeonly) - uint[] compacted instance indices
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

    def is_ready(self) -> bool:
        return self._compute_shader is not None

    def ensure_resources(self, max_instances: int):
        if max_instances <= self._capacity:
            return
        self.release()
        self._capacity = max_instances
        self._count = max_instances
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
        return self._cull_impl(planes, n, world_mat_ssbo)

    def cull_f32(self, view_f32, proj_f32,
                 world_mat_ssbo: moderngl.Buffer) -> bool:
        if not self._compute_shader:
            return False
        n = self._count
        if n <= 0:
            return False
        from core.renderer.batcher import _extract_frustum_planes_f32
        planes = _extract_frustum_planes_f32(view_f32, proj_f32)
        return self._cull_impl(planes, n, world_mat_ssbo)

    def _cull_impl(self, planes, n: int, world_mat_ssbo: moderngl.Buffer) -> bool:
        if not self._compute_shader:
            return False
        self._counter_ssbo.clear()
        self._instance_output_ssbo.clear()

        cs = self._compute_shader
        try:
            if "u_frustum_planes" in cs._members:
                cs["u_frustum_planes"].write(planes.astype(np.float32).tobytes())
            elif "u_frustum_planes[0]" in cs._members:
                cs["u_frustum_planes[0]"].write(planes.astype(np.float32).tobytes())
            else:
                return False
            cs["u_total"].value = n
        except Exception as e:
            print(f"[gpu_culling] cull failed, falling back to CPU: {e!r}")
            return False

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

    def get_instance_indices_ssbo(self) -> Optional[moderngl.Buffer]:
        return self._instance_output_ssbo

    def read_visible_count(self) -> int:
        if self._counter_ssbo is None:
            return -1
        data = self._counter_ssbo.read(4, 0)
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
