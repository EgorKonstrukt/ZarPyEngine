from __future__ import annotations
import os
import time
import numpy as np
import moderngl
from typing import Any, Optional
from collections import defaultdict, OrderedDict
from core.maths.math3d import Mat4

_INSTANCE_ATTRS = ("in_model0", "in_model1", "in_model2", "in_model3")

from core.renderer.gpu_culling import WORLD_MATRIX_BINDING, INDEX_BINDING

_INITIAL_INST_VBO_CAPACITY = 4096
_MAX_VAO_CACHE = 512


def resolve_normal_matrix(cache: dict, ent_id: int, model_d) -> np.ndarray:
    try:
        key = model_d[:3, :3].tobytes()
    except Exception:
        key = None
    if key is not None:
        cached = cache.get(ent_id)
        if cached is not None:
            old_key, nm = cached
            if old_key == key:
                return nm
    from core.math_helpers import mat4_normal_matrix
    try:
        nm = mat4_normal_matrix(model_d)
    except Exception:
        nm = np.eye(3, dtype=np.float32).T
    cache[ent_id] = (key, nm)
    return nm


def _supports_instancing(prog: moderngl.Program) -> bool:
    try:
        locs = prog._attribute_locations
        for a in _INSTANCE_ATTRS:
            if locs.get(a, -1) < 0:
                return False
        return True
    except Exception:
        return False


def _make_instanced_vao(ctx: moderngl.Context, prog: moderngl.Program,
                        mesh, instance_vbo: moderngl.Buffer) -> moderngl.VertexArray:
    vbo = getattr(mesh, '_vbo', None)
    ibo = getattr(mesh, '_ibo', None)
    if vbo is None:
        n_verts = len(mesh.vertices) // 3 if len(mesh.vertices) > 0 else 0
        data = np.zeros((n_verts, 8), dtype=np.float32)
        data[:, 0:3] = mesh.vertices.reshape(-1, 3)
        if len(mesh.normals) == len(mesh.vertices):
            data[:, 3:6] = mesh.normals.reshape(-1, 3)
        if len(mesh.uvs) * 3 == len(mesh.vertices) * 2:
            data[:, 6:8] = mesh.uvs.reshape(-1, 2)
        vbo = ctx.buffer(data.tobytes())
    content = [
        (vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv'),
    ]
    if "in_color" in prog:
        n_verts = len(mesh.vertices) // 3 if len(mesh.vertices) > 0 else vbo.size // 32
        if n_verts > 0:
            col = ctx.buffer(np.full((n_verts, 4), 1.0, dtype=np.float32).tobytes())
            content.append((col, '4f', 'in_color'))
    if _supports_instancing(prog):
        content.append((instance_vbo, '4f 4f 4f 4f /i',
                        'in_model0', 'in_model1', 'in_model2', 'in_model3'))
    if ibo is not None:
        return ctx.vertex_array(prog, content, ibo)
    return ctx.vertex_array(prog, content)


def _extract_frustum_planes_f32(view_f32, proj_f32):
    view_4 = view_f32.reshape(4, 4).T
    proj_4 = proj_f32.reshape(4, 4).T
    vp = np.dot(proj_4, view_4)
    planes = np.empty((6, 4), dtype=np.float32)
    planes[0] = vp[3] + vp[0]
    planes[1] = vp[3] - vp[0]
    planes[2] = vp[3] + vp[1]
    planes[3] = vp[3] - vp[1]
    planes[4] = vp[3] + vp[2]
    planes[5] = vp[3] - vp[2]
    norms = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    planes /= norms
    return planes


def _frustum_cull_instances(group, planes, mesh_radius):
    n = len(group)
    if n == 0:
        return group
    centers = np.empty((n, 3), dtype=np.float64)
    radii = np.empty(n, dtype=np.float64)
    for i, item in enumerate(group):
        wm = item[6]
        d = wm._d
        centers[i, 0] = d[3, 0]
        centers[i, 1] = d[3, 1]
        centers[i, 2] = d[3, 2]
        sx = (d[0, 0] * d[0, 0] + d[1, 0] * d[1, 0] + d[2, 0] * d[2, 0]) ** 0.5
        sy = (d[0, 1] * d[0, 1] + d[1, 1] * d[1, 1] + d[2, 1] * d[2, 1]) ** 0.5
        sz = (d[0, 2] * d[0, 2] + d[1, 2] * d[1, 2] + d[2, 2] * d[2, 2]) ** 0.5
        radii[i] = max(sx, sy, sz) * mesh_radius
    distances = planes[:, :3] @ centers.T + planes[:, 3, None]
    visible_mask = np.all(distances > -radii[None, :], axis=0)
    if np.all(visible_mask):
        return group
    indices = np.where(visible_mask)[0]
    return [group[i] for i in indices]


class RenderBatcher:
    """Groups renderables by mesh+material+shader and renders instanced."""

    def __init__(self, ctx: moderngl.Context, default_prog: moderngl.Program):
        self._ctx = ctx
        self._default_prog = self._ensure_instancing_prog(default_prog)
        self._vao_cache: OrderedDict[tuple[int, int], moderngl.VertexArray] = OrderedDict()
        self._prog_member_cache: dict[int, frozenset] = {}
        self._inst_vbo_capacity: int = _INITIAL_INST_VBO_CAPACITY
        self._shared_inst_vbo = ctx.buffer(reserve=_INITIAL_INST_VBO_CAPACITY * 64)
        self._index_buf: Optional[moderngl.Buffer] = None
        self._index_buf_capacity: int = 0
        self._stats_batches: int = 0
        self._stats_draw_calls: int = 0
        self._stats_instanced: int = 0
        self._total_instances: int = 0
        self._frustum_planes_cache = None
        self._frustum_cache_key = None

    @staticmethod
    def _ensure_instancing_prog(prog: moderngl.Program) -> moderngl.Program:
        if _supports_instancing(prog):
            return prog
        try:
            from core.renderer.mesh_data import SHADER_DIR
            vpath = os.path.join(SHADER_DIR, "default.vert")
            fpath = os.path.join(SHADER_DIR, "default.frag")
            with open(vpath) as f:
                vert = f.read()
            with open(fpath) as f:
                frag = f.read()
            new_prog = prog.ctx.program(vertex_shader=vert, fragment_shader=frag)
            if _supports_instancing(new_prog):
                return new_prog
        except Exception:
            pass
        return prog

    def reset_stats(self):
        self._stats_batches = 0
        self._stats_draw_calls = 0
        self._stats_instanced = 0

    _MAT_NONE = object()

    def _uniform_names(self, prog: moderngl.Program) -> frozenset:
        key = id(prog)
        cached = self._prog_member_cache.get(key)
        if cached is not None:
            return cached
        names = frozenset(prog)
        self._prog_member_cache[key] = names
        return names

    def collect_groups(self, renderables, materials, shaders):
        groups = defaultdict(list)
        for entry in renderables:
            ent, tr, mesh, mr = entry[:4]
            wm = entry[4] if len(entry) > 4 else tr.world_matrix
            sub_idx = entry[5] if len(entry) > 5 else -1
            mat = materials.load_material(mr.get_material_path(sub_idx))
            shader_path = mat.shader_path if mat else ""
            prog = shaders.get_or_compile(shader_path) or self._default_prog
            mat_key = id(mat) if mat else id(self._MAT_NONE)
            key = (id(prog), mat_key, id(mesh), mr.receive_shadows, sub_idx, getattr(mr, 'dynamic_reflections', False))
            groups[key].append((ent, tr, mesh, mr, mat, prog, wm, sub_idx))
        return groups

    def _ensure_index_buffer(self, n: int) -> moderngl.Buffer:
        needed = n * 4
        if self._index_buf is not None and self._index_buf_capacity >= needed:
            return self._index_buf
        if self._index_buf:
            try:
                self._index_buf.release()
            except Exception:
                pass
        self._index_buf = self._ctx.buffer(reserve=needed + 64)
        self._index_buf_capacity = needed + 64
        return self._index_buf

    def _write_shared_vbo(self, matrices: list[Mat4]):
        n = len(matrices)
        needed_cap = n
        if needed_cap > self._inst_vbo_capacity:
            new_cap = self._inst_vbo_capacity
            while new_cap < needed_cap:
                new_cap *= 2
            try:
                self._shared_inst_vbo.release()
            except Exception:
                pass
            self._inst_vbo_capacity = new_cap
            self._shared_inst_vbo = self._ctx.buffer(reserve=new_cap * 64)
            for vao in self._vao_cache.values():
                try:
                    vao.release()
                except Exception:
                    pass
            self._vao_cache.clear()
        try:
            from core._render_utils import batch_mat4_to_f32_flat
            data = batch_mat4_to_f32_flat(matrices).tobytes()
        except ImportError:
            data = Mat4.batch_to_f32(matrices).tobytes()
        self._shared_inst_vbo.write(data)
        return self._shared_inst_vbo

    def _get_vao(self, prog: moderngl.Program, mesh) -> moderngl.VertexArray:
        key = (id(mesh), id(prog))
        cached = self._vao_cache.get(key)
        if cached is not None:
            self._vao_cache.move_to_end(key)
            return cached
        vao = _make_instanced_vao(self._ctx, prog, mesh, self._shared_inst_vbo)
        self._vao_cache[key] = vao
        if len(self._vao_cache) > _MAX_VAO_CACHE:
            _, old_vao = self._vao_cache.popitem(last=False)
            try:
                old_vao.release()
            except Exception:
                pass
        return vao

    def _get_frustum_planes(self, view_f32, proj_f32):
        key = (view_f32.tobytes(), proj_f32.tobytes())
        if self._frustum_cache_key == key and self._frustum_planes_cache is not None:
            return self._frustum_planes_cache
        planes = _extract_frustum_planes_f32(view_f32, proj_f32)
        self._frustum_planes_cache = planes
        self._frustum_cache_key = key
        return planes

    def render_groups(self, groups: dict, view_f32, proj_f32, cam_pos, lights,
                      disable_shadows: bool, set_scene_uniforms_fn,
                      apply_material_fn, normal_cache: dict,
                      selected_entities: set, outline_queue: list,
                      gpu_storage=None, dynamic_cubemaps=None, sky_ibl=None):
        self.reset_stats()
        scene_done = set()
        frustum_planes = self._get_frustum_planes(view_f32, proj_f32)
        for key, group in groups.items():
            _, _, mesh, _, mat, prog, _, _ = group[0]
            dyn_ref = key[5] if len(key) > 5 else False
            self._stats_batches += 1
            n = len(group)
            group_disable_shadows = disable_shadows or not key[3]
            scene_key = (id(prog), group_disable_shadows)
            if scene_key not in scene_done:
                set_scene_uniforms_fn(prog, view_f32, proj_f32, cam_pos, lights,
                                      disable_shadows=group_disable_shadows)
                scene_done.add(scene_key)
            if dyn_ref and dynamic_cubemaps is not None:
                dynamic_cubemaps.bind_ibl(prog)
            elif sky_ibl is not None and sky_ibl.ready:
                sky_ibl.bind(prog)
            elif not dyn_ref:
                names = self._uniform_names(prog)
                try:
                    if "u_irradiance_map_Active" in names:
                        prog["u_irradiance_map_Active"].value = 0
                    if "u_prefilter_map_Active" in names:
                        prog["u_prefilter_map_Active"].value = 0
                    if "u_brdf_lut_Active" in names:
                        prog["u_brdf_lut_Active"].value = 0
                    # Keep the IBL samplers off unit 0: unit 0 may hold a
                    # cubemap left by dynamic-cubemap generation, which makes
                    # the driver reject draws that still reference unit 0.
                    if "u_irradiance_map" in names:
                        prog["u_irradiance_map"].value = 14
                    if "u_prefilter_map" in names:
                        prog["u_prefilter_map"].value = 15
                    if "u_brdf_lut" in names:
                        prog["u_brdf_lut"].value = 16
                except Exception:
                    pass
            if n == 1:
                self._render_single(group[0], prog, mesh, mat,
                                    view_f32, proj_f32, cam_pos, lights,
                                    group_disable_shadows, set_scene_uniforms_fn,
                                    apply_material_fn, normal_cache,
                                    selected_entities, outline_queue,
                                    set_scene=False)
            elif _supports_instancing(prog):
                visible = _frustum_cull_instances(group, frustum_planes,
                                                  mesh.bounding_radius)
                if len(visible) == 0:
                    self._stats_draw_calls += 1
                    continue
                if len(visible) == 1:
                    self._render_single(visible[0], prog, mesh, mat,
                                        view_f32, proj_f32, cam_pos, lights,
                                        group_disable_shadows, set_scene_uniforms_fn,
                                        apply_material_fn, normal_cache,
                                        selected_entities, outline_queue,
                                        set_scene=False)
                else:
                    self._render_instanced(visible, prog, mesh, mat,
                                           view_f32, proj_f32, cam_pos, lights,
                                           group_disable_shadows, set_scene_uniforms_fn,
                                           apply_material_fn,
                                           selected_entities, outline_queue,
                                           gpu_storage=gpu_storage, set_scene=False)
            else:
                for item in group:
                    self._render_single(item, prog, mesh, mat,
                                        view_f32, proj_f32, cam_pos, lights,
                                        group_disable_shadows, set_scene_uniforms_fn,
                                        apply_material_fn, normal_cache,
                                        selected_entities, outline_queue,
                                        set_scene=False)

    def _render_instanced(self, group, prog, mesh, mat,
                          view_f32, proj_f32, cam_pos, lights,
                          disable_shadows, set_scene_uniforms_fn,
                          apply_material_fn,
                          selected_entities, outline_queue,
                          gpu_storage=None, set_scene=True):
        world_ssbo = gpu_storage.get_world_matrix_ssbo() if gpu_storage else None
        names = self._uniform_names(prog)

        if world_ssbo is not None:
            indices = np.array(range(len(group)), dtype=np.uint32)
            idx_buf = self._ensure_index_buffer(len(indices))
            idx_buf.write(indices.tobytes())
            world_ssbo.bind_to_storage_buffer(WORLD_MATRIX_BINDING)
            idx_buf.bind_to_storage_buffer(INDEX_BINDING)
            if "u_use_instancing" in names:
                prog["u_use_instancing"].value = 2
        else:
            model_mats = [item[6] for item in group]
            self._write_shared_vbo(model_mats)
            if "u_use_instancing" in names:
                prog["u_use_instancing"].value = 1
        if "u_use_skinning" in names:
            prog["u_use_skinning"].value = 0

        vao = self._get_vao(prog, mesh)

        if set_scene:
            set_scene_uniforms_fn(prog, view_f32, proj_f32, cam_pos, lights,
                                  disable_shadows=disable_shadows)
        if mesh.is_error_mesh:
            apply_material_fn(None, prog)
            r = 0.1 + 0.9 * abs(np.sin(time.perf_counter() * 3.0))
            if "u_albedo_color" in names:
                prog["u_albedo_color"].write(np.array([r, 0.0, 0.0, 0.8], dtype=np.float32).tobytes())
        else:
            apply_material_fn(mat, prog)

        sub_idx = group[0][7]
        ranges = mesh.sub_mesh_ranges
        ds = bool(mat.properties.get("double_sided") or mat.properties.get("_double_sided")) if mat else False
        cull_on = bool(self._ctx.cull_face)
        if ds and cull_on:
            self._ctx.disable(moderngl.CULL_FACE)
        try:
            if "u_double_sided" in names:
                try:
                    prog["u_double_sided"].value = 1 if ds else 0
                except Exception:
                    pass
            if ranges and sub_idx >= 0 and sub_idx < len(ranges):
                start, count = ranges[sub_idx]
                vao.render(instances=len(group), vertices=count, first=start)
            else:
                vao.render(instances=len(group))
        finally:
            if ds and cull_on:
                self._ctx.enable(moderngl.CULL_FACE)
        self._stats_draw_calls += 1
        self._stats_instanced += len(group)

        if selected_entities:
            for item in group:
                ent, tr, _, _, _, _, wm, _ = item
                if ent in selected_entities:
                    outline_queue.append((mesh, wm))

    def _render_single(self, item, prog, mesh, mat,
                        view_f32, proj_f32, cam_pos, lights,
                        disable_shadows, set_scene_uniforms_fn,
                        apply_material_fn, normal_cache,
                        selected_entities, outline_queue, set_scene=True):
        self._stats_draw_calls += 1
        ent, tr, _, _, _, _, wm, sub_idx = item
        names = self._uniform_names(prog)
        if "u_use_instancing" in names:
            prog["u_use_instancing"].value = 0
        if "u_use_skinning" in names:
            prog["u_use_skinning"].value = 0
        if set_scene:
            set_scene_uniforms_fn(prog, view_f32, proj_f32, cam_pos, lights,
                                  disable_shadows=disable_shadows)
        model = wm
        model_f32 = model.to_f32()
        if "u_model" in names:
            prog["u_model"].write(model_f32)
        nm = resolve_normal_matrix(normal_cache, ent._id, model._d)
        if "u_normal_matrix" in names:
            prog["u_normal_matrix"].write(nm.tobytes())
        if mesh.is_error_mesh:
            apply_material_fn(None, prog)
            r = 0.1 + 0.9 * abs(np.sin(time.perf_counter() * 3.0))
            if "u_albedo_color" in names:
                prog["u_albedo_color"].write(np.array([r, 0.0, 0.0, 0.8], dtype=np.float32).tobytes())
        else:
            apply_material_fn(mat, prog)
        ranges = mesh.sub_mesh_ranges
        ds = bool(mat.properties.get("double_sided") or mat.properties.get("_double_sided")) if mat else False
        cull_on = bool(self._ctx.cull_face)
        if ds and cull_on:
            self._ctx.disable(moderngl.CULL_FACE)
        try:
            if "u_double_sided" in names:
                try:
                    prog["u_double_sided"].value = 1 if ds else 0
                except Exception:
                    pass
            if ranges and sub_idx >= 0 and sub_idx < len(ranges):
                start, count = ranges[sub_idx]
                mesh.render_range(prog, start, count)
            else:
                mesh.render(prog)
        finally:
            if ds and cull_on:
                self._ctx.enable(moderngl.CULL_FACE)
        if selected_entities and ent in selected_entities:
            outline_queue.append((mesh, wm))

    @property
    def draw_calls(self) -> int:
        return self._stats_draw_calls

    @property
    def batches(self) -> int:
        return self._stats_batches

    @property
    def instanced(self) -> int:
        return self._stats_instanced

    def release(self):
        for vao in self._vao_cache.values():
            try:
                vao.release()
            except Exception:
                pass
        self._vao_cache.clear()
        self._prog_member_cache.clear()
        if self._shared_inst_vbo:
            try:
                self._shared_inst_vbo.release()
            except Exception:
                pass
            self._shared_inst_vbo = None
        if self._index_buf:
            try:
                self._index_buf.release()
            except Exception:
                pass
            self._index_buf = None