from __future__ import annotations
import os
import time
import numpy as np
import moderngl
from typing import Any, Optional
from collections import defaultdict
from core.math.math3d import Mat4

_INSTANCE_ATTRS = ("in_model0", "in_model1", "in_model2", "in_model3")

from core.renderer.gpu_culling import WORLD_MATRIX_BINDING, INDEX_BINDING

_MAX_SHARED_INSTANCES = 16384


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
    if _supports_instancing(prog):
        content.append((instance_vbo, '4f 4f 4f 4f /i',
                        'in_model0', 'in_model1', 'in_model2', 'in_model3'))
    if ibo is not None:
        return ctx.vertex_array(prog, content, ibo)
    return ctx.vertex_array(prog, content)


class RenderBatcher:
    """Groups renderables by mesh+material+shader and renders instanced."""

    def __init__(self, ctx: moderngl.Context, default_prog: moderngl.Program):
        self._ctx = ctx
        self._default_prog = self._ensure_instancing_prog(default_prog)
        self._vao_cache: dict[tuple[int, int], moderngl.VertexArray] = {}
        self._shared_inst_vbo = ctx.buffer(reserve=_MAX_SHARED_INSTANCES * 64)
        self._index_buf: Optional[moderngl.Buffer] = None
        self._index_buf_capacity: int = 0
        self._stats_batches: int = 0
        self._stats_draw_calls: int = 0
        self._stats_instanced: int = 0
        self._total_instances: int = 0

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
            key = (id(prog), mat_key, id(mesh), mr.receive_shadows, sub_idx)
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
        data = Mat4.batch_to_f32(matrices).tobytes()
        buf = self._shared_inst_vbo
        if buf.size < len(data):
            buf.release()
            self._shared_inst_vbo = self._ctx.buffer(reserve=len(data) + 64)
            buf = self._shared_inst_vbo
        buf.write(data)
        return buf

    def _get_vao(self, prog: moderngl.Program, mesh) -> moderngl.VertexArray:
        key = (id(mesh), id(prog))
        cached = self._vao_cache.get(key)
        if cached is not None:
            return cached
        vao = _make_instanced_vao(self._ctx, prog, mesh, self._shared_inst_vbo)
        self._vao_cache[key] = vao
        return vao

    def render_groups(self, groups: dict, view_f32, proj_f32, cam_pos, lights,
                      disable_shadows: bool, set_scene_uniforms_fn,
                      apply_material_fn, normal_cache: dict,
                      selected_entities: set, outline_queue: list,
                      gpu_storage=None):
        self.reset_stats()
        scene_done = set()
        for (prog_id, mat_path, mesh_id, receive_shadows, sub_idx), group in groups.items():
            _, _, mesh, _, mat, prog, _, _ = group[0]
            self._stats_batches += 1
            n = len(group)
            group_disable_shadows = disable_shadows or not receive_shadows
            scene_key = (id(prog), group_disable_shadows)
            if scene_key not in scene_done:
                set_scene_uniforms_fn(prog, view_f32, proj_f32, cam_pos, lights,
                                      disable_shadows=group_disable_shadows)
                scene_done.add(scene_key)
            if n == 1:
                self._render_single(group[0], prog, mesh, mat,
                                    view_f32, proj_f32, cam_pos, lights,
                                    group_disable_shadows, set_scene_uniforms_fn,
                                    apply_material_fn, normal_cache,
                                    selected_entities, outline_queue,
                                    set_scene=False)
            elif _supports_instancing(prog):
                self._render_instanced(group, prog, mesh, mat,
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

        if world_ssbo is not None:
            indices = np.array(range(len(group)), dtype=np.uint32)
            idx_buf = self._ensure_index_buffer(len(indices))
            idx_buf.write(indices.tobytes())
            world_ssbo.bind_to_storage_buffer(WORLD_MATRIX_BINDING)
            idx_buf.bind_to_storage_buffer(INDEX_BINDING)
            if "u_use_instancing" in prog:
                prog["u_use_instancing"].value = 2
        else:
            model_mats = [item[6] for item in group]
            self._write_shared_vbo(model_mats)
            if "u_use_instancing" in prog:
                prog["u_use_instancing"].value = 1

        vao = self._get_vao(prog, mesh)

        if set_scene:
            set_scene_uniforms_fn(prog, view_f32, proj_f32, cam_pos, lights,
                                  disable_shadows=disable_shadows)
        if mesh.is_error_mesh:
            apply_material_fn(None, prog)
            r = 0.1 + 0.9 * abs(np.sin(time.perf_counter() * 3.0))
            if "u_albedo_color" in prog:
                prog["u_albedo_color"].write(np.array([r, 0.0, 0.0, 0.8], dtype=np.float32).tobytes())
        else:
            apply_material_fn(mat, prog)

        sub_idx = group[0][7]
        ranges = mesh.sub_mesh_ranges
        if ranges and sub_idx >= 0 and sub_idx < len(ranges):
            start, count = ranges[sub_idx]
            vao.render(instances=len(group), vertices=count, first=start)
        else:
            vao.render(instances=len(group))
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
        if "u_use_instancing" in prog:
            prog["u_use_instancing"].value = 0
        if set_scene:
            set_scene_uniforms_fn(prog, view_f32, proj_f32, cam_pos, lights,
                                  disable_shadows=disable_shadows)
        model = wm
        model_f32 = model.to_f32()
        if "u_model" in prog:
            prog["u_model"].write(model_f32.tobytes())
        nm = normal_cache.get(ent._id)
        if nm is None:
            try:
                from core.math_helpers import mat4_normal_matrix
                nm = mat4_normal_matrix(model._d)
                normal_cache[ent._id] = nm
            except Exception:
                nm = np.eye(3, dtype=np.float32).T
        if "u_normal_matrix" in prog:
            prog["u_normal_matrix"].write(nm.tobytes())
        if mesh.is_error_mesh:
            apply_material_fn(None, prog)
            r = 0.1 + 0.9 * abs(np.sin(time.perf_counter() * 3.0))
            if "u_albedo_color" in prog:
                prog["u_albedo_color"].write(np.array([r, 0.0, 0.0, 0.8], dtype=np.float32).tobytes())
        else:
            apply_material_fn(mat, prog)
        ranges = mesh.sub_mesh_ranges
        if ranges and sub_idx >= 0 and sub_idx < len(ranges):
            start, count = ranges[sub_idx]
            mesh.render_range(prog, start, count)
        else:
            mesh.render(prog)
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
        self._vao_cache.clear()
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
