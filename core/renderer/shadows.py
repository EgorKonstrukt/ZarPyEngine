# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import numpy as np
import moderngl
from typing import Optional, Any
from collections import defaultdict
from core.math.math3d import Vec3, Mat4
from core.components.lighting.light import Light, LightType, LightAreaType

from core.components.rendering.renderers.mesh_filter import MeshFilter
from core.components.rendering.renderers.mesh_renderer import MeshRenderer
from core.renderer.mesh_data import MeshData

_INSTANCE_ATTRS = ("in_model0", "in_model1", "in_model2", "in_model3")

def _shadow_supports_instancing(prog: moderngl.Program) -> bool:
    try:
        locs = prog._attribute_locations
        for a in _INSTANCE_ATTRS:
            if locs.get(a, -1) < 0:
                return False
        return True
    except Exception:
        return False

def _make_shadow_instanced_vao(ctx: moderngl.Context, prog: moderngl.Program,
                                mesh, instance_vbo: moderngl.Buffer) -> moderngl.VertexArray:
    vbo = getattr(mesh, '_vbo', None)
    ibo = getattr(mesh, '_ibo', None)
    if vbo is not None:
        fmt = '3f 3x4 2x4'
        attrs = ('in_position',)
    else:
        n_verts = len(mesh.vertices) // 3 if len(mesh.vertices) > 0 else 0
        data = np.zeros((n_verts, 3), dtype=np.float32)
        data[:, 0:3] = mesh.vertices.reshape(-1, 3)
        vbo = ctx.buffer(data.tobytes())
        fmt = '3f'
        attrs = ('in_position',)
    content = [
        (vbo, fmt, *attrs),
    ]
    if _shadow_supports_instancing(prog):
        content.append((instance_vbo, '4f 4f 4f 4f /i',
                        'in_model0', 'in_model1', 'in_model2', 'in_model3'))
    if ibo is not None:
        return ctx.vertex_array(prog, content, ibo)
    return ctx.vertex_array(prog, content)


class ShadowRenderer:
    def __init__(self, ctx: moderngl.Context, shadow_prog: moderngl.Program,
                 shadow_resolution: int = 4096, shadow_distance: float = 50.0,
                 area_shadow_resolution: int = 8192):
        self._ctx = ctx
        self._prog = shadow_prog
        self._shadow_resolution = shadow_resolution
        self._area_shadow_resolution = area_shadow_resolution
        self._shadow_distance = shadow_distance
        self._shadow_maps: list[Any] = []
        self._shadow_fbos: list[Any] = []
        self._cascade_splits: list[float] = [1000000.0] * 3
        self._light_space_matrices: list[np.ndarray] = [np.eye(4, dtype=np.float32) for _ in range(3)]
        self._point_shadow_resolution: int = 1024
        self._point_shadow_maps: list[Any] = []
        self._point_shadow_fbos: list[Any] = []
        self._point_light_vps: list[np.ndarray] = [np.eye(4, dtype=np.float32) for _ in range(6)]
        self._has_point_shadow: bool = False
        self._point_light_world_pos: Vec3 = Vec3.zero()
        self._point_light_range: float = 10.0
        self._point_light_idx: int = -1
        self._spot_shadow_map: Optional[Any] = None
        self._spot_shadow_fbo: Optional[Any] = None
        self._spot_light_vp: np.ndarray = np.eye(4, dtype=np.float32)
        self._has_spot_shadow: bool = False
        self._spot_light_idx: int = -1
        self._area_shadow_map: Optional[Any] = None
        self._area_shadow_fbo: Optional[Any] = None
        self._area_light_vp: np.ndarray = np.eye(4, dtype=np.float32)
        self._has_area_shadow: bool = False
        self._area_light_idx: int = -1
        self._area_light_size: float = 1.0
        self._area_light_pos: Vec3 = Vec3.zero()
        self._area_light_range: float = 10.0
        self._area_light_near: float = 0.1
        self._area_light_far: float = 10.0
        self._area_light_fov_scale: float = 1.0
        self._area_shadow_bias: float = 0.005
        self._projector_shadow_maps: list[Any] = []
        self._projector_shadow_fbos: list[Any] = []
        self._projector_light_vps: list[np.ndarray] = [np.eye(4, dtype=np.float32) for _ in range(2)]
        self._has_projector_shadow: list[bool] = [False, False]
        self._shadow_vao_cache: dict[tuple[int, int], moderngl.VertexArray] = {}
        self._shadow_inst_vbo: dict[tuple[int, int], moderngl.Buffer] = {}
        self._shadow_groups_cache: Optional[dict] = None
        self._shadow_groups_key: int = 0
        self._shadow_groups_ref: Any = None
        self._create_csm_resources()

    def _create_csm_resources(self):
        for sm in self._shadow_maps:
            try:
                sm.release()
            except Exception:
                pass
        for fbo in self._shadow_fbos:
            try:
                fbo.release()
            except Exception:
                pass
        self._shadow_maps = []
        self._shadow_fbos = []
        res = self._shadow_resolution
        for _ in range(3):
            tex = self._ctx.depth_texture((res, res))
            tex.repeat_x = False
            tex.repeat_y = False
            fbo = self._ctx.framebuffer(depth_attachment=tex)
            self._shadow_maps.append(tex)
            self._shadow_fbos.append(fbo)

    def _create_point_shadow_resources(self):
        res = self._point_shadow_resolution
        for _ in range(6):
            tex = self._ctx.depth_texture((res, res))
            tex.repeat_x = False
            tex.repeat_y = False
            fbo = self._ctx.framebuffer(depth_attachment=tex)
            self._point_shadow_maps.append(tex)
            self._point_shadow_fbos.append(fbo)

    def _create_spot_shadow_resources(self):
        res = self._shadow_resolution
        tex = self._ctx.depth_texture((res, res))
        tex.repeat_x = False
        tex.repeat_y = False
        self._spot_shadow_map = tex
        self._spot_shadow_fbo = self._ctx.framebuffer(depth_attachment=self._spot_shadow_map)

    def _create_projector_shadow_resources(self):
        res = self._shadow_resolution
        for _ in range(2):
            tex = self._ctx.depth_texture((res, res))
            tex.repeat_x = False
            tex.repeat_y = False
            self._projector_shadow_maps.append(tex)
            self._projector_shadow_fbos.append(self._ctx.framebuffer(depth_attachment=tex))

    def _create_area_shadow_resources(self):
        res = self._area_shadow_resolution
        tex = self._ctx.depth_texture((res, res))
        tex.repeat_x = False
        tex.repeat_y = False
        self._area_shadow_map = tex
        self._area_shadow_fbo = self._ctx.framebuffer(depth_attachment=self._area_shadow_map)

    def _build_renderable_shadow(self, scene) -> list[tuple[MeshData, Mat4]]:
        result = []
        for ent in scene.get_entities_with_component(MeshFilter):
            if not ent.active:
                continue
            mf = ent.get_component(MeshFilter)
            mr = ent.get_component(MeshRenderer)
            tr = ent.transform
            if not tr or not mr or not mr.enabled:
                continue
            if not mr.cast_shadows:
                continue
            mp = mf.mesh_path or mf.mesh_name
            _imp = mp + ".import"
            try:
                with open(_imp) as _f:
                    _s = json.load(_f)
                _sk, _scp, _sfu = _s.get("scale", 1.0), _s.get("center_pivot", False), _s.get("flip_uvs", False)
            except Exception:
                _sk, _scp, _sfu = 1.0, False, False
            cache_key = f"{mp}|s={_sk}|cp={_scp}|fu={_sfu}"
            mesh = None
            if hasattr(self, '_get_mesh'):
                mesh = self._get_mesh(cache_key)
                if mesh is None and not mf.mesh_path:
                    mesh = self._get_mesh(mf.mesh_name)
            if mesh:
                result.append((mesh, tr.world_matrix))
        return result

    def _get_mesh(self, cache_key: str) -> Optional[MeshData]:
        return None

    def _build_shadow_groups(self, renderable_shadow: list) -> dict:
        key = id(renderable_shadow)
        if key == self._shadow_groups_key and self._shadow_groups_cache is not None:
            return self._shadow_groups_cache
        groups = defaultdict(list)
        for mesh, model_mat in renderable_shadow:
            groups[id(mesh)].append((mesh, model_mat))
        self._shadow_groups_cache = groups
        self._shadow_groups_key = key
        self._shadow_groups_ref = renderable_shadow
        return groups

    def _build_shadow_instance_vbo(self, key: tuple[int, int],
                                   model_matrices: list) -> moderngl.Buffer:
        data = Mat4.batch_to_f32(model_matrices).tobytes()
        cached = self._shadow_inst_vbo.get(key)
        if cached is not None:
            if cached.size >= len(data):
                try:
                    cached.write(data)
                    return cached
                except Exception:
                    pass
            cached.release()
            self._shadow_inst_vbo.pop(key, None)
            vao_del = self._shadow_vao_cache.pop(key, None)
            if vao_del is not None:
                try: vao_del.release()
                except Exception: pass
        vbo = self._ctx.buffer(data)
        self._shadow_inst_vbo[key] = vbo
        return vbo

    def _get_shadow_vao(self, prog: moderngl.Program, mesh,
                        instance_vbo: moderngl.Buffer) -> moderngl.VertexArray:
        key = (id(mesh), id(prog))
        cached = self._shadow_vao_cache.get(key)
        if cached is not None:
            return cached
        vao = _make_shadow_instanced_vao(self._ctx, prog, mesh, instance_vbo)
        self._shadow_vao_cache[key] = vao
        return vao

    def render_geometry(self, vp: np.ndarray, fbo, renderable_shadow: list, resolution: int = 1024):
        groups = self._build_shadow_groups(renderable_shadow)
        self._render_geometry_with_groups(vp, fbo, groups, resolution)

    def _render_geometry_with_groups(self, vp: np.ndarray, fbo, groups: dict, resolution: int = 1024):
        fbo.clear(depth=1.0)
        fbo.use()
        self._ctx.viewport = (0, 0, resolution, resolution)
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.depth_mask = True
        self._ctx.disable(moderngl.CULL_FACE)
        prog = self._prog
        vp_bytes = vp.astype(np.float32).tobytes()
        prog["u_light_vp"].write(vp_bytes)
        supports_instancing = _shadow_supports_instancing(prog)
        for mesh_id, group in groups.items():
            mesh, _ = group[0]
            n = len(group)
            if n > 1 and supports_instancing:
                key = (mesh_id, id(prog))
                model_mats = [m for _, m in group]
                vbo = self._build_shadow_instance_vbo(key, model_mats)
                vao = self._get_shadow_vao(prog, mesh, vbo)
                if "u_use_instancing" in prog:
                    prog["u_use_instancing"].value = 1
                vao.render(instances=n)
            else:
                if "u_use_instancing" in prog:
                    prog["u_use_instancing"].value = 0
                for _, model_mat in group:
                    prog["u_model"].write(model_mat.to_f32().tobytes())
                    mesh.render(prog)
        self._ctx.enable(moderngl.CULL_FACE)

    def collect_shadow_data(self, scene, meshes: dict) -> list[tuple]:
        self._get_mesh = lambda k: meshes.get(k)
        result = self._build_renderable_shadow(scene)
        self._get_mesh = None
        return result

    def render_shadow_pass(self, renderable_shadow, lights, cam_near: float, cam_far: float, cam_fov: float,
                           aspect: float, view_mat: Mat4, meshes: object = None) -> dict:
        if not self._prog:
            return {}
        if not renderable_shadow:
            self._cascade_splits = [0.0] * 3
            self._has_point_shadow = False
            self._has_spot_shadow = False
            self._has_area_shadow = False
            return {}
        shadow_groups = self._build_shadow_groups(renderable_shadow)
        for l, lt in lights:
            lt_type = l.light_type
            if lt_type == LightType.DIRECTIONAL and l.cast_shadows:
                self._render_directional_shadow(lt, shadow_groups,
                                                cam_near, cam_far, cam_fov, aspect, view_mat)
                break
        else:
            self._cascade_splits = [0.0] * 3
        for l, lt in lights:
            if l.light_type == LightType.POINT and l.cast_shadows:
                self._render_point_shadow(l, lt, shadow_groups, lights)
                break
        else:
            self._has_point_shadow = False
        for l, lt in lights:
            if l.light_type == LightType.SPOT and l.cast_shadows:
                self._render_spot_shadow(l, lt, shadow_groups, lights)
                break
        else:
            self._has_spot_shadow = False
        for l, lt in lights:
            if l.light_type == LightType.AREA and l.cast_shadows:
                self._render_area_shadow(l, lt, shadow_groups, lights)
                break
        else:
            self._has_area_shadow = False
        return shadow_groups

    def _get_frustum_corners(self, near_z: float, far_z: float, cam_fov: float, aspect: float, inv_view: np.ndarray) -> list[np.ndarray]:
        tan_half_fov = math.tan(math.radians(cam_fov) * 0.5)
        corners = []
        for z in (near_z, far_z):
            half_h = tan_half_fov * z
            half_w = half_h * aspect
            for y_sign in (-1, 1):
                for x_sign in (-1, 1):
                    view_pt = np.array([x_sign * half_w, y_sign * half_h, -z, 1.0], dtype=np.float64)
                    world_pt = view_pt @ inv_view
                    world_pt = world_pt / world_pt[3]
                    corners.append(world_pt[:3])
        return corners

    def _cascade_distances(self, cam_near: float, cam_far: float) -> list[float]:
        near_z = max(cam_near, 0.01)
        far_z = max(near_z + 0.1, min(cam_far, self._shadow_distance))
        span = far_z - near_z
        first = near_z + span * 0.14
        second = near_z + span * 0.38
        return [first, max(first + 0.1, second), far_z]

    def _render_directional_shadow(self, sun_transform, shadow_groups,
                                   cam_near, cam_far, cam_fov, aspect, view_mat):
        light_dir = sun_transform.forward.normalized()
        inv_view = np.linalg.inv(view_mat._d)
        splits = self._cascade_distances(cam_near, cam_far)
        self._cascade_splits = splits
        near_z = max(cam_near, 0.01)
        for cascade_idx, split_far in enumerate(splits):
            corners = self._get_frustum_corners(near_z, split_far, cam_fov, aspect, inv_view)
            vp = self._build_directional_cascade(light_dir, corners, split_far - near_z)
            self._light_space_matrices[cascade_idx] = vp
            self._render_geometry_with_groups(vp, self._shadow_fbos[cascade_idx], shadow_groups, resolution=self._shadow_resolution)
            near_z = split_far

    def _build_directional_cascade(self, light_dir: Vec3, corners: list[np.ndarray], depth_span: float) -> np.ndarray:
        corners_np = np.array(corners, dtype=np.float64)
        center = np.mean(corners_np, axis=0)
        radius = 0.0
        for c in corners_np:
            radius = max(radius, float(np.linalg.norm(c - center)))
        radius = max(radius, 0.25)
        radius = math.ceil(radius * 16.0) / 16.0
        light_pos = Vec3(*center) - light_dir * max(radius * 2.0, depth_span + 10.0)
        light_up = Vec3(0.0, 1.0, 0.0)
        if abs(light_dir.dot(light_up)) > 0.999:
            light_up = Vec3(0.0, 0.0, 1.0)
        view = Mat4.look_at(light_pos, Vec3(*center), light_up)
        center_light = np.append(center, 1.0) @ view._d
        texel_size = (radius * 2.0) / max(1, self._shadow_resolution)
        center_light[0] = math.floor(center_light[0] / texel_size) * texel_size
        center_light[1] = math.floor(center_light[1] / texel_size) * texel_size
        left = center_light[0] - radius
        right = center_light[0] + radius
        bottom = center_light[1] - radius
        top = center_light[1] + radius
        corners_light = []
        for c in corners:
            p = np.append(c, 1.0) @ view._d
            corners_light.append(p[:3] / p[3])
        corners_light = np.array(corners_light)
        min_z = float(np.min(corners_light[:, 2]))
        max_z = float(np.max(corners_light[:, 2]))
        z_margin = max(depth_span * 0.45, 6.0)
        n_val = max(-max_z - z_margin, 0.01)
        f_val = max(-min_z + z_margin, n_val + 0.01)
        proj = Mat4.orthographic(left, right, bottom, top, n_val, f_val)
        return view._d @ proj._d

    def _render_point_shadow(self, point_light, point_transform, shadow_groups, lights):
        if not self._point_shadow_maps:
            self._create_point_shadow_resources()
        light_pos = point_transform.position
        light_range = max(point_light.range, 0.1)
        face_configs = [
            (Vec3(1, 0, 0), Vec3(0, -1, 0)),
            (Vec3(-1, 0, 0), Vec3(0, -1, 0)),
            (Vec3(0, 1, 0), Vec3(0, 0, 1)),
            (Vec3(0, -1, 0), Vec3(0, 0, -1)),
            (Vec3(0, 0, 1), Vec3(0, -1, 0)),
            (Vec3(0, 0, -1), Vec3(0, -1, 0)),
        ]
        near_plane = 0.1
        far_plane = light_range
        proj = Mat4.perspective(90.0, 1.0, near_plane, far_plane)
        proj_np = proj._d
        self._has_point_shadow = True
        self._point_light_world_pos = light_pos
        self._point_light_range = light_range
        self._point_light_idx = next(
            (i for i, (l, lt) in enumerate(lights) if l is point_light and lt is point_transform), -1
        )
        for face_idx, (face_dir, face_up) in enumerate(face_configs):
            view = Mat4.look_at(light_pos, light_pos + face_dir, face_up)
            view_np = view._d
            vp = (view_np @ proj_np).astype(np.float32)
            self._point_light_vps[face_idx] = vp
            self._render_geometry_with_groups(vp, self._point_shadow_fbos[face_idx], shadow_groups, resolution=self._point_shadow_resolution)

    def _render_spot_shadow(self, spot_light, spot_transform, shadow_groups, lights):
        if not self._spot_shadow_map:
            self._create_spot_shadow_resources()
        light_pos = spot_transform.position
        light_dir = spot_transform.forward.normalized()
        light_range = max(spot_light.range, 0.1)
        spot_fov = max(spot_light.spot_angle * 2.0, 1.0)
        near_plane = 0.1
        far_plane = light_range
        view = Mat4.look_at(light_pos, light_pos + light_dir, Vec3.up())
        proj = Mat4.perspective(spot_fov, 1.0, near_plane, far_plane)
        vp = (view._d @ proj._d).astype(np.float32)
        self._spot_light_vp = vp
        self._has_spot_shadow = True
        self._spot_light_idx = next(
            (i for i, (l, lt) in enumerate(lights) if l is spot_light and lt is spot_transform), -1
        )
        self._render_geometry_with_groups(vp, self._spot_shadow_fbo, shadow_groups, resolution=self._shadow_resolution)

    def _render_area_shadow(self, area_light, area_transform, shadow_groups, lights):
        if not self._area_shadow_map:
            self._create_area_shadow_resources()
        light_pos = area_transform.position
        light_dir = area_transform.forward.normalized()
        light_up = area_transform.up.normalized()
        if abs(light_dir.dot(light_up)) > 0.999:
            light_up = Vec3(0.0, 0.0, 1.0)
        light_range = max(area_light.range, 0.1)
        near_plane = 0.1
        far_plane = light_range
        self._area_light_near = near_plane
        self._area_light_far = far_plane
        fov = max(90.0, min(150.0, math.degrees(2.0 * math.atan2(max(area_light.area_width, area_light.area_height) * 0.5, near_plane))))
        fov_rad = math.radians(fov)
        tan_half_fov = math.tan(fov_rad * 0.5)
        self._area_light_fov_scale = float(1.0 / (2.0 * tan_half_fov))
        self._area_shadow_bias = float(area_light.area_shadow_bias)
        view = Mat4.look_at(light_pos, light_pos + light_dir, light_up)
        proj = Mat4.perspective(fov, 1.0, near_plane, far_plane)
        vp = (view._d @ proj._d).astype(np.float32)
        self._area_light_vp = vp
        self._has_area_shadow = True
        self._area_light_pos = light_pos
        self._area_light_range = light_range
        self._area_light_size = max(area_light.area_width, area_light.area_height) * 0.5
        self._area_light_idx = next(
            (i for i, (l, lt) in enumerate(lights) if l is area_light and lt is area_transform), -1
        )
        self._render_geometry_with_groups(vp, self._area_shadow_fbo, shadow_groups, resolution=self._area_shadow_resolution)

    def render_projector_shadows(self, projectors, renderable_shadow, shadow_groups: dict = None):
        if shadow_groups is None:
            shadow_groups = self._build_shadow_groups(renderable_shadow)
        if not shadow_groups:
            for i in range(2):
                self._has_projector_shadow[i] = False
            return
        for i, pj in enumerate(projectors[:2]):
            if not pj.cast_shadows:
                self._has_projector_shadow[i] = False
                continue
            if len(self._projector_shadow_maps) <= i:
                self._create_projector_shadow_resources()
            light_pos_vec = Vec3(float(pj.position[0]), float(pj.position[1]), float(pj.position[2]))
            light_dir_vec = Vec3(float(pj.direction[0]), float(pj.direction[1]), float(pj.direction[2]))
            light_dir_vec = light_dir_vec.normalized()
            up_vec = Vec3(float(pj.up[0]), float(pj.up[1]), float(pj.up[2]))
            spot_fov = max(pj.spot_angle, 1.0)
            near_plane = max(pj.near_plane, 0.01)
            far_plane = max(pj.far_plane, near_plane + 0.1)
            view = Mat4.look_at(light_pos_vec, light_pos_vec + light_dir_vec, up_vec)
            proj = Mat4.perspective(spot_fov, pj.aspect_ratio, near_plane, far_plane)
            vp = (view._d @ proj._d).astype(np.float32)
            self._projector_light_vps[i] = vp
            self._has_projector_shadow[i] = True
            self._render_geometry_with_groups(vp, self._projector_shadow_fbos[i], shadow_groups, resolution=self._shadow_resolution)

    def set_uniforms(self, prog):
        has_csm = self._cascade_splits[2] > 0.0
        if has_csm and "u_cascade_count" in prog:
            prog["u_cascade_count"].value = 3
            if "u_light_space_matrices" in prog:
                all_mats = np.array([self._light_space_matrices[ci].astype(np.float32) for ci in range(3)])
                prog["u_light_space_matrices"].write(all_mats.tobytes())
            if "u_cascade_splits" in prog:
                prog["u_cascade_splits"].write(np.array(self._cascade_splits, dtype=np.float32).tobytes())
            for ci in range(3):
                tex_unit = 3 + ci
                self._shadow_maps[ci].use(tex_unit)
                si = f"u_shadow_map_{ci}"
                if si in prog:
                    prog[si].value = tex_unit
        else:
            if "u_cascade_count" in prog:
                prog["u_cascade_count"].value = 0
        if "u_shadow_bias" in prog:
            prog["u_shadow_bias"].value = 0.0008
        if self._has_point_shadow and "u_point_shadow_count" in prog:
            prog["u_point_shadow_count"].value = 1
            if "u_point_light_vps" in prog:
                all_vps = np.array([self._point_light_vps[fi].astype(np.float32) for fi in range(6)])
                prog["u_point_light_vps"].write(all_vps.tobytes())
            for fi in range(6):
                tex_unit = 6 + fi
                self._point_shadow_maps[fi].use(tex_unit)
                si = f"u_point_shadow_map_{fi}"
                if si in prog:
                    prog[si].value = tex_unit
            if "u_point_light_pos" in prog:
                prog["u_point_light_pos"].write(
                    np.array(self._point_light_world_pos.to_array(), dtype=np.float32).tobytes()
                )
            if "u_point_light_range" in prog:
                prog["u_point_light_range"].value = float(self._point_light_range)
            if "u_point_shadow_light_index" in prog:
                prog["u_point_shadow_light_index"].value = self._point_light_idx if self._point_light_idx >= 0 else -1
        else:
            if "u_point_shadow_count" in prog:
                prog["u_point_shadow_count"].value = 0
        if self._has_spot_shadow and "u_spot_shadow_count" in prog:
            prog["u_spot_shadow_count"].value = 1
            tex_unit = 12
            self._spot_shadow_map.use(tex_unit)
            if "u_spot_shadow_map" in prog:
                prog["u_spot_shadow_map"].value = tex_unit
            if "u_spot_light_vp" in prog:
                prog["u_spot_light_vp"].write(self._spot_light_vp.tobytes())
            if "u_spot_shadow_light_index" in prog:
                prog["u_spot_shadow_light_index"].value = self._spot_light_idx if self._spot_light_idx >= 0 else -1
        else:
            if "u_spot_shadow_count" in prog:
                prog["u_spot_shadow_count"].value = 0
        if self._has_area_shadow and "u_area_shadow_light_index" in prog:
            tex_unit = 15
            self._area_shadow_map.use(tex_unit)
            if "u_area_shadow_map" in prog:
                prog["u_area_shadow_map"].value = tex_unit
            if "u_area_light_vp" in prog:
                prog["u_area_light_vp"].write(self._area_light_vp.tobytes())
            if "u_area_light_size" in prog:
                prog["u_area_light_size"].value = float(self._area_light_size)
            if "u_area_light_fov_scale" in prog:
                prog["u_area_light_fov_scale"].value = float(self._area_light_fov_scale)
            if "u_area_light_near_far" in prog:
                prog["u_area_light_near_far"].write(
                    np.array([self._area_light_near, self._area_light_far], dtype=np.float32).tobytes()
                )
            if "u_area_shadow_light_index" in prog:
                prog["u_area_shadow_light_index"].value = self._area_light_idx if self._area_light_idx >= 0 else -1
            if "u_area_shadow_bias" in prog:
                prog["u_area_shadow_bias"].value = float(self._area_shadow_bias)
        else:
            if "u_area_shadow_light_index" in prog:
                prog["u_area_shadow_light_index"].value = -1
        for i in range(2):
            suf = f"u_pj_{i}_shadow_map"
            if self._has_projector_shadow[i] and suf in prog:
                tex_unit = 16 + i
                if i < len(self._projector_shadow_maps):
                    self._projector_shadow_maps[i].use(tex_unit)
                    prog[suf].value = tex_unit
                if f"u_pj_{i}_shadow_vp" in prog:
                    prog[f"u_pj_{i}_shadow_vp"].write(self._projector_light_vps[i].tobytes())
            elif suf in prog:
                prog[suf].value = 0

    def release(self):
        for sm in self._shadow_maps:
            try:
                sm.release()
            except Exception:
                pass
        for fbo in self._shadow_fbos:
            try:
                fbo.release()
            except Exception:
                pass
        for sm in self._point_shadow_maps:
            try:
                sm.release()
            except Exception:
                pass
        for fbo in self._point_shadow_fbos:
            try:
                fbo.release()
            except Exception:
                pass
        if self._spot_shadow_map:
            try:
                self._spot_shadow_map.release()
            except Exception:
                pass
        if self._spot_shadow_fbo:
            try:
                self._spot_shadow_fbo.release()
            except Exception:
                pass
        if self._area_shadow_map:
            try:
                self._area_shadow_map.release()
            except Exception:
                pass
        if self._area_shadow_fbo:
            try:
                self._area_shadow_fbo.release()
            except Exception:
                pass
        for sm in self._projector_shadow_maps:
            try:
                sm.release()
            except Exception:
                pass
        for fbo in self._projector_shadow_fbos:
            try:
                fbo.release()
            except Exception:
                pass
        for vbo in self._shadow_inst_vbo.values():
            try:
                vbo.release()
            except Exception:
                pass
        self._shadow_inst_vbo.clear()
        self._shadow_vao_cache.clear()
