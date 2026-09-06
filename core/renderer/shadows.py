from __future__ import annotations

import json
import math
import os

import numpy as np
import moderngl
from typing import Optional, Any
from collections import defaultdict
from core.maths.math3d import Vec3, Mat4
from core.components.lighting.light import Light, LightType, LightAreaType

from core.components.rendering.renderers.mesh_filter import MeshFilter
from core.components.rendering.renderers.mesh_renderer import MeshRenderer
from core.renderer.mesh_data import MeshData

_INSTANCE_ATTRS = ("in_model0", "in_model1", "in_model2", "in_model3")
MAX_POINT_SHADOWS = 4
MAX_SPOT_SHADOWS = 4
_POINT_FACE_DIRS = (
    (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
)
_POINT_FACE_UPS = (
    (0.0, -1.0, 0.0), (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    (0.0, -1.0, 0.0), (0.0, -1.0, 0.0),
)


def _tr_pos_xyz(tr):
    d = tr.world_matrix._d
    return float(d[3, 0]), float(d[3, 1]), float(d[3, 2])


def _tr_fwd_xyz(tr):
    d = tr.world_matrix._d
    fx = -float(d[2, 0])
    fy = -float(d[2, 1])
    fz = -float(d[2, 2])
    inv = 1.0 / max(1e-12, math.sqrt(fx * fx + fy * fy + fz * fz))
    return fx * inv, fy * inv, fz * inv


def _tr_up_xyz(tr):
    d = tr.world_matrix._d
    ux = float(d[1, 0])
    uy = float(d[1, 1])
    uz = float(d[1, 2])
    inv = 1.0 / max(1e-12, math.sqrt(ux * ux + uy * uy + uz * uz))
    return ux * inv, uy * inv, uz * inv

try:
    from core._shadow_batch import (
        compute_frustum_corners_out,
        build_directional_cascade_fast,
        pack_model_matrices_f32,
        frustum_cull_shadow_groups,
        pack_cascade_vps_transposed,
        compute_frustum_corners,
        build_directional_cascade,
        compute_cascade_distances,
        pack_cascade_matrices_f32,
        pack_cascade_splits_f32,
        pack_point_vps_f32,
        build_shadow_groups as cy_build_shadow_groups,
        face_cull_point_shadow,
        prepare_shadow_flat,
        cull_flat,
        cull_flat_min,
        cull_flat_range_min,
    )
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False


try:
    from core.math_helpers import mat4_inv_fast as _mat4_inv_fast
except ImportError:
    _mat4_inv_fast = None


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
                  shadow_resolution: int = 512, shadow_distance: float = 50.0,
                  area_shadow_resolution: int = 512, cascade_count: int = 2,
                  cascade_splits: list = None, point_shadow_resolution: int = None,
                  spot_shadow_resolution: int = None):
        self._ctx = ctx
        self._prog = shadow_prog
        self._shadow_resolution = shadow_resolution
        self._area_shadow_resolution = area_shadow_resolution
        self._shadow_distance = shadow_distance
        self._cascade_count = max(1, min(cascade_count, 4))
        self._cascade_splits_norm: list[float] | None = cascade_splits
        self._shadow_maps: list[Any] = []
        self._shadow_fbos: list[Any] = []
        self._cascade_splits: list[float] = [0.0] * 4
        self._light_space_matrices: list[np.ndarray] = [np.eye(4, dtype=np.float32) for _ in range(4)]
        self._point_shadow_resolution: int = point_shadow_resolution or shadow_resolution
        self._spot_shadow_resolution: int = spot_shadow_resolution or shadow_resolution
        self._point_shadow_maps: list[Any] = []
        self._point_shadow_fbos: list[Any] = []
        self._point_light_vps: list[np.ndarray] = [np.eye(4, dtype=np.float32) for _ in range(MAX_POINT_SHADOWS * 6)]
        self._has_point_shadow: bool = False
        self._point_shadow_light_positions: list[Vec3] = [Vec3.zero() for _ in range(MAX_POINT_SHADOWS)]
        self._point_shadow_light_ranges: list[float] = [10.0] * MAX_POINT_SHADOWS
        self._point_shadow_light_indices: list[int] = [-1] * MAX_POINT_SHADOWS
        self._point_shadow_count: int = 0
        self._spot_shadow_maps: list[Any] = []
        self._spot_shadow_fbos: list[Any] = []
        self._spot_shadow_vps: list[np.ndarray] = [np.eye(4, dtype=np.float32) for _ in range(MAX_SPOT_SHADOWS)]
        self._has_spot_shadow: bool = False
        self._spot_shadow_light_indices: list[int] = [-1] * MAX_SPOT_SHADOWS
        self._spot_shadow_count: int = 0
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
        self._prog_member_cache: dict[int, frozenset] = {}
        self._shadow_inst_vbo: dict[tuple[int, int], moderngl.Buffer] = {}
        self._shadow_inst_vbo_fp: dict[tuple[int, int], tuple] = {}
        self._skinned_bone_ssbo: Optional[Any] = None
        self._skinned_bone_ssbo_cap: int = 0
        self._pending_skinned: list = []
        self._pending_scene: Any = None
        self._skinning_cache: dict = None
        self._shadow_groups_cache: Optional[dict] = None
        self._cascade_matrices_buf = np.zeros((4, 4, 4), dtype=np.float32)
        self._cascade_splits_buf = np.zeros(4, dtype=np.float32)
        self._point_vps_buf = np.zeros((MAX_POINT_SHADOWS * 6, 4, 4), dtype=np.float32)
        self._point_pos_buf = np.zeros((MAX_POINT_SHADOWS, 3), dtype=np.float32)
        self._point_range_buf = np.zeros(MAX_POINT_SHADOWS, dtype=np.float32)
        self._point_idx_buf = np.zeros(MAX_POINT_SHADOWS, dtype=np.int32)
        self._spot_vps_buf = np.zeros((MAX_SPOT_SHADOWS, 4, 4), dtype=np.float32)
        self._spot_idx_buf = np.zeros(MAX_SPOT_SHADOWS, dtype=np.int32)
        self._area_nearfar_buf = np.zeros(2, dtype=np.float32)
        self._vp_f32_buf = np.zeros((4, 4), dtype=np.float32)
        self._shadow_groups_key: int = 0
        self._shadow_groups_ref: Any = None
        self._inv_view_buf = np.zeros((4, 4), dtype=np.float64)
        self._frustum_corners_buf = np.zeros((8, 3), dtype=np.float64)
        self._cascade_vps_raw = np.zeros((4, 4, 4), dtype=np.float64)
        self._model_pack_buf: Optional[np.ndarray] = None
        self._model_pack_buf_cap: int = 0
        self._temporal_frame: int = 0
        self._temporal_skip_idx: int = -1
        self._cascade_resolutions: list[int] = [2048, 1024, 1024, 512]
        self._last_light_dir: Optional[Vec3] = None
        self._last_cam_view_hash: int = 0
        self._import_cache: dict[str, tuple] = {}
        self._import_cache_mtime: dict[str, float] = {}
        self._shadow_cache_valid: bool = False
        self._type_flags: dict = {'directional': True, 'point': True, 'spot': True, 'area': True}
        self._flat_cap: int = 0
        self._flat_n: int = 0
        self._flat_centers = np.zeros((0, 3), dtype=np.float64)
        self._flat_radii = np.zeros(0, dtype=np.float64)
        self._flat_mats = np.zeros((0, 16), dtype=np.float32)
        self._flat_mesh_ids = np.zeros(0, dtype=np.uint64)
        self._flat_out = np.zeros(0, dtype=np.intp)
        self._flat_mesh_map: dict = {}
        self._mesh_radius_cache: dict = {}
        self._instancing_cache: dict[int, bool] = {}
        self._point_proj_cache: dict = {}
        self._spot_proj_cache: dict = {}
        self._last_view = np.zeros((4, 4), dtype=np.float64)
        self._last_view_valid: bool = False
        self._last_light_dir_xyz: tuple = (0.0, 0.0, 0.0)
        self._cascade_matrices_bytes: bytes = b""
        self._cascade_splits_bytes: bytes = b""
        self._point_vps_bytes: bytes = b""
        self._point_pos_bytes: bytes = b""
        self._point_range_bytes: bytes = b""
        self._point_idx_bytes: bytes = b""
        self._spot_vps_bytes: bytes = b""
        self._spot_idx_bytes: bytes = b""
        self._area_vp_bytes: bytes = b""
        self._area_nearfar_bytes: bytes = b""
        self._create_csm_resources()

    def update_settings(self, shadow_resolution: int = None, shadow_distance: float = None,
                        cascade_count: int = None, area_shadow_resolution: int = None,
                        cascade_splits: list = None, point_shadow_resolution: int = None,
                        spot_shadow_resolution: int = None, type_flags: dict = None):
        changed = False
        if shadow_resolution is not None and shadow_resolution != self._shadow_resolution:
            self._shadow_resolution = shadow_resolution
            base = shadow_resolution
            self._cascade_resolutions = [base, max(512, base//2), max(512, base//2), max(256, base//4)]
            changed = True
        if shadow_distance is not None and shadow_distance != self._shadow_distance:
            self._shadow_distance = shadow_distance
        if cascade_splits is not None:
            self._cascade_splits_norm = list(cascade_splits) if cascade_splits else None
        if cascade_count is not None:
            cc = max(1, min(cascade_count, 4))
            if cc != self._cascade_count:
                self._cascade_count = cc
                changed = True
        if area_shadow_resolution is not None and area_shadow_resolution != self._area_shadow_resolution:
            self._area_shadow_resolution = area_shadow_resolution
            changed = True
        if point_shadow_resolution is not None and point_shadow_resolution != self._point_shadow_resolution:
            self._point_shadow_resolution = point_shadow_resolution
            changed = True
        if spot_shadow_resolution is not None and spot_shadow_resolution != self._spot_shadow_resolution:
            self._spot_shadow_resolution = spot_shadow_resolution
            changed = True
        if type_flags is not None:
            self._type_flags = dict(type_flags)
        if changed:
            try:
                self._create_csm_resources()
            except Exception:
                pass
            try:
                self._create_point_shadow_resources()
            except Exception:
                pass
            try:
                self._create_spot_shadow_resources()
            except Exception:
                pass

    def _ensure_context(self):
        try:
            from core.engine.engine import Engine
            eng = Engine.instance()
            vp = getattr(eng, "viewport", None) if eng else None
            if vp is not None and hasattr(vp, "makeCurrent"):
                try:
                    vp.makeCurrent()
                except Exception:
                    pass
        except Exception:
            pass
    def _create_csm_resources(self):
        self._ensure_context()
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
        base = self._shadow_resolution
        for i in range(self._cascade_count):
            res = self._cascade_resolutions[i] if i < len(self._cascade_resolutions) else base
            res = min(res, base)
            try:
                tex = self._ctx.depth_texture((res, res))
            except Exception:
                try:
                    import time
                    time.sleep(0.01)
                    tex = self._ctx.depth_texture((res, res))
                except Exception:
                    continue
            tex.repeat_x = False
            tex.repeat_y = False
            try:
                fbo = self._ctx.framebuffer(depth_attachment=tex)
            except Exception:
                try:
                    tex.release()
                except Exception:
                    pass
                continue
            self._shadow_maps.append(tex)
            self._shadow_fbos.append(fbo)

    def _create_point_shadow_resources(self):
        self._ensure_context()
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
        self._point_shadow_maps = []
        self._point_shadow_fbos = []
        res = self._point_shadow_resolution
        for _ in range(MAX_POINT_SHADOWS * 6):
            try:
                tex = self._ctx.depth_texture((res, res))
            except Exception:
                continue
            tex.repeat_x = False
            tex.repeat_y = False
            try:
                fbo = self._ctx.framebuffer(depth_attachment=tex)
            except Exception:
                try:
                    tex.release()
                except Exception:
                    pass
                continue
            self._point_shadow_maps.append(tex)
            self._point_shadow_fbos.append(fbo)

    def _create_spot_shadow_resources(self):
        self._ensure_context()
        for sm in self._spot_shadow_maps:
            try:
                sm.release()
            except Exception:
                pass
        for fbo in self._spot_shadow_fbos:
            try:
                fbo.release()
            except Exception:
                pass
        self._spot_shadow_maps = []
        self._spot_shadow_fbos = []
        res = self._spot_shadow_resolution
        for _ in range(MAX_SPOT_SHADOWS):
            try:
                tex = self._ctx.depth_texture((res, res))
            except Exception:
                continue
            tex.repeat_x = False
            tex.repeat_y = False
            try:
                fbo = self._ctx.framebuffer(depth_attachment=tex)
            except Exception:
                try:
                    tex.release()
                except Exception:
                    pass
                continue
            self._spot_shadow_maps.append(tex)
            self._spot_shadow_fbos.append(fbo)

    def _create_projector_shadow_resources(self):
        self._ensure_context()
        res = self._shadow_resolution
        for _ in range(2):
            try:
                tex = self._ctx.depth_texture((res, res))
            except Exception:
                continue
            tex.repeat_x = False
            tex.repeat_y = False
            try:
                self._projector_shadow_maps.append(tex)
                self._projector_shadow_fbos.append(self._ctx.framebuffer(depth_attachment=tex))
            except Exception:
                try:
                    tex.release()
                except Exception:
                    pass
                continue

    def _create_area_shadow_resources(self):
        self._ensure_context()
        res = self._area_shadow_resolution
        try:
            tex = self._ctx.depth_texture((res, res))
        except Exception:
            return
        tex.repeat_x = False
        tex.repeat_y = False
        try:
            self._area_shadow_map = tex
            self._area_shadow_fbo = self._ctx.framebuffer(depth_attachment=self._area_shadow_map)
        except Exception:
            try:
                tex.release()
            except Exception:
                pass

    def _build_renderable_shadow(self, scene) -> list[tuple[MeshData, Mat4]]:
        result = []
        get_entities = scene.get_entities_with_component
        for ent in get_entities(MeshFilter):
            if not ent._active:
                continue
            tm = ent._type_map
            mf_list = tm.get(MeshFilter)
            if not mf_list:
                continue
            mf = mf_list[0]
            mr_list = tm.get(MeshRenderer)
            if not mr_list:
                continue
            mr = mr_list[0]
            if not mr.enabled or not mr.cast_shadows:
                continue
            tt = ent._transform_type
            tr = tm.get(tt, [None])[0] if tt is not None and tt in tm else ent.transform
            if tr is None:
                continue
            mp = mf.mesh_path or mf.mesh_name or ""
            if not mp:
                continue
            _imp = mp + ".import"
            cached = self._import_cache.get(_imp)
            if cached is not None:
                mtime = self._import_cache_mtime.get(_imp, 0)
                try:
                    cur = os.path.getmtime(_imp)
                    if abs(cur - mtime) < 0.001:
                        _sk, _scp, _sfu = cached
                    else:
                        raise KeyError
                except Exception:
                    try:
                        with open(_imp) as _f:
                            _s = json.load(_f)
                        _sk, _scp, _sfu = _s.get("scale", 1.0), _s.get("center_pivot", False), _s.get("flip_uvs", False)
                    except Exception:
                        _sk, _scp, _sfu = 1.0, False, False
                    self._import_cache[_imp] = (_sk, _scp, _sfu)
                    try:
                        self._import_cache_mtime[_imp] = os.path.getmtime(_imp)
                    except Exception:
                        self._import_cache_mtime[_imp] = 0
            else:
                try:
                    with open(_imp) as _f:
                        _s = json.load(_f)
                    _sk, _scp, _sfu = _s.get("scale", 1.0), _s.get("center_pivot", False), _s.get("flip_uvs", False)
                except Exception:
                    _sk, _scp, _sfu = 1.0, False, False
                self._import_cache[_imp] = (_sk, _scp, _sfu)
                try:
                    self._import_cache_mtime[_imp] = os.path.getmtime(_imp)
                except Exception:
                    self._import_cache_mtime[_imp] = 0
            cache_key = f"{mp}|s={_sk}|cp={_scp}|fu={_sfu}"
            mesh = None
            gm = getattr(self, '_get_mesh', None)
            if gm is not None:
                mesh = gm(cache_key)
                if mesh is None and not mf.mesh_path:
                    mesh = gm(mf.mesh_name)
            if mesh is not None:
                result.append((mesh, tr))
        return result

    def _get_mesh(self, cache_key: str) -> Optional[MeshData]:
        return None

    def _build_shadow_groups(self, renderable_shadow: list) -> dict:
        key = id(renderable_shadow)
        if key == self._shadow_groups_key and self._shadow_groups_cache is not None:
            return self._shadow_groups_cache
        if _HAS_CYTHON:
            groups = cy_build_shadow_groups(renderable_shadow)
        else:
            groups = defaultdict(list)
            for mesh, tr in renderable_shadow:
                groups[id(mesh)].append((mesh, tr))
        self._shadow_groups_cache = groups
        self._shadow_groups_key = key
        self._shadow_groups_ref = renderable_shadow
        return groups

    def _ensure_model_pack_buf(self, needed_count: int):
        if self._model_pack_buf is not None and self._model_pack_buf_cap >= needed_count * 16:
            return
        cap = max(64, needed_count) * 16
        self._model_pack_buf = np.empty(cap, dtype=np.float32)
        self._model_pack_buf_cap = cap

    def _ensure_flat_cap(self, n: int):
        if n <= self._flat_cap:
            return
        cap = max(64, int(n * 1.5) + 8)
        self._flat_centers = np.empty((cap, 3), dtype=np.float64)
        self._flat_radii = np.empty(cap, dtype=np.float64)
        self._flat_mats = np.empty((cap, 16), dtype=np.float32)
        self._flat_mesh_ids = np.empty(cap, dtype=np.uint64)
        self._flat_out = np.empty(cap, dtype=np.intp)
        self._flat_cap = cap

    def _prepare_flat(self, renderable_shadow: list) -> int:
        n = len(renderable_shadow)
        if n == 0:
            self._flat_n = 0
            self._flat_mesh_map = {}
            return 0
        self._ensure_flat_cap(n)
        if _HAS_CYTHON:
            try:
                prepare_shadow_flat(renderable_shadow, self._mesh_radius_cache,
                                    self._flat_centers, self._flat_radii,
                                    self._flat_mats, self._flat_mesh_ids)
            except Exception:
                self._prepare_flat_numpy(renderable_shadow, n)
        else:
            self._prepare_flat_numpy(renderable_shadow, n)
        centers = self._flat_centers
        radii = self._flat_radii
        mats = self._flat_mats
        ids = self._flat_mesh_ids
        mesh_map: dict = {}
        for i in range(n):
            mid = int(ids[i])
            if mid not in mesh_map:
                try:
                    mesh_map[mid] = renderable_shadow[i][0]
                except Exception:
                    pass
        self._flat_mesh_map = mesh_map
        self._flat_n = n
        return n

    def _prepare_flat_numpy(self, renderable_shadow: list, n: int):
        centers = self._flat_centers
        radii = self._flat_radii
        mats = self._flat_mats
        ids = self._flat_mesh_ids
        rc = self._mesh_radius_cache
        for i in range(n):
            entry = renderable_shadow[i]
            mesh = entry[0]
            tr = entry[1]
            ids[i] = id(mesh)
            if tr is None:
                centers[i, 0] = 1e30
                centers[i, 1] = 1e30
                centers[i, 2] = 1e30
                radii[i] = 0.0
                mats[i, :] = 0.0
                continue
            wm = tr.world_matrix._d
            centers[i, 0] = wm[3, 0]
            centers[i, 1] = wm[3, 1]
            centers[i, 2] = wm[3, 2]
            sx = math.sqrt(wm[0, 0] * wm[0, 0] + wm[1, 0] * wm[1, 0] + wm[2, 0] * wm[2, 0])
            sy = math.sqrt(wm[0, 1] * wm[0, 1] + wm[1, 1] * wm[1, 1] + wm[2, 1] * wm[2, 1])
            sz = math.sqrt(wm[0, 2] * wm[0, 2] + wm[1, 2] * wm[1, 2] + wm[2, 2] * wm[2, 2])
            ms = sx
            if sy > ms:
                ms = sy
            if sz > ms:
                ms = sz
            br = rc.get(mesh)
            if br is None:
                try:
                    br = float(mesh.bounding_radius)
                except Exception:
                    br = 1.0
                rc[mesh] = br
            radii[i] = ms * br
            m = wm.astype(np.float32, copy=False).reshape(-1)
            mats[i, :] = m

    def _supports_instancing_cached(self, prog) -> bool:
        key = id(prog)
        v = self._instancing_cache.get(key)
        if v is not None:
            return v
        v = _shadow_supports_instancing(prog)
        self._instancing_cache[key] = v
        return v

    def _upload_instanced_mats(self, key: tuple[int, int], mats_slice: np.ndarray) -> moderngl.Buffer:
        data = mats_slice.tobytes()
        cached = self._shadow_inst_vbo.get(key)
        if cached is not None:
            try:
                if cached.size >= len(data):
                    cached.write(data)
                    return cached
            except Exception:
                pass
            try:
                cached.release()
            except Exception:
                pass
            self._shadow_inst_vbo.pop(key, None)
            vao_del = self._shadow_vao_cache.pop(key, None)
            if vao_del is not None:
                try:
                    vao_del.release()
                except Exception:
                    pass
        vbo = self._ctx.buffer(data)
        self._shadow_inst_vbo[key] = vbo
        return vbo

    def _build_shadow_instance_vbo(self, key: tuple[int, int],
                                   model_matrices) -> moderngl.Buffer:
        if isinstance(model_matrices, np.ndarray):
            return self._upload_instanced_mats(key, np.ascontiguousarray(model_matrices, dtype=np.float32))
        n = len(model_matrices)
        self._ensure_model_pack_buf(n)
        if _HAS_CYTHON:
            try:
                pack_model_matrices_f32(model_matrices, self._model_pack_buf)
                data = self._model_pack_buf[:n * 16].tobytes()
            except Exception:
                try:
                    from core._render_utils import batch_mat4_to_f32_flat
                    data = batch_mat4_to_f32_flat(model_matrices).tobytes()
                except ImportError:
                    data = Mat4.batch_to_f32(model_matrices).tobytes()
        else:
            try:
                from core._render_utils import batch_mat4_to_f32_flat
                data = batch_mat4_to_f32_flat(model_matrices).tobytes()
            except ImportError:
                data = Mat4.batch_to_f32(model_matrices).tobytes()
        cached = self._shadow_inst_vbo.get(key)
        if cached is not None:
            try:
                if cached.size >= len(data):
                    cached.write(data)
                    return cached
            except Exception:
                pass
            try:
                cached.release()
            except Exception:
                pass
            self._shadow_inst_vbo.pop(key, None)
            vao_del = self._shadow_vao_cache.pop(key, None)
            if vao_del is not None:
                try:
                    vao_del.release()
                except Exception:
                    pass
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

    def _uniform_names(self, prog: moderngl.Program) -> frozenset:
        key = id(prog)
        cached = self._prog_member_cache.get(key)
        if cached is not None:
            return cached
        names = frozenset(prog)
        self._prog_member_cache[key] = names
        return names

    def _cull_flat_count(self, vp: np.ndarray) -> int:
        n = self._flat_n
        if n == 0:
            return 0
        if _HAS_CYTHON:
            try:
                return int(cull_flat(self._flat_centers[:n], self._flat_radii[:n], vp, self._flat_out[:n]))
            except Exception:
                pass
        from core.renderer.culling import cpu_frustum_cull
        try:
            vis = cpu_frustum_cull(self._flat_centers[:n], self._flat_radii[:n], np.asarray(vp, dtype=np.float64))
            m = len(vis)
            self._flat_out[:m] = vis
            return m
        except Exception:
            return n

    def _cull_flat_min_count(self, vp: np.ndarray, min_radius: float) -> int:
        n = self._flat_n
        if n == 0:
            return 0
        if min_radius <= 0.0:
            return self._cull_flat_count(vp)
        if _HAS_CYTHON:
            try:
                return int(cull_flat_min(self._flat_centers[:n], self._flat_radii[:n], vp, float(min_radius), self._flat_out[:n]))
            except Exception:
                pass
        return self._cull_flat_count(vp)

    def _cull_flat_range_count(self, vp: np.ndarray, lx: float, ly: float, lz: float, range_sq: float, min_radius: float = 0.0) -> int:
        n = self._flat_n
        if n == 0:
            return 0
        if _HAS_CYTHON:
            try:
                return int(cull_flat_range_min(self._flat_centers[:n], self._flat_radii[:n], vp, float(lx), float(ly), float(lz), float(range_sq), float(min_radius), self._flat_out[:n]))
            except Exception:
                pass
        c = self._flat_centers[:n]
        dx = c[:, 0] - lx
        dy = c[:, 1] - ly
        dz = c[:, 2] - lz
        mask = (dx * dx + dy * dy + dz * dz) <= range_sq
        if min_radius > 0.0:
            mask = mask & (self._flat_radii[:n] >= min_radius)
        idx = np.nonzero(mask)[0].astype(np.intp, copy=False)
        if idx.size == 0:
            return 0
        from core.renderer.culling import cpu_frustum_cull
        try:
            vis = cpu_frustum_cull(c[idx, :], self._flat_radii[:n][idx], np.asarray(vp, dtype=np.float64))
            mapped = idx[vis]
            m = len(mapped)
            self._flat_out[:m] = mapped
            return m
        except Exception:
            m = len(idx)
            self._flat_out[:m] = idx
            return m

    def _draw_flat_visible(self, vp: np.ndarray, fbo, resolution: int, visible_count: int):
        if visible_count <= 0:
            return
        n = self._flat_n
        vis = self._flat_out[:visible_count]
        mesh_ids = self._flat_mesh_ids[:n]
        vis_mesh = mesh_ids[vis]
        uniq = np.unique(vis_mesh)
        prog = self._prog
        supports_instancing = self._supports_instancing_cached(prog)
        names = self._uniform_names(prog)
        use_inst = "u_use_instancing" in names
        fbo.clear(depth=1.0)
        fbo.use()
        self._ctx.viewport = (0, 0, resolution, resolution)
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.depth_mask = True
        self._ctx.disable(moderngl.CULL_FACE)
        prog["u_light_vp"].write(vp.tobytes())
        mats = self._flat_mats
        mmap = self._flat_mesh_map
        prog_id = id(prog)
        if supports_instancing:
            if use_inst:
                prog["u_use_instancing"].value = 1
            for mid in uniq:
                mid_i = int(mid)
                mesh = mmap.get(mid_i)
                if mesh is None:
                    continue
                sel = vis[vis_mesh == mid_i]
                if sel.size == 0:
                    continue
                chunk = mats[sel]
                key = (mid_i, prog_id)
                vbo = self._upload_instanced_mats(key, chunk)
                vao = self._get_shadow_vao(prog, mesh, vbo)
                vao.render(instances=int(sel.size))
        else:
            if use_inst:
                prog["u_use_instancing"].value = 0
            umodel = "u_model" in names
            for mid in uniq:
                mid_i = int(mid)
                mesh = mmap.get(mid_i)
                if mesh is None:
                    continue
                sel = vis[vis_mesh == mid_i]
                if sel.size == 0:
                    continue
                if umodel:
                    for fi in sel:
                        prog["u_model"].write(mats[int(fi)].tobytes())
                        mesh.render(prog)
                else:
                    for _fi in sel:
                        mesh.render(prog)
        self._ctx.enable(moderngl.CULL_FACE)

    def render_geometry(self, vp: np.ndarray, fbo, renderable_shadow: list, resolution: int = 1024):
        if _HAS_CYTHON and isinstance(renderable_shadow, list) and len(renderable_shadow) > 32:
            try:
                self._prepare_flat(renderable_shadow)
                vp32 = np.asarray(vp, dtype=np.float32)
                cnt = self._cull_flat_count(vp32)
                if cnt:
                    self._draw_flat_visible(vp32, fbo, resolution, cnt)
                return
            except Exception:
                pass
        groups = self._build_shadow_groups(renderable_shadow)
        self._render_geometry_with_groups(vp, fbo, groups, resolution)

    @staticmethod
    def _filter_by_range(shadow_groups: dict, light_x: float, light_y: float, light_z: float,
                         range_sq: float) -> dict:
        result = {}
        for mid, group in shadow_groups.items():
            near = []
            for mesh, tr in group:
                p = tr.world_matrix._d
                dx = p[3][0] - light_x
                dy = p[3][1] - light_y
                dz = p[3][2] - light_z
                if dx * dx + dy * dy + dz * dz <= range_sq:
                    near.append((mesh, tr))
            if near:
                result[mid] = near
        return result

    def _render_geometry_with_groups(self, vp: np.ndarray, fbo, groups: dict, resolution: int = 1024):
        if not groups:
            return
        fbo.clear(depth=1.0)
        fbo.use()
        self._ctx.viewport = (0, 0, resolution, resolution)
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.depth_mask = True
        self._ctx.disable(moderngl.CULL_FACE)
        prog = self._prog
        prog["u_light_vp"].write(vp.tobytes())
        supports_instancing = self._supports_instancing_cached(prog)
        names = self._uniform_names(prog)
        use_inst = "u_use_instancing" in names
        umodel = "u_model" in names
        prog_id = id(prog)
        if supports_instancing:
            if use_inst:
                prog["u_use_instancing"].value = 1
            for mesh_id, group in groups.items():
                mesh, _ = group[0]
                n = len(group)
                if n == 1:
                    tr = group[0][1]
                    try:
                        wm = tr.world_matrix
                    except Exception:
                        continue
                    key = (mesh_id, prog_id)
                    vbo = self._build_shadow_instance_vbo(key, [wm])
                    vao = self._get_shadow_vao(prog, mesh, vbo)
                    vao.render(instances=1)
                    continue
                key = (mesh_id, prog_id)
                try:
                    model_mats = [tr.world_matrix for _, tr in group]
                except Exception:
                    continue
                vbo = self._build_shadow_instance_vbo(key, model_mats)
                vao = self._get_shadow_vao(prog, mesh, vbo)
                vao.render(instances=n)
        else:
            if use_inst:
                prog["u_use_instancing"].value = 0
            for _mesh_id, group in groups.items():
                mesh, _ = group[0]
                if umodel:
                    for _, tr in group:
                        try:
                            prog["u_model"].write(tr.world_matrix.to_f32().tobytes())
                        except Exception:
                            continue
                        mesh.render(prog)
                else:
                    for _, tr in group:
                        mesh.render(prog)
        self._ctx.enable(moderngl.CULL_FACE)

    def collect_shadow_data(self, scene, meshes: dict) -> list[tuple]:
        self._get_mesh = lambda k: meshes.get(k)
        result = self._build_renderable_shadow(scene)
        self._get_mesh = None
        return result

    def _bind_bone_ssbo(self, flat: np.ndarray):
        n = flat.shape[0]
        needed = max(64, n) * 64
        if self._skinned_bone_ssbo is not None and self._skinned_bone_ssbo_cap >= needed:
            data = flat.tobytes()
            if self._skinned_bone_ssbo.size < len(data):
                try:
                    self._skinned_bone_ssbo.release()
                except Exception:
                    pass
                self._skinned_bone_ssbo = self._ctx.buffer(reserve=len(data) + 64)
                self._skinned_bone_ssbo_cap = len(data) + 64
            self._skinned_bone_ssbo.write(data)
        else:
            if self._skinned_bone_ssbo is not None:
                try:
                    self._skinned_bone_ssbo.release()
                except Exception:
                    pass
            self._skinned_bone_ssbo = self._ctx.buffer(reserve=needed)
            self._skinned_bone_ssbo_cap = needed
            self._skinned_bone_ssbo.write(flat.tobytes())
        self._skinned_bone_ssbo.bind_to_storage_buffer(6)

    def _maybe_render_skinned(self, vp: np.ndarray, fbo, resolution: int):
        if not self._pending_skinned:
            return
        if self._pending_scene is None:
            return
        prog = self._prog
        fbo.use()
        fbo.viewport = (0, 0, resolution, resolution)
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.depth_mask = True
        self._ctx.disable(moderngl.CULL_FACE)
        prog["u_light_vp"].write(vp.tobytes())
        names = self._uniform_names(prog)
        if "u_use_instancing" in names:
            prog["u_use_instancing"].value = 0
        if "u_use_skinning" in names:
            prog["u_use_skinning"].value = 1
        skinning_cache = self._skinning_cache
        for entry in self._pending_skinned:
            mesh, ent, armature, wm = entry[0], entry[1], entry[2], entry[3]
            if armature is None or len(armature.bone_offset_matrices) == 0:
                continue
            cache_key = ent._id
            cached = skinning_cache.get(cache_key) if skinning_cache is not None else None
            if cached is not None:
                flat, n_bones = cached
            else:
                flat, n_bones = armature.compute_skinning_buffer(self._pending_scene, wm)
                if n_bones > 0 and skinning_cache is not None:
                    skinning_cache[cache_key] = (flat, n_bones)
            if n_bones == 0:
                continue
            self._bind_bone_ssbo(flat)
            if "u_bone_count" in names:
                prog["u_bone_count"].value = int(n_bones)
            if "u_model" in names:
                prog["u_model"].write(wm.to_f32().tobytes())
            mesh.render(prog)
        if "u_use_skinning" in names:
            prog["u_use_skinning"].value = 0
        self._ctx.enable(moderngl.CULL_FACE)

    def render_shadow_pass(self, renderable_shadow, lights, cam_near: float, cam_far: float, cam_fov: float,
                           aspect: float, view_mat: Mat4, meshes: object = None,
                           skinned_entries: list = None, scene: object = None,
                           skinning_cache: dict = None) -> dict:
        if not self._prog:
            return {}
        self._pending_skinned = skinned_entries or []
        self._pending_scene = scene
        self._skinning_cache = skinning_cache
        if not renderable_shadow:
            self._flat_n = 0
            self._flat_mesh_map = {}
            self.reset_shadow_state()
            self._cache_uniform_bytes()
            return {}
        try:
            self._prepare_flat(renderable_shadow)
        except Exception:
            self._flat_n = 0
            self._flat_mesh_map = {}
        shadow_groups = self._build_shadow_groups(renderable_shadow)
        flags = self._type_flags
        if flags.get('directional', True):
            for l, lt in lights:
                if l.light_type == LightType.DIRECTIONAL and l.cast_shadows:
                    self._render_directional_shadow(lt, shadow_groups,
                                                    cam_near, cam_far, cam_fov, aspect, view_mat)
                    break
            else:
                self._cascade_splits = [0.0] * 4
        else:
            self._cascade_splits = [0.0] * 4
        try:
            vd = view_mat._d
            if _mat4_inv_fast is not None:
                inv_d = _mat4_inv_fast(np.ascontiguousarray(vd, dtype=np.float64))
            else:
                inv_d = np.linalg.inv(vd)
            cam_x = float(inv_d[3, 0])
            cam_y = float(inv_d[3, 1])
            cam_z = float(inv_d[3, 2])
        except Exception:
            cam_x = 0.0
            cam_y = 0.0
            cam_z = 0.0
        if not self._point_shadow_maps:
            self._create_point_shadow_resources()
        if flags.get('point', True):
            pc = []
            for l, lt in lights:
                if l.light_type == LightType.POINT and l.cast_shadows:
                    try:
                        px, py, pz = _tr_pos_xyz(lt)
                    except Exception:
                        continue
                    dx = px - cam_x
                    dy = py - cam_y
                    dz = pz - cam_z
                    pc.append((l, lt, dx * dx + dy * dy + dz * dz))
            pc.sort(key=lambda x: x[2])
            self._point_shadow_count = min(len(pc), MAX_POINT_SHADOWS)
            for slot in range(self._point_shadow_count):
                l, lt, _ = pc[slot]
                self._render_point_shadow_for_slot(slot, l, lt, shadow_groups, lights)
            for slot in range(self._point_shadow_count, MAX_POINT_SHADOWS):
                self._point_shadow_light_indices[slot] = -1
            self._has_point_shadow = self._point_shadow_count > 0
        else:
            self._point_shadow_count = 0
            self._has_point_shadow = False
            for slot in range(MAX_POINT_SHADOWS):
                self._point_shadow_light_indices[slot] = -1
        if not self._spot_shadow_maps:
            self._create_spot_shadow_resources()
        if flags.get('spot', True):
            sc = []
            for l, lt in lights:
                if l.light_type == LightType.SPOT and l.cast_shadows:
                    try:
                        px, py, pz = _tr_pos_xyz(lt)
                    except Exception:
                        continue
                    dx = px - cam_x
                    dy = py - cam_y
                    dz = pz - cam_z
                    sc.append((l, lt, dx * dx + dy * dy + dz * dz))
            sc.sort(key=lambda x: x[2])
            self._spot_shadow_count = min(len(sc), MAX_SPOT_SHADOWS)
            for slot in range(self._spot_shadow_count):
                l, lt, _ = sc[slot]
                self._render_spot_shadow_for_slot(slot, l, lt, shadow_groups, lights)
            for slot in range(self._spot_shadow_count, MAX_SPOT_SHADOWS):
                self._spot_shadow_light_indices[slot] = -1
            self._has_spot_shadow = self._spot_shadow_count > 0
        else:
            self._spot_shadow_count = 0
            self._has_spot_shadow = False
            for slot in range(MAX_SPOT_SHADOWS):
                self._spot_shadow_light_indices[slot] = -1
        if flags.get('area', True):
            best_area = None
            best_area_dist = float('inf')
            for l, lt in lights:
                if l.light_type == LightType.AREA and l.cast_shadows:
                    try:
                        px, py, pz = _tr_pos_xyz(lt)
                    except Exception:
                        continue
                    dx = px - cam_x
                    dy = py - cam_y
                    dz = pz - cam_z
                    d = dx * dx + dy * dy + dz * dz
                    if d < best_area_dist:
                        best_area_dist = d
                        best_area = (l, lt)
            if best_area:
                self._render_area_shadow(best_area[0], best_area[1], shadow_groups, lights)
            else:
                self._has_area_shadow = False
        else:
            self._has_area_shadow = False
        try:
            self._cache_uniform_bytes()
        except Exception:
            pass
        return shadow_groups

    def reset_shadow_state(self):
        self._cascade_splits = [0.0] * 4
        self._has_point_shadow = False
        self._point_shadow_count = 0
        self._has_spot_shadow = False
        self._spot_shadow_count = 0
        self._has_area_shadow = False

    def _compute_cascade_splits(self, cam_near: float, cam_far: float) -> list[float]:
        if not self._cascade_splits_norm:
            return self._cascade_distances(cam_near, cam_far)
        near_z = max(cam_near, 0.01)
        far_z = max(near_z + 0.1, min(cam_far, self._shadow_distance))
        norm = self._cascade_splits_norm
        splits = []
        for i in range(self._cascade_count):
            if i < len(norm):
                splits.append(near_z + (far_z - near_z) * float(norm[i]))
            else:
                splits.append(far_z)
        return splits

    def _cascade_distances(self, cam_near: float, cam_far: float) -> list[float]:
        near_z = max(cam_near, 0.01)
        far_z = max(near_z + 0.1, min(cam_far, self._shadow_distance))
        if self._cascade_count <= 1:
            return [far_z, far_z, far_z, far_z]
        if self._cascade_count == 2:
            s0 = near_z + (far_z - near_z) * 0.2
            return [s0, far_z, far_z, far_z]
        if self._cascade_count == 3:
            s0 = near_z + (far_z - near_z) * 0.1
            s1 = near_z + (far_z - near_z) * 0.3
            return [s0, s1, far_z, far_z]
        span = far_z - near_z
        lam = 0.9
        splits = []
        for i in range(1, 5):
            p = i / 4.0
            log = near_z * (far_z / near_z) ** p if near_z > 0 else near_z + span * p
            uni = near_z + span * p
            s = lam * log + (1.0 - lam) * uni
            splits.append(s)
        splits[-1] = far_z
        return splits

    def _render_directional_shadow(self, sun_transform, shadow_groups,
                                   cam_near, cam_far, cam_fov, aspect, view_mat):
        try:
            ld_x, ld_y, ld_z = _tr_fwd_xyz(sun_transform)
        except Exception:
            try:
                light_dir = sun_transform.forward.normalized()
                ld_x, ld_y, ld_z = light_dir.x, light_dir.y, light_dir.z
            except Exception:
                return
        try:
            vd = view_mat._d
            if _mat4_inv_fast is not None:
                inv_view = _mat4_inv_fast(np.ascontiguousarray(vd, dtype=np.float64))
            else:
                inv_view = np.linalg.inv(vd)
        except Exception:
            inv_view = np.linalg.inv(view_mat._d)
        splits = self._compute_cascade_splits(cam_near, cam_far)
        self._cascade_splits = splits
        near_z = max(cam_near, 0.01)
        prog = self._prog
        try:
            prog["u_light_vp"].write(self._vp_f32_buf.tobytes())
        except Exception:
            pass
        self._temporal_frame += 1
        try:
            lv = self._last_view
            cam_moved = True
            if self._last_view_valid and lv.shape == vd.shape:
                if bool(np.array_equal(lv, vd)):
                    px, py, pz = self._last_light_dir_xyz
                    if abs(px - ld_x) < 0.002 and abs(py - ld_y) < 0.002 and abs(pz - ld_z) < 0.002:
                        cam_moved = False
            try:
                np.copyto(self._last_view, vd)
            except Exception:
                try:
                    self._last_view = np.array(vd, dtype=np.float64, copy=True)
                except Exception:
                    pass
            self._last_view_valid = True
            self._last_light_dir_xyz = (ld_x, ld_y, ld_z)
        except Exception:
            cam_moved = True
        if cam_moved:
            self._shadow_cache_valid = False
        use_flat = self._flat_n > 0
        try:
            light_dir_v = Vec3(ld_x, ld_y, ld_z)
        except Exception:
            light_dir_v = None
        first_cascade = True
        supports_instancing = self._supports_instancing_cached(prog)
        names = self._uniform_names(prog)
        use_inst = "u_use_instancing" in names
        umodel = "u_model" in names
        prog_id = id(prog)
        mmap = self._flat_mesh_map
        for ci in range(self._cascade_count):
            res = self._cascade_resolutions[ci] if ci < len(self._cascade_resolutions) else self._shadow_resolution
            if not cam_moved:
                if ci == 3 and (self._temporal_frame % 3) != 0:
                    near_z = splits[ci]
                    continue
                if ci == 2 and (self._temporal_frame % 2) != 0:
                    near_z = splits[ci]
                    continue
            if _HAS_CYTHON:
                try:
                    compute_frustum_corners_out(
                        near_z, splits[ci], cam_fov, aspect,
                        inv_view, self._frustum_corners_buf
                    )
                    build_directional_cascade_fast(
                        ld_x, ld_y, ld_z,
                        self._frustum_corners_buf, splits[ci] - near_z, res,
                        self._cascade_vps_raw[ci]
                    )
                    np.copyto(self._vp_f32_buf, self._cascade_vps_raw[ci])
                except Exception:
                    corners = self._get_frustum_corners(near_z, splits[ci], cam_fov, aspect, inv_view)
                    vp = self._build_directional_cascade(light_dir_v, corners, splits[ci] - near_z, res)
                    np.copyto(self._vp_f32_buf, vp)
            else:
                corners = self._get_frustum_corners(near_z, splits[ci], cam_fov, aspect, inv_view)
                vp = self._build_directional_cascade(light_dir_v, corners, splits[ci] - near_z, res)
                np.copyto(self._vp_f32_buf, vp)
            self._light_space_matrices[ci] = self._vp_f32_buf.copy()
            if use_flat:
                thr = 0.0
                if ci == 2:
                    thr = 0.4
                elif ci >= 3:
                    thr = 0.8
                try:
                    if thr > 0.0:
                        cnt = self._cull_flat_min_count(self._vp_f32_buf, thr)
                    else:
                        cnt = self._cull_flat_count(self._vp_f32_buf)
                except Exception:
                    cnt = 0
                if cnt > 0:
                    try:
                        vis = self._flat_out[:cnt]
                        vis_mesh = self._flat_mesh_ids[:self._flat_n][vis]
                        uniq = np.unique(vis_mesh)
                        self._shadow_fbos[ci].clear(depth=1.0)
                        self._shadow_fbos[ci].use()
                        self._ctx.viewport = (0, 0, res, res)
                        if first_cascade:
                            self._ctx.enable(moderngl.DEPTH_TEST)
                            self._ctx.depth_mask = True
                            self._ctx.disable(moderngl.CULL_FACE)
                            first_cascade = False
                        prog["u_light_vp"].write(self._vp_f32_buf.tobytes())
                        if supports_instancing:
                            if use_inst:
                                prog["u_use_instancing"].value = 1
                            for mid in uniq:
                                mid_i = int(mid)
                                mesh = mmap.get(mid_i)
                                if mesh is None:
                                    continue
                                sel = vis[vis_mesh == mid_i]
                                if sel.size == 0:
                                    continue
                                chunk = self._flat_mats[sel]
                                vbo = self._upload_instanced_mats((mid_i, prog_id), chunk)
                                vao = self._get_shadow_vao(prog, mesh, vbo)
                                vao.render(instances=int(sel.size))
                        else:
                            if use_inst:
                                prog["u_use_instancing"].value = 0
                            for mid in uniq:
                                mid_i = int(mid)
                                mesh = mmap.get(mid_i)
                                if mesh is None:
                                    continue
                                sel = vis[vis_mesh == mid_i]
                                if sel.size == 0:
                                    continue
                                if umodel:
                                    for fi in sel:
                                        prog["u_model"].write(self._flat_mats[int(fi)].tobytes())
                                        mesh.render(prog)
                                else:
                                    for _fi in sel:
                                        mesh.render(prog)
                        self._maybe_render_skinned(self._vp_f32_buf, self._shadow_fbos[ci], res)
                    except Exception:
                        pass
                near_z = splits[ci]
                continue
            if _HAS_CYTHON and shadow_groups:
                try:
                    culled = frustum_cull_shadow_groups(shadow_groups, self._vp_f32_buf)
                except Exception:
                    culled = shadow_groups
            else:
                culled = shadow_groups
            if culled:
                self._shadow_fbos[ci].clear(depth=1.0)
                self._shadow_fbos[ci].use()
                self._ctx.viewport = (0, 0, res, res)
                if first_cascade:
                    self._ctx.enable(moderngl.DEPTH_TEST)
                    self._ctx.depth_mask = True
                    self._ctx.disable(moderngl.CULL_FACE)
                    first_cascade = False
                prog["u_light_vp"].write(self._vp_f32_buf.tobytes())
                for mesh_id, group in culled.items():
                    mesh, _ = group[0]
                    n = len(group)
                    if supports_instancing:
                        key = (mesh_id, prog_id)
                        model_mats = [tr.world_matrix for _, tr in group]
                        vbo = self._build_shadow_instance_vbo(key, model_mats)
                        vao = self._get_shadow_vao(prog, mesh, vbo)
                        if use_inst:
                            prog["u_use_instancing"].value = 1
                        vao.render(instances=n)
                    else:
                        if use_inst:
                            prog["u_use_instancing"].value = 0
                        for _, tr in group:
                            prog["u_model"].write(tr.world_matrix.to_f32().tobytes())
                            mesh.render(prog)
                self._maybe_render_skinned(self._vp_f32_buf, self._shadow_fbos[ci], res)
            near_z = splits[ci]
        if not first_cascade:
            self._ctx.enable(moderngl.CULL_FACE)

    def _get_frustum_corners(self, near_z: float, far_z: float, cam_fov: float,
                             aspect: float, inv_view: np.ndarray) -> list[np.ndarray]:
        if _HAS_CYTHON:
            return compute_frustum_corners(near_z, far_z, cam_fov, aspect, inv_view)
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

    def _build_directional_cascade(self, light_dir: Vec3, corners: list[np.ndarray],
                                   depth_span: float, shadow_res: int = None) -> np.ndarray:
        n = len(corners)
        cx = cy = cz = 0.0
        for c in corners:
            cx += c[0]
            cy += c[1]
            cz += c[2]
        cx /= n
        cy /= n
        cz /= n
        radius2 = 0.0
        for c in corners:
            dx = c[0] - cx
            dy = c[1] - cy
            dz = c[2] - cz
            r2 = dx * dx + dy * dy + dz * dz
            if r2 > radius2:
                radius2 = r2
        radius = math.sqrt(radius2)
        radius = max(radius, 0.25)
        radius = math.ceil(radius * 16.0) / 16.0
        light_pos = Vec3(cx, cy, cz) - light_dir * max(radius * 2.0, depth_span + 10.0)
        light_up = Vec3(0.0, 1.0, 0.0)
        if abs(light_dir.dot(light_up)) > 0.999:
            light_up = Vec3(0.0, 0.0, 1.0)
        view = Mat4.look_at(light_pos, Vec3(cx, cy, cz), light_up)
        m = [list(map(float, row)) for row in view._d.tolist()]
        center_light = (
            cx * m[0][0] + cy * m[1][0] + cz * m[2][0] + m[3][0],
            cx * m[0][1] + cy * m[1][1] + cz * m[2][1] + m[3][1],
            cx * m[0][2] + cy * m[1][2] + cz * m[2][2] + m[3][2],
            cx * m[0][3] + cy * m[1][3] + cz * m[2][3] + m[3][3],
        )
        res = shadow_res if shadow_res is not None else self._shadow_resolution
        texel_size = (radius * 2.0) / max(1, res)
        cx_l = math.floor(center_light[0] / texel_size) * texel_size
        cy_l = math.floor(center_light[1] / texel_size) * texel_size
        left = cx_l - radius
        right = cx_l + radius
        bottom = cy_l - radius
        top = cy_l + radius
        min_z = 1e300
        max_z = -1e300
        for c in corners:
            px = c[0] * m[0][0] + c[1] * m[1][0] + c[2] * m[2][0] + m[3][0]
            py = c[0] * m[0][1] + c[1] * m[1][1] + c[2] * m[2][1] + m[3][1]
            pz = c[0] * m[0][2] + c[1] * m[1][2] + c[2] * m[2][2] + m[3][2]
            pw = c[0] * m[0][3] + c[1] * m[1][3] + c[2] * m[2][3] + m[3][3]
            pz = pz / pw if pw != 0.0 else 0.0
            if pz < min_z:
                min_z = pz
            if pz > max_z:
                max_z = pz
        z_margin = max(depth_span * 0.45, 6.0)
        n_val = max(-max_z - z_margin, 0.01)
        f_val = max(-min_z + z_margin, n_val + 0.01)
        proj = Mat4.orthographic(left, right, bottom, top, n_val, f_val)
        return view._d @ proj._d

    def _point_proj(self, light_range: float):
        key = round(float(light_range), 3)
        p = self._point_proj_cache.get(key)
        if p is None:
            p = Mat4.perspective(90.0, 1.0, 0.1, max(float(light_range), 0.1))._d
            self._point_proj_cache[key] = p
            if len(self._point_proj_cache) > 16:
                try:
                    self._point_proj_cache.pop(next(iter(self._point_proj_cache)))
                except Exception:
                    pass
        return p

    def _render_point_shadow_for_slot(self, slot, point_light, point_transform, shadow_groups, lights):
        try:
            lp_x, lp_y, lp_z = _tr_pos_xyz(point_transform)
            light_pos = Vec3(lp_x, lp_y, lp_z)
        except Exception:
            light_pos = point_transform.position
            lp_x, lp_y, lp_z = light_pos.x, light_pos.y, light_pos.z
        light_range = max(float(getattr(point_light, 'range', 10.0)), 0.1)
        lr2 = light_range * light_range
        self._point_shadow_light_positions[slot] = light_pos
        self._point_shadow_light_ranges[slot] = light_range
        self._point_shadow_light_indices[slot] = next(
            (i for i, (l, lt) in enumerate(lights) if l is point_light and lt is point_transform), -1
        )
        if self._flat_n == 0:
            return
        proj_np = self._point_proj(light_range)
        prog = self._prog
        shadow_res = self._point_shadow_resolution
        supports_instancing = self._supports_instancing_cached(prog)
        names = self._uniform_names(prog)
        use_inst = "u_use_instancing" in names
        umodel = "u_model" in names
        prog_id = id(prog)
        mmap = self._flat_mesh_map
        base = slot * 6
        self._ctx.viewport = (0, 0, shadow_res, shadow_res)
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.depth_mask = True
        self._ctx.disable(moderngl.CULL_FACE)
        if use_inst:
            prog["u_use_instancing"].value = 1 if supports_instancing else 0
        for face_idx in range(6):
            try:
                fbo = self._point_shadow_fbos[base + face_idx]
                fbo.use()
                fbo.clear(depth=1.0)
            except Exception:
                pass
        for face_idx in range(6):
            dx, dy, dz = _POINT_FACE_DIRS[face_idx]
            ux, uy, uz = _POINT_FACE_UPS[face_idx]
            try:
                face_dir = Vec3(dx, dy, dz)
                face_up = Vec3(ux, uy, uz)
                vp = (Mat4.look_at(light_pos, light_pos + face_dir, face_up)._d @ proj_np).astype(np.float32)
            except Exception:
                continue
            self._point_light_vps[base + face_idx] = vp
            try:
                cnt = self._cull_flat_range_count(vp, lp_x, lp_y, lp_z, lr2, 0.0)
            except Exception:
                cnt = 0
            if cnt <= 0:
                continue
            try:
                vis = self._flat_out[:cnt]
                vis_mesh = self._flat_mesh_ids[:self._flat_n][vis]
                uniq = np.unique(vis_mesh)
            except Exception:
                continue
            try:
                fbo = self._point_shadow_fbos[base + face_idx]
                fbo.use()
            except Exception:
                continue
            prog["u_light_vp"].write(vp.tobytes())
            if supports_instancing:
                for mid in uniq:
                    mid_i = int(mid)
                    mesh = mmap.get(mid_i)
                    if mesh is None:
                        continue
                    sel = vis[vis_mesh == mid_i]
                    if sel.size == 0:
                        continue
                    chunk = self._flat_mats[sel]
                    vbo = self._upload_instanced_mats((mid_i, prog_id), chunk)
                    vao = self._get_shadow_vao(prog, mesh, vbo)
                    vao.render(instances=int(sel.size))
            else:
                for mid in uniq:
                    mid_i = int(mid)
                    mesh = mmap.get(mid_i)
                    if mesh is None:
                        continue
                    sel = vis[vis_mesh == mid_i]
                    if sel.size == 0:
                        continue
                    if umodel:
                        for fi in sel:
                            prog["u_model"].write(self._flat_mats[int(fi)].tobytes())
                            mesh.render(prog)
                    else:
                        for _fi in sel:
                            mesh.render(prog)
        self._ctx.enable(moderngl.CULL_FACE)

    def _spot_proj(self, fov: float, far: float):
        key = (round(float(fov), 2), round(float(far), 2))
        p = self._spot_proj_cache.get(key)
        if p is None:
            p = Mat4.perspective(max(float(fov), 1.0), 1.0, 0.1, max(float(far), 0.2))._d
            self._spot_proj_cache[key] = p
            if len(self._spot_proj_cache) > 16:
                try:
                    self._spot_proj_cache.pop(next(iter(self._spot_proj_cache)))
                except Exception:
                    pass
        return p

    def _render_spot_shadow_for_slot(self, slot, spot_light, spot_transform, shadow_groups, lights):
        try:
            lp_x, lp_y, lp_z = _tr_pos_xyz(spot_transform)
            light_pos = Vec3(lp_x, lp_y, lp_z)
            fdx, fdy, fdz = _tr_fwd_xyz(spot_transform)
            light_dir = Vec3(fdx, fdy, fdz)
        except Exception:
            light_pos = spot_transform.position
            light_dir = spot_transform.forward.normalized()
            lp_x, lp_y, lp_z = light_pos.x, light_pos.y, light_pos.z
        light_range = max(float(getattr(spot_light, 'range', 10.0)), 0.1)
        spot_fov = max(float(getattr(spot_light, 'spot_angle', 30.0)) * 2.0, 1.0)
        lr2 = light_range * light_range
        if self._flat_n == 0:
            try:
                fbo = self._spot_shadow_fbos[slot]
                fbo.use()
                self._ctx.viewport = (0, 0, self._spot_shadow_resolution, self._spot_shadow_resolution)
                fbo.clear(depth=1.0)
            except Exception:
                pass
            return
        proj_d = self._spot_proj(spot_fov, light_range)
        try:
            view_d = Mat4.look_at(light_pos, light_pos + light_dir, Vec3.up())._d
            vp = (view_d @ proj_d).astype(np.float32)
        except Exception:
            return
        self._spot_shadow_vps[slot] = vp
        self._spot_shadow_light_indices[slot] = next(
            (i for i, (l, lt) in enumerate(lights) if l is spot_light and lt is spot_transform), -1
        )
        try:
            cnt = self._cull_flat_range_count(vp, lp_x, lp_y, lp_z, lr2, 0.0)
        except Exception:
            cnt = 0
        if cnt > 0:
            try:
                self._draw_flat_visible(vp, self._spot_shadow_fbos[slot], self._spot_shadow_resolution, cnt)
                self._maybe_render_skinned(vp, self._spot_shadow_fbos[slot], self._spot_shadow_resolution)
            except Exception:
                pass
        else:
            try:
                fbo = self._spot_shadow_fbos[slot]
                fbo.use()
                self._ctx.viewport = (0, 0, self._spot_shadow_resolution, self._spot_shadow_resolution)
                fbo.clear(depth=1.0)
            except Exception:
                pass

    def _render_area_shadow(self, area_light, area_transform, shadow_groups, lights):
        if not self._area_shadow_map:
            self._create_area_shadow_resources()
            if not self._area_shadow_map:
                self._has_area_shadow = False
                return
        try:
            lp_x, lp_y, lp_z = _tr_pos_xyz(area_transform)
            light_pos = Vec3(lp_x, lp_y, lp_z)
            fdx, fdy, fdz = _tr_fwd_xyz(area_transform)
            light_dir = Vec3(fdx, fdy, fdz)
            udx, udy, udz = _tr_up_xyz(area_transform)
            light_up = Vec3(udx, udy, udz)
        except Exception:
            light_pos = area_transform.position
            light_dir = area_transform.forward.normalized()
            light_up = area_transform.up.normalized()
            lp_x, lp_y, lp_z = light_pos.x, light_pos.y, light_pos.z
        try:
            if abs(light_dir.dot(light_up)) > 0.999:
                light_up = Vec3(0.0, 0.0, 1.0)
        except Exception:
            pass
        light_range = max(float(getattr(area_light, 'range', 10.0)), 0.1)
        near_plane = 0.1
        far_plane = light_range
        self._area_light_near = near_plane
        self._area_light_far = far_plane
        try:
            aw = float(getattr(area_light, 'area_width', 1.0))
            ah = float(getattr(area_light, 'area_height', 1.0))
        except Exception:
            aw = 1.0
            ah = 1.0
        fov = max(90.0, min(150.0, math.degrees(2.0 * math.atan2(max(aw, ah) * 0.5, near_plane))))
        fov_rad = math.radians(fov)
        tan_half_fov = math.tan(fov_rad * 0.5)
        self._area_light_fov_scale = float(1.0 / max(1e-6, (2.0 * tan_half_fov)))
        try:
            self._area_shadow_bias = float(getattr(area_light, 'area_shadow_bias', 0.005))
        except Exception:
            self._area_shadow_bias = 0.005
        try:
            view_d = Mat4.look_at(light_pos, light_pos + light_dir, light_up)._d
            proj_d = Mat4.perspective(fov, 1.0, near_plane, far_plane)._d
            vp = (view_d @ proj_d).astype(np.float32)
        except Exception:
            self._has_area_shadow = False
            return
        self._area_light_vp = vp
        self._has_area_shadow = True
        self._area_light_pos = light_pos
        self._area_light_range = light_range
        self._area_light_size = max(aw, ah) * 0.5
        self._area_light_idx = next(
            (i for i, (l, lt) in enumerate(lights) if l is area_light and lt is area_transform), -1
        )
        lr2 = light_range * light_range
        if self._flat_n == 0:
            return
        try:
            cnt = self._cull_flat_range_count(vp, lp_x, lp_y, lp_z, lr2, 0.0)
        except Exception:
            cnt = 0
        if cnt > 0:
            try:
                self._draw_flat_visible(vp, self._area_shadow_fbo, self._area_shadow_resolution, cnt)
                self._maybe_render_skinned(vp, self._area_shadow_fbo, self._area_shadow_resolution)
            except Exception:
                pass

    def render_projector_shadows(self, projectors, renderable_shadow, shadow_groups: dict = None):
        if self._flat_n == 0:
            try:
                if shadow_groups is None:
                    shadow_groups = self._build_shadow_groups(renderable_shadow)
            except Exception:
                shadow_groups = {}
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
                try:
                    light_pos_vec = Vec3(float(pj.position[0]), float(pj.position[1]), float(pj.position[2]))
                    light_dir_vec = Vec3(float(pj.direction[0]), float(pj.direction[1]), float(pj.direction[2])).normalized()
                    up_vec = Vec3(float(pj.up[0]), float(pj.up[1]), float(pj.up[2]))
                    view = Mat4.look_at(light_pos_vec, light_pos_vec + light_dir_vec, up_vec)
                    proj = Mat4.perspective(max(float(pj.spot_angle), 1.0), float(pj.aspect_ratio), max(float(pj.near_plane), 0.01), max(float(pj.far_plane), max(float(pj.near_plane), 0.01) + 0.1))
                    vp = (view._d @ proj._d).astype(np.float32)
                except Exception:
                    continue
                self._projector_light_vps[i] = vp
                self._has_projector_shadow[i] = True
                try:
                    lr2 = float(pj.far_plane) * float(pj.far_plane)
                    filtered = self._filter_by_range(shadow_groups, float(pj.position[0]), float(pj.position[1]), float(pj.position[2]), lr2)
                except Exception:
                    filtered = shadow_groups
                if filtered:
                    try:
                        self._render_geometry_with_groups(vp, self._projector_shadow_fbos[i], filtered, resolution=self._shadow_resolution)
                    except Exception:
                        pass
            return
        for i, pj in enumerate(projectors[:2]):
            if not pj.cast_shadows:
                self._has_projector_shadow[i] = False
                continue
            if len(self._projector_shadow_maps) <= i:
                self._create_projector_shadow_resources()
            try:
                lp_x = float(pj.position[0])
                lp_y = float(pj.position[1])
                lp_z = float(pj.position[2])
                dx = float(pj.direction[0])
                dy = float(pj.direction[1])
                dz = float(pj.direction[2])
                inv = 1.0 / max(1e-12, math.sqrt(dx * dx + dy * dy + dz * dz))
                light_pos_vec = Vec3(lp_x, lp_y, lp_z)
                light_dir_vec = Vec3(dx * inv, dy * inv, dz * inv)
                up_vec = Vec3(float(pj.up[0]), float(pj.up[1]), float(pj.up[2]))
                spot_fov = max(float(pj.spot_angle), 1.0)
                near_plane = max(float(pj.near_plane), 0.01)
                far_plane = max(float(pj.far_plane), near_plane + 0.1)
                view = Mat4.look_at(light_pos_vec, light_pos_vec + light_dir_vec, up_vec)
                proj = Mat4.perspective(spot_fov, float(pj.aspect_ratio), near_plane, far_plane)
                vp = (view._d @ proj._d).astype(np.float32)
            except Exception:
                continue
            self._projector_light_vps[i] = vp
            self._has_projector_shadow[i] = True
            lr2 = far_plane * far_plane
            try:
                cnt = self._cull_flat_range_count(vp, lp_x, lp_y, lp_z, lr2, 0.0)
            except Exception:
                cnt = 0
            if cnt > 0:
                try:
                    self._draw_flat_visible(vp, self._projector_shadow_fbos[i], self._shadow_resolution, cnt)
                except Exception:
                    pass
        for i in range(len(projectors[:2]), 2):
            try:
                self._has_projector_shadow[i] = False
            except Exception:
                pass

    def _cache_uniform_bytes(self):
        try:
            cm = self._cascade_matrices_buf
            for ci in range(self._cascade_count):
                np.copyto(cm[ci], self._light_space_matrices[ci])
            self._cascade_matrices_bytes = cm.tobytes()
        except Exception:
            pass
        try:
            cs = self._cascade_splits_buf
            cs[0] = self._cascade_splits[0]
            cs[1] = self._cascade_splits[1]
            cs[2] = self._cascade_splits[2]
            cs[3] = self._cascade_splits[3]
            self._cascade_splits_bytes = cs.tobytes()
        except Exception:
            pass
        try:
            pv = self._point_vps_buf
            for slot in range(self._point_shadow_count):
                base = slot * 6
                for fi in range(6):
                    np.copyto(pv[base + fi], self._point_light_vps[base + fi])
            self._point_vps_bytes = pv.tobytes()
        except Exception:
            pass
        try:
            for slot in range(self._point_shadow_count):
                try:
                    pa = self._point_shadow_light_positions[slot]
                    self._point_pos_buf[slot][0] = float(pa.x)
                    self._point_pos_buf[slot][1] = float(pa.y)
                    self._point_pos_buf[slot][2] = float(pa.z)
                except Exception:
                    pass
                try:
                    self._point_range_buf[slot] = float(self._point_shadow_light_ranges[slot])
                except Exception:
                    pass
                try:
                    self._point_idx_buf[slot] = int(self._point_shadow_light_indices[slot])
                except Exception:
                    pass
            self._point_pos_bytes = self._point_pos_buf.tobytes()
            self._point_range_bytes = self._point_range_buf.tobytes()
            self._point_idx_bytes = self._point_idx_buf.tobytes()
        except Exception:
            pass
        try:
            sv = self._spot_vps_buf
            for slot in range(self._spot_shadow_count):
                np.copyto(sv[slot], self._spot_shadow_vps[slot])
            self._spot_vps_bytes = sv.tobytes()
        except Exception:
            pass
        try:
            for slot in range(self._spot_shadow_count):
                try:
                    self._spot_idx_buf[slot] = int(self._spot_shadow_light_indices[slot])
                except Exception:
                    pass
            self._spot_idx_bytes = self._spot_idx_buf.tobytes()
        except Exception:
            pass
        try:
            self._area_vp_bytes = np.ascontiguousarray(self._area_light_vp, dtype=np.float32).tobytes()
        except Exception:
            pass
        try:
            af = self._area_nearfar_buf
            af[0] = float(self._area_light_near)
            af[1] = float(self._area_light_far)
            self._area_nearfar_bytes = af.tobytes()
        except Exception:
            pass

    def set_uniforms(self, prog):
        names = self._uniform_names(prog)
        has_csm = self._cascade_splits[self._cascade_count - 1] > 0.0
        if has_csm and "u_cascade_count" in names and len(self._shadow_maps) >= self._cascade_count:
            prog["u_cascade_count"].value = self._cascade_count
            if "u_light_space_matrices" in names:
                try:
                    if not self._cascade_matrices_bytes:
                        self._cache_uniform_bytes()
                    prog["u_light_space_matrices"].write(self._cascade_matrices_bytes)
                except Exception:
                    pass
            if "u_cascade_splits" in names:
                try:
                    if not self._cascade_splits_bytes:
                        self._cache_uniform_bytes()
                    prog["u_cascade_splits"].write(self._cascade_splits_bytes)
                except Exception:
                    pass
            for ci in range(self._cascade_count):
                tex_unit = 3 + ci
                try:
                    self._shadow_maps[ci].use(tex_unit)
                except Exception:
                    continue
                si = f"u_shadow_map_{ci}"
                if si in names:
                    prog[si].value = tex_unit
        else:
            if "u_cascade_count" in names:
                prog["u_cascade_count"].value = 0
        if "u_shadow_bias" in names:
            prog["u_shadow_bias"].value = 0.0008
        if self._has_point_shadow and "u_point_shadow_count" in names:
            prog["u_point_shadow_count"].value = self._point_shadow_count
            if "u_point_shadow_vps" in names:
                try:
                    if not self._point_vps_bytes:
                        self._cache_uniform_bytes()
                    prog["u_point_shadow_vps"].write(self._point_vps_bytes)
                except Exception:
                    pass
            point_units = [0] * (MAX_POINT_SHADOWS * 6)
            for slot in range(self._point_shadow_count):
                base = slot * 6
                for fi in range(6):
                    tex_unit = 7 + base + fi
                    try:
                        self._point_shadow_maps[base + fi].use(tex_unit)
                    except Exception:
                        pass
                    point_units[base + fi] = tex_unit
            if "u_point_shadow_maps" in names:
                prog["u_point_shadow_maps"].value = point_units
            if "u_point_shadow_light_positions" in names:
                try:
                    if not self._point_pos_bytes:
                        self._cache_uniform_bytes()
                    prog["u_point_shadow_light_positions"].write(self._point_pos_bytes)
                except Exception:
                    pass
            if "u_point_shadow_light_ranges" in names:
                try:
                    if not self._point_range_bytes:
                        self._cache_uniform_bytes()
                    prog["u_point_shadow_light_ranges"].write(self._point_range_bytes)
                except Exception:
                    pass
            if "u_point_shadow_light_indices" in names:
                try:
                    if not self._point_idx_bytes:
                        self._cache_uniform_bytes()
                    prog["u_point_shadow_light_indices"].write(self._point_idx_bytes)
                except Exception:
                    pass
        else:
            if "u_point_shadow_count" in names:
                prog["u_point_shadow_count"].value = 0
        if self._has_spot_shadow and "u_spot_shadow_count" in names:
            prog["u_spot_shadow_count"].value = self._spot_shadow_count
            if "u_spot_shadow_vps" in names:
                try:
                    if not self._spot_vps_bytes:
                        self._cache_uniform_bytes()
                    prog["u_spot_shadow_vps"].write(self._spot_vps_bytes)
                except Exception:
                    pass
            spot_units = [0] * MAX_SPOT_SHADOWS
            for slot in range(self._spot_shadow_count):
                tex_unit = 7 + MAX_POINT_SHADOWS * 6 + slot
                try:
                    self._spot_shadow_maps[slot].use(tex_unit)
                except Exception:
                    pass
                spot_units[slot] = tex_unit
            if "u_spot_shadow_maps" in names:
                prog["u_spot_shadow_maps"].value = spot_units
            if "u_spot_shadow_light_indices" in names:
                try:
                    if not self._spot_idx_bytes:
                        self._cache_uniform_bytes()
                    prog["u_spot_shadow_light_indices"].write(self._spot_idx_bytes)
                except Exception:
                    pass
        else:
            if "u_spot_shadow_count" in names:
                prog["u_spot_shadow_count"].value = 0
        if self._has_area_shadow and "u_area_shadow_light_index" in names:
            tex_unit = 35
            try:
                self._area_shadow_map.use(tex_unit)
            except Exception:
                pass
            if "u_area_shadow_map" in names:
                prog["u_area_shadow_map"].value = tex_unit
            if "u_area_light_vp" in names:
                try:
                    if not self._area_vp_bytes:
                        self._cache_uniform_bytes()
                    prog["u_area_light_vp"].write(self._area_vp_bytes)
                except Exception:
                    pass
            if "u_area_light_size" in names:
                prog["u_area_light_size"].value = float(self._area_light_size)
            if "u_area_light_fov_scale" in names:
                prog["u_area_light_fov_scale"].value = float(self._area_light_fov_scale)
            if "u_area_light_near_far" in names:
                try:
                    if not self._area_nearfar_bytes:
                        self._cache_uniform_bytes()
                    prog["u_area_light_near_far"].write(self._area_nearfar_bytes)
                except Exception:
                    pass
            if "u_area_shadow_light_index" in names:
                prog["u_area_shadow_light_index"].value = self._area_light_idx if self._area_light_idx >= 0 else -1
            if "u_area_shadow_bias" in names:
                prog["u_area_shadow_bias"].value = float(self._area_shadow_bias)
        else:
            if "u_area_shadow_light_index" in names:
                prog["u_area_shadow_light_index"].value = -1
        for i in range(2):
            suf = f"u_pj_{i}_shadow_map"
            if self._has_projector_shadow[i] and suf in names:
                tex_unit = 36 + i
                if i < len(self._projector_shadow_maps):
                    try:
                        self._projector_shadow_maps[i].use(tex_unit)
                    except Exception:
                        pass
                    prog[suf].value = tex_unit
                vpn = f"u_pj_{i}_shadow_vp"
                if vpn in names:
                    try:
                        prog[vpn].write(self._projector_light_vps[i].tobytes())
                    except Exception:
                        pass
            elif suf in names:
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
        for sm in self._spot_shadow_maps:
            try:
                sm.release()
            except Exception:
                pass
        for fbo in self._spot_shadow_fbos:
            try:
                fbo.release()
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
        self._shadow_inst_vbo_fp.clear()
        self._shadow_vao_cache.clear()
        self._prog_member_cache.clear()
        if self._skinned_bone_ssbo is not None:
            try:
                self._skinned_bone_ssbo.release()
            except Exception:
                pass
            self._skinned_bone_ssbo = None
