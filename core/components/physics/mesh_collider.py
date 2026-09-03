# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from enum import Enum
from collections import OrderedDict
import json
import math
import numpy as np
import os
from typing import Optional
from core.maths.math3d import Vec3
from core.ecs.ecs import Component, ComponentRegistry, GizmoPrimitive
from core.components.inspector_meta import FieldType, InspectorField
from core.engine.engine import Engine


class CollisionMode(Enum):
    AUTO = "auto"
    MESH = "mesh"
    CONVEX_HULL = "convex_hull"
    BOX = "box"
    SPHERE = "sphere"


_SENTINEL = object()

class _LRU:
    def __init__(self, maxsize: int = 512):
        self._data: OrderedDict = OrderedDict()
        self._maxsize = maxsize
    def get(self, key):
        if key not in self._data:
            return _SENTINEL
        self._data.move_to_end(key)
        return self._data[key]
    def set(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

_mesh_data = _LRU(256)
_decimated_hull_cache_np = _LRU(128)
_convex_hull_cache_np = _LRU(128)
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))


def _resolve_mesh_path(path: str) -> Optional[str]:
    if not path:
        return None
    if os.path.exists(path):
        return path
    if os.path.isabs(path):
        eng = Engine.instance()
        root = eng.project_root if eng and eng.project_root else _PROJECT_ROOT
        if path[1:2] == ":":
            parts = path.replace("\\", "/").split("/")
            for i in range(len(parts)):
                sub = "/".join(parts[i:])
                if sub:
                    c = os.path.normpath(os.path.join(root, sub))
                    if os.path.exists(c):
                        return c.replace("\\", "/")
        return path if os.path.exists(path) else None
    resolved = os.path.normpath(os.path.join(_PROJECT_ROOT, path))
    if os.path.exists(resolved):
        return resolved
    return None


def _load_mesh_data(path: str) -> Optional[dict]:
    resolved = _resolve_mesh_path(path)
    if not resolved:
        return None
    try:
        from core.assets.asset_importer import _import_meta_signature
        sig = _import_meta_signature(resolved)
    except Exception:
        sig = None
    cache_key = resolved.lower() + "|" + (sig[1] if sig else "-")
    cached = _mesh_data.get(cache_key)
    if cached is not _SENTINEL:
        return cached
    try:
        from core.assets.asset_importer import load_mesh, load_obj
        lower_path = resolved.lower()
        if lower_path.endswith(".obj"):
            data = load_obj(resolved)
        else:
            data = load_mesh(resolved)
    except Exception:
        return None
    scale = 1.0
    center_pivot = False
    from core.assets.asset_importer import _resolve_mesh_import_path
    import_path = _resolve_mesh_import_path(resolved)
    if os.path.exists(import_path):
        try:
            with open(import_path) as f:
                settings = json.load(f)
            scale = settings.get("scale", 1.0)
            center_pivot = bool(settings.get("center_pivot", False))
        except Exception:
            pass
    if data is None or len(data.vertices) == 0:
        result = {"verts": np.empty((0, 3), dtype=np.float32), "edges": [], "num_verts": 0, "indices": None, "import_scale": scale,
                  "mins": np.zeros(3, dtype=np.float32), "maxs": np.zeros(3, dtype=np.float32),
                  "center": np.zeros(3, dtype=np.float32), "radius": 0.0}
        _mesh_data.set(cache_key, result)
        return result
    verts = data.vertices.reshape(-1, 3).astype(np.float32)
    if scale != 1.0:
        verts = verts * scale
    if center_pivot and len(verts) > 0:
        try:
            mean = verts.mean(axis=0)
            if np.all(np.isfinite(mean)):
                verts = verts - mean
        except Exception:
            pass
    try:
        raw_idx = np.asarray(data.indices)
        indices = np.ascontiguousarray(raw_idx.reshape(-1).astype(np.int64)) if raw_idx.size >= 3 else None
    except Exception:
        indices = None
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = verts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(verts - center, axis=1)))
    if len(data.indices) == 0:
        result = {"verts": verts, "edges": [], "edge_verts_np": np.empty((0, 2, 3), dtype=np.float32),
                  "mins": mins, "maxs": maxs, "num_verts": len(verts), "center": center, "radius": radius,
                  "indices": indices, "import_scale": scale}
        _mesh_data.set(cache_key, result)
        return result
    idxs = data.indices
    edges_set = set()
    for i in range(0, len(idxs), 3):
        if i + 2 >= len(idxs):
            break
        a, b, c = int(idxs[i]), int(idxs[i+1]), int(idxs[i+2])
        for ia, ib in ((a, b), (b, c), (c, a)):
            edges_set.add((ia, ib) if ia < ib else (ib, ia))
    edge_indices = np.array(list(edges_set), dtype=np.int32)
    edge_verts_np = verts[edge_indices]
    result = {"verts": verts, "edge_verts_np": edge_verts_np,
              "mins": mins, "maxs": maxs, "num_verts": len(verts), "center": center, "radius": radius,
              "indices": indices, "import_scale": scale}
    _mesh_data.set(cache_key, result)
    return result


def _decimate_verts(verts: np.ndarray, max_vertices: int) -> np.ndarray:
    n = len(verts)
    if n <= max_vertices or max_vertices < 1:
        return verts
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    extent = np.where(maxs - mins < 1e-8, 1.0, maxs - mins)
    cell_vol = extent.prod() / max_vertices
    cell_size = cell_vol ** (1.0 / 3.0)
    grid_res = np.maximum(1, np.ceil(extent / cell_size).astype(np.int32))
    indices = np.floor((verts - mins) / extent * grid_res).astype(np.int32)
    indices = np.clip(indices, 0, grid_res - 1)
    cell_ids = indices[:, 0] * grid_res[1] * grid_res[2] + indices[:, 1] * grid_res[2] + indices[:, 2]
    unique_ids, inverse = np.unique(cell_ids, return_inverse=True)
    centroids = np.zeros((len(unique_ids), 3), dtype=np.float32)
    np.add.at(centroids, inverse, verts)
    counts = np.bincount(inverse, minlength=len(unique_ids)).astype(np.float32)
    centroids /= counts[:, None]
    return centroids


def _compute_hull_edges_np(verts: np.ndarray) -> Optional[np.ndarray]:
    try:
        from core.maths.convex_hull import convex_hull_simplices
        simplices = convex_hull_simplices(verts)
        if len(simplices) == 0:
            return None
        edges_set = set()
        for simplex in simplices:
            a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
            for ia, ib in ((a, b), (b, c), (c, a)):
                edges_set.add((ia, ib) if ia < ib else (ib, ia))
        if not edges_set:
            return None
        edge_indices = np.array(list(edges_set), dtype=np.int32)
        return verts[edge_indices]
    except Exception:
        return None


def _import_sig_of(path: str) -> str:
    try:
        from core.assets.asset_importer import _import_meta_signature, _resolve_mesh_import_path
        resolved = _resolve_mesh_path(path)
        if resolved:
            sig = _import_meta_signature(resolved)
            if sig:
                return sig[1]
            ip = _resolve_mesh_import_path(resolved)
            if not os.path.exists(ip):
                return "-"
    except Exception:
        pass
    return "-"


def load_collision_geometry(path: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    try:
        md = _load_mesh_data(path)
    except Exception:
        return None, None, 1.0
    if md is None or md.get("num_verts", 0) == 0:
        return None, None, 1.0
    try:
        k = float(md.get("import_scale", 1.0) or 1.0)
    except Exception:
        k = 1.0
    if abs(k) < 1e-9:
        k = 1.0
    try:
        verts = np.ascontiguousarray(md["verts"].reshape(-1, 3), dtype=np.float32)
    except Exception:
        return None, None, k
    indices = md.get("indices")
    try:
        if indices is not None:
            raw = np.asarray(indices)
            indices = np.ascontiguousarray(raw.reshape(-1).astype(np.int64)) if raw.size >= 3 else None
        else:
            indices = None
    except Exception:
        indices = None
    return verts, indices, k


def _get_convex_hull_edges_np(path: str) -> Optional[np.ndarray]:
    cache_key = (path.lower().replace("\\", "/"), _import_sig_of(path))
    cached = _convex_hull_cache_np.get(cache_key)
    if cached is not _SENTINEL:
        return cached
    md = _load_mesh_data(path)
    if md is None or md["num_verts"] == 0:
        return None
    result = _compute_hull_edges_np(md["verts"])
    _convex_hull_cache_np.set(cache_key, result)
    return result


def _get_decimated_hull_edges_np(path: str, max_verts: int) -> Optional[np.ndarray]:
    cache_key = (path.lower().replace("\\", "/"), max_verts, _import_sig_of(path))
    cached = _decimated_hull_cache_np.get(cache_key)
    if cached is not _SENTINEL:
        return cached
    try:
        md = _load_mesh_data(path)
        if md is None or md["num_verts"] == 0:
            return None
        verts = md["verts"]
        if len(verts) <= max_verts:
            return None
        dv = _decimate_verts(verts, max_verts)
        if len(dv) < 3:
            _decimated_hull_cache_np.set(cache_key, None)
            return None
        result = _compute_hull_edges_np(dv)
        _decimated_hull_cache_np.set(cache_key, result)
        return result
    except Exception:
        _decimated_hull_cache_np.set(cache_key, None)
        return None


def _triangle_edges_np(verts: np.ndarray, indices: np.ndarray, max_tris: int = 15000) -> Optional[np.ndarray]:
    try:
        tris = np.asarray(indices).reshape(-1, 3)
        if len(tris) == 0:
            return None
        n = len(verts)
        tris = tris[((tris >= 0) & (tris < n)).all(axis=1)]
        if len(tris) == 0:
            return None
        if len(tris) > max_tris:
            tris = tris[::int(np.ceil(len(tris) / max_tris))]
        pairs = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
        return np.ascontiguousarray(verts[pairs].astype(np.float32))
    except Exception:
        return None


def _box_edges_np(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    x0, y0, z0 = float(mins[0]), float(mins[1]), float(mins[2])
    x1, y1, z1 = float(maxs[0]), float(maxs[1]), float(maxs[2])
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1],
    ], dtype=np.float32)
    pairs = np.array([
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int64)
    return np.ascontiguousarray(corners[pairs])


def _sphere_edges_np(center: np.ndarray, radius: float, segments: int = 24) -> np.ndarray:
    r = max(float(radius), 1e-6)
    t = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False, dtype=np.float32)
    c = np.cos(t) * r
    s = np.sin(t) * r
    z = np.zeros_like(t)
    rings = [
        np.stack([center[0] + c, center[1] + z, center[2] + s], axis=1),
        np.stack([center[0] + c, center[1] + s, center[2] + z], axis=1),
        np.stack([center[0] + z, center[1] + s, center[2] + c], axis=1),
    ]
    out = []
    for ring in rings:
        out.append(np.stack([ring, np.roll(ring, -1, axis=0)], axis=1))
    return np.ascontiguousarray(np.concatenate(out, axis=0).astype(np.float32))


def _edge_pairs_np(edge_verts_np: np.ndarray, color: list, pos, rot, sc):
    q = rot
    x, y, z, w = q.x, q.y, q.z, q.w
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n > 1e-10:
        inv = 1.0/n; x *= inv; y *= inv; z *= inv; w *= inv
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                   [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                   [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]], dtype=np.float32)
    T = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
    S = np.array([sc.x, sc.y, sc.z], dtype=np.float32)
    n_edges = edge_verts_np.shape[0]
    verts = edge_verts_np.reshape(-1, 3)
    transformed = verts * S @ R.T + T
    starts = transformed[0::2].reshape(n_edges, 3)
    ends = transformed[1::2].reshape(n_edges, 3)
    c_arr = np.empty((n_edges, 4), dtype=np.float32)
    c_arr[:] = color
    return starts, ends, c_arr


@ComponentRegistry.register
class MeshCollider(Component):
    _icon = "MeshCollider.png"
    _allow_multiple = True
    _gizmo_icon_color = (200, 80, 80)
    _gizmo_icon_label = "C"
    _show_gizmo_icon: bool = False
    _gizmo_pass = "collider"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("mesh_path", "Mesh", FieldType.RESOURCE_PATH, file_filter="Collision Meshes (*.obj *.stl *.gltf *.glb)"),
            InspectorField("collision_mode", "Collision Mode", FieldType.ENUM, enum_class=CollisionMode),
            InspectorField("max_vertices", "Max Vertices", FieldType.INT, min_val=0, max_val=100000, step=100, decimals=0),
            InspectorField("is_trigger", "Is Trigger", FieldType.BOOL),
            InspectorField("layer", "Layer", FieldType.LAYER),
            InspectorField("mask", "Collision Mask", FieldType.LAYER_MASK),
        ]

    def __init__(self):
        super().__init__()
        self.layer: int = 0
        self.mask: int = 0xFFFF
        self.center: Vec3 = Vec3.zero()
        self.mesh_path: str = ""
        self.collision_mode: CollisionMode = CollisionMode.AUTO
        self.max_vertices: int = 2000
        self.is_trigger: bool = False
        self.material_friction: float = 0.6
        self.material_bounciness: float = 0.0

    @property
    def scaled_center(self) -> Vec3:
        tr = self.transform
        s = tr.local_scale if tr else Vec3.one()
        c = self.center if isinstance(self.center, Vec3) else Vec3(*self.center)
        return Vec3(c.x * s.x, c.y * s.y, c.z * s.z)

    def _is_dynamic_body(self) -> bool:
        try:
            if self.is_trigger:
                return False
            ent = self.entity
            if ent is None:
                return False
            from core.components.physics.rigidbody import Rigidbody
            rb = ent.get_component(Rigidbody)
            if rb is None or not getattr(rb, "enabled", True):
                return False
            if getattr(rb, "is_kinematic", False):
                return True
            return float(getattr(rb, "mass", 0.0) or 0.0) > 0.0
        except Exception:
            return False

    def _effective_mode(self) -> CollisionMode:
        if self.collision_mode == CollisionMode.AUTO:
            return CollisionMode.MESH if not self._is_dynamic_body() else CollisionMode.CONVEX_HULL
        if self.collision_mode == CollisionMode.MESH and self._is_dynamic_body():
            return CollisionMode.CONVEX_HULL
        return self.collision_mode

    def _gizmo_sig(self):
        tr = self.transform
        if tr is None:
            return None
        c = self.center
        if isinstance(c, Vec3):
            cx, cy, cz = c.x, c.y, c.z
        else:
            cx, cy, cz = c[0], c[1], c[2]
        return (
            self.mesh_path, self.collision_mode.value, self.max_vertices,
            cx, cy, cz, self._is_dynamic_body(),
            tr.local_position.x, tr.local_position.y, tr.local_position.z,
            tr.local_rotation.x, tr.local_rotation.y, tr.local_rotation.z, tr.local_rotation.w,
            tr.local_scale.x, tr.local_scale.y, tr.local_scale.z,
        )

    def gizmo_primitives(self):
        if not self.mesh_path:
            return None
        tr = self.transform
        if not tr:
            return None
        sig = self._gizmo_sig()
        cached = getattr(self, "_mc_gizmo_cache", None)
        if cached is not None and cached[0] == sig:
            return cached[1]
        path = self.mesh_path
        md = _load_mesh_data(path)
        if md is None or md["num_verts"] == 0:
            return None
        c = self.center
        if isinstance(c, Vec3):
            co = np.array([c.x, c.y, c.z], dtype=np.float32)
        else:
            co = np.array([c[0], c[1], c[2]], dtype=np.float32)
        mode = self._effective_mode()
        edges_np = None
        if mode == CollisionMode.MESH:
            idx = md.get("indices")
            if idx is not None and len(idx) >= 3:
                edges_np = _triangle_edges_np(md["verts"], idx)
            if edges_np is None:
                edges_np = _get_convex_hull_edges_np(path)
        elif mode == CollisionMode.BOX:
            edges_np = _box_edges_np(md["mins"], md["maxs"])
        elif mode == CollisionMode.SPHERE:
            edges_np = _sphere_edges_np(md["center"], md["radius"])
        else:
            if self.max_vertices > 0:
                edges_np = _get_decimated_hull_edges_np(path, self.max_vertices)
            if edges_np is None:
                edges_np = _get_convex_hull_edges_np(path)
        if edges_np is None or len(edges_np) == 0:
            return None
        edges_np = np.ascontiguousarray(edges_np.astype(np.float32) + co)
        color = [0.0, 1.0, 0.0, 0.6]
        result = _edge_pairs_np(edges_np, color, tr.local_position, tr.local_rotation, tr.local_scale)
        self._mc_gizmo_cache = (sig, result)
        return result

    def gizmo_lines(self) -> list[tuple[Vec3, Vec3, list[float]]]:
        prim = self.gizmo_primitives()
        if prim is None:
            return []
        s, e, c = prim
        n = s.shape[0]
        color = [float(c[0, 0]), float(c[0, 1]), float(c[0, 2]), float(c[0, 3])]
        result = []
        for i in range(n):
            result.append((
                Vec3(float(s[i, 0]), float(s[i, 1]), float(s[i, 2])),
                Vec3(float(e[i, 0]), float(e[i, 1]), float(e[i, 2])),
                color,
            ))
        return result

    def gizmo(self):
        try:
            from core.engine.engine import Engine
            from core.components.physics.rigidbody import Rigidbody
            eng = Engine.instance()
            if eng and getattr(eng, 'play_mode', False) and self.entity:
                if self.entity.get_component(Rigidbody):
                    return []
            if eng:
                vp = eng.viewport
                cam = getattr(vp, '_cam', None) if vp else None
                cam_pos = cam.position if cam else None
                if cam_pos and self.entity:
                    tr = self.transform
                    if tr and (tr.position - cam_pos).length() > 20.0:
                        return []
        except Exception:
            pass
        prims = self.gizmo_primitives()
        if prims is None:
            return []
        s, e, c = prims
        if s.shape[0] == 0:
            return []
        return [GizmoPrimitive(s, e, c)]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "center": self.center.to_list(), "mesh_path": self.mesh_path,
            "collision_mode": self.collision_mode.value,
            "max_vertices": self.max_vertices,
            "is_trigger": self.is_trigger,
            "friction": self.material_friction,
            "bounciness": self.material_bounciness,
            "layer": self.layer, "mask": self.mask,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> MeshCollider:
        mc = cls()
        mc.enabled = data.get("enabled", True)
        mc.center = Vec3(*data.get("center", [0, 0, 0]))
        mc.mesh_path = data.get("mesh_path", "") or ""
        mc.collision_mode = CollisionMode(data.get("collision_mode", "auto"))
        mc.max_vertices = data.get("max_vertices", 2000)
        mc.is_trigger = data.get("is_trigger", False)
        mc.material_friction = data.get("friction", 0.6)
        mc.material_bounciness = data.get("bounciness", 0.0)
        mc.layer = data.get("layer", 0)
        mc.mask = data.get("mask", 0xFFFF)
        return mc
