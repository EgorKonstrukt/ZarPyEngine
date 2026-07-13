# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import numpy as np
import threading
from typing import Optional
from core.engine.engine import Engine
from core.foundation.logger import Logger
from core.renderer.mesh_data import MeshData
from core.renderer.meshes import make_cube_mesh, make_sphere_mesh, make_plane_mesh, make_quad_mesh
from core.ecs.pool import asset as _get_asset_pool

_MAX_PENDING_PER_FRAME = 6

_ERROR_MESH_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "3d_models", "ERRORText.fbx"))


def _deferred_call(cb):
    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, cb)
    except ImportError:
        threading.Thread(target=cb, daemon=True).start()


class MeshLoader:
    def __init__(self, ctx, default_prog, outline_prog=None):
        self._ctx = ctx
        self._default_prog = default_prog
        self._outline_prog = outline_prog
        self._meshes: dict[str, MeshData] = {}
        self._pending_async_loads: int = 0
        self._pending_cache_keys: set = set()
        self._async_lock: threading.Lock = threading.Lock()
        self._pending_mesh_queue: list = []
        self._render_callback = None
        self._loaded_generation: int = 0

    def register_primitives(self):
        self._meshes["cube"] = make_cube_mesh()
        self._meshes["sphere"] = make_sphere_mesh()
        self._meshes["plane"] = make_plane_mesh()
        self._meshes["quad"] = make_quad_mesh()
        for m in self._meshes.values():
            m.build_gl(self._ctx, self._default_prog)
            if self._outline_prog:
                m.build_outline_vao(self._ctx, self._outline_prog)

    def get_mesh(self, name: str) -> Optional[MeshData]:
        return self._meshes.get(name)

    def get_or_create(self, name: str, file_path: str = "", scale: float = 1.0,
                      center_pivot: bool = False, flip_uvs: bool = False) -> Optional[MeshData]:
        cache_key = f"{file_path or name}|s={scale}|cp={center_pivot}|fu={flip_uvs}"
        if cache_key in self._meshes:
            return self._meshes[cache_key]
        if not file_path:
            if name == "cube":
                m = make_cube_mesh()
            elif name == "sphere":
                m = make_sphere_mesh()
            elif name == "plane":
                m = make_plane_mesh()
            elif name == "quad":
                m = make_quad_mesh()
            else:
                if cache_key not in self._pending_cache_keys:
                    self._pending_cache_keys.add(cache_key)
                    with self._async_lock:
                        self._pending_async_loads += 1
                    self._load_async(name, "", cache_key, scale, center_pivot, flip_uvs)
                return None
        else:
            if cache_key not in self._pending_cache_keys:
                self._pending_cache_keys.add(cache_key)
                with self._async_lock:
                    self._pending_async_loads += 1
                self._load_async(name, file_path, cache_key, scale, center_pivot, flip_uvs)
            return None
        self._apply_transforms(m, cache_key, scale, center_pivot, flip_uvs)
        self._do_render_request()
        return m

    def _apply_transforms(self, m: MeshData, cache_key: str, scale: float,
                          center_pivot: bool, flip_uvs: bool):
        if scale != 1.0:
            verts = m.vertices.reshape(-1, 3)
            verts = verts * scale
            m.vertices = verts.flatten()
        if center_pivot:
            verts = m.vertices.reshape(-1, 3)
            center = verts.mean(axis=0)
            verts = verts - center
            m.vertices = verts.flatten()
        if flip_uvs and len(m.uvs) > 0:
            uvs_arr = m.uvs.reshape(-1, 2)
            uvs_arr[:, 1] = 1.0 - uvs_arr[:, 1]
            m.uvs = uvs_arr.flatten()
        m.compute_aabb()
        m.build_gl(self._ctx, self._default_prog)
        if self._outline_prog:
            m.build_outline_vao(self._ctx, self._outline_prog)
        self._meshes[cache_key] = m
        from core.spatial.bvh import prebuild_mesh_bvh
        prebuild_mesh_bvh(m.vertices, m.indices)

    def _resolve_path(self, key: str, file_path: str) -> str:
        if file_path and os.path.exists(file_path):
            return file_path
        elif file_path:
            rel = os.path.join(os.getcwd(), file_path)
            if os.path.exists(rel):
                return rel
            eng = Engine.instance()
            root = eng.project_root if eng and eng.project_root else os.getcwd()
            rel_root = os.path.join(root, file_path)
            if os.path.exists(rel_root):
                return os.path.normpath(rel_root).replace("\\", "/")
            if len(file_path) > 1 and file_path[1] == ":":
                parts = file_path.replace("\\", "/").split("/")
                for i in range(len(parts)):
                    sub = "/".join(parts[i:])
                    if sub:
                        c = os.path.normpath(os.path.join(root, sub))
                        if os.path.exists(c):
                            return c.replace("\\", "/")
            for base in ["", "assets/", "assets/models/"]:
                for ext in [".obj", ".fbx", ".stl", ".gltf", ".glb", ".usdz",
                            ".OBJ", ".FBX", ".STL", ".GLTF", ".GLB", ".USDZ"]:
                    candidate = os.path.join(base, key + ext)
                    if os.path.exists(candidate):
                        return candidate
                if file_path.endswith((".obj", ".fbx", ".stl", ".gltf", ".glb", ".usdz",
                                       ".OBJ", ".FBX", ".STL", ".GLTF", ".GLB", ".USDZ")) and os.path.exists(file_path):
                    return file_path
        else:
            for ext in [".obj", ".fbx", ".stl", ".gltf", ".glb", ".usdz",
                        ".OBJ", ".FBX", ".STL", ".GLTF", ".GLB", ".USDZ"]:
                for base in ["assets/models/", "assets/"]:
                    candidate = os.path.join(base, key + ext)
                    if os.path.exists(candidate):
                        return candidate
        return ""

    def _build_mesh_data(self, import_data):
        if import_data is None or len(import_data.vertices) == 0:
            return None
        m = MeshData()
        m.vertices = import_data.vertices.copy()
        m.normals = import_data.normals.copy()
        m.uvs = import_data.uvs.copy()
        m.indices = import_data.indices.copy()
        m.is_error_mesh = import_data.is_error_mesh
        m.sub_mesh_ranges = list(getattr(import_data, 'sub_mesh_ranges', []))
        m.sub_mesh_names = list(getattr(import_data, 'sub_mesh_names', []))
        if getattr(import_data, 'has_skeleton', False) and len(getattr(import_data, 'bone_indices', np.zeros((0, 4), dtype=np.int32))) > 0:
            m.has_skeleton = True
            m.bone_names = list(import_data.bone_names)
            m.bone_parents = list(import_data.bone_parents)
            m.bone_offset_matrices = [np.array(x, dtype=np.float32) for x in import_data.bone_offset_matrices]
            m.bone_bind_local = [np.array(x, dtype=np.float32) for x in import_data.bone_bind_local]
            m.bone_indices = np.array(import_data.bone_indices, dtype=np.int32).copy()
            m.bone_weights = np.array(import_data.bone_weights, dtype=np.float32).copy()
        return m

    def _load_async(self, key: str, file_path: str, cache_key: str,
                    scale: float, cp: bool, fuvs: bool):
        path = self._resolve_path(key, file_path)
        if not path:
            Logger.warning(f"Mesh not found: {key}, using error fallback")
            path = _ERROR_MESH_PATH
        from core.assets.asset_importer import load_obj_future, load_mesh_future
        lower_path = path.lower()

        def _io_done(fut):
            try:
                import_data = fut.result()
            except Exception:
                import_data = None
            is_error = (path == _ERROR_MESH_PATH)
            if not is_error and (import_data is None or len(import_data.vertices) == 0):
                Logger.warning(f"Failed to load mesh, falling back to error mesh")
                from core.assets.asset_importer import load_mesh
                import_data = load_mesh(_ERROR_MESH_PATH)
                is_error = True
            if is_error and import_data is not None and len(import_data.vertices) > 0:
                verts = import_data.vertices.reshape(-1, 3).astype(np.float32)
                norms = import_data.normals.reshape(-1, 3).astype(np.float32)
                rot = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
                verts = verts @ rot.T
                norms = norms @ rot.T
                s = np.array([0.25 / 9.0, 1.0 / 9.0, 1.0 / 9.0], dtype=np.float32)
                import_data.vertices = (verts * s).ravel().astype(np.float32)
                import_data.normals = norms.ravel().astype(np.float32)
                import_data.is_error_mesh = True
            with self._async_lock:
                self._pending_mesh_queue.append((cache_key, import_data, scale, cp, fuvs))
            self._on_async_load_complete()
        if lower_path.endswith(".obj"):
            fut = load_obj_future(path)
        else:
            fut = load_mesh_future(path)
        fut.add_done_callback(_io_done)

    def set_render_callback(self, callback):
        self._render_callback = callback

    def _on_async_load_complete(self):
        with self._async_lock:
            self._pending_async_loads -= 1
            if self._pending_async_loads <= 0 and self._render_callback:
                _deferred_call(self._render_callback)

    def _do_render_request(self):
        if self._render_callback:
            self._render_callback()

    def process_pending(self):
        with self._async_lock:
            pending = list(self._pending_mesh_queue)
            self._pending_mesh_queue.clear()
        processed = 0
        for cache_key, import_data, scale, cp, fuvs in pending:
            if processed >= _MAX_PENDING_PER_FRAME:
                with self._async_lock:
                    self._pending_mesh_queue.insert(0, (cache_key, import_data, scale, cp, fuvs))
                break
            if cache_key not in self._pending_cache_keys:
                continue
            processed += 1
            m = self._build_mesh_data(import_data)
            if not m:
                self._pending_cache_keys.discard(cache_key)
                continue
            self._apply_transforms(m, cache_key, scale, cp, fuvs)
            self._loaded_generation += 1

    def bump_generation(self):
        self._loaded_generation += 1

    def clear_scene_data(self):
        with self._async_lock:
            self._pending_mesh_queue.clear()
            self._pending_async_loads = 0
            self._pending_cache_keys.clear()
        for m in self._meshes.values():
            m.release()
        self._meshes.clear()
        self.register_primitives()
        self._loaded_generation += 1

    def release(self):
        for m in self._meshes.values():
            m.release()
