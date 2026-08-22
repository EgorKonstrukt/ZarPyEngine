# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os
from typing import Any, Optional

import moderngl
import numpy as np

from core.assets.material import Material, MaterialLibrary
from core.assets.texture_import_settings import TextureImportSettings
from core.engine.engine import Engine
from core.foundation.logger import Logger
from core.foundation.progress import task_complete, task_set_detail, task_start


class MaterialManager:
    """Loads, caches and applies materials and textures to shader programs."""

    _TEX_UNIFORM_MAP = {
        "albedo_texture": "u_albedo_tex",
        "normal_texture": "u_normal_tex",
        "roughness_texture": "u_roughness_tex",
        "_BaseMap": "_BaseMap",
        "_NormalMap": "_NormalMap",
        "_OcclusionMap": "_OcclusionMap",
    }

    _WHITE4 = np.array([1, 1, 1, 1], dtype=np.float32).tobytes()
    _ZERO3 = np.zeros(3, dtype=np.float32).tobytes()

    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._material_cache: dict[str, Material] = {}
        self._prog_uniform_names: dict[int, frozenset] = {}
        self._prog_tex_active_names: dict[int, dict] = {}
        self._texture_cache: dict[str, Any] = {}
        self._pending_texture_queue: list = []
        self._async_lock = None
        self._default_white = ctx.texture((1, 1), 4, b'\xff\xff\xff\xff')

    def set_async_lock(self, lock):
        self._async_lock = lock

    def load_material(self, path: str) -> Optional[Material]:
        if not path:
            return None
        cached = self._material_cache.get(path)
        if cached is not None:
            return cached
        eng = Engine.instance()
        root = eng.project_root if eng and eng.project_root else os.getcwd()
        abs_path = self._resolve_material_path(path, root)
        lib_mat = MaterialLibrary._materials.get(abs_path)
        if lib_mat is not None:
            self._material_cache[path] = lib_mat
            return lib_mat
        m = Material.load(abs_path, root)
        if m:
            self._material_cache[path] = m
        return m

    def _resolve_material_path(self, path: str, root: str) -> str:
        if os.path.isabs(path):
            return os.path.normpath(path)
        if os.path.exists(path):
            return os.path.normpath(os.path.abspath(path))
        # Stored Windows absolute path ("C:/Users/...") — probe subpaths under root.
        if len(path) > 1 and path[1] == ":":
            parts = path.replace("\\", "/").split("/")
            for i in range(len(parts)):
                sub = "/".join(parts[i:])
                if sub:
                    c = os.path.normpath(os.path.join(root, sub))
                    if os.path.exists(c):
                        return c
            return os.path.normpath(os.path.join(root, path))
        return os.path.normpath(os.path.join(root, path))

    def load_texture(self, path: str) -> Optional[Any]:
        if not path:
            return None
        abs_path = self._resolve_tex_path(path)
        if not abs_path or not os.path.exists(abs_path):
            return None
        import_mtime = TextureImportSettings.import_mtime(abs_path)
        cached = self._texture_cache.get(abs_path)
        if cached is not None:
            cached_mtime, cached_tex = cached
            if abs(import_mtime - cached_mtime) < 0.001:
                return cached_tex
            try:
                cached_tex.release()
            except Exception:
                pass
        try:
            from PIL import Image
            img = Image.open(abs_path).convert("RGBA")
            import_settings = TextureImportSettings.for_file(abs_path)
            w, h = img.size
            longest = max(w, h)
            if longest > import_settings.max_size:
                scale = import_settings.max_size / longest
                w = max(1, int(w * scale))
                h = max(1, int(h * scale))
                img = img.resize((w, h), Image.LANCZOS)
            try:
                file_size = os.path.getsize(abs_path)
            except OSError:
                file_size = 0
            task_start("tex_load:" + abs_path, f"Loading texture {os.path.basename(abs_path)}...",
                       total=float(file_size) if file_size else None, units="bytes")
            try:
                task_set_detail("tex_load:" + abs_path, f"{w}×{h}")
                tex = self._ctx.texture(img.size, 4, img.tobytes())
                import_settings.apply_to_texture(tex)
                self._texture_cache[abs_path] = (import_mtime, tex)
                return tex
            finally:
                task_complete("tex_load:" + abs_path)
        except Exception:
            return None

    def _resolve_tex_path(self, path: str) -> str:
        if os.path.exists(path):
            return os.path.abspath(path)
        if not os.path.isabs(path):
            candidate = os.path.join(os.getcwd(), path)
            if os.path.exists(candidate):
                return candidate
        eng = Engine.instance()
        root = eng.project_root if eng and eng.project_root else os.getcwd()
        if len(path) > 1 and path[1] == ":":
            parts = path.replace("\\", "/").split("/")
            for i in range(len(parts)):
                sub = "/".join(parts[i:])
                if sub:
                    c = os.path.normpath(os.path.join(root, sub))
                    if os.path.exists(c):
                        return c.replace("\\", "/")
        candidate = os.path.normpath(os.path.join(root, path))
        if os.path.exists(candidate):
            return candidate
        return path

    def load_texture_async(self, path: str, callback) -> None:
        if not path:
            callback(None)
            return
        abs_path = path
        if not os.path.isabs(path):
            abs_path = os.path.join(os.getcwd(), path)
        if not os.path.exists(abs_path):
            callback(None)
            return
        cached = self._texture_cache.get(abs_path)
        if cached is not None:
            cached_mtime, cached_tex = cached
            import_mtime = TextureImportSettings.import_mtime(abs_path)
            if abs(import_mtime - cached_mtime) < 0.001:
                callback(cached_tex)
                return
            try:
                cached_tex.release()
            except Exception:
                pass
        from core.ecs.pool import asset as _get_asset_pool
        try:
            file_size = os.path.getsize(abs_path)
        except OSError:
            file_size = 0
        task_start("tex_load:" + abs_path, f"Loading texture {os.path.basename(abs_path)}...",
                   total=float(file_size) if file_size else None, units="bytes")

        def _task():
            try:
                from PIL import Image
                img = Image.open(abs_path).convert("RGBA")
            except (ImportError, OSError, ValueError):
                task_complete("tex_load:" + abs_path)
                callback(None)
                return
            task_set_detail("tex_load:" + abs_path, f"{img.size[0]}×{img.size[1]}")
            with self._async_lock:
                self._pending_texture_queue.append((abs_path, callback, img))
        _get_asset_pool().submit(_task)

    def process_texture_pending(self) -> None:
        if not self._pending_texture_queue:
            return
        with self._async_lock:
            items = list(self._pending_texture_queue)
            self._pending_texture_queue.clear()
        for abs_path, callback, img in items:
            try:
                import_settings = TextureImportSettings.for_file(abs_path)
                w, h = img.size
                longest = max(w, h)
                if longest > import_settings.max_size:
                    scale = import_settings.max_size / longest
                    w = max(1, int(w * scale))
                    h = max(1, int(h * scale))
                    img = img.resize((w, h), Image.LANCZOS)
                tex = self._ctx.texture(img.size, 4, img.tobytes())
                import_settings.apply_to_texture(tex)
                import_mtime = TextureImportSettings.import_mtime(abs_path)
                self._texture_cache[abs_path] = (import_mtime, tex)
                callback(tex)
                task_complete("tex_load:" + abs_path)
            except Exception:
                task_complete("tex_load:" + abs_path)
                callback(None)

    # Maps URP-style/PBR property names to default shader uniform names
    _UNIFORM_ALIASES = {
        "_EmissionColor": "u_emission",
        "_EmissionIntensity": None,
        "_Metallic": "u_metallic",
        "_Smoothness": "u_smoothness",
        "_BaseColor": "u_albedo_color",
        "_DoubleSided": "u_double_sided",
        "_Transmission": None,
        "_IOR": None,
    }

    def apply_material(self, mat: Optional[Material], prog: moderngl.Program):
        pid = id(prog)
        names = self._prog_uniform_names.get(pid)
        if names is None:
            try:
                names = frozenset(prog)
            except Exception:
                names = frozenset()
            self._prog_uniform_names[pid] = names
        self._default_white.use(0)
        white4 = self._WHITE4
        zero3 = self._ZERO3
        if "u_albedo_tex" in names:
            prog["u_albedo_tex"].value = 0
        if "u_albedo_color" in names:
            prog["u_albedo_color"].write(white4)
        if "u_metallic" in names:
            prog["u_metallic"].value = 0.0
        if "u_smoothness" in names:
            prog["u_smoothness"].value = 0.5
        if "u_emission" in names:
            prog["u_emission"].write(zero3)
        if "u_normal_tex" in names:
            prog["u_normal_tex"].value = 0
        if "u_roughness_tex" in names:
            prog["u_roughness_tex"].value = 0
        if "u_use_albedo_tex" in names:
            prog["u_use_albedo_tex"].value = 0
        if "u_use_normal_tex" in names:
            prog["u_use_normal_tex"].value = 0
        if "u_use_roughness_tex" in names:
            prog["u_use_roughness_tex"].value = 0
        if "_BaseMap" in names:
            prog["_BaseMap"].value = 0
        if "_BaseColor" in names:
            prog["_BaseColor"].write(white4)
        if "_Metallic" in names:
            prog["_Metallic"].value = 0.0
        if "_Smoothness" in names:
            prog["_Smoothness"].value = 0.5
        if "_EmissionColor" in names:
            prog["_EmissionColor"].write(zero3)
        if "_EmissionIntensity" in names:
            prog["_EmissionIntensity"].value = 0.0
        if "_Transmission" in names:
            prog["_Transmission"].value = 0.0
        if "_IOR" in names:
            prog["_IOR"].value = 1.5
        if "_NormalMap" in names:
            prog["_NormalMap"].value = 0
        if "_OcclusionMap" in names:
            prog["_OcclusionMap"].value = 0
        if "_BaseMap_Active" in names:
            prog["_BaseMap_Active"].value = 0
        if "_NormalMap_Active" in names:
            prog["_NormalMap_Active"].value = 0
        if "_OcclusionMap_Active" in names:
            prog["_OcclusionMap_Active"].value = 0
        if "_HeightMap_Active" in names:
            prog["_HeightMap_Active"].value = 0
        if "_EmissionMap_Active" in names:
            prog["_EmissionMap_Active"].value = 0
        if "_DetailAlbedoMap_Active" in names:
            prog["_DetailAlbedoMap_Active"].value = 0
        if "_DetailNormalMap_Active" in names:
            prog["_DetailNormalMap_Active"].value = 0
        if mat is None:
            return
        props = mat.properties
        tex_unit = 1
        tex_uniform_map = self._TEX_UNIFORM_MAP
        active_names = self._prog_tex_active_names.setdefault(pid, {})
        for key, value in props.items():
            if isinstance(value, str):
                if not value:
                    tex_name = tex_uniform_map.get(key, key)
                    tex_active = 0
                    candidates = active_names.get(tex_name)
                    if candidates is None:
                        cand = []
                        a1 = f"{tex_name}_Active"
                        if a1 in names:
                            cand.append(a1)
                        if tex_name.startswith("u_"):
                            a2 = f"u_use_{tex_name[2:]}"
                            if a2 in names:
                                cand.append(a2)
                        active_names[tex_name] = cand
                        candidates = cand
                    for aname in candidates:
                        prog[aname].value = 0
                    continue
                tex_name = tex_uniform_map.get(key, key)
                if tex_name not in names:
                    continue
                tex = self.load_texture(value)
                if tex is not None:
                    tex.use(tex_unit)
                    prog[tex_name].value = tex_unit
                    tex_unit += 1
                    tex_active = 1
                else:
                    prog[tex_name].value = 0
                    tex_active = 0
                candidates = active_names.get(tex_name)
                if candidates is None:
                    cand = []
                    a1 = f"{tex_name}_Active"
                    if a1 in names:
                        cand.append(a1)
                    if tex_name.startswith("u_"):
                        a2 = f"u_use_{tex_name[2:]}"
                        if a2 in names:
                            cand.append(a2)
                    active_names[tex_name] = cand
                    candidates = cand
                for aname in candidates:
                    prog[aname].value = tex_active
                continue
            if key in names:
                self._set_uniform_value(prog, key, value)
            else:
                ukey = f"u_{key}"
                if ukey in names:
                    self._set_uniform_value(prog, ukey, value)
                else:
                    alias = self._UNIFORM_ALIASES.get(key)
                    if alias is not None and alias in names:
                        self._set_uniform_value(prog, alias, value)

    def _has_uniform(self, prog, name: str) -> bool:
        names = self._prog_uniform_names.get(id(prog))
        if names is None:
            try:
                names = frozenset(prog)
            except Exception:
                names = frozenset()
            self._prog_uniform_names[id(prog)] = names
        return name in names

    def _set_uniform_value(self, prog, name: str, value):
        if isinstance(value, (float, int)):
            if self._has_uniform(prog, name):
                try:
                    prog[name].value = value
                except Exception as e:
                    Logger.error(f"set_uniform {name}={value} float failed: {e}")
        elif isinstance(value, (list, tuple)):
            if self._has_uniform(prog, name):
                try:
                    arr = np.array(value, dtype=np.float32)
                    uni = prog[name]
                    expected = uni.dimension
                    if len(arr) != expected:
                        arr = arr[:expected] if len(arr) > expected else np.pad(arr, (0, expected - len(arr)), 'constant')
                    uni.write(arr.tobytes())
                except Exception as e:
                    Logger.error(f"set_uniform {name}={value} list failed: {e}")
        elif isinstance(value, bool):
            if self._has_uniform(prog, name):
                try:
                    prog[name].value = 1 if value else 0
                except Exception as e:
                    Logger.error(f"set_uniform {name}={value} bool failed: {e}")

    def clear_caches(self):
        """Release GPU textures and clear all caches for scene reload."""
        for _mtime, tex in self._texture_cache.values():
            try:
                tex.release()
            except Exception:
                pass
        self._texture_cache.clear()
        self._material_cache.clear()

    def release(self):
        for _mtime, tex in self._texture_cache.values():
            try:
                tex.release()
            except Exception:
                pass
        try:
            self._default_white.release()
        except Exception:
            pass