# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import numpy as np
import moderngl
from typing import Optional, Any
from core.engine.engine import Engine
from core.foundation.logger import Logger
from core.assets.material import Material, MaterialLibrary
from core.assets.texture_import_settings import TextureImportSettings


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

    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._material_cache: dict[str, Material] = {}
        self._texture_cache: dict[str, Any] = {}
        self._pending_texture_queue: list = []
        self._async_lock = None
        self._default_white = ctx.texture((1, 1), 4, b'\xff\xff\xff\xff')

    def set_async_lock(self, lock):
        self._async_lock = lock

    def load_material(self, path: str) -> Optional[Material]:
        if not path:
            return None
        eng = Engine.instance()
        root = eng.project_root if eng and eng.project_root else os.getcwd()
        abs_path = os.path.normpath(path if os.path.isabs(path) else os.path.join(root, path))
        lib_mat = MaterialLibrary._materials.get(abs_path)
        cached = self._material_cache.get(abs_path)
        if lib_mat is not None and lib_mat is not cached:
            self._material_cache[abs_path] = lib_mat
            return lib_mat
        if cached is not None:
            return cached
        m = Material.load(abs_path, root)
        if m:
            self._material_cache[abs_path] = m
        return m

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
            tex = self._ctx.texture(img.size, 4, img.tobytes())
            import_settings.apply_to_texture(tex)
            self._texture_cache[abs_path] = (import_mtime, tex)
            return tex
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
        from core.assets.asset_importer import _get_thread_pool

        def _task():
            try:
                from PIL import Image
                img = Image.open(abs_path).convert("RGBA")
                with self._async_lock:
                    self._pending_texture_queue.append((abs_path, callback, img))
            except Exception:
                callback(None)
        _get_thread_pool().submit(_task)

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
            except Exception:
                callback(None)

    # Maps URP-style/PBR property names to default shader uniform names
    _UNIFORM_ALIASES = {
        "_EmissionColor": "u_emission",
        "_EmissionIntensity": None,
        "_Metallic": "u_metallic",
        "_Smoothness": "u_smoothness",
        "_BaseColor": "u_albedo_color",
    }

    def apply_material(self, mat: Optional[Material], prog: moderngl.Program):
        self._default_white.use(0)
        if "u_albedo_tex" in prog:
            prog["u_albedo_tex"].value = 0
        if "u_albedo_color" in prog:
            prog["u_albedo_color"].write(np.array([1, 1, 1, 1], dtype=np.float32).tobytes())
        if "u_metallic" in prog:
            prog["u_metallic"].value = 0.0
        if "u_smoothness" in prog:
            prog["u_smoothness"].value = 0.5
        if "u_emission" in prog:
            prog["u_emission"].write(np.zeros(3, dtype=np.float32).tobytes())
        if "u_normal_tex" in prog:
            prog["u_normal_tex"].value = 0
        if "u_roughness_tex" in prog:
            prog["u_roughness_tex"].value = 0
        for _old_active in ("u_use_albedo_tex", "u_use_normal_tex", "u_use_roughness_tex"):
            if _old_active in prog:
                prog[_old_active].value = 0
        if "_BaseMap" in prog:
            prog["_BaseMap"].value = 0
        if "_BaseColor" in prog:
            prog["_BaseColor"].write(np.array([1, 1, 1, 1], dtype=np.float32).tobytes())
        if "_Metallic" in prog:
            prog["_Metallic"].value = 0.0
        if "_Smoothness" in prog:
            prog["_Smoothness"].value = 0.5
        if "_EmissionColor" in prog:
            prog["_EmissionColor"].write(np.zeros(3, dtype=np.float32).tobytes())
        if "_EmissionIntensity" in prog:
            prog["_EmissionIntensity"].value = 0.0
        if "_NormalMap" in prog:
            prog["_NormalMap"].value = 0
        if "_OcclusionMap" in prog:
            prog["_OcclusionMap"].value = 0
        for _active in ("_BaseMap_Active", "_NormalMap_Active", "_OcclusionMap_Active",
                        "_HeightMap_Active", "_EmissionMap_Active", "_DetailAlbedoMap_Active",
                        "_DetailNormalMap_Active"):
            if _active in prog:
                prog[_active].value = 0
        if mat is None:
            return
        props = mat.properties
        tex_unit = 0
        tex_uniform_map = self._TEX_UNIFORM_MAP
        for key, value in props.items():
            if isinstance(value, str):
                tex_name = tex_uniform_map.get(key, key)
                tex = None
                if value and tex_name in prog:
                    tex = self.load_texture(value)
                    if tex:
                        tex.use(tex_unit)
                        prog[tex_name].value = tex_unit
                        tex_unit += 1
                tex_active = 1 if tex else 0
                for aname in (f"{tex_name}_Active", f"u_use_{tex_name[2:]}" if tex_name.startswith("u_") else None):
                    if aname and aname in prog:
                        prog[aname].value = tex_active
                continue
            if key in prog:
                self._set_uniform_value(prog, key, value)
            elif f"u_{key}" in prog:
                self._set_uniform_value(prog, f"u_{key}", value)
            else:
                alias = self._UNIFORM_ALIASES.get(key)
                if alias and alias in prog:
                    self._set_uniform_value(prog, alias, value)

    def _set_uniform_value(self, prog, name: str, value):
        if isinstance(value, (float, int)):
            if name in prog:
                try:
                    prog[name].value = value
                except Exception as e:
                    Logger.error(f"set_uniform {name}={value} float failed: {e}")
        elif isinstance(value, (list, tuple)):
            if name in prog:
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
            if name in prog:
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
