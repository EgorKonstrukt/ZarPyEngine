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
from core.maths.math3d import Mat4
from core.components.rendering.renderers.svg_renderer import SvgRenderer

from core.assets.texture_import_settings import TextureImportSettings


class SvgRendererGL:
    def __init__(self, ctx: moderngl.Context, prog: moderngl.Program):
        self._ctx = ctx
        self._prog = prog
        # cache uniform availability (checked once)
        self._has_view = "u_view" in prog
        self._has_proj = "u_proj" in prog
        self._has_model = "u_model" in prog
        self._has_color = "u_color" in prog
        self._has_flip = "u_flip" in prog
        self._has_alpha_cutoff = "u_alpha_cutoff" in prog
        self._has_texture = "u_texture" in prog
        # pre-allocated reusable arrays
        self._flip_arr = np.array([0.0, 0.0], dtype=np.float32)
        self._alpha_cutoff_val = 0.01
        # quad geometry
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None
        self._vao: Optional[moderngl.VertexArray] = None
        # caches: texture by (abs_path, pixels_per_unit), paths by raw_path
        self._texture_cache: dict[tuple[str, float], tuple[str, float, Any]] = {}
        self._path_cache: dict[str, Optional[str]] = {}
        self._build_buffers()

    def _build_buffers(self):
        quad = np.array([
            -0.5, -0.5, 0.0, 0.0, 0.0,
             0.5, -0.5, 0.0, 1.0, 0.0,
             0.5,  0.5, 0.0, 1.0, 1.0,
            -0.5,  0.5, 0.0, 0.0, 1.0,
        ], dtype=np.float32)
        idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self._vbo = self._ctx.buffer(quad.tobytes())
        self._ibo = self._ctx.buffer(idx.tobytes())
        self._vao = self._ctx.vertex_array(
            self._prog,
            [(self._vbo, "3f 2f", "in_position", "in_uv")],
            self._ibo
        )

    def _resolve_path(self, path: str) -> Optional[str]:
        if not path:
            return None
        cached = self._path_cache.get(path)
        if cached is not None:
            return cached
        abs_path: Optional[str] = None
        if os.path.exists(path):
            abs_path = os.path.abspath(path)
        elif not os.path.isabs(path):
            candidate = os.path.join(os.getcwd(), path)
            if os.path.exists(candidate):
                abs_path = candidate
            else:
                from core.engine.engine import Engine
                eng = Engine.instance()
                root = eng.project_root if eng and eng.project_root else os.getcwd()
                candidate = os.path.normpath(os.path.join(root, path))
                if os.path.exists(candidate):
                    abs_path = candidate
        self._path_cache[path] = abs_path
        return abs_path

    def _rasterize_svg(self, abs_path: str, pixels_per_unit: float) -> Optional[tuple[bytes, int, int]]:
        from PyQt6.QtGui import QImage, QColor, QPainter
        from PyQt6.QtSvg import QSvgRenderer
        renderer = QSvgRenderer(abs_path)
        if not renderer.isValid():
            return None
        ds = renderer.defaultSize()
        if ds.isValid() and ds.width() > 0 and ds.height() > 0:
            longest = max(ds.width(), ds.height())
            tex_size = max(int(pixels_per_unit), 16)
            w = max(1, int(ds.width() * tex_size / longest))
            h = max(1, int(ds.height() * tex_size / longest))
        else:
            w = h = max(int(pixels_per_unit), 16)
        img = QImage(w, h, QImage.Format.Format_RGBA8888)
        img.fill(0)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(p)
        p.end()
        return (bytes(img.constBits().asstring(w * h * 4)), w, h)

    def _get_texture(self, abs_path: str, pixels_per_unit: float) -> Optional[Any]:
        try:
            mtime = os.path.getmtime(abs_path)
        except OSError:
            mtime = 0.0
        import_mtime = TextureImportSettings.import_mtime(abs_path)
        key = (abs_path, pixels_per_unit)
        cached = self._texture_cache.get(key)
        if cached is not None and len(cached) >= 4 and cached[0] == mtime and cached[3] == import_mtime:
            return cached[2]
        if cached is not None and len(cached) >= 3:
            old_tex = cached[2]
            if old_tex is not None:
                try:
                    old_tex.release()
                except Exception:
                    pass
        result = self._rasterize_svg(abs_path, pixels_per_unit)
        if result is None:
            self._texture_cache[key] = (mtime, pixels_per_unit, None, import_mtime)
            return None
        data, w, h = result
        import_settings = TextureImportSettings.for_file(abs_path)
        if import_settings.max_size < max(w, h):
            scale = import_settings.max_size / max(w, h)
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            from PIL import Image
            pil_img = Image.frombuffer("RGBA", (w, h), data)
            data = pil_img.resize((nw, nh), Image.LANCZOS).tobytes()
            w, h = nw, nh
        tex = self._ctx.texture((w, h), 4, data)
        import_settings.apply_to_texture(tex)
        self._texture_cache[key] = (mtime, pixels_per_unit, tex, import_mtime)
        return tex

    def resolve_path(self, path: str) -> Optional[str]:
        return self._resolve_path(path)

    def render_snapshot(self, svg_items: list, view_mat: Mat4, proj_mat: Mat4):
        if not self._vao or not svg_items:
            return
        prog = self._prog
        if self._has_view:
            prog["u_view"].write(view_mat.to_f32().tobytes())
        if self._has_proj:
            prog["u_proj"].write(proj_mat.to_f32().tobytes())
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._ctx.enable(moderngl.DEPTH_TEST)
        if self._has_alpha_cutoff:
            prog["u_alpha_cutoff"].value = self._alpha_cutoff_val
        for item in svg_items:
            if not item.abs_path:
                continue
            tex = self._get_texture(item.abs_path, item.pixels_per_unit)
            if tex is None:
                continue
            if self._has_model:
                prog["u_model"].write(item.world_matrix.to_f32().tobytes())
            if self._has_color:
                prog["u_color"].write(np.array(item.color, dtype=np.float32).tobytes())
            if self._has_flip:
                self._flip_arr[0] = 1.0 if item.flip_x else 0.0
                self._flip_arr[1] = 1.0 if item.flip_y else 0.0
                prog["u_flip"].write(self._flip_arr.tobytes())
            tex.use(0)
            if self._has_texture:
                prog["u_texture"].value = 0
            self._vao.render(moderngl.TRIANGLES)
        self._ctx.enable(moderngl.CULL_FACE)

    def render(self, scene, view_mat: Mat4, proj_mat: Mat4):
        if not self._vao:
            return
        prog = self._prog
        # write view/proj once
        if self._has_view:
            prog["u_view"].write(view_mat.to_f32().tobytes())
        if self._has_proj:
            prog["u_proj"].write(proj_mat.to_f32().tobytes())
        # GL state
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._ctx.enable(moderngl.DEPTH_TEST)
        # alpha cutoff once
        if self._has_alpha_cutoff:
            prog["u_alpha_cutoff"].value = self._alpha_cutoff_val
        # entity loop вЂ” batch writes
        entities = scene.get_entities_with_component(SvgRenderer)
        for ent in entities:
            if not ent.active:
                continue
            sr = ent.get_component(SvgRenderer)
            if not sr or not sr.enabled:
                continue
            tr = ent.transform
            if not tr:
                continue
            abs_path = self._resolve_path(sr.svg_path)
            if not abs_path:
                continue
            tex = self._get_texture(abs_path, sr.pixels_per_unit)
            if tex is None:
                continue
            if self._has_model:
                prog["u_model"].write(tr.world_matrix.to_f32().tobytes())
            if self._has_color:
                c = sr.color
                prog["u_color"].write(np.array(c, dtype=np.float32).tobytes())
            if self._has_flip:
                self._flip_arr[0] = 1.0 if sr.flip_x else 0.0
                self._flip_arr[1] = 1.0 if sr.flip_y else 0.0
                prog["u_flip"].write(self._flip_arr.tobytes())
            tex.use(0)
            if self._has_texture:
                prog["u_texture"].value = 0
            self._vao.render(moderngl.TRIANGLES)
        self._ctx.enable(moderngl.CULL_FACE)

    def release(self):
        for entry in self._texture_cache.values():
            tex = entry[2]
            if tex is not None:
                try:
                    tex.release()
                except Exception:
                    pass
        self._texture_cache.clear()
        self._path_cache.clear()
        if self._vao:
            self._vao.release()
        if self._vbo:
            self._vbo.release()
        if self._ibo:
            self._ibo.release()
