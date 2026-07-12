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
from core.math.math3d import Mat4


class IconRenderer:
    """Renders screen-space icon overlays (gizmo icons)."""

    def __init__(self, ctx: moderngl.Context, prog: moderngl.Program):
        self._ctx = ctx
        self._prog = prog
        self._vbo: Optional[moderngl.Buffer] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._batch_vbo: Optional[moderngl.Buffer] = None
        self._batch_vao: Optional[moderngl.VertexArray] = None
        self._batch_cap: int = 0
        self._textures: dict[str, Any] = {}
        self._build_buffers()

    def _build_buffers(self):
        quad = np.array([
            -0.5, -0.5, 0.0,  0.0, 0.0, 1.0,
             0.5, -0.5, 0.0,  1.0, 0.0, 1.0,
             0.5,  0.5, 0.0,  1.0, 1.0, 1.0,
            -0.5,  0.5, 0.0,  0.0, 1.0, 1.0,
        ], dtype=np.float32)
        idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self._vbo = self._ctx.buffer(quad.tobytes())
        ibo = self._ctx.buffer(idx.tobytes())
        self._vao = self._ctx.vertex_array(
            self._prog,
            [(self._vbo, "3f 2f 1f", "in_position", "in_uv", "in_alpha")],
            ibo
        )
        self._batch_cap = 6 * 1024
        self._batch_vbo = self._ctx.buffer(
            reserve=self._batch_cap * 6 * 4, dynamic=True)
        self._batch_vao = self._ctx.vertex_array(
            self._prog,
            [(self._batch_vbo, "3f 2f 1f", "in_position", "in_uv", "in_alpha")]
        )

    def _ensure_batch_cap(self, verts: int):
        if verts <= self._batch_cap:
            return
        new_cap = self._batch_cap
        while new_cap < verts:
            new_cap *= 2
        self._batch_vbo.release()
        self._batch_vao.release()
        self._batch_cap = new_cap
        self._batch_vbo = self._ctx.buffer(reserve=new_cap * 6 * 4, dynamic=True)
        self._batch_vao = self._ctx.vertex_array(
            self._prog,
            [(self._batch_vbo, "3f 2f 1f", "in_position", "in_uv", "in_alpha")]
        )

    def create_texture_from_data(self, rgba_data: bytes, w: int, h: int, key: str) -> Any:
        if key in self._textures:
            return self._textures[key]
        tex = self._ctx.texture((w, h), 4, rgba_data)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        self._textures[key] = tex
        return tex

    def create_texture_from_png(self, path: str) -> Optional[Any]:
        if path in self._textures:
            return self._textures[path]
        abs_path = path
        if not os.path.isabs(path):
            abs_path = os.path.join(os.getcwd(), path)
            if not os.path.exists(abs_path):
                alt = os.path.join("assets", path)
                if os.path.exists(alt):
                    abs_path = alt
        if not os.path.exists(abs_path):
            return None
        try:
            from PIL import Image
            img = Image.open(abs_path).convert("RGBA")
            tex = self._ctx.texture(img.size, 4, img.tobytes())
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False
            self._textures[path] = tex
            return tex
        except Exception:
            return None

    def render(self, texture: Any, sx: float, sy: float, size: float, alpha: float,
               viewport_w: int, viewport_h: int):
        if not self._prog or not self._vao or not texture:
            return
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.disable(moderngl.CULL_FACE)
        vp_w = int(viewport_w)
        vp_h = int(viewport_h)
        self._ctx.viewport = (0, 0, vp_w, vp_h)
        ortho = Mat4.orthographic(0.0, vp_w, vp_h, 0.0, -1.0, 1.0)
        model = Mat4.identity()
        model._d[0, 0] = size
        model._d[1, 1] = size
        model._d[3, 0] = sx
        model._d[3, 1] = sy
        mvp = model * ortho
        self._prog["u_mvp"].write(mvp.to_f32().tobytes())
        texture.use(0)
        self._prog["u_texture"].value = 0
        self._prog["u_color"].write(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32).tobytes())
        self._prog["u_alpha"].value = alpha
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._vao.render(moderngl.TRIANGLES, vertices=6)
        self._ctx.disable(moderngl.BLEND)

    def render_batched(self, batches: list, viewport_w: int, viewport_h: int):
        if not self._prog or not self._batch_vao or not batches:
            return
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.disable(moderngl.CULL_FACE)
        vp_w = int(viewport_w)
        vp_h = int(viewport_h)
        self._ctx.viewport = (0, 0, vp_w, vp_h)
        ortho = Mat4.orthographic(0.0, vp_w, vp_h, 0.0, -1.0, 1.0)
        self._prog["u_mvp"].write(ortho.to_f32().tobytes())
        self._prog["u_color"].write(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32).tobytes())
        self._prog["u_alpha"].value = 1.0
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        for texture, quads in batches:
            n = len(quads)
            if n == 0 or not texture:
                continue
            verts = n * 6
            self._ensure_batch_cap(verts)
            data = np.empty((verts, 6), dtype=np.float32)
            for i, (sx, sy, size, alpha) in enumerate(quads):
                h = size * 0.5
                o = i * 6
                data[o + 0] = (sx - h, sy - h, 0.0, 0.0, 0.0, alpha)
                data[o + 1] = (sx + h, sy - h, 0.0, 1.0, 0.0, alpha)
                data[o + 2] = (sx + h, sy + h, 0.0, 1.0, 1.0, alpha)
                data[o + 3] = (sx - h, sy - h, 0.0, 0.0, 0.0, alpha)
                data[o + 4] = (sx + h, sy + h, 0.0, 1.0, 1.0, alpha)
                data[o + 5] = (sx - h, sy + h, 0.0, 0.0, 1.0, alpha)
            self._batch_vbo.write(data.tobytes())
            texture.use(0)
            self._prog["u_texture"].value = 0
            self._batch_vao.render(moderngl.TRIANGLES, vertices=verts)
        self._ctx.disable(moderngl.BLEND)

    def release(self):
        for tex in self._textures.values():
            try:
                tex.release()
            except Exception:
                pass
