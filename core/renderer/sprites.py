# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
import moderngl
from typing import Optional, Any
from core.maths.math3d import Mat4
from core.components.rendering.renderers.sprite_renderer import SpriteRenderer



class SpriteRendererGL:
    """Renders 2D sprites in 3D space."""

    def __init__(self, ctx: moderngl.Context, prog: moderngl.Program):
        self._ctx = ctx
        self._prog = prog
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._texture_loader = None
        self._build_buffers()

    def _build_buffers(self):
        sprite_quad = np.array([
            -0.5, -0.5, 0.0, 0.0, 0.0,
             0.5, -0.5, 0.0, 1.0, 0.0,
             0.5,  0.5, 0.0, 1.0, 1.0,
            -0.5,  0.5, 0.0, 0.0, 1.0,
        ], dtype=np.float32)
        sprite_idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self._vbo = self._ctx.buffer(sprite_quad.tobytes())
        self._ibo = self._ctx.buffer(sprite_idx.tobytes())
        self._vao = self._ctx.vertex_array(
            self._prog,
            [(self._vbo, "3f 2f", "in_position", "in_uv")],
            self._ibo
        )

    def set_texture_loader(self, loader):
        self._texture_loader = loader

    def render_snapshot(self, sprite_items: list, view_mat: Mat4, proj_mat: Mat4):
        if not self._prog or not self._vao or not sprite_items:
            return
        prog = self._prog
        view_f32 = view_mat.to_f32()
        proj_f32 = proj_mat.to_f32()
        if "u_view" in prog:
            prog["u_view"].write(view_f32.tobytes())
        if "u_proj" in prog:
            prog["u_proj"].write(proj_f32.tobytes())
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.depth_mask = True
        for item in sprite_items:
            tex = self._texture_loader(item.texture_path) if item.texture_path and self._texture_loader else None
            if tex is None:
                continue
            model_f32 = item.world_matrix.to_f32()
            if "u_model" in prog:
                prog["u_model"].write(model_f32.tobytes())
            if "u_color" in prog:
                prog["u_color"].write(np.array(item.color, dtype=np.float32).tobytes())
            if "u_flip" in prog:
                prog["u_flip"].write(np.array(
                    [1.0 if item.flip_x else 0.0, 1.0 if item.flip_y else 0.0],
                    dtype=np.float32
                ).tobytes())
            if "u_alpha_cutoff" in prog:
                prog["u_alpha_cutoff"].value = 0.01
            tex.use(0)
            if "u_texture" in prog:
                prog["u_texture"].value = 0
            self._vao.render(moderngl.TRIANGLES)
        self._ctx.enable(moderngl.CULL_FACE)

    def render(self, scene, view_mat: Mat4, proj_mat: Mat4):
        if not self._prog or not self._vao:
            return
        prog = self._prog
        view_f32 = view_mat.to_f32()
        proj_f32 = proj_mat.to_f32()
        if "u_view" in prog:
            prog["u_view"].write(view_f32.tobytes())
        if "u_proj" in prog:
            prog["u_proj"].write(proj_f32.tobytes())
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.depth_mask = True
        for ent in scene.get_entities_with_component(SpriteRenderer):
            if not ent.active:
                continue
            sr = ent.get_component(SpriteRenderer)
            if not sr or not sr.enabled:
                continue
            tr = ent.transform
            if not tr:
                continue
            tex = self._texture_loader(sr.texture_path) if sr.texture_path and self._texture_loader else None
            if tex is None:
                continue
            model_f32 = tr.world_matrix.to_f32()
            if "u_model" in prog:
                prog["u_model"].write(model_f32.tobytes())
            color = sr.color
            if "u_color" in prog:
                prog["u_color"].write(np.array(color, dtype=np.float32).tobytes())
            if "u_flip" in prog:
                prog["u_flip"].write(np.array(
                    [1.0 if sr.flip_x else 0.0, 1.0 if sr.flip_y else 0.0],
                    dtype=np.float32
                ).tobytes())
            if "u_alpha_cutoff" in prog:
                prog["u_alpha_cutoff"].value = 0.01
            tex.use(0)
            if "u_texture" in prog:
                prog["u_texture"].value = 0
            self._vao.render(moderngl.TRIANGLES)
        self._ctx.enable(moderngl.CULL_FACE)

    def release(self):
        if self._vao:
            self._vao.release()
        if self._vbo:
            self._vbo.release()
        if self._ibo:
            self._ibo.release()
