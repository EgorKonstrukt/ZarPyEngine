# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun
#
# Radiance Cascades Global Illumination
# Based on the paper by Alexander Sannikov (Grinding Gear Games)
# https://github.com/Raikiri/RadianceCascadesPaper

from __future__ import annotations
import os
import numpy as np
import moderngl
from typing import Optional
from core.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.logger import Logger


@ComponentRegistry.register
class RadianceCascadesGI(Component):
    _allow_multiple = False

    NUM_CASCADES = 7

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("enabled", "Enabled", FieldType.BOOL),
            InspectorField("_compute_shader_path", "Compute Shader", FieldType.RESOURCE_PATH,
                           file_filter="Compute (*.compute)"),
            InspectorField("_resolution_scale", "Resolution Scale", FieldType.FLOAT, 0.25, 1.0),
            InspectorField("_intensity", "GI Intensity", FieldType.FLOAT, 0.0, 5.0),
            InspectorField("_step_size", "Step Size", FieldType.FLOAT, 0.1, 2.0),
            InspectorField("_depth_threshold", "Depth Threshold", FieldType.FLOAT, 0.01, 1.0),
            InspectorField("_show_overlay", "Show Overlay", FieldType.BOOL),
            InspectorField("_debug_mode", "Debug Mode", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self._compute_shader_path: str = "core/shaders/RadianceCascades.compute"
        self._resolution_scale: float = 0.5
        self._intensity: float = 1.0
        self._step_size: float = 0.5
        self._depth_threshold: float = 0.15
        self._show_overlay: bool = False
        self._debug_mode: bool = False

        self._program: Optional[moderngl.ComputeShader] = None
        self._cascade_atlas: Optional[moderngl.Texture] = None
        self._gi_output_tex: Optional[moderngl.Texture] = None
        self._gi_output_fbo: Optional[moderngl.Framebuffer] = None
        self._fullscreen_quad: Optional[moderngl.VertexArray] = None
        self._fullscreen_prog: Optional[moderngl.Program] = None

        self._ctx_id = 0
        self._prev_width: int = 0
        self._prev_height: int = 0

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "compute_shader_path": self._compute_shader_path,
            "resolution_scale": self._resolution_scale,
            "intensity": self._intensity,
            "step_size": self._step_size,
            "depth_threshold": self._depth_threshold,
            "show_overlay": self._show_overlay,
            "debug_mode": self._debug_mode,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> RadianceCascadesGI:
        r = cls()
        r.enabled = data.get("enabled", True)
        r._compute_shader_path = data.get("compute_shader_path", "core/shaders/RadianceCascades.compute")
        r._resolution_scale = float(data.get("resolution_scale", 0.5))
        r._intensity = float(data.get("intensity", 1.0))
        r._step_size = float(data.get("step_size", 0.5))
        r._depth_threshold = float(data.get("depth_threshold", 0.15))
        r._show_overlay = data.get("show_overlay", False)
        r._debug_mode = data.get("debug_mode", False)
        return r

    def _compile_compute(self, ctx: moderngl.Context, path: str) -> Optional[moderngl.ComputeShader]:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            Logger.error(f"Compute shader not found: {abs_path}")
            return None
        try:
            with open(abs_path) as f:
                src = f.read()
            glsl_start = src.find("GLSLPROGRAM")
            glsl_end = src.find("ENDGLSL", glsl_start)
            if glsl_start < 0 or glsl_end < 0:
                Logger.error("Invalid .compute file: no GLSLPROGRAM/ENDGLSL")
                return None
            source = src[glsl_start + len("GLSLPROGRAM"):glsl_end].strip()
            return ctx.compute_shader(source)
        except Exception as e:
            Logger.error(f"Failed to compile compute shader: {e}")
            return None

    def _ensure_resources(self, ctx: moderngl.Context, width: int, height: int):
        rw = max(1, int(width * self._resolution_scale))
        rh = max(1, int(height * self._resolution_scale))

        if self._program is None:
            prog = self._compile_compute(ctx, self._compute_shader_path)
            if prog is None:
                return False
            self._program = prog

        if self._fullscreen_prog is None:
            self._fullscreen_prog = ctx.program(
                vertex_shader="""
                #version 460 core
                in vec2 in_position;
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    v_uv = in_uv;
                }
                """,
                fragment_shader="""
                #version 460 core
                in vec2 v_uv;
                uniform sampler2D u_tex;
                out vec4 frag_color;
                void main() {
                    frag_color = texture(u_tex, v_uv);
                }
                """,
            )

        if self._fullscreen_quad is None:
            fs_verts = np.array([
                -1, -1, 0, 0,
                 1, -1, 1, 0,
                 1,  1, 1, 1,
                -1, -1, 0, 0,
                 1,  1, 1, 1,
                -1,  1, 0, 1,
            ], dtype=np.float32)
            vbo = ctx.buffer(fs_verts.tobytes())
            self._fullscreen_quad = ctx.vertex_array(
                self._fullscreen_prog,
                [(vbo, "2f 2f", "in_position", "in_uv")],
            )

        if (self._gi_output_tex is None or self._prev_width != rw or self._prev_height != rh):
            if self._gi_output_tex:
                self._gi_output_tex.release()
            if self._gi_output_fbo:
                self._gi_output_fbo.release()
            self._gi_output_tex = ctx.texture((rw, rh), 4, dtype="f4")
            self._gi_output_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._gi_output_tex.repeat_x = False
            self._gi_output_tex.repeat_y = False
            self._gi_output_fbo = ctx.framebuffer(color_attachments=[self._gi_output_tex])
            self._prev_width = rw
            self._prev_height = rh

        if (self._cascade_atlas is None or self._cascade_atlas.width != rw or self._cascade_atlas.height != rh):
            if self._cascade_atlas:
                self._cascade_atlas.release()
            self._cascade_atlas = ctx.texture((rw, rh), 4, dtype="f4")
            self._cascade_atlas.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._cascade_atlas.repeat_x = False
            self._cascade_atlas.repeat_y = False

        return True

    def _dispatch(self, ctx: moderngl.Context, width: int, height: int,
                  view_mat, proj_mat, cam_pos, scene, renderer) -> bool:
        ctx_id = id(ctx)
        if self._ctx_id != ctx_id:
            self._release_gl()
            self._ctx_id = ctx_id

        if not self._ensure_resources(ctx, width, height):
            return False

        rw = max(1, int(width * self._resolution_scale))
        rh = max(1, int(height * self._resolution_scale))

        prog = self._program
        if prog is None:
            return False

        d = view_mat._d
        cam_right = (float(d[0, 0]), float(d[1, 0]), float(d[2, 0]))
        cam_forward = (-float(d[0, 2]), -float(d[1, 2]), -float(d[2, 2]))

        ctx.disable(moderngl.DEPTH_TEST)

        try:
            prog["u_screen_size"] = (float(rw), float(rh))
            prog["u_camera_right"] = cam_right
            prog["u_camera_forward"] = cam_forward
            prog["u_step_size"] = self._step_size
            prog["u_depth_threshold"] = self._depth_threshold
            prog["u_intensity"] = self._intensity
            prog["u_num_cascades"] = self.NUM_CASCADES

            view_f32 = view_mat.to_f32().reshape(4, 4).T
            proj_f32 = proj_mat.to_f32().reshape(4, 4).T
            vp = proj_f32 @ view_f32
            inv_vp = np.linalg.inv(vp)
            prog["u_inv_view_proj"].write(inv_vp.astype(np.float32).flatten(order='F').tobytes())
            prog["u_view_proj"].write(vp.astype(np.float32).flatten(order='F').tobytes())
        except KeyError as e:
            Logger.warning(f"RadianceCascades uniform missing: {e}")
            return False

        depth_tex = getattr(renderer, '_scene_depth_tex', None)
        color_tex = getattr(renderer, '_scene_color_tex', None)
        if depth_tex is None or color_tex is None:
            return False

        depth_tex.use(0)
        color_tex.use(1)
        try:
            prog["u_depth_tex"] = 0
            prog["u_color_tex"] = 1
        except KeyError as e:
            Logger.warning(f"RadianceCascades texture uniform missing: {e}")
            return False

        self._cascade_atlas.bind_to_image(2, read=False, write=True)

        prog["u_mode"] = 0
        prog.run(group_x=(rw + 7) // 8, group_y=(rh + 7) // 8, group_z=1)
        ctx.memory_barrier(moderngl.ALL_BARRIER_BITS)

        self._cascade_atlas.bind_to_image(2, read=True, write=False)
        self._gi_output_tex.bind_to_image(3, read=False, write=True)

        if self._debug_mode:
            prog["u_mode"] = 2
        elif self._show_overlay:
            prog["u_mode"] = 3
        else:
            prog["u_mode"] = 1
        prog.run(group_x=(rw + 7) // 8, group_y=(rh + 7) // 8, group_z=1)
        ctx.memory_barrier(moderngl.ALL_BARRIER_BITS)

        return True

    def _blit_to_fbo(self, ctx: moderngl.Context, target_fbo: moderngl.Framebuffer, width: int, height: int):
        if not self._gi_output_fbo or not self._fullscreen_prog:
            return
        old_fbo = ctx.fbo
        target_fbo.use()
        target_fbo.viewport = (0, 0, width, height)
        self._gi_output_tex.use(0)
        self._fullscreen_prog["u_tex"].value = 0
        ctx.viewport = (0, 0, width, height)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self._fullscreen_quad.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)
        if old_fbo is not None:
            old_fbo.use()

    def blit_to_screen(self, ctx: moderngl.Context, width: int, height: int):
        if not self._gi_output_fbo or not self._fullscreen_prog:
            return
        self._gi_output_tex.use(0)
        self._fullscreen_prog["u_tex"].value = 0
        ctx.viewport = (0, 0, width, height)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        if self._show_overlay or self._debug_mode:
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        else:
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self._fullscreen_quad.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)

    def on_destroy(self):
        self._release_gl()

    def on_disable(self):
        self._release_gl()

    def _release_gl(self):
        if self._cascade_atlas:
            self._cascade_atlas.release()
            self._cascade_atlas = None
        if self._gi_output_tex:
            self._gi_output_tex.release()
            self._gi_output_tex = None
        if self._gi_output_fbo:
            self._gi_output_fbo.release()
            self._gi_output_fbo = None
        if self._program:
            self._program.release()
            self._program = None
        if self._fullscreen_prog:
            self._fullscreen_prog.release()
            self._fullscreen_prog = None
        if self._fullscreen_quad:
            self._fullscreen_quad.release()
            self._fullscreen_quad = None