# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import json
import threading
import time
import traceback

import numpy as np
import moderngl
from typing import Optional, Any, Callable

from core.components import LightType, LightAreaType
from core.components.lighting import Light, Projector
from core.engine import Engine
from core.logger import Logger
from core.components.transform import Transform
from core.components.rendering.mesh_filter import MeshFilter
from core.components.rendering.mesh_renderer import MeshRenderer
from core.components.rendering.sprite_renderer import SpriteRenderer
from core.components.rendering.svg_renderer import SvgRenderer
from core.components.rendering.particle_system import ParticleSystem
from core.components.rendering.particle_force_field import ParticleForceField, FORCE_FIELD_DTYPE, MAX_FORCE_FIELDS
from core.components.mesh_editor import ProBuilderMesh
from core.components.rendering.graphics_effect import GraphicsEffect
from core.components.rendering.video_renderer import VideoRenderer
from core.math3d import Mat4, Vec3

from core.renderer.types import RenderMode
from core.renderer.mesh_data import MeshData, read_shader
from core.renderer.meshes import make_cube_mesh, make_sphere_mesh, make_plane_mesh, make_quad_mesh
from core.renderer.grid import GridRenderer
from core.renderer.gizmo import GizmoRenderer, FATLINE_VERT, FATLINE_FRAG
from core.renderer.shadows import ShadowRenderer
from core.components.rendering.sky import Sky
from core.components.rendering.clouds import Cloud
from core.renderer.particles import ParticleRenderer
from core.renderer.sprites import SpriteRendererGL
from core.renderer.svgs import SvgRendererGL
from core.renderer.video import VideoRendererGL
from core.renderer.icons import IconRenderer
from core.renderer.text import TextRendererGL
from core.renderer.materials import MaterialManager
from core.renderer.shaders import ShaderManager
from core.renderer.mesh_loader import MeshLoader
from core.renderer.batcher import RenderBatcher
from core.renderer.culling import cpu_frustum_cull
from core.renderer.gpu_culling import GpuStorage, GpuCulling


class _SpriteItem:
    __slots__ = ('world_matrix', 'color', 'flip_x', 'flip_y', 'texture_path')
    def __init__(self, world_matrix, color, flip_x, flip_y, texture_path):
        self.world_matrix = world_matrix
        self.color = list(color) if color else [1, 1, 1, 1]
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.texture_path = texture_path

class _SvgItem:
    __slots__ = ('world_matrix', 'color', 'flip_x', 'flip_y', 'abs_path', 'pixels_per_unit')
    def __init__(self, world_matrix, color, flip_x, flip_y, abs_path, pixels_per_unit):
        self.world_matrix = world_matrix
        self.color = list(color) if color else [1, 1, 1, 1]
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.abs_path = abs_path
        self.pixels_per_unit = pixels_per_unit

class _VideoItem:
    __slots__ = ('world_matrix', 'color', 'flip_x', 'flip_y', 'video_path', 'entity_id', 'loop', 'volume', 'offset', 'audio_source_entity_id')
    def __init__(self, world_matrix, color, flip_x, flip_y, video_path, entity_id, loop, volume, offset=0.0, audio_source_entity_id=""):
        self.world_matrix = world_matrix
        self.color = list(color) if color else [1, 1, 1, 1]
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.video_path = video_path
        self.entity_id = entity_id
        self.loop = loop
        self.volume = volume
        self.offset = offset
        self.audio_source_entity_id = audio_source_entity_id

class _ProjectorItem:
    __slots__ = ('texture_path', 'color', 'intensity', 'range', 'spot_angle',
                 'aspect_ratio', 'near_plane', 'far_plane', 'vp_matrix',
                 'position', 'direction', 'flip_y', 'flip_x', 'cast_shadows')
    def __init__(self, texture_path, color, intensity, range, spot_angle,
                 aspect_ratio, near_plane, far_plane, vp_matrix, position, direction,
                 flip_y=True, flip_x=False, cast_shadows=True):
        self.texture_path = texture_path
        c = list(color) if color else [1, 1, 1]
        self.color = c[:3]
        self.intensity = intensity
        self.range = range
        self.spot_angle = spot_angle
        self.aspect_ratio = aspect_ratio
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.vp_matrix = vp_matrix
        self.position = np.array(position.to_array(), dtype=np.float32)
        self.direction = np.array(direction.to_array(), dtype=np.float32)
        self.flip_y = flip_y
        self.flip_x = flip_x
        self.cast_shadows = cast_shadows

class _RenderSnapshot:
    __slots__ = (
        'lights', 'dir_light', 'sky_component', 'sky_entity', 'cloud_components',
        'renderable', 'shadow_renderables', 'sprite_items', 'video_items',
        'svg_items', 'text_items', 'particle_systems', 'force_fields',
        'projectors',
    )
    def __init__(self):
        self.lights: list = []
        self.dir_light = None
        self.sky_component = None
        self.sky_entity = None
        self.cloud_components: list = []
        self.renderable: list = []
        self.shadow_renderables: list = []
        self.sprite_items: list = []
        self.video_items: list = []
        self.svg_items: list = []
        self.text_items: list = []
        self.particle_systems: list = []
        self.force_fields: list = []
        self.projectors: list = []


class Renderer:
    """Central renderer composing all rendering subsystems."""

    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._default_prog: Optional[moderngl.Program] = None
        self._grid_prog: Optional[moderngl.Program] = None
        self._gizmo_prog: Optional[moderngl.Program] = None
        self._skybox_cube: Optional[MeshData] = None
        self._wireframe_prog: Optional[moderngl.Program] = None
        self._outline_prog: Optional[moderngl.Program] = None
        self._gizmo_fatline_prog: Optional[moderngl.Program] = None
        self._gizmo_solid_prog: Optional[moderngl.Program] = None
        self._shadow_prog: Optional[moderngl.Program] = None
        self._particle_prog: Optional[moderngl.Program] = None
        self._icon_prog: Optional[moderngl.Program] = None
        self._icon_textures: dict = {}
        self._sprite_prog: Optional[moderngl.Program] = None
        self._video_prog: Optional[moderngl.Program] = None
        self._text_prog: Optional[moderngl.Program] = None
        self._overlay_prog: Optional[moderngl.Program] = None
        self._projector_prog: Optional[moderngl.Program] = None
        self._projector_vao: Optional[moderngl.VertexArray] = None
        self._quad_vbo: Optional[moderngl.Buffer] = None
        self._quad_ibo: Optional[moderngl.Buffer] = None
        self._quad_vao: Optional[moderngl.VertexArray] = None
        self._scene_fbo: Optional[moderngl.Framebuffer] = None
        self._scene_color_tex: Optional[moderngl.Texture] = None
        self._scene_depth_tex: Optional[moderngl.Texture] = None
        self._scene_fbo_size: tuple = (0, 0)
        self._initialized: bool = False
        self._render_mode: RenderMode = RenderMode.SHADED
        self._max_lights: int = 8
        self._light_uniforms = [
            {
                "type": f"u_lights[{i}].type",
                "position": f"u_lights[{i}].position",
                "direction": f"u_lights[{i}].direction",
                "color": f"u_lights[{i}].color",
                "intensity": f"u_lights[{i}].intensity",
                "range": f"u_lights[{i}].range",
                "spot_angle": f"u_lights[{i}].spot_angle",
                "spot_inner_angle": f"u_lights[{i}].spot_inner_angle",
                "right": f"u_lights[{i}].right",
                "up": f"u_lights[{i}].up",
                "area_width": f"u_lights[{i}].area_width",
                "area_height": f"u_lights[{i}].area_height",
                "area_type": f"u_lights[{i}].area_type",
                "area_samples": f"u_lights[{i}].area_samples",
                "area_double_sided": f"u_lights[{i}].area_double_sided",
            }
            for i in range(self._max_lights)
        ]
        self._ambient: list[float] = [0.26, 0.28, 0.34]
        self._selection_outline_color: list[float] = [0.8, 0.5, 0.1, 1.0]
        self._selection_outline_thickness: float = 0.03
        self._draw_calls: int = 0
        self._triangles_drawn: int = 0
        self._vertices_drawn: int = 0
        self._particle_count: int = 0
        self._culled_total: int = 0
        self._culled_visible: int = 0
        self._render_callback: Optional[Callable] = None
        self._shadow_resolution: int = 4096
        self._shadow_distance: float = 50.0
        self._line_width: float = 0.6667
        self._clear_color: list = [0.18, 0.18, 0.18]
        self._import_meta_cache: dict[str, tuple] = {}
        self._normal_cache: dict[int, np.ndarray] = {}

        self._pp_fbo_a: Optional[moderngl.Framebuffer] = None
        self._pp_fbo_b: Optional[moderngl.Framebuffer] = None
        self._pp_color_tex_a: Optional[moderngl.Texture] = None
        self._pp_color_tex_b: Optional[moderngl.Texture] = None
        self._pp_fbo_size: tuple = (0, 0)
        self._pp_copy_prog: Optional[moderngl.Program] = None
        self._pp_copy_vao: Optional[moderngl.VertexArray] = None

        self._grid: Optional[GridRenderer] = None
        self._gizmo: Optional[GizmoRenderer] = None
        self._shadows: Optional[ShadowRenderer] = None
        self._skybox_enabled: bool = True
        self._particles: Optional[ParticleRenderer] = None
        self._sprites: Optional[SpriteRendererGL] = None
        self._videos: Optional[VideoRendererGL] = None
        self._text: Optional[TextRendererGL] = None
        self._svgs: Optional[SvgRendererGL] = None
        self._icons: Optional[IconRenderer] = None
        self._materials: Optional[MaterialManager] = None
        self._shaders: Optional[ShaderManager] = None
        self._mesh_loader: Optional[MeshLoader] = None
        self._cloud_quad: Optional[MeshData] = None
        self._cloud_plane: Optional[MeshData] = None
        self._batcher: Optional[RenderBatcher] = None
        self._gpu_storage: Optional[GpuStorage] = None
        self._gpu_culling: Optional[GpuCulling] = None
        self._render_count: int = 0

    def load_config(self, config) -> None:
        self._ambient = [
            config.get("rendering.ambient_r", self._ambient[0]),
            config.get("rendering.ambient_g", self._ambient[1]),
            config.get("rendering.ambient_b", self._ambient[2]),
        ]
        self._selection_outline_color = [
            config.get("rendering.selection_outline_r", self._selection_outline_color[0]),
            config.get("rendering.selection_outline_g", self._selection_outline_color[1]),
            config.get("rendering.selection_outline_b", self._selection_outline_color[2]),
            config.get("rendering.selection_outline_a", self._selection_outline_color[3]),
        ]
        self._selection_outline_thickness = config.get("rendering.selection_outline_thickness", self._selection_outline_thickness)
        self._max_lights = config.get("rendering.max_lights", self._max_lights)
        self._shadow_resolution = config.get("rendering.shadow_resolution", self._shadow_resolution)
        self._shadow_distance = config.get("rendering.shadow_distance", self._shadow_distance)
        self._line_width = config.get("gizmo.line_width", self._line_width)

    def initialize(self):
        try:
            self._default_prog = self._ctx.program(
                vertex_shader=read_shader("default.vert"),
                fragment_shader=read_shader("default.frag")
            )
            self._grid_prog = self._ctx.program(
                vertex_shader=read_shader("grid.vert"),
                fragment_shader=read_shader("grid.frag")
            )
            self._gizmo_prog = self._ctx.program(
                vertex_shader=read_shader("gizmo.vert"),
                fragment_shader=read_shader("gizmo.frag")
            )
            self._gizmo_fatline_prog = self._ctx.program(
                vertex_shader=FATLINE_VERT,
                fragment_shader=FATLINE_FRAG
            )
            self._gizmo_solid_prog = self._ctx.program(
                vertex_shader=read_shader("gizmo_solid.vert"),
                fragment_shader=read_shader("gizmo_solid.frag")
            )
            self._wireframe_prog = self._ctx.program(
                vertex_shader=read_shader("gizmo.vert"),
                fragment_shader=read_shader("gizmo.frag")
            )
            self._outline_prog = self._ctx.program(
                vertex_shader=read_shader("outline.vert"),
                fragment_shader=read_shader("outline.frag")
            )
            self._shadow_prog = self._ctx.program(
                vertex_shader=read_shader("shadow.vert"),
                fragment_shader=read_shader("shadow.frag")
            )
            self._particle_prog = self._ctx.program(
                vertex_shader=read_shader("particle_gpu.vert"),
                fragment_shader=read_shader("particle.frag")
            )
            self._icon_prog = self._ctx.program(
                vertex_shader=read_shader("icon.vert"),
                fragment_shader=read_shader("icon.frag")
            )
            self._sprite_prog = self._ctx.program(
                vertex_shader=read_shader("sprite.vert"),
                fragment_shader=read_shader("sprite.frag")
            )
            self._video_prog = self._ctx.program(
                vertex_shader=read_shader("video.vert"),
                fragment_shader=read_shader("video.frag")
            )
            self._text_prog = self._ctx.program(
                vertex_shader=read_shader("text.vert"),
                fragment_shader=read_shader("text.frag")
            )
            self._overlay_prog = self._ctx.program(
                vertex_shader=read_shader("shadow_overlay.vert"),
                fragment_shader=read_shader("shadow_overlay.frag")
            )
            self._projector_prog = self._ctx.program(
                vertex_shader=read_shader("projector.vert"),
                fragment_shader=read_shader("projector.frag")
            )
            PP_COPY_FRAG = """
#version 460 core
uniform sampler2D u_input_tex;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    frag_color = vec4(texture(u_input_tex, v_uv).rgb, 1.0);
}
"""
            self._pp_copy_prog = self._ctx.program(
                vertex_shader=read_shader("shadow_overlay.vert"),
                fragment_shader=PP_COPY_FRAG
            )
            quad_verts = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32)
            quad_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
            self._quad_vbo = self._ctx.buffer(quad_verts.tobytes())
            self._quad_ibo = self._ctx.buffer(quad_indices.tobytes())
            self._pp_copy_vao = self._ctx.vertex_array(
                self._pp_copy_prog,
                [(self._quad_vbo, '2f', 'in_position')],
                self._quad_ibo
            )
            self._quad_vao = self._ctx.vertex_array(
                self._overlay_prog,
                [(self._quad_vbo, '2f', 'in_position')],
                self._quad_ibo
            )
            self._projector_vao = self._ctx.vertex_array(
                self._projector_prog,
                [(self._quad_vbo, '2f', 'in_position')],
                self._quad_ibo
            )
            self._shaders = ShaderManager(self._ctx)
            self._shaders.store("core/shaders/default", self._default_prog)
            self._materials = MaterialManager(self._ctx)
            self._mesh_loader = MeshLoader(self._ctx, self._default_prog, self._outline_prog)
            self._mesh_loader.register_primitives()
            self._batcher = RenderBatcher(self._ctx, self._default_prog)
            self._default_prog = self._batcher._default_prog
            self._gpu_storage = GpuStorage(self._ctx)
            self._gpu_culling = self._gpu_storage.get_or_create_culling()
            self._grid = GridRenderer(self._ctx, self._grid_prog)
            self._load_grid_config()
            self._gizmo = GizmoRenderer(self._ctx, self._gizmo_prog, self._gizmo_fatline_prog, self._gizmo_solid_prog)
            self._gizmo._line_width = self._line_width
            self._gizmo.initialize_instanced_meshes()
            self._gizmo.initialize_instanced_lines()
            self._shadows = ShadowRenderer(self._ctx, self._shadow_prog, self._shadow_resolution, self._shadow_distance)
            self._skybox_cube = make_cube_mesh()
            self._skybox_cube.build_gl(self._ctx, self._default_prog)
            self._cloud_quad = make_quad_mesh(2.0)
            self._cloud_quad.build_gl(self._ctx, self._default_prog)
            self._cloud_plane = make_plane_mesh(1.0)
            self._cloud_plane.build_gl(self._ctx, self._default_prog)
            self._particles = ParticleRenderer(self._ctx, self._particle_prog)
            self._particles.load_compute_shader(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "shaders", "particle.compute")
            )
            self._sprites = SpriteRendererGL(self._ctx, self._sprite_prog)
            self._sprites.set_texture_loader(self._materials.load_texture)
            self._videos = VideoRendererGL(self._ctx, self._video_prog)
            self._text = TextRendererGL(self._ctx, self._text_prog)
            self._svgs = SvgRendererGL(self._ctx, self._sprite_prog)
            self._icons = IconRenderer(self._ctx, self._icon_prog)
            self._initialized = True
            Logger.info("Renderer initialized.")
        except Exception as e:
            Logger.error(f"Renderer init error: {e}", e)

    def _load_grid_config(self):
        eng = Engine.instance()
        config = eng.config if eng and hasattr(eng, 'config') else None
        if not config:
            return
        if self._grid:
            self._grid.show = config.get("rendering.show_grid", self._grid.show)
            self._grid.grid_size = config.get("rendering.grid_size", self._grid.grid_size)
            self._grid.grid_2d_mode = config.get("rendering.grid_2d_mode", self._grid.grid_2d_mode)
            self._grid.grid_zoom_distance = config.get("rendering.grid_zoom_distance", self._grid.grid_zoom_distance)
        self._skybox_enabled = config.get("rendering.show_skybox", self._skybox_enabled)

    def _ensure_scene_fbo(self, w: int, h: int):
        if self._scene_fbo_size == (w, h) and self._scene_fbo:
            return
        self._release_scene_fbo()
        self._scene_color_tex = self._ctx.texture((w, h), 4, dtype='f1')
        self._scene_depth_tex = self._ctx.depth_texture((w, h))
        self._scene_fbo = self._ctx.framebuffer(self._scene_color_tex, self._scene_depth_tex)
        self._scene_fbo_size = (w, h)

    def _release_scene_fbo(self):
        for obj in [self._scene_fbo, self._scene_color_tex, self._scene_depth_tex]:
            if obj:
                try:
                    obj.release()
                except Exception:
                    pass
        self._scene_fbo = None
        self._scene_color_tex = None
        self._scene_depth_tex = None
        self._scene_fbo_size = (0, 0)

    def _ensure_pp_fbo(self, w: int, h: int):
        if self._pp_fbo_size == (w, h) and self._pp_fbo_a:
            return
        self._release_pp_fbo()
        self._pp_color_tex_a = self._ctx.texture((w, h), 4, dtype='f1')
        self._pp_color_tex_a.repeat_x = False
        self._pp_color_tex_a.repeat_y = False
        self._pp_fbo_a = self._ctx.framebuffer(self._pp_color_tex_a)
        self._pp_color_tex_b = self._ctx.texture((w, h), 4, dtype='f1')
        self._pp_color_tex_b.repeat_x = False
        self._pp_color_tex_b.repeat_y = False
        self._pp_fbo_b = self._ctx.framebuffer(self._pp_color_tex_b)
        self._pp_fbo_size = (w, h)

    def _release_pp_fbo(self):
        for obj in [self._pp_fbo_a, self._pp_color_tex_a, self._pp_fbo_b, self._pp_color_tex_b]:
            if obj:
                try:
                    obj.release()
                except Exception:
                    pass
        self._pp_fbo_a = None
        self._pp_color_tex_a = None
        self._pp_fbo_b = None
        self._pp_color_tex_b = None
        self._pp_fbo_size = (0, 0)

    def _set_overlay_uniforms(self, overlay_prog, view_f32, inv_vp_f32):
        overlay_prog["u_inv_vp"].write(inv_vp_f32.tobytes())
        overlay_prog["u_view"].write(view_f32.tobytes())
        overlay_prog["u_scene_color"] = 13
        overlay_prog["u_depth_tex"] = 14
        self._scene_color_tex.use(13)
        self._scene_depth_tex.use(14)
        self._shadows.set_uniforms(overlay_prog)
        if "u_shadow_bias" in overlay_prog:
            overlay_prog["u_shadow_bias"].value = 0.0008

    def get_or_create_mesh(self, name: str, file_path: str = "", scale: float = 1.0,
                           center_pivot: bool = False, flip_uvs: bool = False) -> Optional[MeshData]:
        if self._mesh_loader:
            return self._mesh_loader.get_or_create(name, file_path, scale, center_pivot, flip_uvs)
        return None

    def request_render(self, callback: Callable) -> None:
        self._render_callback = callback
        if self._mesh_loader:
            self._mesh_loader.set_render_callback(callback)

    def _set_scene_uniforms(self, prog, view_f32, proj_f32, cam_pos, lights, disable_shadows=False):
        if "u_view" in prog:
            prog["u_view"].write(view_f32.tobytes())
        if "u_proj" in prog:
            prog["u_proj"].write(proj_f32.tobytes())
        if "u_camera_pos" in prog:
            prog["u_camera_pos"].write(np.array(cam_pos.to_array(), dtype=np.float32).tobytes())
        n_lights = min(len(lights), self._max_lights)
        if self._render_mode == RenderMode.FLAT:
            if "u_ambient" in prog:
                prog["u_ambient"].write(np.array([1.0, 1.0, 1.0], dtype=np.float32).tobytes())
            if "u_light_count" in prog:
                prog["u_light_count"].value = 0
            n_lights = 0
        else:
            if "u_ambient" in prog:
                prog["u_ambient"].write(np.array(self._ambient, dtype=np.float32).tobytes())
            if "u_light_count" in prog:
                prog["u_light_count"].value = n_lights
        if disable_shadows:
            shadow_light_idx = -1
        else:
            shadow_light_idx = -1
            for i, (l, lt) in enumerate(lights):
                if l.light_type == LightType.DIRECTIONAL and l.cast_shadows:
                    shadow_light_idx = i
                    break
        if "u_shadow_light_index" in prog:
            prog["u_shadow_light_index"].value = shadow_light_idx if shadow_light_idx >= 0 else -1
        for i in range(n_lights):
            l, lt = lights[i]
            unames = self._light_uniforms[i]
            if l.light_type == LightType.DIRECTIONAL:
                ltype_int = 0
            elif l.light_type == LightType.POINT:
                ltype_int = 1
            elif l.light_type == LightType.SPOT:
                ltype_int = 2
            else:
                ltype_int = 3
            if unames["type"] in prog:
                prog[unames["type"]].value = ltype_int
            pos = lt.position
            fwd = lt.forward
            if unames["position"] in prog:
                prog[unames["position"]].write(np.array([pos.x, pos.y, pos.z], dtype=np.float32).tobytes())
            if unames["direction"] in prog:
                prog[unames["direction"]].write(np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32).tobytes())
            if l.procedural_sky_lighting and l.light_type == LightType.DIRECTIONAL:
                effective_color, effective_intensity = Light.compute_sun_light(-fwd)
            else:
                effective_color = l.color
                effective_intensity = l.intensity
            if unames["color"] in prog:
                prog[unames["color"]].write(np.array(effective_color, dtype=np.float32).tobytes())
            if unames["intensity"] in prog:
                prog[unames["intensity"]].value = float(effective_intensity)
            if unames["range"] in prog:
                prog[unames["range"]].value = float(l.range)
            if unames["spot_angle"] in prog:
                prog[unames["spot_angle"]].value = float(l.spot_angle)
            if unames["spot_inner_angle"] in prog:
                prog[unames["spot_inner_angle"]].value = float(l.spot_inner_angle)
            if unames["right"] in prog:
                rv = lt.right
                prog[unames["right"]].write(np.array([rv.x, rv.y, rv.z], dtype=np.float32).tobytes())
            if unames["up"] in prog:
                uv = lt.up
                prog[unames["up"]].write(np.array([uv.x, uv.y, uv.z], dtype=np.float32).tobytes())
            if unames["area_width"] in prog:
                prog[unames["area_width"]].value = float(l.area_width)
            if unames["area_height"] in prog:
                prog[unames["area_height"]].value = float(l.area_height)
            if unames["area_type"] in prog:
                prog[unames["area_type"]].value = 0 if l.area_type == LightAreaType.RECT else 1
            if unames["area_samples"] in prog:
                prog[unames["area_samples"]].value = int(l.area_samples)
            if unames["area_double_sided"] in prog:
                prog[unames["area_double_sided"]].value = 1.0 if l.area_double_sided else 0.0
        if not disable_shadows:
            self._shadows.set_uniforms(prog)

    def _collect_snapshot(self, scene, cam_near, cam_far, cam_fov, view_mat, proj_mat, cam_pos) -> _RenderSnapshot:
        snap = _RenderSnapshot()
        scene.flush_transforms()
        if not self._import_meta_cache:
            self._preload_import_meta(scene)
        for ent in scene.get_entities_with_component(Light):
            if not ent.active:
                continue
            l = ent.get_component(Light)
            t = ent.get_component(Transform)
            if l and l.enabled and t:
                snap.lights.append((l, t))
                if snap.dir_light is None and l.light_type == LightType.DIRECTIONAL:
                    snap.dir_light = (l, t)
        for ent in scene.get_entities_with_component(Sky):
            if ent.active:
                snap.sky_component = ent.get_component(Sky)
                snap.sky_entity = ent
                break
        for ent in scene.get_entities_with_component(Cloud):
            if ent.active:
                cloud = ent.get_component(Cloud)
                if cloud and cloud.enabled:
                    snap.cloud_components.append(cloud)
        self._sync_probuilder_meshes(scene)
        needs_shadow = any(l.cast_shadows for l, _ in snap.lights)
        for ent in scene.get_entities_with_component(MeshFilter):
            if not ent.active:
                continue
            mr = ent.get_component(MeshRenderer)
            tr = ent.get_component(Transform)
            if not tr or not mr or not mr.enabled:
                continue
            mf = ent.get_component(MeshFilter)
            mesh_name = mf.mesh_name
            scale, cp, fuvs = 1.0, False, False
            mesh_path = mf.mesh_path or ""
            if mesh_path:
                meta = self._import_meta_cache.get(mesh_path)
                if meta is None:
                    meta = (1.0, False, False)
                    self._import_meta_cache[mesh_path] = meta
                scale, cp, fuvs = meta
            if not mesh_name and not mesh_path:
                mesh_name = "cube"
            elif not mesh_name and mesh_path:
                mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
            mesh = self.get_or_create_mesh(mesh_name, mesh_path, scale, cp, fuvs)
            if mesh:
                wm = tr.world_matrix
                snap.renderable.append((ent, tr, mesh, mr, wm))
                if needs_shadow and mr.cast_shadows:
                    snap.shadow_renderables.append((mesh, wm))
        for ent in scene.get_entities_with_component(SpriteRenderer):
            if not ent.active:
                continue
            sr = ent.get_component(SpriteRenderer)
            if not sr or not sr.enabled:
                continue
            tr = ent.get_component(Transform)
            if not tr:
                continue
            snap.sprite_items.append(_SpriteItem(
                tr.world_matrix, sr.color, sr.flip_x, sr.flip_y, sr.texture_path))
        for ent in scene.get_entities_with_component(VideoRenderer):
            if not ent.active:
                continue
            vr = ent.get_component(VideoRenderer)
            if not vr or not vr.enabled:
                continue
            tr = ent.get_component(Transform)
            if not tr:
                continue
            snap.video_items.append(_VideoItem(
                tr.world_matrix, vr.color, vr.flip_x, vr.flip_y,
                vr.video_path, ent._id, vr.loop, vr.volume, vr.offset,
                vr.audio_source_entity_id))
        for ent in scene.get_entities_with_component(SvgRenderer):
            if not ent.active:
                continue
            sr = ent.get_component(SvgRenderer)
            if not sr or not sr.enabled:
                continue
            tr = ent.get_component(Transform)
            if not tr:
                continue
            abs_path = self._svgs.resolve_path(sr.svg_path)
            snap.svg_items.append(_SvgItem(
                tr.world_matrix, sr.color, sr.flip_x, sr.flip_y,
                abs_path or "", sr.pixels_per_unit))
        for ent in scene.get_entities_with_component(Projector):
            if not ent.active:
                continue
            pj = ent.get_component(Projector)
            if not pj or not pj.enabled:
                continue
            tr = ent.get_component(Transform)
            if not tr:
                continue
            pos = tr.position
            fwd = tr.forward
            up = tr.up
            view = Mat4.look_at(pos, pos + fwd, up)
            proj = Mat4.perspective(pj.spot_angle, pj.aspect_ratio, pj.near_plane, pj.far_plane)
            vp = (view @ proj).to_f32()
            snap.projectors.append(_ProjectorItem(
                pj.texture_path, pj.color, pj.intensity, pj.range,
                pj.spot_angle, pj.aspect_ratio, pj.near_plane, pj.far_plane,
                vp, pos, fwd, flip_y=pj.flip_y, flip_x=pj.flip_x,
                cast_shadows=pj.cast_shadows))
        for ent in scene.get_entities_with_component(ParticleSystem):
            if not ent.active:
                continue
            ps = ent.get_component(ParticleSystem)
            if not ps or not ps.enabled:
                continue
            if ps._alive_count == 0:
                continue
            snap.particle_systems.append(ps)
        for ent in scene.get_entities_with_component(ParticleForceField):
            if not ent.active:
                continue
            ff = ent.get_component(ParticleForceField)
            if ff and ff.enabled:
                snap.force_fields.append(ff)
        return snap

    def render_scene(self, scene, view_mat: Mat4, proj_mat: Mat4, cam_pos: Vec3,
                     viewport_w: int, viewport_h: int, fbo=None,
                     selected_entities: Optional[set] = None,
                     cam_near: float = 0.01, cam_far: float = 1000.0, cam_fov: float = 60.0):
        if not self._initialized:
            return
        _render_t0 = time.perf_counter()
        eng = Engine.instance()
        prof = eng._profiler if eng and hasattr(eng, '_profiler') else None
        if self._gizmo:
            self._gizmo._stat_lines = 0
            self._gizmo._stat_instances = 0
            self._gizmo._stat_mesh_verts = 0
            self._gizmo._stat_draws = 0
        if prof:
            prof.start("render_scene")
        snap = _RenderSnapshot()
        if scene:
            with eng._scene_lock:
                snap = self._collect_snapshot(scene, cam_near, cam_far, cam_fov, view_mat, proj_mat, cam_pos)
        lights = snap.lights
        dir_light = snap.dir_light
        sky_component = snap.sky_component
        sky_entity = snap.sky_entity
        cloud_components = snap.cloud_components
        renderable = snap.renderable
        if prof:
            prof.start("gl_state_setup")
        if fbo is not None:
            fbo.use()
            fbo.viewport = (0, 0, viewport_w, viewport_h)
        self._ctx.viewport = (0, 0, viewport_w, viewport_h)
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.enable(moderngl.CULL_FACE)
        self._ctx.cull_face = 'back'
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        if self._render_mode == RenderMode.SHADED:
            prog = self._default_prog
            fill_mode = moderngl.TRIANGLES
            use_polygon_mode = False
        elif self._render_mode == RenderMode.SHADED_WIREFRAME:
            prog = self._default_prog
            fill_mode = moderngl.TRIANGLES
            use_polygon_mode = True
        else:
            prog = self._default_prog
            fill_mode = moderngl.TRIANGLES
            use_polygon_mode = False
        view_f32 = view_mat.to_f32()
        proj_f32 = proj_mat.to_f32()
        if prof:
            prof.stop("gl_state_setup")
        self._ensure_scene_fbo(viewport_w, viewport_h)
        self._scene_fbo.clear(0.0, 0.0, 0.0, 1.0, 1.0)
        self._scene_fbo.use()
        aspect = viewport_w / max(1, viewport_h)
        if prof:
            prof.start("render_shadow_pass")
        shadow_groups = self._shadows.render_shadow_pass(snap.shadow_renderables, snap.lights, cam_near, cam_far, cam_fov, aspect, view_mat, {})
        if snap.projectors:
            self._shadows.render_projector_shadows(snap.projectors, snap.shadow_renderables, shadow_groups)
        if prof:
            prof.stop("render_shadow_pass")
        self._scene_fbo.use()
        self._ctx.viewport = (0, 0, viewport_w, viewport_h)
        if sky_component and sky_component.enabled and self._skybox_cube and self._skybox_enabled:
            if prof:
                prof.start("render_skybox")
            sky_component.render_sky(self._ctx, self._shaders, view_mat, proj_mat, dir_light, self._skybox_cube)
            if prof:
                prof.stop("render_skybox")
        self._ctx.viewport = (0, 0, viewport_w, viewport_h)
        if use_polygon_mode:
            self._ctx.wireframe = True
        if prof:
            prof.start("process_pending_textures")
        self._materials.process_texture_pending()
        if prof:
            prof.stop("process_pending_textures")
        if prof:
            prof.start("mesh_async_load")
        self._mesh_loader.process_pending()
        if prof:
            prof.stop("mesh_async_load")

        renderable = [(ent, tr, mesh, mr, wm, i)
                       for i, (ent, tr, mesh, mr, wm) in enumerate(renderable)]
        self._culled_total = len(renderable) if renderable else 0
        self._culled_visible = self._culled_total
        if renderable:
            try:
                n = len(renderable)
                wm_stack = np.array([entry[4]._d for entry in renderable])
                centers = wm_stack[:, 3, :3].astype(np.float32)
                cols = wm_stack[:, :3, :3]
                sx = np.linalg.norm(cols[:, :, 0], axis=1)
                sy = np.linalg.norm(cols[:, :, 1], axis=1)
                sz = np.linalg.norm(cols[:, :, 2], axis=1)
                max_scale = np.maximum(np.maximum(sx, sy), sz)
                radii = np.array([entry[2].bounding_radius for entry in renderable], dtype=np.float32) * max_scale
                vp = proj_mat._d.T @ view_mat._d.T
                if self._gpu_storage:
                    self._gpu_storage.upload_world_matrices(
                        [entry[4] for entry in renderable], radii, self._render_count)
                    self._render_count += 1
                visible = cpu_frustum_cull(centers, radii, vp)
                self._culled_visible = len(visible)
                if len(visible) < n:
                    renderable = [renderable[idx] for idx in visible]
            except Exception:
                pass

        if prof:
            prof.start("render_meshes")
        outline_queue: list[tuple[MeshData, Mat4]] = []
        if self._batcher:
            groups = self._batcher.collect_groups(
                renderable, self._materials, self._shaders)
            self._batcher.render_groups(
                groups, view_f32, proj_f32, cam_pos, lights, False,
                self._set_scene_uniforms, self._materials.apply_material,
                self._normal_cache,
                selected_entities or set(), outline_queue,
                gpu_storage=self._gpu_storage)
        else:
            for entry in renderable:
                ent, tr, mesh, mr = entry[:4]
                wm = entry[4]
                try:
                    mat = self._materials.load_material(mr.material_path)
                    shader_path = mat.shader_path if mat else ""
                    prog = self._shaders.get_or_compile(shader_path if shader_path else "") or self._default_prog
                    self._set_scene_uniforms(prog, view_f32, proj_f32, cam_pos, lights, disable_shadows=not mr.receive_shadows)
                    model = wm
                    model_f32 = model.to_f32()
                    if "u_model" in prog:
                        prog["u_model"].write(model_f32.tobytes())
                    try:
                        nm = self._normal_cache.get(ent._id)
                        if nm is None:
                            nm3x3 = model._d[:3, :3].copy()
                            nm3x3[0] /= max(1e-10, float(np.linalg.norm(nm3x3[:, 0])))
                            nm3x3[1] /= max(1e-10, float(np.linalg.norm(nm3x3[:, 1])))
                            nm3x3[2] /= max(1e-10, float(np.linalg.norm(nm3x3[:, 2])))
                            nm = nm3x3.T.astype(np.float32)
                            self._normal_cache[ent._id] = nm
                    except Exception:
                        nm = np.eye(3, dtype=np.float32).T
                    if "u_normal_matrix" in prog:
                        prog["u_normal_matrix"].write(nm.tobytes())
                    self._materials.apply_material(mat, prog)
                    mesh.render(prog)
                    if selected_entities and ent in selected_entities:
                        outline_queue.append((mesh, wm))
                except Exception:
                    prog = self._default_prog
                    self._set_scene_uniforms(prog, view_f32, proj_f32, cam_pos, lights, disable_shadows=not mr.receive_shadows)
                    model = wm
                    model_f32 = model.to_f32()
                    if "u_model" in prog:
                        prog["u_model"].write(model_f32.tobytes())
                    if "u_normal_matrix" in prog:
                        prog["u_normal_matrix"].write(np.eye(3, dtype=np.float32).tobytes())
                    self._materials.apply_material(None, prog)
                    mesh.render(prog)
                    if selected_entities and ent in selected_entities:
                        outline_queue.append((mesh, wm))
        if prof:
            prof.stop("render_meshes")
        if use_polygon_mode:
            self._ctx.wireframe = False
        if snap.projectors:
            self._scene_fbo.use()
            self._scene_fbo.viewport = (0, 0, viewport_w, viewport_h)
            self._ctx.viewport = (0, 0, viewport_w, viewport_h)
            self._ctx.disable(moderngl.DEPTH_TEST)
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            proj = self._projector_prog
            combined = view_mat @ proj_mat
            inv_vp = combined.inverted().to_f32()
            proj["u_inv_vp"].write(inv_vp.tobytes())
            proj["u_depth_tex"] = 14
            self._scene_depth_tex.use(14)
            self._shadows.set_uniforms(proj)
            count = min(len(snap.projectors), 2)
            proj["u_projector_count"].value = count
            for i in range(count):
                px = snap.projectors[i]
                suf = f"u_pj_{i}_"
                proj[f"{suf}vp"].write(px.vp_matrix.tobytes())
                proj[f"{suf}pos"].value = tuple(float(v) for v in px.position)
                proj[f"{suf}dir"].value = tuple(float(v) for v in px.direction)
                proj[f"{suf}color"].value = tuple(float(v) for v in px.color)
                proj[f"{suf}intensity"].value = float(px.intensity)
                proj[f"{suf}range"].value = float(px.range)
                proj[f"{suf}spot_angle"].value = float(px.spot_angle)
                tex_loaded = False
                if px.texture_path and self._materials:
                    tex = self._materials.load_texture(px.texture_path)
                    if tex:
                        tex_unit = 20 + i
                        tex.use(tex_unit)
                        proj[f"{suf}tex"].value = tex_unit
                        tex_loaded = True
                proj[f"{suf}has_tex"].value = 1.0 if tex_loaded else 0.0
                proj[f"{suf}flip_y"].value = 1.0 if px.flip_y else 0.0
                proj[f"{suf}flip_x"].value = 1.0 if px.flip_x else 0.0
            self._projector_vao.render()
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self._ctx.enable(moderngl.DEPTH_TEST)
        if prof:
            prof.start("render_text_world")
        if self._text and scene:
            with eng._scene_lock:
                self._text.render(scene, view_mat, proj_mat, viewport_w, viewport_h, world_space_only=True)
        if prof:
            prof.stop("render_text_world")
        if prof:
            prof.start("render_sprites")
        self._sprites.render_snapshot(snap.sprite_items, view_mat, proj_mat)
        if prof:
            prof.stop("render_sprites")
        if prof:
            prof.start("render_videos")
        self._videos.render_snapshot(snap.video_items, view_mat, proj_mat)
        if prof:
            prof.stop("render_videos")
        if self._grid and self._grid.show:
            if prof:
                prof.start("render_grid")
            if "u_scene_color" in self._grid_prog:
                self._grid_prog["u_scene_color"] = 12
                self._scene_color_tex.use(12)
            if "u_viewport_size" in self._grid_prog:
                self._grid_prog["u_viewport_size"].value = (float(viewport_w), float(viewport_h))
            self._grid.render(view_f32, proj_f32, cam_pos, self._clear_color, viewport_h, cam_fov)
            if prof:
                prof.stop("render_grid")
        if cloud_components and self._cloud_plane and self._skybox_enabled:
            if prof:
                prof.start("render_cloud_layer")
            for cloud_component in cloud_components:
                cloud_component.render_cloud_layer(self._ctx, self._shaders, view_mat, proj_mat, dir_light, cam_pos, self._cloud_plane)
            if prof:
                prof.stop("render_cloud_layer")
        if cloud_components and self._cloud_quad and self._skybox_enabled:
            if prof:
                prof.start("render_clouds")
            self._ensure_pp_fbo(viewport_w, viewport_h)
            self._ctx.copy_framebuffer(self._pp_fbo_a, self._scene_fbo)
            self._pp_fbo_a.use()
            self._pp_fbo_a.viewport = (0, 0, viewport_w, viewport_h)
            self._ctx.viewport = (0, 0, viewport_w, viewport_h)
            self._ctx.disable(moderngl.DEPTH_TEST)
            for cloud_component in cloud_components:
                cloud_component.render_clouds(self._ctx, self._shaders, view_mat, proj_mat, dir_light, cam_pos, self._cloud_quad, self._shadows, self._scene_depth_tex, (viewport_w, viewport_h))
            self._scene_fbo.use()
            self._scene_fbo.viewport = (0, 0, viewport_w, viewport_h)
            self._ctx.viewport = (0, 0, viewport_w, viewport_h)
            self._ctx.disable(moderngl.DEPTH_TEST)
            self._pp_copy_prog["u_input_tex"] = 0
            self._pp_color_tex_a.use(0)
            self._pp_copy_vao.render()
            self._ctx.enable(moderngl.DEPTH_TEST)
            if prof:
                prof.stop("render_clouds")
        if prof:
            prof.start("render_overlay")
        if fbo is not None:
            fbo.use()
            fbo.viewport = (0, 0, viewport_w, viewport_h)
        else:
            self._ctx.screen.use()
        self._ctx.viewport = (0, 0, viewport_w, viewport_h)
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._pp_copy_prog["u_input_tex"] = 0
        self._scene_color_tex.use(0)
        self._pp_copy_vao.render()
        self._ctx.enable(moderngl.DEPTH_TEST)
        if prof:
            prof.stop("render_overlay")

        if scene:
            from core.components.rendering.raytracing_renderer import RaytracingRenderer
            for ent in scene.get_entities_with_component(RaytracingRenderer):
                if not ent.active:
                    continue
                rtr = ent.get_component(RaytracingRenderer)
                if rtr and rtr.enabled:
                    if rtr._dispatch(self._ctx, viewport_w, viewport_h, view_mat, proj_mat, cam_pos, scene, self):
                        rtr.blit_to_screen(self._ctx, viewport_w, viewport_h)
                        rtr._blit_to_fbo(self._ctx, self._scene_fbo, viewport_w, viewport_h)
                    break

        if scene:
            from core.components.rendering.radiance_cascades_gi import RadianceCascadesGI
            for ent in scene.get_entities_with_component(RadianceCascadesGI):
                if not ent.active:
                    continue
                gi = ent.get_component(RadianceCascadesGI)
                if gi and gi.enabled:
                    if gi._dispatch(self._ctx, viewport_w, viewport_h, view_mat, proj_mat, cam_pos, scene, self):
                        gi._blit_to_fbo(self._ctx, self._scene_fbo, viewport_w, viewport_h)
                        gi.blit_to_screen(self._ctx, viewport_w, viewport_h)
                    break

        if prof:
            prof.start("render_stats")
        skybox_call = 1 if (self._skybox_enabled and self._skybox_cube) else 0
        if self._batcher:
            self._draw_calls = self._batcher.draw_calls + skybox_call
        else:
            self._draw_calls = len(renderable) + skybox_call
        total_tris = 0
        total_verts = 0
        for entry in renderable:
            mesh = entry[2]
            if hasattr(mesh, 'indices') and mesh.indices is not None and len(mesh.indices) > 0:
                total_tris += len(mesh.indices) // 3
            if hasattr(mesh, 'vertices') and mesh.vertices is not None and len(mesh.vertices) > 0:
                total_verts += len(mesh.vertices) // 3
        self._triangles_drawn = total_tris
        self._vertices_drawn = total_verts
        if prof:
            prof.stop("render_stats")
        if GraphicsEffect._registry and not self._effects_disabled:
            GraphicsEffect.increment_frame()
            if prof:
                prof.start("render_graphics_effects")
            self._ctx.disable(moderngl.DEPTH_TEST)
            additive_effects = []
            screen_effects = []
            for e in list(GraphicsEffect._registry):
                if not e.enabled or not e.entity or not e.entity.active:
                    continue
                if e.should_skip():
                    continue
                if getattr(e, 'render_type', 'additive') == 'screen':
                    screen_effects.append(e)
                else:
                    additive_effects.append(e)
            for effect in additive_effects:
                try:
                    effect.render(self._ctx, self._scene_color_tex, self._scene_depth_tex,
                                  view_mat, proj_mat, cam_pos, viewport_w, viewport_h)
                except Exception as e:
                    Logger.error(f"GraphicsEffect.render error: {e}")
            if screen_effects:
                self._ensure_pp_fbo(viewport_w, viewport_h)
                if fbo is not None:
                    self._ctx.copy_framebuffer(self._pp_fbo_a, fbo)
                else:
                    self._pp_fbo_a.use()
                    self._pp_fbo_a.viewport = (0, 0, viewport_w, viewport_h)
                    self._pp_copy_prog["u_input_tex"] = 0
                    self._scene_color_tex.use(0)
                    self._pp_copy_vao.render()
                src_fbo = self._pp_fbo_a
                dst_fbo = self._pp_fbo_b
                src_tex = self._pp_color_tex_a
                for effect in screen_effects:
                    dst_fbo.use()
                    dst_fbo.viewport = (0, 0, viewport_w, viewport_h)
                    try:
                        if prof:
                            prof.start(f"effect_{effect.__class__.__name__}")
                        effect.render(self._ctx, self._scene_color_tex, self._scene_depth_tex,
                                      view_mat, proj_mat, cam_pos, viewport_w, viewport_h,
                                      input_tex=src_tex, output_fbo=dst_fbo)
                        if prof:
                            time_elapsed = prof.stop(f"effect_{effect.__class__.__name__}")
                            if time_elapsed is not None and time_elapsed > 0.01:
                                Logger.warning(f"{effect.__class__.__name__} took {time_elapsed*1000:.1f}ms")
                    except Exception as e:
                        Logger.error(f"GraphicsEffect.render error: {e}")
                    src_fbo, dst_fbo = dst_fbo, src_fbo
                    src_tex = src_fbo.color_attachments[0]
                self._ctx.disable(moderngl.BLEND)
                if fbo is not None:
                    fbo.use()
                    fbo.viewport = (0, 0, viewport_w, viewport_h)
                elif self._ctx.screen is not None:
                    self._ctx.screen.use()
                self._pp_copy_prog["u_input_tex"] = 0
                src_tex.use(0)
                self._pp_copy_vao.render()
                self._ctx.enable(moderngl.BLEND)
                self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            if prof:
                prof.stop("render_graphics_effects")
        if prof:
            prof.start("render_text")
        if self._text and scene:
            with eng._scene_lock:
                self._text.render(scene, view_mat, proj_mat, viewport_w, viewport_h, world_space_only=False)
        if prof:
            prof.stop("render_text")
        if prof:
            prof.start("render_svgs")
        self._svgs.render_snapshot(snap.svg_items, view_mat, proj_mat)
        if prof:
            prof.stop("render_svgs")
        if prof:
            prof.start("render_particles")
        if self._particles and snap.particle_systems:
            ff_list = snap.force_fields
            num_ff = min(len(ff_list), MAX_FORCE_FIELDS)
            if num_ff > 0:
                ff_data_arr = np.zeros(num_ff, dtype=FORCE_FIELD_DTYPE)
                for i in range(num_ff):
                    ff_data_arr[i] = ff_list[i].to_gpu_data()
                self._particles.upload_force_fields(ff_data_arr)
            frame = eng.frame_count if eng else 0
            if frame != self._particles._last_frame:
                now = time.perf_counter()
                self._particles._render_dt = min(now - self._particles._last_particle_time, 0.05)
                self._particles._last_particle_time = now
                self._particles._last_frame = frame
            self._particles.begin_frame(view_mat, proj_mat)
            for ps in snap.particle_systems:
                params = ps.get_compute_params(eng.fixed_dt if eng else 0.02, ps._last_delta_pos)
                params['num_force_fields'] = num_ff
                n = len(ps._particles) if ps._particles is not None else 0
                self._particles._ensure_buffers(n)
                self._particles.upload_all(ps._particles)
                self._particles.dispatch(params)
                self._particles.readback_all(ps._particles)
                dead = self._particles.read_dead_list()
                ps.replenish_free_list(dead)
                self._particles.render_single(ps)
            self._particles.end_frame()
            self._particle_count = sum(ps._alive_count for ps in snap.particle_systems)
        else:
            self._particle_count = 0
        if prof:
            prof.stop("render_particles")
        if outline_queue and self._outline_prog:
            if prof:
                prof.start("render_outlines")
            old_depth_mask = self._ctx.depth_mask
            old_wireframe = self._ctx.wireframe
            self._ctx.depth_mask = False
            self._ctx.wireframe = True
            for mesh, model_mat in outline_queue:
                mvp = model_mat * view_mat * proj_mat
                if "u_mvp" in self._outline_prog:
                    self._outline_prog["u_mvp"].write(mvp.to_f32().tobytes())
                if "u_outline_color" in self._outline_prog:
                    self._outline_prog["u_outline_color"].write(np.array(self._selection_outline_color, dtype=np.float32).tobytes())
                mesh.render_outline()
            self._ctx.depth_mask = old_depth_mask
            self._ctx.wireframe = old_wireframe
            if prof:
                prof.stop("render_outlines")
        if prof:
            prof.set_value("render_scene", (time.perf_counter() - _render_t0) * 1000.0)
            prof.stop("render_scene")

    def _render_single_outline(self, mesh: MeshData, model_mat: Mat4, view_mat: Mat4, proj_mat: Mat4):
        if not self._outline_prog or not mesh:
            return
        outline_color = self._selection_outline_color
        old_wireframe = self._ctx.wireframe
        old_depth_mask = self._ctx.depth_mask
        try:
            mvp = model_mat * view_mat * proj_mat
            if "u_mvp" in self._outline_prog:
                self._outline_prog["u_mvp"].write(mvp.to_f32().tobytes())
            if "u_outline_color" in self._outline_prog:
                self._outline_prog["u_outline_color"].write(np.array(outline_color, dtype=np.float32).tobytes())
            self._ctx.depth_mask = False
            self._ctx.wireframe = True
            mesh.render_outline()
        except Exception as e:
            Logger.error("Outline render failed in _render_single_outline", e)
        finally:
            self._ctx.wireframe = old_wireframe
            self._ctx.depth_mask = old_depth_mask

    def _preload_import_meta(self, scene):
        paths = set()
        for ent in scene.get_entities_with_component(MeshFilter):
            mf = ent.get_component(MeshFilter)
            if mf and mf.mesh_path:
                paths.add(mf.mesh_path)
        for mesh_path in paths:
            if mesh_path in self._import_meta_cache:
                continue
            import_cache = mesh_path + ".import"
            if os.path.exists(import_cache):
                try:
                    with open(import_cache) as _f:
                        _s = json.load(_f)
                    self._import_meta_cache[mesh_path] = (
                        _s.get("scale", 1.0), _s.get("center_pivot", False), _s.get("flip_uvs", False)
                    )
                except Exception:
                    self._import_meta_cache[mesh_path] = (1.0, False, False)
            else:
                self._import_meta_cache[mesh_path] = (1.0, False, False)

    def _lookup_outline_mesh(self, mf) -> Optional[MeshData]:
        if not self._mesh_loader:
            return None
        meshes = self._mesh_loader._meshes
        mesh_name = mf.mesh_name or "cube"
        mesh_path = mf.mesh_path or ""
        mesh = meshes.get(mesh_name)
        if mesh:
            return mesh
        if mesh_path:
            cache_key = f"{mesh_path}|s=1.0|cp=False|fu=False"
            mesh = meshes.get(cache_key)
            if mesh:
                return mesh
        return meshes.get("cube")

    def render_entity_outline(self, entity, model_mat: Mat4, view_mat: Mat4, proj_mat: Mat4, color: list[float]):
        if not self._outline_prog:
            return
        from core.components.rendering.mesh_filter import MeshFilter
        from core.components.rendering.mesh_renderer import MeshRenderer
        mf = entity.get_component(MeshFilter)
        mr = entity.get_component(MeshRenderer)
        if not mf or not mr or not mr.enabled:
            return
        mesh = self._lookup_outline_mesh(mf)
        if not mesh:
            return
        old_wireframe = self._ctx.wireframe
        old_depth_mask = self._ctx.depth_mask
        try:
            mvp = model_mat * view_mat * proj_mat
            if "u_mvp" in self._outline_prog:
                self._outline_prog["u_mvp"].write(mvp.to_f32().tobytes())
            if "u_outline_color" in self._outline_prog:
                self._outline_prog["u_outline_color"].write(np.array(color, dtype=np.float32).tobytes())
            self._ctx.depth_mask = False
            self._ctx.wireframe = True
            mesh.render_outline()
        except Exception as e:
            Logger.error("render_entity_outline failed", e)
        finally:
            self._ctx.wireframe = old_wireframe
            self._ctx.depth_mask = old_depth_mask

    def render_gizmo_lines(self, lines, vp_mat: Mat4, cam_pos: Optional[Vec3] = None,
                           fw: int = 1920, fh: int = 1080, thickness_multiplier: float = 1.0):
        if self._gizmo:
            self._gizmo.render_lines(lines, vp_mat, fw, fh, thickness_multiplier)

    def render_gizmo_arrays(self, starts: np.ndarray, ends: np.ndarray, colors: np.ndarray,
                             vp_mat: Mat4, fw: int = 1920, fh: int = 1080, thickness_multiplier: float = 1.0,
                             dash_opts: Optional[dict] = None):
        if self._gizmo:
            desired_pixels = max(1.0, float(self._line_width) * 1.5 * thickness_multiplier)
            self._gizmo._render_lines_np(starts, ends, colors, vp_mat, fw, fh, desired_pixels, dash_opts)

    def render_instanced_gizmo(self, mesh_type: str, instance_data: np.ndarray, vp_mat: Mat4, num_instances: int):
        if not self._gizmo:
            return
        mesh_map = {
            'cone': self._gizmo._cone_mesh,
            'cylinder': self._gizmo._cylinder_mesh,
            'cube': self._gizmo._cube_mesh,
            'quad': self._gizmo._quad_mesh,
            'circle': self._gizmo._circle_mesh,
        }
        mesh = mesh_map.get(mesh_type)
        if mesh is not None:
            self._gizmo.render_instanced(mesh, instance_data, vp_mat, num_instances)

    def render_gizmo_meshes(self, meshes: list[tuple], vp_mat: Mat4):
        if self._gizmo:
            self._gizmo.render_meshes(meshes, vp_mat)

    def render_gizmo_mesh_np(self, v_data: np.ndarray, idx_arr: np.ndarray, vp_mat: Mat4):
        if self._gizmo:
            self._gizmo.render_mesh_np(v_data, idx_arr, vp_mat)

    def render_instanced_gizmo_lines(self, shape_type: str, instance_data: np.ndarray,
                                      num_instances: int, vp_mat: Mat4,
                                      fw: int = 1920, fh: int = 1080,
                                      thickness_multiplier: float = 1.0,
                                      cam_pos: Vec3 = Vec3(0, 0, 0)):
        if self._gizmo:
            self._gizmo.render_instanced_lines(shape_type, instance_data, num_instances,
                                                vp_mat, fw, fh, thickness_multiplier, cam_pos)

    def render_wireframe_box(self, center: Vec3, size: Vec3, color: list[float], vp_mat: Mat4):
        if self._gizmo:
            self._gizmo.render_wireframe_box(center, size, color, vp_mat)

    def create_icon_texture_from_data(self, rgba_data: bytes, w: int, h: int, key: str):
        if self._icons:
            return self._icons.create_texture_from_data(rgba_data, w, h, key)
        return None

    def create_icon_texture_from_png(self, path: str):
        if self._icons:
            return self._icons.create_texture_from_png(path)
        return None

    def render_icon(self, texture, sx: float, sy: float, size: float, alpha: float,
                    viewport_w: int, viewport_h: int):
        if self._icons:
            self._icons.render(texture, sx, sy, size, alpha, viewport_w, viewport_h)

    _render_icon = render_icon

    @property
    def _meshes(self):
        if self._mesh_loader:
            return self._mesh_loader._meshes
        return {}

    @property
    def show_grid(self) -> bool:
        return self._grid.show if self._grid else False

    @show_grid.setter
    def show_grid(self, v: bool):
        if self._grid:
            self._grid.show = v

    @property
    def grid_2d_mode(self) -> bool:
        return self._grid.grid_2d_mode if self._grid else False

    @grid_2d_mode.setter
    def grid_2d_mode(self, v: bool):
        if self._grid:
            self._grid.grid_2d_mode = v

    @property
    def grid_zoom_distance(self) -> float:
        return self._grid.grid_zoom_distance if self._grid else 0.0

    @grid_zoom_distance.setter
    def grid_zoom_distance(self, v: float):
        if self._grid:
            self._grid.grid_zoom_distance = v

    @property
    def grid_size(self) -> float:
        return self._grid.grid_size if self._grid else 10.0

    @grid_size.setter
    def grid_size(self, v: float):
        if self._grid:
            self._grid.grid_size = v

    @property
    def clear_color(self) -> list:
        return self._clear_color

    @clear_color.setter
    def clear_color(self, v: list):
        self._clear_color = list(v[:3]) if v else [0.18, 0.18, 0.18]

    @property
    def ambient(self) -> list[float]:
        return self._ambient

    @ambient.setter
    def ambient(self, v: list[float]):
        self._ambient = v

    @property
    def render_mode(self) -> RenderMode:
        return self._render_mode

    @render_mode.setter
    def render_mode(self, v: RenderMode):
        self._render_mode = v

    @property
    def skybox_enabled(self) -> bool:
        return self._skybox_enabled

    @skybox_enabled.setter
    def skybox_enabled(self, v: bool):
        self._skybox_enabled = v

    def clear_scene_caches(self):
        """Clear per-frame caches on scene reload. Does NOT clear mesh/material
        caches to avoid reloading all 3D models after play mode toggle."""
        self._normal_cache.clear()
        self._import_meta_cache.clear()

    def release_all_caches(self):
        """Clear mesh, material and texture caches. Called when loading a
        completely different scene (not on play mode toggle)."""
        self._normal_cache.clear()
        self._import_meta_cache.clear()
        if self._materials:
            self._materials.clear_caches()
        if self._mesh_loader:
            self._mesh_loader.clear_scene_data()

    _effects_disabled: bool = False

    def set_effects_enabled(self, enabled: bool):
        self._effects_disabled = not enabled

    @property
    def effects_enabled(self) -> bool:
        return not self._effects_disabled

    def _sync_probuilder_meshes(self, scene):
        mesh_loader = self._mesh_loader
        if not mesh_loader:
            return
        if not hasattr(self, '_pb_scale_cache'):
            self._pb_scale_cache = {}
        for ent in scene.get_entities_with_component(ProBuilderMesh):
            if not ent.active:
                continue
            pb = ent.get_component(ProBuilderMesh)
            if not pb or not pb.enabled or pb.vertex_count == 0:
                continue
            tr = ent.get_component_by_name("Transform")
            if tr:
                s = tr.local_scale
                scale_key = (s.x, s.y, s.z)
                prev_scale = self._pb_scale_cache.get(ent.id)
                if prev_scale != scale_key:
                    self._pb_scale_cache[ent.id] = scale_key
                    pb.rebuild_uvs(world_scale=np.array([s.x, s.y, s.z], dtype=np.float32))
                    pb._gpu_dirty = True
            if not pb._gpu_dirty:
                continue
            mf = ent.get_component(MeshFilter)
            if not mf:
                mf = MeshFilter()
                ent.add_component(mf)
            mesh_name = f"ProBuilder_{ent.id[:6]}"
            mf.mesh_name = mesh_name
            gpu_mesh = pb.to_gpu_mesh()
            gpu_mesh.build_gl(self._ctx, self._default_prog)
            if self._outline_prog:
                gpu_mesh.build_outline_vao(self._ctx, self._outline_prog)
            mr = ent.get_component(MeshRenderer)
            if not mr:
                mr = MeshRenderer()
                ent.add_component(mr)
            cache_key = f"{mesh_name}|s=1.0|cp=False|fu=False"
            mesh_loader._meshes[cache_key] = gpu_mesh
            pb._gpu_dirty = False
        active_ids = {ent.id for ent in scene.get_entities_with_component(ProBuilderMesh) if ent.active}
        stale = [k for k in self._pb_scale_cache if k not in active_ids]
        for k in stale:
            del self._pb_scale_cache[k]

    def release(self):
        self._release_scene_fbo()
        self._release_pp_fbo()
        if self._batcher:
            self._batcher.release()
        if self._mesh_loader:
            self._mesh_loader.release()
        if self._grid:
            pass
        if self._gizmo:
            self._gizmo.release()
        if self._shadows:
            self._shadows.release()
        if self._particles:
            self._particles.release()
        if self._svgs:
            self._svgs.release()
        if self._text:
            self._text.release()
        if self._cloud_quad:
            self._cloud_quad.release()
        if self._cloud_plane:
            self._cloud_plane.release()
        if self._icons:
            self._icons.release()
        if self._materials:
            self._materials.release()
        if self._shaders:
            self._shaders.release()
        for buff in [self._quad_vbo, self._quad_ibo]:
            if buff:
                try:
                    buff.release()
                except Exception:
                    pass
        if self._quad_vao:
            try:
                self._quad_vao.release()
            except Exception:
                pass
        if self._pp_copy_vao:
            try:
                self._pp_copy_vao.release()
            except Exception:
                pass
        if self._projector_vao:
            try:
                self._projector_vao.release()
            except Exception:
                pass
        if self._gpu_storage:
            self._gpu_storage.release()
        for prog in [self._default_prog, self._grid_prog, self._gizmo_prog,
                     self._wireframe_prog, self._outline_prog,
                     self._gizmo_fatline_prog, self._gizmo_solid_prog,
                     self._shadow_prog, self._particle_prog, self._icon_prog, self._sprite_prog,
                     self._text_prog, self._overlay_prog, self._projector_prog, self._pp_copy_prog]:
            if prog:
                try:
                    prog.release()
                except Exception:
                    pass
        Logger.info("Renderer released.")
