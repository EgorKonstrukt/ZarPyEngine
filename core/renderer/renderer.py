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
from core.engine.engine import Engine
from core.foundation.logger import Logger

from core.components.rendering.renderers.mesh_filter import MeshFilter
from core.components.rendering.renderers.mesh_renderer import MeshRenderer
from core.components.rendering.renderers.skinned_mesh_renderer import SkinnedMeshRenderer
from core.components.rendering.skeleton.armature import Armature
from core.components.rendering.renderers.sprite_renderer import SpriteRenderer
from core.components.rendering.renderers.svg_renderer import SvgRenderer
from core.components.rendering.particles.particle_system import ParticleSystem
from core.components.rendering.particles.particle_force_field import ParticleForceField, FORCE_FIELD_DTYPE, MAX_FORCE_FIELDS
from core.components.mesh_editor import ProBuilderMesh
from core.components.rendering.postfx.graphics_effect import GraphicsEffect
from core.components.rendering.renderers.video_renderer import VideoRenderer
from core.components.rendering.effects.object_effect import ObjectEffect
from core.components.rendering.effects.voxelize_effect import VoxelizeEffect
from core.components.rendering.effects.voxel_cpu import compute_voxel_instances
from core.math.math3d import Mat4, Vec3

from core.renderer.types import RenderMode
from core.renderer.mesh_data import MeshData, read_shader
from core.renderer.meshes import make_cube_mesh, make_sphere_mesh, make_plane_mesh, make_quad_mesh, make_water_plane, make_water_box
from core.renderer.grid import GridRenderer
from core.renderer.gizmo import GizmoRenderer, FATLINE_VERT, FATLINE_FRAG
from core.renderer.shadows import ShadowRenderer
from core.components.rendering.environment.sky import Sky
from core.components.rendering.environment.clouds import Cloud
from core.components.rendering.environment.water import Water
from core.components.environment.wind_zone import WindZone
from core.components.physics.sphere_collider import SphereCollider
from core.components.physics.box_collider import BoxCollider
from core.components.physics.capsule_collider import CapsuleCollider
from core.components.physics.rigidbody import Rigidbody
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


def _halton(index: int, base: int) -> float:
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


_TAAU_JITTER = [(_halton(i, 2) - 0.5, _halton(i, 3) - 0.5) for i in range(1, 9)]


class _SpriteItem:
    __slots__ = ('world_matrix', 'color', 'flip_x', 'flip_y', 'texture_path', '_tr')
    def __init__(self, world_matrix, color, flip_x, flip_y, texture_path, tr=None):
        self.world_matrix = world_matrix
        self._tr = tr
        self.color = list(color) if color else [1, 1, 1, 1]
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.texture_path = texture_path

class _SvgItem:
    __slots__ = ('world_matrix', 'color', 'flip_x', 'flip_y', 'abs_path', 'pixels_per_unit', '_tr')
    def __init__(self, world_matrix, color, flip_x, flip_y, abs_path, pixels_per_unit, tr=None):
        self.world_matrix = world_matrix
        self._tr = tr
        self.color = list(color) if color else [1, 1, 1, 1]
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.abs_path = abs_path
        self.pixels_per_unit = pixels_per_unit

class _VideoItem:
    __slots__ = ('world_matrix', 'color', 'flip_x', 'flip_y', 'video_path', 'entity_id', 'loop', 'volume', 'offset', 'audio_source_entity_id', '_tr')
    def __init__(self, world_matrix, color, flip_x, flip_y, video_path, entity_id, loop, volume, offset=0.0, audio_source_entity_id="", tr=None):
        self.world_matrix = world_matrix
        self._tr = tr
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
                 'position', 'direction', 'up', 'flip_y', 'flip_x', 'cast_shadows', '_tr')
    def __init__(self, texture_path, color, intensity, range, spot_angle,
                 aspect_ratio, near_plane, far_plane, vp_matrix, position, direction, up,
                 flip_y=True, flip_x=False, cast_shadows=True, tr=None):
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
        self.up = np.array(up.to_array(), dtype=np.float32)
        self.flip_y = flip_y
        self.flip_x = flip_x
        self.cast_shadows = cast_shadows
        self._tr = tr

    def refresh_vp(self):
        tr = self._tr
        if tr is None:
            return
        pos = tr.position
        fwd = tr.forward
        up = tr.up
        view = Mat4.look_at(pos, pos + fwd, up)
        proj = Mat4.perspective(self.spot_angle, self.aspect_ratio, self.near_plane, self.far_plane)
        self.vp_matrix = (view @ proj).to_f32()
        self.position = np.array(pos.to_array(), dtype=np.float32)
        self.direction = np.array(fwd.to_array(), dtype=np.float32)
        self.up = np.array(up.to_array(), dtype=np.float32)

class _RenderSnapshot:
    __slots__ = (
        'lights', 'dir_light', 'sky_component', 'sky_entity', 'cloud_components',
        'water_components', 'wind_zones', 'renderable', 'shadow_renderables',         'sprite_items', 'video_items',
        'svg_items', 'text_items', 'particle_systems', 'force_fields',
        'projectors', 'skinned_renderables', 'skinned_shadow_renderables',
        'interactors',
    )
    def __init__(self):
        self.lights: list = []
        self.dir_light = None
        self.sky_component = None
        self.sky_entity = None
        self.cloud_components: list = []
        self.water_components: list = []
        self.wind_zones: list = []
        self.renderable: list = []
        self.shadow_renderables: list = []
        self.skinned_renderables: list = []
        self.skinned_shadow_renderables: list = []
        self.sprite_items: list = []
        self.video_items: list = []
        self.svg_items: list = []
        self.text_items: list = []
        self.particle_systems: list = []
        self.force_fields: list = []
        self.projectors: list = []
        self.interactors: list = []


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
        self._object_fx_prog: Optional[moderngl.Program] = None
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
        self._shadow_resolution: int = 1024
        self._shadow_distance: float = 50.0
        self._render_scale: float = 1.0
        self._line_width: float = 0.6667
        self._clear_color: list = [0.18, 0.18, 0.18]
        self._import_meta_cache: dict[str, tuple] = {}
        self._import_meta_mtime: dict[str, float] = {}
        self._snap_cache: Optional[_RenderSnapshot] = None
        self._snap_version: int = -1
        self._snap_struct_version: int = -1
        self._snap_mesh_gen: int = -1
        self._snap_scene: object = None
        self._normal_cache: dict[int, np.ndarray] = {}

        self._pp_fbo_a: Optional[moderngl.Framebuffer] = None
        self._pp_fbo_b: Optional[moderngl.Framebuffer] = None
        self._pp_color_tex_a: Optional[moderngl.Texture] = None
        self._pp_color_tex_b: Optional[moderngl.Texture] = None
        self._pp_fbo_size: tuple = (0, 0)
        self._pp_copy_prog: Optional[moderngl.Program] = None
        self._pp_copy_vao: Optional[moderngl.VertexArray] = None

        self._se_fbo_a: Optional[moderngl.Framebuffer] = None
        self._se_fbo_b: Optional[moderngl.Framebuffer] = None
        self._se_color_tex_a: Optional[moderngl.Texture] = None
        self._se_color_tex_b: Optional[moderngl.Texture] = None
        self._se_fbo_size: tuple = (0, 0)

        self._velocity_tex: Optional[moderngl.Texture] = None
        self._velocity_fbo: Optional[moderngl.Framebuffer] = None
        self._velocity_depth: Optional[moderngl.Renderbuffer] = None
        self._velocity_prog: Optional[moderngl.Program] = None
        self._velocity_vao: Optional[moderngl.VertexArray] = None
        self._velocity_geom_prog: Optional[moderngl.Program] = None
        self._prev_view_proj_by_target: dict[int, Mat4] = {}
        self._prev_model_by_entity: dict[int, Mat4] = {}
        self._velocity_fbo_size: tuple = (0, 0)

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
        self._water_plane: Optional[MeshData] = None
        self._water_chunk_mesh: Optional[MeshData] = None
        self._water_box_mesh: Optional[MeshData] = None
        self._water_mesh_cache: dict = {}
        self._water_fbo: Optional[moderngl.Framebuffer] = None
        self._water_color_tex: Optional[moderngl.Texture] = None
        self._water_sim_prog: Optional[moderngl.Program] = None
        self._water_sim_vao: Optional[moderngl.VertexArray] = None
        self._water_sim_a: Optional[moderngl.Texture] = None
        self._water_sim_b: Optional[moderngl.Texture] = None
        self._water_sim_fbo_a: Optional[moderngl.Framebuffer] = None
        self._water_sim_fbo_b: Optional[moderngl.Framebuffer] = None
        self._water_sim_size: int = 512
        self._water_sim_world: dict = {}
        self._water_depth_rb: Optional[moderngl.Renderbuffer] = None
        self._water_fbo_size: tuple = (0, 0)
        self._batcher: Optional[RenderBatcher] = None
        self._gpu_storage: Optional[GpuStorage] = None
        self._gpu_culling: Optional[GpuCulling] = None
        self._skinned_bone_ssbo: Optional[Any] = None
        self._skinned_bone_ssbo_capacity: int = 0
        self._render_count: int = 0
        self._vox_prog: Optional[moderngl.Program] = None
        self._vox_vao: Optional[moderngl.VertexArray] = None
        self._vox_cube_vbo: Optional[moderngl.Buffer] = None
        self._vox_cube_ibo: Optional[moderngl.Buffer] = None
        self._vox_inst_vbo: Optional[moderngl.Buffer] = None
        self._vox_inst_cap: int = 0
        self._vox_compute: Optional[moderngl.ComputeShader] = None

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
        self._render_scale = config.get("rendering.render_scale", self._render_scale)
        self._line_width = config.get("gizmo.line_width", self._line_width)

    def initialize(self):
        try:
            default_frag_src = read_shader("default.frag")
            default_frag_src = ShaderManager._inject_area_shadows(default_frag_src)
            self._default_prog = self._ctx.program(
                vertex_shader=read_shader("default.vert"),
                fragment_shader=default_frag_src
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
            try:
                self._underwater_prog = self._ctx.program(
                    vertex_shader=read_shader("shadow_overlay.vert"),
                    fragment_shader=read_shader("underwater.frag")
                )
            except Exception:
                self._underwater_prog = None
            if self._underwater_prog is not None:
                self._underwater_vao = self._ctx.vertex_array(
                    self._underwater_prog,
                    [(self._quad_vbo, '2f', 'in_position')],
                    self._quad_ibo
                )
            else:
                self._underwater_vao = None
            try:
                self._caustics_prog = self._ctx.program(
                    vertex_shader=read_shader("shadow_overlay.vert"),
                    fragment_shader=read_shader("caustics.frag")
                )
            except Exception:
                self._caustics_prog = None
            if self._caustics_prog is not None:
                self._caustics_vao = self._ctx.vertex_array(
                    self._caustics_prog,
                    [(self._quad_vbo, '2f', 'in_position')],
                    self._quad_ibo
                )
            else:
                self._caustics_vao = None
            VELOCITY_FRAG = """
#version 460 core
uniform sampler2D u_depth_tex;
uniform mat4 u_inv_view_proj;
uniform mat4 u_prev_view_proj;
uniform vec2 u_pixel_size;
in vec2 v_uv;
out vec2 frag_velocity;
void main() {
    float depth = texture(u_depth_tex, v_uv).r;
    if (depth >= 0.99999) {
        frag_velocity = vec2(0.0);
        return;
    }
    vec4 clip_pos = vec4(v_uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 world_pos = u_inv_view_proj * clip_pos;
    world_pos /= world_pos.w;
    vec4 prev_clip = u_prev_view_proj * world_pos;
    vec2 prev_uv = prev_clip.xy / prev_clip.w * 0.5 + 0.5;
    vec2 velocity = v_uv - prev_uv;
    if (isnan(velocity.x) || isnan(velocity.y) || isinf(velocity.x) || isinf(velocity.y)) {
        frag_velocity = vec2(0.0);
        return;
    }
    float max_v = 64.0 * max(u_pixel_size.x, u_pixel_size.y);
    velocity = clamp(velocity, -vec2(max_v), vec2(max_v));
    frag_velocity = velocity;
}
"""
            self._velocity_prog = self._ctx.program(
                vertex_shader=read_shader("shadow_overlay.vert"),
                fragment_shader=VELOCITY_FRAG
            )
            self._velocity_vao = self._ctx.vertex_array(
                self._velocity_prog,
                [(self._quad_vbo, '2f', 'in_position')],
                self._quad_ibo
            )
            VELOCITY_GEOM_VERT = """
#version 460 core
in vec3 in_position;
uniform mat4 u_view_proj;
uniform mat4 u_prev_view_proj;
uniform mat4 u_model;
uniform mat4 u_prev_model;
out vec2 v_velocity;
void main() {
    vec4 cur_clip = u_view_proj * u_model * vec4(in_position, 1.0);
    vec4 prev_clip = u_prev_view_proj * u_prev_model * vec4(in_position, 1.0);
    vec2 cur_ndc = cur_clip.xy / cur_clip.w;
    vec2 prev_ndc = prev_clip.xy / prev_clip.w;
    v_velocity = (prev_ndc - cur_ndc) * 0.5;
    gl_Position = cur_clip;
}
"""
            VELOCITY_GEOM_FRAG = """
#version 460 core
in vec2 v_velocity;
out vec2 frag_velocity;
void main() {
    vec2 vel = v_velocity;
    if (isnan(vel.x) || isnan(vel.y) || isinf(vel.x) || isinf(vel.y))
        vel = vec2(0.0);
    frag_velocity = vel;
}
"""
            self._velocity_geom_prog = self._ctx.program(
                vertex_shader=VELOCITY_GEOM_VERT,
                fragment_shader=VELOCITY_GEOM_FRAG
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
            self._compile_object_fx()
            self._init_voxel_instancing()
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
            self._water_plane = self._get_water_plane_mesh(200)
            self._water_chunk_mesh = self._get_water_plane_mesh(32)
            self._water_box_mesh = self._get_water_box_mesh(128)
            self._init_water_sim()
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

    def _compile_object_fx(self):
        self._fx_prog_cache: dict = {}
    _VOX_MAX_DIM = 128

    _VOX_MAX_INSTANCES = 2200000

    _FX_PASSTHROUGH_GEOM = """#version 460 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
in vec3 gs_world_pos[3];
in vec3 gs_normal[3];
in vec2 gs_uv[3];
in vec3 gs_view_pos[3];
in vec3 gs_local_pos[3];
out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out vec3 v_view_pos;
out vec3 v_local_pos;
void main() {
    for (int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;
        v_world_pos = gs_world_pos[i];
        v_normal = gs_normal[i];
        v_uv = gs_uv[i];
        v_view_pos = gs_view_pos[i];
        v_local_pos = gs_local_pos[i];
        EmitVertex();
    }
    EndPrimitive();
}
"""

    _VOXEL_INST_VERT = """#version 460 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec4 i_cell;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_vox_size;
uniform vec3 u_vox_scale;

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out vec3 v_view_pos;
out vec3 v_local_pos;

    void main() {
        vec3 s = max(vec3(0.01), u_vox_scale);
        vec3 h = u_vox_size * 0.9 * s;
    vec3 center = i_cell.xyz;
    vec3 local = center + vec3(in_position.x * h.x, in_position.y * h.y, in_position.z * h.z);
    vec4 world = u_model * vec4(local, 1.0);
    v_world_pos = world.xyz;
    v_local_pos = local;
    v_normal = in_normal;
    v_uv = in_uv;
    vec4 view_pos = u_view * world;
    v_view_pos = view_pos.xyz;
    gl_Position = u_proj * view_pos;
    if (i_cell.w < 0.0) {
        gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
    }
}
"""

    _VOXEL_INST_FRAG = """#version 460 core
in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;
in vec3 v_view_pos;
in vec3 v_local_pos;

uniform vec3 u_camera_pos;
uniform vec3 u_vox_color;
uniform float u_vox_amount;
uniform float u_vox_emission;
uniform float u_vox_rim;

out vec4 frag_color;

    void main() {
        vec3 N = normalize(v_normal);
        vec3 V = normalize(u_camera_pos - v_world_pos);
        vec3 L1 = normalize(vec3(0.5, 0.9, 0.4));
        vec3 L2 = normalize(vec3(-0.4, -0.3, -0.7));
        float ndl = max(dot(N, L1), 0.0) * 0.8 + max(dot(N, L2), 0.0) * 0.2;
        float fres = pow(1.0 - max(dot(N, V), 0.0), max(0.1, u_vox_rim));
        float edge = smoothstep(0.0, 0.06, v_uv.x) * smoothstep(0.0, 0.06, v_uv.y)
                   * smoothstep(0.0, 0.06, 1.0 - v_uv.x) * smoothstep(0.0, 0.06, 1.0 - v_uv.y);
        vec3 base = u_vox_color * (0.22 + 0.78 * ndl) * mix(0.82, 1.0, edge);
        float glow = u_vox_amount * (u_vox_emission + fres * u_vox_emission * 1.5);
        vec3 col = base + u_vox_color * glow;
        frag_color = vec4(col, 1.0);
    }
"""

    def _get_fx_program(self, fx_list):
        key = tuple(sorted(type(fx).__name__ for fx in fx_list))
        cached = self._fx_prog_cache.get(key)
        if cached is not None:
            return cached
        prog = None
        try:
            fx_vert = read_shader("object_fx.vert")
            fx_frag = read_shader("object_fx.frag")
            fx_frag = ShaderManager._inject_area_shadows(fx_frag)
            fx_frag = ShaderManager._inject_caustics(fx_frag)
            uniforms_block = "\n".join(fx.fx_fragment_uniforms() for fx in fx_list)
            main_block = "\n".join(fx.fx_fragment_snippet() for fx in fx_list)
            fx_frag = fx_frag.replace("// @FX_UNIFORMS", uniforms_block)
            fx_frag = fx_frag.replace("// @FX_MAIN", main_block)
            geom = self._FX_PASSTHROUGH_GEOM
            for fx in fx_list:
                gs = fx.fx_geometry_shader()
                if gs:
                    geom = gs
                    break
            prog = self._ctx.program(
                vertex_shader=fx_vert,
                fragment_shader=fx_frag,
                geometry_shader=geom,
            )
        except Exception as e:
            Logger.error(f"Failed to compile object_fx program for {key}: {e}", e)
            prog = None
        self._fx_prog_cache[key] = prog
        return prog


    def _build_voxel_cube(self):
        s = 0.5
        faces = [
            ((0.0, 0.0, 1.0), [(-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]),
            ((0.0, 0.0, -1.0), [(s, -s, -s), (-s, -s, -s), (-s, s, -s), (s, s, -s)]),
            ((1.0, 0.0, 0.0), [(s, -s, s), (s, -s, -s), (s, s, -s), (s, s, s)]),
            ((-1.0, 0.0, 0.0), [(-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s)]),
            ((0.0, 1.0, 0.0), [(-s, s, s), (s, s, s), (s, s, -s), (-s, s, -s)]),
            ((0.0, -1.0, 0.0), [(-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s)]),
        ]
        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        data = []
        idx = []
        base = 0
        for nrm, corners in faces:
            for i, c in enumerate(corners):
                data.append([c[0], c[1], c[2], nrm[0], nrm[1], nrm[2], uvs[i][0], uvs[i][1]])
            idx.extend([base, base + 1, base + 2, base, base + 2, base + 3])
            base += 4
        return np.array(data, dtype=np.float32), np.array(idx, dtype=np.uint32)

    def _init_voxel_instancing(self):
        try:
            self._vox_prog = self._ctx.program(
                vertex_shader=self._VOXEL_INST_VERT,
                fragment_shader=self._VOXEL_INST_FRAG,
            )
            cube_data, cube_idx = self._build_voxel_cube()
            self._vox_cube_vbo = self._ctx.buffer(cube_data.tobytes())
            self._vox_cube_ibo = self._ctx.buffer(cube_idx.astype(np.uint32).tobytes())
            self._vox_inst_vbo = self._ctx.buffer(reserve=1024 * 16)
            self._vox_inst_cap = 1024
            self._vox_vao = self._ctx.vertex_array(
                self._vox_prog,
                [
                    (self._vox_cube_vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv"),
                    (self._vox_inst_vbo, "4f /i", "i_cell"),
                ],
                self._vox_cube_ibo,
            )
            self._vox_compute = None
            comp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shaders", "voxelize.comp")
            if os.path.exists(comp_path):
                try:
                    with open(comp_path) as f:
                        src = f.read()
                    self._vox_compute = self._ctx.compute_shader(src)
                except Exception as e:
                    Logger.error(f"Voxel compute shader failed: {e}", e)
                    self._vox_compute = None
        except Exception as e:
            Logger.error(f"Voxel instancing init failed: {e}", e)
            self._vox_prog = None
            self._vox_vao = None
            self._vox_compute = None

    def _vox_ensure_res(self, fx, nverts, nidx, total):
        res = getattr(fx, "_vox_res", None)
        need_inst = min(total, self._VOX_MAX_INSTANCES)
        if res is None or res["nverts"] != nverts or res["nidx"] != nidx \
                or res["grid_cap"] < total or res["inst_cap"] < need_inst:
            if res is not None:
                for k in ("pos", "idx", "grid", "inst", "vao"):
                    obj = res.get(k)
                    if obj is not None:
                        try:
                            obj.release()
                        except Exception:
                            pass
            pos = self._ctx.buffer(reserve=max(1, nverts) * 16)
            idxf = self._ctx.buffer(reserve=max(1, nidx) * 4)
            grid_cap = max(total, res["grid_cap"] if res else 0)
            inst_cap = max(need_inst, res["inst_cap"] if res else 0)
            grid = self._ctx.buffer(reserve=max(1, grid_cap) * 4)
            inst = self._ctx.buffer(reserve=max(1, inst_cap) * 16)
            vao = self._ctx.vertex_array(
                self._vox_prog,
                [
                    (self._vox_cube_vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv"),
                    (inst, "4f /i", "i_cell"),
                ],
                self._vox_cube_ibo,
            )
            res = {
                "pos": pos, "idx": idxf, "grid": grid, "inst": inst,
                "vao": vao, "nverts": nverts, "nidx": nidx,
                "grid_cap": grid_cap, "inst_cap": inst_cap,
                "pos_key": None, "idx_key": None,
            }
            fx._vox_res = res
        return res

    def _render_voxel_instances(self, entry, fx, view_f32, proj_f32, cam_pos):
        mesh = entry[2]
        wm = entry[4]
        verts = getattr(mesh, "vertices", None)
        idx = getattr(mesh, "indices", None)
        if verts is None or len(verts) < 9 or self._vox_prog is None:
            return
        size = float(fx.voxel_size)
        if size <= 0.0:
            return
        world_grid = bool(fx.world_grid)
        model = np.asarray(wm._d, dtype=np.float32).T
        ctx = self._ctx
        prev_cull = ctx.cull_face
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        ctx.cull_face = 'back'
        ctx.disable(moderngl.BLEND)
        try:
            if self._vox_compute is not None:
                self._render_voxel_gpu(entry, fx, verts, idx, model, size, world_grid, view_f32, proj_f32, cam_pos)
            else:
                self._render_voxel_cpu(entry, fx, verts, idx, model, size, world_grid, view_f32, proj_f32, cam_pos)
        finally:
            ctx.enable(moderngl.DEPTH_TEST)
            if prev_cull:
                ctx.cull_face = prev_cull
            else:
                ctx.disable(moderngl.CULL_FACE)
            ctx.disable(moderngl.BLEND)

    def _set_vox_uniforms(self, prog, fx, wm, world_grid, size, view_f32, proj_f32, cam_pos):
        model = np.asarray(wm.to_f32(), dtype=np.float32).reshape(4, 4)
        prog["u_view"].write(view_f32.tobytes())
        prog["u_proj"].write(proj_f32.tobytes())
        prog["u_camera_pos"].write(np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32).tobytes())
        if world_grid:
            prog["u_model"].write(np.eye(4, dtype=np.float32).tobytes())
        else:
            prog["u_model"].write(model.tobytes())
        prog["u_vox_size"].value = size
        sc = fx.scale
        if hasattr(sc, "x"):
            sc = np.array([sc.x, sc.y, sc.z], dtype=np.float32)
        else:
            sc = np.asarray(sc, dtype=np.float32)
            if sc.shape[0] < 3:
                sc = np.full(3, float(sc[0]) if sc.shape[0] else 1.0, dtype=np.float32)
        prog["u_vox_scale"].write(sc[:3].tobytes())
        col = np.array(fx.color, dtype=np.float32)
        if col.shape[0] < 3:
            col = np.array([0.4, 1.0, 0.6], dtype=np.float32)
        prog["u_vox_color"].write(col[:3].tobytes())
        prog["u_vox_amount"].value = float(fx.amount)
        prog["u_vox_emission"].value = float(fx.emission)
        prog["u_vox_rim"].value = float(fx.rim)

    def _render_voxel_cpu(self, entry, fx, verts, idx, model, size, world_grid, view_f32, proj_f32, cam_pos):
        wm = entry[4]
        grid_min = None
        if world_grid:
            V = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
            ones = np.ones((V.shape[0], 1), dtype=np.float32)
            S = (model @ np.concatenate([V, ones], axis=1).T).T[:, :3]
            grid_min = np.floor(S.min(axis=0) / size) * size
        cells = fx.get_voxel_instances(verts, idx, model, size, world_grid, float(fx.jitter), grid_min)
        n = cells.shape[0]
        if n == 0 or self._vox_vao is None:
            return
        if n > self._vox_inst_cap:
            self._vox_inst_vbo.orphan(n * 16)
            self._vox_inst_cap = n
        self._vox_inst_vbo.write(cells[:n].tobytes())
        self._set_vox_uniforms(self._vox_prog, fx, wm, world_grid, size, view_f32, proj_f32, cam_pos)
        self._vox_vao.render(moderngl.TRIANGLES, instances=n)

    def _render_voxel_gpu(self, entry, fx, verts, idx, model, size, world_grid, view_f32, proj_f32, cam_pos):
        wm = entry[4]
        V = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
        nverts = V.shape[0]
        if idx is not None and len(idx) >= 3:
            nt = (len(idx) // 3) * 3
            I = np.asarray(idx[:nt], dtype=np.uint32)
        else:
            I = np.arange(nverts, dtype=np.uint32)
        ntris = I.shape[0] // 3
        if ntris == 0:
            return

        if world_grid:
            m = model
            ones = np.ones((nverts, 1), dtype=np.float32)
            S = (m @ np.concatenate([V, ones], axis=1).T).T[:, :3]
            smin = np.floor(S.min(axis=0) / size) * size
            smax = np.floor(S.max(axis=0) / size) * size + size
        else:
            S = V
            smin = S.min(axis=0)
            smax = S.max(axis=0)
        extent = np.maximum(smax - smin, 1e-4)
        cell = float(size)
        dims = np.ceil(extent / cell).astype(np.int64)
        over = np.maximum(dims - self._VOX_MAX_DIM, 0)
        if np.any(over > 0):
            factor = float(np.max(dims)) / self._VOX_MAX_DIM
            cell = cell * factor
            dims = np.ceil(extent / cell).astype(np.int64)
        dims = np.clip(dims, 1, self._VOX_MAX_DIM).astype(np.int32)
        total = int(dims[0]) * int(dims[1]) * int(dims[2])
        if total <= 0:
            return

        res = self._vox_ensure_res(fx, nverts, I.shape[0], total)

        pkey = (id(verts), nverts)
        if res["pos_key"] != pkey:
            P = np.zeros((nverts, 4), dtype=np.float32)
            P[:, :3] = V
            res["pos"].write(P.tobytes())
            res["pos_key"] = pkey
        ikey = (id(idx) if idx is not None else -1, I.shape[0])
        if res["idx_key"] != ikey:
            res["idx"].write(I.tobytes())
            res["idx_key"] = ikey

        res["grid"].clear()

        cs = self._vox_compute
        cs["u_pass"].value = 0
        cs["u_num_tris"].value = ntris
        cs["u_total_cells"].value = total
        cs["u_grid_min"].write(smin.astype(np.float32).tobytes())
        cs["u_cell_size"].value = cell
        cs["u_grid_dims"].write(dims.astype(np.int32).tobytes())
        cs["u_jitter"].value = 0.0
        cs["u_world_grid"].value = 1 if world_grid else 0
        cs["u_model"].write(wm.to_f32().tobytes())
        res["pos"].bind_to_storage_buffer(0)
        res["idx"].bind_to_storage_buffer(1)
        res["grid"].bind_to_storage_buffer(2)
        res["inst"].bind_to_storage_buffer(3)
        cs.run((ntris + 63) // 64, 1, 1)
        self._ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)

        cs["u_pass"].value = 1
        cs["u_jitter"].value = float(fx.jitter)
        cs.run((total + 63) // 64, 1, 1)
        try:
            bar = moderngl.SHADER_STORAGE_BARRIER_BIT | moderngl.VERTEX_ATTRIB_ARRAY_BARRIER_BIT
        except AttributeError:
            bar = moderngl.SHADER_STORAGE_BARRIER_BIT
        self._ctx.memory_barrier(bar)

        if total > res["inst_cap"]:
            return
        self._set_vox_uniforms(self._vox_prog, fx, wm, world_grid, cell, view_f32, proj_f32, cam_pos)
        res["vao"].render(moderngl.TRIANGLES, instances=total)



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
        self._scene_color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
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
        self._pp_color_tex_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._pp_color_tex_a.repeat_x = False
        self._pp_color_tex_a.repeat_y = False
        self._pp_fbo_a = self._ctx.framebuffer(self._pp_color_tex_a)
        self._pp_color_tex_b = self._ctx.texture((w, h), 4, dtype='f1')
        self._pp_color_tex_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
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

    def _ensure_se_fbo(self, w: int, h: int):
        if self._se_fbo_size == (w, h) and self._se_fbo_a:
            return
        self._release_se_fbo()
        self._se_color_tex_a = self._ctx.texture((w, h), 4, dtype='f1')
        self._se_color_tex_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._se_color_tex_a.repeat_x = False
        self._se_color_tex_a.repeat_y = False
        self._se_fbo_a = self._ctx.framebuffer(self._se_color_tex_a)
        self._se_color_tex_b = self._ctx.texture((w, h), 4, dtype='f1')
        self._se_color_tex_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._se_color_tex_b.repeat_x = False
        self._se_color_tex_b.repeat_y = False
        self._se_fbo_b = self._ctx.framebuffer(self._se_color_tex_b)
        self._se_fbo_size = (w, h)

    def _release_se_fbo(self):
        for obj in [self._se_fbo_a, self._se_color_tex_a, self._se_fbo_b, self._se_color_tex_b]:
            if obj:
                try:
                    obj.release()
                except Exception:
                    pass
        self._se_fbo_a = None
        self._se_color_tex_a = None
        self._se_fbo_b = None
        self._se_color_tex_b = None
        self._se_fbo_size = (0, 0)

    def _ensure_water_fbo(self, w: int, h: int):
        if self._water_fbo_size == (w, h) and self._water_fbo:
            return
        self._release_water_fbo()
        self._water_color_tex = self._ctx.texture((w, h), 4, dtype='f1')
        self._water_color_tex.repeat_x = False
        self._water_color_tex.repeat_y = False
        self._water_depth_rb = self._ctx.depth_renderbuffer((w, h))
        self._water_fbo = self._ctx.framebuffer(self._water_color_tex, depth_attachment=self._water_depth_rb)
        self._water_fbo_size = (w, h)

    def _release_water_fbo(self):
        for obj in [self._water_fbo, self._water_color_tex, self._water_depth_rb]:
            if obj:
                try:
                    obj.release()
                except Exception:
                    pass
        self._water_fbo = None
        self._water_color_tex = None
        self._water_depth_rb = None
        self._water_fbo_size = (0, 0)

    def _init_water_sim(self):
        try:
            self._water_sim_prog = self._ctx.program(
                vertex_shader=read_shader("shadow_overlay.vert"),
                fragment_shader=self._load_water_sim_frag(),
            )
        except Exception as e:
            Logger.error(f"Failed to init water sim: {e}", e)
            self._water_sim_prog = None
        if self._water_sim_prog is not None:
            quad_verts = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32)
            quad_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
            vbo = self._ctx.buffer(quad_verts.tobytes())
            ibo = self._ctx.buffer(quad_indices.tobytes())
            self._sim_quad_vbo = vbo
            self._sim_quad_ibo = ibo
            self._water_sim_vao = self._ctx.vertex_array(
                self._water_sim_prog,
                [(vbo, '2f', 'in_position')],
                ibo
            )
        self._ensure_water_sim_textures(self._water_sim_size)

    def _load_water_sim_frag(self) -> str:
        from core.assets.material import _extract_glsl_from_shader
        import os as _os
        path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "core", "shaders", "WaterSim.shader")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        _, frag = _extract_glsl_from_shader(text)
        return frag

    def _ensure_water_sim_textures(self, size: int):
        if self._water_sim_a is not None and self._water_sim_size == size:
            return
        for obj in [self._water_sim_a, self._water_sim_b, self._water_sim_fbo_a, self._water_sim_fbo_b]:
            if obj:
                try:
                    obj.release()
                except Exception:
                    pass
        self._water_sim_a = self._ctx.texture((size, size), 2, dtype='f2')
        self._water_sim_b = self._ctx.texture((size, size), 2, dtype='f2')
        self._water_sim_a.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._water_sim_b.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._water_sim_a.repeat_x = False
        self._water_sim_a.repeat_y = False
        self._water_sim_b.repeat_x = False
        self._water_sim_b.repeat_y = False
        self._water_sim_fbo_a = self._ctx.framebuffer(self._water_sim_a)
        self._water_sim_fbo_b = self._ctx.framebuffer(self._water_sim_b)
        self._water_sim_size = size
        self._water_sim_fbo_a.clear(color=(0.0, 0.0, 0.0, 1.0))
        self._water_sim_fbo_b.clear(color=(0.0, 0.0, 0.0, 1.0))

    def _step_water_sim(self, water_component, grid_center, grid_size, rest_y, dt):
        prog = self._water_sim_prog
        if prog is None or self._water_sim_vao is None:
            return None
        if not getattr(water_component, "interaction_enabled", True):
            return None
        size = self._water_sim_size
        self._ensure_water_sim_textures(size)
        interactors = getattr(self, "_last_interactors", [])
        count = min(len(interactors), 64)
        pos_arr = np.zeros((64, 4), dtype=np.float32)
        vel_arr = np.zeros((64, 4), dtype=np.float32)
        for i in range(count):
            ent, center, radius, vel = interactors[i][:4]
            vradius = interactors[i][4] if len(interactors[i]) > 4 else radius
            vy = vel.y if vel is not None else 0.0
            vx = vel.x if vel is not None else 0.0
            vz = vel.z if vel is not None else 0.0
            depth = center.y - rest_y
            pos_arr[i] = [center.x, center.z, max(radius, 0.05), depth]
            vel_arr[i] = [vx, vy, vz, max(vradius, 0.05)]
        prev = self._water_sim_a
        dst_fbo = self._water_sim_fbo_b
        prev.use(10)
        prog["_PrevState"] = 10
        prog["_Texel"] = (1.0 / size, 1.0 / size)
        prog["_Dt"] = float(min(dt, 0.05))
        prog["_Damping"] = float(getattr(water_component, "sim_damping", 0.04))
        prog["_Propagation"] = float(getattr(water_component, "sim_propagation", 18.0))
        prog["_Saturation"] = float(getattr(water_component, "sim_saturation", 4.0))
        prog["_GridSize"] = float(grid_size)
        prog["_GridCenter"] = (float(grid_center[0]), float(grid_center[2]))
        prog["_InteractionStrength"] = float(getattr(water_component, "interaction_strength", 1.0))
        prog["_InteractorCount"] = int(count)
        try:
            prog["_Interactors"].write(pos_arr.tobytes())
            prog["_InteractorVel"].write(vel_arr.tobytes())
        except Exception:
            pass
        old_fbo = self._ctx.fbo
        old_vp = self._ctx.viewport
        dst_fbo.use()
        dst_fbo.viewport = (0, 0, size, size)
        self._ctx.viewport = (0, 0, size, size)
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.disable(moderngl.BLEND)
        self._water_sim_vao.render()
        old_fbo.use()
        self._ctx.viewport = old_vp
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._water_sim_a, self._water_sim_b = self._water_sim_b, self._water_sim_a
        self._water_sim_fbo_a, self._water_sim_fbo_b = self._water_sim_fbo_b, self._water_sim_fbo_a
        return self._water_sim_a

    def _get_water_plane_mesh(self, res: int) -> MeshData:
        key = ("plane", int(res))
        m = self._water_mesh_cache.get(key)
        if m is None:
            m = make_water_plane(1.0, max(2, int(res)))
            m.build_gl(self._ctx, self._default_prog)
            self._water_mesh_cache[key] = m
        return m

    def _get_water_box_mesh(self, res: int) -> MeshData:
        key = ("box", int(res))
        m = self._water_mesh_cache.get(key)
        if m is None:
            m = make_water_box(max(2, int(res)), max(2, int(res) // 8), 1.0)
            m.build_gl(self._ctx, self._default_prog)
            self._water_mesh_cache[key] = m
        return m

    def _compute_water_chunk_models(self, cam_pos, water_y, ocean_size, chunk_size=200):
        from core.math.math3d import Mat4, Vec3
        grid_radius = max(1, int(round(ocean_size / (2.0 * chunk_size))))
        gcx = round(cam_pos.x / chunk_size) * chunk_size
        gcz = round(cam_pos.z / chunk_size) * chunk_size
        models = []
        for i in range(-grid_radius, grid_radius + 1):
            for j in range(-grid_radius, grid_radius + 1):
                cx = gcx + i * chunk_size
                cz = gcz + j * chunk_size
                model = Mat4.scale(Vec3(chunk_size, 1.0, chunk_size)) * Mat4.translation(Vec3(cx, water_y, cz))
                models.append(model)
        return models

    def _render_underwater_pass(self, w, h, view_f32, proj_f32, cam_pos,
                                sun_dir, sun_color, sun_intensity,
                                fog_color, caustic_color, depth_below,
                                cam_near, cam_far,
                                water_component=None, inv_view_proj_f32=None):
        if self._underwater_prog is None or self._underwater_vao is None:
            return
        self._ensure_pp_fbo(w, h)
        self._ctx.copy_framebuffer(self._pp_fbo_a, self._scene_fbo)
        self._scene_fbo.use()
        self._scene_fbo.viewport = (0, 0, w, h)
        self._ctx.viewport = (0, 0, w, h)
        old_mask = self._ctx.depth_mask
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.depth_mask = False
        prog = self._underwater_prog
        if "u_scene" in prog:
            self._pp_color_tex_a.use(0)
            prog["u_scene"] = 0
        if "u_depth" in prog:
            self._scene_depth_tex.use(1)
            prog["u_depth"] = 1
        # Caustics are driven by the same wave field as the surface, so they
        # follow the real waves and the sun. Use the water component's own
        # time origin to stay in phase with the surface displacement.
        t = time.time()
        if water_component is not None:
            try:
                t = time.time() - water_component._time_origin
            except Exception:
                pass
        if "u_time" in prog:
            prog["u_time"].value = t
        if "u_resolution" in prog:
            prog["u_resolution"].value = (float(w), float(h))
        if "u_cam_pos" in prog:
            prog["u_cam_pos"].write(np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32).tobytes())
        if "u_sun_dir" in prog:
            prog["u_sun_dir"].write(np.array([sun_dir.x, sun_dir.y, sun_dir.z], dtype=np.float32).tobytes())
        if "u_sun_color" in prog:
            prog["u_sun_color"].write(np.array(sun_color, dtype=np.float32).tobytes())
        if "u_sun_intensity" in prog:
            prog["u_sun_intensity"].value = float(sun_intensity)
        if "u_fog_color" in prog:
            prog["u_fog_color"].write(np.array(fog_color, dtype=np.float32).tobytes())
        if "u_caustic_color" in prog:
            prog["u_caustic_color"].write(np.array(caustic_color, dtype=np.float32).tobytes())
        if "u_depth_below" in prog:
            prog["u_depth_below"].value = float(depth_below)
        if "u_cam_near" in prog:
            prog["u_cam_near"].value = float(cam_near)
        if "u_cam_far" in prog:
            prog["u_cam_far"].value = float(cam_far)
        if "u_view" in prog:
            prog["u_view"].write(view_f32)
        if "u_proj" in prog:
            prog["u_proj"].write(proj_f32)
        if "u_inv_view_proj" in prog and inv_view_proj_f32 is not None:
            prog["u_inv_view_proj"].write(inv_view_proj_f32)
        if "u_fog_density" in prog:
            prog["u_fog_density"].value = 0.045
        # Projected caustics: feed the wave field + water level so the shader
        # can march the sun ray through the actual surface.
        if water_component is not None:
            surface_y = 0.0
            tr = water_component.transform
            if tr is not None:
                surface_y = float(tr.position.y)
            if "u_surface_y" in prog:
                prog["u_surface_y"].value = surface_y
            if "u_caustics_strength" in prog:
                prog["u_caustics_strength"].value = float(getattr(water_component, "caustics", 0.0))
            ws = water_component.get_wave_uniforms(t, cam_pos.x, cam_pos.z)
            if "_WaveCount" in prog:
                prog["_WaveCount"].value = ws["wave_count"]
                prog["_WaveDirection"].write(ws["dirs"].tobytes())
                prog["_WaveParams"].write(ws["params"].tobytes())
            if "_WindDir" in prog:
                prog["_WindDir"].write(np.array([ws["wind_dir"][0], ws["wind_dir"][1]], dtype=np.float32).tobytes())
            if "_WindSpeed" in prog:
                prog["_WindSpeed"].value = ws["wind_speed"]
            if "_WindGust" in prog:
                prog["_WindGust"].value = ws["wind_gust"]
            if "_WindTurbulence" in prog:
                prog["_WindTurbulence"].value = ws["wind_turbulence"]
            if "_WindAlign" in prog:
                prog["_WindAlign"].value = ws["wind_align"]
            if "_Choppiness" in prog:
                prog["_Choppiness"].value = ws["choppiness"]
            if "_MacroWave" in prog:
                prog["_MacroWave"].value = ws["macro_wave"]
            if "_Chaos" in prog:
                prog["_Chaos"].value = ws["chaos"]
        try:
            self._underwater_vao.render()
        finally:
            self._ctx.enable(moderngl.DEPTH_TEST)
            self._ctx.depth_mask = old_mask

    def _ensure_velocity_fbo(self, w: int, h: int):
        if self._velocity_fbo_size == (w, h) and self._velocity_fbo:
            return
        self._release_velocity_fbo()
        self._velocity_tex = self._ctx.texture((w, h), 2, dtype='f2')
        self._velocity_tex.repeat_x = False
        self._velocity_tex.repeat_y = False
        self._velocity_depth = self._ctx.depth_renderbuffer((w, h))
        self._velocity_fbo = self._ctx.framebuffer(self._velocity_tex, depth_attachment=self._velocity_depth)
        self._velocity_fbo_size = (w, h)

    def _release_velocity_fbo(self):
        for obj in [self._velocity_fbo, self._velocity_tex, self._velocity_depth]:
            if obj:
                try:
                    obj.release()
                except Exception:
                    pass
        self._velocity_fbo = None
        self._velocity_tex = None
        self._velocity_depth = None
        self._velocity_fbo_size = (0, 0)

    def _render_caustics_pass(self, w, h, view_f32, proj_f32, cam_pos,
                              sun_dir, sun_color, sun_intensity,
                              caustic_color, caustics_strength, surface_y,
                              water_component, inv_view_proj_f32,
                              cam_near, cam_far):
        """Material-independent underwater caustics.

        Projects physically based caustics (see caustics.glsl) onto every
        submerged pixel of the scene, regardless of the material / shader used
        by the geometry below the water. Runs from above or below the surface.
        """
        if self._caustics_prog is None or self._caustics_vao is None:
            return
        if water_component is None or caustics_strength <= 0.0:
            return
        self._ensure_pp_fbo(w, h)
        self._ctx.copy_framebuffer(self._pp_fbo_a, self._scene_fbo)
        self._scene_fbo.use()
        self._scene_fbo.viewport = (0, 0, w, h)
        self._ctx.viewport = (0, 0, w, h)
        old_mask = self._ctx.depth_mask
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.depth_mask = False
        prog = self._caustics_prog
        if "u_scene" in prog:
            self._pp_color_tex_a.use(0)
            prog["u_scene"] = 0
        if "u_depth" in prog:
            self._scene_depth_tex.use(1)
            prog["u_depth"] = 1
        if "u_time" in prog:
            try:
                prog["u_time"].value = time.time() - water_component._time_origin
            except Exception:
                prog["u_time"].value = time.time()
        if "u_resolution" in prog:
            prog["u_resolution"].value = (float(w), float(h))
        if "u_cam_pos" in prog:
            prog["u_cam_pos"].write(np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32).tobytes())
        if "u_sun_dir" in prog:
            prog["u_sun_dir"].write(np.array([sun_dir.x, sun_dir.y, sun_dir.z], dtype=np.float32).tobytes())
        if "u_sun_color" in prog:
            prog["u_sun_color"].write(np.array(sun_color, dtype=np.float32).tobytes())
        if "u_sun_intensity" in prog:
            prog["u_sun_intensity"].value = float(sun_intensity)
        if "u_caustic_tint" in prog:
            prog["u_caustic_tint"].write(np.array(caustic_color, dtype=np.float32).tobytes())
        if "u_caustics_strength" in prog:
            prog["u_caustics_strength"].value = float(caustics_strength)
        if "u_surface_y" in prog:
            prog["u_surface_y"].value = float(surface_y)
        if "u_cam_near" in prog:
            prog["u_cam_near"].value = float(cam_near)
        if "u_cam_far" in prog:
            prog["u_cam_far"].value = float(cam_far)
        if "u_inv_view_proj" in prog and inv_view_proj_f32 is not None:
            prog["u_inv_view_proj"].write(inv_view_proj_f32)
        # Feed the wave field so the caustic focusing matches the real surface.
        ws = water_component.get_wave_uniforms(time.time() - water_component._time_origin, cam_pos.x, cam_pos.z)
        if "_WaveCount" in prog:
            prog["_WaveCount"].value = ws["wave_count"]
            prog["_WaveDirection"].write(ws["dirs"].tobytes())
            prog["_WaveParams"].write(ws["params"].tobytes())
        if "_WindDir" in prog:
            prog["_WindDir"].write(np.array([ws["wind_dir"][0], ws["wind_dir"][1]], dtype=np.float32).tobytes())
        if "_WindSpeed" in prog:
            prog["_WindSpeed"].value = ws["wind_speed"]
        if "_WindGust" in prog:
            prog["_WindGust"].value = ws["wind_gust"]
        if "_WindTurbulence" in prog:
            prog["_WindTurbulence"].value = ws["wind_turbulence"]
        if "_WindAlign" in prog:
            prog["_WindAlign"].value = ws["wind_align"]
        if "_Choppiness" in prog:
            prog["_Choppiness"].value = ws["choppiness"]
        if "_MacroWave" in prog:
            prog["_MacroWave"].value = ws["macro_wave"]
        if "_Chaos" in prog:
            prog["_Chaos"].value = ws["chaos"]
        try:
            self._caustics_vao.render()
        finally:
            self._ctx.enable(moderngl.DEPTH_TEST)
            self._ctx.depth_mask = old_mask

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

    def _ensure_skinned_bone_ssbo(self, n_bones: int):
        needed = max(64, n_bones) * 64
        if self._skinned_bone_ssbo is not None and self._skinned_bone_ssbo_capacity >= needed:
            return
        if self._skinned_bone_ssbo is not None:
            try:
                self._skinned_bone_ssbo.release()
            except Exception:
                pass
        self._skinned_bone_ssbo = self._ctx.buffer(reserve=needed)
        self._skinned_bone_ssbo_capacity = needed

    def _bind_bone_ssbo(self, flat: np.ndarray):
        n = flat.shape[0]
        self._ensure_skinned_bone_ssbo(n)
        data = flat.tobytes()
        if self._skinned_bone_ssbo.size < len(data):
            try:
                self._skinned_bone_ssbo.release()
            except Exception:
                pass
            self._skinned_bone_ssbo = self._ctx.buffer(reserve=len(data) + 64)
            self._skinned_bone_ssbo_capacity = len(data) + 64
        self._skinned_bone_ssbo.write(data)
        self._skinned_bone_ssbo.bind_to_storage_buffer(6)

    def _render_skinned_meshes(self, snap, view_f32, proj_f32, cam_pos, lights):
        eng = Engine.instance()
        scene = eng.scene if eng and hasattr(eng, 'scene') else None
        if scene is None:
            return
        last_prog = None
        for entry in snap.skinned_renderables:
            ent, tr, mesh, smr, wm, armature, sub_idx = entry[:7]
            if armature is None or len(armature.bone_offset_matrices) == 0:
                continue
            flat, n_bones = armature.compute_skinning_buffer(scene, wm)
            if n_bones == 0:
                continue
            mat = self._materials.load_material(smr.get_material_path(sub_idx if sub_idx >= 0 else 0))
            shader_path = mat.shader_path if mat else ""
            prog = self._shaders.get_or_compile(shader_path) if shader_path else self._default_prog
            if prog is None:
                prog = self._default_prog
            if prog is not last_prog:
                self._set_scene_uniforms(prog, view_f32, proj_f32, cam_pos, lights,
                                          disable_shadows=not smr.receive_shadows)
                last_prog = prog
            model_f32 = wm.to_f32()
            if "u_model" in prog:
                prog["u_model"].write(model_f32.tobytes())
            try:
                from core.math_helpers import mat4_normal_matrix
                nm = mat4_normal_matrix(wm._d)
            except Exception:
                nm = np.eye(3, dtype=np.float32).T
            if "u_normal_matrix" in prog:
                prog["u_normal_matrix"].write(nm.tobytes())
            if "u_use_skinning" in prog:
                prog["u_use_skinning"].value = 1
            if "u_bone_count" in prog:
                prog["u_bone_count"].value = int(n_bones)
            self._bind_bone_ssbo(flat)
            self._materials.apply_material(mat, prog)
            if "u_use_instancing" in prog:
                prog["u_use_instancing"].value = 0
            if sub_idx >= 0 and mesh.sub_mesh_ranges:
                start, count = mesh.sub_mesh_ranges[sub_idx]
                mesh.render_range(prog, start, count)
            else:
                mesh.render(prog)
        if last_prog is not None and "u_use_skinning" in last_prog:
            last_prog["u_use_skinning"].value = 0


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

    def _find_object_effect(self, ent) -> list:
        result = []
        for comp in ent.get_all_components():
            if isinstance(comp, ObjectEffect) and comp.enabled:
                result.append(comp)
        return result

    def _collect_snapshot(self, scene, cam_near, cam_far, cam_fov, view_mat, proj_mat, cam_pos) -> _RenderSnapshot:
        n_updated = scene.flush_transforms()
        struct_version = scene._render_version
        mesh_gen = self._mesh_loader._loaded_generation if self._mesh_loader else 0
        if (self._snap_cache is not None
                and self._snap_scene is scene
                and self._snap_struct_version == struct_version
                and self._snap_mesh_gen == mesh_gen):
            self._refresh_snapshot_world_matrices(scene, n_updated)
            self._collect_interactors(self._snap_cache, scene)
            return self._snap_cache
        snap = _RenderSnapshot()
        if not self._import_meta_cache:
            self._preload_import_meta(scene)
        for ent in scene.get_entities_with_component(Light):
            if not ent.active:
                continue
            l = ent.get_component(Light)
            t = ent.transform
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
        for ent in scene.get_entities_with_component(Water):
            if ent.active:
                water = ent.get_component(Water)
                if water and water.enabled:
                    snap.water_components.append(water)
        for ent in scene.get_entities_with_component(WindZone):
            if ent.active:
                wz = ent.get_component(WindZone)
                if wz and wz.enabled:
                    snap.wind_zones.append(wz)
        self._collect_interactors(snap, scene)
        self._sync_probuilder_meshes(scene)
        needs_shadow = any(l.cast_shadows for l, _ in snap.lights)
        if not needs_shadow:
            for ent in scene.get_entities_with_component(Projector):
                if ent.active:
                    pj = ent.get_component(Projector)
                    if pj and pj.enabled and pj.cast_shadows:
                        needs_shadow = True
                        break
        for ent in scene.get_entities_with_component(MeshFilter):
            if not ent.active:
                continue
            mr = ent.get_component(MeshRenderer)
            tr = ent.transform
            if not tr or not mr or not mr.enabled:
                continue
            mf = ent.get_component(MeshFilter)
            mesh_name = mf.mesh_name
            scale, cp, fuvs = 1.0, False, False
            mesh_path = mf.mesh_path or ""
            if mesh_path:
                _meta = self._sync_import_meta(mesh_path)
                scale, cp, fuvs = _meta[0], _meta[1], _meta[2]
            if not mesh_name and not mesh_path:
                mesh_name = "cube"
            elif not mesh_name and mesh_path:
                mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
            mesh = self.get_or_create_mesh(mesh_name, mesh_path, scale, cp, fuvs)
            if mesh:
                wm = tr.world_matrix
                sub_ranges = mesh.sub_mesh_ranges
                fx_list = self._find_object_effect(ent)
                if sub_ranges:
                    for sub_idx in range(len(sub_ranges)):
                        snap.renderable.append([ent, tr, mesh, mr, wm, sub_idx, fx_list])
                else:
                    snap.renderable.append([ent, tr, mesh, mr, wm, -1, fx_list])
                if needs_shadow and mr.cast_shadows:
                    snap.shadow_renderables.append([mesh, tr])
        for ent in scene.get_entities_with_component(SkinnedMeshRenderer):
            if not ent.active:
                continue
            smr = ent.get_component(SkinnedMeshRenderer)
            tr = ent.transform
            if not tr or not smr or not smr.enabled:
                continue
            armature = ent.get_component(Armature)
            mesh_name = smr.mesh_name
            mesh_path = smr.mesh_path or ""
            if mesh_path:
                _meta = self._sync_import_meta(mesh_path)
                scale, cp, fuvs = _meta[0], _meta[1], _meta[2]
            else:
                scale, cp, fuvs = 1.0, False, False
            if not mesh_name and not mesh_path:
                continue
            elif not mesh_name and mesh_path:
                mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
            mesh = self.get_or_create_mesh(mesh_name, mesh_path, 1.0, False, fuvs)
            if not mesh or not getattr(mesh, 'has_skeleton', False):
                continue
            wm = tr.world_matrix
            sub_ranges = mesh.sub_mesh_ranges
            if sub_ranges:
                for sub_idx in range(len(sub_ranges)):
                    snap.skinned_renderables.append([ent, tr, mesh, smr, wm, armature, sub_idx])
            else:
                snap.skinned_renderables.append([ent, tr, mesh, smr, wm, armature, -1])
            if needs_shadow and smr.cast_shadows:
                snap.skinned_shadow_renderables.append([mesh, ent, armature, wm])
        for ent in scene.get_entities_with_component(SpriteRenderer):
            if not ent.active:
                continue
            sr = ent.get_component(SpriteRenderer)
            if not sr or not sr.enabled:
                continue
            tr = ent.transform
            if not tr:
                continue
            snap.sprite_items.append(_SpriteItem(
                tr.world_matrix, sr.color, sr.flip_x, sr.flip_y, sr.texture_path, tr))
        for ent in scene.get_entities_with_component(VideoRenderer):
            if not ent.active:
                continue
            vr = ent.get_component(VideoRenderer)
            if not vr or not vr.enabled:
                continue
            tr = ent.transform
            if not tr:
                continue
            snap.video_items.append(_VideoItem(
                tr.world_matrix, vr.color, vr.flip_x, vr.flip_y,
                vr.video_path, ent._id, vr.loop, vr.volume, vr.offset,
                vr.audio_source_entity_id, tr))
        for ent in scene.get_entities_with_component(SvgRenderer):
            if not ent.active:
                continue
            sr = ent.get_component(SvgRenderer)
            if not sr or not sr.enabled:
                continue
            tr = ent.transform
            if not tr:
                continue
            abs_path = self._svgs.resolve_path(sr.svg_path)
            snap.svg_items.append(_SvgItem(
                tr.world_matrix, sr.color, sr.flip_x, sr.flip_y,
                abs_path or "", sr.pixels_per_unit, tr))
        for ent in scene.get_entities_with_component(Projector):
            if not ent.active:
                continue
            pj = ent.get_component(Projector)
            if not pj or not pj.enabled:
                continue
            tr = ent.transform
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
                vp, pos, fwd, up, flip_y=pj.flip_y, flip_x=pj.flip_x,
                cast_shadows=pj.cast_shadows, tr=tr))
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
        self._snap_cache = snap
        self._snap_version = struct_version
        self._snap_struct_version = struct_version
        self._snap_mesh_gen = mesh_gen
        self._snap_scene = scene
        return snap

    def _collect_interactors(self, snap, scene):
        interactors = []
        collider_types = (SphereCollider, BoxCollider, CapsuleCollider)
        for ent in scene.get_entities_with_component(SphereCollider):
            c = ent.get_component(SphereCollider)
            if not c or not c.enabled or not ent.active:
                continue
            tr = ent.transform
            if not tr:
                continue
            rb = ent.get_component(Rigidbody)
            vel = rb.velocity if rb else Vec3.zero()
            center = tr.position + c.scaled_center
            interactors.append((ent, center, c.scaled_radius, vel, c.scaled_radius))
        for ent in scene.get_entities_with_component(BoxCollider):
            c = ent.get_component(BoxCollider)
            if not c or not c.enabled or not ent.active:
                continue
            tr = ent.transform
            if not tr:
                continue
            rb = ent.get_component(Rigidbody)
            vel = rb.velocity if rb else Vec3.zero()
            center = tr.position + c.scaled_center
            hsize = c.scaled_size * 0.5
            radius = max(hsize.x, hsize.z)
            if radius <= 0.0:
                continue
            interactors.append((ent, center, radius, vel, hsize.y))
        for ent in scene.get_entities_with_component(CapsuleCollider):
            c = ent.get_component(CapsuleCollider)
            if not c or not c.enabled or not ent.active:
                continue
            tr = ent.transform
            if not tr:
                continue
            rb = ent.get_component(Rigidbody)
            vel = rb.velocity if rb else Vec3.zero()
            center = tr.position + c.scaled_center
            interactors.append((ent, center, c.scaled_radius, vel, c.scaled_radius))
        capped = sorted(interactors, key=lambda it: abs(it[1].y), reverse=False)[:64]
        snap.interactors = capped

    def _refresh_snapshot_world_matrices(self, scene, n_updated: int):
        snap = self._snap_cache
        if snap is None:
            return
        renderable = snap.renderable
        for entry in renderable:
            tr = entry[1]
            if tr is not None:
                entry[4] = tr.world_matrix
        for entry in snap.skinned_renderables:
            tr = entry[1]
            if tr is not None:
                entry[4] = tr.world_matrix
        for entry in snap.skinned_shadow_renderables:
            ent = entry[1]
            tr = ent.transform if ent is not None else None
            if tr is not None:
                entry[3] = tr.world_matrix
        for item in snap.sprite_items:
            tr = item._tr
            if tr is not None:
                item.world_matrix = tr.world_matrix
        for item in snap.video_items:
            tr = item._tr
            if tr is not None:
                item.world_matrix = tr.world_matrix
        for item in snap.svg_items:
            tr = item._tr
            if tr is not None:
                item.world_matrix = tr.world_matrix
        for item in snap.projectors:
            item.refresh_vp()

    def _mat_double_sided(self, mat) -> bool:
        if mat is None:
            return False
        props = mat.properties
        return bool(props.get("double_sided") or props.get("_double_sided"))

    def _render_mesh_double_sided(self, prog, mesh, double_sided: bool):
        ctx = self._ctx
        cull_on = bool(ctx.cull_face)
        if double_sided and cull_on:
            ctx.disable(moderngl.CULL_FACE)
        try:
            if "u_double_sided" in prog:
                try:
                    prog["u_double_sided"].value = 1 if double_sided else 0
                except Exception:
                    pass
            mesh.render(prog)
        finally:
            if double_sided and cull_on:
                ctx.enable(moderngl.CULL_FACE)

    def _render_object_effects(self, entries, view_f32, proj_f32, cam_pos, lights,
                                 selected_entities, outline_queue):
        if not entries:
            return
        t = time.time()
        by_prog: dict = {}
        for entry in entries:
            fx_list = entry[6] if len(entry) > 6 else None
            if not fx_list:
                continue
            key = tuple(sorted(type(fx).__name__ for fx in fx_list))
            by_prog.setdefault(key, []).append(entry)
        for key, group in by_prog.items():
            fx_list = group[0][6]
            prog = self._get_fx_program(fx_list)
            if prog is None:
                continue
            self._set_scene_uniforms(prog, view_f32, proj_f32, cam_pos, lights, disable_shadows=False)
            for entry in group:
                ent, tr, mesh, mr = entry[:4]
                wm = entry[4]
                try:
                    mat = self._materials.load_material(mr.get_material_path(0))
                    model_f32 = wm.to_f32()
                    if "u_model" in prog:
                        prog["u_model"].write(model_f32.tobytes())
                    nm = np.eye(3, dtype=np.float32).T
                    try:
                        m3 = wm._d[:3, :3].copy()
                        m3[:, 0] /= max(1e-10, float(np.linalg.norm(m3[:, 0])))
                        m3[:, 1] /= max(1e-10, float(np.linalg.norm(m3[:, 1])))
                        m3[:, 2] /= max(1e-10, float(np.linalg.norm(m3[:, 2])))
                        nm = m3.T.astype(np.float32)
                    except Exception:
                        pass
                    if "u_normal_matrix" in prog:
                        prog["u_normal_matrix"].write(nm.tobytes())
                    center = [wm._d[0, 3], wm._d[1, 3], wm._d[2, 3]]
                    scale = max(
                        float(np.linalg.norm(wm._d[:3, 0])),
                        float(np.linalg.norm(wm._d[:3, 1])),
                        float(np.linalg.norm(wm._d[:3, 2])),
                        1e-4,
                    )
                    if "u_obj_center" in prog:
                        prog["u_obj_center"].write(np.array(center, dtype=np.float32).tobytes())
                    if "u_obj_scale" in prog:
                        prog["u_obj_scale"].value = scale
                    if "u_use_instancing" in prog:
                        prog["u_use_instancing"].value = 0
                    if "u_use_skinning" in prog:
                        prog["u_use_skinning"].value = 0
                    self._materials.apply_material(mat, prog)
                    ObjectEffect.reset_all_defaults(prog)
                    for fx in fx_list:
                        fx.bind(prog, t)
                    vox = next((f for f in fx_list if isinstance(f, VoxelizeEffect)), None)
                    ds = any(getattr(fx, 'double_sided', False) for fx in fx_list)
                    show_base = (vox is None) or (not vox.enabled) or vox.show_base_mesh
                    if show_base:
                        self._render_mesh_double_sided(prog, mesh, ds)
                    if vox is not None and vox.enabled and vox.amount > 0.0:
                        self._render_voxel_instances(entry, vox, view_f32, proj_f32, cam_pos)
                    if selected_entities and ent in selected_entities:
                        outline_queue.append((mesh, wm))
                except Exception as e:
                    Logger.error(f"ObjectEffect render failed on '{getattr(ent, 'name', '?')}': {e}", e)

    def render_scene(self, scene, view_mat: Mat4, proj_mat: Mat4, cam_pos: Vec3,
                     viewport_w: int, viewport_h: int, fbo=None,
                     selected_entities: Optional[set] = None,
                     cam_near: float = 0.01, cam_far: float = 1000.0, cam_fov: float = 60.0,
                      display_w: int = None, display_h: int = None):
        if not self._initialized:
            return
        _scale = self._render_scale
        rw = max(1, int(round(viewport_w * _scale)))
        rh = max(1, int(round(viewport_h * _scale)))
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
        water_components = snap.water_components
        renderable = snap.renderable
        if prof:
            prof.start("gl_state_setup")
        if fbo is not None:
            fbo.use()
            fbo.viewport = (0, 0, viewport_w, viewport_h)
        self._ctx.viewport = (0, 0, rw, rh)
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
        unjit_proj = proj_mat
        jitter_active = (not self._effects_disabled) and (display_w and display_h) and any(
            getattr(e, '_is_upscaler', False) and e.enabled and e.entity and e.entity.active
            for e in GraphicsEffect._registry
        )
        if jitter_active:
            self._jitter_index = (getattr(self, '_jitter_index', 0) + 1) % len(_TAAU_JITTER)
            hx, hy = _TAAU_JITTER[self._jitter_index]
            jx = hx * 2.0 / max(1, viewport_w)
            jy = hy * 2.0 / max(1, viewport_h)
            jd = proj_mat._d.copy()
            jd[:, 0] += jx * jd[:, 3]
            jd[:, 1] += jy * jd[:, 3]
            proj_mat = Mat4(jd)
        view_f32 = view_mat.to_f32()
        proj_f32 = proj_mat.to_f32()
        if prof:
            prof.stop("gl_state_setup")
        self._ensure_scene_fbo(rw, rh)
        self._ctx.disable(moderngl.BLEND)
        self._scene_fbo.use()
        self._scene_fbo.clear(0.0, 0.0, 0.0, 1.0, 1.0)
        aspect = rw / max(1, rh)
        if prof:
            prof.start("mesh_async_load")
        self._mesh_loader.process_pending()
        if prof:
            prof.stop("mesh_async_load")
        if prof:
            prof.start("render_shadow_pass")
        shadow_groups = {}
        try:
            shadow_groups = self._shadows.render_shadow_pass(snap.shadow_renderables, snap.lights, cam_near, cam_far, cam_fov, aspect, view_mat, {}, skinned_entries=snap.skinned_shadow_renderables, scene=scene)
            if snap.projectors:
                self._shadows.render_projector_shadows(snap.projectors, snap.shadow_renderables, shadow_groups)
        except Exception as _sh_err:
            import traceback as _tb
            _tb.print_exc()
        if prof:
            prof.stop("render_shadow_pass")
        self._scene_fbo.use()
        self._ctx.viewport = (0, 0, rw, rh)
        if sky_component and sky_component.enabled and self._skybox_cube and self._skybox_enabled:
            if prof:
                prof.start("render_skybox")
            sky_component.render_sky(self._ctx, self._shaders, view_mat, proj_mat, dir_light, self._skybox_cube)
            if prof:
                prof.stop("render_skybox")
        self._ctx.viewport = (0, 0, rw, rh)
        if use_polygon_mode:
            self._ctx.wireframe = True
        if prof:
            prof.start("process_pending_textures")
        self._materials.process_texture_pending()
        if prof:
            prof.stop("process_pending_textures")


        self._culled_total = len(renderable) if renderable else 0
        self._culled_visible = self._culled_total
        if renderable:
            try:
                n = len(renderable)
                from core._render_utils import build_frustum_cull_inputs as _bfci
                radii_in = np.array([entry[2].bounding_radius for entry in renderable], dtype=np.float64)
                centers, radii = _bfci(renderable, radii_in)
                vp = proj_mat._d.T @ view_mat._d.T
                visible = cpu_frustum_cull(centers, radii, vp)
                self._culled_visible = len(visible)
                if len(visible) < n:
                    renderable = [renderable[idx] for idx in visible]
            except Exception:
                import traceback; traceback.print_exc()

        if prof:
            prof.start("render_meshes")
        outline_queue: list[tuple[MeshData, Mat4]] = []
        fx_renderable = [e for e in renderable if len(e) > 6 and e[6]]
        if fx_renderable:
            self._render_object_effects(fx_renderable, view_f32, proj_f32, cam_pos, lights, selected_entities, outline_queue)
            renderable = [e for e in renderable if not (len(e) > 6 and e[6])]
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
                    mat = self._materials.load_material(mr.get_material_path(0))
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
                    ds = self._mat_double_sided(mat)
                    self._render_mesh_double_sided(prog, mesh, ds)
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
        if snap.skinned_renderables:
            if prof:
                prof.start("render_skinned")
            self._render_skinned_meshes(snap, view_f32, proj_f32, cam_pos, lights)
            if prof:
                prof.stop("render_skinned")
        if snap.projectors:
            self._scene_fbo.use()
            self._scene_fbo.viewport = (0, 0, rw, rh)
            self._ctx.viewport = (0, 0, rw, rh)
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
                self._text.render(scene, view_mat, proj_mat, rw, rh, world_space_only=True)
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
        if water_components and self._water_plane:
            if prof:
                prof.start("render_water")
            self._ensure_water_fbo(rw, rh)
            self._ctx.copy_framebuffer(self._water_fbo, self._scene_fbo)
            self._water_fbo.use()
            self._water_fbo.viewport = (0, 0, rw, rh)
            self._ctx.viewport = (0, 0, rw, rh)
            self._ctx.enable(moderngl.DEPTH_TEST)
            self._ctx.enable(moderngl.CULL_FACE)
            self._ctx.cull_face = 'back'
            self._last_interactors = snap.interactors
            now_t = time.time()
            sim_dt = max(0.001, min(now_t - getattr(self, "_last_water_sim_time", now_t - 0.016), 0.05))
            self._last_water_sim_time = now_t
            for water_component in water_components:
                tr = water_component.transform
                water_y = tr.position.y if tr else 0.0
                if water_component.infinite_ocean and water_component.surface_type == "Ocean":
                    chunk_models = self._compute_water_chunk_models(cam_pos, water_y, water_component.ocean_size)
                    res = int(getattr(water_component, "mesh_resolution", 32.0))
                    mesh = self._get_water_plane_mesh(max(2, res))
                    grid_center = cam_pos
                    grid_size = float(getattr(water_component, "ocean_size", 2000.0))
                    sim_tex = self._step_water_sim(water_component, grid_center, grid_size, water_y, sim_dt)
                    water_component.render_water(self._ctx, self._shaders, view_mat, proj_mat,
                                                 dir_light, cam_pos, mesh,
                                                 self._scene_color_tex, self._scene_depth_tex,
                                                 (rw, rh), cam_near, cam_far,
                                                 snap.wind_zones, snap.lights, chunk_models,
                                                 sim_tex=sim_tex, sim_grid_center=grid_center,
                                                 sim_grid_size=grid_size,
                                                 sim_disp_scale=getattr(water_component, "sim_disp_scale", 1.0),
                                                 sim_normal_scale=getattr(water_component, "sim_normal_scale", 1.0))
                elif water_component.surface_type == "Pond":
                    res = int(getattr(water_component, "mesh_resolution", 128.0))
                    mesh = self._get_water_box_mesh(res)
                    grid_center = tr.position if tr else cam_pos
                    grid_size = float(getattr(water_component, "pond_size", 20.0))
                    sim_tex = self._step_water_sim(water_component, grid_center, grid_size, water_y, sim_dt)
                    water_component.render_water(self._ctx, self._shaders, view_mat, proj_mat,
                                                 dir_light, cam_pos, mesh,
                                                 self._scene_color_tex, self._scene_depth_tex,
                                                 (rw, rh), cam_near, cam_far,
                                                 snap.wind_zones, snap.lights, None, is_box=True,
                                                 sim_tex=sim_tex, sim_grid_center=grid_center,
                                                 sim_grid_size=grid_size,
                                                 sim_disp_scale=getattr(water_component, "sim_disp_scale", 1.0),
                                                 sim_normal_scale=getattr(water_component, "sim_normal_scale", 1.0))
                else:
                    res = int(getattr(water_component, "mesh_resolution", 200.0))
                    mesh = self._get_water_plane_mesh(res)
                    grid_center = tr.position if tr else cam_pos
                    grid_size = float(getattr(water_component, "pond_size", 200.0))
                    sim_tex = self._step_water_sim(water_component, grid_center, grid_size, water_y, sim_dt)
                    water_component.render_water(self._ctx, self._shaders, view_mat, proj_mat,
                                                 dir_light, cam_pos, mesh,
                                                 self._scene_color_tex, self._scene_depth_tex,
                                                 (rw, rh), cam_near, cam_far,
                                                 snap.wind_zones, snap.lights, None,
                                                 sim_tex=sim_tex, sim_grid_center=grid_center,
                                                 sim_grid_size=grid_size,
                                                 sim_disp_scale=getattr(water_component, "sim_disp_scale", 1.0),
                                                 sim_normal_scale=getattr(water_component, "sim_normal_scale", 1.0))
            self._scene_fbo.use()
            self._scene_fbo.viewport = (0, 0, rw, rh)
            self._ctx.viewport = (0, 0, rw, rh)
            self._ctx.disable(moderngl.DEPTH_TEST)
            self._pp_copy_prog["u_input_tex"] = 0
            self._water_color_tex.use(0)
            self._pp_copy_vao.render()
            self._ctx.enable(moderngl.DEPTH_TEST)
            if prof:
                prof.stop("render_water")

        # Material-independent underwater caustics: project wave/sun-driven
        # caustics onto every submerged pixel of the scene (works with any
        # material / shader on the geometry below the water, from above or
        # below the surface). Uses the topmost water surface as the projector.
        caustic_water = None
        caustic_best_y = float("-inf")
        for wc in water_components:
            if getattr(wc, "caustics", 0.0) <= 0.0:
                continue
            tr = wc.transform
            sy = tr.position.y if tr else 0.0
            if sy > caustic_best_y:
                caustic_best_y = sy
                caustic_water = wc
        if caustic_water is not None and self._scene_depth_tex is not None:
            if prof:
                prof.start("render_caustics")
            csun_dir = Vec3(0.0, 1.0, 0.0)
            csun_color = [1.0, 1.0, 1.0]
            csun_intensity = 1.0
            if dir_light:
                dl, dt = dir_light
                csun_dir = -dt.forward
                if dl.procedural_sky_lighting:
                    sc, si = Light.compute_sun_light(-dt.forward)
                    csun_color = sc
                    csun_intensity = si
                else:
                    csun_color = dl.color
                    csun_intensity = dl.intensity
            c_tint = getattr(caustic_water, "sss_color", [0.0, 0.55, 0.45])
            c_strength = float(getattr(caustic_water, "caustics", 0.0))
            c_surface_y = caustic_best_y
            try:
                c_inv_vp = (view_mat @ proj_mat).inverted().to_f32()
            except Exception:
                c_inv_vp = None
            self._render_caustics_pass(
                rw, rh, view_f32, proj_f32, cam_pos,
                csun_dir, csun_color, csun_intensity,
                c_tint, c_strength, c_surface_y,
                caustic_water, c_inv_vp, cam_near, cam_far
            )
            if prof:
                prof.stop("render_caustics")
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
            self._ensure_pp_fbo(rw, rh)
            self._ctx.copy_framebuffer(self._pp_fbo_a, self._scene_fbo)
            self._pp_fbo_a.use()
            self._pp_fbo_a.viewport = (0, 0, rw, rh)
            self._ctx.viewport = (0, 0, rw, rh)
            self._ctx.disable(moderngl.DEPTH_TEST)
            for cloud_component in cloud_components:
                cloud_component.render_clouds(self._ctx, self._shaders, view_mat, proj_mat, dir_light, cam_pos, self._cloud_quad, self._shadows, self._scene_depth_tex, (rw, rh))
            self._scene_fbo.use()
            self._scene_fbo.viewport = (0, 0, rw, rh)
            self._ctx.viewport = (0, 0, rw, rh)
            self._ctx.disable(moderngl.DEPTH_TEST)
            self._pp_copy_prog["u_input_tex"] = 0
            self._pp_color_tex_a.use(0)
            self._pp_copy_vao.render()
            self._ctx.enable(moderngl.DEPTH_TEST)
            if prof:
                prof.stop("render_clouds")
        underwater_water = None
        best_depth = float("inf")
        for wc in water_components:
            tr = wc.transform
            surface_y = tr.position.y if tr else 0.0
            if cam_pos.y < surface_y:
                depth_below = surface_y - cam_pos.y
                if depth_below < best_depth:
                    best_depth = depth_below
                    underwater_water = wc
        if underwater_water is not None and self._scene_depth_tex is not None:
            if prof:
                prof.start("render_underwater")
            sun_dir = Vec3(0.0, 1.0, 0.0)
            sun_color = [1.0, 1.0, 1.0]
            sun_intensity = 1.0
            if dir_light:
                dl, dt = dir_light
                sun_dir = -dt.forward
                if dl.procedural_sky_lighting:
                    sc, si = Light.compute_sun_light(-dt.forward)
                    sun_color = sc
                    sun_intensity = si
                else:
                    sun_color = dl.color
                    sun_intensity = dl.intensity
            fog_color = getattr(underwater_water, "deep_color", [0.02, 0.18, 0.28])
            caustic_color = getattr(underwater_water, "sss_color", [0.0, 0.55, 0.45])
            try:
                inv_vp_f32 = (view_mat @ proj_mat).inverted().to_f32()
            except Exception:
                inv_vp_f32 = None
            self._render_underwater_pass(
                rw, rh, view_f32, proj_f32, cam_pos,
                sun_dir, sun_color, sun_intensity, fog_color, caustic_color,
                best_depth, cam_near, cam_far,
                water_component=underwater_water, inv_view_proj_f32=inv_vp_f32
            )
            if prof:
                prof.stop("render_underwater")
        if prof:
            prof.start("render_overlay")
        dw = display_w if display_w else viewport_w
        dh = display_h if display_h else viewport_h
        if fbo is not None:
            fbo.use()
            fbo.viewport = (0, 0, dw, dh)
        else:
            self._ctx.screen.use()
        self._ctx.viewport = (0, 0, dw, dh)
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._pp_copy_prog["u_input_tex"] = 0
        self._scene_color_tex.use(0)
        self._pp_copy_vao.render()
        self._ctx.enable(moderngl.DEPTH_TEST)
        if prof:
            prof.stop("render_overlay")

        if scene:
            from core.components.rendering.renderers.raytracing_renderer import RaytracingRenderer
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
            from core.components.rendering.environment.radiance_cascades_gi import RadianceCascadesGI
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
        self._triangles_drawn = 0
        self._vertices_drawn = 0
        for entry in renderable:
            mesh = entry[2]
            if hasattr(mesh, 'indices') and mesh.indices is not None and len(mesh.indices) > 0:
                self._triangles_drawn += len(mesh.indices) // 3
            if hasattr(mesh, 'vertices') and mesh.vertices is not None and len(mesh.vertices) > 0:
                self._vertices_drawn += len(mesh.vertices) // 3
        if prof:
            prof.stop("render_stats")
        _pv_key = id(fbo) if fbo is not None else 0
        prev_view_proj = self._prev_view_proj_by_target.get(_pv_key)
        self._prev_view_proj_by_target[_pv_key] = unjit_proj @ view_mat
        if GraphicsEffect._registry and not self._effects_disabled:
            GraphicsEffect.increment_frame()
            if prof:
                prof.start("render_graphics_effects")
            self._ctx.disable(moderngl.DEPTH_TEST)

            has_velocity_effects = any(
                getattr(e, '_use_velocity', False) for e in GraphicsEffect._registry
                if e.enabled and e.entity and e.entity.active
            )
            velocity_tex = None
            if has_velocity_effects and renderable and prev_view_proj is not None and self._velocity_geom_prog:
                try:
                    self._ensure_velocity_fbo(rw, rh)
                    self._velocity_fbo.use()
                    self._velocity_fbo.viewport = (0, 0, rw, rh)
                    self._velocity_fbo.clear(red=0.0, green=0.0, depth=1.0)
                    self._ctx.enable(moderngl.DEPTH_TEST)
                    self._ctx.enable(moderngl.CULL_FACE)
                    self._ctx.cull_face = 'back'
                    cur_vp = unjit_proj @ view_mat
                    prog = self._velocity_geom_prog
                    prog["u_view_proj"].write(cur_vp.to_f32().tobytes())
                    prog["u_prev_view_proj"].write(prev_view_proj.to_f32().tobytes())
                    for entry in renderable:
                        ent = entry[0]
                        wm = entry[4]
                        mesh = entry[2]
                        prev_model = self._prev_model_by_entity.get(id(ent))
                        if prev_model is None:
                            continue
                        prog["u_model"].write(wm.to_f32().tobytes())
                        prog["u_prev_model"].write(prev_model.to_f32().tobytes())
                        mesh.render(prog)
                    for entry in renderable:
                        ent = entry[0]
                        wm = entry[4]
                        self._prev_model_by_entity[id(ent)] = wm
                    velocity_tex = self._velocity_tex
                    self._scene_fbo.use()
                    self._scene_fbo.viewport = (0, 0, rw, rh)
                    self._ctx.disable(moderngl.DEPTH_TEST)
                except Exception as e:
                    Logger.error(f"Velocity geometry pass error: {e}")
                    import traceback; traceback.print_exc()
                    velocity_tex = None

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
                    extra = {}
                    if getattr(effect, '_use_velocity', False):
                        extra['velocity_tex'] = velocity_tex
                        extra['prev_view_proj'] = prev_view_proj
                    effect.render(self._ctx, self._scene_color_tex, self._scene_depth_tex,
                                  view_mat, proj_mat, cam_pos, viewport_w, viewport_h,
                                  **extra)
                except Exception as e:
                    Logger.error(f"GraphicsEffect.render error: {e}")
            disp_w = display_w if display_w else viewport_w
            disp_h = display_h if display_h else viewport_h
            composite_src = self._scene_color_tex
            if screen_effects:
                has_upscaler = any(getattr(e, '_is_upscaler', False) for e in screen_effects)
                pipe_w = disp_w if has_upscaler else viewport_w
                pipe_h = disp_h if has_upscaler else viewport_h
                self._ensure_se_fbo(pipe_w, pipe_h)
                self._ctx.disable(moderngl.BLEND)
                self._se_fbo_a.use()
                self._se_fbo_a.viewport = (0, 0, pipe_w, pipe_h)
                self._se_fbo_a.clear(0.0, 0.0, 0.0, 0.0, 1.0)
                self._se_fbo_b.use()
                self._se_fbo_b.viewport = (0, 0, pipe_w, pipe_h)
                self._se_fbo_b.clear(0.0, 0.0, 0.0, 0.0, 1.0)
                self._se_fbo_a.use()
                self._se_fbo_a.viewport = (0, 0, pipe_w, pipe_h)
                self._pp_copy_prog["u_input_tex"] = 0
                self._scene_color_tex.use(0)
                self._pp_copy_vao.render()
                src_fbo = self._se_fbo_a
                dst_fbo = self._se_fbo_b
                src_tex = self._se_color_tex_a
                for effect in screen_effects:
                    dst_fbo.use()
                    dst_fbo.viewport = (0, 0, pipe_w, pipe_h)
                    try:
                        if prof:
                            prof.start(f"effect_{effect.__class__.__name__}")
                        extra = {}
                        if getattr(effect, '_use_velocity', False):
                            extra['velocity_tex'] = velocity_tex
                            extra['prev_view_proj'] = prev_view_proj
                        effect.render(self._ctx, self._scene_color_tex, self._scene_depth_tex,
                                      view_mat, proj_mat, cam_pos, pipe_w, pipe_h,
                                      input_tex=src_tex, output_fbo=dst_fbo,
                                      render_w=viewport_w, render_h=viewport_h,
                                      **extra)
                        if prof:
                            time_elapsed = prof.stop(f"effect_{effect.__class__.__name__}")
                            if time_elapsed is not None and time_elapsed > 0.01:
                                Logger.warning(f"{effect.__class__.__name__} took {time_elapsed*1000:.1f}ms")
                    except Exception as e:
                        Logger.error(f"GraphicsEffect.render error: {e}")
                    src_fbo, dst_fbo = dst_fbo, src_fbo
                    src_tex = src_fbo.color_attachments[0]
                composite_src = src_tex
            self._ctx.disable(moderngl.BLEND)
            if fbo is not None:
                fbo.use()
                fbo.viewport = (0, 0, disp_w, disp_h)
            elif self._ctx.screen is not None:
                self._ctx.screen.use()
                self._ctx.viewport = (0, 0, disp_w, disp_h)
            self._pp_copy_prog["u_input_tex"] = 0
            composite_src.use(0)
            self._pp_copy_vao.render()
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            if prof:
                prof.stop("render_graphics_effects")

        if prof:
            prof.start("render_text")
        if self._text and scene:
            with eng._scene_lock:
                self._text.render(scene, view_mat, proj_mat, rw, rh, world_space_only=False)
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

    def _resolve_import_meta_path(self, mesh_path: str) -> str:
        direct = mesh_path + ".import"
        if os.path.exists(direct):
            return direct
        eng = None
        try:
            from core.engine.engine import Engine
            eng = Engine.instance()
        except Exception:
            pass
        root = (eng.project_root if eng and getattr(eng, "project_root", None) else os.getcwd())
        base = os.path.basename(mesh_path)
        candidates = [os.path.join(root, mesh_path + ".import")]
        for sub in ["", "assets/", "assets/models/", "models/"]:
            candidates.append(os.path.join(root, sub, mesh_path + ".import"))
            candidates.append(os.path.join(root, sub, base + ".import"))
        for c in candidates:
            if os.path.exists(c):
                return c
        return direct

    def _sync_import_meta(self, mesh_path: str) -> tuple:
        if not mesh_path:
            return (1.0, False, True, 30.0, True, True)
        import_cache = self._resolve_import_meta_path(mesh_path)
        try:
            mtime = os.path.getmtime(import_cache) if os.path.exists(import_cache) else -1.0
        except OSError:
            mtime = -1.0
        cached_mtime = self._import_meta_mtime.get(mesh_path)
        if cached_mtime == mtime and mesh_path in self._import_meta_cache:
            return self._import_meta_cache[mesh_path]
        self._import_meta_mtime[mesh_path] = mtime
        if os.path.exists(import_cache):
            try:
                with open(import_cache) as _f:
                    _s = json.load(_f)
                meta = (
                    float(_s.get("scale", 1.0)),
                    bool(_s.get("center_pivot", False)),
                    bool(_s.get("flip_uvs", False)),
                    float(_s.get("smooth_angle", 30.0)),
                    bool(_s.get("gen_normals", True)),
                    bool(_s.get("gen_uvs", True)),
                )
            except Exception:
                meta = (1.0, False, True, 30.0, True, True)
        else:
            meta = (1.0, False, True, 30.0, True, True)
        old = self._import_meta_cache.get(mesh_path)
        self._import_meta_cache[mesh_path] = meta
        if old is not None and old != meta and self._mesh_loader is not None:
            prefix = mesh_path + "|"
            for k in [k for k in list(self._mesh_loader._meshes.keys()) if k.startswith(prefix)]:
                self._mesh_loader._meshes.pop(k, None)
        return meta

    def _preload_import_meta(self, scene):
        paths = set()
        for ent in scene.get_entities_with_component(MeshFilter):
            mf = ent.get_component(MeshFilter)
            if mf and mf.mesh_path:
                paths.add(mf.mesh_path)
        for mesh_path in paths:
            self._sync_import_meta(mesh_path)

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
            _meta = self._sync_import_meta(mesh_path)
            cache_key = f"{mesh_path}|s={_meta[0]}|cp={_meta[1]}|fu={_meta[2]}"
            mesh = meshes.get(cache_key)
            if mesh:
                return mesh
        return meshes.get("cube")

    def render_entity_outline(self, entity, model_mat: Mat4, view_mat: Mat4, proj_mat: Mat4, color: list[float]):
        if not self._outline_prog:
            return
        from core.components.rendering.renderers.mesh_filter import MeshFilter
        from core.components.rendering.renderers.mesh_renderer import MeshRenderer
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

    def render_icons_batched(self, batches: list, viewport_w: int, viewport_h: int):
        if self._icons:
            self._icons.render_batched(batches, viewport_w, viewport_h)

    _render_icons_batched = render_icons_batched

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
        self._import_meta_mtime.clear()
        self._snap_cache = None
        self._snap_version = -1
        self._snap_scene = None

    def release_all_caches(self):
        """Clear mesh, material and texture caches. Called when loading a
        completely different scene (not on play mode toggle)."""
        self._normal_cache.clear()
        self._import_meta_cache.clear()
        self._import_meta_mtime.clear()
        self._snap_cache = None
        self._snap_version = -1
        self._snap_scene = None
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
            tr = ent.transform
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
            mesh_loader.bump_generation()
            pb._gpu_dirty = False
        active_ids = {ent.id for ent in scene.get_entities_with_component(ProBuilderMesh) if ent.active}
        stale = [k for k in self._pb_scale_cache if k not in active_ids]
        for k in stale:
            del self._pb_scale_cache[k]

    def release(self):
        GraphicsEffect.cleanup_registry()
        self._prev_view_proj_by_target.clear()
        self._release_scene_fbo()
        self._release_pp_fbo()
        self._release_se_fbo()
        self._release_water_fbo()
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
        if self._velocity_vao:
            try:
                self._velocity_vao.release()
            except Exception:
                pass
        self._release_velocity_fbo()
        if self._gpu_storage:
            self._gpu_storage.release()
        for prog in [self._default_prog, self._grid_prog, self._gizmo_prog,
                     self._wireframe_prog, self._outline_prog,
                     self._gizmo_fatline_prog, self._gizmo_solid_prog,
                     self._shadow_prog, self._particle_prog, self._icon_prog, self._sprite_prog,
                     self._text_prog, self._overlay_prog, self._projector_prog, self._pp_copy_prog,
                     self._velocity_prog]:
            if prog:
                try:
                    prog.release()
                except Exception:
                    pass
        Logger.info("Renderer released.")