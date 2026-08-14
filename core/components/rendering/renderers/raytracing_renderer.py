# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import numpy as np
import moderngl
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.components.rendering.renderers.mesh_filter import MeshFilter
from core.components.rendering.renderers.mesh_renderer import MeshRenderer
from core.components.lighting.light import Light, LightType

from core.maths.math3d import Mat4, Vec3
from core.foundation.logger import Logger
import math

_INST_STRIDE = 46
_MAX_INSTANCES = 256
_MAX_LIGHTS = 8
_MAT_STRIDE = 20
_MAX_ALBEDO_LAYERS = 32

try:
    from core._raytracing_data import prepare_raytrace_data as _cy_prepare
    _USE_CYTHON_RT = True
except ImportError:
    _USE_CYTHON_RT = False


@ComponentRegistry.register
class RaytracingRenderer(Component):
    _allow_multiple = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("enabled", "Enabled", FieldType.BOOL),
            InspectorField("_compute_shader_path", "Compute Shader", FieldType.RESOURCE_PATH, file_filter="Compute (*.compute)"),
            InspectorField("_resolution_scale", "Resolution Scale", FieldType.FLOAT, 0.1, 1.0),
            InspectorField("_max_bounces", "Max Bounces", FieldType.INT, 1, 16),
            InspectorField("_samples_per_pixel", "Samples Per Pixel", FieldType.INT, 1, 16),
            InspectorField("_accumulate", "Accumulate Frames", FieldType.BOOL),
            InspectorField("_show_overlay", "Show Overlay", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self._compute_shader_path: str = "core/shaders/Raytracing.compute"
        self._resolution_scale: float = 0.5
        self._max_bounces: int = 1
        self._samples_per_pixel: int = 1
        self._accumulate: bool = False
        self._show_overlay: bool = True

        self._program: Optional[moderngl.ComputeShader] = None
        self._output_tex: Optional[moderngl.Texture] = None
        self._output_fbo: Optional[moderngl.Framebuffer] = None
        self._emissive_tex: Optional[moderngl.Texture] = None
        self._fullscreen_quad: Optional[moderngl.VertexArray] = None
        self._fullscreen_prog: Optional[moderngl.Program] = None

        self._sky_env_tex: Optional[moderngl.Texture] = None
        self._sky_env_prog: Optional[moderngl.ComputeShader] = None

        self._albedo_array_tex: Optional[moderngl.TextureArray] = None
        self._albedo_tex_map: dict = {}
        self._albedo_array_size: tuple = (256, 256)
        self._albedo_count: int = 0

        self._bvh_buf: Optional[moderngl.Buffer] = None
        self._vert_buf: Optional[moderngl.Buffer] = None
        self._idx_buf: Optional[moderngl.Buffer] = None
        self._mat_buf: Optional[moderngl.Buffer] = None
        self._inst_buf: Optional[moderngl.Buffer] = None
        self._light_buf: Optional[moderngl.Buffer] = None

        self._ctx_id = 0
        self._bvh_np: Optional[np.ndarray] = None
        self._vert_np: Optional[np.ndarray] = None
        self._idx_np: Optional[np.ndarray] = None
        self._mat_np: Optional[np.ndarray] = None
        self._inst_np: Optional[np.ndarray] = None
        self._light_np: Optional[np.ndarray] = None

        self._fixed_bvh: Optional[np.ndarray] = None
        self._fixed_vert: Optional[np.ndarray] = None
        self._fixed_idx: Optional[np.ndarray] = None
        self._fixed_mat: Optional[np.ndarray] = None
        self._fixed_inst: Optional[np.ndarray] = None
        self._fixed_light: Optional[np.ndarray] = None

        self._accum_frame: int = 1
        self._frame: int = 0
        self._prev_view_proj: Optional[np.ndarray] = None
        self._prev_width: int = 0
        self._prev_height: int = 0
        self._rays_per_frame: int = 0

        self._mesh_geo_cache: dict[int, tuple] = {}
        self._geo_signature: Optional[tuple] = None
        self._geo_offsets: list = []
        self._geo_dirty: bool = True

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "compute_shader_path": self._compute_shader_path,
            "resolution_scale": self._resolution_scale,
            "samples_per_pixel": self._samples_per_pixel,
            "accumulate": self._accumulate,
            "show_overlay": self._show_overlay,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> RaytracingRenderer:
        r = cls()
        r.enabled = data.get("enabled", True)
        r._compute_shader_path = data.get("compute_shader_path", "core/shaders/Raytracing.compute")
        r._resolution_scale = float(data.get("resolution_scale", 0.5))
        r._samples_per_pixel = int(data.get("samples_per_pixel", 1))
        r._accumulate = data.get("accumulate", False)
        r._show_overlay = data.get("show_overlay", True)
        return r

    def _ensure_resources(self, ctx: moderngl.Context, width: int, height: int):
        rw = max(1, int(width * self._resolution_scale))
        rh = max(1, int(height * self._resolution_scale))

        if self._program is None:
            path = os.path.abspath(self._compute_shader_path)
            if not os.path.exists(path):
                Logger.error(f"Raytracing compute shader not found: {path}")
                return False
            try:
                with open(path) as f:
                    src = f.read()
                glsl_start = src.find("GLSLPROGRAM")
                glsl_end = src.find("ENDGLSL", glsl_start)
                if glsl_start < 0 or glsl_end < 0:
                    Logger.error("Invalid .compute file: no GLSLPROGRAM/ENDGLSL")
                    return False
                source = src[glsl_start + len("GLSLPROGRAM"):glsl_end].strip()
                self._program = ctx.compute_shader(source)
            except Exception as e:
                Logger.error(f"Failed to compile compute shader: {e}")
                return False

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

        if self._output_tex is None or self._prev_width != rw or self._prev_height != rh:
            if self._output_tex:
                self._output_tex.release()
            if self._output_fbo:
                self._output_fbo.release()
            if self._emissive_tex:
                self._emissive_tex.release()
            self._output_tex = ctx.texture((rw, rh), 4, dtype="f4")
            self._output_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._output_tex.repeat_x = False
            self._output_tex.repeat_y = False
            self._output_fbo = ctx.framebuffer(color_attachments=[self._output_tex])
            self._emissive_tex = ctx.texture((rw, rh), 4, dtype="f4")
            self._emissive_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._emissive_tex.repeat_x = False
            self._emissive_tex.repeat_y = False
            self._prev_width = rw
            self._prev_height = rh
            self._accum_frame = 1

        if self._albedo_array_tex is None:
            self._albedo_array_tex = ctx.texture_array((*self._albedo_array_size, _MAX_ALBEDO_LAYERS), 4, dtype="f1")
            self._albedo_array_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._albedo_array_tex.repeat_x = False
            self._albedo_array_tex.repeat_y = False

        if self._sky_env_tex is None:
            self._sky_env_tex = ctx.texture((256, 128), 4, dtype="f4")
            self._sky_env_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._sky_env_tex.repeat_x = False
            self._sky_env_tex.repeat_y = False

        if self._sky_env_prog is None:
            env_path = os.path.abspath("core/shaders/SkyEnv.compute")
            if not os.path.exists(env_path):
                Logger.error(f"SkyEnv compute shader not found: {env_path}")
                return False
            try:
                with open(env_path) as f:
                    src = f.read()
                glsl_start = src.find("GLSLPROGRAM")
                glsl_end = src.find("ENDGLSL", glsl_start)
                if glsl_start < 0 or glsl_end < 0:
                    Logger.error("Invalid SkyEnv.compute: no GLSLPROGRAM/ENDGLSL")
                    return False
                source = src[glsl_start + len("GLSLPROGRAM"):glsl_end].strip()
                self._sky_env_prog = ctx.compute_shader(source)
            except Exception as e:
                Logger.error(f"Failed to compile SkyEnv compute shader: {e}")
                return False

        return True

    def _build_mesh_geo(self, mesh, bvh) -> tuple:
        verts3 = mesh.vertices.reshape(-1, 3)
        idxs = mesh.indices.reshape(-1, 3)
        nv = verts3.shape[0]
        nt = idxs.shape[0]
        vert8 = np.zeros((nv, 8), dtype=np.float32)
        vert8[:, :3] = verts3
        norms = getattr(mesh, 'normals', None)
        if norms is not None and norms.shape[0] == nv * 3:
            vert8[:, 3:6] = norms.reshape(-1, 3)
        else:
            f0 = verts3[0::3]; f1 = verts3[1::3]; f2 = verts3[2::3]
            e1 = f1 - f0
            e2 = f2 - f0
            face_norms = np.cross(e1, e2)
            fn_len = np.linalg.norm(face_norms, axis=1, keepdims=True)
            fn_len[fn_len == 0] = 1
            vert8[:, 3:6] = np.repeat(face_norms / fn_len, 3, axis=0)
        uvs = getattr(mesh, 'uvs', None)
        if uvs is not None and len(uvs) >= nv * 2:
            vert8[:, 6:8] = uvs.reshape(-1, 2)[:nv]
        if bvh.tri_indices is not None and len(bvh.tri_indices) == nt:
            idx_local = idxs[bvh.tri_indices].astype(np.uint32, copy=False)
        else:
            idx_local = idxs.astype(np.uint32, copy=False)
        bvh_flat = np.asarray(bvh.flatten_for_gpu(), dtype=np.float32)
        nn = bvh_flat.shape[0]
        internal_mask = bvh_flat[:, 7] >= 0
        root = bvh.nodes[-1]
        lbmin = np.asarray(root.bmin, dtype=np.float32)
        lbmax = np.asarray(root.bmax, dtype=np.float32)
        return vert8, idx_local, bvh_flat, internal_mask, nn, nt, nv, bvh, lbmin, lbmax

    def _rebuild_geometry_buffers(self, geo_entries: list) -> None:
        total_v = sum(g[6] for g in geo_entries)
        total_t = sum(g[5] for g in geo_entries)
        total_n = sum(g[4] for g in geo_entries)
        vert_np = np.empty((total_v, 8), dtype=np.float32)
        idx_np = np.empty((total_t, 3), dtype=np.uint32)
        bvh_np = np.empty((total_n, 8), dtype=np.float32)
        offsets = []
        vo, io, bo = 0, 0, 0
        for vert8, idx_local, bvh_flat, internal_mask, nn, nt, nv, _bvh, _lbmin, _lbmax in geo_entries:
            vert_np[vo:vo + nv] = vert8
            idx_np[io:io + nt] = idx_local + vo
            if bo > 0 and nn > 0 and internal_mask.any():
                blk = bvh_flat.copy()
                blk[internal_mask, 6] += bo
                blk[internal_mask, 7] += bo
            else:
                blk = bvh_flat
            bvh_np[bo:bo + nn] = blk
            offsets.append((vo, io, bo, nn, nt))
            vo += nv; io += nt; bo += nn
        self._vert_np = vert_np
        self._idx_np = idx_np.reshape(-1)
        self._bvh_np = bvh_np.reshape(-1)
        self._geo_offsets = offsets
        self._geo_dirty = True

    def _collect_and_upload(self, ctx: moderngl.Context, scene, view_mat: Mat4, proj_mat: Mat4, cam_pos: Vec3,
                            renderer):
        from core.spatial.bvh import get_mesh_bvh

        instances = []
        material_map = {}

        mf_list = scene.get_entities_with_component(MeshFilter)
        for ent in mf_list:
            if len(instances) >= _MAX_INSTANCES:
                break
            mr = ent.get_component(MeshRenderer)
            tr = ent.transform
            if not tr or not mr or not mr.enabled:
                continue
            mf = ent.get_component(MeshFilter)
            mesh_name = mf.mesh_name
            mesh_path = mf.mesh_path or ""
            scale, cp, fuvs = 1.0, False, False
            if mesh_path:
                meta = renderer._import_meta_cache.get(mesh_path) if hasattr(renderer, '_import_meta_cache') else None
                if meta is None:
                    meta = (1.0, False, False, 30.0, True, True)
                scale, cp, fuvs = meta[0], meta[1], meta[2]
            if not mesh_name:
                mesh_name = "cube" if not mesh_path else os.path.splitext(os.path.basename(mesh_path))[0]
            mesh = renderer.get_or_create_mesh(mesh_name, mesh_path, scale, cp, fuvs)
            if not mesh or mesh.vertices is None or len(mesh.vertices) < 3:
                continue
            bvh = get_mesh_bvh(mesh.vertices, mesh.indices)
            if not bvh or not bvh.nodes:
                continue

            mat_path = mr.get_material_path(0)
            if mat_path and mat_path not in material_map:
                material_map[mat_path] = len(material_map)

            instances.append((Mat4(tr.world_matrix._d), mat_path, mesh, bvh))

        if not instances:
            return False

        n_inst = len(instances)
        geo_entries = []
        sig_parts = []
        cache = self._mesh_geo_cache
        for _wm, _mat_path, mesh, bvh in instances:
            key = id(mesh)
            geo = cache.get(key)
            if geo is None or geo[7] is not bvh:
                geo = self._build_mesh_geo(mesh, bvh)
                cache[key] = geo
            geo_entries.append(geo)
            sig_parts.append(key)
        signature = tuple(sig_parts)

        if signature != self._geo_signature:
            self._rebuild_geometry_buffers(geo_entries)
            self._geo_signature = signature
            if len(cache) > max(64, n_inst * 4):
                active = set(sig_parts)
                for stale_key in [k for k in cache if k not in active]:
                    del cache[stale_key]

        offsets = self._geo_offsets

        n_mats = max(len(material_map), 1)
        mat_np = np.zeros((n_mats, _MAT_STRIDE), dtype=np.float32)
        mat_np[:, :3] = 0.8
        mat_np[:, 3] = 0.0
        mat_np[:, 4] = 0.5
        mat_np[:, 10] = 1.0
        mat_np[:, 11] = 1.0
        mat_np[:, 13] = 1.5
        mat_np[:, 9] = -1.0

        mmgr = getattr(renderer, '_materials', None)
        if mmgr:
            for mat_path, mi in material_map.items():
                mat = mmgr.load_material(mat_path)
                if mat:
                    props = mat.properties
                    bc = props.get("_BaseColor", None)
                    if bc is None:
                        bc = props.get("albedo_color", (0.8, 0.8, 0.8, 1.0))
                    mat_np[mi, 0] = float(bc[0])
                    mat_np[mi, 1] = float(bc[1])
                    mat_np[mi, 2] = float(bc[2])
                    mat_np[mi, 3] = float(props.get("_Metallic", 0.0))
                    mat_np[mi, 4] = float(props.get("_Smoothness", 0.5))
                    ec = props.get("_EmissionColor")
                    if ec is None:
                        ec = props.get("emission_color", (0, 0, 0, 0))
                    if ec is None:
                        ec = (0, 0, 0, 0)
                    mat_np[mi, 5] = float(ec[0])
                    mat_np[mi, 6] = float(ec[1])
                    mat_np[mi, 7] = float(ec[2])
                    mat_np[mi, 8] = float(props.get("_EmissionIntensity", 0.0))
                    mat_np[mi, 10] = float(props.get("_OcclusionStrength", 1.0))
                    mat_np[mi, 11] = float(bc[3]) if bc is not None and len(bc) > 3 else 1.0
                    mat_np[mi, 12] = float(props.get("_Transmission", 0.0))
                    mat_np[mi, 13] = float(props.get("_IOR", 1.5))
                    albedo_tex_path = props.get("_BaseMap")
                    if not albedo_tex_path:
                        albedo_tex_path = props.get("albedo_texture", "")
                    if albedo_tex_path and isinstance(albedo_tex_path, str):
                        layer = self._get_or_load_albedo_layer(albedo_tex_path, mmgr)
                        if layer >= 0:
                            mat_np[mi, 9] = float(layer)
        self._mat_np = mat_np

        wm_list = np.array([wm._d for wm, _, _, _ in instances])
        inv_wm_list = np.linalg.inv(wm_list)

        inst_np = np.empty((n_inst, _INST_STRIDE), dtype=np.float32)
        wm_f32 = np.array([Mat4(w).to_f32() for w in wm_list])
        inv_w_f32 = np.array([Mat4(w).to_f32() for w in inv_wm_list])
        inst_np[:, :16] = wm_f32.reshape(n_inst, 16)
        inst_np[:, 16:32] = inv_w_f32.reshape(n_inst, 16)

        _bvh_roots = np.array([float(offsets[i][2] + offsets[i][3] - 1) for i in range(n_inst)])
        _vert_offs = np.array([float(offsets[i][0]) for i in range(n_inst)])
        _idx_offs = np.array([float(offsets[i][1]) for i in range(n_inst)])
        _mat_idxs = np.array([float(material_map.get(instances[i][1], 0)) for i in range(n_inst)])
        _tri_counts = np.array([float(offsets[i][4]) for i in range(n_inst)])
        _node_counts = np.array([float(offsets[i][3]) for i in range(n_inst)])
        inst_np[:, 32] = _bvh_roots
        inst_np[:, 33] = _vert_offs
        inst_np[:, 34] = _idx_offs
        inst_np[:, 35] = _mat_idxs
        inst_np[:, 36] = _tri_counts
        inst_np[:, 37] = _node_counts

        lbmins = np.array([geo_entries[i][8] for i in range(n_inst)])
        lbmaxs = np.array([geo_entries[i][9] for i in range(n_inst)])
        centers = (lbmins + lbmaxs) * 0.5
        extents = (lbmaxs - lbmins) * 0.5
        r3 = wm_list[:, :3, :3]
        wcenters = np.einsum('ij,ikj->ik', centers, r3) + wm_list[:, 3, :3]
        wextents = np.einsum('ij,ikj->ik', extents, np.abs(r3))
        inst_np[:, 38:41] = wcenters - wextents
        inst_np[:, 41:44] = wcenters + wextents

        self._inst_np = inst_np

        lights_list = []
        lights_ents = scene.get_entities_with_component(Light)
        for ent in lights_ents:
            if len(lights_list) >= _MAX_LIGHTS:
                break
            if not ent.active:
                continue
            l = ent.get_component(Light)
            t = ent.transform
            if not l or not l.enabled or not t:
                continue
            lt = 0
            if l.light_type == LightType.DIRECTIONAL:
                lt = 0
            elif l.light_type == LightType.POINT:
                lt = 1
            elif l.light_type == LightType.SPOT:
                lt = 2
            fwd = t.forward
            c = l.color
            spot_cos = math.cos(math.radians(l.spot_angle))
            inner_cos = math.cos(math.radians(l.spot_inner_angle))
            _, eff_int = Light.shader_radiance(l, t)
            lights_list.append([
                float(lt), t.position.x, t.position.y, t.position.z,
                fwd.x, fwd.y, fwd.z,
                c[0], c[1], c[2],
                eff_int, l.range, spot_cos, inner_cos,
            ])
        n_lights = min(len(lights_list), _MAX_LIGHTS)
        light_np = np.zeros((max(n_lights, 1), 14), dtype=np.float32)
        if n_lights > 0:
            light_np[:n_lights] = np.array(lights_list[:n_lights], dtype=np.float32)
        self._light_np = light_np

        if self._geo_dirty:
            self._upload_or_realloc(ctx, '_bvh_buf', self._bvh_np)
            self._upload_or_realloc(ctx, '_vert_buf', self._vert_np)
            self._upload_or_realloc(ctx, '_idx_buf', self._idx_np)
            self._geo_dirty = False
        self._upload_or_realloc(ctx, '_mat_buf', self._mat_np)
        self._upload_or_realloc(ctx, '_inst_buf', self._inst_np)
        self._upload_or_realloc(ctx, '_light_buf', self._light_np)

        return True

    def _upload_or_realloc(self, ctx: moderngl.Context, buf_attr: str, data: np.ndarray) -> None:
        buf = getattr(self, buf_attr)
        nbytes = data.nbytes
        if buf is None or nbytes > buf.size:
            if buf:
                buf.release()
            nbuf = ctx.buffer(data, dynamic=True)
            setattr(self, buf_attr, nbuf)
        else:
            buf.write(data)

    def _dispatch(self, ctx: moderngl.Context, width: int, height: int,
                  view_mat: Mat4, proj_mat: Mat4, cam_pos: Vec3, scene, renderer) -> bool:
        ctx_id = id(ctx)
        if self._ctx_id != ctx_id:
            self._release_gl()
            self._ctx_id = ctx_id

        rw = max(1, int(width * self._resolution_scale))
        rh = max(1, int(height * self._resolution_scale))

        if self._accumulate:
            cur_vp = (proj_mat.to_f32().reshape(4, 4) @ view_mat.to_f32().reshape(4, 4))
            if self._prev_view_proj is not None:
                if np.max(np.abs(cur_vp - self._prev_view_proj)) > 1e-5:
                    self._accum_frame = 1
            self._prev_view_proj = cur_vp.copy()

        if not self._ensure_resources(ctx, width, height):
            return False

        if not self._collect_and_upload(ctx, scene, view_mat, proj_mat, cam_pos, renderer):
            return False

        if self._sky_env_tex is None or self._sky_env_prog is None:
            return False

        view_f32 = view_mat.to_f32().reshape(4, 4).T
        proj_f32 = proj_mat.to_f32().reshape(4, 4).T

        rectified, _ = np.linalg.qr(view_f32)
        if np.linalg.det(rectified) < 0:
            rectified[:, 0] = -rectified[:, 0]
        inv_vp = np.linalg.inv(proj_f32 @ view_f32)

        prog = self._program
        try:
            prog["u_camera_pos"] = (cam_pos.x, cam_pos.y, cam_pos.z)
            prog["u_inv_view_proj"].write(inv_vp.astype(np.float32, copy=False).flatten(order='F').tobytes())
            prog["u_screen_width"] = rw
            prog["u_screen_height"] = rh
            prog["u_instance_count"] = self._inst_np.shape[0]
            prog["u_light_count"] = self._light_np.shape[0]
            prog["u_max_bounces"] = self._max_bounces
            prog["u_accum_frame"] = self._accum_frame
            prog["u_frame"] = self._frame
            prog["u_accumulate"] = 1 if self._accumulate else 0
            prog["u_samples_per_pixel"] = self._samples_per_pixel
        except KeyError as e:
            Logger.warning(f"Raytracing uniform missing: {e}")
            return False

        sun_dir = Vec3(0, -0.3, -1)
        sky_color, sky_intensity = [1.0, 0.95, 0.85], 1.0
        for ent in scene.get_entities_with_component(Light):
            l = ent.get_component(Light)
            t = ent.transform
            if l and l.enabled and t and l.light_type == LightType.DIRECTIONAL:
                sun_dir = -t.forward
                sky_color, sky_intensity = Light.shader_radiance(l, t)
                break
        # Honour the Atmosphere component's sun-disc settings and intensity
        # multiplier on the procedural SkyEnv as well.
        sun_radius = 0.27
        sun_limb = 0.7
        sun_conv = 0.5
        try:
            from core.components.rendering.environment.atmosphere import Atmosphere
            atmos = next((a for a in Atmosphere._registry
                          if a.enabled and a.entity and a.entity.active), None)
            if atmos is not None:
                sky_intensity = sky_intensity * float(getattr(atmos, "_sun_intensity", 1.0))
                sun_radius = float(getattr(atmos, "_sun_angular_radius", 0.27))
                sun_limb = float(getattr(atmos, "_sun_limb_darkening", 0.7))
                sun_conv = float(getattr(atmos, "_sun_convergence", 0.5))
        except Exception:
            pass
        try:
            self._sky_env_prog["u_sun_direction"] = (sun_dir.x, sun_dir.y, sun_dir.z)
            self._sky_env_prog["u_sun_color"] = (sky_color[0], sky_color[1], sky_color[2])
            self._sky_env_prog["u_sun_intensity"] = sky_intensity
            self._sky_env_prog["u_sun_angular_radius"] = sun_radius
            self._sky_env_prog["u_sun_limb_darkening"] = sun_limb
            self._sky_env_prog["u_sun_convergence"] = sun_conv
        except KeyError as e:
            Logger.warning(f"SkyEnv uniform missing: {e}")
        self._sky_env_tex.bind_to_image(0, read=False, write=True)
        self._sky_env_prog.run(group_x=(256 + 7) // 8, group_y=(128 + 7) // 8, group_z=1)
        ctx.memory_barrier(moderngl.ALL_BARRIER_BITS)
        self._sky_env_tex.use(1)
        try:
            prog["u_sky_env"] = 1
        except KeyError:
            pass

        self._bvh_buf.bind_to_storage_buffer(0)
        self._vert_buf.bind_to_storage_buffer(1)
        self._idx_buf.bind_to_storage_buffer(2)
        self._mat_buf.bind_to_storage_buffer(3)
        self._inst_buf.bind_to_storage_buffer(4)
        self._light_buf.bind_to_storage_buffer(5)
        self._output_tex.bind_to_image(0, read=False, write=True)
        self._emissive_tex.bind_to_image(1, read=False, write=True)

        self._albedo_array_tex.use(2)
        try:
            prog["u_albedo_array"] = 2
            prog["u_albedo_tex_count"] = self._albedo_count
        except KeyError:
            pass

        if hasattr(renderer, '_raytracing_emissive_tex'):
            renderer._raytracing_emissive_tex = self._emissive_tex

        prog.run(group_x=(rw + 7) // 8, group_y=(rh + 7) // 8, group_z=1)
        ctx.memory_barrier(moderngl.ALL_BARRIER_BITS)

        self._rays_per_frame = rw * rh * self._samples_per_pixel * (self._max_bounces + 1)

        self._frame += 1
        if self._accumulate:
            self._accum_frame += 1
        return True

    def _blit_to_fbo(self, ctx: moderngl.Context, target_fbo: moderngl.Framebuffer, width: int, height: int):
        if not self._output_fbo or not self._fullscreen_prog:
            return
        old_fbo = ctx.fbo
        target_fbo.use()
        target_fbo.viewport = (0, 0, width, height)
        self._output_tex.use(0)
        self._fullscreen_prog["u_tex"].value = 0
        ctx.viewport = (0, 0, width, height)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._fullscreen_quad.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)
        if old_fbo is not None:
            old_fbo.use()

    def blit_to_screen(self, ctx: moderngl.Context, width: int, height: int):
        if not self._show_overlay or not self._output_fbo or not self._fullscreen_prog:
            return
        self._output_tex.use(0)
        self._fullscreen_prog["u_tex"].value = 0
        ctx.viewport = (0, 0, width, height)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._fullscreen_quad.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)

    def _get_or_load_albedo_layer(self, tex_path: str, mmgr) -> int:
        if tex_path in self._albedo_tex_map:
            return self._albedo_tex_map[tex_path]
        if self._albedo_count >= _MAX_ALBEDO_LAYERS:
            return -1
        abs_path = mmgr._resolve_tex_path(tex_path) if mmgr else tex_path
        if not abs_path or not os.path.exists(abs_path):
            self._albedo_tex_map[tex_path] = -1
            return -1
        try:
            from PIL import Image
            img = Image.open(abs_path).convert("RGBA")
        except Exception:
            self._albedo_tex_map[tex_path] = -1
            return -1
        w, h = img.size
        aw, ah = self._albedo_array_size
        if w > aw or h > ah:
            scale = min(aw / w, ah / h)
            w = max(1, int(w * scale))
            h = max(1, int(h * scale))
            img = img.resize((w, h), Image.LANCZOS)
        data = img.tobytes()
        x_off = (aw - w) // 2
        y_off = (ah - h) // 2
        try:
            layer = self._albedo_count
            self._albedo_array_tex.write(data, viewport=(x_off, y_off, layer, w, h, 1))
            self._albedo_count += 1
            self._albedo_tex_map[tex_path] = layer
            return layer
        except Exception:
            return -1

    def on_destroy(self):
        self._release_gl()

    def on_disable(self):
        self._release_gl()

    def _release_gl(self):
        for buf in [self._bvh_buf, self._vert_buf, self._idx_buf, self._mat_buf, self._inst_buf, self._light_buf]:
            if buf:
                buf.release()
        self._bvh_buf = self._vert_buf = self._idx_buf = self._mat_buf = self._inst_buf = self._light_buf = None
        if self._output_tex:
            self._output_tex.release()
            self._output_tex = None
        if self._output_fbo:
            self._output_fbo.release()
            self._output_fbo = None
        if self._program:
            self._program.release()
            self._program = None
        if self._fullscreen_prog:
            self._fullscreen_prog.release()
            self._fullscreen_prog = None
        if self._fullscreen_quad:
            self._fullscreen_quad.release()
            self._fullscreen_quad = None
        if self._sky_env_tex:
            self._sky_env_tex.release()
            self._sky_env_tex = None
        if self._sky_env_prog:
            self._sky_env_prog.release()
            self._sky_env_prog = None
        if self._albedo_array_tex:
            self._albedo_array_tex.release()
            self._albedo_array_tex = None
        self._albedo_tex_map.clear()
        self._albedo_count = 0
        self._mesh_geo_cache.clear()
        self._geo_signature = None
        self._geo_offsets = []
        self._geo_dirty = True
