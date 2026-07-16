# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import numpy as np
import moderngl
from typing import Optional, Tuple
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.components.rendering.postfx.graphics_effect import GraphicsEffect
from core.components.rendering.effects.voxel_cpu import compute_voxel_instances
from core.components.rendering.renderers.mesh_filter import MeshFilter
from core.engine.engine import Engine
from core.math.math3d import Vec3


VR_VERT = """
#version 330 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

VR_FRAG = """
#version 330 core

uniform sampler2D u_depth_tex;
uniform sampler3D u_grid;
uniform mat4 u_inv_view_proj;
uniform vec3 u_cam_pos;
uniform vec3 u_grid_min;
uniform vec3 u_grid_dims;
uniform float u_cell_size;
uniform vec3 u_vox_color;
uniform float u_vox_emission;
uniform float u_vox_rim;
uniform float u_amount;
uniform float u_opacity;
uniform vec3 u_light_dir;
uniform int u_max_steps;

in vec2 v_uv;
out vec4 frag_color;

bool solidAt(ivec3 c) {
    if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, ivec3(u_grid_dims))))
        return false;
    return texelFetch(u_grid, c, 0).r > 0.5;
}

vec3 cellNormal(ivec3 c) {
    vec3 n = vec3(0.0);
    if (!solidAt(c + ivec3(1, 0, 0))) n += vec3(1.0, 0.0, 0.0);
    if (!solidAt(c + ivec3(-1, 0, 0))) n += vec3(-1.0, 0.0, 0.0);
    if (!solidAt(c + ivec3(0, 1, 0))) n += vec3(0.0, 1.0, 0.0);
    if (!solidAt(c + ivec3(0, -1, 0))) n += vec3(0.0, -1.0, 0.0);
    if (!solidAt(c + ivec3(0, 0, 1))) n += vec3(0.0, 0.0, 1.0);
    if (!solidAt(c + ivec3(0, 0, -1))) n += vec3(0.0, 0.0, -1.0);
    return length(n) > 0.0001 ? normalize(n) : vec3(0.0, 1.0, 0.0);
}

void main() {
    ivec3 dims = ivec3(u_grid_dims);
    float cell = u_cell_size;
    vec3 gmin = u_grid_min;
    vec3 gmax = gmin + u_grid_dims * cell;

    float depth = texture(u_depth_tex, v_uv).r;
    vec2 ndc = v_uv * 2.0 - 1.0;

    vec4 nearH = u_inv_view_proj * vec4(ndc, -1.0, 1.0);
    vec4 farH  = u_inv_view_proj * vec4(ndc,  1.0, 1.0);
    vec3 nearP = nearH.xyz / nearH.w;
    vec3 farP  = farH.xyz / farH.w;

    vec3 ro = u_cam_pos;
    vec3 rd = normalize(farP - nearP);

    vec3 invD = 1.0 / rd;
    vec3 t0 = (gmin - ro) * invD;
    vec3 t1 = (gmax - ro) * invD;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tenter = max(max(tmin.x, tmin.y), tmin.z);
    float texit  = min(min(tmax.x, tmax.y), tmax.z);
    if (texit < 0.0 || tenter > texit) {
        frag_color = vec4(0.0);
        return;
    }

    float maxd;
    if (depth < 1.0) {
        vec4 sH = u_inv_view_proj * vec4(ndc, depth * 2.0 - 1.0, 1.0);
        vec3 sp = sH.xyz / sH.w;
        maxd = length(sp - ro);
    } else {
        maxd = 1e6;
    }

    float t = max(tenter, 0.0);
    float tstop = min(texit, maxd);
    if (tstop < t) {
        frag_color = vec4(0.0);
        return;
    }

    float stepSize = max(cell * 0.5, 1e-4);
    int steps = int(ceil((tstop - t) / stepSize));
    steps = min(steps, u_max_steps);

    vec3 col = vec3(0.0);
    float alpha = 0.0;
    for (int i = 0; i < steps; i++) {
        vec3 p = ro + rd * (t + float(i) * stepSize);
        vec3 rel = (p - gmin) / cell;
        ivec3 c = ivec3(floor(rel));
        if (all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, dims))) {
            if (texelFetch(u_grid, c, 0).r > 0.5) {
                vec3 n = cellNormal(c);
                vec3 L = normalize(u_light_dir);
                float diff = max(dot(n, L), 0.0);
                vec3 V = normalize(ro - p);
                float rim = pow(1.0 - max(dot(n, V), 0.0), max(u_vox_rim, 0.0001));
                col = u_vox_color * (0.25 + 0.75 * diff)
                    + u_vox_color * u_vox_emission
                    + vec3(rim) * u_vox_color;
                alpha = clamp(u_opacity * u_amount, 0.0, 1.0);
                break;
            }
        }
    }
    frag_color = vec4(col, alpha);
}
"""


@ComponentRegistry.register
class VoxelRaymarchEffect(GraphicsEffect):
    _allow_multiple = True
    _gizmo_icon_label = "VR"
    _intensity_prop = "amount"
    render_type = "additive"

    def __init__(self):
        super().__init__()
        self.amount: float = 1.0
        self.voxel_size: float = 0.3
        self.color: list[float] = [0.4, 1.0, 0.6]
        self.emission: float = 0.6
        self.rim: float = 2.5
        self.opacity: float = 1.0
        self.max_steps: int = 256
        self.light_dir: Vec3 = Vec3(0.4, 0.8, 0.3)
        self._grid: Optional[Tuple] = None
        self._scene_fbos: dict = {}
        self._ctx: Optional[moderngl.Context] = None
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("amount", "Amount", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("voxel_size", "Voxel Size", FieldType.FLOAT, step=0.01, decimals=3),
            InspectorField("color", "Voxel Color", FieldType.COLOR),
            InspectorField("emission", "Glow", FieldType.FLOAT, step=0.05, decimals=3),
            InspectorField("rim", "Rim Power", FieldType.FLOAT, step=0.05, decimals=3),
            InspectorField("opacity", "Opacity", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("max_steps", "Max Steps", FieldType.INT, min_val=32, max_val=1024, step=32),
            InspectorField("light_dir", "Light Dir", FieldType.VEC3, step=0.05, decimals=3),
        ]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "amount": self.amount,
            "voxel_size": self.voxel_size,
            "color": list(self.color),
            "emission": self.emission,
            "rim": self.rim,
            "opacity": self.opacity,
            "max_steps": self.max_steps,
            "light_dir": [self.light_dir.x, self.light_dir.y, self.light_dir.z],
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> "VoxelRaymarchEffect":
        fx = cls()
        fx.enabled = data.get("enabled", True)
        fx.amount = data.get("amount", 1.0)
        fx.voxel_size = data.get("voxel_size", 0.3)
        fx.color = list(data.get("color", [0.4, 1.0, 0.6]))
        fx.emission = data.get("emission", 0.6)
        fx.rim = data.get("rim", 2.5)
        fx.opacity = data.get("opacity", 1.0)
        fx.max_steps = int(data.get("max_steps", 256))
        fx.light_dir = Vec3(*data.get("light_dir", [0.4, 0.8, 0.3])[:3])
        fx._prog = None
        fx._vao = None
        fx._vbo = None
        fx._ibo = None
        return fx

    _res_cache: dict = {}

    def _ensure_resources(self, ctx: moderngl.Context):
        ctx_id = id(ctx)
        cached = self._res_cache.get(ctx_id)
        if cached is not None and cached.get('_prog') is not None:
            self._ctx = ctx
            self._prog = cached['_prog']
            self._vao = cached['_vao']
            self._vbo = cached['_vbo']
            self._ibo = cached['_ibo']
            return
        self._ctx = ctx
        self._prog = ctx.program(vertex_shader=VR_VERT, fragment_shader=VR_FRAG)
        verts = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        self._vbo = ctx.buffer(verts.tobytes())
        self._ibo = ctx.buffer(indices.tobytes())
        self._vao = ctx.vertex_array(self._prog, [(self._vbo, '2f', 'in_position')], self._ibo)
        self._res_cache[ctx_id] = {
            '_prog': self._prog,
            '_vao': self._vao,
            '_vbo': self._vbo,
            '_ibo': self._ibo,
        }
        if len(self._res_cache) > 4:
            oldest = next(iter(self._res_cache))
            for obj in self._res_cache[oldest].values():
                if obj is not None and hasattr(obj, 'release'):
                    try:
                        obj.release()
                    except Exception:
                        pass
            del self._res_cache[oldest]

    def _get_host_mesh(self):
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vr_debug.log"), "a") as _f:
                _f.write(f"[host] entity={self.entity is not None} eng={Engine.instance() is not None}\n")
        except Exception:
            pass
        if self.entity is None:
            return None, None
        mf = self.entity.get_component(MeshFilter)
        if mf is None:
            try:
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vr_debug.log"), "a") as _f:
                    _f.write(f"[host] NO MeshFilter on entity\n")
            except Exception:
                pass
            return None, None
        eng = Engine.instance()
        if eng is None:
            return None, None
        rndr = getattr(eng, '_renderer', None)
        if rndr is None:
            return None, None
        loader = getattr(rndr, '_mesh_loader', None)
        if loader is None:
            return None, None
        mesh = loader.get_or_create(mf.mesh_name or '', mf.mesh_path or '', 1.0, False, False)
        return mesh, self.entity.transform

    def _build_grid(self, ctx, world_verts, idx, cell):
        V = np.asarray(world_verts, dtype=np.float32).reshape(-1, 3)
        if V.shape[0] < 3:
            return None
        aabb_min = V.min(axis=0)
        aabb_max = V.max(axis=0)
        extent = aabb_max - aabb_min
        extent = np.maximum(extent, 1e-4)
        max_dim = 192
        need = np.ceil(extent / cell).astype(int)
        if np.any(need > max_dim):
            cell = float(np.max(extent) / max_dim)
        base = np.floor(aabb_min / cell).astype(int)
        dims = np.ceil((aabb_max - base * cell) / cell).astype(int)
        dims = np.maximum(dims, 1)

        centers = compute_voxel_instances(V, idx if idx is not None else None, None, cell, False, 0.0)
        if centers.shape[0] == 0:
            return None
        rel = np.floor(centers[:, :3] / cell + 1e-5).astype(int) - base
        mask = np.all((rel >= 0) & (rel < dims), axis=1)
        rel = rel[mask]
        if rel.shape[0] == 0:
            return None

        occ = np.zeros((int(dims[2]), int(dims[1]), int(dims[0])), dtype=np.uint8)
        occ[rel[:, 2], rel[:, 1], rel[:, 0]] = 255

        tex = ctx.texture3d((int(dims[0]), int(dims[1]), int(dims[2])), 1, occ.tobytes())
        return (tex, base * cell, dims.astype(int), float(cell))

    def render(self, ctx, scene_color_tex, scene_depth_tex,
               view_mat, proj_mat, cam_pos, viewport_w, viewport_h, **kwargs):
        _dbg = True
        _dlog = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vr_debug.log")
        if _dbg:
            try:
                with open(_dlog, "a") as _f:
                    _f.write(f"[render] called enabled={self.enabled} entity={self.entity is not None} active={getattr(self.entity,'active',None)} amount={self.amount} opacity={self.opacity} vsize={self.voxel_size}\n")
            except Exception:
                pass
        if not self.enabled or not self.entity or not self.entity.active:
            return
        if self.amount <= 0.001 or self.opacity <= 0.001 or self.voxel_size <= 1e-5:
            return

        self._ensure_resources(ctx)

        mesh, transform = self._get_host_mesh()
        if _dbg:
            try:
                with open(_dlog, "a") as _f:
                    _f.write(f"[render] mesh={mesh is not None} vcount={getattr(getattr(mesh,'vertices',None),'size',None)} tf={transform is not None}\n")
            except Exception:
                pass
        if mesh is None or getattr(mesh, 'vertices', None) is None:
            return
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        if verts.size < 9:
            return
        idx = getattr(mesh, 'indices', None)
        wm = np.asarray(transform.world_matrix._d, dtype=np.float32).reshape(4, 4)
        V = verts.reshape(-1, 3)
        ones = np.ones((V.shape[0], 1), dtype=np.float32)
        world_verts = (wm @ np.concatenate([V, ones], axis=1).T).T[:, :3]

        cell = float(self.voxel_size)
        key = (
            id(verts), verts.shape[0],
            round(cell, 4),
            tuple(np.round(wm, 3).reshape(16).tolist()),
        )
        if self._grid is None or self._grid[0] != key:
            old = self._grid
            built = self._build_grid(ctx, world_verts, idx, cell)
            if built is None:
                if old is not None:
                    try:
                        old[1].release()
                    except Exception:
                        pass
                    self._grid = None
                return
            if old is not None:
                try:
                    old[1].release()
                except Exception:
                    pass
            self._grid = (key, built[0], built[1], built[2], built[3])
            if _dbg:
                try:
                    with open(_dlog, "a") as _f:
                        _f.write(f"[render] grid built dims={built[2]} voxels={int(np.frombuffer(built[0].read(),dtype=np.uint8).sum())/255}\n")
                except Exception:
                    pass
        tex, gmin, dims, cell = self._grid[1], self._grid[2], self._grid[3], self._grid[4]

        vp = proj_mat * view_mat
        inv_vp = vp.inverted()
        inv_vp_f32 = inv_vp.to_f32()

        prog = self._prog
        old_mask = ctx.depth_mask
        prev_fbo = ctx.fbo
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.depth_mask = False
        try:
            scene_fbo = self._scene_fbos.get(id(scene_color_tex))
            if scene_fbo is None or scene_fbo.glo == 0 or id(ctx) != id(self._ctx):
                try:
                    if scene_fbo is not None:
                        scene_fbo.release()
                except Exception:
                    pass
                scene_fbo = ctx.framebuffer(scene_color_tex)
                self._scene_fbos[id(scene_color_tex)] = scene_fbo
            scene_fbo.use()
            scene_fbo.viewport = (0, 0, viewport_w, viewport_h)
            ctx.viewport = (0, 0, viewport_w, viewport_h)
            prog["u_inv_view_proj"].write(inv_vp_f32.tobytes())
            prog["u_cam_pos"].write(np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32).tobytes())
            prog["u_grid_min"].write(np.asarray(gmin, dtype=np.float32).tobytes())
            prog["u_grid_dims"].write(np.asarray(dims, dtype=np.float32).tobytes())
            prog["u_cell_size"] = float(cell)
            col = np.asarray(self.color, dtype=np.float32)
            if col.shape[0] < 3:
                col = np.array([0.4, 1.0, 0.6], dtype=np.float32)
            prog["u_vox_color"].write(col[:3].tobytes())
            prog["u_vox_emission"] = float(self.emission)
            prog["u_vox_rim"] = float(self.rim)
            prog["u_amount"] = float(self.amount)
            prog["u_opacity"] = float(self.opacity)
            prog["u_light_dir"].write(np.array([self.light_dir.x, self.light_dir.y, self.light_dir.z], dtype=np.float32).tobytes())
            prog["u_max_steps"] = int(self.max_steps)

            prog["u_grid"] = 0
            tex.use(0)
            prog["u_depth_tex"] = 1
            scene_depth_tex.use(1)

            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self._vao.render()
            ctx.disable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            if _dbg:
                try:
                    with open(_dlog, "a") as _f:
                        _f.write(f"[render] DREW into scene_fbo ok\n")
                except Exception:
                    pass
        except Exception as e:
            if _dbg:
                try:
                    with open(_dlog, "a") as _f:
                        _f.write(f"[render] EXCEPTION: {e}\n")
                except Exception:
                    pass
            from core.foundation.logger import Logger
            Logger.error(f"VoxelRaymarchEffect error: {e}")
        finally:
            if prev_fbo is not None:
                prev_fbo.use()
            ctx.depth_mask = old_mask

    def _release_gl(self):
        self._res_cache.pop(id(self._ctx), None)
        for fb in self._scene_fbos.values():
            try:
                if fb is not None:
                    fb.release()
            except Exception:
                pass
        self._scene_fbos.clear()
        for obj in (self._prog, self._vao, self._vbo, self._ibo):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        if self._grid is not None:
            try:
                self._grid[1].release()
            except Exception:
                pass
        self._ctx = None
        self._prog = None
        self._vao = None
        self._vbo = None
        self._ibo = None
        self._grid = None
