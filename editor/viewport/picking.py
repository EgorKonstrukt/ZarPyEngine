# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import numpy as np
from core.maths.math3d import Vec3
from editor.viewport.projection import screen_to_ray, world_to_screen

try:
    from core import _raycast as _raycast_cy
except ImportError:
    _raycast_cy = None

_font_atlas_cache: dict[tuple[str, int], "FontAtlas"] = {}


def _ray_aabb_min(ox: float, oy: float, oz: float,
                  dx: float, dy: float, dz: float,
                  bmin_x: float, bmin_y: float, bmin_z: float,
                  bmax_x: float, bmax_y: float, bmax_z: float) -> float:
    tmin = -1e30
    tmax = 1e30
    if abs(dx) > 1e-30:
        t1 = (bmin_x - ox) / dx
        t2 = (bmax_x - ox) / dx
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
    elif ox < bmin_x or ox > bmax_x:
        return -1.0
    if abs(dy) > 1e-30:
        t1 = (bmin_y - oy) / dy
        t2 = (bmax_y - oy) / dy
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
    elif oy < bmin_y or oy > bmax_y:
        return -1.0
    if abs(dz) > 1e-30:
        t1 = (bmin_z - oz) / dz
        t2 = (bmax_z - oz) / dz
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
    elif oz < bmin_z or oz > bmax_z:
        return -1.0
    if tmin > tmax:
        return -1.0
    return tmin if tmin > 0.0 else (tmax if tmax > 0.0 else -1.0)


def _world_aabb_of(entity, only_expanded: bool = False) -> tuple | None:
    from core.components.transform import Transform
    from core.components.rendering.renderers.mesh_filter import MeshFilter
    from core.components.rendering.renderers.mesh_renderer import MeshRenderer
    from core.components.rendering.renderers.text_renderer import TextRenderer
    from core.components.physics.box_collider import BoxCollider
    from core.components.physics.sphere_collider import SphereCollider
    from core.components.rendering.skeleton.armature import Bone
    if entity.get_component(Bone):
        return None
    t = entity.transform
    if not t:
        return None
    wp = t.position
    bmin = np.array([wp.x, wp.y, wp.z])
    bmax = np.array([wp.x, wp.y, wp.z])
    expanded = False
    mf = entity.get_component(MeshFilter)
    mr = entity.get_component(MeshRenderer)
    if mf and mr and mr.enabled:
        mesh_name = mf.mesh_name or "cube"
        mesh = _get_mesh_for(entity, mesh_name, mf.mesh_path)
        if mesh is not None:
            wm = t.world_matrix._d
            ax, ay, az = mesh.aabb_min
            bx, by, bz = mesh.aabb_max
            corners = np.array([
                [ax, ay, az, 1], [bx, ay, az, 1], [bx, by, az, 1], [ax, by, az, 1],
                [ax, ay, bz, 1], [bx, ay, bz, 1], [bx, by, bz, 1], [ax, by, bz, 1],
            ], dtype=np.float32)
            pts = corners @ wm
            np.minimum(bmin, pts[:, :3].min(axis=0), out=bmin)
            np.maximum(bmax, pts[:, :3].max(axis=0), out=bmax)
            expanded = True
    from core.components.rendering.renderers.skinned_mesh_renderer import SkinnedMeshRenderer
    smr = entity.get_component(SkinnedMeshRenderer)
    if smr and smr.enabled:
        mesh_name = smr.mesh_name or "cube"
        mesh = _get_mesh_for(entity, mesh_name, smr.mesh_path)
        if mesh is not None:
            wm = t.world_matrix._d
            ax, ay, az = mesh.aabb_min
            bx, by, bz = mesh.aabb_max
            corners = np.array([
                [ax, ay, az, 1], [bx, ay, az, 1], [bx, by, az, 1], [ax, by, az, 1],
                [ax, ay, bz, 1], [bx, ay, bz, 1], [bx, by, bz, 1], [ax, by, bz, 1],
            ], dtype=np.float32)
            pts = corners @ wm
            np.minimum(bmin, pts[:, :3].min(axis=0), out=bmin)
            np.maximum(bmax, pts[:, :3].max(axis=0), out=bmax)
            expanded = True
    from core.components.rendering.renderers.sprite_renderer import SpriteRenderer
    sr = entity.get_component(SpriteRenderer)
    if sr and sr.enabled and sr.texture_path:
        wm = t.world_matrix._d
        corners = np.array([
            [-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [0.5, 0.5, 0, 1], [-0.5, 0.5, 0, 1],
        ], dtype=np.float32)
        pts = corners @ wm
        np.minimum(bmin, pts[:, :3].min(axis=0), out=bmin)
        np.maximum(bmax, pts[:, :3].max(axis=0), out=bmax)
        expanded = True
    from core.components.rendering.renderers.video_renderer import VideoRenderer
    vr = entity.get_component(VideoRenderer)
    if vr and vr.enabled and vr.video_path:
        wm = t.world_matrix._d
        corners = np.array([
            [-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [0.5, 0.5, 0, 1], [-0.5, 0.5, 0, 1],
        ], dtype=np.float32)
        pts = corners @ wm
        np.minimum(bmin, pts[:, :3].min(axis=0), out=bmin)
        np.maximum(bmax, pts[:, :3].max(axis=0), out=bmax)
        expanded = True
    from core.components.rendering.renderers.text_renderer import TextRenderer
    from core.assets.font_atlas import FontAtlas
    from core.assets.font_atlas import get_default_font_path as get_def_font
    tr_comp = entity.get_component(TextRenderer)
    if tr_comp and tr_comp.enabled and tr_comp.text:
        fp = tr_comp.font_path or get_def_font()
        base_size = getattr(tr_comp, "atlas_resolution", 128)
        ak = (fp, base_size)
        atlas = _font_atlas_cache.get(ak)
        if atlas is None and fp:
            try:
                atlas = FontAtlas(fp, base_size)
                _font_atlas_cache[ak] = atlas
            except Exception:
                pass
        if atlas is not None:
            inv_lh = 1.0 / atlas.line_height if atlas.line_height > 0 else 1.0
            scale = float(tr_comp.font_size) * inv_lh * 0.01
            lines = tr_comp.text.split("\n")
            total_w_raw = 0.0
            for line in lines:
                lw = 0.0
                for c in line:
                    g = atlas.get_glyph(c)
                    if g:
                        lw += g["advance"]
                if lw > total_w_raw:
                    total_w_raw = lw
            total_w = total_w_raw * scale
            line_h = atlas.line_height * scale * tr_comp.line_spacing
            total_h = (len(lines) - 1) * line_h + atlas.line_height * scale
            hw = total_w * 0.5
            hh = total_h * 0.5
            wm = t.world_matrix._d
            corners = np.array([
                [-hw, -hh, 0, 1], [hw, -hh, 0, 1], [hw, hh, 0, 1], [-hw, hh, 0, 1],
            ], dtype=np.float32)
            pts = corners @ wm
            np.minimum(bmin, pts[:, :3].min(axis=0), out=bmin)
            np.maximum(bmax, pts[:, :3].max(axis=0), out=bmax)
            expanded = True
    bc = entity.get_component(BoxCollider)
    if bc:
        hx, hy, hz = bc.size.x * 0.5, bc.size.y * 0.5, bc.size.z * 0.5
        wm = t.world_matrix._d
        corners = np.array([
            [-hx, -hy, -hz, 1], [hx, -hy, -hz, 1], [hx, hy, -hz, 1], [-hx, hy, -hz, 1],
            [-hx, -hy, hz, 1], [hx, -hy, hz, 1], [hx, hy, hz, 1], [-hx, hy, hz, 1],
        ], dtype=np.float32)
        pts = corners @ wm
        np.minimum(bmin, pts[:, :3].min(axis=0), out=bmin)
        np.maximum(bmax, pts[:, :3].max(axis=0), out=bmax)
        expanded = True
    sc = entity.get_component(SphereCollider)
    if sc:
        r = sc.radius
        wm = t.world_matrix._d
        center = (np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) @ wm)[:3]
        np.minimum(bmin, center - r, out=bmin)
        np.maximum(bmax, center + r, out=bmax)
        expanded = True
    for child in entity.children:
        child_box = _world_aabb_of(child)
        if child_box:
            np.minimum(bmin, child_box[0], out=bmin)
            np.maximum(bmax, child_box[1], out=bmax)
            expanded = True
    if not expanded:
        if only_expanded:
            return None
        s = t.local_scale
        half = max(max(abs(s.x), abs(s.y), abs(s.z)) * 0.5, 0.5)
        bmin = np.array([wp.x - half, wp.y - half, wp.z - half])
        bmax = np.array([wp.x + half, wp.y + half, wp.z + half])
    return (bmin, bmax)


_mesh_lookup_cache: dict[tuple[int, int], dict] = {}
_MESH_LOOKUP_SENTINEL = object()


def _resolve_mesh_key(meshes, prefix: str):
    for key, m in meshes.items():
        if key == prefix or key.startswith(prefix + "|"):
            return key
    return _MESH_LOOKUP_SENTINEL


def _get_mesh_for(entity, mesh_name: str, mesh_path: str):
    from core.engine.engine import Engine
    engine = Engine.instance()
    if not engine:
        return None
    renderer = getattr(engine, '_renderer', None)
    if renderer is None:
        vp = getattr(engine, 'viewport', None)
        if vp:
            renderer = getattr(vp, '_renderer', None)
    if renderer is None:
        return None
    meshes = renderer._meshes
    if not meshes:
        return None
    mesh = meshes.get(mesh_name)
    if mesh is not None:
        return mesh
    if mesh_path:
        mesh = meshes.get(mesh_path)
        if mesh is not None:
            return mesh
    sig = (id(meshes), len(meshes))
    cache = _mesh_lookup_cache.get(sig)
    if cache is None:
        cache = {}
        _mesh_lookup_cache[sig] = cache
    if mesh_path:
        key = cache.get(("p", mesh_path))
        if key is None:
            key = _resolve_mesh_key(meshes, mesh_path)
            cache[("p", mesh_path)] = key
        if key is not _MESH_LOOKUP_SENTINEL:
            return meshes[key]
    if mesh_name and mesh_name != "cube":
        key = cache.get(("n", mesh_name))
        if key is None:
            key = _resolve_mesh_key(meshes, mesh_name)
            cache[("n", mesh_name)] = key
        if key is not _MESH_LOOKUP_SENTINEL:
            return meshes[key]
    return meshes.get("cube")


def _world_aabb_from_mesh(mesh, wm):
    ax, ay, az = mesh.aabb_min
    bx, by, bz = mesh.aabb_max
    corners = np.array([
        [ax, ay, az, 1], [bx, ay, az, 1], [bx, by, az, 1], [ax, by, az, 1],
        [ax, ay, bz, 1], [bx, ay, bz, 1], [bx, by, bz, 1], [ax, by, bz, 1],
    ], dtype=np.float32)
    pts = corners @ wm
    return pts[:, :3].min(axis=0), pts[:, :3].max(axis=0)


def _test_mesh_hit(wm, ro, rd, mesh):
    bmin, bmax = _world_aabb_from_mesh(mesh, wm)
    d = _ray_aabb_min(ro[0], ro[1], ro[2], rd[0], rd[1], rd[2],
                      bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2])
    if d < 0:
        return -1.0
    wm_inv = _raycast_cy.inv_affine4(wm) if _raycast_cy is not None else np.linalg.inv(wm)
    local_o = ro @ wm_inv
    local_d = rd @ wm_inv
    if mesh.indices is not None and len(mesh.indices) > 0:
        from core.spatial.bvh import get_mesh_bvh, get_mesh_bvh_sync
        bvh = get_mesh_bvh(mesh.vertices, mesh.indices)
        if bvh and bvh.nodes:
            return bvh.intersect(local_o[0], local_o[1], local_o[2],
                                 local_d[0], local_d[1], local_d[2],
                                 mesh.vertices, mesh.indices)
        if _raycast_cy is not None:
            verts = np.ascontiguousarray(mesh.vertices)
            if verts.dtype == np.float32 and verts.ndim == 2:
                indices = mesh.indices
                if indices.dtype != np.uint32:
                    indices = indices.astype(np.uint32)
                else:
                    indices = np.ascontiguousarray(indices)
                return _raycast_cy.triangles_intersect(
                    verts.reshape(-1), indices,
                    float(local_o[0]), float(local_o[1]), float(local_o[2]),
                    float(local_d[0]), float(local_d[1]), float(local_d[2]))
        bvh = get_mesh_bvh_sync(mesh.vertices, mesh.indices)
        if bvh and bvh.nodes:
            return bvh.intersect(local_o[0], local_o[1], local_o[2],
                                 local_d[0], local_d[1], local_d[2],
                                 mesh.vertices, mesh.indices)
    return _ray_aabb_min(local_o[0], local_o[1], local_o[2],
                         local_d[0], local_d[1], local_d[2],
                         mesh.aabb_min[0], mesh.aabb_min[1], mesh.aabb_min[2],
                         mesh.aabb_max[0], mesh.aabb_max[1], mesh.aabb_max[2])


def _test_entity_pick(entity, ro, rd, ray_origin, ray_dir):
    from core.components.transform import Transform
    from core.components.rendering.renderers.mesh_filter import MeshFilter
    from core.components.rendering.renderers.mesh_renderer import MeshRenderer
    from core.components.physics.mesh_collider import MeshCollider
    t = entity.transform
    if not t:
        return -1.0
    mf = entity.get_component(MeshFilter)
    mr = entity.get_component(MeshRenderer)
    mesh = None
    has_mesh = False
    if mf:
        mesh_name = mf.mesh_name or "cube"
        mesh = _get_mesh_for(entity, mesh_name, mf.mesh_path)
        has_mesh = bool(mesh and mr and mr.enabled)
    if has_mesh:
        wm = t.world_matrix._d
        d = _test_mesh_hit(wm, ro, rd, mesh)
        return d if d > 0 else -1.0
    from core.components.rendering.renderers.skinned_mesh_renderer import SkinnedMeshRenderer
    smr = entity.get_component(SkinnedMeshRenderer)
    if smr and smr.enabled:
        mesh_name = smr.mesh_name or "cube"
        mesh = _get_mesh_for(entity, mesh_name, smr.mesh_path)
        if mesh is not None:
            wm = t.world_matrix._d
            d = _test_mesh_hit(wm, ro, rd, mesh)
            return d if d > 0 else -1.0
    mc = entity.get_component(MeshCollider)
    if mc:
        mf2 = entity.get_component(MeshFilter)
        if mf2:
            mesh2 = _get_mesh_for(entity, mf2.mesh_name or "cube", mf2.mesh_path)
            if mesh2 is not None and mesh2.indices is not None and len(mesh2.indices) > 0:
                wm = t.world_matrix._d
                d = _test_mesh_hit(wm, ro, rd, mesh2)
                return d if d > 0 else -1.0
    box = _world_aabb_of(entity, only_expanded=True)
    if box is not None:
        d = _ray_aabb_min(ray_origin.x, ray_origin.y, ray_origin.z,
                          ray_dir.x, ray_dir.y, ray_dir.z,
                          box[0][0], box[0][1], box[0][2],
                          box[1][0], box[1][1], box[1][2])
        return d if d > 0 else -1.0
    wp = t.position
    half = 0.5
    d = _ray_aabb_min(ray_origin.x, ray_origin.y, ray_origin.z,
                      ray_dir.x, ray_dir.y, ray_dir.z,
                      wp.x - half, wp.y - half, wp.z - half,
                      wp.x + half, wp.y + half, wp.z + half)
    return d if d > 0 else -1.0


def _mesh_of_entity(entity):
    from core.components.rendering.renderers.mesh_filter import MeshFilter
    from core.components.rendering.renderers.mesh_renderer import MeshRenderer
    from core.components.rendering.renderers.skinned_mesh_renderer import SkinnedMeshRenderer
    from core.components.physics.mesh_collider import MeshCollider
    mf = entity.get_component(MeshFilter)
    mr = entity.get_component(MeshRenderer)
    mesh = None
    if mf:
        m = _get_mesh_for(entity, mf.mesh_name or "cube", mf.mesh_path)
        if m is not None and mr and mr.enabled:
            mesh = m
    if mesh is None:
        smr = entity.get_component(SkinnedMeshRenderer)
        if smr and smr.enabled:
            m = _get_mesh_for(entity, smr.mesh_name or "cube", smr.mesh_path)
            if m is not None:
                mesh = m
    if mesh is None:
        mc = entity.get_component(MeshCollider)
        if mc:
            mf2 = entity.get_component(MeshFilter)
            if mf2:
                m = _get_mesh_for(entity, mf2.mesh_name or "cube", mf2.mesh_path)
                if m is not None and m.indices is not None and len(m.indices) > 0:
                    mesh = m
    return mesh


def _batch_mesh_aabb_miss(entities, ro, rd) -> set:
    if _raycast_cy is None:
        return set()
    idx = []
    bmins = []
    bmaxs = []
    wms = []
    for i, entity in enumerate(entities):
        mesh = _mesh_of_entity(entity)
        if mesh is None:
            continue
        t = entity.transform
        if not t:
            continue
        idx.append(i)
        bmins.append(mesh.aabb_min)
        bmaxs.append(mesh.aabb_max)
        wms.append(t.world_matrix._d)
    if not idx:
        return set()
    wmn, wmx = _raycast_cy.world_aabbs(
        np.array(bmins, dtype=np.float64),
        np.array(bmaxs, dtype=np.float64),
        np.array(wms, dtype=np.float64),
    )
    hits = _raycast_cy.ray_aabbs(ro[0], ro[1], ro[2], rd[0], rd[1], rd[2], wmn, wmx)
    return {idx[i] for i in range(len(idx)) if not hits[i]}


def pick_entity(vp, sx: int, sy: int):
    scene = vp._engine.scene
    if not scene:
        return None
    ray_origin, ray_dir = screen_to_ray(vp, sx, sy)
    ro = np.array([ray_origin.x, ray_origin.y, ray_origin.z, 1.0], dtype=np.float64)
    rd = np.array([ray_dir.x, ray_dir.y, ray_dir.z, 0.0], dtype=np.float64)
    candidates = scene.spatial_raycast(ray_origin, ray_dir, 1000.0)
    candidate_ids = {eid for eid, _ in candidates}
    best_entity = None
    best_dist = float("inf")
    for eid, _ in candidates:
        entity = scene.get_entity(eid)
        if not entity or not entity.active:
            continue
        d = _test_entity_pick(entity, ro, rd, ray_origin, ray_dir)
        if d > 0 and d < best_dist:
            best_dist = d
            best_entity = entity
    all_ents = scene.get_all_entities()
    active = [e for e in all_ents if e.active]
    miss = _batch_mesh_aabb_miss(active, ro, rd)
    for i, entity in enumerate(active):
        if entity.id in candidate_ids or i in miss:
            continue
        d = _test_entity_pick(entity, ro, rd, ray_origin, ray_dir)
        if d > 0 and d < best_dist:
            best_dist = d
            best_entity = entity
    return best_entity


def pick_entity_hit(vp, sx: int, sy: int):
    """Returns (entity, hit_world_pos) or (None, None)."""
    scene = vp._engine.scene
    if not scene:
        return None, None
    ray_origin, ray_dir = screen_to_ray(vp, sx, sy)
    ro = np.array([ray_origin.x, ray_origin.y, ray_origin.z, 1.0], dtype=np.float64)
    rd = np.array([ray_dir.x, ray_dir.y, ray_dir.z, 0.0], dtype=np.float64)
    candidates = scene.spatial_raycast(ray_origin, ray_dir, 1000.0)
    candidate_ids = {eid for eid, _ in candidates}
    best_entity = None
    best_dist = float("inf")
    for eid, _ in candidates:
        entity = scene.get_entity(eid)
        if not entity or not entity.active:
            continue
        d = _test_entity_pick(entity, ro, rd, ray_origin, ray_dir)
        if d > 0 and d < best_dist:
            best_dist = d
            best_entity = entity
    all_ents = scene.get_all_entities()
    active = [e for e in all_ents if e.active]
    miss = _batch_mesh_aabb_miss(active, ro, rd)
    for i, entity in enumerate(active):
        if entity.id in candidate_ids or i in miss:
            continue
        d = _test_entity_pick(entity, ro, rd, ray_origin, ray_dir)
        if d > 0 and d < best_dist:
            best_dist = d
            best_entity = entity
    if best_entity is None:
        return None, None
    hit_pos = ray_origin + ray_dir * best_dist
    return best_entity, hit_pos


def _screen_aabb_of(vp, entity) -> tuple | None:
    box = _world_aabb_of(entity)
    if box is None:
        return None
    corners = [
        (box[0][0], box[0][1], box[0][2]),
        (box[1][0], box[0][1], box[0][2]),
        (box[0][0], box[1][1], box[0][2]),
        (box[0][0], box[0][1], box[1][2]),
        (box[1][0], box[1][1], box[0][2]),
        (box[1][0], box[0][1], box[1][2]),
        (box[0][0], box[1][1], box[1][2]),
        (box[1][0], box[1][1], box[1][2]),
    ]
    sx_min = sy_min = float('inf')
    sx_max = sy_max = float('-inf')
    for c in corners:
        sp = world_to_screen(vp, Vec3(*c))
        if sp is None:
            continue
        sx, sy = sp
        sx_min = min(sx_min, sx)
        sy_min = min(sy_min, sy)
        sx_max = max(sx_max, sx)
        sy_max = max(sy_max, sy)
    if sx_min == float('inf'):
        return None
    return (sx_min, sy_min, sx_max, sy_max)


def pick_entities_in_rect(vp, rx: int, ry: int, rw: int, rh: int) -> list:
    scene = vp._engine.scene
    if not scene:
        return []
    entities = []
    boxes = []
    for entity in scene.get_all_entities():
        if not entity.active:
            continue
        t = entity.transform
        if not t:
            continue
        entities.append(entity)
        boxes.append(_world_aabb_of(entity))
    if not entities:
        return []
    w = float(vp.width())
    h = float(vp.height())
    aspect = w / max(1.0, h)
    vp_mat = (vp._cam.get_view_matrix() * vp._cam.get_projection_matrix(aspect))._d

    corner_rows = []
    corner_off = []
    pos_rows = []
    off = 0
    for i, (entity, box) in enumerate(zip(entities, boxes)):
        t = entity.transform
        p = t.position
        pos_rows.append((p.x, p.y, p.z))
        if box is None:
            corner_off.append(-1)
            continue
        (ax, ay, az), (bx, by, bz) = box
        corner_rows.extend([
            (ax, ay, az), (bx, ay, az), (ax, by, az), (ax, ay, bz),
            (bx, by, az), (bx, ay, bz), (ax, by, bz), (bx, by, bz),
        ])
        corner_off.append(off)
        off += 8

    if _raycast_cy is not None:
        pts = np.array(corner_rows + pos_rows, dtype=np.float64)
        sx, sy, ok = _raycast_cy.project_points(pts, vp_mat, w, h)
    else:
        pts = np.array(corner_rows + pos_rows, dtype=np.float64)
        sx = np.empty(len(pts), dtype=np.float64)
        sy = np.empty(len(pts), dtype=np.float64)
        ok = np.zeros(len(pts), dtype=np.uint8)
        for i in range(len(pts)):
            sp = world_to_screen(vp, Vec3(pts[i][0], pts[i][1], pts[i][2]))
            if sp is not None:
                sx[i] = sp[0]
                sy[i] = sp[1]
                ok[i] = 1

    n_corner = off
    result = []
    for i, entity in enumerate(entities):
        c_off = corner_off[i]
        if c_off < 0:
            pi = n_corner + i
            if ok[pi] and rx <= sx[pi] <= rx + rw and ry <= sy[pi] <= ry + rh:
                result.append(entity)
            continue
        sx_min = float('inf')
        sy_min = float('inf')
        sx_max = float('-inf')
        sy_max = float('-inf')
        any_ok = False
        for k in range(8):
            pi = c_off + k
            if not ok[pi]:
                continue
            any_ok = True
            px = sx[pi]
            py = sy[pi]
            if px < sx_min:
                sx_min = px
            if px > sx_max:
                sx_max = px
            if py < sy_min:
                sy_min = py
            if py > sy_max:
                sy_max = py
        if not any_ok:
            pi = n_corner + i
            if ok[pi] and rx <= sx[pi] <= rx + rw and ry <= sy[pi] <= ry + rh:
                result.append(entity)
            continue
        if sx_min <= rx + rw and sx_max >= rx and sy_min <= ry + rh and sy_max >= ry:
            result.append(entity)
    return result
