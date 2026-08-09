# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import numpy as np
from core.maths.math3d import Vec3
from editor.viewport.projection import screen_to_ray, world_to_screen

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
        for key, m in meshes.items():
            if key == mesh_path or key.startswith(mesh_path + "|"):
                return m
    if mesh_name and mesh_name != "cube":
        for key, m in meshes.items():
            if key.startswith(mesh_name + "|"):
                return m
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
    wm_inv = np.linalg.inv(wm)
    local_o = ro @ wm_inv
    local_d = rd @ wm_inv
    if mesh.indices is not None and len(mesh.indices) > 0:
        from core.spatial.bvh import get_mesh_bvh
        bvh = get_mesh_bvh(mesh.vertices, mesh.indices)
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
    for entity in all_ents:
        if entity.id in candidate_ids or not entity.active:
            continue
        d = _test_entity_pick(entity, ro, rd, ray_origin, ray_dir)
        if d > 0 and d < best_dist:
            best_dist = d
            best_entity = entity
    if best_entity is not None:
        return best_entity
    for entity in all_ents:
        if entity.id in candidate_ids or not entity.active:
            continue
        from core.components.transform import Transform
        t = entity.transform
        if not t:
            continue
        from core.components.rendering.renderers.mesh_filter import MeshFilter
        from core.components.rendering.renderers.mesh_renderer import MeshRenderer
        from core.components.physics.mesh_collider import MeshCollider
        from core.components.physics.box_collider import BoxCollider
        from core.components.physics.sphere_collider import SphereCollider
        from core.components.rendering.renderers.sprite_renderer import SpriteRenderer
        mf = entity.get_component(MeshFilter)
        mr = entity.get_component(MeshRenderer)
        mc = entity.get_component(MeshCollider)
        bc = entity.get_component(BoxCollider)
        sc = entity.get_component(SphereCollider)
        sr = entity.get_component(SpriteRenderer)
        if mf or mr or mc or bc or sc or sr:
            continue
        half = 0.5
        wp = t.position
        d = _ray_aabb_min(ray_origin.x, ray_origin.y, ray_origin.z,
                          ray_dir.x, ray_dir.y, ray_dir.z,
                          wp.x - half, wp.y - half, wp.z - half,
                          wp.x + half, wp.y + half, wp.z + half)
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
    for entity in all_ents:
        if entity.id in candidate_ids or not entity.active:
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
    result = []
    from core.components.transform import Transform
    for entity in scene.get_all_entities():
        if not entity.active:
            continue
        t = entity.transform
        if not t:
            continue
        saabb = _screen_aabb_of(vp, entity)
        if saabb is None:
            sp = world_to_screen(vp, t.position)
            if sp and rx <= sp[0] <= rx + rw and ry <= sp[1] <= ry + rh:
                result.append(entity)
            continue
        ex1, ey1, ex2, ey2 = saabb
        if ex1 <= rx + rw and ex2 >= rx and ey1 <= ry + rh and ey2 >= ry:
            result.append(entity)
    return result
