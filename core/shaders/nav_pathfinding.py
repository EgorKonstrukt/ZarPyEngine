# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import heapq
import uuid
import numpy as np
from typing import Optional
from core.math.math3d import Vec3
from core.spatial.octree import AABB

_COLLIDER_TYPES = ("BoxCollider", "SphereCollider", "CapsuleCollider", "MeshCollider")

_NEIGHBOR_OFFSETS_3D = [
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
]

_NEIGHBOR_OFFSETS_2D = [
    (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
]


class NavRect:
    __slots__ = ("gx", "gz", "gw", "gd", "ground_gy", "rid", "cx", "cz")
    def __init__(self, gx: int, gz: int, gw: int, gd: int, ground_gy: int, rid: int):
        self.gx = gx
        self.gz = gz
        self.gw = gw
        self.gd = gd
        self.ground_gy = ground_gy
        self.rid = rid
        self.cx = gx + gw * 0.5
        self.cz = gz + gd * 0.5

    def world_aabb(self, grid: "NavGrid") -> AABB:
        cs = grid.cell_size
        half = cs * 0.5
        c1 = grid.grid_to_world(self.gx, self.ground_gy, self.gz)
        c2 = grid.grid_to_world(self.gx + self.gw - 1, self.ground_gy, self.gz + self.gd - 1)
        return AABB(
            Vec3(c1.x - half, c1.y - half, c1.z - half),
            Vec3(c2.x + half, c2.y + half, c2.z + half),
        )

    def rect_aabb(self, grid: "NavGrid") -> AABB:
        cs = grid.cell_size
        c1 = grid.grid_to_world(self.gx, self.ground_gy, self.gz)
        c2 = grid.grid_to_world(self.gx + self.gw - 1, self.ground_gy, self.gz + self.gd - 1)
        return AABB(
            Vec3(c1.x, c1.y, c1.z),
            Vec3(c2.x, c2.y, c2.z),
        )


class NavGrid:
    def __init__(self, resolution: int = 48, world_size: float = 500.0):
        self.resolution = resolution
        self.world_size = world_size
        self.cell_size = world_size / resolution
        self.half_world = world_size * 0.5
        self._grid: np.ndarray = np.zeros((resolution, resolution, resolution), dtype=np.uint32)
        self._raw_grid: Optional[np.ndarray] = None
        self._dirty = True

    def cell_aabb(self, gx: int, gy: int, gz: int) -> AABB:
        c = self.grid_to_world(gx, gy, gz)
        h = self.cell_size * 0.5
        return AABB(c - Vec3(h, h, h), c + Vec3(h, h, h))

    def world_to_grid(self, pos: Vec3) -> tuple[int, int, int]:
        gx = int((pos.x + self.half_world) / self.cell_size)
        gy = int((pos.y + self.half_world) / self.cell_size)
        gz = int((pos.z + self.half_world) / self.cell_size)
        return max(0, min(self.resolution - 1, gx)), max(0, min(self.resolution - 1, gy)), max(0, min(self.resolution - 1, gz))

    def grid_to_world(self, gx: int, gy: int, gz: int) -> Vec3:
        return Vec3(
            -self.half_world + (gx + 0.5) * self.cell_size,
            -self.half_world + (gy + 0.5) * self.cell_size,
            -self.half_world + (gz + 0.5) * self.cell_size,
        )

    def _aabb_to_cell_range(self, aabb_min: Vec3, aabb_max: Vec3) -> tuple[int, int, int, int, int, int]:
        gx1, gy1, gz1 = self.world_to_grid(aabb_min)
        gx2, gy2, gz2 = self.world_to_grid(aabb_max)
        r = self.resolution
        return (
            max(0, min(r - 1, gx1)), max(0, min(r - 1, gy1)), max(0, min(r - 1, gz1)),
            max(0, min(r - 1, gx2)), max(0, min(r - 1, gy2)), max(0, min(r - 1, gz2)),
        )

    def mark_blocked(self, aabb_min: Vec3, aabb_max: Vec3):
        x1, y1, z1, x2, y2, z2 = self._aabb_to_cell_range(aabb_min, aabb_max)
        self._grid[x1:x2 + 1, y1:y2 + 1, z1:z2 + 1] = 1
        self._dirty = True

    def clear(self):
        self._grid.fill(0)
        self._dirty = True

    def is_blocked(self, gx: int, gy: int, gz: int) -> bool:
        if gx < 0 or gx >= self.resolution or gy < 0 or gy >= self.resolution or gz < 0 or gz >= self.resolution:
            return True
        return bool(self._grid[gx, gy, gz])

    def find_nearest_unblocked(self, gx: int, gy: int, gz: int, max_radius: int = 10) -> tuple[int, int, int]:
        if not self.is_blocked(gx, gy, gz):
            return (gx, gy, gz)
        r = self.resolution
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in (-radius, radius):
                        nx, ny, nz = gx + dx, gy + dy, gz + dz
                        if 0 <= nx < r and 0 <= ny < r and 0 <= nz < r and not self._grid[nx, ny, nz]:
                            return (nx, ny, nz)
            for dx in range(-radius, radius + 1):
                for dz in range(-radius + 1, radius):
                    for dy in (-radius, radius):
                        nx, ny, nz = gx + dx, gy + dy, gz + dz
                        if 0 <= nx < r and 0 <= ny < r and 0 <= nz < r and not self._grid[nx, ny, nz]:
                            return (nx, ny, nz)
            for dy in range(-radius + 1, radius):
                for dz in range(-radius + 1, radius):
                    for dx in (-radius, radius):
                        nx, ny, nz = gx + dx, gy + dy, gz + dz
                        if 0 <= nx < r and 0 <= ny < r and 0 <= nz < r and not self._grid[nx, ny, nz]:
                            return (nx, ny, nz)
        return (gx, gy, gz)

    def find_ground_cell(self, gx: int, gy: int, gz: int) -> tuple[int, int, int]:
        r = self.resolution
        if not self.is_blocked(gx, gy, gz):
            return (gx, gy, gz)
        highest_blocked = -1
        for y in range(gy, -1, -1):
            if self._grid[gx, y, gz]:
                highest_blocked = y
                break
        for walk_y in range(highest_blocked + 1, r):
            if not self._grid[gx, walk_y, gz]:
                return (gx, walk_y, gz)
        for y in range(gy + 1, r):
            if not self._grid[gx, y, gz]:
                return (gx, y, gz)
        return (gx, gy, gz)

    @staticmethod
    def _dilate_2d(walkable: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return walkable
        r = walkable.shape[0]
        try:
            import scipy.ndimage as ndi
            blocked = (walkable == 0).astype(np.uint8)
            dilated = ndi.binary_dilation(blocked, structure=np.ones((2 * radius + 1,) * 2)).astype(np.uint32)
            return (1 - dilated).astype(np.uint32)
        except ImportError:
            dilated = walkable.copy()
            indices = np.where(dilated == 0)
            for idx in range(len(indices[0])):
                gx, gz = indices[0][idx], indices[1][idx]
                x1 = max(0, gx - radius)
                x2 = min(r, gx + radius + 1)
                z1 = max(0, gz - radius)
                z2 = min(r, gz + radius + 1)
                dilated[x1:x2, z1:z2] = 0
            return dilated

    def build_ground_obstacle_grid(self, agent_height_cells: int, max_climb_cells: int,
                                    max_slope_deg: float = 45.0,
                                    start_gx: int = 0, start_gy: int = 0, start_gz: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
        r = self.resolution
        raw = self._raw_grid if self._raw_grid is not None else self._grid
        raw_bool = raw.astype(np.bool_)

        walk_gy = start_gy
        for y in range(max(0, start_gy - 1), -1, -1):
            if raw[start_gx, y, start_gz]:
                walk_gy = y + 1
                break
        walk_gy = max(0, min(r - 1, walk_gy))

        hc = agent_height_cells + max_climb_cells
        ground_gy = np.full((r, r), -1, dtype=np.int32)
        for gy in range(r - 1, -1, -1):
            blocked_at_gy = raw_bool[:, gy, :]
            if not np.any(blocked_at_gy):
                continue
            top = min(r, gy + 2 + hc)
            if top > gy + 1:
                clear_above = ~np.any(raw_bool[:, gy + 1:top, :], axis=1)
            else:
                clear_above = np.ones((r, r), dtype=np.bool_)
            valid = blocked_at_gy & clear_above
            mask = valid & (ground_gy == -1)
            ground_gy[mask] = gy

        has_ground = ground_gy >= 0

        walkable = np.ones((r, r), dtype=np.uint32)
        walkable[~has_ground] = 0

        if 0 < max_slope_deg < 90:
            max_hdiff = math.tan(math.radians(max_slope_deg))
            if max_hdiff > 0:
                gv = ground_gy[has_ground]
                if gv.size > 0 and gv.max() - gv.min() > max_hdiff:
                    changed = True
                    while changed:
                        changed = False
                        for gx in range(r):
                            for gz in range(r):
                                if not walkable[gx, gz]:
                                    continue
                                h0 = ground_gy[gx, gz]
                                if h0 < 0:
                                    continue
                                for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                                    nx, nz = gx + dx, gz + dz
                                    if 0 <= nx < r and 0 <= nz < r and walkable[nx, nz]:
                                        h1 = ground_gy[nx, nz]
                                        if h1 >= 0 and abs(h1 - h0) > max_hdiff:
                                            walkable[gx, gz] = 0
                                            changed = True
                                            break

        return walkable, ground_gy, walk_gy

    def save_raw(self):
        self._raw_grid = self._grid.copy()

    def dilate_obstacles(self, radius_cells: int):
        if radius_cells <= 0:
            return
        self.save_raw()
        try:
            import scipy.ndimage as ndi
            self._grid = ndi.binary_dilation(self._grid, structure=np.ones((2 * radius_cells + 1,) * 3)).astype(np.uint32)
        except ImportError:
            simple = self._grid.copy()
            r = self.resolution
            d = radius_cells
            indices = np.where(self._grid > 0)
            for idx in range(len(indices[0])):
                x, y, z = indices[0][idx], indices[1][idx], indices[2][idx]
                x1, x2 = max(0, x - d), min(r, x + d + 1)
                y1, y2 = max(0, y - d), min(r, y + d + 1)
                z1, z2 = max(0, z - d), min(r, z + d + 1)
                simple[x1:x2, y1:y2, z1:z2] = 1
            self._grid = simple
        self._dirty = True

    @property
    def flat_data(self) -> np.ndarray:
        return self._grid.ravel().astype(np.uint32)


class NavWorld:
    _instance: Optional[NavWorld] = None

    @classmethod
    def instance(cls) -> Optional[NavWorld]:
        return cls._instance

    def __init__(self, resolution: int = 48, world_size: float = 500.0):
        NavWorld._instance = self
        self._grid = NavGrid(resolution, world_size)
        self._scene = None
        self._last_grid_version = 0
        self._last_scene_version: int = -1
        self._pending_results: dict[str, Optional[list[Vec3]]] = {}
        self._path_cells: list[tuple[int, int, int]] = []
        self._raw_path_cells: list[tuple[int, int, int]] = []
        self._path_rects: list[NavRect] = []
        self._path_aabbs: list[tuple[AABB, int]] = []
        self._is_flying: bool = False

    def find_path_gpu_deferred(self, start_world: Vec3, end_world: Vec3,
                                agent_radius: float = 0.5, agent_height: float = 2.0,
                                flying: bool = False,
                                max_climb: float = 0.5, max_slope: float = 45.0,
                                agent_padding: Optional[float] = None) -> str:
        req_id = uuid.uuid4().hex[:12]
        path = self.find_path(start_world, end_world, agent_radius, agent_height, flying,
                              max_climb, max_slope, agent_padding)
        self._pending_results[req_id] = path if path else []
        return req_id

    def poll_result(self, req_id: str) -> Optional[list[Vec3]]:
        return self._pending_results.pop(req_id, None)

    def get_path_cells(self) -> list[tuple[int, int, int]]:
        return list(self._path_cells)

    def _build_path_aabbs(self):
        self._path_aabbs.clear()
        if self._path_rects:
            for rct in self._path_rects:
                self._path_aabbs.append((rct.world_aabb(self._grid), 0))
            for rct in self._path_rects:
                self._path_aabbs.append((rct.rect_aabb(self._grid), 1))
            return
        max_cells = 200
        n = min(len(self._raw_path_cells), max_cells)
        step = max(1, len(self._raw_path_cells) // max_cells) if len(self._raw_path_cells) > max_cells else 1
        for idx in range(0, len(self._raw_path_cells), step):
            gx, gy, gz = self._raw_path_cells[idx]
            self._path_aabbs.append((self._grid.cell_aabb(gx, gy, gz), 0))

    def get_path_aabbs(self) -> list[tuple[AABB, int]]:
        return self._path_aabbs

    def set_scene(self, scene):
        self._scene = scene
        self._rebuild_grid()

    def _rebuild_grid(self):
        if not self._scene:
            return
        sv = getattr(self._scene, '_render_version', -1)
        if sv == self._last_scene_version:
            return
        self._last_scene_version = sv
        self._grid.clear()
        entities = self._scene.get_all_entities()
        for entity in entities:
            for comp in entity.get_all_components():
                cname = type(comp).__name__
                if cname not in _COLLIDER_TYPES:
                    continue
                aabb = self._get_collider_world_aabb(comp)
                if aabb:
                    self._grid.mark_blocked(aabb.min, aabb.max)
        self._last_grid_version += 1

    def _get_collider_world_aabb(self, comp) -> Optional[AABB]:
        entity = comp._entity
        if not entity:
            return None
        tr = entity.transform
        if not tr:
            return None
        pos = tr.local_position
        rot = tr.local_rotation
        scale = tr.local_scale
        cname = type(comp).__name__

        if cname == "BoxCollider":
            hx = comp.size.x * scale.x * 0.5
            hy = comp.size.y * scale.y * 0.5
            hz = comp.size.z * scale.z * 0.5
            cl = Vec3(comp.center.x * scale.x, comp.center.y * scale.y, comp.center.z * scale.z)
            c = pos + rot.rotate_vec3(cl)
            local_corners = [
                Vec3(-hx, -hy, -hz), Vec3(hx, -hy, -hz), Vec3(-hx, hy, -hz), Vec3(hx, hy, -hz),
                Vec3(-hx, -hy, hz), Vec3(hx, -hy, hz), Vec3(-hx, hy, hz), Vec3(hx, hy, hz),
            ]
            ws = [c + rot.rotate_vec3(lc) for lc in local_corners]
            bmin = Vec3(min(v.x for v in ws), min(v.y for v in ws), min(v.z for v in ws))
            bmax = Vec3(max(v.x for v in ws), max(v.y for v in ws), max(v.z for v in ws))
            return AABB(bmin, bmax)
        elif cname == "SphereCollider":
            r = comp.radius * max(scale.x, scale.y, scale.z)
            c = pos + rot.rotate_vec3(Vec3(comp.center.x * scale.x, comp.center.y * scale.y, comp.center.z * scale.z))
            return AABB(c - Vec3(r, r, r), c + Vec3(r, r, r))
        elif cname == "CapsuleCollider":
            r = comp.radius
            hh = comp.height * 0.5
            cl = Vec3(comp.center.x * scale.x, comp.center.y * scale.y, comp.center.z * scale.z)
            c = pos + rot.rotate_vec3(cl)
            if comp.direction == 1:
                axis_local = Vec3(0, hh * scale.y, 0)
            elif comp.direction == 0:
                axis_local = Vec3(hh * scale.x, 0, 0)
            else:
                axis_local = Vec3(0, 0, hh * scale.z)
            rs = r * max(scale.x, scale.y, scale.z)
            axis_world = rot.rotate_vec3(axis_local)
            a = c + axis_world
            b = c - axis_world
            rv = Vec3(rs, rs, rs)
            aabb_a = AABB(a - rv, a + rv)
            aabb_b = AABB(b - rv, b + rv)
            return AABB(
                Vec3(min(aabb_a.min.x, aabb_b.min.x), min(aabb_a.min.y, aabb_b.min.y), min(aabb_a.min.z, aabb_b.min.z)),
                Vec3(max(aabb_a.max.x, aabb_b.max.x), max(aabb_a.max.y, aabb_b.max.y), max(aabb_a.max.z, aabb_b.max.z)),
            )
        elif cname == "MeshCollider":
            if hasattr(comp, 'mesh_path') and comp.mesh_path:
                try:
                    from core.engine.engine import Engine
                    eng = Engine.instance()
                    if eng and hasattr(eng, '_renderer') and eng._renderer:
                        mesh = eng._renderer._meshes.get(comp.mesh_path)
                        if mesh and hasattr(mesh, 'aabb_min') and hasattr(mesh, 'aabb_max'):
                            corners = np.array([
                                [mesh.aabb_min[i] for i in range(3)] + [1],
                                [mesh.aabb_max[0], mesh.aabb_min[1], mesh.aabb_min[2], 1],
                                [mesh.aabb_max[0], mesh.aabb_max[1], mesh.aabb_min[2], 1],
                                [mesh.aabb_min[0], mesh.aabb_max[1], mesh.aabb_min[2], 1],
                                [mesh.aabb_min[0], mesh.aabb_min[1], mesh.aabb_max[2], 1],
                                [mesh.aabb_max[0], mesh.aabb_min[1], mesh.aabb_max[2], 1],
                                [mesh.aabb_max[0], mesh.aabb_max[1], mesh.aabb_max[2], 1],
                                [mesh.aabb_min[0], mesh.aabb_max[1], mesh.aabb_max[2], 1],
                            ], dtype=np.float32)
                            wm = tr.world_matrix._d
                            pts = corners @ wm
                            bmin = pts[:, :3].min(axis=0)
                            bmax = pts[:, :3].max(axis=0)
                            return AABB(Vec3(float(bmin[0]), float(bmin[1]), float(bmin[2])),
                                        Vec3(float(bmax[0]), float(bmax[1]), float(bmax[2])))
                except Exception:
                    pass
            return None
        return None

    def rebuild_grid(self):
        self._rebuild_grid()

    def dilate_for_agent(self, radius: float):
        radius_cells = int(math.ceil(radius / self._grid.cell_size))
        self._grid.dilate_obstacles(radius_cells)

    def get_path_cell_aabbs(self) -> list[AABB]:
        result = []
        for rct in self._path_rects:
            result.append(rct.world_aabb(self._grid))
        return result

    def get_path_rects(self) -> list[NavRect]:
        return list(self._path_rects)

    def _heuristic(self, ax: int, ay: int, az: int, bx: int, by: int, bz: int) -> float:
        dx = abs(ax - bx)
        dy = abs(ay - by)
        dz = abs(az - bz)
        return float(dx + dy + dz)

    def _astar(self, sx: int, sy: int, sz: int, ex: int, ey: int, ez: int, flying: bool) -> Optional[list[tuple[int, int, int]]]:
        res = self._grid.resolution
        if flying:
            sx, sy, sz = self._grid.find_nearest_unblocked(sx, sy, sz)
            ex, ey, ez = self._grid.find_nearest_unblocked(ex, ey, ez)

        offsets = _NEIGHBOR_OFFSETS_3D if flying else _NEIGHBOR_OFFSETS_2D
        start = (sx, sy, sz)
        end = (ex, ey, ez)

        open_heap = [(0.0, 0, start)]
        g_scores = {start: 0}
        came_from = {start: None}

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            cx, cy, cz = current

            if current == end:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dx, dy, dz in offsets:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                neighbor = (nx, ny, nz)
                if nx < 0 or nx >= res or ny < 0 or ny >= res or nz < 0 or nz >= res:
                    continue
                if self._grid.is_blocked(nx, ny, nz):
                    continue

                ng = g_scores[current] + 1
                if neighbor not in g_scores or ng < g_scores[neighbor]:
                    g_scores[neighbor] = ng
                    f_score = ng + self._heuristic(nx, ny, nz, ex, ey, ez)
                    heapq.heappush(open_heap, (f_score, id(neighbor), neighbor))
                    came_from[neighbor] = current

        return None

    def _astar_2d(self, sx: int, sz: int, ex: int, ez: int,
                  walkable_2d: np.ndarray) -> Optional[list[tuple[int, int, int]]]:
        r = walkable_2d.shape[0]

        def snap(gx, gz):
            if 0 <= gx < r and 0 <= gz < r and walkable_2d[gx, gz]:
                return (gx, gz)
            for radius in range(1, min(r, 21)):
                for dx in range(-radius, radius + 1):
                    for dz in (-radius, radius):
                        nx, nz = gx + dx, gz + dz
                        if 0 <= nx < r and 0 <= nz < r and walkable_2d[nx, nz]:
                            return (nx, nz)
                for dz in range(-radius + 1, radius):
                    for dx in (-radius, radius):
                        nx, nz = gx + dx, gz + dz
                        if 0 <= nx < r and 0 <= nz < r and walkable_2d[nx, nz]:
                            return (nx, nz)
            return None

        start = snap(sx, sz)
        end = snap(ex, ez)
        if start is None or end is None:
            return None

        open_heap = [(0.0, 0, start)]
        g_scores = {start: 0}
        came_from = {start: None}

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            cx, cz = current

            if current == end:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, nz = cx + dx, cz + dz
                neighbor = (nx, nz)
                if nx < 0 or nx >= r or nz < 0 or nz >= r:
                    continue
                if not walkable_2d[nx, nz]:
                    continue

                ng = g_scores[current] + 1
                if neighbor not in g_scores or ng < g_scores[neighbor]:
                    g_scores[neighbor] = ng
                    f_score = ng + abs(nx - ex) + abs(nz - ez)
                    heapq.heappush(open_heap, (f_score, id(neighbor), neighbor))
                    came_from[neighbor] = current

        return None

    @staticmethod
    def _decompose_walkable(walkable: np.ndarray, ground_gy: np.ndarray) -> list[NavRect]:
        r = walkable.shape[0]
        assigned = np.zeros((r, r), dtype=np.bool_)
        rects = []
        rid = 0
        for gz in range(r):
            for gx in range(r):
                if not walkable[gx, gz] or assigned[gx, gz]:
                    continue
                cell_gy = ground_gy[gx, gz]
                if cell_gy < 0:
                    assigned[gx, gz] = True
                    continue
                gw = 1
                while gx + gw < r and walkable[gx + gw, gz] and not assigned[gx + gw, gz] and ground_gy[gx + gw, gz] == cell_gy:
                    gw += 1
                gd = 1
                while gz + gd < r:
                    row_ok = True
                    for dx in range(gw):
                        if not walkable[gx + dx, gz + gd] or assigned[gx + dx, gz + gd] or ground_gy[gx + dx, gz + gd] != cell_gy:
                            row_ok = False
                            break
                    if not row_ok:
                        break
                    gd += 1
                assigned[gx:gx + gw, gz:gz + gd] = True
                rects.append(NavRect(gx, gz, gw, gd, cell_gy, rid))
                rid += 1
        return rects

    @staticmethod
    def _build_rect_grid(rects: list[NavRect], r: int) -> np.ndarray:
        grid = np.full((r, r), -1, dtype=np.int32)
        for rect in rects:
            grid[rect.gx:rect.gx + rect.gw, rect.gz:rect.gz + rect.gd] = rect.rid
        return grid

    @staticmethod
    def _build_rect_adjacency(rects: list[NavRect], rect_grid: np.ndarray, r: int) -> dict[int, list[int]]:
        adj = {rect.rid: [] for rect in rects}
        for rect in rects:
            rid = rect.rid
            if rect.gx > 0:
                for dz in range(rect.gd):
                    nid = rect_grid[rect.gx - 1, rect.gz + dz]
                    if nid >= 0 and nid not in adj[rid]:
                        adj[rid].append(nid)
            right = rect.gx + rect.gw
            if right < r:
                for dz in range(rect.gd):
                    nid = rect_grid[right, rect.gz + dz]
                    if nid >= 0 and nid not in adj[rid]:
                        adj[rid].append(nid)
            if rect.gz > 0:
                for dx in range(rect.gw):
                    nid = rect_grid[rect.gx + dx, rect.gz - 1]
                    if nid >= 0 and nid not in adj[rid]:
                        adj[rid].append(nid)
            bottom = rect.gz + rect.gd
            if bottom < r:
                for dx in range(rect.gw):
                    nid = rect_grid[rect.gx + dx, bottom]
                    if nid >= 0 and nid not in adj[rid]:
                        adj[rid].append(nid)
        return adj

    @staticmethod
    def _astar_rects(start_rid: int, end_rid: int, rects: list[NavRect],
                      adj: dict[int, list[int]]) -> Optional[list[int]]:
        if start_rid == end_rid:
            return [start_rid]

        def heur(a: int, b: int) -> float:
            ra, rb = rects[a], rects[b]
            return math.sqrt((ra.cx - rb.cx) ** 2 + (ra.cz - rb.cz) ** 2)

        open_heap = [(0.0, 0, start_rid)]
        g_scores = {start_rid: 0.0}
        came_from = {start_rid: None}

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current == end_rid:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for nid in adj.get(current, []):
                dx = rects[current].cx - rects[nid].cx
                dz = rects[current].cz - rects[nid].cz
                ng = g_scores[current] + math.sqrt(dx * dx + dz * dz) + 0.001
                if nid not in g_scores or ng < g_scores[nid]:
                    g_scores[nid] = ng
                    f_score = ng + heur(nid, end_rid)
                    heapq.heappush(open_heap, (f_score, id(rects[nid]), nid))
                    came_from[nid] = current

        return None

    @staticmethod
    def _rect_path_to_waypoints(rect_path: list[int], rects: list[NavRect],
                                 ground_gy: np.ndarray, grid: NavGrid,
                                 start_world: Vec3, end_world: Vec3) -> list[Vec3]:
        if not rect_path:
            return [end_world]

        cs = grid.cell_size
        hw = grid.half_world
        r = ground_gy.shape[0] if ground_gy is not None else 0

        def get_gy(gx: int, gz: int) -> int:
            if ground_gy is not None and 0 <= gx < r and 0 <= gz < r and ground_gy[gx, gz] >= 0:
                return int(ground_gy[gx, gz])
            return 0

        def grid_xz(world_x: float, world_z: float) -> tuple[int, int]:
            return int(round((world_x + hw) / cs)), int(round((world_z + hw) / cs))

        def world_pt(gx: float, gz: float) -> Vec3:
            return Vec3(-hw + gx * cs, 0, -hw + gz * cs)

        def portal_endpoint(gx: float, gz: float) -> Vec3:
            wp = world_pt(gx, gz)
            ix, iz = grid_xz(wp.x, wp.z)
            gy = get_gy(ix, iz)
            wp.y = -hw + (gy + 1) * cs
            return wp

        def cross_xz(a: Vec3, b: Vec3, c: Vec3) -> float:
            return (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x)

        portals = []
        for i in range(len(rect_path) - 1):
            a = rects[rect_path[i]]
            b = rects[rect_path[i + 1]]
            ox1 = max(a.gx, b.gx)
            ox2 = min(a.gx + a.gw, b.gx + b.gw)
            oz1 = max(a.gz, b.gz)
            oz2 = min(a.gz + a.gd, b.gz + b.gd)
            if oz2 > oz1:
                portals.append((portal_endpoint(ox1, oz1), portal_endpoint(ox1, oz2)))
            else:
                portals.append((portal_endpoint(ox1, oz1), portal_endpoint(ox2, oz1)))

        if not portals:
            return [start_world, end_world]

        waypoints = [start_world]
        apex = start_world
        i = 0
        while i < len(portals):
            left, right = portals[i]
            if cross_xz(apex, left, right) > 0:
                left, right = right, left
            found = False
            for j in range(i + 1, len(portals)):
                pl, pr = portals[j]
                if cross_xz(apex, pl, pr) > 0:
                    pl, pr = pr, pl
                if cross_xz(apex, right, pl) < 0:
                    waypoints.append(right)
                    apex = right
                    i = j
                    found = True
                    break
                if cross_xz(apex, left, pr) > 0:
                    waypoints.append(left)
                    apex = left
                    i = j
                    found = True
                    break
                if cross_xz(apex, left, pl) > 0:
                    left = pl
                if cross_xz(apex, right, pr) < 0:
                    right = pr
            if not found:
                break
        waypoints.append(end_world)
        return waypoints

    def _navmesh_path_to_world(self, path_indices: list) -> list[Vec3]:
        res = self._grid.resolution
        waypoints = []
        for gx, gy, gz in path_indices:
            waypoints.append(self._grid.grid_to_world(gx, gy, gz))
        return waypoints

    def find_path(self, start_world: Vec3, end_world: Vec3,
                  agent_radius: float = 0.5, agent_height: float = 2.0,
                  flying: bool = False,
                  max_climb: float = 0.5, max_slope: float = 45.0,
                  agent_padding: Optional[float] = None) -> Optional[list[Vec3]]:
        grid = self._grid
        sx, sy, sz = grid.world_to_grid(start_world)
        ex, ey, ez = grid.world_to_grid(end_world)
        self._path_cells = []
        self._raw_path_cells = []
        self._path_aabbs.clear()
        self._path_rects = []
        self._is_flying = flying

        padding = agent_padding if agent_padding is not None else agent_radius

        if flying:
            sx, sy, sz = grid.find_nearest_unblocked(sx, sy, sz)
            ex, ey, ez = grid.find_nearest_unblocked(ex, ey, ez)
            path_cells = self._astar(sx, sy, sz, ex, ey, ez, True)
            if not path_cells:
                return None
            self._raw_path_cells = path_cells
            self._path_cells = path_cells
            waypoints = self._navmesh_path_to_world(path_cells)
            self._build_path_aabbs()
            return waypoints

        hc = int(math.ceil(agent_height / grid.cell_size))
        cc = int(math.ceil(max_climb / grid.cell_size))
        radius_cells = int(math.ceil(padding / grid.cell_size))
        walkable_2d, ground_gy, walk_gy = grid.build_ground_obstacle_grid(hc, cc, max_slope, sx, sy, sz)
        walkable_2d = NavGrid._dilate_2d(walkable_2d, radius_cells)
        walkable_2d = self._flood_fill_ground(walkable_2d, ground_gy, sx, sz, max_climb, grid.cell_size, grid.half_world)

        if self._has_line_of_sight_2d(walkable_2d, sx, sz, ex, ez):
            self._path_cells = [(sx, sy, sz), (ex, ey, ez)]
            self._raw_path_cells = [(sx, sy, sz), (ex, ey, ez)]
            return [start_world, end_world]

        rects = self._decompose_walkable(walkable_2d, ground_gy)
        if not rects:
            return None
        rect_grid = self._build_rect_grid(rects, grid.resolution)
        adj = self._build_rect_adjacency(rects, rect_grid, grid.resolution)

        def find_rect(gx: int, gz: int) -> Optional[int]:
            if 0 <= gx < grid.resolution and 0 <= gz < grid.resolution:
                rid = rect_grid[gx, gz]
                if rid >= 0:
                    return int(rid)
            for radius in range(1, min(grid.resolution, 21)):
                for dx in range(-radius, radius + 1):
                    for dz in (-radius, radius):
                        ngx, ngz = gx + dx, gz + dz
                        if 0 <= ngx < grid.resolution and 0 <= ngz < grid.resolution:
                            rid = rect_grid[ngx, ngz]
                            if rid >= 0:
                                return int(rid)
                for dz in range(-radius + 1, radius):
                    for dx in (-radius, radius):
                        ngx, ngz = gx + dx, gz + dz
                        if 0 <= ngx < grid.resolution and 0 <= ngz < grid.resolution:
                            rid = rect_grid[ngx, ngz]
                            if rid >= 0:
                                return int(rid)
            return None

        start_rid = find_rect(sx, sz)
        end_rid = find_rect(ex, ez)
        if start_rid is None or end_rid is None:
            return None

        rect_path = self._astar_rects(start_rid, end_rid, rects, adj)
        if not rect_path:
            return None

        self._path_rects = [rects[rid] for rid in rect_path]
        self._build_path_aabbs()

        waypoints = self._rect_path_to_waypoints(rect_path, rects, ground_gy, grid, start_world, end_world)
        waypoints = self._simplify_path_los(waypoints, walkable_2d, grid)

        for rid in rect_path:
            rct = rects[rid]
            self._path_cells.append((rct.gx, max(0, rct.ground_gy), rct.gz))

        return waypoints

    @staticmethod
    def _has_line_of_sight_2d(walkable: np.ndarray, sx: int, sz: int, ex: int, ez: int) -> bool:
        res = walkable.shape[0]
        if not (0 <= sx < res and 0 <= sz < res and 0 <= ex < res and 0 <= ez < res):
            return False
        dx = ex - sx
        dz = ez - sz
        if dx == 0 and dz == 0:
            return bool(walkable[sx, sz])
        step_x = 1 if dx > 0 else -1
        step_z = 1 if dz > 0 else -1
        t_delta_x = 1.0 / abs(dx) if dx != 0 else float('inf')
        t_delta_z = 1.0 / abs(dz) if dz != 0 else float('inf')
        t_max_x = 0.5 * t_delta_x if dx != 0 else float('inf')
        t_max_z = 0.5 * t_delta_z if dz != 0 else float('inf')
        gx, gz = sx, sz
        while True:
            if not (0 <= gx < res and 0 <= gz < res) or not walkable[gx, gz]:
                return False
            if gx == ex and gz == ez:
                return True
            if t_max_x < t_max_z:
                gx += step_x
                t_max_x += t_delta_x
            elif t_max_z < t_max_x:
                gz += step_z
                t_max_z += t_delta_z
            else:
                gx += step_x
                gz += step_z
                t_max_x += t_delta_x
                t_max_z += t_delta_z
                if not (0 <= gx < res and 0 <= gz < res) or not walkable[gx, gz]:
                    return False

    @staticmethod
    def _flood_fill_ground(walkable: np.ndarray, ground_gy: np.ndarray,
                            start_gx: int, start_gz: int,
                            max_climb: float, cell_size: float, half_world: float) -> np.ndarray:
        r = ground_gy.shape[0]
        if not (0 <= start_gx < r and 0 <= start_gz < r) or ground_gy[start_gx, start_gz] < 0:
            return np.zeros((r, r), dtype=np.uint32)
        ground_y = -half_world + (ground_gy + 1) * cell_size
        result = np.zeros((r, r), dtype=np.uint32)
        from collections import deque
        q = deque()
        q.append((start_gx, start_gz))
        result[start_gx, start_gz] = 1
        while q:
            gx, gz = q.popleft()
            h0 = ground_y[gx, gz]
            for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, nz = gx + dx, gz + dz
                if 0 <= nx < r and 0 <= nz < r and result[nx, nz] == 0:
                    if ground_gy[nx, nz] >= 0 and walkable[nx, nz]:
                        h1 = ground_y[nx, nz]
                        if h1 > h0 and h1 - h0 > max_climb:
                            continue
                        result[nx, nz] = 1
                        q.append((nx, nz))
        return result

    def _simplify_path_los(self, waypoints: list[Vec3], walkable: np.ndarray, grid: "NavGrid") -> list[Vec3]:
        if len(waypoints) <= 2:
            return waypoints

        def to_grid(p: Vec3) -> tuple[int, int]:
            gx, _, gz = grid.world_to_grid(p)
            return gx, gz

        res = [waypoints[0]]
        for i in range(1, len(waypoints)):
            lx, lz = to_grid(res[-1])
            nx, nz = to_grid(waypoints[i])
            if not self._has_line_of_sight_2d(walkable, lx, lz, nx, nz):
                prev = waypoints[i - 1]
                if prev != res[-1]:
                    res.append(prev)
        res.append(waypoints[-1])
        return res

    def find_path_gpu(self, start_world: Vec3, end_world: Vec3,
                      agent_radius: float = 0.5, agent_height: float = 2.0,
                      flying: bool = False,
                      max_climb: float = 0.5, max_slope: float = 45.0,
                      agent_padding: Optional[float] = None) -> Optional[list[Vec3]]:
        return self.find_path(start_world, end_world, agent_radius, agent_height,
                              flying, max_climb, max_slope, agent_padding)

    @staticmethod
    def _backtrack(dist: np.ndarray, res: int, start_idx: int, end_idx: int) -> Optional[list[int]]:
        if dist[end_idx] == 0xFFFFFFFF:
            return None
        path = [end_idx]
        current = end_idx
        while current != start_idx:
            gz = current // (res * res)
            gy = (current - gz * res * res) // res
            gx = current - gz * res * res - gy * res
            best_n = -1
            best_d = dist[current]
            for dx, dy, dz in _NEIGHBOR_OFFSETS_3D:
                nx, ny, nz = gx + dx, gy + dy, gz + dz
                if nx < 0 or nx >= res or ny < 0 or ny >= res or nz < 0 or nz >= res:
                    continue
                nidx = nx + ny * res + nz * res * res
                d = dist[nidx]
                if d < best_d:
                    best_d = d
                    best_n = nidx
            if best_n < 0:
                if current == end_idx:
                    return [start_idx, end_idx]
                return None
            path.append(best_n)
            current = best_n
        path.reverse()
        return path

    def release(self):
        NavWorld._instance = None
