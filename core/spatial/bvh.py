# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future

import numpy as np
import xxhash

from core.ecs.pool import bvh as _get_bvh_pool
from core.ecs.pool import bvh_parallel as _get_bvh_parallel_pool
from core.foundation.progress import task_complete, task_start, task_update
from core.foundation.progress import get_progress_queue

_BVH_LOG = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")),
                        "zarin_bvh_log.txt")
try:
    with open(_BVH_LOG, "w", encoding="utf-8") as _lf:
        _lf.write(f"=== BVH LOG RESET ===\n")
except OSError:
    pass


def _bvh_log(msg: str) -> None:
    try:
        with open(_BVH_LOG, "a", encoding="utf-8") as _f:
            _f.write(f"{time.monotonic():.3f} {msg}\n")
    except OSError:
        pass

_bvh_progress_queue: object | None = None

_LEAF_SIZE = 32
_MAX_DEPTH = 48
_SAH_BINS = 12
_SAH_TRAV = 1.0
_SAH_HIT = 1.0
_PAR_THRESH = 50000
_BVH_CACHE_VERSION = 2


def _spread_bits_vec(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint64)
    v = (v | (v << 16)) & np.uint64(0x0000FFFF0000FFFF)
    v = (v | (v << 8)) & np.uint64(0x00FF00FF00FF00FF)
    v = (v | (v << 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v | (v << 2)) & np.uint64(0x3333333333333333)
    v = (v | (v << 1)) & np.uint64(0x5555555555555555)
    return v.astype(np.uint32)


def _get_bvh_build_mode() -> str:
    try:
        from core.config.config import get_global_config
        return get_global_config().get("rendering.bvh_build_mode", "fast")
    except Exception:
        return "fast"


class _BuildCtx:
    __slots__ = ('last_frac', 'last_t', 'lock', 'n_verts', 'node_count', 'nodes',
                 'progress_done', 'task_id', 'total_tris', 'tri_indices', 'tri_offset')

    def __init__(self, max_nodes, max_tris):
        self.nodes = np.empty((max_nodes, 8), dtype=np.float32)
        self.tri_indices = np.empty(max_tris, dtype=np.uint32)
        self.node_count = 0
        self.tri_offset = 0
        self.lock = threading.Lock()
        self.progress_done = 0
        self.total_tris = 0
        self.n_verts = 0
        self.task_id = ""
        self.last_frac = 0.0
        self.last_t = 0.0


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{int(v)}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{int(v)}k" if v == int(v) else f"{v:.1f}k"
    return str(n)


def _pq_put(msg):
    q = _bvh_progress_queue
    if q is not None:
        try:
            q.put_nowait(msg)
        except Exception:
            pass


def _bvh_task_update(task_id, fraction=None, detail=None, total=None, units=None):
    if _bvh_progress_queue is not None:
        kw = {}
        if fraction is not None:
            kw["fraction"] = fraction
        if detail is not None:
            kw["detail"] = detail
        if total is not None:
            kw["total"] = total
        if units is not None:
            kw["units"] = units
        _pq_put(("update", (task_id,), kw))
    else:
        task_update(task_id, fraction=fraction, detail=detail, total=total, units=units)


def _report_build_progress(ctx):
    if ctx.total_tris <= 0:
        return
    frac = ctx.progress_done / ctx.total_tris
    now = time.monotonic()
    if frac - ctx.last_frac >= 0.02 or now - ctx.last_t >= 0.25:
        ctx.last_frac = frac
        ctx.last_t = now
        detail = (f"{_fmt_count(ctx.progress_done)} / {_fmt_count(ctx.total_tris)} tris"
                  f" · {_fmt_count(ctx.n_verts)} verts")
        _bvh_task_update(ctx.task_id, frac, detail=detail)


def _surface_area(bmin, bmax):
    d = bmax - bmin
    return 2.0 * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2])


def _ray_aabb_min(ox, oy, oz, dx, dy, dz, bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z):
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


def _moller_trumbore(ox, oy, oz, dx, dy, dz, ax, ay, az, bx, by, bz, cx, cy, cz):
    e1x = bx - ax
    e1y = by - ay
    e1z = bz - az
    e2x = cx - ax
    e2y = cy - ay
    e2z = cz - az
    px = dy * e2z - dz * e2y
    py = dz * e2x - dx * e2z
    pz = dx * e2y - dy * e2x
    det = e1x * px + e1y * py + e1z * pz
    if abs(det) < 1e-12:
        return -1.0
    inv_det = 1.0 / det
    tx = ox - ax
    ty = oy - ay
    tz = oz - az
    u = (tx * px + ty * py + tz * pz) * inv_det
    if u < 0.0 or u > 1.0:
        return -1.0
    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv_det
    if v < 0.0 or u + v > 1.0:
        return -1.0
    t = (e2x * qx + e2y * qy + e2z * qz) * inv_det
    return t if t > 0 else -1.0


def _find_morton_split(sorted_morton: np.ndarray, start: int, end: int) -> int:
    common = int(sorted_morton[start] ^ sorted_morton[end - 1])
    if common == 0:
        return (start + end) // 2
    split_bit = 63 - (common.bit_length() - 1)
    split_mask = np.uint64(1 << split_bit)
    masks = sorted_morton[start:end] & split_mask
    indices = np.flatnonzero(masks)
    if len(indices) > 0:
        return start + int(indices[0])
    return (start + end) // 2


class BVH:
    __slots__ = ('_nodes', '_tri_indices', '_tri_bmin', '_tri_bmax', '_centroids',
                 '_vertices', '_indices', '_vert_key', '_idx_key', '_task_id',
                 '_cached_depths', '_node_views', '_tri_v0', '_tri_v1', '_tri_v2',
                 '_root_idx')

    def __init__(self, vertices: np.ndarray, indices: np.ndarray,
                 task_id: str | None = None):
        self._vertices = vertices
        self._indices = indices
        self._cached_depths: list[int] | None = None
        self._node_views: list | None = None
        self._vert_key = id(vertices)
        self._idx_key = id(indices)
        self._task_id = task_id or str(self._vert_key)

        n_tris = len(indices) // 3
        if n_tris == 0:
            self._nodes = np.empty((0, 8), dtype=np.float32)
            self._tri_indices = np.array([], dtype=np.uint32)
            self._tri_bmin = np.empty((0, 3), dtype=np.float32)
            self._tri_bmax = np.empty((0, 3), dtype=np.float32)
            self._centroids = np.empty((0, 3), dtype=np.float32)
            self._tri_v0 = np.array([], dtype=np.uint32)
            self._tri_v1 = np.array([], dtype=np.uint32)
            self._tri_v2 = np.array([], dtype=np.uint32)
            self._root_idx = 0
            return

        verts3 = vertices.reshape(-1, 3)
        tri_i = indices.reshape(n_tris, 3).astype(np.intp)
        v0 = verts3[tri_i[:, 0]]
        v1 = verts3[tri_i[:, 1]]
        v2 = verts3[tri_i[:, 2]]
        self._tri_bmin = np.minimum(np.minimum(v0, v1), v2)
        self._tri_bmax = np.maximum(np.maximum(v0, v1), v2)
        self._centroids = (v0 + v1 + v2) / 3.0

        tri_order = np.arange(n_tris, dtype=np.intp)
        if _get_bvh_build_mode() == "fast":
            self._build_lbvh(tri_order, n_tris)
        else:
            self._build(tri_order, n_tris)

        tri_u32 = np.asarray(indices, dtype=np.uint32)
        slot_idx = self._tri_indices.astype(np.intp) * 3
        self._tri_v0 = tri_u32[slot_idx]
        self._tri_v1 = tri_u32[slot_idx + 1]
        self._tri_v2 = tri_u32[slot_idx + 2]

    @classmethod
    def from_cache(cls, nodes: np.ndarray, tri_indices: np.ndarray,
                   vertices: np.ndarray, indices: np.ndarray,
                   root_idx: int = -1) -> BVH:
        bvh = object.__new__(cls)
        bvh._nodes = nodes
        bvh._tri_indices = tri_indices
        bvh._vertices = vertices
        bvh._indices = indices
        bvh._vert_key = id(vertices)
        bvh._idx_key = id(indices)
        bvh._task_id = ""
        bvh._cached_depths = None
        bvh._node_views = None
        bvh._tri_bmin = np.empty((0, 3), dtype=np.float32)
        bvh._tri_bmax = np.empty((0, 3), dtype=np.float32)
        bvh._centroids = np.empty((0, 3), dtype=np.float32)
        bvh._root_idx = root_idx if root_idx >= 0 else len(nodes) - 1
        tri_u32 = np.asarray(indices, dtype=np.uint32)
        slot_idx = tri_indices.astype(np.intp) * 3
        bvh._tri_v0 = tri_u32[slot_idx]
        bvh._tri_v1 = tri_u32[slot_idx + 1]
        bvh._tri_v2 = tri_u32[slot_idx + 2]
        bvh._root_idx = len(nodes) - 1
        return bvh

    def _build(self, tri_order, n_tris):
        import sys
        sys.setrecursionlimit(1000000)

        ctx = _BuildCtx(n_tris * 2 + 1, n_tris)
        ctx.total_tris = n_tris
        ctx.n_verts = len(self._vertices.reshape(-1, 3))
        ctx.task_id = self._task_id
        ctx.last_t = time.monotonic()
        _bvh_task_update(ctx.task_id, 0.0,
                         detail=f"0 / {_fmt_count(n_tris)} tris · {_fmt_count(ctx.n_verts)} verts",
                         total=n_tris, units="tris")

        def _alloc_node(ctx):
            with ctx.lock:
                ni = ctx.node_count
                ctx.node_count += 1
            return ni

        def _make_leaf(ctx, tris):
            ni = _alloc_node(ctx)
            bmin = self._tri_bmin[tris].min(axis=0)
            bmax = self._tri_bmax[tris].max(axis=0)
            ctx.nodes[ni, 0:3] = bmin
            ctx.nodes[ni, 3:6] = bmax
            with ctx.lock:
                tri_start = ctx.tri_offset
                ctx.tri_offset += len(tris)
                ctx.progress_done += len(tris)
            ctx.nodes[ni, 6] = float(tri_start)
            ctx.nodes[ni, 7] = -float(len(tris)) - 1.0
            ctx.tri_indices[tri_start:tri_start + len(tris)] = tris.astype(np.uint32)
            _report_build_progress(ctx)
            return ni

        def _sah_build(tris, depth=0):
            n = len(tris)
            bmin = self._tri_bmin[tris].min(axis=0)
            bmax = self._tri_bmax[tris].max(axis=0)

            if n <= _LEAF_SIZE or depth >= _MAX_DEPTH:
                return _make_leaf(ctx, tris)

            parent_sa = _surface_area(bmin, bmax)
            if parent_sa < 1e-12:
                return _make_leaf(ctx, tris)

            if _USE_CYTHON_BVH:
                best_axis, best_split, lmask = _sah_cython_split(
                    np.asarray(tris, dtype=np.intp),
                    self._tri_bmin, self._tri_bmax, self._centroids,
                    n, parent_sa, _SAH_BINS
                )
                if best_axis < 0:
                    return _make_leaf(ctx, tris)
                left_mask = lmask
            else:
                c_cent = self._centroids[tris]
                c_bmin = c_cent.min(axis=0)
                c_bmax = c_cent.max(axis=0)
                c_range = c_bmax - c_bmin

                best_axis = -1
                best_split = 0.0
                best_cost = float('inf')

                for axis in range(3):
                    if c_range[axis] < 1e-12:
                        continue
                    bins = _SAH_BINS
                    scale = bins / c_range[axis]
                    cent_vals = c_cent[:, axis]
                    bin_idx = np.floor((cent_vals - c_bmin[axis]) * scale).astype(np.intp)
                    bin_idx = np.clip(bin_idx, 0, bins - 1)

                    bin_cnt = np.bincount(bin_idx, minlength=bins).astype(np.intp)
                    sort_idx = np.argsort(bin_idx, kind='stable')
                    sorted_bin = bin_idx[sort_idx]
                    sorted_tris = tris[sort_idx]
                    tri_bm = self._tri_bmin[sorted_tris]
                    tri_bx = self._tri_bmax[sorted_tris]
                    boundaries = np.searchsorted(sorted_bin, np.arange(bins))
                    bin_bmin = np.full((bins, 3), 1e30, dtype=np.float32)
                    bin_bmax = np.full((bins, 3), -1e30, dtype=np.float32)
                    for bi in range(bins):
                        s = boundaries[bi]
                        e = boundaries[bi + 1] if bi + 1 < bins else n
                        if s < e:
                            bin_bmin[bi] = tri_bm[s:e].min(axis=0)
                            bin_bmax[bi] = tri_bx[s:e].max(axis=0)

                    left_cs = np.cumsum(bin_cnt[:-1])
                    right_cs = n - left_cs
                    valid = (left_cs > 0) & (right_cs > 0)
                    pmin_acc = np.minimum.accumulate(bin_bmin[:-1], axis=0)
                    pmax_acc = np.maximum.accumulate(bin_bmax[:-1], axis=0)
                    d_l = pmax_acc - pmin_acc
                    left_sa_arr = 2.0 * (d_l[:, 0] * d_l[:, 1] + d_l[:, 0] * d_l[:, 2] + d_l[:, 1] * d_l[:, 2])
                    left_cost = left_sa_arr * left_cs
                    smin_rev = np.minimum.accumulate(bin_bmin[:0:-1], axis=0)[::-1]
                    smax_rev = np.maximum.accumulate(bin_bmax[:0:-1], axis=0)[::-1]
                    d_r = smax_rev - smin_rev
                    right_sa_arr = 2.0 * (d_r[:, 0] * d_r[:, 1] + d_r[:, 0] * d_r[:, 2] + d_r[:, 1] * d_r[:, 2])
                    right_cost = right_sa_arr * right_cs
                    costs = np.full(bins - 1, np.inf, dtype=np.float64)
                    costs[valid] = _SAH_TRAV + (left_cost[valid] + right_cost[valid]) / parent_sa
                    i_best = np.argmin(costs)
                    if costs[i_best] < best_cost:
                        best_cost = costs[i_best]
                        best_axis = axis
                        best_split = c_bmin[axis] + (i_best + 1) * c_range[axis] / bins

                leaf_cost = n * _SAH_HIT
                if best_axis < 0 or best_cost >= leaf_cost:
                    return _make_leaf(ctx, tris)

                axis = best_axis
                left_mask = c_cent[:, axis] < best_split
            left_n = left_mask.sum()
            if left_n == 0 or left_n == n:
                if _USE_CYTHON_BVH:
                    cent_axis = self._centroids[tris, best_axis]
                else:
                    cent_axis = c_cent[:, axis]
                mid = n // 2
                order = np.argsort(cent_axis)
                left_tris = tris[order[:mid]]
                right_tris = tris[order[mid:]]
            else:
                left_tris = tris[left_mask]
                right_tris = tris[~left_mask]

            use_parallel = depth == 0 and n >= _PAR_THRESH
            if use_parallel:
                pool = _get_bvh_parallel_pool()
                lf = pool.submit(_sah_build, left_tris, depth + 1)
                rf = pool.submit(_sah_build, right_tris, depth + 1)
                lc = lf.result()
                rc = rf.result()
                ni = _alloc_node(ctx)
                ctx.nodes[ni, 0:3] = bmin
                ctx.nodes[ni, 3:6] = bmax
                ctx.nodes[ni, 6] = float(lc)
                ctx.nodes[ni, 7] = float(rc)
            else:
                lc = _sah_build(left_tris, depth + 1)
                rc = _sah_build(right_tris, depth + 1)
                ni = _alloc_node(ctx)
                ctx.nodes[ni, 0:3] = bmin
                ctx.nodes[ni, 3:6] = bmax
                ctx.nodes[ni, 6] = float(lc)
                ctx.nodes[ni, 7] = float(rc)

            return ni

        _sah_build(tri_order)
        self._nodes = ctx.nodes[:ctx.node_count]
        self._tri_indices = ctx.tri_indices[:ctx.tri_offset]
        self._root_idx = ctx.node_count - 1

    def _build_lbvh(self, tri_order, n_tris):
        verts3 = self._vertices.reshape(-1, 3)
        tri_i = self._indices.reshape(n_tris, 3).astype(np.intp)
        v0 = verts3[tri_i[:, 0]]
        v1 = verts3[tri_i[:, 1]]
        v2 = verts3[tri_i[:, 2]]

        scene_min = np.minimum(np.minimum(v0, v1), v2).min(axis=0)
        scene_max = np.maximum(np.maximum(v0, v1), v2).max(axis=0)
        scene_ext = np.maximum(scene_max - scene_min, 1e-10)
        centroids = (v0 + v1 + v2) * (1.0 / 3.0)
        norm = (centroids - scene_min) / scene_ext
        bits = 10
        scale = float((1 << bits) - 1)
        cx = np.floor(norm[:, 0] * scale).astype(np.uint32)
        cy = np.floor(norm[:, 1] * scale).astype(np.uint32)
        cz = np.floor(norm[:, 2] * scale).astype(np.uint32)
        morton = _spread_bits_vec(cx) | (_spread_bits_vec(cy) << 1) | (_spread_bits_vec(cz) << 2)
        morton_u64 = morton.astype(np.uint64)

        order = np.argsort(morton_u64, kind='stable')
        sorted_tris = tri_order[order]
        sorted_morton = morton_u64[order]

        if n_tris == 1:
            ctx = _BuildCtx(1, 1)
            ctx.total_tris = 1
            ctx.n_verts = len(self._vertices.reshape(-1, 3))
            ctx.task_id = self._task_id
            ctx.tri_indices[0] = sorted_tris[0]
            ctx.progress_done = 1
            _report_build_progress(ctx)
            bmin, bmax = self._tri_bmin[0], self._tri_bmax[0]
            self._nodes = np.array([[bmin[0], bmin[1], bmin[2],
                                     bmax[0], bmax[1], bmax[2],
                                     0.0, -1.0]], dtype=np.float32)
            self._tri_indices = ctx.tri_indices[:1]
            return

        max_nodes = n_tris * 2
        node_bmin = np.empty((max_nodes, 3), dtype=np.float32)
        node_bmax = np.empty((max_nodes, 3), dtype=np.float32)
        left_child = np.full(max_nodes, -1, dtype=np.int32)
        right_child = np.full(max_nodes, -1, dtype=np.int32)
        leaf_flag = np.ones(max_nodes, dtype=bool)
        leaf_tri_start = np.full(max_nodes, -1, dtype=np.int32)
        leaf_tri_count = np.zeros(max_nodes, dtype=np.int32)
        node_count = 0

        ctx = _BuildCtx(max_nodes, n_tris)
        ctx.total_tris = n_tris
        ctx.n_verts = len(self._vertices.reshape(-1, 3))
        ctx.task_id = self._task_id
        ctx.last_t = time.monotonic()
        _bvh_task_update(ctx.task_id, 0.0,
                         detail=f"0 / {_fmt_count(n_tris)} tris · {_fmt_count(ctx.n_verts)} verts",
                         total=n_tris, units="tris")

        td_build_stack = [(0, n_tris, -1, False)]
        while td_build_stack:
            start, end, parent_idx, is_right = td_build_stack.pop()
            count = end - start
            ni = node_count
            node_count += 1
            if parent_idx >= 0:
                if is_right:
                    right_child[parent_idx] = ni
                else:
                    left_child[parent_idx] = ni

            if count <= _LEAF_SIZE:
                leaf_flag[ni] = True
                tri_range = sorted_tris[start:end]
                node_bmin[ni] = self._tri_bmin[tri_range].min(axis=0)
                node_bmax[ni] = self._tri_bmax[tri_range].max(axis=0)
                leaf_tri_start[ni] = start
                leaf_tri_count[ni] = count
                ctx.progress_done += count
                _report_build_progress(ctx)
            else:
                leaf_flag[ni] = False
                split = _find_morton_split(sorted_morton, start, end)
                td_build_stack.append((split, end, ni, True))
                td_build_stack.append((start, split, ni, False))

        root = 0
        nodes_out = np.empty((node_count, 8), dtype=np.float32)
        tri_offset = 0

        _flatten_stack = [(root, False)]
        while _flatten_stack:
            ni, visited_right = _flatten_stack.pop()
            if leaf_flag[ni]:
                tc = int(leaf_tri_count[ni])
                nodes_out[ni, 0:3] = node_bmin[ni]
                nodes_out[ni, 3:6] = node_bmax[ni]
                nodes_out[ni, 6] = float(tri_offset)
                nodes_out[ni, 7] = float(-tc - 1)
                ts = int(leaf_tri_start[ni])
                ctx.tri_indices[tri_offset:tri_offset + tc] = sorted_tris[ts:ts + tc]
                tri_offset += tc
                continue
            lc = int(left_child[ni])
            rc = int(right_child[ni])
            if not visited_right:
                nodes_out[ni, 6] = float(lc)
                nodes_out[ni, 7] = float(rc)
                _flatten_stack.append((ni, True))
                _flatten_stack.append((rc, False))
                _flatten_stack.append((lc, False))
            else:
                np.minimum(nodes_out[lc, 0:3], nodes_out[rc, 0:3], out=nodes_out[ni, 0:3])
                np.maximum(nodes_out[lc, 3:6], nodes_out[rc, 3:6], out=nodes_out[ni, 3:6])

        self._nodes = nodes_out[:node_count]
        self._tri_indices = ctx.tri_indices[:tri_offset]
        self._root_idx = 0

    def intersect(self, ox: float, oy: float, oz: float,
                  dx: float, dy: float, dz: float,
                  vertices: np.ndarray | None = None,
                  indices: np.ndarray | None = None) -> float:
        nodes = self._nodes
        if len(nodes) == 0:
            return -1.0
        v_arr = self._vertices if vertices is None else vertices
        i_arr = self._indices if indices is None else indices
        if (_raycast_mod is not None and v_arr is self._vertices and i_arr is self._indices
                and v_arr.dtype == np.float32 and v_arr.flags.c_contiguous):
            return _raycast_mod.bvh_intersect(
                nodes, self._tri_v0, self._tri_v1, self._tri_v2,
                np.ascontiguousarray(v_arr).reshape(-1), ox, oy, oz, dx, dy, dz,
                self._root_idx)
        verts3 = v_arr.reshape(-1, 3)

        best_t = float('inf')
        stack = np.empty(64, dtype=np.intp)
        sp = 0
        stack[sp] = self._root_idx
        sp += 1

        while sp > 0:
            sp -= 1
            ni = stack[sp]
            nd = nodes[ni]
            t = _ray_aabb_min(ox, oy, oz, dx, dy, dz,
                              nd[0], nd[1], nd[2],
                              nd[3], nd[4], nd[5])
            if t < 0 or t >= best_t:
                continue

            left_or_start = int(nd[6])
            right_or_count = int(nd[7])
            is_leaf = right_or_count < 0

            if is_leaf:
                tri_start = left_or_start
                tri_end = tri_start + (-right_or_count - 1)
                for ti in range(tri_start, tri_end):
                    t_idx = int(self._tri_indices[ti])
                    vi0 = int(i_arr[t_idx * 3])
                    vi1 = int(i_arr[t_idx * 3 + 1])
                    vi2 = int(i_arr[t_idx * 3 + 2])
                    v0 = verts3[vi0]
                    v1 = verts3[vi1]
                    v2 = verts3[vi2]
                    tt = _moller_trumbore(ox, oy, oz, dx, dy, dz,
                                          v0[0], v0[1], v0[2],
                                          v1[0], v1[1], v1[2],
                                          v2[0], v2[1], v2[2])
                    if tt > 0 and tt < best_t:
                        best_t = tt
            else:
                n_left = left_or_start
                n_right = right_or_count
                ln = nodes[n_left]
                rn = nodes[n_right]
                tl = _ray_aabb_min(ox, oy, oz, dx, dy, dz,
                                   ln[0], ln[1], ln[2],
                                   ln[3], ln[4], ln[5])
                tr = _ray_aabb_min(ox, oy, oz, dx, dy, dz,
                                   rn[0], rn[1], rn[2],
                                   rn[3], rn[4], rn[5])
                if tl < 0 and tr < 0:
                    continue
                if tl < 0:
                    stack[sp] = n_right
                    sp += 1
                elif tr < 0:
                    stack[sp] = n_left
                    sp += 1
                else:
                    if tl < tr:
                        stack[sp] = n_right
                        sp += 1
                        stack[sp] = n_left
                        sp += 1
                    else:
                        stack[sp] = n_left
                        sp += 1
                        stack[sp] = n_right
                        sp += 1

        return -1.0 if best_t == float('inf') else best_t

    def intersect_any(self, ox: float, oy: float, oz: float,
                      dx: float, dy: float, dz: float,
                      vertices: np.ndarray | None = None,
                      indices: np.ndarray | None = None) -> bool:
        nodes = self._nodes
        if len(nodes) == 0:
            return False
        v_arr = self._vertices if vertices is None else vertices
        i_arr = self._indices if indices is None else indices
        if (_raycast_mod is not None and v_arr is self._vertices and i_arr is self._indices
                and v_arr.dtype == np.float32 and v_arr.flags.c_contiguous):
            return _raycast_mod.bvh_intersect_any(
                nodes, self._tri_v0, self._tri_v1, self._tri_v2,
                np.ascontiguousarray(v_arr).reshape(-1), ox, oy, oz, dx, dy, dz,
                self._root_idx)
        verts3 = v_arr.reshape(-1, 3)

        stack = np.empty(64, dtype=np.intp)
        sp = 0
        stack[sp] = self._root_idx
        sp += 1

        while sp > 0:
            sp -= 1
            ni = stack[sp]
            nd = nodes[ni]
            t = _ray_aabb_min(ox, oy, oz, dx, dy, dz,
                              nd[0], nd[1], nd[2],
                              nd[3], nd[4], nd[5])
            if t < 0:
                continue

            left_or_start = int(nd[6])
            right_or_count = int(nd[7])
            is_leaf = right_or_count < 0

            if is_leaf:
                tri_start = left_or_start
                tri_end = tri_start + (-right_or_count - 1)
                for ti in range(tri_start, tri_end):
                    t_idx = int(self._tri_indices[ti])
                    vi0 = int(i_arr[t_idx * 3])
                    vi1 = int(i_arr[t_idx * 3 + 1])
                    vi2 = int(i_arr[t_idx * 3 + 2])
                    v0 = verts3[vi0]
                    v1 = verts3[vi1]
                    v2 = verts3[vi2]
                    tt = _moller_trumbore(ox, oy, oz, dx, dy, dz,
                                          v0[0], v0[1], v0[2],
                                          v1[0], v1[1], v1[2],
                                          v2[0], v2[1], v2[2])
                    if tt > 0:
                        return True
            else:
                n_left = left_or_start
                n_right = right_or_count
                ln = nodes[n_left]
                rn = nodes[n_right]
                tl = _ray_aabb_min(ox, oy, oz, dx, dy, dz,
                                   ln[0], ln[1], ln[2],
                                   ln[3], ln[4], ln[5])
                tr = _ray_aabb_min(ox, oy, oz, dx, dy, dz,
                                   rn[0], rn[1], rn[2],
                                   rn[3], rn[4], rn[5])
                if tl < 0 and tr < 0:
                    continue
                if tl < 0:
                    stack[sp] = n_right
                    sp += 1
                elif tr < 0:
                    stack[sp] = n_left
                    sp += 1
                else:
                    if tl < tr:
                        stack[sp] = n_right
                        sp += 1
                        stack[sp] = n_left
                        sp += 1
                    else:
                        stack[sp] = n_left
                        sp += 1
                        stack[sp] = n_right
                        sp += 1

        return False

    def flatten_for_gpu(self):
        return self._nodes

    def flatten_for_gpu_stackless(self) -> np.ndarray:
        nodes = self._nodes
        n = len(nodes)
        if n == 0:
            return np.empty((0, 8), dtype=np.float32)

        sizes = np.empty(n, dtype=np.int64)
        for i in range(n):
            right_or_count = nodes[i, 7]
            if right_or_count < 0:
                sizes[i] = 1
            else:
                l = int(nodes[i, 6])
                r = int(right_or_count)
                sizes[i] = 1 + sizes[l] + sizes[r]

        order = np.empty(n, dtype=np.int64)
        stack = [n - 1]
        w = 0
        while stack:
            old = stack.pop()
            order[w] = old
            w += 1
            right_or_count = nodes[old, 7]
            if right_or_count >= 0:
                stack.append(int(right_or_count))
                stack.append(int(nodes[old, 6]))

        out = np.empty((n, 8), dtype=np.float32)
        out[:, 0:6] = nodes[order, 0:6]
        right_or_count_ord = nodes[order, 7]
        leaf_mask = right_or_count_ord < 0
        out[leaf_mask, 6] = nodes[order[leaf_mask], 6]
        out[leaf_mask, 7] = right_or_count_ord[leaf_mask]
        internal_new_idx = np.nonzero(~leaf_mask)[0]
        out[internal_new_idx, 6] = (internal_new_idx + sizes[order[internal_new_idx]]).astype(np.float32)
        out[internal_new_idx, 7] = 0.0
        return out


    @property
    def tri_indices(self) -> np.ndarray:
        return self._tri_indices

    def node_count(self) -> int:
        return len(self._nodes)

    def depth(self) -> int:
        nodes = self._nodes
        if len(nodes) == 0:
            return 0

        def _max_depth(ni, d):
            if ni < 0 or ni >= len(nodes):
                return d
            nd = nodes[ni]
            if nd[7] < 0:
                return d
            return max(_max_depth(int(nd[6]), d + 1), _max_depth(int(nd[7]), d + 1))

        return _max_depth(self._root_idx, 0)

    @property
    def node_depths(self) -> list[int]:
        if self._cached_depths is None:
            self._cached_depths = _compute_node_depths(self)
        return self._cached_depths

    @property
    def nodes(self):
        if self._node_views is None:
            class _NodeView:
                __slots__ = ('_data',)
                def __init__(self, data):
                    self._data = data
                @property
                def bmin(self):
                    return self._data[0:3]
                @property
                def bmax(self):
                    return self._data[3:6]
                @property
                def left(self):
                    return int(self._data[6])
                @property
                def right(self):
                    return int(self._data[7])
                @property
                def tri_start(self):
                    return int(self._data[6])
                @property
                def tri_count(self):
                    return 0 if self._data[7] >= 0 else int(-self._data[7] - 1)
                @property
                def is_leaf(self):
                    return self._data[7] < 0
            self._node_views = [_NodeView(self._nodes[i]) for i in range(len(self._nodes))]
        return self._node_views

    def enumerate_nodes(self):
        for i in range(len(self._nodes)):
            yield i, self.nodes[i]


try:
    from core._bvh_build import sah_compute_best_split as _sah_cython_split
    _USE_CYTHON_BVH = True
except ImportError:
    _USE_CYTHON_BVH = False

try:
    from core import _raycast as _raycast_mod
except ImportError:
    _raycast_mod = None


_BVH_CACHE: dict[str, BVH | Future | None] = {}
_BVH_LOCK = threading.Lock()


def _bvh_cache_key(vertices: np.ndarray, indices: np.ndarray) -> str:
    h = xxhash.xxh128()
    h.update(vertices.data)
    h.update(indices.data)
    return h.hexdigest()


def _bvh_cache_dir() -> str | None:
    try:
        from core.engine.engine import Engine
        eng = Engine.instance()
        if eng is None:
            return None
        root = eng.project_root
        d = os.path.join(root, "cache", "bvh")
        os.makedirs(d, exist_ok=True)
        return d
    except (ImportError, OSError, AttributeError, TypeError):
        return None

def _save_bvh_disk(key: str, bvh: BVH, cache_dir: str) -> None:
    prefix = f"v{_BVH_CACHE_VERSION}_{key}"
    nodes_path = os.path.join(cache_dir, prefix + "_n.npy")
    tmp_nodes = nodes_path + ".tmp"
    tris_path = os.path.join(cache_dir, prefix + "_t.npy")
    tmp_tris = tris_path + ".tmp"
    root_path = os.path.join(cache_dir, prefix + "_r.npy")
    tmp_root = root_path + ".tmp"
    try:
        np.save(tmp_nodes, bvh._nodes, allow_pickle=False)
        np.save(tmp_tris, bvh._tri_indices, allow_pickle=False)
        np.save(tmp_root, np.array(bvh._root_idx, dtype=np.int32), allow_pickle=False)
        os.replace(tmp_nodes + ".npy", nodes_path)
        os.replace(tmp_tris + ".npy", tris_path)
        os.replace(tmp_root + ".npy", root_path)
    except (OSError, ValueError):
        for p in (tmp_nodes + ".npy", tmp_tris + ".npy", tmp_root + ".npy",
                   tmp_nodes, tmp_tris, tmp_root):
            try:
                os.remove(p)
            except OSError:
                pass


def _load_bvh_disk(key: str, cache_dir: str,
                   vertices: np.ndarray, indices: np.ndarray) -> BVH | None:
    prefix = f"v{_BVH_CACHE_VERSION}_{key}"
    nodes_path = os.path.join(cache_dir, prefix + "_n.npy")
    tris_path = os.path.join(cache_dir, prefix + "_t.npy")
    root_path = os.path.join(cache_dir, prefix + "_r.npy")
    try:
        nodes = np.load(nodes_path, allow_pickle=False)
        tri_indices = np.load(tris_path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    root_idx = len(nodes) - 1
    try:
        root_idx = int(np.load(root_path, allow_pickle=False))
    except (OSError, ValueError):
        pass
    return BVH.from_cache(np.array(nodes), tri_indices, vertices, indices, root_idx)



def _build_bvh(vertices: np.ndarray, indices: np.ndarray, title: str | None = None,
               cache_key: str | None = None,
               progress_queue: object | None = None):
    global _bvh_progress_queue
    _bvh_progress_queue = progress_queue
    task_id = cache_key or str(id(vertices))
    _bvh_log(f"BUILD start key={task_id} verts={len(vertices)} idx={len(indices)}")
    if progress_queue is not None:
        _pq_put(("start", (task_id, title or "Building BVH..."),
                 {"fraction": 0.0}))
    else:
        task_start(task_id, title or "Building BVH...", fraction=0.0)
    try:
        bvh = BVH(vertices, indices, task_id=task_id)
        result = bvh._nodes.copy(), bvh._tri_indices.copy(), bvh._root_idx
        _bvh_log(f"BUILD done key={task_id} nodes={result[0].shape[0]}")
        return result
    except Exception as exc:
        _bvh_log(f"BUILD FAIL key={task_id} {type(exc).__name__}: {exc}")
        return None
    finally:
        _bvh_progress_queue = None
        try:
            if progress_queue is not None:
                progress_queue.put_nowait(("complete", (task_id,), {}))
        except Exception:
            pass


def prebuild_mesh_bvh(vertices: np.ndarray, indices: np.ndarray, title: str | None = None) -> None:
    """Trigger async BVH pre-build. Safe to call from any thread."""
    if vertices is None or len(vertices) < 3 or indices is None or len(indices) < 3:
        key = _bvh_cache_key(
            np.zeros(3, dtype=np.float32) if vertices is None else vertices,
            np.zeros(3, dtype=np.uint32) if indices is None else indices,
        )
        with _BVH_LOCK:
            _BVH_CACHE[key] = None
        _bvh_log(f"PREBUILD skip (tiny mesh)")
        return
    key = _bvh_cache_key(vertices, indices)
    with _BVH_LOCK:
        if key in _BVH_CACHE:
            _bvh_log(f"PREBUILD key={key} ALREADY_CACHED")
            return
        cache_dir = _bvh_cache_dir()
        if cache_dir:
            cached = _load_bvh_disk(key, cache_dir, vertices, indices)
            if cached is not None:
                _BVH_CACHE[key] = cached
                _bvh_log(f"PREBUILD key={key} DISK_HIT")
                return
        pool = _get_bvh_pool()
        pq = get_progress_queue()
        _bvh_log(f"PREBUILD key={key} SUBMIT verts={len(vertices)} idx={len(indices)} pool={type(pool).__name__}")
        _BVH_CACHE[key] = pool.submit(_build_bvh, vertices, indices, title, key,
                                       progress_queue=pq)


def get_mesh_bvh(vertices: np.ndarray, indices: np.ndarray) -> BVH | None:
    if vertices is None or len(vertices) < 3 or indices is None or len(indices) < 3:
        return None
    key = _bvh_cache_key(vertices, indices)
    with _BVH_LOCK:
        if key not in _BVH_CACHE:
            cache_dir = _bvh_cache_dir()
            if cache_dir:
                cached = _load_bvh_disk(key, cache_dir, vertices, indices)
                if cached is not None:
                    _BVH_CACHE[key] = cached
                    _bvh_log(f"GET key={key} DISK HIT")
                    return cached
            _bvh_log(f"GET key={key} SUBMIT verts={len(vertices)} idx={len(indices)}")
            _BVH_CACHE[key] = _get_bvh_pool().submit(_build_bvh, vertices, indices,
                                                     cache_key=key,
                                                     progress_queue=get_progress_queue())
        entry = _BVH_CACHE[key]
    if isinstance(entry, Future):
        if entry.done():
            try:
                result = entry.result()
            except Exception as exc:
                _bvh_log(f"GET key={key} FUTURE FAIL: {type(exc).__name__}: {exc}")
                result = None
            if isinstance(result, tuple) and len(result) == 3:
                result = BVH.from_cache(result[0], result[1], vertices, indices, result[2])
            elif isinstance(result, tuple):
                result = BVH.from_cache(result[0], result[1], vertices, indices)
            with _BVH_LOCK:
                _BVH_CACHE[key] = result
            if result is not None:
                cache_dir = _bvh_cache_dir()
                if cache_dir:
                    _save_bvh_disk(key, result, cache_dir)
                _bvh_log(f"GET key={key} BUILT nodes={result.node_count()}")
            else:
                _bvh_log(f"GET key={key} result=None")
            return result
        _bvh_log(f"GET key={key} PENDING")
        return None
    if entry is None:
        _bvh_log(f"GET key={key} CACHED_NONE")
    return entry


def get_mesh_bvh_sync(vertices: np.ndarray, indices: np.ndarray, timeout: float = 60.0) -> BVH | None:
    if vertices is None or len(vertices) < 3 or indices is None or len(indices) < 3:
        return None
    key = _bvh_cache_key(vertices, indices)
    with _BVH_LOCK:
        if key not in _BVH_CACHE:
            cache_dir = _bvh_cache_dir()
            if cache_dir:
                cached = _load_bvh_disk(key, cache_dir, vertices, indices)
                if cached is not None:
                    _BVH_CACHE[key] = cached
                    return cached
            _bvh_log(f"SYNC key={key} SUBMIT verts={len(vertices)} idx={len(indices)}")
            _BVH_CACHE[key] = _get_bvh_pool().submit(_build_bvh, vertices, indices,
                                                     cache_key=key,
                                                     progress_queue=get_progress_queue())
        entry = _BVH_CACHE[key]
    if isinstance(entry, Future):
        try:
            result = entry.result(timeout=timeout)
        except Exception as exc:
            _bvh_log(f"SYNC key={key} FUTURE FAIL: {type(exc).__name__}: {exc}")
            result = None
        if isinstance(result, tuple) and len(result) == 3:
            result = BVH.from_cache(result[0], result[1], vertices, indices, result[2])
        elif isinstance(result, tuple):
            result = BVH.from_cache(result[0], result[1], vertices, indices)
        with _BVH_LOCK:
            _BVH_CACHE[key] = result
        if result is not None:
            cache_dir = _bvh_cache_dir()
            if cache_dir:
                _save_bvh_disk(key, result, cache_dir)
            _bvh_log(f"SYNC key={key} BUILT nodes={result.node_count()}")
        return result
    return entry


def _build_bvh_lines(bvh: BVH, depth_filter: int = -1) -> list[tuple]:
    depths = _compute_node_depths(bvh) if depth_filter >= 0 else None
    lines = []
    from core.maths.math3d import Vec3
    n_nodes = len(bvh._nodes)
    for ni in range(n_nodes):
        if depth_filter >= 0 and depths[ni] != depth_filter:
            continue
        nd = bvh._nodes[ni]
        mn = nd[0:3]
        mx = nd[3:6]
        corners = [
            (mn[0], mn[1], mn[2]), (mx[0], mn[1], mn[2]),
            (mx[0], mx[1], mn[2]), (mn[0], mx[1], mn[2]),
            (mn[0], mn[1], mx[2]), (mx[0], mn[1], mx[2]),
            (mx[0], mx[1], mx[2]), (mn[0], mx[1], mx[2]),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        frac = ni / max(1, n_nodes)
        hue = frac * 0.66
        r, g, b = _hsv_to_rgb(hue, 0.8, 0.6 + 0.4 * (1.0 - frac))
        color = [r, g, b]
        for e in edges:
            s = Vec3(*corners[e[0]])
            t = Vec3(*corners[e[1]])
            lines.append((s, t, color))
    return lines


def _node_depth(bvh, ni):
    nodes = bvh._nodes
    if ni < 0 or ni >= len(nodes):
        return 0
    nd = nodes[ni]
    if nd[7] < 0:
        return 0
    return 1 + max(_node_depth(bvh, int(nd[6])), _node_depth(bvh, int(nd[7])))


def _compute_node_depths(bvh):
    nodes = bvh._nodes
    depths = [0] * len(nodes)

    def walk(ni, d):
        if ni < 0 or ni >= len(nodes):
            return
        depths[ni] = d
        nd = nodes[ni]
        if nd[7] >= 0:
            walk(int(nd[6]), d + 1)
            walk(int(nd[7]), d + 1)

    if len(nodes) > 0:
        walk(bvh._root_idx, 0)
    return depths


def _hsv_to_rgb(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    if h_i == 0:
        return v, t, p
    elif h_i == 1:
        return q, v, p
    elif h_i == 2:
        return p, v, t
    elif h_i == 3:
        return p, q, v
    elif h_i == 4:
        return t, p, v
    else:
        return v, p, q


_EDGE_PAIRS = np.array([
    0, 1, 1, 2, 2, 3, 3, 0,
    4, 5, 5, 6, 6, 7, 7, 4,
    0, 4, 1, 5, 2, 6, 3, 7,
], dtype=np.intp).reshape(-1, 2)


def build_bvh_arrays(bvh, max_depth: int = -1):
    n = len(bvh._nodes)
    if n == 0:
        return None, None, None
    if max_depth < 0:
        mask = None
        count = n
    else:
        depths = bvh.node_depths
        mask = np.array([d <= max_depth for d in depths], dtype=bool)
        count = int(mask.sum())
        if count == 0:
            return None, None, None
    corners = np.empty((count, 8, 3), dtype=np.float32)
    if mask is None:
        for i in range(n):
            nd = bvh._nodes[i]
            mn, mx = nd[0:3], nd[3:6]
            corners[i, 0] = (mn[0], mn[1], mn[2])
            corners[i, 1] = (mx[0], mn[1], mn[2])
            corners[i, 2] = (mx[0], mx[1], mn[2])
            corners[i, 3] = (mn[0], mx[1], mn[2])
            corners[i, 4] = (mn[0], mn[1], mx[2])
            corners[i, 5] = (mx[0], mn[1], mx[2])
            corners[i, 6] = (mx[0], mx[1], mx[2])
            corners[i, 7] = (mn[0], mx[1], mx[2])
    else:
        idx = 0
        for i in range(n):
            if not mask[i]:
                continue
            nd = bvh._nodes[i]
            mn, mx = nd[0:3], nd[3:6]
            corners[idx, 0] = (mn[0], mn[1], mn[2])
            corners[idx, 1] = (mx[0], mn[1], mn[2])
            corners[idx, 2] = (mx[0], mx[1], mn[2])
            corners[idx, 3] = (mn[0], mx[1], mn[2])
            corners[idx, 4] = (mn[0], mn[1], mx[2])
            corners[idx, 5] = (mx[0], mn[1], mx[2])
            corners[idx, 6] = (mx[0], mx[1], mx[2])
            corners[idx, 7] = (mn[0], mx[1], mx[2])
            idx += 1
    starts = corners[:, _EDGE_PAIRS[:, 0], :].reshape(-1, 3)
    ends = corners[:, _EDGE_PAIRS[:, 1], :].reshape(-1, 3)
    frac = np.arange(count, dtype=np.float32) / max(1, count)
    hue = frac * 0.66
    sat = np.full(count, 0.8, dtype=np.float32)
    val = 0.6 + 0.4 * (1.0 - frac)
    h_i = (hue * 6).astype(np.intp)
    f = hue * 6 - h_i
    p = val * (1 - sat)
    q = val * (1 - f * sat)
    t = val * (1 - (1 - f) * sat)
    r = np.select([h_i == 0, h_i == 1, h_i == 2, h_i == 3, h_i == 4, h_i >= 5],
                  [val, q, p, p, t, val])
    g = np.select([h_i == 0, h_i == 1, h_i == 2, h_i == 3, h_i == 4, h_i >= 5],
                  [t, val, val, q, p, p])
    b = np.select([h_i == 0, h_i == 1, h_i == 2, h_i == 3, h_i == 4, h_i >= 5],
                  [p, p, t, val, val, q])
    colors = np.column_stack([r, g, b])
    colors = np.repeat(colors[:, None, :], 12, axis=1).reshape(-1, 3)
    return starts, ends, colors