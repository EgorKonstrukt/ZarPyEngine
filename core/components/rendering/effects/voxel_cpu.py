# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np


def tri_box_overlap(center: np.ndarray, half: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> bool:
    t0 = v0 - center
    t1 = v1 - center
    t2 = v2 - center

    tmin = np.minimum(np.minimum(t0, t1), t2)
    tmax = np.maximum(np.maximum(t0, t1), t2)
    if (tmin > half).any() or (tmax < -half).any():
        return False

    e0 = t1 - t0
    e1 = t2 - t1
    e2 = t0 - t2
    n = np.cross(e0, e1)
    rad = abs(n[0]) * half[0] + abs(n[1]) * half[1] + abs(n[2]) * half[2]
    d0 = n[0] * t0[0] + n[1] * t0[1] + n[2] * t0[2]
    d1 = n[0] * t1[0] + n[1] * t1[1] + n[2] * t1[2]
    d2 = n[0] * t2[0] + n[1] * t2[1] + n[2] * t2[2]
    if min(d0, d1, d2) > rad or max(d0, d1, d2) < -rad:
        return False

    box_axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    edges = (e0, e1, e2)
    for ux, uy, uz in box_axes:
        for e in edges:
            ax = (uy * e[2] - uz * e[1], uz * e[0] - ux * e[2], ux * e[1] - uy * e[0])
            if ax[0] == 0.0 and ax[1] == 0.0 and ax[2] == 0.0:
                continue
            p0 = ax[0] * t0[0] + ax[1] * t0[1] + ax[2] * t0[2]
            p1 = ax[0] * t1[0] + ax[1] * t1[1] + ax[2] * t1[2]
            p2 = ax[0] * t2[0] + ax[1] * t2[1] + ax[2] * t2[2]
            pmin = min(p0, p1, p2)
            pmax = max(p0, p1, p2)
            er = abs(ax[0]) * half[0] + abs(ax[1]) * half[1] + abs(ax[2]) * half[2]
            if pmin > er or pmax < -er:
                return False
    return True


def compute_voxel_instances(verts, idx, model, size: float, world_grid: bool, jitter: float, seed: int = 0) -> np.ndarray:
    if verts is None or size <= 1e-5:
        return np.zeros((0, 4), dtype=np.float32)

    V = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    if V.shape[0] < 3:
        return np.zeros((0, 4), dtype=np.float32)
    if world_grid and model is not None:
        m = np.asarray(model, dtype=np.float32).reshape(4, 4)
        ones = np.ones((V.shape[0], 1), dtype=np.float32)
        V = (m @ np.concatenate([V, ones], axis=1).T).T[:, :3]

    if idx is not None and len(idx) >= 3:
        nt = (len(idx) // 3) * 3
        tris = np.asarray(idx[:nt], dtype=np.int32).reshape(-1, 3)
    else:
        n = V.shape[0] // 3
        if n == 0:
            return np.zeros((0, 4), dtype=np.float32)
        tris = np.arange(n * 3, dtype=np.int32).reshape(-1, 3)

    inv = 1.0 / float(size)
    half = np.array([size * 0.5, size * 0.5, size * 0.5], dtype=np.float32)
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)

    cell_set: set = set()
    for t in tris:
        a = V[t[0]]
        b = V[t[1]]
        c = V[t[2]]
        tmn = np.minimum(np.minimum(a, b), c)
        tmx = np.maximum(np.maximum(a, b), c)
        cmin = np.floor(tmn * inv).astype(np.int64)
        cmax = np.floor(tmx * inv).astype(np.int64)
        span = (cmax[0] - cmin[0] + 1) * (cmax[1] - cmin[1] + 1) * (cmax[2] - cmin[2] + 1)
        if span > 4096:
            continue
        for ix in range(cmin[0], cmax[0] + 1):
            cx = (ix + 0.5) * size
            for iy in range(cmin[1], cmax[1] + 1):
                cy = (iy + 0.5) * size
                for iz in range(cmin[2], cmax[2] + 1):
                    cz = (iz + 0.5) * size
                    if tri_box_overlap(np.array([cx, cy, cz], dtype=np.float32), half, a, b, c):
                        cell_set.add((int(ix), int(iy), int(iz)))

    if not cell_set:
        return np.zeros((0, 4), dtype=np.float32)

    cells = np.array(list(cell_set), dtype=np.int64)
    centers = (cells.astype(np.float32) + 0.5) * size
    rnd = (rng.random((len(cells), 3)).astype(np.float32) - 0.5) * jitter * size

    out = np.zeros((len(cells), 4), dtype=np.float32)
    out[:, :3] = centers + rnd
    return out
