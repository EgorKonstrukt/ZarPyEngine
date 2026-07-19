# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You cannot obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


_MESHLET_QUADS = 8


@dataclass
class MeshletLevel:
    resolution: int
    heightfield: np.ndarray
    vertices: np.ndarray
    meshlet_indices: np.ndarray
    meshlet_count: int
    meshlet_aabb: np.ndarray
    meshlet_cone: np.ndarray
    lod: int


@dataclass
class TerrainMeshletData:
    size: float
    levels: List[MeshletLevel] = field(default_factory=list)
    meshlet_offset: np.ndarray = None
    total_meshlets: int = 0
    base_resolution: int = 0
    morph_range: float = 0.35


def _build_mip_chain(hf: np.ndarray) -> List[np.ndarray]:
    levels = [hf.astype(np.float32)]
    cur = hf
    while cur.shape[0] > 16:
        half = cur.shape[0] // 2
        down = np.zeros((half, half), dtype=np.float32)
        s = cur.shape[0]
        for j in range(half):
            for i in range(half):
                down[j, i] = cur[2 * j, 2 * i]
        levels.append(down)
        cur = down
    return levels


def _build_level(heightfield: np.ndarray, size: float, lod: int) -> MeshletLevel:
    res = heightfield.shape[0]
    step = size / (res - 1)
    half = size * 0.5
    xs = (np.arange(res) * step - half).astype(np.float32)
    zs = (np.arange(res) * step - half).astype(np.float32)
    gx, gz = np.meshgrid(xs, zs)
    verts = np.empty((res * res, 3), dtype=np.float32)
    verts[:, 0] = gx.ravel()
    verts[:, 1] = 0.0
    verts[:, 2] = gz.ravel()

    q = _MESHLET_QUADS
    mc = max(1, res // q)
    nx = mc
    nz = mc
    if nx * q < res - 1:
        nx += 1
    if nz * q < res - 1:
        nz += 1
    meshlet_count = nx * nz

    meshlet_indices = np.empty(meshlet_count * q * q * 6, dtype=np.uint32)
    meshlet_aabb = np.empty((meshlet_count, 6), dtype=np.float32)
    meshlet_cone = np.empty((meshlet_count, 4), dtype=np.float32)

    pos = 0
    idx = 0
    for mz in range(nz):
        for mx in range(nx):
            x0 = mx * q
            z0 = mz * q
            x1 = min(x0 + q, res - 1)
            z1 = min(z0 + q, res - 1)
            cxs = (xs[x0] + xs[x1]) * 0.5
            czs = (zs[z0] + zs[z1]) * 0.5
            miny = float(heightfield[z0:z1 + 1, x0:x1 + 1].min())
            maxy = float(heightfield[z0:z1 + 1, x0:x1 + 1].max())
            meshlet_aabb[idx, 0] = xs[x0]
            meshlet_aabb[idx, 1] = miny
            meshlet_aabb[idx, 2] = zs[z0]
            meshlet_aabb[idx, 3] = xs[x1]
            meshlet_aabb[idx, 4] = maxy
            meshlet_aabb[idx, 5] = zs[z1]
            cz = z0 + (z1 - z0) // 2
            cx = x0 + (x1 - x0) // 2
            hy = float(heightfield[cz, cx])
            if cz + 1 < res and cz - 1 >= 0 and cx + 1 < res and cx - 1 >= 0:
                dx = (heightfield[cz, cx + 1] - heightfield[cz, cx - 1]) * 0.5
                dz = (heightfield[cz + 1, cx] - heightfield[cz - 1, cx]) * 0.5
            else:
                dx = 0.0
                dz = 0.0
            n = np.array([-dx, step, -dz], dtype=np.float32)
            ln = float(np.linalg.norm(n))
            if ln > 1e-6:
                n /= ln
            meshlet_cone[idx, 0] = cxs
            meshlet_cone[idx, 1] = hy
            meshlet_cone[idx, 2] = czs
            meshlet_cone[idx, 3] = max(-1.0, min(1.0, n[1]))
            for j in range(z0, z1):
                for i in range(x0, x1):
                    a = j * res + i
                    b = j * res + i + 1
                    c = (j + 1) * res + i + 1
                    d = (j + 1) * res + i
                    meshlet_indices[pos] = a
                    meshlet_indices[pos + 1] = c
                    meshlet_indices[pos + 2] = b
                    meshlet_indices[pos + 3] = a
                    meshlet_indices[pos + 4] = d
                    meshlet_indices[pos + 5] = c
                    pos += 6
            idx += 1
    return MeshletLevel(
        resolution=res,
        heightfield=heightfield,
        vertices=verts,
        meshlet_indices=meshlet_indices[:pos].copy(),
        meshlet_count=meshlet_count,
        meshlet_aabb=meshlet_aabb,
        meshlet_cone=meshlet_cone,
        lod=lod,
    )


def build_meshlets(heightfield: np.ndarray, size: float) -> TerrainMeshletData:
    chain = _build_mip_chain(heightfield)
    levels = []
    for lod, hf in enumerate(chain):
        levels.append(_build_level(hf, size, lod))
    total = sum(lv.meshlet_count for lv in levels)
    offsets = np.zeros(total, dtype=np.uint32)
    o = 0
    for lv in levels:
        for _ in range(lv.meshlet_count):
            offsets[o] = lv.lod
            o += 1
    return TerrainMeshletData(
        size=size,
        levels=levels,
        meshlet_offset=offsets,
        total_meshlets=total,
        base_resolution=heightfield.shape[0],
    )
