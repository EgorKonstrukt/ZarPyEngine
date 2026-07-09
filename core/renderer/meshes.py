# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

import numpy as np
from core.renderer.mesh_data import MeshData


def make_cube_mesh() -> MeshData:
    """Create a unit cube mesh with positions, normals and UVs."""
    v = np.array([
        -0.5,-0.5,-0.5, -0.5, 0.5,-0.5,  0.5, 0.5,-0.5,  0.5,-0.5,-0.5,
         0.5,-0.5, 0.5,  0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5,-0.5, 0.5,
        -0.5,-0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5,-0.5, -0.5,-0.5,-0.5,
         0.5,-0.5,-0.5,  0.5, 0.5,-0.5,  0.5, 0.5, 0.5,  0.5,-0.5, 0.5,
        -0.5,-0.5,-0.5,  0.5,-0.5,-0.5,  0.5,-0.5, 0.5, -0.5,-0.5, 0.5,
        -0.5, 0.5, 0.5,  0.5, 0.5, 0.5,  0.5, 0.5,-0.5, -0.5, 0.5,-0.5,
    ], dtype=np.float32)
    n = np.array([
        0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1,
        0,0, 1, 0,0, 1, 0,0, 1, 0,0, 1,
        -1,0,0, -1,0,0, -1,0,0, -1,0,0,
         1,0,0,  1,0,0,  1,0,0,  1,0,0,
        0,-1,0, 0,-1,0, 0,-1,0, 0,-1,0,
        0, 1,0, 0, 1,0, 0, 1,0, 0, 1,0,
    ], dtype=np.float32)
    uv = np.array([
        0,0, 0,1, 1,1, 1,0,
        0,0, 0,1, 1,1, 1,0,
        0,0, 0,1, 1,1, 1,0,
        0,0, 0,1, 1,1, 1,0,
        0,0, 1,0, 1,1, 0,1,
        0,0, 1,0, 1,1, 0,1,
    ], dtype=np.float32)
    idx = []
    for f in range(6):
        b = f * 4
        idx += [b,b+1,b+2, b,b+2,b+3]
    mesh = MeshData()
    mesh.vertices = v
    mesh.normals = n
    mesh.uvs = uv
    mesh.indices = np.array(idx, dtype=np.uint32)
    return mesh


def make_sphere_mesh(segments: int = 16) -> MeshData:
    """Create a UV sphere mesh with given segment count."""
    verts, norms, uvs_arr, idxs = [], [], [], []
    for i in range(segments+1):
        lat = np.pi * (-0.5 + float(i)/segments)
        for j in range(segments+1):
            lon = 2*np.pi * float(j)/segments
            x = np.cos(lat)*np.cos(lon)
            y = np.sin(lat)
            z = np.cos(lat)*np.sin(lon)
            verts += [x*0.5, y*0.5, z*0.5]
            norms += [x, y, z]
            uvs_arr += [float(j)/segments, float(i)/segments]
    for i in range(segments):
        for j in range(segments):
            a = i*(segments+1)+j
            b = a+segments+1
            idxs += [a, b, a+1, b, b+1, a+1]
    mesh = MeshData()
    mesh.vertices = np.array(verts, dtype=np.float32)
    mesh.normals = np.array(norms, dtype=np.float32)
    mesh.uvs = np.array(uvs_arr, dtype=np.float32)
    mesh.indices = np.array(idxs, dtype=np.uint32)
    return mesh


def make_plane_mesh(size: float = 1.0) -> MeshData:
    """Create a horizontal plane mesh facing up."""
    h = size * 0.5
    v = np.array([-h,0,-h, h,0,-h, h,0,h, -h,0,h], dtype=np.float32)
    n = np.array([0,1,0, 0,1,0, 0,1,0, 0,1,0], dtype=np.float32)
    uv = np.array([0,0, 1,0, 1,1, 0,1], dtype=np.float32)
    mesh = MeshData()
    mesh.vertices = v
    mesh.normals = n
    mesh.uvs = uv
    mesh.indices = np.array([0,1,2, 0,2,3], dtype=np.uint32)
    return mesh


def make_water_plane(size: float = 1.0, segments: int = 200) -> MeshData:
    """Create a tessellated horizontal plane (y=0) for water/ocean rendering.

    Provides positions, up-normals and 0..1 UVs. Vertex shader displaces it
    (Gerstner waves), so a single large plane can represent an ocean.
    """
    seg = max(1, int(segments))
    half = size * 0.5
    step = size / seg
    verts = []
    uvs = []
    norms = []
    idxs = []
    for z in range(seg + 1):
        for x in range(seg + 1):
            px = -half + x * step
            pz = -half + z * step
            verts.append(px)
            verts.append(0.0)
            verts.append(pz)
            uvs.append(float(x) / seg)
            uvs.append(float(z) / seg)
            norms.append(0.0)
            norms.append(1.0)
            norms.append(0.0)
    def vid(x, z):
        return z * (seg + 1) + x
    for z in range(seg):
        for x in range(seg):
            a = vid(x, z)
            b = vid(x + 1, z)
            c = vid(x + 1, z + 1)
            d = vid(x, z + 1)
            idxs.append(a)
            idxs.append(b)
            idxs.append(c)
            idxs.append(a)
            idxs.append(c)
            idxs.append(d)
    mesh = MeshData()
    mesh.vertices = np.array(verts, dtype=np.float32)
    mesh.normals = np.array(norms, dtype=np.float32)
    mesh.uvs = np.array(uvs, dtype=np.float32)
    mesh.indices = np.array(idxs, dtype=np.uint32)
    return mesh


def make_quad_mesh(size: float = 1.0) -> MeshData:
    """Create a screen-aligned quad facing +Z."""
    h = size * 0.5
    v = np.array([-h, -h, 0,  h, -h, 0,  h, h, 0,  -h, h, 0], dtype=np.float32)
    n = np.array([0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1], dtype=np.float32)
    uv = np.array([0, 0,  1, 0,  1, 1,  0, 1], dtype=np.float32)
    mesh = MeshData()
    mesh.vertices = v
    mesh.normals = n
    mesh.uvs = uv
    mesh.indices = np.array([0, 1, 2,  0, 2, 3], dtype=np.uint32)
    return mesh
