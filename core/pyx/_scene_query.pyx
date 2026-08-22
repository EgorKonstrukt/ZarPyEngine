# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, overflowcheck=False, infer_types=True
# distutils: extra_compile_args = -O3 -ffast-math -march=native -fopenmp
# distutils: extra_link_args = -fopenmp
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel import prange
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
FTYPE = np.float32
ctypedef np.float32_t FTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_get_entities_with_component(dict component_indices, dict entities, str comp_name, int render_version, dict frame_cache):
    cdef object s = component_indices.get(comp_name)
    if s is None:
        return []
    cdef tuple tag = (comp_name, render_version)
    cdef object cached = frame_cache.get(tag)
    if cached is not None:
        return cached
    cdef list result = []
    cdef object eid
    cdef object ent
    for eid in s:
        ent = entities.get(eid)
        if ent is not None:
            result.append(ent)
    frame_cache[tag] = result
    if len(frame_cache) > 256:
        frame_cache.clear()
        frame_cache[tag] = result
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_get_transforms(list entities):
    cdef int n = len(entities)
    if n == 0:
        return []
    cdef list out = [None]*n
    cdef int i
    cdef object e, tt, lst
    for i in range(n):
        e = entities[i]
        tt = e._transform_type
        if tt is not None:
            lst = e._type_map.get(tt)
            if lst is not None and len(lst) > 0:
                out[i] = lst[0]
                continue
        out[i] = e.transform
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_compute_world_aabbs(list entities, list meshes, list world_mats):
    cdef int n = len(entities)
    if n == 0:
        return np.zeros((0, 6), dtype=np.float32)
    cdef np.ndarray[FTYPE_t, ndim=2] out = np.empty((n, 6), dtype=np.float32)
    cdef int i, k
    cdef object mesh, wm
    cdef DTYPE_t[:, :] d
    cdef FTYPE_t ax, ay, az, bx, by, bz
    cdef FTYPE_t c0x, c0y, c0z, c1x, c1y, c1z, c2x, c2y, c2z, c3x, c3y, c3z
    cdef FTYPE_t c4x, c4y, c4z, c5x, c5y, c5z, c6x, c6y, c6z, c7x, c7y, c7z
    cdef FTYPE_t minx, miny, minz, maxx, maxy, maxz, x, y, z, w
    for i in range(n):
        mesh = meshes[i]
        wm = world_mats[i]
        d = wm._d
        ax = <FTYPE_t>mesh.aabb_min[0]
        ay = <FTYPE_t>mesh.aabb_min[1]
        az = <FTYPE_t>mesh.aabb_min[2]
        bx = <FTYPE_t>mesh.aabb_max[0]
        by = <FTYPE_t>mesh.aabb_max[1]
        bz = <FTYPE_t>mesh.aabb_max[2]
        c0x = <FTYPE_t>(ax*d[0,0] + ay*d[1,0] + az*d[2,0] + d[3,0])
        c0y = <FTYPE_t>(ax*d[0,1] + ay*d[1,1] + az*d[2,1] + d[3,1])
        c0z = <FTYPE_t>(ax*d[0,2] + ay*d[1,2] + az*d[2,2] + d[3,2])
        minx = c0x; maxx = c0x; miny = c0y; maxy = c0y; minz = c0z; maxz = c0z
        c1x = <FTYPE_t>(bx*d[0,0] + ay*d[1,0] + az*d[2,0] + d[3,0])
        c1y = <FTYPE_t>(bx*d[0,1] + ay*d[1,1] + az*d[2,1] + d[3,1])
        c1z = <FTYPE_t>(bx*d[0,2] + ay*d[1,2] + az*d[2,2] + d[3,2])
        if c1x < minx: minx = c1x
        if c1x > maxx: maxx = c1x
        if c1y < miny: miny = c1y
        if c1y > maxy: maxy = c1y
        if c1z < minz: minz = c1z
        if c1z > maxz: maxz = c1z
        c2x = <FTYPE_t>(bx*d[0,0] + by*d[1,0] + az*d[2,0] + d[3,0])
        c2y = <FTYPE_t>(bx*d[0,1] + by*d[1,1] + az*d[2,1] + d[3,1])
        c2z = <FTYPE_t>(bx*d[0,2] + by*d[1,2] + az*d[2,2] + d[3,2])
        if c2x < minx: minx = c2x
        if c2x > maxx: maxx = c2x
        if c2y < miny: miny = c2y
        if c2y > maxy: maxy = c2y
        if c2z < minz: minz = c2z
        if c2z > maxz: maxz = c2z
        c3x = <FTYPE_t>(ax*d[0,0] + by*d[1,0] + az*d[2,0] + d[3,0])
        c3y = <FTYPE_t>(ax*d[0,1] + by*d[1,1] + az*d[2,1] + d[3,1])
        c3z = <FTYPE_t>(ax*d[0,2] + by*d[1,2] + az*d[2,2] + d[3,2])
        if c3x < minx: minx = c3x
        if c3x > maxx: maxx = c3x
        if c3y < miny: miny = c3y
        if c3y > maxy: maxy = c3y
        if c3z < minz: minz = c3z
        if c3z > maxz: maxz = c3z
        c4x = <FTYPE_t>(ax*d[0,0] + ay*d[1,0] + bz*d[2,0] + d[3,0])
        c4y = <FTYPE_t>(ax*d[0,1] + ay*d[1,1] + bz*d[2,1] + d[3,1])
        c4z = <FTYPE_t>(ax*d[0,2] + ay*d[1,2] + bz*d[2,2] + d[3,2])
        if c4x < minx: minx = c4x
        if c4x > maxx: maxx = c4x
        if c4y < miny: miny = c4y
        if c4y > maxy: maxy = c4y
        if c4z < minz: minz = c4z
        if c4z > maxz: maxz = c4z
        c5x = <FTYPE_t>(bx*d[0,0] + ay*d[1,0] + bz*d[2,0] + d[3,0])
        c5y = <FTYPE_t>(bx*d[0,1] + ay*d[1,1] + bz*d[2,1] + d[3,1])
        c5z = <FTYPE_t>(bx*d[0,2] + ay*d[1,2] + bz*d[2,2] + d[3,2])
        if c5x < minx: minx = c5x
        if c5x > maxx: maxx = c5x
        if c5y < miny: miny = c5y
        if c5y > maxy: maxy = c5y
        if c5z < minz: minz = c5z
        if c5z > maxz: maxz = c5z
        c6x = <FTYPE_t>(bx*d[0,0] + by*d[1,0] + bz*d[2,0] + d[3,0])
        c6y = <FTYPE_t>(bx*d[0,1] + by*d[1,1] + bz*d[2,1] + d[3,1])
        c6z = <FTYPE_t>(bx*d[0,2] + by*d[1,2] + bz*d[2,2] + d[3,2])
        if c6x < minx: minx = c6x
        if c6x > maxx: maxx = c6x
        if c6y < miny: miny = c6y
        if c6y > maxy: maxy = c6y
        if c6z < minz: minz = c6z
        if c6z > maxz: maxz = c6z
        c7x = <FTYPE_t>(ax*d[0,0] + by*d[1,0] + bz*d[2,0] + d[3,0])
        c7y = <FTYPE_t>(ax*d[0,1] + by*d[1,1] + bz*d[2,1] + d[3,1])
        c7z = <FTYPE_t>(ax*d[0,2] + by*d[1,2] + bz*d[2,2] + d[3,2])
        if c7x < minx: minx = c7x
        if c7x > maxx: maxx = c7x
        if c7y < miny: miny = c7y
        if c7y > maxy: maxy = c7y
        if c7z < minz: minz = c7z
        if c7z > maxz: maxz = c7z
        out[i, 0] = minx; out[i, 1] = miny; out[i, 2] = minz
        out[i, 3] = maxx; out[i, 4] = maxy; out[i, 5] = maxz
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_extract_frustum_planes(np.ndarray[DTYPE_t, ndim=2] view_proj):
    cdef np.ndarray[DTYPE_t, ndim=2] planes = np.empty((6, 4), dtype=np.float64)
    cdef int i
    cdef DTYPE_t inv
    for i in range(6):
        if i == 0:
            planes[i, 0] = view_proj[3, 0] + view_proj[0, 0]
            planes[i, 1] = view_proj[3, 1] + view_proj[0, 1]
            planes[i, 2] = view_proj[3, 2] + view_proj[0, 2]
            planes[i, 3] = view_proj[3, 3] + view_proj[0, 3]
        elif i == 1:
            planes[i, 0] = view_proj[3, 0] - view_proj[0, 0]
            planes[i, 1] = view_proj[3, 1] - view_proj[0, 1]
            planes[i, 2] = view_proj[3, 2] - view_proj[0, 2]
            planes[i, 3] = view_proj[3, 3] - view_proj[0, 3]
        elif i == 2:
            planes[i, 0] = view_proj[3, 0] + view_proj[1, 0]
            planes[i, 1] = view_proj[3, 1] + view_proj[1, 1]
            planes[i, 2] = view_proj[3, 2] + view_proj[1, 2]
            planes[i, 3] = view_proj[3, 3] + view_proj[1, 3]
        elif i == 3:
            planes[i, 0] = view_proj[3, 0] - view_proj[1, 0]
            planes[i, 1] = view_proj[3, 1] - view_proj[1, 1]
            planes[i, 2] = view_proj[3, 2] - view_proj[1, 2]
            planes[i, 3] = view_proj[3, 3] - view_proj[1, 3]
        elif i == 4:
            planes[i, 0] = view_proj[3, 0] + view_proj[2, 0]
            planes[i, 1] = view_proj[3, 1] + view_proj[2, 1]
            planes[i, 2] = view_proj[3, 2] + view_proj[2, 2]
            planes[i, 3] = view_proj[3, 3] + view_proj[2, 3]
        else:
            planes[i, 0] = view_proj[3, 0] - view_proj[2, 0]
            planes[i, 1] = view_proj[3, 1] - view_proj[2, 1]
            planes[i, 2] = view_proj[3, 2] - view_proj[2, 2]
            planes[i, 3] = view_proj[3, 3] - view_proj[2, 3]
        inv = 1.0 / sqrt(planes[i,0]*planes[i,0] + planes[i,1]*planes[i,1] + planes[i,2]*planes[i,2])
        planes[i, 0] *= inv; planes[i, 1] *= inv; planes[i, 2] *= inv; planes[i, 3] *= inv
    return planes

def batch_dot3(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] b):
    cdef int n = a.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef int i
    with nogil:
        for i in prange(n, schedule='static'):
            out[i] = a[i,0]*b[i,0] + a[i,1]*b[i,1] + a[i,2]*b[i,2]
    return out
