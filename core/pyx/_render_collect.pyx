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
def pack_world_matrices_f32(list matrices):
    cdef int n = len(matrices)
    if n == 0:
        return np.zeros((0, 16), dtype=np.float32)
    cdef np.ndarray[FTYPE_t, ndim=2] out = np.empty((n, 16), dtype=np.float32)
    cdef int i, r, c, idx
    cdef object wm
    cdef DTYPE_t[:, :] d
    for i in range(n):
        wm = matrices[i]
        d = wm._d
        idx = 0
        for r in range(4):
            for c in range(4):
                out[i, idx] = <FTYPE_t>d[r, c]
                idx += 1
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_lod_distances(np.ndarray[DTYPE_t, ndim=2] centers, DTYPE_t cam_x, DTYPE_t cam_y, DTYPE_t cam_z):
    cdef int n = centers.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef int i
    cdef DTYPE_t dx, dy, dz
    with nogil:
        for i in prange(n, schedule='static'):
            dx = centers[i, 0] - cam_x
            dy = centers[i, 1] - cam_y
            dz = centers[i, 2] - cam_z
            out[i] = sqrt(dx*dx + dy*dy + dz*dz)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def sort_by_distance(np.ndarray[DTYPE_t, ndim=1] dists, list entries):
    cdef int n = dists.shape[0]
    cdef np.ndarray[np.intp_t, ndim=1] idx = np.argsort(dists)
    cdef list out = [None]*n
    cdef int i
    for i in range(n):
        out[i] = entries[idx[i]]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_interpolate_vec3(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] b, DTYPE_t t):
    cdef int n = a.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((n, 3), dtype=np.float64)
    cdef int i
    cdef DTYPE_t it = 1.0 - t
    with nogil:
        for i in prange(n, schedule='static'):
            out[i, 0] = a[i,0]*it + b[i,0]*t
            out[i, 1] = a[i,1]*it + b[i,1]*t
            out[i, 2] = a[i,2]*it + b[i,2]*t
    return out
