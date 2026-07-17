# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def compute_bounding_spheres(list matrices, np.ndarray[DTYPE_t, ndim=1] bounding_radii):
    """Build (n,4) float32 sphere data for GPU culling.

    Each sphere: x, y, z = center, w = radius.
    """
    cdef int n = len(matrices)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=2] spheres = np.empty((n, 4), dtype=np.float32)
    cdef int i
    cdef object wm, _d

    for i in range(n):
        wm = matrices[i]
        _d = wm._d
        spheres[i, 0] = <np.float32_t>_d[3, 0]
        spheres[i, 1] = <np.float32_t>_d[3, 1]
        spheres[i, 2] = <np.float32_t>_d[3, 2]
        spheres[i, 3] = <np.float32_t>bounding_radii[i]

    return spheres


def build_frustum_cull_inputs(list entries, np.ndarray[DTYPE_t, ndim=1] bounding_radii):
    cdef int n = len(entries)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=2] centers = np.empty((n, 3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] radii = np.empty(n, dtype=np.float32)
    cdef int i
    cdef double sx, sy, sz, ms
    cdef object entry, wm, d

    for i in range(n):
        entry = entries[i]
        wm = entry[4]
        d = wm._d
        centers[i, 0] = <np.float32_t>d[3, 0]
        centers[i, 1] = <np.float32_t>d[3, 1]
        centers[i, 2] = <np.float32_t>d[3, 2]
        sx = sqrt(d[0, 0] * d[0, 0] + d[1, 0] * d[1, 0] + d[2, 0] * d[2, 0])
        sy = sqrt(d[0, 1] * d[0, 1] + d[1, 1] * d[1, 1] + d[2, 1] * d[2, 1])
        sz = sqrt(d[0, 2] * d[0, 2] + d[1, 2] * d[1, 2] + d[2, 2] * d[2, 2])
        ms = sx
        if sy > ms:
            ms = sy
        if sz > ms:
            ms = sz
        radii[i] = <np.float32_t>(ms * bounding_radii[i])

    return centers, radii


def batch_mat4_to_f32(list matrices):
    cdef int n = len(matrices)
    if n == 0:
        return np.zeros((0, 4, 4), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] out = np.empty((n, 4, 4), dtype=np.float32)
    cdef int i
    cdef object wm, d
    for i in range(n):
        wm = matrices[i]
        d = wm._d
        out[i, 0, 0] = <np.float32_t>d[0, 0]
        out[i, 0, 1] = <np.float32_t>d[0, 1]
        out[i, 0, 2] = <np.float32_t>d[0, 2]
        out[i, 0, 3] = <np.float32_t>d[0, 3]
        out[i, 1, 0] = <np.float32_t>d[1, 0]
        out[i, 1, 1] = <np.float32_t>d[1, 1]
        out[i, 1, 2] = <np.float32_t>d[1, 2]
        out[i, 1, 3] = <np.float32_t>d[1, 3]
        out[i, 2, 0] = <np.float32_t>d[2, 0]
        out[i, 2, 1] = <np.float32_t>d[2, 1]
        out[i, 2, 2] = <np.float32_t>d[2, 2]
        out[i, 2, 3] = <np.float32_t>d[2, 3]
        out[i, 3, 0] = <np.float32_t>d[3, 0]
        out[i, 3, 1] = <np.float32_t>d[3, 1]
        out[i, 3, 2] = <np.float32_t>d[3, 2]
        out[i, 3, 3] = <np.float32_t>d[3, 3]
    return out
