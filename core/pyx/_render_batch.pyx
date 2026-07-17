# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def build_instance_matrices(list entries):
    cdef int n = len(entries)
    if n == 0:
        return np.zeros((0, 4, 4), dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=3] out = np.empty((n, 4, 4), dtype=np.float32)
    cdef int i
    cdef object entry, wm, d
    cdef int j, k

    for i in range(n):
        entry = entries[i]
        wm = entry[4]
        d = wm._d
        for j in range(4):
            for k in range(4):
                out[i, j, k] = <np.float32_t>d[j, k]
    return out

def build_bounding_spheres_batch(list entries, np.ndarray[DTYPE_t, ndim=1] bounding_radii):
    cdef int n = len(entries)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=2] centers = np.empty((n, 3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] spheres = np.empty((n, 4), dtype=np.float32)
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
        spheres[i, 0] = <np.float32_t>d[3, 0]
        spheres[i, 1] = <np.float32_t>d[3, 1]
        spheres[i, 2] = <np.float32_t>d[3, 2]
        sx = sqrt(d[0, 0] * d[0, 0] + d[1, 0] * d[1, 0] + d[2, 0] * d[2, 0])
        sy = sqrt(d[0, 1] * d[0, 1] + d[1, 1] * d[1, 1] + d[2, 1] * d[2, 1])
        sz = sqrt(d[0, 2] * d[0, 2] + d[1, 2] * d[1, 2] + d[2, 2] * d[2, 2])
        ms = sx
        if sy > ms: ms = sy
        if sz > ms: ms = sz
        radii[i] = <np.float32_t>(ms * bounding_radii[i])
        spheres[i, 3] = radii[i]

    return spheres, centers, radii

def pack_instance_vbo(np.ndarray[np.float32_t, ndim=3] matrices):
    cdef int n = matrices.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] out = np.empty((n, 16), dtype=np.float32)
    cdef int i, j, k, flat_idx
    for i in range(n):
        flat_idx = 0
        for j in range(4):
            for k in range(4):
                out[i, flat_idx] = matrices[i, k, j]
                flat_idx += 1
    return out

def pack_instance_vbo_flat(np.ndarray[np.float32_t, ndim=3] matrices):
    cdef int n = matrices.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] out = np.empty(n * 16, dtype=np.float32)
    cdef int i, j, k, idx
    idx = 0
    for i in range(n):
        for j in range(4):
            for k in range(4):
                out[idx] = matrices[i, k, j]
                idx += 1
    return out

def compute_normal_matrices_batch(np.ndarray[DTYPE_t, ndim=3] model_matrices):
    cdef int n = model_matrices.shape[0]
    cdef np.ndarray[np.float32_t, ndim=3] out = np.empty((n, 3, 3), dtype=np.float32)
    cdef int i, r, c
    cdef DTYPE_t norm_val
    cdef DTYPE_t m00, m01, m02, m10, m11, m12, m20, m21, m22

    for i in range(n):
        m00 = model_matrices[i, 0, 0]; m01 = model_matrices[i, 0, 1]; m02 = model_matrices[i, 0, 2]
        m10 = model_matrices[i, 1, 0]; m11 = model_matrices[i, 1, 1]; m12 = model_matrices[i, 1, 2]
        m20 = model_matrices[i, 2, 0]; m21 = model_matrices[i, 2, 1]; m22 = model_matrices[i, 2, 2]

        norm_val = sqrt(m00*m00 + m10*m10 + m20*m20)
        if norm_val < 1e-10: norm_val = 1e-10
        m00 /= norm_val; m10 /= norm_val; m20 /= norm_val

        norm_val = sqrt(m01*m01 + m11*m11 + m21*m21)
        if norm_val < 1e-10: norm_val = 1e-10
        m01 /= norm_val; m11 /= norm_val; m21 /= norm_val

        norm_val = sqrt(m02*m02 + m12*m12 + m22*m22)
        if norm_val < 1e-10: norm_val = 1e-10
        m02 /= norm_val; m12 /= norm_val; m22 /= norm_val

        out[i, 0, 0] = <np.float32_t>m00
        out[i, 0, 1] = <np.float32_t>m01
        out[i, 0, 2] = <np.float32_t>m02
        out[i, 1, 0] = <np.float32_t>m10
        out[i, 1, 1] = <np.float32_t>m11
        out[i, 1, 2] = <np.float32_t>m12
        out[i, 2, 0] = <np.float32_t>m20
        out[i, 2, 1] = <np.float32_t>m21
        out[i, 2, 2] = <np.float32_t>m22

    return out
