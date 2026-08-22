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
from libc.math cimport sqrt, sin, cos, tan
from cython.parallel import prange
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef inline void _mat4_mul_nogil(const DTYPE_t a[4][4], const DTYPE_t b[4][4], DTYPE_t out[4][4]) noexcept nogil:
    cdef int i, j, k
    for i in range(4):
        for j in range(4):
            out[i][j] = 0
            for k in range(4):
                out[i][j] += a[i][k] * b[k][j]

cdef inline void _mat4_transpose_nogil(const DTYPE_t a[4][4], DTYPE_t out[4][4]) noexcept nogil:
    cdef int i, j
    for i in range(4):
        for j in range(4):
            out[i][j] = a[j][i]

def mat4_identity_f32():
    return np.eye(4, dtype=np.float32)

def mat4_batch_mul(np.ndarray[DTYPE_t, ndim=3] A, np.ndarray[DTYPE_t, ndim=3] B):
    cdef int n = A.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] out = np.empty((n, 4, 4), dtype=np.float64)
    cdef int i, r, c, k
    for i in range(n):
        for r in range(4):
            for c in range(4):
                out[i, r, c] = A[i, r, 0]*B[i, 0, c] + A[i, r, 1]*B[i, 1, c] + A[i, r, 2]*B[i, 2, c] + A[i, r, 3]*B[i, 3, c]
    return out

def mat4_batch_transpose(np.ndarray[DTYPE_t, ndim=3] A):
    cdef int n = A.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] out = np.empty((n, 4, 4), dtype=np.float64)
    cdef int i, r, c
    with nogil:
        for i in prange(n, schedule='static'):
            for r in range(4):
                for c in range(4):
                    out[i, r, c] = A[i, c, r]
    return out

def mat4_batch_perspective(np.ndarray[DTYPE_t, ndim=1] fovs, np.ndarray[DTYPE_t, ndim=1] aspects, DTYPE_t near, DTYPE_t far):
    cdef int n = fovs.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] out = np.zeros((n, 4, 4), dtype=np.float64)
    cdef int i
    cdef DTYPE_t f, nf
    with nogil:
        for i in prange(n, schedule='static'):
            f = 1.0 / tan(fovs[i] * 0.5)
            nf = 1.0 / (near - far)
            out[i, 0, 0] = f / aspects[i]
            out[i, 1, 1] = f
            out[i, 2, 2] = (far + near) * nf
            out[i, 2, 3] = -1.0
            out[i, 3, 2] = 2.0 * far * near * nf
    return out

def mat4_batch_inverse(np.ndarray[DTYPE_t, ndim=3] M):
    cdef int n = M.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] out = np.empty((n, 4, 4), dtype=np.float64)
    cdef int i
    cdef DTYPE_t m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33
    cdef DTYPE_t t00, t10, t20, t30, det, inv
    for i in range(n):
        m00 = M[i,0,0]; m01 = M[i,0,1]; m02 = M[i,0,2]; m03 = M[i,0,3]
        m10 = M[i,1,0]; m11 = M[i,1,1]; m12 = M[i,1,2]; m13 = M[i,1,3]
        m20 = M[i,2,0]; m21 = M[i,2,1]; m22 = M[i,2,2]; m23 = M[i,2,3]
        m30 = M[i,3,0]; m31 = M[i,3,1]; m32 = M[i,3,2]; m33 = M[i,3,3]
        t00 = m11*m22*m33 - m11*m23*m32 - m12*m21*m33 + m12*m23*m31 + m13*m21*m32 - m13*m22*m31
        t10 = -m10*m22*m33 + m10*m23*m32 + m12*m20*m33 - m12*m23*m30 - m13*m20*m32 + m13*m22*m30
        t20 = m10*m21*m33 - m10*m23*m31 - m11*m20*m33 + m11*m23*m30 + m13*m20*m31 - m13*m21*m30
        t30 = -m10*m21*m32 + m10*m22*m31 + m11*m20*m32 - m11*m22*m30 - m12*m20*m31 + m12*m21*m30
        det = m00*t00 + m01*t10 + m02*t20 + m03*t30
        if det == 0:
            out[i] = np.eye(4, dtype=np.float64)
            continue
        inv = 1.0/det
        out[i,0,0] = t00*inv
        out[i,1,0] = t10*inv
        out[i,2,0] = t20*inv
        out[i,3,0] = t30*inv
        out[i,0,1] = (-m01*m22*m33 + m01*m23*m32 + m02*m21*m33 - m02*m23*m31 - m03*m21*m32 + m03*m22*m31)*inv
        out[i,1,1] = ( m00*m22*m33 - m00*m23*m32 - m02*m20*m33 + m02*m23*m30 + m03*m20*m32 - m03*m22*m30)*inv
        out[i,2,1] = (-m00*m21*m33 + m00*m23*m31 + m01*m20*m33 - m01*m23*m30 - m03*m20*m31 + m03*m21*m30)*inv
        out[i,3,1] = ( m00*m21*m32 - m00*m22*m31 - m01*m20*m32 + m01*m22*m30 + m02*m20*m31 - m02*m21*m30)*inv
        out[i,0,2] = ( m01*m12*m33 - m01*m13*m32 - m02*m11*m33 + m02*m13*m31 + m03*m11*m32 - m03*m12*m31)*inv
        out[i,1,2] = (-m00*m12*m33 + m00*m13*m32 + m02*m10*m33 - m02*m13*m30 - m03*m10*m32 + m03*m12*m30)*inv
        out[i,2,2] = ( m00*m11*m33 - m00*m13*m31 - m01*m10*m33 + m01*m13*m30 + m03*m10*m31 - m03*m11*m30)*inv
        out[i,3,2] = (-m00*m11*m32 + m00*m12*m31 + m01*m10*m32 - m01*m12*m30 - m02*m10*m31 + m02*m11*m30)*inv
        out[i,0,3] = (-m01*m12*m23 + m01*m13*m22 + m02*m11*m23 - m02*m13*m21 - m03*m11*m22 + m03*m12*m21)*inv
        out[i,1,3] = ( m00*m12*m23 - m00*m13*m22 - m02*m10*m23 + m02*m13*m20 + m03*m10*m22 - m03*m12*m20)*inv
        out[i,2,3] = (-m00*m11*m23 + m00*m13*m21 + m01*m10*m23 - m01*m13*m20 - m03*m10*m21 + m03*m11*m20)*inv
        out[i,3,3] = ( m00*m11*m22 - m00*m12*m21 - m01*m10*m22 + m01*m12*m20 + m02*m10*m21 - m02*m11*m20)*inv
    return out
