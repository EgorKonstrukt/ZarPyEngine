# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, overflowcheck=False
# distutils: extra_compile_args = -O3 -ffast-math -march=native -fopenmp
# distutils: extra_link_args = -fopenmp
import numpy as np
cimport numpy as np
from cython.parallel import prange

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef inline void _mat4_mul(const DTYPE_t a[4][4], const DTYPE_t[:, :] b,
                            DTYPE_t out[4][4]) noexcept nogil:
    cdef int i
    cdef DTYPE_t a0, a1, a2, a3
    for i in range(4):
        a0 = a[i][0]; a1 = a[i][1]; a2 = a[i][2]; a3 = a[i][3]
        out[i][0] = a0*b[0,0] + a1*b[1,0] + a2*b[2,0] + a3*b[3,0]
        out[i][1] = a0*b[0,1] + a1*b[1,1] + a2*b[2,1] + a3*b[3,1]
        out[i][2] = a0*b[0,2] + a1*b[1,2] + a2*b[2,2] + a3*b[3,2]
        out[i][3] = a0*b[0,3] + a1*b[1,3] + a2*b[2,3] + a3*b[3,3]


cdef inline void _local_matrix(
    DTYPE_t tx, DTYPE_t ty, DTYPE_t tz,
    DTYPE_t rx, DTYPE_t ry, DTYPE_t rz, DTYPE_t rw,
    DTYPE_t sx, DTYPE_t sy, DTYPE_t sz,
    DTYPE_t out[4][4],
) noexcept nogil:
    cdef DTYPE_t rm[4][4], temp[4][4]
    cdef int i
    cdef DTYPE_t a0, a1, a2, a3
    cdef DTYPE_t x2 = rx + rx, y2 = ry + ry, z2 = rz + rz
    cdef DTYPE_t xx = rx * x2, xy = rx * y2, xz = rx * z2
    cdef DTYPE_t yy = ry * y2, yz = ry * z2, zz = rz * z2
    cdef DTYPE_t wx = rw * x2, wy = rw * y2, wz = rw * z2

    rm[0][0] = 1.0 - (yy + zz); rm[0][1] = xy + wz;      rm[0][2] = xz - wy;  rm[0][3] = 0.0
    rm[1][0] = xy - wz;         rm[1][1] = 1.0 - (xx + zz); rm[1][2] = yz + wx;  rm[1][3] = 0.0
    rm[2][0] = xz + wy;         rm[2][1] = yz - wx;         rm[2][2] = 1.0 - (xx + yy); rm[2][3] = 0.0
    rm[3][0] = 0.0; rm[3][1] = 0.0; rm[3][2] = 0.0; rm[3][3] = 1.0

    temp[0][0] = sx*rm[0][0]; temp[0][1] = sx*rm[0][1]; temp[0][2] = sx*rm[0][2]; temp[0][3] = 0.0
    temp[1][0] = sy*rm[1][0]; temp[1][1] = sy*rm[1][1]; temp[1][2] = sy*rm[1][2]; temp[1][3] = 0.0
    temp[2][0] = sz*rm[2][0]; temp[2][1] = sz*rm[2][1]; temp[2][2] = sz*rm[2][2]; temp[2][3] = 0.0
    temp[3][0] = 0.0; temp[3][1] = 0.0; temp[3][2] = 0.0; temp[3][3] = 1.0

    for i in range(4):
        a0 = temp[i][0]; a1 = temp[i][1]; a2 = temp[i][2]; a3 = temp[i][3]
        out[i][0] = a0 + a3*tx
        out[i][1] = a1 + a3*ty
        out[i][2] = a2 + a3*tz
        out[i][3] = a3


def batch_update_world_matrices(
    int n,
    np.ndarray[DTYPE_t, ndim=1] pos_x,
    np.ndarray[DTYPE_t, ndim=1] pos_y,
    np.ndarray[DTYPE_t, ndim=1] pos_z,
    np.ndarray[DTYPE_t, ndim=1] rot_x,
    np.ndarray[DTYPE_t, ndim=1] rot_y,
    np.ndarray[DTYPE_t, ndim=1] rot_z,
    np.ndarray[DTYPE_t, ndim=1] rot_w,
    np.ndarray[DTYPE_t, ndim=1] sc_x,
    np.ndarray[DTYPE_t, ndim=1] sc_y,
    np.ndarray[DTYPE_t, ndim=1] sc_z,
    np.ndarray[np.int32_t, ndim=1] has_parent,
    np.ndarray[np.int32_t, ndim=1] parent_idx,
    np.ndarray[DTYPE_t, ndim=3] parent_outside,
):
    cdef np.ndarray[DTYPE_t, ndim=3] world_mats = np.empty((n, 4, 4), dtype=DTYPE)
    cdef int i, pi, j, k
    cdef DTYPE_t local[4][4], result[4][4]

    for i in range(n):
        _local_matrix(
            pos_x[i], pos_y[i], pos_z[i],
            rot_x[i], rot_y[i], rot_z[i], rot_w[i],
            sc_x[i], sc_y[i], sc_z[i],
            local,
        )
        if not has_parent[i]:
            for j in range(4):
                for k in range(4):
                    world_mats[i, j, k] = local[j][k]
        elif parent_idx[i] >= 0:
            pi = parent_idx[i]
            _mat4_mul(local, world_mats[pi], result)
            for j in range(4):
                for k in range(4):
                    world_mats[i, j, k] = result[j][k]
        else:
            _mat4_mul(local, parent_outside[i], result)
            for j in range(4):
                for k in range(4):
                    world_mats[i, j, k] = result[j][k]

    return world_mats
