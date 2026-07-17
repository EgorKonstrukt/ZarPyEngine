# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
from libc.math cimport sqrt, sin, cos, fabs, atan2, asin, acos

ctypedef double DTYPE_t

cdef inline DTYPE_t _dot3(DTYPE_t ax, DTYPE_t ay, DTYPE_t az,
                           DTYPE_t bx, DTYPE_t by, DTYPE_t bz) noexcept nogil:
    return ax*bx + ay*by + az*bz

cdef inline void _cross3(DTYPE_t ax, DTYPE_t ay, DTYPE_t az,
                          DTYPE_t bx, DTYPE_t by, DTYPE_t bz,
                          DTYPE_t* rx, DTYPE_t* ry, DTYPE_t* rz) noexcept nogil:
    rx[0] = ay*bz - az*by
    ry[0] = az*bx - ax*bz
    rz[0] = ax*by - ay*bx

cdef inline void _normalize3(DTYPE_t x, DTYPE_t y, DTYPE_t z,
                              DTYPE_t* rx, DTYPE_t* ry, DTYPE_t* rz) noexcept nogil:
    cdef DTYPE_t l = sqrt(x*x + y*y + z*z)
    if l > 1e-10:
        l = 1.0 / l
        rx[0] = x*l; ry[0] = y*l; rz[0] = z*l
    else:
        rx[0] = 0.0; ry[0] = 0.0; rz[0] = 0.0

cdef inline void _mat4_mul_flat(const DTYPE_t* l, const DTYPE_t* r,
                                 DTYPE_t* o) noexcept nogil:
    o[0]  = l[0]*r[0]  + l[1]*r[4]  + l[2]*r[8]  + l[3]*r[12]
    o[1]  = l[0]*r[1]  + l[1]*r[5]  + l[2]*r[9]  + l[3]*r[13]
    o[2]  = l[0]*r[2]  + l[1]*r[6]  + l[2]*r[10] + l[3]*r[14]
    o[3]  = l[0]*r[3]  + l[1]*r[7]  + l[2]*r[11] + l[3]*r[15]
    o[4]  = l[4]*r[0]  + l[5]*r[4]  + l[6]*r[8]  + l[7]*r[12]
    o[5]  = l[4]*r[1]  + l[5]*r[5]  + l[6]*r[9]  + l[7]*r[13]
    o[6]  = l[4]*r[2]  + l[5]*r[6]  + l[6]*r[10] + l[7]*r[14]
    o[7]  = l[4]*r[3]  + l[5]*r[7]  + l[6]*r[11] + l[7]*r[15]
    o[8]  = l[8]*r[0]  + l[9]*r[4]  + l[10]*r[8] + l[11]*r[12]
    o[9]  = l[8]*r[1]  + l[9]*r[5]  + l[10]*r[9] + l[11]*r[13]
    o[10] = l[8]*r[2]  + l[9]*r[6]  + l[10]*r[10]+ l[11]*r[14]
    o[11] = l[8]*r[3]  + l[9]*r[7]  + l[10]*r[11]+ l[11]*r[15]
    o[12] = l[12]*r[0] + l[13]*r[4] + l[14]*r[8] + l[15]*r[12]
    o[13] = l[12]*r[1] + l[13]*r[5] + l[14]*r[9] + l[15]*r[13]
    o[14] = l[12]*r[2] + l[13]*r[6] + l[14]*r[10]+ l[15]*r[14]
    o[15] = l[12]*r[3] + l[13]*r[7] + l[14]*r[11]+ l[15]*r[15]

cdef inline void _mat4_mul_vec_flat(const DTYPE_t* m,
                                     DTYPE_t x, DTYPE_t y, DTYPE_t z, DTYPE_t w,
                                     DTYPE_t* rx, DTYPE_t* ry, DTYPE_t* rz, DTYPE_t* rw) noexcept nogil:
    rx[0] = m[0]*x + m[4]*y + m[8]*z  + m[12]*w
    ry[0] = m[1]*x + m[5]*y + m[9]*z  + m[13]*w
    rz[0] = m[2]*x + m[6]*y + m[10]*z + m[14]*w
    rw[0] = m[3]*x + m[7]*y + m[11]*z + m[15]*w

cdef inline void _mat4_inv_flat(const DTYPE_t* m, DTYPE_t* o) noexcept nogil:
    cdef DTYPE_t m00=m[0], m01=m[1], m02=m[2], m03=m[3]
    cdef DTYPE_t m10=m[4], m11=m[5], m12=m[6], m13=m[7]
    cdef DTYPE_t m20=m[8], m21=m[9], m22=m[10], m23=m[11]
    cdef DTYPE_t m30=m[12], m31=m[13], m32=m[14], m33=m[15]
    cdef DTYPE_t t00 = m11*m22*m33 - m11*m23*m32 - m12*m21*m33 + m12*m23*m31 + m13*m21*m32 - m13*m22*m31
    cdef DTYPE_t t10 = -m10*m22*m33 + m10*m23*m32 + m12*m20*m33 - m12*m23*m30 - m13*m20*m32 + m13*m22*m30
    cdef DTYPE_t t20 = m10*m21*m33 - m10*m23*m31 - m11*m20*m33 + m11*m23*m30 + m13*m20*m31 - m13*m21*m30
    cdef DTYPE_t t30 = -m10*m21*m32 + m10*m22*m31 + m11*m20*m32 - m11*m22*m30 - m12*m20*m31 + m12*m21*m30
    cdef DTYPE_t det = m00*t00 + m01*t10 + m02*t20 + m03*t30
    if fabs(det) < 1e-15:
        o[0]=1; o[1]=0; o[2]=0; o[3]=0
        o[4]=0; o[5]=1; o[6]=0; o[7]=0
        o[8]=0; o[9]=0; o[10]=1; o[11]=0
        o[12]=0; o[13]=0; o[14]=0; o[15]=1
        return
    cdef DTYPE_t inv_det = 1.0 / det
    o[0]  = t00 * inv_det
    o[4]  = t10 * inv_det
    o[8]  = t20 * inv_det
    o[12] = t30 * inv_det
    o[1]  = (-m01*m22*m33 + m01*m23*m32 + m02*m21*m33 - m02*m23*m31 - m03*m21*m32 + m03*m22*m31) * inv_det
    o[5]  = ( m00*m22*m33 - m00*m23*m32 - m02*m20*m33 + m02*m23*m30 + m03*m20*m32 - m03*m22*m30) * inv_det
    o[9]  = (-m00*m21*m33 + m00*m23*m31 + m01*m20*m33 - m01*m23*m30 - m03*m20*m31 + m03*m21*m30) * inv_det
    o[13] = ( m00*m21*m32 - m00*m22*m31 - m01*m20*m32 + m01*m22*m30 + m02*m20*m31 - m02*m21*m30) * inv_det
    o[2]  = ( m01*m12*m33 - m01*m13*m32 - m02*m11*m33 + m02*m13*m31 + m03*m11*m32 - m03*m12*m31) * inv_det
    o[6]  = (-m00*m12*m33 + m00*m13*m32 + m02*m10*m33 - m02*m13*m30 - m03*m10*m32 + m03*m12*m30) * inv_det
    o[10] = ( m00*m11*m33 - m00*m13*m31 - m01*m10*m33 + m01*m13*m30 + m03*m10*m31 - m03*m11*m30) * inv_det
    o[14] = (-m00*m11*m32 + m00*m12*m31 + m01*m10*m32 - m01*m12*m30 - m02*m10*m31 + m02*m11*m30) * inv_det
    o[3]  = (-m01*m12*m23 + m01*m13*m22 + m02*m11*m23 - m02*m13*m21 - m03*m11*m22 + m03*m12*m21) * inv_det
    o[7]  = ( m00*m12*m23 - m00*m13*m22 - m02*m10*m23 + m02*m13*m20 + m03*m10*m22 - m03*m12*m20) * inv_det
    o[11] = (-m00*m11*m23 + m00*m13*m21 + m01*m10*m23 - m01*m13*m20 - m03*m10*m21 + m03*m11*m20) * inv_det
    o[15] = ( m00*m11*m22 - m00*m12*m21 - m01*m10*m22 + m01*m12*m20 + m02*m10*m21 - m02*m11*m20) * inv_det

cdef inline void _mat4_look_at_flat(DTYPE_t ex, DTYPE_t ey, DTYPE_t ez,
                                     DTYPE_t ax, DTYPE_t ay, DTYPE_t az,
                                     DTYPE_t ux, DTYPE_t uy, DTYPE_t uz,
                                     int right_handed,
                                     DTYPE_t* o) noexcept nogil:
    cdef DTYPE_t fx, fy, fz, rx, ry, rz, vx, vy, vz, zsign
    fx = ax - ex; fy = ay - ey; fz = az - ez
    _normalize3(fx, fy, fz, &fx, &fy, &fz)
    if right_handed:
        _cross3(fx, fy, fz, ux, uy, uz, &rx, &ry, &rz)
    else:
        _cross3(ux, uy, uz, fx, fy, fz, &rx, &ry, &rz)
    _normalize3(rx, ry, rz, &rx, &ry, &rz)
    if right_handed:
        _cross3(rx, ry, rz, fx, fy, fz, &vx, &vy, &vz)
    else:
        _cross3(fx, fy, fz, rx, ry, rz, &vx, &vy, &vz)
    zsign = -1.0 if right_handed else 1.0
    o[0] = rx; o[1] = vx; o[2] = zsign*fx; o[3] = 0.0
    o[4] = ry; o[5] = vy; o[6] = zsign*fy; o[7] = 0.0
    o[8] = rz; o[9] = vz; o[10] = zsign*fz; o[11] = 0.0
    o[12] = -_dot3(rx, ry, rz, ex, ey, ez)
    o[13] = -_dot3(vx, vy, vz, ex, ey, ez)
    o[14] = -zsign*_dot3(fx, fy, fz, ex, ey, ez)
    o[15] = 1.0

cdef inline void _rotate_around_axis_flat(DTYPE_t vx, DTYPE_t vy, DTYPE_t vz,
                                           DTYPE_t ax, DTYPE_t ay, DTYPE_t az,
                                           DTYPE_t angle,
                                           DTYPE_t* rx, DTYPE_t* ry, DTYPE_t* rz) noexcept nogil:
    cdef DTYPE_t nx, ny, nz, c, s, d
    _normalize3(ax, ay, az, &nx, &ny, &nz)
    c = cos(angle); s = sin(angle)
    d = _dot3(nx, ny, nz, vx, vy, vz)
    rx[0] = vx*c + (ny*vz - nz*vy)*s + nx*d*(1.0-c)
    ry[0] = vy*c + (nz*vx - nx*vz)*s + ny*d*(1.0-c)
    rz[0] = vz*c + (nx*vy - ny*vx)*s + nz*d*(1.0-c)


import numpy as np
cimport numpy as np

DTYPE = np.float64

def mat4_mul_flat(np.ndarray[DTYPE_t, ndim=1] l,
                  np.ndarray[DTYPE_t, ndim=1] r):
    cdef np.ndarray[DTYPE_t, ndim=1] o = np.empty(16, dtype=DTYPE)
    _mat4_mul_flat(&l[0], &r[0], &o[0])
    return o

def mat4_mul_vec_flat(np.ndarray[DTYPE_t, ndim=1] m,
                      DTYPE_t x, DTYPE_t y, DTYPE_t z, DTYPE_t w=1.0):
    cdef DTYPE_t rx, ry, rz, rw
    _mat4_mul_vec_flat(&m[0], x, y, z, w, &rx, &ry, &rz, &rw)
    return (rx, ry, rz, rw)

def mat4_invert_flat(np.ndarray[DTYPE_t, ndim=1] m):
    cdef np.ndarray[DTYPE_t, ndim=1] o = np.empty(16, dtype=DTYPE)
    _mat4_inv_flat(&m[0], &o[0])
    return o

def dot3_flat(DTYPE_t ax, DTYPE_t ay, DTYPE_t az,
              DTYPE_t bx, DTYPE_t by, DTYPE_t bz):
    return _dot3(ax, ay, az, bx, by, bz)

def cross3_flat(DTYPE_t ax, DTYPE_t ay, DTYPE_t az,
                DTYPE_t bx, DTYPE_t by, DTYPE_t bz):
    cdef DTYPE_t rx, ry, rz
    _cross3(ax, ay, az, bx, by, bz, &rx, &ry, &rz)
    return (rx, ry, rz)

def normalize3_flat(DTYPE_t x, DTYPE_t y, DTYPE_t z):
    cdef DTYPE_t rx, ry, rz
    _normalize3(x, y, z, &rx, &ry, &rz)
    return (rx, ry, rz)

def rotate_around_axis_flat(DTYPE_t vx, DTYPE_t vy, DTYPE_t vz,
                            DTYPE_t ax, DTYPE_t ay, DTYPE_t az,
                            DTYPE_t angle):
    cdef DTYPE_t rx, ry, rz
    _rotate_around_axis_flat(vx, vy, vz, ax, ay, az, angle, &rx, &ry, &rz)
    return (rx, ry, rz)

def look_at_flat(DTYPE_t ex, DTYPE_t ey, DTYPE_t ez,
                 DTYPE_t ax, DTYPE_t ay, DTYPE_t az,
                 DTYPE_t ux, DTYPE_t uy, DTYPE_t uz,
                 int cs):
    cdef np.ndarray[DTYPE_t, ndim=1] o = np.empty(16, dtype=DTYPE)
    cdef int rh = 1 if cs in (0, 1, 2) else 0
    _mat4_look_at_flat(ex, ey, ez, ax, ay, az, ux, uy, uz, rh, &o[0])
    return o

def point_in_circle(DTYPE_t cx, DTYPE_t cy, DTYPE_t radius,
                    DTYPE_t px, DTYPE_t py):
    cdef DTYPE_t dx = px - cx, dy = py - cy
    return dx*dx + dy*dy <= radius*radius


def mat4_mul_flat_batch(np.ndarray[DTYPE_t, ndim=2] L,
                        np.ndarray[DTYPE_t, ndim=2] R):
    cdef int n = L.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] O = np.empty((n, 16), dtype=DTYPE)
    cdef int i
    for i in range(n):
        _mat4_mul_flat(&L[i, 0], &R[i, 0], &O[i, 0])
    return O

def mat4_invert_flat_batch(np.ndarray[DTYPE_t, ndim=2] M):
    cdef int n = M.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] O = np.empty((n, 16), dtype=DTYPE)
    cdef int i
    for i in range(n):
        _mat4_inv_flat(&M[i, 0], &O[i, 0])
    return O

def mat4_mul_vec_flat_batch(np.ndarray[DTYPE_t, ndim=2] M,
                            np.ndarray[DTYPE_t, ndim=1] X,
                            np.ndarray[DTYPE_t, ndim=1] Y,
                            np.ndarray[DTYPE_t, ndim=1] Z,
                            np.ndarray[DTYPE_t, ndim=1] W):
    cdef int n = M.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] O = np.empty((n, 4), dtype=DTYPE)
    cdef int i
    cdef DTYPE_t rx, ry, rz, rw
    for i in range(n):
        _mat4_mul_vec_flat(&M[i, 0], X[i], Y[i], Z[i], W[i],
                           &rx, &ry, &rz, &rw)
        O[i, 0] = rx; O[i, 1] = ry; O[i, 2] = rz; O[i, 3] = rw
    return O

def mat4_to_f32_col_major_flat(np.ndarray[DTYPE_t, ndim=1] m):
    cdef np.ndarray[np.float32_t, ndim=1] o = np.empty(16, dtype=np.float32)
    cdef int i
    for i in range(4):
        o[i*4]     = <np.float32_t>m[i]
        o[i*4 + 1] = <np.float32_t>m[4 + i]
        o[i*4 + 2] = <np.float32_t>m[8 + i]
        o[i*4 + 3] = <np.float32_t>m[12 + i]
    return o

def batch_flat_to_f32_col_major(np.ndarray[DTYPE_t, ndim=2] M):
    cdef int n = M.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] O = np.empty((n, 16), dtype=np.float32)
    cdef int i, j
    for i in range(n):
        for j in range(4):
            O[i, j*4]     = <np.float32_t>M[i, j]
            O[i, j*4 + 1] = <np.float32_t>M[i, 4 + j]
            O[i, j*4 + 2] = <np.float32_t>M[i, 8 + j]
            O[i, j*4 + 3] = <np.float32_t>M[i, 12 + j]
    return O

def vec3_batch_normalize(np.ndarray[DTYPE_t, ndim=1] X,
                          np.ndarray[DTYPE_t, ndim=1] Y,
                          np.ndarray[DTYPE_t, ndim=1] Z):
    cdef int n = X.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] O = np.empty((n, 3), dtype=DTYPE)
    cdef int i
    cdef DTYPE_t rx, ry, rz
    for i in range(n):
        _normalize3(X[i], Y[i], Z[i], &rx, &ry, &rz)
        O[i, 0] = rx; O[i, 1] = ry; O[i, 2] = rz
    return O

def vec3_batch_cross(np.ndarray[DTYPE_t, ndim=1] AX,
                      np.ndarray[DTYPE_t, ndim=1] AY,
                      np.ndarray[DTYPE_t, ndim=1] AZ,
                      np.ndarray[DTYPE_t, ndim=1] BX,
                      np.ndarray[DTYPE_t, ndim=1] BY,
                      np.ndarray[DTYPE_t, ndim=1] BZ):
    cdef int n = AX.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] O = np.empty((n, 3), dtype=DTYPE)
    cdef int i
    cdef DTYPE_t rx, ry, rz
    for i in range(n):
        _cross3(AX[i], AY[i], AZ[i], BX[i], BY[i], BZ[i], &rx, &ry, &rz)
        O[i, 0] = rx; O[i, 1] = ry; O[i, 2] = rz
    return O

def vec3_batch_dot(np.ndarray[DTYPE_t, ndim=1] AX,
                    np.ndarray[DTYPE_t, ndim=1] AY,
                    np.ndarray[DTYPE_t, ndim=1] AZ,
                    np.ndarray[DTYPE_t, ndim=1] BX,
                    np.ndarray[DTYPE_t, ndim=1] BY,
                    np.ndarray[DTYPE_t, ndim=1] BZ):
    cdef int n = AX.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] O = np.empty(n, dtype=DTYPE)
    cdef int i
    for i in range(n):
        O[i] = _dot3(AX[i], AY[i], AZ[i], BX[i], BY[i], BZ[i])
    return O

def axis_vector_flat(int axis, DTYPE_t sign=1.0):
    if axis == 0:
        return (sign, 0.0, 0.0)
    if axis == 1:
        return (0.0, sign, 0.0)
    return (0.0, 0.0, sign)

def coordinate_system_axes(int cs):
    if cs == 1:
        return (1, 2, 0)
    if cs == 2:
        return (2, 0, 1)
    if cs == 3:
        return (0, 2, 1)
    if cs == 4:
        return (1, 0, 2)
    if cs == 5:
        return (2, 1, 0)
    return (0, 1, 2)

def is_right_handed_cs(int cs):
    return cs in (0, 1, 2)

def mat4_normal_batch(np.ndarray[DTYPE_t, ndim=3] matrices):
    cdef int n = matrices.shape[0]
    cdef np.ndarray[np.float32_t, ndim=3] out = np.empty((n, 3, 3), dtype=np.float32)
    cdef int i, r, c
    cdef DTYPE_t norm_val
    cdef DTYPE_t m00, m01, m02, m10, m11, m12, m20, m21, m22
    for i in range(n):
        m00 = matrices[i, 0, 0]; m01 = matrices[i, 0, 1]; m02 = matrices[i, 0, 2]
        m10 = matrices[i, 1, 0]; m11 = matrices[i, 1, 1]; m12 = matrices[i, 1, 2]
        m20 = matrices[i, 2, 0]; m21 = matrices[i, 2, 1]; m22 = matrices[i, 2, 2]
        norm_val = sqrt(m00*m00 + m10*m10 + m20*m20)
        if norm_val < 1e-10:
            norm_val = 1e-10
        m00 /= norm_val; m10 /= norm_val; m20 /= norm_val
        norm_val = sqrt(m01*m01 + m11*m11 + m21*m21)
        if norm_val < 1e-10:
            norm_val = 1e-10
        m01 /= norm_val; m11 /= norm_val; m21 /= norm_val
        norm_val = sqrt(m02*m02 + m12*m12 + m22*m22)
        if norm_val < 1e-10:
            norm_val = 1e-10
        m02 /= norm_val; m12 /= norm_val; m22 /= norm_val
        out[i, 0, 0] = <np.float32_t>m00
        out[i, 0, 1] = <np.float32_t>m10
        out[i, 0, 2] = <np.float32_t>m20
        out[i, 1, 0] = <np.float32_t>m01
        out[i, 1, 1] = <np.float32_t>m11
        out[i, 1, 2] = <np.float32_t>m21
        out[i, 2, 0] = <np.float32_t>m02
        out[i, 2, 1] = <np.float32_t>m12
        out[i, 2, 2] = <np.float32_t>m22
    return out
