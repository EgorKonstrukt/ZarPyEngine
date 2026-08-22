# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, overflowcheck=False
# distutils: extra_compile_args = -O3 -ffast-math -march=native -fopenmp
# distutils: extra_link_args = -fopenmp
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, sin, cos, tan, acos, fabs, atan2, asin
from cython.parallel import prange

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef DTYPE_t _PI = 3.14159265358979323846
cdef DTYPE_t _DEG2RAD = _PI / 180.0
cdef DTYPE_t _RAD2DEG = 180.0 / _PI
ctypedef np.int32_t INT32_t
ctypedef np.uint8_t UINT8_t


# ── Pure-scalar internal helpers (nogil) ──────────────────────────────

cdef inline void _mat4_mul(const DTYPE_t[:, :] a, const DTYPE_t[:, :] b,
                            DTYPE_t[:, :] r) noexcept nogil:
    cdef int i
    cdef DTYPE_t a0, a1, a2, a3
    for i in range(4):
        a0 = a[i, 0]; a1 = a[i, 1]; a2 = a[i, 2]; a3 = a[i, 3]
        r[i, 0] = a0*b[0,0] + a1*b[1,0] + a2*b[2,0] + a3*b[3,0]
        r[i, 1] = a0*b[0,1] + a1*b[1,1] + a2*b[2,1] + a3*b[3,1]
        r[i, 2] = a0*b[0,2] + a1*b[1,2] + a2*b[2,2] + a3*b[3,2]
        r[i, 3] = a0*b[0,3] + a1*b[1,3] + a2*b[2,3] + a3*b[3,3]


cdef inline void _mat4_inv(const DTYPE_t[:, :] m,
                            DTYPE_t[:, :] inv) noexcept nogil:
    cdef DTYPE_t m00 = m[0,0], m01 = m[0,1], m02 = m[0,2], m03 = m[0,3]
    cdef DTYPE_t m10 = m[1,0], m11 = m[1,1], m12 = m[1,2], m13 = m[1,3]
    cdef DTYPE_t m20 = m[2,0], m21 = m[2,1], m22 = m[2,2], m23 = m[2,3]
    cdef DTYPE_t m30 = m[3,0], m31 = m[3,1], m32 = m[3,2], m33 = m[3,3]
    cdef DTYPE_t t00 = m11*m22*m33 - m11*m23*m32 - m12*m21*m33 + m12*m23*m31 + m13*m21*m32 - m13*m22*m31
    cdef DTYPE_t t10 = -m10*m22*m33 + m10*m23*m32 + m12*m20*m33 - m12*m23*m30 - m13*m20*m32 + m13*m22*m30
    cdef DTYPE_t t20 = m10*m21*m33 - m10*m23*m31 - m11*m20*m33 + m11*m23*m30 + m13*m20*m31 - m13*m21*m30
    cdef DTYPE_t t30 = -m10*m21*m32 + m10*m22*m31 + m11*m20*m32 - m11*m22*m30 - m12*m20*m31 + m12*m21*m30
    cdef DTYPE_t det = m00*t00 + m01*t10 + m02*t20 + m03*t30
    if fabs(det) < 1e-15:
        inv[0,0]=1; inv[0,1]=0; inv[0,2]=0; inv[0,3]=0
        inv[1,0]=0; inv[1,1]=1; inv[1,2]=0; inv[1,3]=0
        inv[2,0]=0; inv[2,1]=0; inv[2,2]=1; inv[2,3]=0
        inv[3,0]=0; inv[3,1]=0; inv[3,2]=0; inv[3,3]=1
        return
    cdef DTYPE_t inv_det = 1.0 / det
    inv[0,0] = t00 * inv_det
    inv[1,0] = t10 * inv_det
    inv[2,0] = t20 * inv_det
    inv[3,0] = t30 * inv_det
    inv[0,1] = (-m01*m22*m33 + m01*m23*m32 + m02*m21*m33 - m02*m23*m31 - m03*m21*m32 + m03*m22*m31) * inv_det
    inv[1,1] = ( m00*m22*m33 - m00*m23*m32 - m02*m20*m33 + m02*m23*m30 + m03*m20*m32 - m03*m22*m30) * inv_det
    inv[2,1] = (-m00*m21*m33 + m00*m23*m31 + m01*m20*m33 - m01*m23*m30 - m03*m20*m31 + m03*m21*m30) * inv_det
    inv[3,1] = ( m00*m21*m32 - m00*m22*m31 - m01*m20*m32 + m01*m22*m30 + m02*m20*m31 - m02*m21*m30) * inv_det
    inv[0,2] = ( m01*m12*m33 - m01*m13*m32 - m02*m11*m33 + m02*m13*m31 + m03*m11*m32 - m03*m12*m31) * inv_det
    inv[1,2] = (-m00*m12*m33 + m00*m13*m32 + m02*m10*m33 - m02*m13*m30 - m03*m10*m32 + m03*m12*m30) * inv_det
    inv[2,2] = ( m00*m11*m33 - m00*m13*m31 - m01*m10*m33 + m01*m13*m30 + m03*m10*m31 - m03*m11*m30) * inv_det
    inv[3,2] = (-m00*m11*m32 + m00*m12*m31 + m01*m10*m32 - m01*m12*m30 - m02*m10*m31 + m02*m11*m30) * inv_det
    inv[0,3] = (-m01*m12*m23 + m01*m13*m22 + m02*m11*m23 - m02*m13*m21 - m03*m11*m22 + m03*m12*m21) * inv_det
    inv[1,3] = ( m00*m12*m23 - m00*m13*m22 - m02*m10*m23 + m02*m13*m20 + m03*m10*m22 - m03*m12*m20) * inv_det
    inv[2,3] = (-m00*m11*m23 + m00*m13*m21 + m01*m10*m23 - m01*m13*m20 - m03*m10*m21 + m03*m11*m20) * inv_det
    inv[3,3] = ( m00*m11*m22 - m00*m12*m21 - m01*m10*m22 + m01*m12*m20 + m02*m10*m21 - m02*m11*m20) * inv_det


# ── Public 4×4 matrix functions ───────────────────────────────────────

def mat4_mul_fast(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((4, 4), dtype=DTYPE)
    _mat4_mul(a, b, out)
    return out


def mat4_inv_fast(m: np.ndarray) -> np.ndarray:
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((4, 4), dtype=DTYPE)
    _mat4_inv(m, out)
    return out


def mat4_translation(pos_x: float, pos_y: float, pos_z: float) -> np.ndarray:
    cdef np.ndarray[DTYPE_t, ndim=2] m = np.eye(4, dtype=DTYPE)
    m[3, 0] = pos_x
    m[3, 1] = pos_y
    m[3, 2] = pos_z
    return m


def mat4_scale_mat(sx: float, sy: float, sz: float) -> np.ndarray:
    cdef np.ndarray[DTYPE_t, ndim=2] m = np.eye(4, dtype=DTYPE)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def mat4_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    cdef np.ndarray[DTYPE_t, ndim=2] m = np.eye(4, dtype=DTYPE)
    cdef DTYPE_t x2 = x + x, y2 = y + y, z2 = z + z
    cdef DTYPE_t xx = x * x2, xy = x * y2, xz = x * z2
    cdef DTYPE_t yy = y * y2, yz = y * z2, zz = z * z2
    cdef DTYPE_t wx = w * x2, wy = w * y2, wz = w * z2
    m[0, 0] = 1.0 - (yy + zz)
    m[0, 1] = xy + wz
    m[0, 2] = xz - wy
    m[1, 0] = xy - wz
    m[1, 1] = 1.0 - (xx + zz)
    m[1, 2] = yz + wx
    m[2, 0] = xz + wy
    m[2, 1] = yz - wx
    m[2, 2] = 1.0 - (xx + yy)
    return m


def mat4_to_f32_col_major(np.ndarray[DTYPE_t, ndim=2] m) -> np.ndarray:
    cdef np.ndarray[np.float32_t, ndim=1] r = np.empty(16, dtype=np.float32)
    cdef int i
    for i in range(4):
        r[i*4]     = <np.float32_t>m[0, i]
        r[i*4 + 1] = <np.float32_t>m[1, i]
        r[i*4 + 2] = <np.float32_t>m[2, i]
        r[i*4 + 3] = <np.float32_t>m[3, i]
    return r


def mat4_mul_vec3(np.ndarray[DTYPE_t, ndim=2] m,
                  float vx, float vy, float vz):
    return (
        m[0,0]*vx + m[1,0]*vy + m[2,0]*vz + m[3,0],
        m[0,1]*vx + m[1,1]*vy + m[2,1]*vz + m[3,1],
        m[0,2]*vx + m[1,2]*vy + m[2,2]*vz + m[3,2],
    )


def mat4_look_at(float eye_x, float eye_y, float eye_z,
                  float center_x, float center_y, float center_z,
                  float up_x, float up_y, float up_z) -> np.ndarray:
    cdef DTYPE_t fx = center_x - eye_x
    cdef DTYPE_t fy = center_y - eye_y
    cdef DTYPE_t fz = center_z - eye_z
    cdef DTYPE_t flen = sqrt(fx*fx + fy*fy + fz*fz)
    if flen > 1e-10:
        fx /= flen; fy /= flen; fz /= flen
    cdef DTYPE_t rx = up_y*fz - up_z*fy
    cdef DTYPE_t ry = up_z*fx - up_x*fz
    cdef DTYPE_t rz = up_x*fy - up_y*fx
    cdef DTYPE_t rlen = sqrt(rx*rx + ry*ry + rz*rz)
    if rlen > 1e-10:
        rx /= rlen; ry /= rlen; rz /= rlen
    cdef DTYPE_t ux = fy*rz - fz*ry
    cdef DTYPE_t uy = fz*rx - fx*rz
    cdef DTYPE_t uz = fx*ry - fy*rx
    cdef np.ndarray[DTYPE_t, ndim=2] m = np.eye(4, dtype=DTYPE)
    m[0,0] = rx;  m[1,0] = ry;  m[2,0] = rz
    m[0,1] = ux;  m[1,1] = uy;  m[2,1] = uz
    m[0,2] = -fx; m[1,2] = -fy; m[2,2] = -fz
    m[3,0] = -(rx*eye_x + ry*eye_y + rz*eye_z)
    m[3,1] = -(ux*eye_x + uy*eye_y + uz*eye_z)
    m[3,2] = fx*eye_x + fy*eye_y + fz*eye_z
    return m


def mat4_perspective(float fov_rad, float aspect, float near, float far) -> np.ndarray:
    cdef DTYPE_t f = 1.0 / tan(fov_rad * 0.5)
    cdef DTYPE_t nf = 1.0 / (near - far)
    cdef np.ndarray[DTYPE_t, ndim=2] m = np.zeros((4, 4), dtype=DTYPE)
    m[0,0] = f / aspect
    m[1,1] = f
    m[2,2] = (far + near) * nf
    m[2,3] = -1.0
    m[3,2] = 2.0 * far * near * nf
    return m


# ── 3×3 matrix ────────────────────────────────────────────────────────

def mat3x3_inv(float m00, float m01, float m02,
                float m10, float m11, float m12,
                float m20, float m21, float m22) -> np.ndarray:
    cdef DTYPE_t det = (m00*(m11*m22 - m12*m21)
                        - m01*(m10*m22 - m12*m20)
                        + m02*(m10*m21 - m11*m20))
    cdef np.ndarray[DTYPE_t, ndim=2] r = np.empty((3, 3), dtype=DTYPE)
    if fabs(det) < 1e-15:
        r[0,0]=1; r[0,1]=0; r[0,2]=0
        r[1,0]=0; r[1,1]=1; r[1,2]=0
        r[2,0]=0; r[2,1]=0; r[2,2]=1
        return r
    cdef DTYPE_t inv_det = 1.0 / det
    r[0,0] = (m11*m22 - m12*m21) * inv_det
    r[0,1] = -(m01*m22 - m02*m21) * inv_det
    r[0,2] = (m01*m12 - m02*m11) * inv_det
    r[1,0] = -(m10*m22 - m12*m20) * inv_det
    r[1,1] = (m00*m22 - m02*m20) * inv_det
    r[1,2] = -(m00*m12 - m02*m10) * inv_det
    r[2,0] = (m10*m21 - m11*m20) * inv_det
    r[2,1] = -(m00*m21 - m01*m20) * inv_det
    r[2,2] = (m00*m11 - m01*m10) * inv_det
    return r


cdef inline void _quat_conjugate(DTYPE_t x, DTYPE_t y, DTYPE_t z, DTYPE_t w,
                                  DTYPE_t* rx, DTYPE_t* ry, DTYPE_t* rz, DTYPE_t* rw) noexcept nogil:
    rx[0] = -x; ry[0] = -y; rz[0] = -z; rw[0] = w


def quat_conjugate(float x, float y, float z, float w):
    return (-x, -y, -z, w)


# ── Quaternion helpers ────────────────────────────────────────────────

cdef inline void _quat_mul(DTYPE_t ax, DTYPE_t ay, DTYPE_t az, DTYPE_t aw,
                            DTYPE_t bx, DTYPE_t by, DTYPE_t bz, DTYPE_t bw,
                            DTYPE_t* rx, DTYPE_t* ry, DTYPE_t* rz, DTYPE_t* rw) noexcept nogil:
    rx[0] = aw*bx + ax*bw + ay*bz - az*by
    ry[0] = aw*by - ax*bz + ay*bw + az*bx
    rz[0] = aw*bz + ax*by - ay*bx + az*bw
    rw[0] = aw*bw - ax*bx - ay*by - az*bz


def quat_mul(float ax, float ay, float az, float aw,
              float bx, float by, float bz, float bw):
    cdef DTYPE_t rx, ry, rz, rw
    _quat_mul(ax, ay, az, aw, bx, by, bz, bw, &rx, &ry, &rz, &rw)
    return (rx, ry, rz, rw)


cdef inline void _quat_normalize(DTYPE_t x, DTYPE_t y, DTYPE_t z, DTYPE_t w,
                                  DTYPE_t* rx, DTYPE_t* ry, DTYPE_t* rz, DTYPE_t* rw) noexcept nogil:
    cdef DTYPE_t n = sqrt(x*x + y*y + z*z + w*w)
    if n > 1e-10:
        n = 1.0 / n
        rx[0] = x*n; ry[0] = y*n; rz[0] = z*n; rw[0] = w*n
    else:
        rx[0] = 0; ry[0] = 0; rz[0] = 0; rw[0] = 1.0


def quat_normalize(float x, float y, float z, float w):
    cdef DTYPE_t rx, ry, rz, rw
    _quat_normalize(x, y, z, w, &rx, &ry, &rz, &rw)
    return (rx, ry, rz, rw)


cdef inline void _quat_rotate_vec3(DTYPE_t qx, DTYPE_t qy, DTYPE_t qz, DTYPE_t qw,
                                    DTYPE_t vx, DTYPE_t vy, DTYPE_t vz,
                                    DTYPE_t* rx, DTYPE_t* ry, DTYPE_t* rz) noexcept nogil:
    cdef DTYPE_t tx = 2.0*(qy*vz - qz*vy)
    cdef DTYPE_t ty = 2.0*(qz*vx - qx*vz)
    cdef DTYPE_t tz = 2.0*(qx*vy - qy*vx)
    rx[0] = vx + qw*tx + qy*tz - qz*ty
    ry[0] = vy + qw*ty + qz*tx - qx*tz
    rz[0] = vz + qw*tz + qx*ty - qy*tx


def quat_rotate_vec3(float qx, float qy, float qz, float qw,
                      float vx, float vy, float vz):
    cdef DTYPE_t rx, ry, rz
    _quat_rotate_vec3(qx, qy, qz, qw, vx, vy, vz, &rx, &ry, &rz)
    return (rx, ry, rz)


def quat_slerp(float ax, float ay, float az, float aw,
                float bx, float by, float bz, float bw, float t):
    cdef DTYPE_t rx, ry, rz, rw, n
    cdef DTYPE_t d = ax*bx + ay*by + az*bz + aw*bw
    cdef DTYPE_t _bx = bx, _by = by, _bz = bz, _bw = bw
    if d < 0.0:
        _bx = -bx; _by = -by; _bz = -bz; _bw = -bw
        d = -d
    if d > 0.9995:
        rx = ax + t*(_bx - ax)
        ry = ay + t*(_by - ay)
        rz = az + t*(_bz - az)
        rw = aw + t*(_bw - aw)
        n = sqrt(rx*rx + ry*ry + rz*rz + rw*rw)
        if n > 1e-10:
            n = 1.0 / n
            return (rx*n, ry*n, rz*n, rw*n)
        return (ax, ay, az, aw)
    cdef DTYPE_t theta0 = acos(d)
    cdef DTYPE_t sin_theta0 = sin(theta0)
    cdef DTYPE_t inv_sin = 1.0 / sin_theta0
    cdef DTYPE_t theta = theta0 * t
    cdef DTYPE_t s0 = (cos(theta) - d * sin(theta) * inv_sin)
    cdef DTYPE_t s1 = sin(theta) * inv_sin
    rx = s0*ax + s1*_bx
    ry = s0*ay + s1*_by
    rz = s0*az + s1*_bz
    rw = s0*aw + s1*_bw
    n = sqrt(rx*rx + ry*ry + rz*rz + rw*rw)
    if n > 1e-10:
        n = 1.0 / n
        return (rx*n, ry*n, rz*n, rw*n)
    return (ax, ay, az, aw)


def quat_from_euler(float x_deg, float y_deg, float z_deg):
    cdef DTYPE_t hx = (x_deg * _DEG2RAD) * 0.5
    cdef DTYPE_t hy = (y_deg * _DEG2RAD) * 0.5
    cdef DTYPE_t hz = (z_deg * _DEG2RAD) * 0.5
    cdef DTYPE_t sx = sin(hx)
    cdef DTYPE_t cx = cos(hx)
    cdef DTYPE_t sy = sin(hy)
    cdef DTYPE_t cy = cos(hy)
    cdef DTYPE_t sz = sin(hz)
    cdef DTYPE_t cz = cos(hz)
    return (sx*cy*cz - cx*sy*sz,
            cx*sy*cz + sx*cy*sz,
            cx*cy*sz - sx*sy*cz,
            cx*cy*cz + sx*sy*sz)


def quat_to_euler(float x, float y, float z, float w):
    cdef DTYPE_t sinx_cosp = 2*(w*x + y*z)
    cdef DTYPE_t cosx_cosp = 1 - 2*(x*x + y*y)
    cdef DTYPE_t rx = atan2(sinx_cosp, cosx_cosp) * _RAD2DEG
    cdef DTYPE_t siny_cosp = 2*(w*y - z*x)
    cdef DTYPE_t ry = asin(max(-1.0, min(1.0, siny_cosp))) * _RAD2DEG
    cdef DTYPE_t sinz_cosp = 2*(w*z + x*y)
    cdef DTYPE_t cosz_cosp = 1 - 2*(y*y + z*z)
    cdef DTYPE_t rz = atan2(sinz_cosp, cosz_cosp) * _RAD2DEG
    return rx, ry, rz


def quat_from_euler_rad(float x_rad, float y_rad, float z_rad):
    cdef DTYPE_t hx = x_rad * 0.5
    cdef DTYPE_t hy = y_rad * 0.5
    cdef DTYPE_t hz = z_rad * 0.5
    cdef DTYPE_t sx = sin(hx)
    cdef DTYPE_t cx = cos(hx)
    cdef DTYPE_t sy = sin(hy)
    cdef DTYPE_t cy = cos(hy)
    cdef DTYPE_t sz = sin(hz)
    cdef DTYPE_t cz = cos(hz)
    return (sx*cy*cz - cx*sy*sz,
            cx*sy*cz + sx*cy*sz,
            cx*cy*sz - sx*sy*cz,
            cx*cy*cz + sx*sy*sz)


def quat_to_euler_rad(float x, float y, float z, float w):
    cdef DTYPE_t sinx_cosp = 2*(w*x + y*z)
    cdef DTYPE_t cosx_cosp = 1 - 2*(x*x + y*y)
    cdef DTYPE_t rx = atan2(sinx_cosp, cosx_cosp)
    cdef DTYPE_t siny_cosp = 2*(w*y - z*x)
    cdef DTYPE_t ry = asin(max(-1.0, min(1.0, siny_cosp)))
    cdef DTYPE_t sinz_cosp = 2*(w*z + x*y)
    cdef DTYPE_t cosz_cosp = 1 - 2*(y*y + z*z)
    cdef DTYPE_t rz = atan2(sinz_cosp, cosz_cosp)
    return rx, ry, rz


# ── Vec3 helpers ──────────────────────────────────────────────────────

cdef inline void _vec3_normalize(DTYPE_t x, DTYPE_t y, DTYPE_t z,
                                  DTYPE_t* rx, DTYPE_t* ry, DTYPE_t* rz) noexcept nogil:
    cdef DTYPE_t n = sqrt(x*x + y*y + z*z)
    if n > 1e-10:
        n = 1.0 / n
        rx[0] = x*n; ry[0] = y*n; rz[0] = z*n
    else:
        rx[0] = 0; ry[0] = 0; rz[0] = 0


def vec3_normalize(float x, float y, float z):
    cdef DTYPE_t rx, ry, rz
    _vec3_normalize(x, y, z, &rx, &ry, &rz)
    return (rx, ry, rz)


def vec3_sub(float ax, float ay, float az, float bx, float by, float bz):
    return (ax - bx, ay - by, az - bz)


def vec3_add(float ax, float ay, float az, float bx, float by, float bz):
    return (ax + bx, ay + by, az + bz)


def vec3_scale(float x, float y, float z, float s):
    return (x*s, y*s, z*s)


def vec3_dot(float ax, float ay, float az, float bx, float by, float bz) -> float:
    return ax*bx + ay*by + az*bz


def vec3_cross(float ax, float ay, float az, float bx, float by, float bz):
    return (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)


# ── Ray intersection helpers ──────────────────────────────────────────

def ray_triangle_intersect(float ox, float oy, float oz,
                            float dx, float dy, float dz,
                            float v0x, float v0y, float v0z,
                            float v1x, float v1y, float v1z,
                            float v2x, float v2y, float v2z) -> float:
    cdef DTYPE_t e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z
    cdef DTYPE_t e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z
    cdef DTYPE_t px = dy*e2z - dz*e2y
    cdef DTYPE_t py = dz*e2x - dx*e2z
    cdef DTYPE_t pz = dx*e2y - dy*e2x
    cdef DTYPE_t det = e1x*px + e1y*py + e1z*pz
    if fabs(det) < 1e-12:
        return -1.0
    cdef DTYPE_t inv_det = 1.0 / det
    cdef DTYPE_t tx = ox - v0x, ty = oy - v0y, tz = oz - v0z
    cdef DTYPE_t u = (tx*px + ty*py + tz*pz) * inv_det
    if u < 0.0 or u > 1.0:
        return -1.0
    cdef DTYPE_t qx = ty*e1z - tz*e1y
    cdef DTYPE_t qy = tz*e1x - tx*e1z
    cdef DTYPE_t qz = tx*e1y - ty*e1x
    cdef DTYPE_t v = (dx*qx + dy*qy + dz*qz) * inv_det
    if v < 0.0 or u + v > 1.0:
        return -1.0
    cdef DTYPE_t t = (e2x*qx + e2y*qy + e2z*qz) * inv_det
    return t if t > 0.0 else -1.0


def ray_mesh_intersect(float ox, float oy, float oz,
                        float dx, float dy, float dz,
                        np.ndarray[DTYPE_t, ndim=1] verts,
                        np.ndarray[INT32_t, ndim=1] indices) -> float:
    cdef int n_indices = indices.shape[0]
    cdef DTYPE_t best_t = -1.0, t
    cdef int i, i0, i1, i2
    cdef DTYPE_t v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z
    for i in range(0, n_indices, 3):
        i0 = indices[i]; i1 = indices[i+1]; i2 = indices[i+2]
        v0x = verts[i0*3]; v0y = verts[i0*3+1]; v0z = verts[i0*3+2]
        v1x = verts[i1*3]; v1y = verts[i1*3+1]; v1z = verts[i1*3+2]
        v2x = verts[i2*3]; v2y = verts[i2*3+1]; v2z = verts[i2*3+2]
        t = ray_triangle_intersect(ox, oy, oz, dx, dy, dz,
                                    v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z)
        if t > 0 and (best_t < 0 or t < best_t):
            best_t = t
    return best_t


def ray_aabb_intersect(float ox, float oy, float oz,
                        float dx, float dy, float dz,
                        float bmin_x, float bmin_y, float bmin_z,
                        float bmax_x, float bmax_y, float bmax_z) -> float:
    cdef DTYPE_t tmin = -1e30, tmax = 1e30
    cdef DTYPE_t t1, t2
    if fabs(dx) > 1e-30:
        t1 = (bmin_x - ox) / dx
        t2 = (bmax_x - ox) / dx
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin: tmin = t1
        if t2 < tmax: tmax = t2
    elif ox < bmin_x or ox > bmax_x:
        return -1.0
    if fabs(dy) > 1e-30:
        t1 = (bmin_y - oy) / dy
        t2 = (bmax_y - oy) / dy
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin: tmin = t1
        if t2 < tmax: tmax = t2
    elif oy < bmin_y or oy > bmax_y:
        return -1.0
    if fabs(dz) > 1e-30:
        t1 = (bmin_z - oz) / dz
        t2 = (bmax_z - oz) / dz
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin: tmin = t1
        if t2 < tmax: tmax = t2
    elif oz < bmin_z or oz > bmax_z:
        return -1.0
    if tmin > tmax:
        return -1.0
    return tmin if tmin > 0.0 else (tmax if tmax > 0.0 else -1.0)


def ray_sphere_intersect_batch(
    np.ndarray[DTYPE_t, ndim=1] origins_x,
    np.ndarray[DTYPE_t, ndim=1] origins_y,
    np.ndarray[DTYPE_t, ndim=1] origins_z,
    np.ndarray[DTYPE_t, ndim=1] dirs_x,
    np.ndarray[DTYPE_t, ndim=1] dirs_y,
    np.ndarray[DTYPE_t, ndim=1] dirs_z,
    np.ndarray[DTYPE_t, ndim=1] centers_x,
    np.ndarray[DTYPE_t, ndim=1] centers_y,
    np.ndarray[DTYPE_t, ndim=1] centers_z,
    np.ndarray[DTYPE_t, ndim=1] radii,
    np.ndarray[DTYPE_t, ndim=1] results,
):
    cdef int i, n = radii.shape[0]
    cdef DTYPE_t ocx, ocy, ocz, b, c, disc, sq, t, t2
    for i in range(n):
        ocx = origins_x[i] - centers_x[i]
        ocy = origins_y[i] - centers_y[i]
        ocz = origins_z[i] - centers_z[i]
        b = ocx*dirs_x[i] + ocy*dirs_y[i] + ocz*dirs_z[i]
        c = ocx*ocx + ocy*ocy + ocz*ocz - radii[i]*radii[i]
        disc = b*b - c
        if disc < 0.0:
            results[i] = -1.0
            continue
        sq = sqrt(disc)
        t = -b - sq
        if t > 0.0:
            results[i] = t
        else:
            t2 = -b + sq
            results[i] = t2 if t2 > 0.0 else -1.0


# ── Normal matrix ─────────────────────────────────────────────────────

def mat4_normal_matrix(np.ndarray[DTYPE_t, ndim=2] model):
    cdef np.ndarray[DTYPE_t, ndim=2] m = model[:3, :3].copy()
    cdef np.ndarray[np.float32_t, ndim=2] nm = np.empty((3, 3), dtype=np.float32)
    cdef DTYPE_t a, b, c, d, e, f, g, h, i
    cdef DTYPE_t A, B, C, D, E, F, G, H, I, det, inv_det
    a = m[0,0]; b = m[0,1]; c = m[0,2]
    d = m[1,0]; e = m[1,1]; f = m[1,2]
    g = m[2,0]; h = m[2,1]; i = m[2,2]
    # Cofactor matrix (transpose of the adjugate), used for (M^-1)^T.
    A = e*i - f*h
    B = -(d*i - f*g)
    C = d*h - e*g
    D = -(b*i - c*h)
    E = a*i - c*g
    F = -(a*h - b*g)
    G = b*f - c*e
    H = -(a*f - c*d)
    I = a*e - b*d
    det = a*A + b*B + c*C
    if det < 1e-12 and det > -1e-12:
        return np.eye(3, dtype=np.float32)
    inv_det = 1.0 / det
    nm[0,0] = <np.float32_t>(A * inv_det)
    nm[0,1] = <np.float32_t>(B * inv_det)
    nm[0,2] = <np.float32_t>(C * inv_det)
    nm[1,0] = <np.float32_t>(D * inv_det)
    nm[1,1] = <np.float32_t>(E * inv_det)
    nm[1,2] = <np.float32_t>(F * inv_det)
    nm[2,0] = <np.float32_t>(G * inv_det)
    nm[2,1] = <np.float32_t>(H * inv_det)
    nm[2,2] = <np.float32_t>(I * inv_det)
    return nm


# ── Batch matrices to float32 ─────────────────────────────────────────

def batch_matrices_to_f32(np.ndarray[DTYPE_t, ndim=3] matrices):
    cdef int n = matrices.shape[0]
    if n == 0:
        return np.empty((0, 16), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] out = np.empty((n, 16), dtype=np.float32)
    cdef int i, r, c
    if n > 512:
        with nogil:
            for i in prange(n, schedule='static'):
                for r in range(4):
                    for c in range(4):
                        out[i, r*4 + c] = <np.float32_t>matrices[i, r, c]
        return out
    for i in range(n):
        for r in range(4):
            for c in range(4):
                out[i, r*4 + c] = <np.float32_t>matrices[i, r, c]
    return out
