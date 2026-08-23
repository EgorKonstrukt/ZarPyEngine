# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: initializedcheck = False
# cython: emit_code_comments = False

import numpy as np
cimport numpy as np
cimport cython

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def build_eye_view_proj(double ex, double ey, double ez,
                        double rx, double ry, double rz,
                        double ux, double uy, double uz,
                        double fx, double fy, double fz,
                        double al, double ar, double au, double ad,
                        double near_z, double far_z):
    cdef double fl = ar - al
    cdef double fh = au - ad
    cdef double nl = near_z
    cdef double fr = far_z
    if fl < 1e-8:
        fl = 1e-8
    if fh < 1e-8:
        fh = 1e-8

    cdef double proj[16]
    proj[0] = 2.0 / fl
    proj[1] = 0.0
    proj[2] = 0.0
    proj[3] = 0.0
    proj[4] = 0.0
    proj[5] = 2.0 / fh
    proj[6] = 0.0
    proj[7] = 0.0
    proj[8] = (ar + al) / fl
    proj[9] = (au + ad) / fh
    proj[10] = -(fr + nl) / (fr - nl)
    proj[11] = -1.0
    proj[12] = 0.0
    proj[13] = 0.0
    proj[14] = -(2.0 * fr * nl) / (fr - nl)
    proj[15] = 0.0

    cdef double fmag = (fx * fx + fy * fy + fz * fz) ** 0.5
    if fmag < 1e-12:
        fmag = 1e-12
    cdef double f[3]
    f[0] = fx / fmag
    f[1] = fy / fmag
    f[2] = fz / fmag

    cdef double rmag = (rx * rx + ry * ry + rz * rz) ** 0.5
    if rmag < 1e-12:
        rmag = 1e-12
    cdef double rraw[3]
    rraw[0] = rx / rmag
    rraw[1] = ry / rmag
    rraw[2] = rz / rmag

    cdef double r[3]
    r[0] = f[1] * uz - f[2] * uy
    r[1] = f[2] * ux - f[0] * uz
    r[2] = f[0] * uy - f[1] * ux
    cdef double rnorm = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) ** 0.5
    if rnorm < 1e-12:
        rnorm = 1e-12
    r[0] /= rnorm
    r[1] /= rnorm
    r[2] /= rnorm

    cdef double u[3]
    u[0] = r[1] * f[2] - r[2] * f[1]
    u[1] = r[2] * f[0] - r[0] * f[2]
    u[2] = r[0] * f[1] - r[1] * f[0]

    cdef double m[16]
    m[0] = r[0]
    m[4] = r[1]
    m[8] = r[2]
    m[12] = -(r[0] * ex + r[1] * ey + r[2] * ez)

    m[1] = u[0]
    m[5] = u[1]
    m[9] = u[2]
    m[13] = -(u[0] * ex + u[1] * ey + u[2] * ez)

    m[2] = -f[0]
    m[6] = -f[1]
    m[10] = -f[2]
    m[14] = (f[0] * ex + f[1] * ey + f[2] * ez)

    m[3] = 0.0
    m[7] = 0.0
    m[11] = 0.0
    m[15] = 1.0

    cdef np.ndarray[np.float64_t, ndim=2] view = np.empty((4, 4), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] projm = np.empty((4, 4), dtype=np.float64)
    cdef int i
    for i in range(16):
        view.reshape(-1)[i] = m[i]
        projm.reshape(-1)[i] = proj[i]
    return view, projm


@cython.boundscheck(False)
@cython.wraparound(False)
def build_controller_mvp(double[::1] view, double[::1] proj, double[::1] model):
    cdef np.ndarray[np.float64_t, ndim=2] v = np.asarray(view).reshape((4, 4), order='F')
    cdef np.ndarray[np.float64_t, ndim=2] p = np.asarray(proj).reshape((4, 4), order='F')
    cdef np.ndarray[np.float64_t, ndim=2] mo = np.asarray(model).reshape((4, 4), order='F')
    cdef np.ndarray[np.float64_t, ndim=2] mvp = p @ v @ mo
    return np.ascontiguousarray(mvp).ravel(order='F')
