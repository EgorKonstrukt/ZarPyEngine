# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef inline void _extract_frustum_planes(
    const DTYPE_t[:, :] vp,
    DTYPE_t[:, :] planes_out,
) noexcept nogil:
    cdef int i
    cdef DTYPE_t inv_norm
    for i in range(6):
        if i == 0:
            planes_out[i, 0] = vp[3, 0] + vp[0, 0]
            planes_out[i, 1] = vp[3, 1] + vp[0, 1]
            planes_out[i, 2] = vp[3, 2] + vp[0, 2]
            planes_out[i, 3] = vp[3, 3] + vp[0, 3]
        elif i == 1:
            planes_out[i, 0] = vp[3, 0] - vp[0, 0]
            planes_out[i, 1] = vp[3, 1] - vp[0, 1]
            planes_out[i, 2] = vp[3, 2] - vp[0, 2]
            planes_out[i, 3] = vp[3, 3] - vp[0, 3]
        elif i == 2:
            planes_out[i, 0] = vp[3, 0] + vp[1, 0]
            planes_out[i, 1] = vp[3, 1] + vp[1, 1]
            planes_out[i, 2] = vp[3, 2] + vp[1, 2]
            planes_out[i, 3] = vp[3, 3] + vp[1, 3]
        elif i == 3:
            planes_out[i, 0] = vp[3, 0] - vp[1, 0]
            planes_out[i, 1] = vp[3, 1] - vp[1, 1]
            planes_out[i, 2] = vp[3, 2] - vp[1, 2]
            planes_out[i, 3] = vp[3, 3] - vp[1, 3]
        elif i == 4:
            planes_out[i, 0] = vp[3, 0] + vp[2, 0]
            planes_out[i, 1] = vp[3, 1] + vp[2, 1]
            planes_out[i, 2] = vp[3, 2] + vp[2, 2]
            planes_out[i, 3] = vp[3, 3] + vp[2, 3]
        else:
            planes_out[i, 0] = vp[3, 0] - vp[2, 0]
            planes_out[i, 1] = vp[3, 1] - vp[2, 1]
            planes_out[i, 2] = vp[3, 2] - vp[2, 2]
            planes_out[i, 3] = vp[3, 3] - vp[2, 3]
        inv_norm = 1.0 / sqrt(planes_out[i,0]*planes_out[i,0]
                            + planes_out[i,1]*planes_out[i,1]
                            + planes_out[i,2]*planes_out[i,2])
        planes_out[i, 0] *= inv_norm
        planes_out[i, 1] *= inv_norm
        planes_out[i, 2] *= inv_norm
        planes_out[i, 3] *= inv_norm


def cpu_frustum_cull(
    np.ndarray[DTYPE_t, ndim=2] centers,
    np.ndarray[DTYPE_t, ndim=1] radii,
    np.ndarray[DTYPE_t, ndim=2] view_proj,
):
    cdef int n = centers.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.intp)

    cdef np.ndarray[DTYPE_t, ndim=2] planes = np.empty((6, 4), dtype=DTYPE)
    _extract_frustum_planes(view_proj, planes)

    cdef np.ndarray[np.intp_t, ndim=1] visible = np.empty(n, dtype=np.intp)
    cdef int count = 0, i, j
    cdef DTYPE_t dist
    cdef bint inside

    for i in range(n):
        inside = True
        for j in range(6):
            dist = (planes[j,0]*centers[i,0] + planes[j,1]*centers[i,1]
                    + planes[j,2]*centers[i,2] + planes[j,3])
            if dist < -radii[i]:
                inside = False
                break
        if inside:
            visible[count] = i
            count += 1

    return visible[:count]


def extract_frustum_planes_c(np.ndarray[DTYPE_t, ndim=2] view_proj):
    cdef np.ndarray[DTYPE_t, ndim=2] planes = np.empty((6, 4), dtype=DTYPE)
    _extract_frustum_planes(view_proj, planes)
    return planes
