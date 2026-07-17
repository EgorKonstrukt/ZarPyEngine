# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef inline int aabb_intersects_one(
    DTYPE_t a_min_x, DTYPE_t a_min_y, DTYPE_t a_min_z,
    DTYPE_t a_max_x, DTYPE_t a_max_y, DTYPE_t a_max_z,
    DTYPE_t b_min_x, DTYPE_t b_min_y, DTYPE_t b_min_z,
    DTYPE_t b_max_x, DTYPE_t b_max_y, DTYPE_t b_max_z,
) noexcept nogil:
    if (a_min_x <= b_max_x and a_min_y <= b_max_y and a_min_z <= b_max_z and
            a_max_x >= b_min_x and a_max_y >= b_min_y and a_max_z >= b_min_z):
        return 1
    return 0

cdef inline int aabb_contains_point_one(
    DTYPE_t min_x, DTYPE_t min_y, DTYPE_t min_z,
    DTYPE_t max_x, DTYPE_t max_y, DTYPE_t max_z,
    DTYPE_t px, DTYPE_t py, DTYPE_t pz,
) noexcept nogil:
    if (min_x <= px and min_y <= py and min_z <= pz and
            max_x >= px and max_y >= py and max_z >= pz):
        return 1
    return 0

cdef inline DTYPE_t aabb_ray_one(
    DTYPE_t a_min_x, DTYPE_t a_min_y, DTYPE_t a_min_z,
    DTYPE_t a_max_x, DTYPE_t a_max_y, DTYPE_t a_max_z,
    DTYPE_t ox, DTYPE_t oy, DTYPE_t oz,
    DTYPE_t dx, DTYPE_t dy, DTYPE_t dz,
    DTYPE_t max_dist,
) noexcept nogil:
    cdef DTYPE_t t_near = -1e30
    cdef DTYPE_t t_far = 1e30
    cdef DTYPE_t d, o, mn, mx, inv, t1, t2

    d = dx
    o = ox
    mn = a_min_x
    mx = a_max_x
    if d > -1e-12 and d < 1e-12:
        if o < mn or o > mx:
            return -1.0
    else:
        inv = 1.0 / d
        t1 = (mn - o) * inv
        t2 = (mx - o) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_near:
            t_near = t1
        if t2 < t_far:
            t_far = t2
        if t_near > t_far or t_far < 0:
            return -1.0

    d = dy
    o = oy
    mn = a_min_y
    mx = a_max_y
    if d > -1e-12 and d < 1e-12:
        if o < mn or o > mx:
            return -1.0
    else:
        inv = 1.0 / d
        t1 = (mn - o) * inv
        t2 = (mx - o) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_near:
            t_near = t1
        if t2 < t_far:
            t_far = t2
        if t_near > t_far or t_far < 0:
            return -1.0

    d = dz
    o = oz
    mn = a_min_z
    mx = a_max_z
    if d > -1e-12 and d < 1e-12:
        if o < mn or o > mx:
            return -1.0
    else:
        inv = 1.0 / d
        t1 = (mn - o) * inv
        t2 = (mx - o) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_near:
            t_near = t1
        if t2 < t_far:
            t_far = t2
        if t_near > t_far or t_far < 0:
            return -1.0

    if t_near < 0:
        t_near = t_far
    if t_near <= max_dist:
        return t_near
    return -1.0

def aabb_intersects_batch(
    np.ndarray[DTYPE_t, ndim=1] a_min_x,
    np.ndarray[DTYPE_t, ndim=1] a_min_y,
    np.ndarray[DTYPE_t, ndim=1] a_min_z,
    np.ndarray[DTYPE_t, ndim=1] a_max_x,
    np.ndarray[DTYPE_t, ndim=1] a_max_y,
    np.ndarray[DTYPE_t, ndim=1] a_max_z,
    np.ndarray[DTYPE_t, ndim=1] b_min_x,
    np.ndarray[DTYPE_t, ndim=1] b_min_y,
    np.ndarray[DTYPE_t, ndim=1] b_min_z,
    np.ndarray[DTYPE_t, ndim=1] b_max_x,
    np.ndarray[DTYPE_t, ndim=1] b_max_y,
    np.ndarray[DTYPE_t, ndim=1] b_max_z,
):
    cdef int n = a_min_x.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] result = np.empty(n, dtype=np.int32)
    cdef int i
    for i in range(n):
        result[i] = aabb_intersects_one(
            a_min_x[i], a_min_y[i], a_min_z[i],
            a_max_x[i], a_max_y[i], a_max_z[i],
            b_min_x[i], b_min_y[i], b_min_z[i],
            b_max_x[i], b_max_y[i], b_max_z[i],
        )
    return result

def aabb_contains_point_batch(
    np.ndarray[DTYPE_t, ndim=1] min_x,
    np.ndarray[DTYPE_t, ndim=1] min_y,
    np.ndarray[DTYPE_t, ndim=1] min_z,
    np.ndarray[DTYPE_t, ndim=1] max_x,
    np.ndarray[DTYPE_t, ndim=1] max_y,
    np.ndarray[DTYPE_t, ndim=1] max_z,
    np.ndarray[DTYPE_t, ndim=1] px,
    np.ndarray[DTYPE_t, ndim=1] py,
    np.ndarray[DTYPE_t, ndim=1] pz,
):
    cdef int n = min_x.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] result = np.empty(n, dtype=np.int32)
    cdef int i
    for i in range(n):
        result[i] = aabb_contains_point_one(
            min_x[i], min_y[i], min_z[i],
            max_x[i], max_y[i], max_z[i],
            px[i], py[i], pz[i],
        )
    return result

def aabb_ray_batch(
    np.ndarray[DTYPE_t, ndim=1] min_x,
    np.ndarray[DTYPE_t, ndim=1] min_y,
    np.ndarray[DTYPE_t, ndim=1] min_z,
    np.ndarray[DTYPE_t, ndim=1] max_x,
    np.ndarray[DTYPE_t, ndim=1] max_y,
    np.ndarray[DTYPE_t, ndim=1] max_z,
    np.ndarray[DTYPE_t, ndim=1] ox,
    np.ndarray[DTYPE_t, ndim=1] oy,
    np.ndarray[DTYPE_t, ndim=1] oz,
    np.ndarray[DTYPE_t, ndim=1] dx,
    np.ndarray[DTYPE_t, ndim=1] dy,
    np.ndarray[DTYPE_t, ndim=1] dz,
    DTYPE_t max_dist,
):
    cdef int n = min_x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.empty(n, dtype=DTYPE)
    cdef int i
    for i in range(n):
        result[i] = aabb_ray_one(
            min_x[i], min_y[i], min_z[i],
            max_x[i], max_y[i], max_z[i],
            ox[i], oy[i], oz[i],
            dx[i], dy[i], dz[i],
            max_dist,
        )
    return result

def aabb_intersects_point_batch(
    np.ndarray[DTYPE_t, ndim=1] a_min_x,
    np.ndarray[DTYPE_t, ndim=1] a_min_y,
    np.ndarray[DTYPE_t, ndim=1] a_min_z,
    np.ndarray[DTYPE_t, ndim=1] a_max_x,
    np.ndarray[DTYPE_t, ndim=1] a_max_y,
    np.ndarray[DTYPE_t, ndim=1] a_max_z,
    DTYPE_t px, DTYPE_t py, DTYPE_t pz,
):
    cdef int n = a_min_x.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] result = np.empty(n, dtype=np.int32)
    cdef int i
    for i in range(n):
        result[i] = aabb_contains_point_one(
            a_min_x[i], a_min_y[i], a_min_z[i],
            a_max_x[i], a_max_y[i], a_max_z[i],
            px, py, pz,
        )
    return result

def aabb_intersects_sphere_batch(
    np.ndarray[DTYPE_t, ndim=1] a_min_x,
    np.ndarray[DTYPE_t, ndim=1] a_min_y,
    np.ndarray[DTYPE_t, ndim=1] a_min_z,
    np.ndarray[DTYPE_t, ndim=1] a_max_x,
    np.ndarray[DTYPE_t, ndim=1] a_max_y,
    np.ndarray[DTYPE_t, ndim=1] a_max_z,
    DTYPE_t cx, DTYPE_t cy, DTYPE_t cz, DTYPE_t radius,
):
    cdef int n = a_min_x.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] result = np.empty(n, dtype=np.int32)
    cdef int i
    cdef DTYPE_t dx, dy, dz, dist_sq
    for i in range(n):
        dx = 0.0
        dy = 0.0
        dz = 0.0
        if cx < a_min_x[i]:
            dx = a_min_x[i] - cx
        elif cx > a_max_x[i]:
            dx = cx - a_max_x[i]
        if cy < a_min_y[i]:
            dy = a_min_y[i] - cy
        elif cy > a_max_y[i]:
            dy = cy - a_max_y[i]
        if cz < a_min_z[i]:
            dz = a_min_z[i] - cz
        elif cz > a_max_z[i]:
            dz = cz - a_max_z[i]
        dist_sq = dx*dx + dy*dy + dz*dz
        if dist_sq <= radius*radius:
            result[i] = 1
        else:
            result[i] = 0
    return result
