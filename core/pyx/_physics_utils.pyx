# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
from libc.math cimport fabs


def ray_aabb_intersect(float ox, float oy, float oz,
                        float dx, float dy, float dz,
                        float bmin_x, float bmin_y, float bmin_z,
                        float bmax_x, float bmax_y, float bmax_z):
    """Returns tmin if hit, -1.0 if miss (same as _ray_aabb_entry)."""
    cdef float tmin = -1e9, tmax = 1e9
    cdef float t1, t2, o, d, mn, mx

    o = ox; d = dx; mn = bmin_x; mx = bmax_x
    if fabs(d) < 1e-10:
        if o < mn or o > mx:
            return -1.0
    else:
        t1 = (mn - o) / d
        t2 = (mx - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin: tmin = t1
        if t2 < tmax: tmax = t2
        if tmin > tmax:
            return -1.0

    o = oy; d = dy; mn = bmin_y; mx = bmax_y
    if fabs(d) < 1e-10:
        if o < mn or o > mx:
            return -1.0
    else:
        t1 = (mn - o) / d
        t2 = (mx - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin: tmin = t1
        if t2 < tmax: tmax = t2
        if tmin > tmax:
            return -1.0

    o = oz; d = dz; mn = bmin_z; mx = bmax_z
    if fabs(d) < 1e-10:
        if o < mn or o > mx:
            return -1.0
    else:
        t1 = (mn - o) / d
        t2 = (mx - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin: tmin = t1
        if t2 < tmax: tmax = t2
        if tmin > tmax:
            return -1.0

    return tmin if tmax >= 0.0 else -1.0
