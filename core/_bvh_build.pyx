# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np

cdef inline float _surface_area(float bx, float by, float bz,
                                 float mx, float my, float mz) noexcept nogil:
    cdef float dx = mx - bx, dy = my - by, dz = mz - bz
    return 2.0 * (dx * dy + dx * dz + dy * dz)


def sah_compute_best_split(
    np.intp_t[:] tris,
    float[:, :] tri_bmin,
    float[:, :] tri_bmax,
    float[:, :] centroids,
    int n,
    float parent_sa,
    int sah_bins = 12,
):
    """
    Binned SAH — single pass per axis, no temporary allocations.

    Returns (best_axis, best_split, left_mask)
      best_axis = -1  means 'make leaf'
    """
    cdef int i, axis, bi, ti, b, bi2, pc, sc
    cdef float cmin_x, cmin_y, cmin_z, cmax_x, cmax_y, cmax_z
    cdef float crange_x, crange_y, crange_z
    cdef float best_cost = 1e30, best_split = 0.0
    cdef int best_axis = -1
    cdef float cost, leaf_cost
    cdef float cv, scale, c_min_c, crange_c

    cdef float bbi_min_x[12], bbi_min_y[12], bbi_min_z[12]
    cdef float bbi_max_x[12], bbi_max_y[12], bbi_max_z[12]
    cdef int bbi_cnt[12]

    cdef float left_sa[12], right_sa[12]
    cdef int left_cnt[12]
    cdef float pmin_x, pmin_y, pmin_z, pmax_x, pmax_y, pmax_z
    cdef float smin_x, smin_y, smin_z, smax_x, smax_y, smax_z

    cdef np.ndarray left_mask_arr
    cdef np.uint8_t[:] left_mask

    cmin_x = 1e30; cmin_y = 1e30; cmin_z = 1e30
    cmax_x = -1e30; cmax_y = -1e30; cmax_z = -1e30

    for i in range(n):
        ti = tris[i]
        cv = centroids[ti, 0]
        if cv < cmin_x: cmin_x = cv
        if cv > cmax_x: cmax_x = cv
        cv = centroids[ti, 1]
        if cv < cmin_y: cmin_y = cv
        if cv > cmax_y: cmax_y = cv
        cv = centroids[ti, 2]
        if cv < cmin_z: cmin_z = cv
        if cv > cmax_z: cmax_z = cv

    crange_x = cmax_x - cmin_x
    crange_y = cmax_y - cmin_y
    crange_z = cmax_z - cmin_z

    leaf_cost = <float>n * 1.0

    for axis in range(3):
        if axis == 0:
            if crange_x < 1e-12: continue
            c_min_c = cmin_x; crange_c = crange_x
        elif axis == 1:
            if crange_y < 1e-12: continue
            c_min_c = cmin_y; crange_c = crange_y
        else:
            if crange_z < 1e-12: continue
            c_min_c = cmin_z; crange_c = crange_z

        scale = <float>sah_bins / crange_c

        for bi in range(sah_bins):
            bbi_min_x[bi] = 1e30; bbi_min_y[bi] = 1e30; bbi_min_z[bi] = 1e30
            bbi_max_x[bi] = -1e30; bbi_max_y[bi] = -1e30; bbi_max_z[bi] = -1e30
            bbi_cnt[bi] = 0

        for i in range(n):
            ti = tris[i]
            if axis == 0:
                cv = centroids[ti, 0]
            elif axis == 1:
                cv = centroids[ti, 1]
            else:
                cv = centroids[ti, 2]

            bi = <int>((cv - c_min_c) * scale)
            if bi < 0: bi = 0
            if bi >= sah_bins: bi = sah_bins - 1

            bbi_cnt[bi] += 1
            if tri_bmin[ti, 0] < bbi_min_x[bi]: bbi_min_x[bi] = tri_bmin[ti, 0]
            if tri_bmin[ti, 1] < bbi_min_y[bi]: bbi_min_y[bi] = tri_bmin[ti, 1]
            if tri_bmin[ti, 2] < bbi_min_z[bi]: bbi_min_z[bi] = tri_bmin[ti, 2]
            if tri_bmax[ti, 0] > bbi_max_x[bi]: bbi_max_x[bi] = tri_bmax[ti, 0]
            if tri_bmax[ti, 1] > bbi_max_y[bi]: bbi_max_y[bi] = tri_bmax[ti, 1]
            if tri_bmax[ti, 2] > bbi_max_z[bi]: bbi_max_z[bi] = tri_bmax[ti, 2]

        pmin_x = 1e30; pmin_y = 1e30; pmin_z = 1e30
        pmax_x = -1e30; pmax_y = -1e30; pmax_z = -1e30
        pc = 0
        for b in range(sah_bins - 1):
            if bbi_cnt[b]:
                if bbi_min_x[b] < pmin_x: pmin_x = bbi_min_x[b]
                if bbi_min_y[b] < pmin_y: pmin_y = bbi_min_y[b]
                if bbi_min_z[b] < pmin_z: pmin_z = bbi_min_z[b]
                if bbi_max_x[b] > pmax_x: pmax_x = bbi_max_x[b]
                if bbi_max_y[b] > pmax_y: pmax_y = bbi_max_y[b]
                if bbi_max_z[b] > pmax_z: pmax_z = bbi_max_z[b]
                pc += bbi_cnt[b]
            left_sa[b] = _surface_area(pmin_x, pmin_y, pmin_z,
                                        pmax_x, pmax_y, pmax_z)
            left_cnt[b] = pc

        smin_x = 1e30; smin_y = 1e30; smin_z = 1e30
        smax_x = -1e30; smax_y = -1e30; smax_z = -1e30
        sc = 0
        for b in range(sah_bins - 2, -1, -1):
            bi2 = b + 1
            if bbi_cnt[bi2]:
                if bbi_min_x[bi2] < smin_x: smin_x = bbi_min_x[bi2]
                if bbi_min_y[bi2] < smin_y: smin_y = bbi_min_y[bi2]
                if bbi_min_z[bi2] < smin_z: smin_z = bbi_min_z[bi2]
                if bbi_max_x[bi2] > smax_x: smax_x = bbi_max_x[bi2]
                if bbi_max_y[bi2] > smax_y: smax_y = bbi_max_y[bi2]
                if bbi_max_z[bi2] > smax_z: smax_z = bbi_max_z[bi2]
                sc += bbi_cnt[bi2]
            right_sa[b] = _surface_area(smin_x, smin_y, smin_z,
                                         smax_x, smax_y, smax_z)

        for b in range(sah_bins - 1):
            if left_cnt[b] == 0 or (n - left_cnt[b]) == 0:
                continue
            cost = 1.0 + (left_sa[b] * <float>left_cnt[b]
                          + right_sa[b] * <float>(n - left_cnt[b])) / parent_sa
            if cost < best_cost:
                best_cost = cost
                best_axis = axis
                best_split = c_min_c + (<float>(b + 1)) * crange_c / <float>sah_bins

    if best_axis < 0 or best_cost >= leaf_cost:
        return -1, 0.0, None

    left_mask_arr = np.zeros(n, dtype=np.uint8)
    left_mask = left_mask_arr
    for i in range(n):
        ti = tris[i]
        if best_axis == 0:
            if centroids[ti, 0] < best_split:
                left_mask[i] = 1
        elif best_axis == 1:
            if centroids[ti, 1] < best_split:
                left_mask[i] = 1
        else:
            if centroids[ti, 2] < best_split:
                left_mask[i] = 1

    return best_axis, best_split, left_mask_arr.view(np.bool_)
