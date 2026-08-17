# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np
from libc.math cimport fabs


cdef inline float _slab(float ox, float oy, float oz,
                        float dx, float dy, float dz,
                        float bx0, float by0, float bz0,
                        float bx1, float by1, float bz1) noexcept nogil:
    cdef float tmin = -1e30
    cdef float tmax = 1e30
    cdef float t1, t2
    if fabs(dx) > 1e-30:
        t1 = (bx0 - ox) / dx
        t2 = (bx1 - ox) / dx
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
    elif ox < bx0 or ox > bx1:
        return -1.0
    if fabs(dy) > 1e-30:
        t1 = (by0 - oy) / dy
        t2 = (by1 - oy) / dy
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
    elif oy < by0 or oy > by1:
        return -1.0
    if fabs(dz) > 1e-30:
        t1 = (bz0 - oz) / dz
        t2 = (bz1 - oz) / dz
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
    elif oz < bz0 or oz > bz1:
        return -1.0
    if tmin > tmax:
        return -1.0
    if tmin > 0.0:
        return tmin
    return tmax if tmax > 0.0 else -1.0


cdef inline float _moll_trum(float ox, float oy, float oz,
                             float dx, float dy, float dz,
                             float ax, float ay, float az,
                             float bx, float by, float bz,
                             float cx, float cy, float cz) noexcept nogil:
    cdef float e1x = bx - ax, e1y = by - ay, e1z = bz - az
    cdef float e2x = cx - ax, e2y = cy - ay, e2z = cz - az
    cdef float px = dy * e2z - dz * e2y
    cdef float py = dz * e2x - dx * e2z
    cdef float pz = dx * e2y - dy * e2x
    cdef float det = e1x * px + e1y * py + e1z * pz
    cdef float inv_det
    cdef float tx, ty, tz
    cdef float u, v, t
    cdef float qx, qy, qz
    if fabs(det) < 1e-12:
        return -1.0
    inv_det = 1.0 / det
    tx = ox - ax
    ty = oy - ay
    tz = oz - az
    u = (tx * px + ty * py + tz * pz) * inv_det
    if u < 0.0 or u > 1.0:
        return -1.0
    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv_det
    if v < 0.0 or u + v > 1.0:
        return -1.0
    t = (e2x * qx + e2y * qy + e2z * qz) * inv_det
    if t > 0.0:
        return t
    return -1.0


cdef inline float _traverse(
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    const float[:, :] nds,
    const np.uint32_t[:] tv0,
    const np.uint32_t[:] tv1,
    const np.uint32_t[:] tv2,
    const float[:] verts,
    int n_nodes,
    np.intp_t[:] stack,
    bint early_out,
    int root_idx,
) noexcept nogil:
    cdef int sp = 0
    cdef float best_t = -1.0
    cdef int ni, left, right, k, tri_start, count
    cdef float t, tl, tr, tt
    cdef np.intp_t vi0, vi1, vi2

    stack[sp] = root_idx
    sp += 1

    while sp > 0:
        sp -= 1
        ni = stack[sp]
        t = _slab(ox, oy, oz, dx, dy, dz,
                  nds[ni, 0], nds[ni, 1], nds[ni, 2],
                  nds[ni, 3], nds[ni, 4], nds[ni, 5])
        if t < 0:
            continue
        if best_t >= 0.0 and t >= best_t:
            continue
        left = <int>nds[ni, 6]
        right = <int>nds[ni, 7]
        if right < 0:
            tri_start = left
            count = -right - 1
            for k in range(tri_start, tri_start + count):
                vi0 = tv0[k]
                vi1 = tv1[k]
                vi2 = tv2[k]
                tt = _moll_trum(ox, oy, oz, dx, dy, dz,
                                verts[vi0 * 3], verts[vi0 * 3 + 1], verts[vi0 * 3 + 2],
                                verts[vi1 * 3], verts[vi1 * 3 + 1], verts[vi1 * 3 + 2],
                                verts[vi2 * 3], verts[vi2 * 3 + 1], verts[vi2 * 3 + 2])
                if tt > 0 and (best_t < 0.0 or tt < best_t):
                    best_t = tt
                    if early_out:
                        return best_t
        else:
            tl = _slab(ox, oy, oz, dx, dy, dz,
                       nds[left, 0], nds[left, 1], nds[left, 2],
                       nds[left, 3], nds[left, 4], nds[left, 5])
            tr = _slab(ox, oy, oz, dx, dy, dz,
                       nds[right, 0], nds[right, 1], nds[right, 2],
                       nds[right, 3], nds[right, 4], nds[right, 5])
            if tl < 0 and tr < 0:
                continue
            if tl < 0:
                stack[sp] = right
                sp += 1
            elif tr < 0:
                stack[sp] = left
                sp += 1
            else:
                if tl < tr:
                    stack[sp] = right
                    sp += 1
                    stack[sp] = left
                    sp += 1
                else:
                    stack[sp] = left
                    sp += 1
                    stack[sp] = right
                    sp += 1

    return best_t


def bvh_intersect(
    np.ndarray[np.float32_t, ndim=2] nodes,
    np.ndarray[np.uint32_t, ndim=1] tri_v0,
    np.ndarray[np.uint32_t, ndim=1] tri_v1,
    np.ndarray[np.uint32_t, ndim=1] tri_v2,
    np.ndarray[np.float32_t, ndim=1] vertices,
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    int root_idx=0,
):
    cdef int n_nodes = nodes.shape[0]
    if n_nodes == 0:
        return -1.0
    cdef float[:, :] nds = nodes
    cdef np.uint32_t[:] tv0 = tri_v0
    cdef np.uint32_t[:] tv1 = tri_v1
    cdef np.uint32_t[:] tv2 = tri_v2
    cdef float[:] verts = vertices
    cdef np.intp_t[:] stack = np.empty(n_nodes + 1, dtype=np.intp)
    return _traverse(ox, oy, oz, dx, dy, dz,
                     nds, tv0, tv1, tv2, verts, n_nodes, stack, 0, root_idx)


def bvh_intersect_any(
    np.ndarray[np.float32_t, ndim=2] nodes,
    np.ndarray[np.uint32_t, ndim=1] tri_v0,
    np.ndarray[np.uint32_t, ndim=1] tri_v1,
    np.ndarray[np.uint32_t, ndim=1] tri_v2,
    np.ndarray[np.float32_t, ndim=1] vertices,
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    int root_idx=0,
):
    cdef int n_nodes = nodes.shape[0]
    if n_nodes == 0:
        return False
    cdef float[:, :] nds = nodes
    cdef np.uint32_t[:] tv0 = tri_v0
    cdef np.uint32_t[:] tv1 = tri_v1
    cdef np.uint32_t[:] tv2 = tri_v2
    cdef float[:] verts = vertices
    cdef np.intp_t[:] stack = np.empty(n_nodes + 1, dtype=np.intp)
    return _traverse(ox, oy, oz, dx, dy, dz,
                     nds, tv0, tv1, tv2, verts, n_nodes, stack, 1, root_idx) > 0.0


def triangles_intersect(
    np.ndarray[np.float32_t, ndim=1] vertices,
    np.ndarray[np.uint32_t, ndim=1] indices,
    float ox, float oy, float oz,
    float dx, float dy, float dz,
):
    cdef int n_tri = indices.shape[0] // 3
    cdef float[:] verts = vertices
    cdef np.uint32_t[:] idx = indices
    cdef int k
    cdef np.uint32_t vi0, vi1, vi2
    cdef float best_t = -1.0
    cdef float tt
    for k in range(n_tri):
        vi0 = idx[k * 3]
        vi1 = idx[k * 3 + 1]
        vi2 = idx[k * 3 + 2]
        tt = _moll_trum(ox, oy, oz, dx, dy, dz,
                        verts[vi0 * 3], verts[vi0 * 3 + 1], verts[vi0 * 3 + 2],
                        verts[vi1 * 3], verts[vi1 * 3 + 1], verts[vi1 * 3 + 2],
                        verts[vi2 * 3], verts[vi2 * 3 + 1], verts[vi2 * 3 + 2])
        if tt > 0 and (best_t < 0.0 or tt < best_t):
            best_t = tt
    return best_t


cdef inline double _slab_d(double ox, double oy, double oz,
                           double dx, double dy, double dz,
                           double bx0, double by0, double bz0,
                           double bx1, double by1, double bz1) noexcept nogil:
    cdef double tmin = -1e30
    cdef double tmax = 1e30
    cdef double t1, t2
    if fabs(dx) > 1e-30:
        t1 = (bx0 - ox) / dx
        t2 = (bx1 - ox) / dx
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
    elif ox < bx0 or ox > bx1:
        return -1.0
    if fabs(dy) > 1e-30:
        t1 = (by0 - oy) / dy
        t2 = (by1 - oy) / dy
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
    elif oy < by0 or oy > by1:
        return -1.0
    if fabs(dz) > 1e-30:
        t1 = (bz0 - oz) / dz
        t2 = (bz1 - oz) / dz
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
    elif oz < bz0 or oz > bz1:
        return -1.0
    if tmin > tmax:
        return -1.0
    if tmin > 0.0:
        return tmin
    return tmax if tmax > 0.0 else -1.0


def world_aabbs(
    np.ndarray[np.float64_t, ndim=2] bmins,
    np.ndarray[np.float64_t, ndim=2] bmaxs,
    np.ndarray[np.float64_t, ndim=3] wms,
):
    cdef int n = bmins.shape[0]
    cdef np.float64_t[:, :] wmn = np.empty((n, 3), dtype=np.float64)
    cdef np.float64_t[:, :] wmx = np.empty((n, 3), dtype=np.float64)
    cdef double[:, :] bmin = bmins
    cdef double[:, :] bmax = bmaxs
    cdef double[:, :, :] wm = wms
    cdef int i, k
    cdef double ax, ay, az, bx, by, bz
    cdef double c0, c1, c2
    cdef double wx, wy, wz
    cdef double minx, miny, minz, maxx, maxy, maxz
    for i in range(n):
        ax = bmin[i, 0]
        ay = bmin[i, 1]
        az = bmin[i, 2]
        bx = bmax[i, 0]
        by = bmax[i, 1]
        bz = bmax[i, 2]
        minx = 1e300
        miny = 1e300
        minz = 1e300
        maxx = -1e300
        maxy = -1e300
        maxz = -1e300
        for k in range(8):
            c0 = ax if (k & 1) == 0 else bx
            c1 = ay if (k & 2) == 0 else by
            c2 = az if (k & 4) == 0 else bz
            wx = wm[i, 0, 0] * c0 + wm[i, 1, 0] * c1 + wm[i, 2, 0] * c2 + wm[i, 3, 0]
            wy = wm[i, 0, 1] * c0 + wm[i, 1, 1] * c1 + wm[i, 2, 1] * c2 + wm[i, 3, 1]
            wz = wm[i, 0, 2] * c0 + wm[i, 1, 2] * c1 + wm[i, 2, 2] * c2 + wm[i, 3, 2]
            if wx < minx:
                minx = wx
            if wy < miny:
                miny = wy
            if wz < minz:
                minz = wz
            if wx > maxx:
                maxx = wx
            if wy > maxy:
                maxy = wy
            if wz > maxz:
                maxz = wz
        wmn[i, 0] = minx
        wmn[i, 1] = miny
        wmn[i, 2] = minz
        wmx[i, 0] = maxx
        wmx[i, 1] = maxy
        wmx[i, 2] = maxz
    return np.asarray(wmn), np.asarray(wmx)


def ray_aabbs(
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    np.ndarray[np.float64_t, ndim=2] bmins,
    np.ndarray[np.float64_t, ndim=2] bmaxs,
):
    cdef int n = bmins.shape[0]
    cdef np.uint8_t[:] hits = np.empty(n, dtype=np.uint8)
    cdef double[:, :] bmin = bmins
    cdef double[:, :] bmax = bmaxs
    cdef int i
    for i in range(n):
        hits[i] = (1 if _slab_d(ox, oy, oz, dx, dy, dz,
                                bmin[i, 0], bmin[i, 1], bmin[i, 2],
                                bmax[i, 0], bmax[i, 1], bmax[i, 2]) > 0.0 else 0)
    return np.asarray(hits)


def project_points(
    np.ndarray[np.float64_t, ndim=2] points,
    np.ndarray[np.float64_t, ndim=2] vp_mat,
    double vw, double vh,
):
    cdef int n = points.shape[0]
    cdef np.float64_t[:] sx = np.empty(n, dtype=np.float64)
    cdef np.float64_t[:] sy = np.empty(n, dtype=np.float64)
    cdef np.uint8_t[:] ok = np.empty(n, dtype=np.uint8)
    cdef double[:, :] p = points
    cdef double[:, :] m = vp_mat
    cdef int i
    cdef double x, y, z, cx, cy, cz, cw, ndcx, ndcy, ndcz
    for i in range(n):
        x = p[i, 0]
        y = p[i, 1]
        z = p[i, 2]
        cx = m[0, 0] * x + m[0, 1] * y + m[0, 2] * z + m[0, 3]
        cy = m[1, 0] * x + m[1, 1] * y + m[1, 2] * z + m[1, 3]
        cz = m[2, 0] * x + m[2, 1] * y + m[2, 2] * z + m[2, 3]
        cw = m[3, 0] * x + m[3, 1] * y + m[3, 2] * z + m[3, 3]
        if fabs(cw) < 1e-6:
            ok[i] = 0
            continue
        ndcx = cx / cw
        ndcy = cy / cw
        ndcz = cz / cw
        if ndcz < -1.0 or ndcz > 1.0:
            ok[i] = 0
            continue
        sx[i] = (ndcx + 1.0) * 0.5 * vw
        sy[i] = (1.0 - ndcy) * 0.5 * vh
        ok[i] = 1
    return np.asarray(sx), np.asarray(sy), np.asarray(ok)


def inv_affine4(np.ndarray[np.float64_t, ndim=2] m):
    cdef np.ndarray[np.float64_t, ndim=2] out = np.empty((4, 4), dtype=np.float64)
    cdef double[:, :] src = m
    cdef double[:, :] dst = out
    cdef double a, b, c, d, e, f, g, h, i2, det
    a = src[0, 0]; b = src[0, 1]; c = src[0, 2]
    d = src[1, 0]; e = src[1, 1]; f = src[1, 2]
    g = src[2, 0]; h = src[2, 1]; i2 = src[2, 2]
    det = a * (e * i2 - f * h) - b * (d * i2 - f * g) + c * (d * h - e * g)
    if fabs(det) < 1e-30:
        return np.linalg.inv(src)
    det = 1.0 / det
    dst[0, 0] = (e * i2 - f * h) * det
    dst[0, 1] = (c * h - b * i2) * det
    dst[0, 2] = (b * f - c * e) * det
    dst[1, 0] = (f * g - d * i2) * det
    dst[1, 1] = (a * i2 - c * g) * det
    dst[1, 2] = (c * d - a * f) * det
    dst[2, 0] = (d * h - e * g) * det
    dst[2, 1] = (b * g - a * h) * det
    dst[2, 2] = (a * e - b * d) * det
    dst[3, 0] = -(dst[0, 0] * src[3, 0] + dst[1, 0] * src[3, 1] + dst[2, 0] * src[3, 2])
    dst[3, 1] = -(dst[0, 1] * src[3, 0] + dst[1, 1] * src[3, 1] + dst[2, 1] * src[3, 2])
    dst[3, 2] = -(dst[0, 2] * src[3, 0] + dst[1, 2] * src[3, 1] + dst[2, 2] * src[3, 2])
    dst[3, 3] = 1.0
    return out
