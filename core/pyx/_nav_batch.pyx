# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, overflowcheck=False, infer_types=True
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs
cimport cython

cdef float _INF = 1e30
cdef float _SQRT2 = 1.41421356
cdef float _SQRT3 = 1.73205081

_DUMMY2 = np.zeros((1, 1), dtype=np.int32)

cdef int _DX2[8]
cdef int _DZ2[8]
_DX2[0] = 1
_DX2[1] = -1
_DX2[2] = 0
_DX2[3] = 0
_DX2[4] = 1
_DX2[5] = 1
_DX2[6] = -1
_DX2[7] = -1
_DZ2[0] = 0
_DZ2[1] = 0
_DZ2[2] = 1
_DZ2[3] = -1
_DZ2[4] = 1
_DZ2[5] = -1
_DZ2[6] = 1
_DZ2[7] = -1

cdef inline void _hpush(float *hf, float *hh, np.int32_t *hi, np.int32_t *ps, int *n, float f, float h, int idx) noexcept nogil:
    cdef int i, p
    if ps[idx] >= 0:
        i = ps[idx]
        if f >= hf[i]:
            return
        while i > 0:
            p = (i - 1) >> 1
            if hf[p] < f or (hf[p] == f and hh[p] <= h):
                break
            hf[i] = hf[p]
            hh[i] = hh[p]
            hi[i] = hi[p]
            ps[hi[i]] = i
            i = p
        hf[i] = f
        hh[i] = h
        hi[i] = idx
        ps[idx] = i
        return
    i = n[0]
    n[0] = i + 1
    while i > 0:
        p = (i - 1) >> 1
        if hf[p] < f or (hf[p] == f and hh[p] <= h):
            break
        hf[i] = hf[p]
        hh[i] = hh[p]
        hi[i] = hi[p]
        ps[hi[i]] = i
        i = p
    hf[i] = f
    hh[i] = h
    hi[i] = idx
    ps[idx] = i

cdef inline int _hpop(float *hf, float *hh, np.int32_t *hi, np.int32_t *ps, int *n, float *f, float *h) noexcept nogil:
    cdef int nn = n[0] - 1
    cdef int idx = hi[0]
    cdef float lf, lh
    cdef int li, i = 0, l, r, m
    f[0] = hf[0]
    h[0] = hh[0]
    ps[idx] = -1
    if nn <= 0:
        n[0] = 0
        return idx
    lf = hf[nn]
    lh = hh[nn]
    li = hi[nn]
    n[0] = nn
    while True:
        l = (i << 1) + 1
        if l >= nn:
            break
        r = l + 1
        m = l
        if r < nn and (hf[r] < hf[l] or (hf[r] == hf[l] and hh[r] < hh[l])):
            m = r
        if hf[m] > lf or (hf[m] == lf and hh[m] >= lh):
            break
        hf[i] = hf[m]
        hh[i] = hh[m]
        hi[i] = hi[m]
        ps[hi[i]] = i
        i = m
    hf[i] = lf
    hh[i] = lh
    hi[i] = li
    ps[li] = i
    return idx

cdef inline float _octile2(int dx, int dz) noexcept nogil:
    if dx < 0:
        dx = -dx
    if dz < 0:
        dz = -dz
    if dx > dz:
        return <float>dx + 0.41421356 * <float>dz
    return <float>dz + 0.41421356 * <float>dx

cdef inline float _octile3(int dx, int dy, int dz) noexcept nogil:
    cdef int t
    if dx < 0:
        dx = -dx
    if dy < 0:
        dy = -dy
    if dz < 0:
        dz = -dz
    if dx < dy:
        t = dx
        dx = dy
        dy = t
    if dy < dz:
        t = dy
        dy = dz
        dz = t
    if dx < dy:
        t = dx
        dx = dy
        dy = t
    return <float>dx + 0.41421356 * <float>dy + 0.31783725 * <float>dz

def build_ground_fields(np.ndarray[np.uint8_t, ndim=3] raw, int hc, float max_hdiff, float cell, float w_slope, float w_climb,
                        np.ndarray[np.uint8_t, ndim=2] walk, np.ndarray[np.int32_t, ndim=2] ground, np.ndarray[np.float32_t, ndim=2] cost):
    cdef int r = raw.shape[0]
    cdef unsigned char[:, :, :] rv = raw
    cdef unsigned char[:, :] wv = walk
    cdef np.int32_t[:, :] gv = ground
    cdef float[:, :] cv = cost
    cdef int x, z, y, lim, gy
    cdef bint ok
    cdef int g0, g1
    cdef float dh, tot, up, n
    with nogil:
        for x in range(r):
            for z in range(r):
                gv[x, z] = -1
                wv[x, z] = 0
                cv[x, z] = 1.0
                for y in range(r - 1, -1, -1):
                    if rv[x, y, z]:
                        lim = y + 2 + hc
                        if lim > r:
                            lim = r
                        ok = True
                        gy = y + 1
                        while gy < lim:
                            if rv[x, gy, z]:
                                ok = False
                                break
                            gy += 1
                        if ok:
                            gv[x, z] = y
                            wv[x, z] = 1
                            break
        for x in range(r):
            for z in range(r):
                if wv[x, z]:
                    g0 = gv[x, z]
                    if x > 0 and gv[x - 1, z] >= 0 and fabs(<float>(g0 - gv[x - 1, z])) > max_hdiff:
                        wv[x, z] = 0
                    elif x + 1 < r and gv[x + 1, z] >= 0 and fabs(<float>(g0 - gv[x + 1, z])) > max_hdiff:
                        wv[x, z] = 0
                    elif z > 0 and gv[x, z - 1] >= 0 and fabs(<float>(g0 - gv[x, z - 1])) > max_hdiff:
                        wv[x, z] = 0
                    elif z + 1 < r and gv[x, z + 1] >= 0 and fabs(<float>(g0 - gv[x, z + 1])) > max_hdiff:
                        wv[x, z] = 0
        for x in range(r):
            for z in range(r):
                if wv[x, z]:
                    g0 = gv[x, z]
                    tot = 0.0
                    up = 0.0
                    n = 0.0
                    if x > 0 and gv[x - 1, z] >= 0:
                        dh = <float>(gv[x - 1, z] - g0) * cell
                        tot += fabs(dh)
                        if dh > up:
                            up = dh
                        n += 1.0
                    if x + 1 < r and gv[x + 1, z] >= 0:
                        dh = <float>(gv[x + 1, z] - g0) * cell
                        tot += fabs(dh)
                        if dh > up:
                            up = dh
                        n += 1.0
                    if z > 0 and gv[x, z - 1] >= 0:
                        dh = <float>(gv[x, z - 1] - g0) * cell
                        tot += fabs(dh)
                        if dh > up:
                            up = dh
                        n += 1.0
                    if z + 1 < r and gv[x, z + 1] >= 0:
                        dh = <float>(gv[x, z + 1] - g0) * cell
                        tot += fabs(dh)
                        if dh > up:
                            up = dh
                        n += 1.0
                    if n > 0.0:
                        cv[x, z] = 1.0 + w_slope * (tot / n) + w_climb * up
                    else:
                        cv[x, z] = 1.0

def collect_candidates(np.ndarray[np.uint8_t, ndim=3] raw, int hc,
                        np.ndarray[np.int32_t, ndim=3] cands, np.ndarray[np.uint8_t, ndim=2] ncnt):
    cdef int r = raw.shape[0]
    cdef unsigned char[:, :, :] rv = raw
    cdef np.int32_t[:, :, :] cv = cands
    cdef unsigned char[:, :] nv = ncnt
    cdef int x, z, y, lim, gy, n
    cdef bint ok
    with nogil:
        for x in range(r):
            for z in range(r):
                n = 0
                for y in range(r - 1, -1, -1):
                    if rv[x, y, z]:
                        lim = y + 2 + hc
                        if lim > r:
                            lim = r
                        ok = True
                        gy = y + 1
                        while gy < lim:
                            if rv[x, gy, z]:
                                ok = False
                                break
                            gy += 1
                        if ok:
                            cv[x, z, n] = y
                            n += 1
                            if n >= 4:
                                break
                nv[x, z] = n

def flood_levels(np.ndarray[np.int32_t, ndim=3] cands, np.ndarray[np.uint8_t, ndim=2] ncnt,
                 int sx, int sz, int ref_y, float cell, float climb,
                 np.ndarray[np.int32_t, ndim=2] ground, np.ndarray[np.uint8_t, ndim=2] walk):
    cdef int r = ground.shape[0]
    cdef np.int32_t[:, :, :] cv = cands
    cdef unsigned char[:, :] nv = ncnt
    cdef np.int32_t[:, :] gv = ground
    cdef unsigned char[:, :] wv = walk
    cdef int x, z, n0, i, y0, best, bd, dd
    cdef int rad, dx, dz, nx, nz
    cdef bint found = False
    for x in range(r):
        for z in range(r):
            gv[x, z] = -1
            wv[x, z] = 0
    n0 = 0
    if 0 <= sx < r and 0 <= sz < r:
        n0 = nv[sx, sz]
    if n0 <= 0:
        for rad in range(1, 25):
            if found:
                break
            for dx in range(-rad, rad + 1):
                for dz in (-rad, rad):
                    nx = sx + dx
                    nz = sz + dz
                    if 0 <= nx < r and 0 <= nz < r and nv[nx, nz] > 0:
                        sx = nx
                        sz = nz
                        n0 = nv[nx, nz]
                        found = True
                        break
                if found:
                    break
            if found:
                break
            for dz in range(-rad + 1, rad):
                for dx in (-rad, rad):
                    nx = sx + dx
                    nz = sz + dz
                    if 0 <= nx < r and 0 <= nz < r and nv[nx, nz] > 0:
                        sx = nx
                        sz = nz
                        n0 = nv[nx, nz]
                        found = True
                        break
                if found:
                    break
    if n0 <= 0:
        return False
    best = cv[sx, sz, 0]
    bd = best - ref_y
    if bd < 0:
        bd = -bd
    for i in range(1, n0):
        y0 = cv[sx, sz, i]
        dd = y0 - ref_y
        if dd < 0:
            dd = -dd
        if dd < bd or (dd == bd and y0 < best):
            bd = dd
            best = y0
    cdef np.ndarray[np.int32_t, ndim=1] stack = np.empty(r * r, dtype=np.int32)
    cdef np.int32_t[:] st = stack
    cdef int sp = 0, cur, cx, cz, d, cy2, k, nk, cand, dd2, bk, bd2
    cdef float dh
    gv[sx, sz] = best
    wv[sx, sz] = 1
    st[sp] = sx * r + sz
    sp += 1
    while sp > 0:
        sp -= 1
        cur = st[sp]
        cx = cur // r
        cz = cur - cx * r
        cy2 = gv[cx, cz]
        for d in range(4):
            nx = cx + _DX2[d]
            nz = cz + _DZ2[d]
            if nx < 0 or nx >= r or nz < 0 or nz >= r:
                continue
            if wv[nx, nz]:
                continue
            nk = nv[nx, nz]
            if nk <= 0:
                continue
            bk = -1
            bd2 = 0
            for k in range(nk):
                cand = cv[nx, nz, k]
                dh = <float>(cand - cy2) * cell
                if dh > climb:
                    continue
                dd2 = cand - cy2
                if dd2 < 0:
                    dd2 = -dd2
                if bk < 0 or dd2 < bd2:
                    bd2 = dd2
                    bk = cand
            if bk < 0:
                continue
            gv[nx, nz] = bk
            wv[nx, nz] = 1
            st[sp] = nx * r + nz
            sp += 1
    return True

def finalize_ground(np.ndarray[np.int32_t, ndim=2] ground, np.ndarray[np.uint8_t, ndim=2] walk,
                    float max_hdiff, float cell, float w_slope, float w_climb,
                    np.ndarray[np.float32_t, ndim=2] cost):
    cdef int r = ground.shape[0]
    cdef np.int32_t[:, :] gv = ground
    cdef unsigned char[:, :] wv = walk
    cdef float[:, :] cv = cost
    cdef int x, z, g0
    cdef float dh, tot, up, n
    with nogil:
        for x in range(r):
            for z in range(r):
                if wv[x, z]:
                    g0 = gv[x, z]
                    if x > 0 and gv[x - 1, z] >= 0 and fabs(<float>(g0 - gv[x - 1, z])) > max_hdiff:
                        wv[x, z] = 0
                    elif x + 1 < r and gv[x + 1, z] >= 0 and fabs(<float>(g0 - gv[x + 1, z])) > max_hdiff:
                        wv[x, z] = 0
                    elif z > 0 and gv[x, z - 1] >= 0 and fabs(<float>(g0 - gv[x, z - 1])) > max_hdiff:
                        wv[x, z] = 0
                    elif z + 1 < r and gv[x, z + 1] >= 0 and fabs(<float>(g0 - gv[x, z + 1])) > max_hdiff:
                        wv[x, z] = 0
        for x in range(r):
            for z in range(r):
                if wv[x, z]:
                    g0 = gv[x, z]
                    tot = 0.0
                    up = 0.0
                    n = 0.0
                    if x > 0 and gv[x - 1, z] >= 0:
                        dh = <float>(gv[x - 1, z] - g0) * cell
                        tot += fabs(dh)
                        if dh > up:
                            up = dh
                        n += 1.0
                    if x + 1 < r and gv[x + 1, z] >= 0:
                        dh = <float>(gv[x + 1, z] - g0) * cell
                        tot += fabs(dh)
                        if dh > up:
                            up = dh
                        n += 1.0
                    if z > 0 and gv[x, z - 1] >= 0:
                        dh = <float>(gv[x, z - 1] - g0) * cell
                        tot += fabs(dh)
                        if dh > up:
                            up = dh
                        n += 1.0
                    if z + 1 < r and gv[x, z + 1] >= 0:
                        dh = <float>(gv[x, z + 1] - g0) * cell
                        tot += fabs(dh)
                        if dh > up:
                            up = dh
                        n += 1.0
                    if n > 0.0:
                        cv[x, z] = 1.0 + w_slope * (tot / n) + w_climb * up
                    else:
                        cv[x, z] = 1.0

def raster_box_obb(np.ndarray[np.uint8_t, ndim=3] grid, float cell, float half,
                   float cx, float cy, float cz,
                   float hx, float hy, float hz,
                   float qx, float qy, float qz, float qw):
    cdef int r = grid.shape[0]
    cdef unsigned char[:, :, :] gv = grid
    cdef double nq = <double>qx * qx + <double>qy * qy + <double>qz * qz + <double>qw * qw
    cdef double inv = 1.0
    cdef double xx, yy, zz, ww, xy, xz, xw, yz, yw, zw
    cdef double m00, m01, m02, m10, m11, m12, m20, m21, m22
    cdef double lx, ly, lz, wx, wy, wz, dx, dy, dz
    cdef double bminx = 1e30, bminy = 1e30, bminz = 1e30
    cdef double bmaxx = -1e30, bmaxy = -1e30, bmaxz = -1e30
    cdef double px, py, pz, qx_, qy_, qz_
    cdef int ix, iy, iz, x1, x2, y1, y2, z1, z2, sx, sy, sz, filled = 0
    cdef int thin = 0
    cdef double ox[2]
    if nq > 1e-12:
        inv = 1.0 / sqrt(nq)
    xx = qx * inv
    yy = qy * inv
    zz = qz * inv
    ww = qw * inv
    xy = xx * yy
    xz = xx * zz
    xw = xx * ww
    yz = yy * zz
    yw = yy * ww
    zw = zz * ww
    m00 = 1.0 - 2.0 * (yy * yy + zz * zz)
    m01 = 2.0 * (xy - zw)
    m02 = 2.0 * (xz + yw)
    m10 = 2.0 * (xy + zw)
    m11 = 1.0 - 2.0 * (xx * xx + zz * zz)
    m12 = 2.0 * (yz - xw)
    m20 = 2.0 * (xz - yw)
    m21 = 2.0 * (yz + xw)
    m22 = 1.0 - 2.0 * (xx * xx + yy * yy)
    for ix in range(8):
        px = hx if (ix & 1) else -hx
        py = hy if (ix & 2) else -hy
        pz = hz if (ix & 4) else -hz
        qx_ = m00 * px + m01 * py + m02 * pz + cx
        qy_ = m10 * px + m11 * py + m12 * pz + cy
        qz_ = m20 * px + m21 * py + m22 * pz + cz
        if qx_ < bminx:
            bminx = qx_
        if qx_ > bmaxx:
            bmaxx = qx_
        if qy_ < bminy:
            bminy = qy_
        if qy_ > bmaxy:
            bmaxy = qy_
        if qz_ < bminz:
            bminz = qz_
        if qz_ > bmaxz:
            bmaxz = qz_
    x1 = <int>((bminx + half) / cell)
    y1 = <int>((bminy + half) / cell)
    z1 = <int>((bminz + half) / cell)
    x2 = <int>((bmaxx + half) / cell)
    y2 = <int>((bmaxy + half) / cell)
    z2 = <int>((bmaxz + half) / cell)
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if z1 < 0:
        z1 = 0
    if x2 >= r:
        x2 = r - 1
    if y2 >= r:
        y2 = r - 1
    if z2 >= r:
        z2 = r - 1
    if x1 > x2 or y1 > y2 or z1 > z2:
        return 0
    if hx < cell * 0.75 or hy < cell * 0.75 or hz < cell * 0.75:
        thin = 1
    ox[0] = -0.25 * cell
    ox[1] = 0.25 * cell
    for ix in range(x1, x2 + 1):
        wx = -half + (<double>ix + 0.5) * <double>cell
        for iy in range(y1, y2 + 1):
            wy = -half + (<double>iy + 0.5) * <double>cell
            for iz in range(z1, z2 + 1):
                wz = -half + (<double>iz + 0.5) * <double>cell
                dx = wx - cx
                dy = wy - cy
                dz = wz - cz
                lx = m00 * dx + m10 * dy + m20 * dz
                ly = m01 * dx + m11 * dy + m21 * dz
                lz = m02 * dx + m12 * dy + m22 * dz
                if fabs(lx) <= hx and fabs(ly) <= hy and fabs(lz) <= hz:
                    if gv[ix, iy, iz] == 0:
                        gv[ix, iy, iz] = 1
                        filled += 1
                    continue
                if thin:
                    for sx in range(2):
                        if gv[ix, iy, iz]:
                            break
                        for sy in range(2):
                            if gv[ix, iy, iz]:
                                break
                            for sz in range(2):
                                dx = wx + ox[sx] - cx
                                dy = wy + ox[sy] - cy
                                dz = wz + ox[sz] - cz
                                lx = m00 * dx + m10 * dy + m20 * dz
                                ly = m01 * dx + m11 * dy + m21 * dz
                                lz = m02 * dx + m12 * dy + m22 * dz
                                if fabs(lx) <= hx and fabs(ly) <= hy and fabs(lz) <= hz:
                                    gv[ix, iy, iz] = 1
                                    filled += 1
                                    break
    return filled

def dilate2d(np.ndarray[np.uint8_t, ndim=2] walk, int radius, np.ndarray[np.uint8_t, ndim=2] out):
    cdef int r = walk.shape[0]
    cdef unsigned char[:, :] wv = walk
    cdef unsigned char[:, :] ov = out
    cdef np.ndarray[np.int32_t, ndim=2] sat = np.zeros((r + 1, r + 1), dtype=np.int32)
    cdef np.int32_t[:, :] sv = sat
    cdef int x, z, x1, x2, z1, z2, s
    with nogil:
        for x in range(r):
            for z in range(r):
                sv[x + 1, z + 1] = (<int>(1 - wv[x, z])) + sv[x, z + 1] + sv[x + 1, z] - sv[x, z]
        for x in range(r):
            x1 = x - radius
            if x1 < 0:
                x1 = 0
            x2 = x + radius + 1
            if x2 > r:
                x2 = r
            for z in range(r):
                z1 = z - radius
                if z1 < 0:
                    z1 = 0
                z2 = z + radius + 1
                if z2 > r:
                    z2 = r
                s = sv[x2, z2] - sv[x1, z2] - sv[x2, z1] + sv[x1, z1]
                ov[x, z] = 0 if s > 0 else 1

def label_components(np.ndarray[np.uint8_t, ndim=2] walk, np.ndarray[np.int32_t, ndim=2] ground, float cell, float climb,
                     np.ndarray[np.int32_t, ndim=2] labels, np.ndarray[np.int32_t, ndim=1] sizes):
    cdef int r = walk.shape[0]
    cdef unsigned char[:, :] wv = walk
    cdef np.int32_t[:, :] gv = ground
    cdef np.int32_t[:, :] lv = labels
    cdef np.int32_t[:] szv = sizes
    cdef np.ndarray[np.int32_t, ndim=1] stack = np.empty(r * r, dtype=np.int32)
    cdef np.int32_t[:] st = stack
    cdef int x, z, ncomp = 0, sp, cur, cx, cz, nx, nz, d
    cdef float dh
    cdef bint use_climb = climb >= 0.0
    for x in range(r):
        for z in range(r):
            lv[x, z] = -1
    for x in range(r):
        for z in range(r):
            szv[x * r + z] = 0
    for x in range(r):
        for z in range(r):
            if wv[x, z] == 0 or lv[x, z] >= 0:
                continue
            sp = 0
            st[sp] = x * r + z
            sp += 1
            lv[x, z] = ncomp
            while sp > 0:
                sp -= 1
                cur = st[sp]
                cx = cur // r
                cz = cur - cx * r
                szv[ncomp] += 1
                for d in range(4):
                    nx = cx + _DX2[d]
                    nz = cz + _DZ2[d]
                    if nx < 0 or nx >= r or nz < 0 or nz >= r:
                        continue
                    if wv[nx, nz] == 0 or lv[nx, nz] >= 0:
                        continue
                    if use_climb and gv[cx, cz] >= 0 and gv[nx, nz] >= 0:
                        dh = <float>(gv[nx, nz] - gv[cx, cz]) * cell
                        if dh > climb:
                            continue
                    lv[nx, nz] = ncomp
                    st[sp] = nx * r + nz
                    sp += 1
            ncomp += 1
    return ncomp

def snap2d(np.ndarray[np.uint8_t, ndim=2] walk, int gx, int gz, int max_rad):
    cdef int r = walk.shape[0]
    cdef unsigned char[:, :] wv = walk
    cdef int rad, dx, dz, nx, nz
    if 0 <= gx < r and 0 <= gz < r and wv[gx, gz]:
        return (gx, gz)
    for rad in range(1, max_rad + 1):
        for dx in range(-rad, rad + 1):
            for dz in (-rad, rad):
                nx = gx + dx
                nz = gz + dz
                if 0 <= nx < r and 0 <= nz < r and wv[nx, nz]:
                    return (nx, nz)
        for dz in range(-rad + 1, rad):
            for dx in (-rad, rad):
                nx = gx + dx
                nz = gz + dz
                if 0 <= nx < r and 0 <= nz < r and wv[nx, nz]:
                    return (nx, nz)
    return (gx, gz)

cdef int _astar2d_run(unsigned char *walk, float *cost, np.int32_t *ground, bint use_climb, float climb_cells,
                      int r, int sx, int sz, int ex, int ez, int max_exp,
                      int x0, int x1, int z0, int z1,
                      float *gf, float *ff, float *hf, np.int32_t *hi, np.int32_t *ps,
                      np.int32_t *came, unsigned char *closed,
                      np.int32_t *path, int *out_exp, int *out_len) noexcept nogil:
    cdef int N = r * r
    cdef int i, hn = 0, exp = 0, cur, cx, cz, d, nx, nz, nidx, status = 1
    cdef int ci, ni
    cdef float f, h, gcur, ng, step, best_h, hh
    cdef int best_idx = sx * r + sz
    cdef int start = sx * r + sz
    cdef int goal = ex * r + ez
    for i in range(N):
        gf[i] = _INF
        ps[i] = -1
    best_h = _octile2(ex - sx, ez - sz)
    gf[start] = 0.0
    _hpush(ff, hf, hi, ps, &hn, best_h, best_h, start)
    while hn > 0:
        if exp >= max_exp:
            status = 2
            break
        cur = _hpop(ff, hf, hi, ps, &hn, &f, &h)
        cx = cur // r
        cz = cur - cx * r
        if closed[cur]:
            continue
        closed[cur] = 1
        exp += 1
        if cur == goal:
            status = 0
            best_idx = cur
            break
        if h < best_h:
            best_h = h
            best_idx = cur
        gcur = gf[cur]
        ci = cur
        for d in range(8):
            nx = cx + _DX2[d]
            nz = cz + _DZ2[d]
            if nx < x0 or nx > x1 or nz < z0 or nz > z1:
                continue
            nidx = nx * r + nz
            if walk[nidx] == 0 or closed[nidx]:
                continue
            if d >= 4:
                if walk[cx * r + nz] == 0 or walk[nx * r + cz] == 0:
                    continue
            if use_climb and ground[ci] >= 0 and ground[nidx] >= 0:
                if <float>(ground[nidx] - ground[ci]) > climb_cells:
                    continue
            step = 1.0 if d < 4 else _SQRT2
            ng = gcur + step * (0.5 * (cost[ci] + cost[nidx]))
            if ng < gf[nidx]:
                gf[nidx] = ng
                came[nidx] = cur
                hh = _octile2(ex - nx, ez - nz)
                _hpush(ff, hf, hi, ps, &hn, ng + hh, hh, nidx)
    out_exp[0] = exp
    cdef int plen = 0, c = best_idx
    if came[c] == -1 and c != start:
        path[0] = start
        out_len[0] = 1
        return status if status == 2 else 1
    while True:
        path[plen] = c
        plen += 1
        if c == start:
            break
        c = came[c]
    cdef int a = 0, b = plen - 1, t
    while a < b:
        t = path[a]
        path[a] = path[b]
        path[b] = t
        a += 1
        b -= 1
    out_len[0] = plen
    return status

def astar2d(np.ndarray[np.uint8_t, ndim=2] walk, np.ndarray[np.float32_t, ndim=2] cost, object ground,
            int sx, int sz, int ex, int ez, int max_exp, float climb_cells,
            int x0, int x1, int z0, int z1):
    cdef int r = walk.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=2] walk_c = np.ascontiguousarray(walk, dtype=np.uint8)
    cdef np.ndarray[np.float32_t, ndim=2] cost_c = np.ascontiguousarray(cost, dtype=np.float32)
    cdef bint use_climb = ground is not None and climb_cells >= 0.0
    cdef np.ndarray[np.int32_t, ndim=2] ground_c = np.ascontiguousarray(ground, dtype=np.int32) if use_climb else _DUMMY2
    cdef int N = r * r
    cdef np.ndarray[np.float32_t, ndim=1] gflat = np.full(N, 1e30, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] hheap = np.empty(N, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] fheap = np.empty(N, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] iheap = np.empty(N, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] pos = np.full(N, -1, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] came = np.full(N, -1, dtype=np.int32)
    cdef np.ndarray[np.uint8_t, ndim=1] closed = np.zeros(N, dtype=np.uint8)
    cdef np.ndarray[np.int32_t, ndim=1] path = np.empty(N, dtype=np.int32)
    cdef unsigned char *wptr = &walk_c[0, 0]
    cdef float *cptr = &cost_c[0, 0]
    cdef np.int32_t *gptr = &ground_c[0, 0]
    cdef float *gf = &gflat[0]
    cdef float *ff = &fheap[0]
    cdef float *hf = &hheap[0]
    cdef np.int32_t *hi = &iheap[0]
    cdef np.int32_t *ps = &pos[0]
    cdef np.int32_t *cm = &came[0]
    cdef unsigned char *cl = &closed[0]
    cdef np.int32_t *pv = &path[0]
    cdef int out_exp = 0
    cdef int out_len = 0
    cdef int status
    with nogil:
        status = _astar2d_run(wptr, cptr, gptr, use_climb, climb_cells, r, sx, sz, ex, ez, max_exp,
                              x0, x1, z0, z1, gf, ff, hf, hi, ps, cm, cl, pv, &out_exp, &out_len)
    return (status, path[:out_len], out_exp)

def los2d(np.ndarray[np.uint8_t, ndim=2] walk, int x0, int z0, int x1, int z1):
    cdef int r = walk.shape[0]
    cdef unsigned char[:, :] wv = walk
    cdef float dx = <float>(x1 - x0), dz = <float>(z1 - z0)
    cdef int sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
    cdef int sz = 1 if dz > 0 else (-1 if dz < 0 else 0)
    cdef float tdx = 1e30, tdz = 1e30, tmx = 1e30, tmz = 1e30
    cdef int gx = x0, gz = z0, guard = 0
    if gx < 0 or gx >= r or gz < 0 or gz >= r or wv[gx, gz] == 0:
        return False
    if gx == x1 and gz == z1:
        return True
    if dx != 0.0:
        tdx = fabs(1.0 / dx)
        tmx = 0.5 * tdx
    if dz != 0.0:
        tdz = fabs(1.0 / dz)
        tmz = 0.5 * tdz
    while guard < 4 * r + 16:
        guard += 1
        if tmx < tmz:
            gx += sx
            tmx += tdx
        elif tmz < tmx:
            gz += sz
            tmz += tdz
        else:
            gx += sx
            tmx += tdx
        if gx < 0 or gx >= r or gz < 0 or gz >= r or wv[gx, gz] == 0:
            return False
        if gx == x1 and gz == z1:
            return True
    return False

def los2d_climb(np.ndarray[np.uint8_t, ndim=2] walk, np.ndarray[np.int32_t, ndim=2] ground,
                int x0, int z0, int x1, int z1, float climb_cells):
    cdef int r = walk.shape[0]
    cdef unsigned char[:, :] wv = walk
    cdef np.int32_t[:, :] gv = ground
    cdef float dx = <float>(x1 - x0), dz = <float>(z1 - z0)
    cdef int sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
    cdef int sz = 1 if dz > 0 else (-1 if dz < 0 else 0)
    cdef float tdx = 1e30, tdz = 1e30, tmx = 1e30, tmz = 1e30
    cdef int gx = x0, gz = z0, guard = 0
    cdef int px = x0, pz = z0
    cdef float dh
    if gx < 0 or gx >= r or gz < 0 or gz >= r or wv[gx, gz] == 0:
        return False
    if gx == x1 and gz == z1:
        return True
    if dx != 0.0:
        tdx = fabs(1.0 / dx)
        tmx = 0.5 * tdx
    if dz != 0.0:
        tdz = fabs(1.0 / dz)
        tmz = 0.5 * tdz
    while guard < 4 * r + 16:
        guard += 1
        if tmx < tmz:
            gx += sx
            tmx += tdx
        elif tmz < tmx:
            gz += sz
            tmz += tdz
        else:
            gx += sx
            tmx += tdx
        if gx < 0 or gx >= r or gz < 0 or gz >= r or wv[gx, gz] == 0:
            return False
        if gv[gx, gz] >= 0 and gv[px, pz] >= 0:
            dh = <float>(gv[gx, gz] - gv[px, pz])
            if dh > climb_cells:
                return False
        px = gx
        pz = gz
        if gx == x1 and gz == z1:
            return True
    return False

def los3d_clear(np.ndarray[np.uint8_t, ndim=3] blocked, int x0, int y0, int z0, int x1, int y1, int z1, int rad):
    cdef int r = blocked.shape[0]
    cdef unsigned char[:, :, :] bv = blocked
    cdef float dx = <float>(x1 - x0), dy = <float>(y1 - y0), dz = <float>(z1 - z0)
    cdef int sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
    cdef int sy = 1 if dy > 0 else (-1 if dy < 0 else 0)
    cdef int sz = 1 if dz > 0 else (-1 if dz < 0 else 0)
    cdef float tdx = 1e30, tdy = 1e30, tdz = 1e30, tmx = 1e30, tmy = 1e30, tmz = 1e30
    cdef int gx = x0, gy = y0, gz = z0, guard = 0
    cdef int ix, iy, iz, x1b, x2b, y1b, y2b, z1b, z2b
    if rad < 0:
        rad = 0
    if gx < 0 or gx >= r or gy < 0 or gy >= r or gz < 0 or gz >= r:
        return False
    if gx == x1 and gy == y1 and gz == z1:
        return bv[gx, gy, gz] == 0
    if dx != 0.0:
        tdx = fabs(1.0 / dx)
        tmx = 0.5 * tdx
    if dy != 0.0:
        tdy = fabs(1.0 / dy)
        tmy = 0.5 * tdy
    if dz != 0.0:
        tdz = fabs(1.0 / dz)
        tmz = 0.5 * tdz
    while guard < 6 * r + 32:
        guard += 1
        if tmx < tmy and tmx < tmz:
            gx += sx
            tmx += tdx
        elif tmy < tmz:
            gy += sy
            tmy += tdy
        elif tmz < tmy:
            gz += sz
            tmz += tdz
        elif tmx <= tmy and tmx <= tmz:
            gx += sx
            tmx += tdx
        elif tmy <= tmz:
            gy += sy
            tmy += tdy
        else:
            gz += sz
            tmz += tdz
        if gx < 0 or gx >= r or gy < 0 or gy >= r or gz < 0 or gz >= r:
            return False
        if rad == 0:
            if bv[gx, gy, gz]:
                return False
        else:
            x1b = gx - rad
            if x1b < 0:
                x1b = 0
            x2b = gx + rad + 1
            if x2b > r:
                x2b = r
            y1b = gy - rad
            if y1b < 0:
                y1b = 0
            y2b = gy + rad + 1
            if y2b > r:
                y2b = r
            z1b = gz - rad
            if z1b < 0:
                z1b = 0
            z2b = gz + rad + 1
            if z2b > r:
                z2b = r
            for ix in range(x1b, x2b):
                for iy in range(y1b, y2b):
                    for iz in range(z1b, z2b):
                        if bv[ix, iy, iz]:
                            return False
        if gx == x1 and gy == y1 and gz == z1:
            return True
    return False

def smooth2d(np.ndarray[np.int32_t, ndim=1] path, int n, np.ndarray[np.uint8_t, ndim=2] walk):
    cdef int r = walk.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] out = np.empty(n, dtype=np.int32)
    cdef int i = 0, j, m = 0
    cdef int ax, az, bx, bz
    while i < n - 1:
        j = n - 1
        ax = path[i] // r
        az = path[i] - ax * r
        while j > i + 1:
            bx = path[j] // r
            bz = path[j] - bx * r
            if los2d(walk, ax, az, bx, bz):
                break
            j -= 1
        out[m] = path[i]
        m += 1
        i = j
    out[m] = path[n - 1]
    m += 1
    return out[:m]

def smooth2d_climb(np.ndarray[np.int32_t, ndim=1] path, int n, np.ndarray[np.uint8_t, ndim=2] walk,
                   np.ndarray[np.int32_t, ndim=2] ground, float climb_cells):
    cdef int r = walk.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] out = np.empty(n, dtype=np.int32)
    cdef int i = 0, j, m = 0
    cdef int ax, az, bx, bz
    while i < n - 1:
        j = n - 1
        ax = path[i] // r
        az = path[i] - ax * r
        while j > i + 1:
            bx = path[j] // r
            bz = path[j] - bx * r
            if los2d_climb(walk, ground, ax, az, bx, bz, climb_cells):
                break
            j -= 1
        out[m] = path[i]
        m += 1
        i = j
    out[m] = path[n - 1]
    m += 1
    return out[:m]

def nearest3d(np.ndarray[np.uint8_t, ndim=3] blocked, int gx, int gy, int gz, int max_rad):
    cdef int r = blocked.shape[0]
    cdef unsigned char[:, :, :] bv = blocked
    cdef int rad, dx, dy, dz, nx, ny, nz
    if 0 <= gx < r and 0 <= gy < r and 0 <= gz < r and bv[gx, gy, gz] == 0:
        return (gx, gy, gz)
    for rad in range(1, max_rad + 1):
        for dx in range(-rad, rad + 1):
            for dy in range(-rad, rad + 1):
                for dz in (-rad, rad):
                    nx = gx + dx
                    ny = gy + dy
                    nz = gz + dz
                    if 0 <= nx < r and 0 <= ny < r and 0 <= nz < r and bv[nx, ny, nz] == 0:
                        return (nx, ny, nz)
        for dx in range(-rad, rad + 1):
            for dz in range(-rad + 1, rad):
                for dy in (-rad, rad):
                    nx = gx + dx
                    ny = gy + dy
                    nz = gz + dz
                    if 0 <= nx < r and 0 <= ny < r and 0 <= nz < r and bv[nx, ny, nz] == 0:
                        return (nx, ny, nz)
        for dy in range(-rad + 1, rad):
            for dz in range(-rad + 1, rad):
                for dx in (-rad, rad):
                    nx = gx + dx
                    ny = gy + dy
                    nz = gz + dz
                    if 0 <= nx < r and 0 <= ny < r and 0 <= nz < r and bv[nx, ny, nz] == 0:
                        return (nx, ny, nz)
    return (gx, gy, gz)

cdef int _astar3d_run(unsigned char *bv, int r,
                      int sx, int sy, int sz, int ex, int ey, int ez, int max_exp,
                      int x0, int x1, int y0, int y1, int z0, int z1,
                      float *gf, float *ff, float *hf, np.int32_t *hi, np.int32_t *ps,
                      np.int32_t *came, unsigned char *closed,
                      np.int32_t *path, int *out_exp, int *out_len) noexcept nogil:
    cdef int N = r * r * r
    cdef int rr = r * r
    cdef int i, hn = 0, exp = 0, cur, cx, cy, cz, ax, ay, az, nx, ny, nz, nidx, status = 1
    cdef int c0, c1, c2
    cdef float f, h, gcur, ng, step, best_h
    cdef int start = (sx * r + sy) * r + sz
    cdef int goal = (ex * r + ey) * r + ez
    cdef int best_idx = start
    for i in range(N):
        gf[i] = _INF
        ps[i] = -1
    best_h = _octile3(ex - sx, ey - sy, ez - sz)
    gf[start] = 0.0
    _hpush(ff, hf, hi, ps, &hn, best_h, best_h, start)
    while hn > 0:
        if exp >= max_exp:
            status = 2
            break
        cur = _hpop(ff, hf, hi, ps, &hn, &f, &h)
        cx = cur // rr
        cy = (cur - cx * rr) // r
        cz = cur - cx * rr - cy * r
        if closed[cur]:
            continue
        closed[cur] = 1
        exp += 1
        if cur == goal:
            status = 0
            best_idx = cur
            break
        if h < best_h:
            best_h = h
            best_idx = cur
        gcur = gf[cur]
        c0 = cur
        for ax in range(-1, 2):
            nx = cx + ax
            if nx < x0 or nx > x1:
                continue
            for ay in range(-1, 2):
                ny = cy + ay
                if ny < y0 or ny > y1:
                    continue
                for az in range(-1, 2):
                    if ax == 0 and ay == 0 and az == 0:
                        continue
                    nz = cz + az
                    if nz < z0 or nz > z1:
                        continue
                    nidx = (nx * r + ny) * r + nz
                    if bv[nidx] or closed[nidx]:
                        continue
                    if ax != 0:
                        c1 = c0 + ax * rr
                        if bv[c1]:
                            continue
                    if ay != 0:
                        c1 = c0 + ay * r
                        if bv[c1]:
                            continue
                    if az != 0:
                        c1 = c0 + az
                        if bv[c1]:
                            continue
                    step = sqrt(<float>(ax * ax + ay * ay + az * az))
                    ng = gcur + step
                    if ng < gf[nidx]:
                        gf[nidx] = ng
                        came[nidx] = cur
                        h = _octile3(ex - nx, ey - ny, ez - nz)
                        _hpush(ff, hf, hi, ps, &hn, ng + h, h, nidx)
    out_exp[0] = exp
    cdef int plen = 0, c = best_idx
    if came[c] == -1 and c != start:
        path[0] = start
        out_len[0] = 1
        return status
    while True:
        path[plen] = c
        plen += 1
        if c == start:
            break
        c = came[c]
    cdef int a = 0, b = plen - 1, t
    while a < b:
        t = path[a]
        path[a] = path[b]
        path[b] = t
        a += 1
        b -= 1
    out_len[0] = plen
    return status

def astar3d(np.ndarray[np.uint8_t, ndim=3] blocked,
            int sx, int sy, int sz, int ex, int ey, int ez, int max_exp,
            int x0, int x1, int y0, int y1, int z0, int z1):
    cdef int r = blocked.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=3] blk = np.ascontiguousarray(blocked, dtype=np.uint8)
    cdef int N = r * r * r
    cdef np.ndarray[np.float32_t, ndim=1] gflat = np.full(N, 1e30, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] fheap = np.empty(N, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] hheap = np.empty(N, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] iheap = np.empty(N, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] pos = np.full(N, -1, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] came = np.full(N, -1, dtype=np.int32)
    cdef np.ndarray[np.uint8_t, ndim=1] closed = np.zeros(N, dtype=np.uint8)
    cdef np.ndarray[np.int32_t, ndim=1] path = np.empty(N, dtype=np.int32)
    cdef unsigned char *bv = &blk[0, 0, 0]
    cdef float *gf = &gflat[0]
    cdef float *ff = &fheap[0]
    cdef float *hf = &hheap[0]
    cdef np.int32_t *hi = &iheap[0]
    cdef np.int32_t *ps = &pos[0]
    cdef np.int32_t *cm = &came[0]
    cdef unsigned char *cl = &closed[0]
    cdef np.int32_t *pv = &path[0]
    cdef int out_exp = 0
    cdef int out_len = 0
    cdef int status
    with nogil:
        status = _astar3d_run(bv, r, sx, sy, sz, ex, ey, ez, max_exp, x0, x1, y0, y1, z0, z1,
                              gf, ff, hf, hi, ps, cm, cl, pv, &out_exp, &out_len)
    return (status, path[:out_len], out_exp)

def los3d(np.ndarray[np.uint8_t, ndim=3] blocked, int x0, int y0, int z0, int x1, int y1, int z1):
    cdef int r = blocked.shape[0]
    cdef unsigned char[:, :, :] bv = blocked
    cdef float dx = <float>(x1 - x0), dy = <float>(y1 - y0), dz = <float>(z1 - z0)
    cdef int sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
    cdef int sy = 1 if dy > 0 else (-1 if dy < 0 else 0)
    cdef int sz = 1 if dz > 0 else (-1 if dz < 0 else 0)
    cdef float tdx = 1e30, tdy = 1e30, tdz = 1e30, tmx = 1e30, tmy = 1e30, tmz = 1e30
    cdef int gx = x0, gy = y0, gz = z0, guard = 0
    if gx < 0 or gx >= r or gy < 0 or gy >= r or gz < 0 or gz >= r or bv[gx, gy, gz]:
        return False
    if gx == x1 and gy == y1 and gz == z1:
        return True
    if dx != 0.0:
        tdx = fabs(1.0 / dx)
        tmx = 0.5 * tdx
    if dy != 0.0:
        tdy = fabs(1.0 / dy)
        tmy = 0.5 * tdy
    if dz != 0.0:
        tdz = fabs(1.0 / dz)
        tmz = 0.5 * tdz
    while guard < 6 * r + 32:
        guard += 1
        if tmx < tmy and tmx < tmz:
            gx += sx
            tmx += tdx
        elif tmy < tmz:
            gy += sy
            tmy += tdy
        elif tmz < tmy:
            gz += sz
            tmz += tdz
        elif tmx <= tmy and tmx <= tmz:
            gx += sx
            tmx += tdx
        elif tmy <= tmz:
            gy += sy
            tmy += tdy
        else:
            gz += sz
            tmz += tdz
        if gx < 0 or gx >= r or gy < 0 or gy >= r or gz < 0 or gz >= r or bv[gx, gy, gz]:
            return False
        if gx == x1 and gy == y1 and gz == z1:
            return True
    return False

def smooth3d(np.ndarray[np.int32_t, ndim=1] path, int n, np.ndarray[np.uint8_t, ndim=3] blocked):
    cdef int r = blocked.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] out = np.empty(n, dtype=np.int32)
    cdef int i = 0, j, m = 0
    cdef int ax, ay, az, bx, by, bz, t
    while i < n - 1:
        j = n - 1
        t = path[i]
        ax = t // (r * r)
        ay = (t - ax * r * r) // r
        az = t - ax * r * r - ay * r
        while j > i + 1:
            t = path[j]
            bx = t // (r * r)
            by = (t - bx * r * r) // r
            bz = t - bx * r * r - by * r
            if los3d(blocked, ax, ay, az, bx, by, bz):
                break
            j -= 1
        out[m] = path[i]
        m += 1
        i = j
    out[m] = path[n - 1]
    m += 1
    return out[:m]

def seg3d_free(np.ndarray[np.uint8_t, ndim=3] blocked,
               float ax, float ay, float az, float bx, float by, float bz, int h):
    cdef int r = blocked.shape[0]
    cdef unsigned char[:, :, :] bv = blocked
    cdef float dx = bx - ax, dy = by - ay, dz = bz - az
    cdef int sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
    cdef int sy = 1 if dy > 0 else (-1 if dy < 0 else 0)
    cdef int sz = 1 if dz > 0 else (-1 if dz < 0 else 0)
    cdef float tdx = 1e30, tdy = 1e30, tdz = 1e30, tmx = 1e30, tmy = 1e30, tmz = 1e30
    cdef int ix = <int>(ax + 1e-4), iy = <int>(ay + 1e-4), iz = <int>(az + 1e-4)
    cdef int ex = <int>(bx + 1e-4), ey = <int>(by + 1e-4), ez = <int>(bz + 1e-4)
    cdef int guard = 0, k, yy
    if h < 0:
        h = 0
    if ix < 0 or ix >= r or iz < 0 or iz >= r:
        return False
    if iy < -h or iy >= r:
        return False
    for k in range(h + 1):
        yy = iy + k
        if yy >= r:
            break
        if yy < 0 or bv[ix, yy, iz]:
            return False
    if ix == ex and iy == ey and iz == ez:
        return True
    if dx > 0.0:
        tdx = 1.0 / dx
        tmx = (<float>(ix + 1) - ax) * tdx
    elif dx < 0.0:
        tdx = -1.0 / dx
        tmx = (ax - <float>ix) * tdx
    if dy > 0.0:
        tdy = 1.0 / dy
        tmy = (<float>(iy + 1) - ay) * tdy
    elif dy < 0.0:
        tdy = -1.0 / dy
        tmy = (ay - <float>iy) * tdy
    if dz > 0.0:
        tdz = 1.0 / dz
        tmz = (<float>(iz + 1) - az) * tdz
    elif dz < 0.0:
        tdz = -1.0 / dz
        tmz = (az - <float>iz) * tdz
    while guard < 8 * r + 64:
        guard += 1
        if tmx < tmy and tmx < tmz:
            ix += sx
            tmx += tdx
        elif tmy < tmz:
            iy += sy
            tmy += tdy
        elif tmz < tmy:
            iz += sz
            tmz += tdz
        elif tmx <= tmy and tmx <= tmz:
            ix += sx
            tmx += tdx
        elif tmy <= tmz:
            iy += sy
            tmy += tdy
        else:
            iz += sz
            tmz += tdz
        if ix < 0 or ix >= r or iz < 0 or iz >= r:
            return False
        if iy < -h or iy >= r:
            return False
        for k in range(h + 1):
            yy = iy + k
            if yy >= r:
                break
            if yy < 0 or bv[ix, yy, iz]:
                return False
        if ix == ex and iy == ey and iz == ez:
            return True
    return False

def segwalk2d(np.ndarray[np.uint8_t, ndim=2] walk, np.ndarray[np.int32_t, ndim=2] ground,
              float x0, float y0, float z0, float x1, float y1, float z1, float climb):
    cdef int r = walk.shape[0]
    cdef unsigned char[:, :] wv = walk
    cdef np.int32_t[:, :] gv = ground
    cdef float dx = x1 - x0, dy = y1 - y0, dz = z1 - z0
    cdef int sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
    cdef int sz = 1 if dz > 0 else (-1 if dz < 0 else 0)
    cdef float tdx = 1e30, tdz = 1e30, tmx = 1e30, tmz = 1e30, t = 0.0, y, lo
    cdef int gx = <int>(x0 + 1e-4), gz = <int>(z0 + 1e-4), guard = 0
    cdef int ex = <int>(x1 + 1e-4), ez = <int>(z1 + 1e-4)
    cdef int surf, psurf
    if climb < 0.0:
        climb = 0.0
    lo = -(climb + 0.5) if dy >= 0.0 else -0.35
    if gx < 0 or gx >= r or gz < 0 or gz >= r or wv[gx, gz] == 0:
        return False
    surf = gv[gx, gz]
    if surf < 0 or y0 - <float>(surf + 1) < lo or y0 - <float>(surf + 1) > climb + 0.5:
        return False
    psurf = surf
    if gx == ex and gz == ez:
        return True
    if dx != 0.0:
        tdx = fabs(1.0 / dx)
        if sx > 0:
            tmx = (<float>(gx + 1) - x0) * tdx
        else:
            tmx = (x0 - <float>gx) * tdx
    if dz != 0.0:
        tdz = fabs(1.0 / dz)
        if sz > 0:
            tmz = (<float>(gz + 1) - z0) * tdz
        else:
            tmz = (z0 - <float>gz) * tdz
    while guard < 4 * r + 16:
        guard += 1
        if tmx < tmz:
            gx += sx
            t = tmx
            tmx += tdx
        elif tmz < tmx:
            gz += sz
            t = tmz
            tmz += tdz
        else:
            gx += sx
            t = tmx
            tmx += tdx
        if gx < 0 or gx >= r or gz < 0 or gz >= r or wv[gx, gz] == 0:
            return False
        surf = gv[gx, gz]
        if surf < 0:
            return False
        if <float>(surf - psurf) > climb:
            return False
        psurf = surf
        y = y0 + t * dy
        if y - <float>(surf + 1) < lo or y - <float>(surf + 1) > climb + 0.5:
            return False
        if gx == ex and gz == ez:
            return True
    return False

def smooth3d_clear(np.ndarray[np.int32_t, ndim=1] path, int n, np.ndarray[np.uint8_t, ndim=3] blocked, int rad):
    cdef int r = blocked.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] out = np.empty(n, dtype=np.int32)
    cdef int i = 0, j, m = 0
    cdef int ax, ay, az, bx, by, bz, t
    while i < n - 1:
        j = n - 1
        t = path[i]
        ax = t // (r * r)
        ay = (t - ax * r * r) // r
        az = t - ax * r * r - ay * r
        while j > i + 1:
            t = path[j]
            bx = t // (r * r)
            by = (t - bx * r * r) // r
            bz = t - bx * r * r - by * r
            if los3d_clear(blocked, ax, ay, az, bx, by, bz, rad):
                break
            j -= 1
        out[m] = path[i]
        m += 1
        i = j
    out[m] = path[n - 1]
    m += 1
    return out[:m]

def downsample_any3d(np.ndarray[np.uint8_t, ndim=3] src, int factor):
    cdef int r = src.shape[0]
    cdef int cr = (r + factor - 1) // factor
    cdef np.ndarray[np.uint8_t, ndim=3] dst = np.zeros((cr, cr, cr), dtype=np.uint8)
    cdef unsigned char[:, :, :] sv = src
    cdef unsigned char[:, :, :] dv = dst
    cdef int cx, cy, cz, x, y, z, x1, y1, z1
    cdef bint hit
    with nogil:
        for cx in range(cr):
            x1 = cx * factor + factor
            if x1 > r:
                x1 = r
            for cy in range(cr):
                y1 = cy * factor + factor
                if y1 > r:
                    y1 = r
                for cz in range(cr):
                    z1 = cz * factor + factor
                    if z1 > r:
                        z1 = r
                    hit = False
                    for x in range(cx * factor, x1):
                        if hit:
                            break
                        for y in range(cy * factor, y1):
                            if hit:
                                break
                            for z in range(cz * factor, z1):
                                if sv[x, y, z]:
                                    hit = True
                                    break
                    if hit:
                        dv[cx, cy, cz] = 1
    return dst

def dilate3d(np.ndarray[np.uint8_t, ndim=3] grid, int radius):
    cdef int r = grid.shape[0]
    if radius <= 0:
        return grid
    cdef np.ndarray[np.uint8_t, ndim=3] out = grid.copy()
    cdef unsigned char[:, :, :] gv = grid
    cdef unsigned char[:, :, :] ov = out
    cdef int x, y, z, dx, dy, dz, x1, x2, y1, y2, z1, z2
    with nogil:
        for x in range(r):
            for y in range(r):
                for z in range(r):
                    if gv[x, y, z]:
                        x1 = x - radius
                        if x1 < 0:
                            x1 = 0
                        x2 = x + radius + 1
                        if x2 > r:
                            x2 = r
                        y1 = y - radius
                        if y1 < 0:
                            y1 = 0
                        y2 = y + radius + 1
                        if y2 > r:
                            y2 = r
                        z1 = z - radius
                        if z1 < 0:
                            z1 = 0
                        z2 = z + radius + 1
                        if z2 > r:
                            z2 = r
                        for dx in range(x1, x2):
                            for dy in range(y1, y2):
                                for dz in range(z1, z2):
                                    ov[dx, dy, dz] = 1
    return out
