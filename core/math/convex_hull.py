# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

import numpy as np

_coro = None
try:
    from core._convex_hull import convex_hull_simplices as _compiled
    _coro = _compiled
except ImportError:
    pass


def convex_hull_simplices(points):
    if _coro is not None:
        return _coro(points)
    return _py_convex_hull_simplices(points)


def _py_convex_hull_simplices(points):
    n = points.shape[0]
    if n < 4:
        return np.empty((0, 3), dtype=np.int32)
    pts = np.asarray(points, dtype=np.float64)

    p0 = 0
    p1 = 1
    max_d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum((pts[i] - pts[j]) ** 2)
            if d > max_d:
                max_d = d
                p0 = i
                p1 = j

    max_d = 0.0
    p2 = p0
    for i in range(n):
        if i == p0 or i == p1:
            continue
        d = np.linalg.norm(np.cross(pts[p1] - pts[p0], pts[p0] - pts[i])) / max(
            np.linalg.norm(pts[p1] - pts[p0]), 1e-20
        )
        if d > max_d:
            max_d = d
            p2 = i

    v1 = pts[p1] - pts[p0]
    v2 = pts[p2] - pts[p0]
    nrm = np.cross(v1, v2)
    nl = np.linalg.norm(nrm)
    if nl < 1e-10:
        return np.empty((0, 3), dtype=np.int32)
    nrm = nrm / nl
    pd_val = np.dot(nrm, pts[p0])

    max_d = 0.0
    p3 = p0
    for i in range(n):
        if i == p0 or i == p1 or i == p2:
            continue
        d = abs(np.dot(pts[i], nrm) - pd_val)
        if d > max_d:
            max_d = d
            p3 = i
    if max_d < 1e-10:
        return np.empty((0, 3), dtype=np.int32)

    tetra = [p0, p1, p2, p3]
    faces = []
    for a, b, c, opp in [(0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2), (1, 3, 2, 0)]:
        i0, i1, i2 = tetra[a], tetra[b], tetra[c]
        v1 = pts[i1] - pts[i0]
        v2 = pts[i2] - pts[i0]
        fn = np.cross(v1, v2)
        fl = np.linalg.norm(fn)
        if fl > 1e-10:
            fn = fn / fl
        fd = np.dot(fn, pts[i0])
        opp_idx = tetra[opp]
        if np.dot(pts[opp_idx], fn) - fd > 1e-10:
            faces.append([i2, i1, i0])
        else:
            faces.append([i0, i1, i2])

    processed = [False] * n
    for t in tetra:
        processed[t] = True

    centroid = np.mean(pts[tetra], axis=0)

    for idx in range(n):
        if processed[idx]:
            continue
        visible = []
        for fi, f in enumerate(faces):
            a, b, c = f
            v1 = pts[b] - pts[a]
            v2 = pts[c] - pts[a]
            fn = np.cross(v1, v2)
            fl = np.linalg.norm(fn)
            if fl > 1e-10:
                fn = fn / fl
            fd = np.dot(fn, pts[a])
            sd = np.dot(pts[idx], fn) - fd
            if sd > 1e-10:
                visible.append(fi)

        if not visible:
            continue

        edge_count = {}
        for fi in visible:
            f = faces[fi]
            for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
                key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
                edge_count[key] = edge_count.get(key, 0) + 1

        horizon = [e for e, cnt in edge_count.items() if cnt == 1]

        saved_visible = [faces[fi][:] for fi in visible]

        for fi in sorted(visible, reverse=True):
            faces.pop(fi)

        for ha, hb in horizon:
            oriented = False
            for fv in saved_visible:
                if (fv[0] == ha and fv[1] == hb) or (fv[1] == ha and fv[2] == hb) or (fv[2] == ha and fv[0] == hb):
                    faces.append([ha, hb, idx])
                    oriented = True
                    break
                if (fv[0] == hb and fv[1] == ha) or (fv[1] == hb and fv[2] == ha) or (fv[2] == hb and fv[0] == ha):
                    faces.append([hb, ha, idx])
                    oriented = True
                    break
            if not oriented:
                v1 = pts[hb] - pts[ha]
                v2 = pts[idx] - pts[ha]
                fn = np.cross(v1, v2)
                fl = np.linalg.norm(fn)
                if fl > 1e-10:
                    fn = fn / fl
                fd = np.dot(fn, pts[ha])
                sd = np.dot(centroid, fn) - fd
                if sd < 0:
                    faces.append([ha, hb, idx])
                else:
                    faces.append([hb, ha, idx])

        processed[idx] = True

    return np.array(faces, dtype=np.int32)
