# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs

cdef inline void cross(double ax, double ay, double az, double bx, double by, double bz,
                        double* cx, double* cy, double* cz) noexcept nogil:
    cx[0] = ay*bz - az*by
    cy[0] = az*bx - ax*bz
    cz[0] = ax*by - ay*bx

cdef inline double dist2_pts(const double[:, :] p, int a, int b) noexcept nogil:
    cdef double dx = p[a,0]-p[b,0], dy = p[a,1]-p[b,1], dz = p[a,2]-p[b,2]
    return dx*dx + dy*dy + dz*dz

cdef inline double point_plane_dist(const double[:] p, double nx, double ny, double nz, double d) noexcept nogil:
    return nx*p[0] + ny*p[1] + nz*p[2] - d

cdef inline double line_dist(const double[:, :] p, int i, int a, int b) noexcept nogil:
    cdef double ux = p[b,0]-p[a,0], uy = p[b,1]-p[a,1], uz = p[b,2]-p[a,2]
    cdef double vx = p[a,0]-p[i,0], vy = p[a,1]-p[i,1], vz = p[a,2]-p[i,2]
    cdef double cx, cy, cz
    cross(ux, uy, uz, vx, vy, vz, &cx, &cy, &cz)
    cdef double lensq = ux*ux + uy*uy + uz*uz
    if lensq < 1e-20:
        return 0.0
    return sqrt((cx*cx + cy*cy + cz*cz) / lensq)

cdef inline void face_norm(const double[:, :] p, int a, int b, int c,
                            double* nx, double* ny, double* nz, double* d) noexcept nogil:
    cdef double ux = p[b,0]-p[a,0], uy = p[b,1]-p[a,1], uz = p[b,2]-p[a,2]
    cdef double vx = p[c,0]-p[a,0], vy = p[c,1]-p[a,1], vz = p[c,2]-p[a,2]
    cross(ux, uy, uz, vx, vy, vz, nx, ny, nz)
    cdef double l = sqrt(nx[0]*nx[0] + ny[0]*ny[0] + nz[0]*nz[0])
    if l > 1e-10:
        nx[0] /= l; ny[0] /= l; nz[0] /= l
    d[0] = nx[0]*p[a,0] + ny[0]*p[a,1] + nz[0]*p[a,2]

def convex_hull_simplices(points):
    cdef int n = points.shape[0]
    if n < 4:
        return np.empty((0, 3), dtype=np.int32)
    cdef double[:, :] pv = np.asarray(points, dtype=np.float64)

    cdef int i, j, p0, p1, p2, p3
    cdef double max_d, d, nx, ny, nz, pd, sd

    p0 = 0; p1 = 1; max_d = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = dist2_pts(pv, i, j)
            if d > max_d:
                max_d = d; p0 = i; p1 = j

    max_d = 0.0; p2 = p0
    for i in range(n):
        if i == p0 or i == p1: continue
        d = line_dist(pv, i, p0, p1)
        if d > max_d:
            max_d = d; p2 = i

    face_norm(pv, p0, p1, p2, &nx, &ny, &nz, &pd)
    if sqrt(nx*nx + ny*ny + nz*nz) < 1e-10:
        return np.empty((0, 3), dtype=np.int32)

    max_d = 0.0; p3 = p0
    for i in range(n):
        if i == p0 or i == p1 or i == p2: continue
        d = fabs(pv[i,0]*nx + pv[i,1]*ny + pv[i,2]*nz - pd)
        if d > max_d:
            max_d = d; p3 = i
    if max_d < 1e-10:
        return np.empty((0, 3), dtype=np.int32)

    tetra = [p0, p1, p2, p3]
    faces = []
    for (a, b, c, opp) in [(0,1,2,3), (0,2,3,1), (0,3,1,2), (1,3,2,0)]:
        i0, i1, i2 = tetra[a], tetra[b], tetra[c]
        face_norm(pv, i0, i1, i2, &nx, &ny, &nz, &pd)
        opp_idx = tetra[opp]
        sd = point_plane_dist(pv[opp_idx,:], nx, ny, nz, pd)
        if sd > 1e-10:
            faces.append([i2, i1, i0])
        else:
            faces.append([i0, i1, i2])

    processed = [False] * n
    for t in tetra:
        processed[t] = True

    cdef double cx, cy, cz
    cx = 0.0; cy = 0.0; cz = 0.0
    for t in tetra:
        cx += pv[t,0]; cy += pv[t,1]; cz += pv[t,2]
    cx /= 4.0; cy /= 4.0; cz /= 4.0

    for idx in range(n):
        if processed[idx]:
            continue
        visible = []
        for fi, f in enumerate(faces):
            a, b, c = f
            face_norm(pv, a, b, c, &nx, &ny, &nz, &pd)
            sd = point_plane_dist(pv[idx,:], nx, ny, nz, pd)
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
                face_norm(pv, ha, hb, idx, &nx, &ny, &nz, &pd)
                sd = nx*cx + ny*cy + nz*cz - pd
                if sd < 0:
                    faces.append([ha, hb, idx])
                else:
                    faces.append([hb, ha, idx])

        processed[idx] = True

    cdef int nf = len(faces)
    cdef np.ndarray[np.int32_t, ndim=2] result = np.empty((nf, 3), dtype=np.int32)
    cdef int[:, :] rv = result
    for i in range(nf):
        rv[i,0] = faces[i][0]
        rv[i,1] = faces[i][1]
        rv[i,2] = faces[i][2]
    return result
