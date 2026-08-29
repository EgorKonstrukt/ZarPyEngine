# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun
#
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# cython: language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs, acos, cos, sin, fmin, fmax

from core._math_vec import Vec3, Quat

cdef double _EPS = 1e-10


# ---------------------------------------------------------------------------
# Vector / quaternion primitives (raw doubles, nogil)
# ---------------------------------------------------------------------------

cdef inline double _vec_len(const double* a) noexcept nogil:
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])


cdef inline void _vec_norm(const double* a, double* o) noexcept nogil:
    cdef double n = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
    if n > _EPS:
        o[0] = a[0] / n
        o[1] = a[1] / n
        o[2] = a[2] / n
    else:
        o[0] = 0.0; o[1] = 0.0; o[2] = 0.0


cdef inline void _vec_sub(const double* a, const double* b, double* o) noexcept nogil:
    o[0] = a[0] - b[0]
    o[1] = a[1] - b[1]
    o[2] = a[2] - b[2]


cdef inline double _vec_dot(const double* a, const double* b) noexcept nogil:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


cdef inline void _vec_cross(const double* a, const double* b, double* o) noexcept nogil:
    o[0] = a[1]*b[2] - a[2]*b[1]
    o[1] = a[2]*b[0] - a[0]*b[2]
    o[2] = a[0]*b[1] - a[1]*b[0]


cdef inline void _vec_scale(const double* a, double s, double* o) noexcept nogil:
    o[0] = a[0] * s
    o[1] = a[1] * s
    o[2] = a[2] * s


cdef inline void _quat_identity(double* o) noexcept nogil:
    o[0] = 0.0; o[1] = 0.0; o[2] = 0.0; o[3] = 1.0


cdef inline void _quat_norm_inplace(double* q) noexcept nogil:
    cdef double n = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if n > _EPS:
        q[0] /= n; q[1] /= n; q[2] /= n; q[3] /= n
    else:
        q[0] = 0.0; q[1] = 0.0; q[2] = 0.0; q[3] = 1.0


cdef inline void _quat_mul(const double* a, const double* b, double* o) noexcept nogil:
    cdef double aw = a[3], ax = a[0], ay = a[1], az = a[2]
    cdef double bw = b[3], bx = b[0], by = b[1], bz = b[2]
    o[0] = aw*bx + ax*bw + ay*bz - az*by
    o[1] = aw*by - ax*bz + ay*bw + az*bx
    o[2] = aw*bz + ax*by - ay*bx + az*bw
    o[3] = aw*bw - ax*bx - ay*by - az*bz


cdef inline void _quat_conj(const double* a, double* o) noexcept nogil:
    o[0] = -a[0]; o[1] = -a[1]; o[2] = -a[2]; o[3] = a[3]


cdef inline void _quat_from_axis_angle(const double* axis, double angle,
                                       double* o) noexcept nogil:
    cdef double n = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
    cdef double s
    if n < _EPS:
        _quat_identity(o)
        return
    s = sin(angle * 0.5) / n
    o[0] = axis[0] * s
    o[1] = axis[1] * s
    o[2] = axis[2] * s
    o[3] = cos(angle * 0.5)


cdef inline void _quat_from_two_vecs(const double* v0, const double* v1,
                                     double* o) noexcept nogil:
    cdef double n0 = sqrt(v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2])
    cdef double n1 = sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2])
    if (n0 < _EPS) or (n1 < _EPS):
        _quat_identity(o)
        return
    cdef double ux = v0[0]/n0, uy = v0[1]/n0, uz = v0[2]/n0
    cdef double wx = v1[0]/n1, wy = v1[1]/n1, wz = v1[2]/n1
    cdef double c = ux*wx + uy*wy + uz*wz
    if c > 0.99999999:
        _quat_identity(o)
        return
    cdef double cx = uy*wz - uz*wy
    cdef double cy = uz*wx - ux*wz
    cdef double cz = ux*wy - uy*wx
    cdef double den = sqrt(2.0 * (1.0 + c))
    cdef double ref[3], ax2[3]
    if den < _EPS:
        ref[0] = 1.0; ref[1] = 0.0; ref[2] = 0.0
        _vec_cross(ref, v0, ax2)
        if _vec_len(ax2) < _EPS:
            ref[0] = 0.0; ref[1] = 1.0; ref[2] = 0.0
            _vec_cross(ref, v0, ax2)
        _quat_from_axis_angle(ax2, 3.141592653589793, o)
        return
    o[0] = cx / den
    o[1] = cy / den
    o[2] = cz / den
    o[3] = den * 0.5


cdef inline void _rotate_vec_by_quat(const double* v, const double* q,
                                     double* o) noexcept nogil:
    cdef double tx = 2.0 * (q[1]*v[2] - q[2]*v[1])
    cdef double ty = 2.0 * (q[2]*v[0] - q[0]*v[2])
    cdef double tz = 2.0 * (q[0]*v[1] - q[1]*v[0])
    o[0] = v[0] + q[3]*tx + (q[1]*tz - q[2]*ty)
    o[1] = v[1] + q[3]*ty + (q[2]*tx - q[0]*tz)
    o[2] = v[2] + q[3]*tz + (q[0]*ty - q[1]*tx)


cdef inline void _quat_mix(const double* a, const double* b, double t, double* o) noexcept nogil:
    cdef double d = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
    cdef double s0 = 1.0 - t
    cdef double s1 = t
    if d < 0.0:
        s1 = -s1
    o[0] = a[0]*s0 + b[0]*s1
    o[1] = a[1]*s0 + b[1]*s1
    o[2] = a[2]*s0 + b[2]*s1
    o[3] = a[3]*s0 + b[3]*s1
    _quat_norm_inplace(o)


cdef inline void _quat_from_mat_rows(double m00, double m01, double m02,
                                     double m10, double m11, double m12,
                                     double m20, double m21, double m22,
                                     double* o) noexcept nogil:
    cdef double trace = m00 + m11 + m22
    cdef double s
    if trace > 0.0:
        s = 0.5 / sqrt(trace + 1.0)
        o[3] = 0.25 / s
        o[0] = (m21 - m12) * s
        o[1] = (m02 - m20) * s
        o[2] = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * sqrt(1.0 + m00 - m11 - m22)
        o[3] = (m21 - m12) / s
        o[0] = 0.25 * s
        o[1] = (m01 + m10) / s
        o[2] = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * sqrt(1.0 + m11 - m00 - m22)
        o[3] = (m02 - m20) / s
        o[0] = (m01 + m10) / s
        o[1] = 0.25 * s
        o[2] = (m12 + m21) / s
    else:
        s = 2.0 * sqrt(1.0 + m22 - m00 - m11)
        o[3] = (m10 - m01) / s
        o[0] = (m02 + m20) / s
        o[1] = (m12 + m21) / s
        o[2] = 0.25 * s
    _quat_norm_inplace(o)


cdef inline void _look_rotation(double fx, double fy, double fz,
                                double ux, double uy, double uz,
                                double* o) noexcept nogil:
    cdef double rx, ry, rz, rl
    cdef double m00, m01, m02, m10, m11, m12, m20, m21, m22
    rl = sqrt(ux*ux + uy*uy + uz*uz)
    if rl < _EPS:
        _quat_identity(o)
        return
    ux /= rl; uy /= rl; uz /= rl
    rl = sqrt(fx*fx + fy*fy + fz*fz)
    if rl < _EPS:
        _quat_identity(o)
        return
    fx /= rl; fy /= rl; fz /= rl
    rx = uy*fz - uz*fy
    ry = uz*fx - ux*fz
    rz = ux*fy - uy*fx
    rl = sqrt(rx*rx + ry*ry + rz*rz)
    if rl < _EPS:
        _quat_identity(o)
        return
    rx /= rl; ry /= rl; rz /= rl
    m00 = rx;   m01 = ux;  m02 = -fx
    m10 = ry;   m11 = uy;  m12 = -fy
    m20 = rz;   m21 = uz;  m22 = -fz
    _quat_from_mat_rows(m00, m01, m02, m10, m11, m12, m20, m21, m22, o)


cdef inline void _quat_to_local(const double* parent_wq, const double* world_q,
                                double* out) noexcept nogil:
    # engine world = parent * local  =>  local = parent^-1 * world_q
    cdef double inv[4]
    _quat_conj(parent_wq, inv)
    _quat_mul(inv, world_q, out)
    _quat_norm_inplace(out)


# ---------------------------------------------------------------------------
# Two-bone analytic geometry (pure math, nogil)
# ---------------------------------------------------------------------------

cdef int _rotate_axis_make(const double* axis_in, double ang, const double* v, double* out) noexcept nogil:
    cdef double q[4]
    _quat_from_axis_angle(axis_in, ang, q)
    _rotate_vec_by_quat(v, q, out)
    return 1


cdef void _two_bone_geometry(
        const double* a, const double* b, const double* c,
        const double* T, const double* pole, int has_pole,
        double stretch, double bend,
        double* b_new, double* qa, double* qb) noexcept nogil:
    cdef double L1 = sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2 + (b[2]-a[2])**2)
    cdef double L2 = sqrt((c[0]-b[0])**2 + (c[1]-b[1])**2 + (c[2]-b[2])**2)
    cdef double d = sqrt((T[0]-a[0])**2 + (T[1]-a[1])**2 + (T[2]-a[2])**2)
    _quat_identity(qa)
    _quat_identity(qb)
    b_new[0] = b[0]; b_new[1] = b[1]; b_new[2] = b[2]
    if L1 < _EPS or L2 < _EPS:
        return

    cdef double u[3]
    if d < _EPS:
        return
    u[0] = (T[0]-a[0]) / d
    u[1] = (T[1]-a[1]) / d
    u[2] = (T[2]-a[2]) / d

    cdef double L1e = L1, L2e = L2
    cdef double max_reach = L1 + L2
    cdef double min_reach = fabs(L1 - L2)
    cdef double extra
    if d > max_reach and stretch > 0.0:
        extra = d - max_reach
        L1e = L1 + extra * stretch * (L1 / max_reach)
        L2e = L2 + extra * stretch * (L2 / max_reach)
        max_reach = L1e + L2e
    if d > max_reach:
        d = max_reach
    if d < min_reach:
        d = min_reach
    if d < _EPS:
        return

    cdef double prev[3], pref[3], axis[3], nrm[3], pd[3]
    prev[0] = b[0]-a[0]; prev[1] = b[1]-a[1]; prev[2] = b[2]-a[2]
    _vec_norm(prev, prev)

    cdef double use_pole = 0
    if has_pole:
        _vec_sub(pole, a, pd)
        _vec_norm(pd, pd)
        if _vec_len(pd) > 1e-6:
            use_pole = 1
    if use_pole:
        pref[0] = pd[0]; pref[1] = pd[1]; pref[2] = pd[2]
        _vec_cross(u, pd, axis)
        if _vec_len(axis) < 1e-6:
            _vec_cross(u, prev, axis)
    else:
        _vec_cross(u, prev, axis)
        if _vec_len(axis) < 1e-6:
            axis[0] = 0.0; axis[1] = 0.0; axis[2] = 1.0
            _vec_cross(u, axis, axis)
    _vec_norm(axis, nrm)
    if _vec_len(nrm) < 1e-6:
        nrm[0] = 0.0; nrm[1] = 0.0; nrm[2] = 1.0

    cdef double cos_a = (L1e*L1e + d*d - L2e*L2e) / (2.0 * L1e * d)
    if cos_a > 1.0: cos_a = 1.0
    if cos_a < -1.0: cos_a = -1.0
    cdef double ang = acos(cos_a)

    cdef double qr[4], c1[3], c2[3]
    _quat_from_axis_angle(nrm, ang, qr)
    _rotate_vec_by_quat(u, qr, c1)
    cdef double qinv[4]
    qinv[0] = -qr[0]; qinv[1] = -qr[1]; qinv[2] = -qr[2]; qinv[3] = qr[3]
    _rotate_vec_by_quat(u, qinv, c2)

    cdef double* dirn = c1
    if use_pole:
        if _vec_dot(c2, pref) > _vec_dot(c1, pref):
            dirn = c2
    elif bend > 0.0:
        dirn = c1
    elif bend < 0.0:
        dirn = c2
    else:
        if _vec_dot(c2, prev) > _vec_dot(c1, prev):
            dirn = c2

    b_new[0] = a[0] + L1e * dirn[0]
    b_new[1] = a[1] + L1e * dirn[1]
    b_new[2] = a[2] + L1e * dirn[2]

    cdef double v0[3], v1[3], aim[3]
    v0[0] = b[0]-a[0]; v0[1] = b[1]-a[1]; v0[2] = b[2]-a[2]
    v1[0] = b_new[0]-a[0]; v1[1] = b_new[1]-a[1]; v1[2] = b_new[2]-a[2]
    _quat_from_two_vecs(v0, v1, qa)

    aim[0] = T[0]-b_new[0]; aim[1] = T[1]-b_new[1]; aim[2] = T[2]-b_new[2]
    _vec_norm(aim, aim)
    v0[0] = c[0]-b[0]; v0[1] = c[1]-b[1]; v0[2] = c[2]-b[2]
    _quat_from_two_vecs(v0, aim, qb)


# ---------------------------------------------------------------------------
# FABRIK iterative chain (pure math, nogil)
# ---------------------------------------------------------------------------

cdef void _fabrik_positions(double* p, const double* orig, int n,
                            const double* lens, const double* T,
                            int iters, double tol) noexcept nogil:
    cdef int i, it
    cdef double total = 0.0, d0, tpar
    cdef double base[3], dir[3]
    for i in range(n - 1):
        total += lens[i]
    base[0] = orig[0]; base[1] = orig[1]; base[2] = orig[2]
    d0 = sqrt((T[0]-base[0])**2 + (T[1]-base[1])**2 + (T[2]-base[2])**2)
    if d0 >= total:
        _vec_sub(T, base, dir)
        _vec_norm(dir, dir)
        for i in range(n):
            tpar = total * i / (n - 1) if n > 1 else 0.0
            p[i*3+0] = base[0] + dir[0] * tpar
            p[i*3+1] = base[1] + dir[1] * tpar
            p[i*3+2] = base[2] + dir[2] * tpar
        return
    for it in range(iters):
        p[(n-1)*3+0] = T[0]
        p[(n-1)*3+1] = T[1]
        p[(n-1)*3+2] = T[2]
        for i in range(n - 2, -1, -1):
            _vec_sub(&p[(i+1)*3], &p[i*3], dir)
            _vec_norm(dir, dir)
            p[i*3+0] = p[(i+1)*3+0] - dir[0] * lens[i]
            p[i*3+1] = p[(i+1)*3+1] - dir[1] * lens[i]
            p[i*3+2] = p[(i+1)*3+2] - dir[2] * lens[i]
        p[0] = base[0]; p[1] = base[1]; p[2] = base[2]
        for i in range(n - 1):
            _vec_sub(&p[(i+1)*3], &p[i*3], dir)
            _vec_norm(dir, dir)
            p[(i+1)*3+0] = p[i*3+0] + dir[0] * lens[i]
            p[(i+1)*3+1] = p[i*3+1] + dir[1] * lens[i]
            p[(i+1)*3+2] = p[i*3+2] + dir[2] * lens[i]
        if sqrt((p[(n-1)*3]-T[0])**2 + (p[(n-1)*3+1]-T[1])**2 + (p[(n-1)*3+2]-T[2])**2) < tol:
            break


# ---------------------------------------------------------------------------
# Engine bridge: read position / world rotation, write local rotation
# ---------------------------------------------------------------------------

cdef void _read_world_pos(object t, double* out) noexcept:
    cdef object p = t.position
    if p is None:
        out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
        return
    out[0] = float(p._x)
    out[1] = float(p._y)
    out[2] = float(p._z)


cdef void _read_world_quat_semantic(object t, double* out) noexcept:
    # The engine composes world = parent quat * local quat (Hamilton), and its
    # world matrix stores R(q)^T. Shepperd on the row-major 3x3 sub-block
    # returns the CONJUGATE of the true semantic quaternion, so negate the
    # vector part here to recover q_world used by placement math below.
    cdef object wm = t.world_matrix
    cdef double[:, ::1] m = wm._d
    _quat_from_mat_rows(
        m[0, 0], m[0, 1], m[0, 2],
        m[1, 0], m[1, 1], m[1, 2],
        m[2, 0], m[2, 1], m[2, 2],
        out)
    out[0] = -out[0]
    out[1] = -out[1]
    out[2] = -out[2]


cdef void _read_local_quat(object t, double* out) noexcept:
    cdef object lr = t.local_rotation
    if lr is None:
        _quat_identity(out)
        return
    out[0] = lr._x
    out[1] = lr._y
    out[2] = lr._z
    out[3] = lr._w


cdef void _apply_world_delta(object t, const double* qd, double weight) noexcept:
    # local_new = local_cur * conj(world_cur) * qd * world_cur
    cdef double ql[4], qw[4], qwc[4], tmp[4], tmp2[4], out[4]
    _read_local_quat(t, ql)
    _read_world_quat_semantic(t, qw)
    _quat_conj(qw, qwc)
    _quat_mul(qwc, qd, tmp)
    _quat_mul(tmp, qw, tmp2)
    _quat_mul(ql, tmp2, out)
    _quat_norm_inplace(out)
    _apply_local(t, out, weight)


cdef void _apply_local(object t, const double* q, double weight) noexcept:
    cdef object lr
    cdef double cur[4], mixed[4]
    if t is None:
        return
    if weight < 0.999999:
        lr = t.local_rotation
        if lr is not None:
            cur[0] = lr._x; cur[1] = lr._y; cur[2] = lr._z; cur[3] = lr._w
            _quat_mix(cur, q, weight, mixed)
            t.local_rotation = Quat._make(mixed[0], mixed[1], mixed[2], mixed[3])
            return
    t.local_rotation = Quat._make(q[0], q[1], q[2], q[3])


cdef object _resolve_entity(object c, str attr):
    cdef object ent = c._entity
    cdef object scene
    cdef object idv
    if ent is None:
        return None
    scene = ent._scene
    idv = getattr(c, attr, None)
    if not idv or scene is None:
        return None
    return scene.get_entity(idv)


cdef int _read_comp_vec3(object c, str attr, double* out) noexcept:
    cdef object v = getattr(c, attr, None)
    if v is None:
        return 0
    if isinstance(v, Vec3):
        out[0] = v.x
        out[1] = v.y
        out[2] = v.z
    else:
        try:
            out[0] = float(v[0]); out[1] = float(v[1]); out[2] = float(v[2])
        except Exception:
            return 0
    return 1


cdef int _read_source_pos(object c, str ent_attr, str vec_attr, double* out) noexcept:
    cdef object tgt = _resolve_entity(c, ent_attr)
    cdef object tt
    if tgt is not None:
        tt = tgt.transform
        if tt is not None:
            _read_world_pos(tt, out)
            return 1
    return _read_comp_vec3(c, vec_attr, out)


cdef object _resolve_bone_path(object ent, str bone_path):
    cdef str root_name
    cdef list parts
    cdef object current
    cdef str seg
    cdef object child
    cdef int found
    if not bone_path:
        return ent
    root_name = ent.name if ent is not None else ""
    parts = [p for p in bone_path.split("/") if p]
    current = ent
    if parts and parts[0] and parts[0] == root_name:
        parts = parts[1:]
    for seg in parts:
        found = 0
        for child in current._children:
            if child.name == seg:
                current = child
                found = 1
                break
        if not found:
            return current
    return current


cdef object _find_named_descendant(object ent, str name):
    """Depth-first search for an entity named `name` anywhere below `ent`."""
    cdef list stack, children
    cdef object node
    cdef Py_ssize_t i
    if ent is None or not name:
        return None
    stack = [ent]
    while stack:
        node = stack.pop()
        children = node._children
        if children:
            for i in range(len(children)):
                stack.append(children[i])
        if node is not ent and node.name == name:
            return node
    return None


cdef object _resolve_mid_or_tip(object ent, object root_e, str path):
    """Resolve a mid/tip bone path: first relative to the component entity,
    then falling back to a name search in the resolved root's subtree."""
    cdef str last
    cdef object node, p
    cdef list parts
    cdef int i
    if not path:
        return None
    parts = []
    for p in path.split("/"):
        if p:
            parts.append(p)
    if not parts:
        return None
    last = parts[len(parts) - 1]
    node = _resolve_bone_path(ent, path)
    if (node is not None and node is not ent and
            node.name == last and last != ent.name):
        return node
    return _find_named_descendant(root_e, last) if root_e is not None else None


# ---------------------------------------------------------------------------
# Component solves
# ---------------------------------------------------------------------------

cdef void _update_two_bone(object c) noexcept:
    cdef object ent = c._entity
    cdef str root_path, mid_path, tip_path
    cdef object root_e, mid_e, tip_e
    cdef object rt
    cdef double a[3], bpos[3], cpos[3], T[3], pole[3]
    cdef int has_pole, target_ok
    cdef double stretch, weight, bend
    cdef object bendv
    cdef double b_new[3], qa[4], qb[4]
    cdef double d0[3], d1[3], qd[4]

    if ent is None:
        return
    root_path = getattr(c, "root_bone", "")
    mid_path = getattr(c, "mid_bone", "")
    tip_path = getattr(c, "tip_bone", "")
    root_e = _resolve_bone_path(ent, root_path)
    if root_e is None:
        return
    rt = root_e.transform
    if rt is None:
        return
    mid_e = _resolve_mid_or_tip(ent, root_e, mid_path)
    tip_e = _resolve_mid_or_tip(ent, root_e, tip_path)
    if mid_e is None:
        mid_e = root_e._children[0] if root_e._children else None
    if tip_e is None and mid_e is not None:
        tip_e = mid_e._children[0] if mid_e._children else None
    if mid_e is None or tip_e is None:
        return
    if mid_e.transform is None or tip_e.transform is None:
        return

    _read_world_pos(rt, a)
    _read_world_pos(mid_e.transform, bpos)
    _read_world_pos(tip_e.transform, cpos)
    has_pole = _read_source_pos(c, "pole_entity_id", "pole_position", pole)
    target_ok = _read_source_pos(c, "target_entity_id", "target_position", T)
    if not target_ok:
        return
    stretch = float(getattr(c, "stretch", 0.0))
    weight = float(getattr(c, "weight", 1.0))
    bendv = getattr(c, "bend_positive", True)
    bend = 1.0
    if isinstance(bendv, bool):
        bend = 1.0 if bendv else -1.0

    _two_bone_geometry(a, bpos, cpos, T, pole, has_pole, stretch, bend, b_new, qa, qb)

    # Root bone: rotate original a->mid direction onto b_new - a.
    d0[0] = bpos[0] - a[0]; d0[1] = bpos[1] - a[1]; d0[2] = bpos[2] - a[2]
    _vec_norm(d0, d0)
    d1[0] = b_new[0] - a[0]; d1[1] = b_new[1] - a[1]; d1[2] = b_new[2] - a[2]
    _vec_norm(d1, d1)
    _quat_from_two_vecs(d0, d1, qd)
    _apply_world_delta(rt, qd, weight)

    # Mid bone: rotate its CURRENT (post-root) mid->tip direction onto T - b_new.
    _read_world_pos(mid_e.transform, bpos)
    _read_world_pos(tip_e.transform, cpos)
    d0[0] = cpos[0] - bpos[0]; d0[1] = cpos[1] - bpos[1]; d0[2] = cpos[2] - bpos[2]
    _vec_norm(d0, d0)
    d1[0] = T[0] - b_new[0]; d1[1] = T[1] - b_new[1]; d1[2] = T[2] - b_new[2]
    _vec_norm(d1, d1)
    _quat_from_two_vecs(d0, d1, qd)
    _apply_world_delta(mid_e.transform, qd, weight)


cdef void _update_fabrik(object c) noexcept:
    cdef object ent = c._entity
    cdef str root_path
    cdef list paths
    cdef int n, chain_len, k
    cdef object cur, bones
    cdef double T[3], pole[3]
    cdef int has_pole, target_ok
    cdef int iters
    cdef double tol, weight
    cdef list weights
    cdef Py_ssize_t i
    cdef double pw[3]
    cdef double qd[4], target[4], local[4], parent_wq[4]
    cdef double d0[3], d1[3], up[3], pole_rel[3], chain_dir[3]
    cdef double dotp
    cdef double w_apply
    cdef double[:, ::1] Pv, Ov, WQv
    cdef double[::1] Lv
    cdef object tt, root_par
    cdef int use_pole_up

    if ent is None:
        return
    root_path = getattr(c, "root_bone", "")
    paths = getattr(c, "bones", None) or []
    cur = _resolve_bone_path(ent, root_path)
    if cur is None:
        return
    if paths:
        bones = [None] * len(paths)
        for i in range(len(paths)):
            bones[i] = _resolve_bone_path(ent, paths[i])
    else:
        chain_len = int(getattr(c, "chain_length", 0))
        bones = [cur]
        for k in range(1, chain_len):
            if not cur._children:
                break
            cur = cur._children[0]
            bones.append(cur)
    n = len(bones)
    if n < 2:
        return
    for i in range(n):
        if bones[i] is None or bones[i].transform is None:
            return

    cdef np.ndarray[double, ndim=2, mode="c"] P = np.empty((n, 3), dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] ORIG = np.empty((n, 3), dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] LEN = np.empty((n - 1), dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] WQ = np.empty((n, 4), dtype=np.float64)
    Pv = P
    Ov = ORIG
    Lv = LEN
    WQv = WQ

    for i in range(n):
        tt = bones[i].transform
        _read_world_pos(tt, pw)
        Pv[i, 0] = pw[0]; Pv[i, 1] = pw[1]; Pv[i, 2] = pw[2]
        Ov[i, 0] = pw[0]; Ov[i, 1] = pw[1]; Ov[i, 2] = pw[2]
        _read_world_quat_semantic(tt, pw)
        WQv[i, 0] = pw[0]; WQv[i, 1] = pw[1]; WQv[i, 2] = pw[2]; WQv[i, 3] = pw[3]
    for i in range(n - 1):
        Lv[i] = sqrt((Ov[i+1,0]-Ov[i,0])**2 + (Ov[i+1,1]-Ov[i,1])**2 + (Ov[i+1,2]-Ov[i,2])**2)
        if Lv[i] < _EPS:
            Lv[i] = _EPS

    has_pole = _read_source_pos(c, "pole_entity_id", "pole_position", pole)
    target_ok = _read_source_pos(c, "target_entity_id", "target_position", T)
    if not target_ok:
        return
    iters = int(getattr(c, "iterations", 8))
    tol = float(getattr(c, "tolerance", 1e-4))
    weight = float(getattr(c, "weight", 1.0))
    weights = getattr(c, "bone_weights", None) or []

    _fabrik_positions(&Pv[0,0], &Ov[0,0], n, &Lv[0], T, iters, tol)

    chain_dir[0] = Pv[n-1,0] - Ov[0,0]
    chain_dir[1] = Pv[n-1,1] - Ov[0,1]
    chain_dir[2] = Pv[n-1,2] - Ov[0,2]
    _vec_norm(chain_dir, chain_dir)

    use_pole_up = 0
    if has_pole:
        _vec_sub(pole, &Ov[0,0], pole_rel)
        _vec_norm(pole_rel, pole_rel)
        _vec_cross(chain_dir, pole_rel, d0)
        if _vec_len(d0) > 1e-4:
            use_pole_up = 1

    root_par = bones[0]._parent
    if root_par is not None and root_par.transform is not None:
        _read_world_quat_semantic(root_par.transform, parent_wq)
    else:
        _quat_identity(parent_wq)

    for i in range(n - 1):
        d0[0] = Ov[i+1,0] - Ov[i,0]
        d0[1] = Ov[i+1,1] - Ov[i,1]
        d0[2] = Ov[i+1,2] - Ov[i,2]
        _vec_norm(d0, d0)
        d1[0] = Pv[i+1,0] - Pv[i,0]
        d1[1] = Pv[i+1,1] - Pv[i,1]
        d1[2] = Pv[i+1,2] - Pv[i,2]
        _vec_norm(d1, d1)

        if use_pole_up:
            dotp = d1[0]*pole_rel[0] + d1[1]*pole_rel[1] + d1[2]*pole_rel[2]
            up[0] = pole_rel[0] - d1[0]*dotp
            up[1] = pole_rel[1] - d1[1]*dotp
            up[2] = pole_rel[2] - d1[2]*dotp
            _vec_norm(up, up)
            if _vec_len(up) > 1e-6:
                # aim along segment with up rolled toward the pole
                _look_rotation(d1[0], d1[1], d1[2], up[0], up[1], up[2], qd)
            else:
                _quat_from_two_vecs(d0, d1, qd)
        else:
            _quat_from_two_vecs(d0, d1, qd)

        # target world rotation of this bone in its ORIGINAL frame:
        _quat_mul(qd, &WQv[i,0], target)
        # engine composes world = parent * local, so local = conj(parent) * target.
        # parent is the bone before us (or the real scene parent for the root).
        _quat_to_local(parent_wq, target, local)
        _quat_norm_inplace(local)

        w_apply = weight
        if len(weights) > 0:
            try:
                if i < len(weights):
                    w_apply = weight * float(weights[i])
            except Exception:
                pass
        _apply_local(bones[i].transform, local, w_apply)

        # this bone's target world becomes the parent world for the next bone
        parent_wq[0] = target[0]
        parent_wq[1] = target[1]
        parent_wq[2] = target[2]
        parent_wq[3] = target[3]

    # tip bone keeps its orientation (maintains child rotation)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

cdef void _update_one(object c) noexcept:
    cdef str tname = type(c).__name__
    if tname == "TwoBoneIK":
        _update_two_bone(c)
    elif tname == "FABRIKChain":
        _update_fabrik(c)


# ---------------------------------------------------------------------------
# Public API (used by tests and the pure-Python component fallback)
# ---------------------------------------------------------------------------

def two_bone_flat(a, b, c, target, pole=None, double stretch=0.0, double bend=1.0):
    """Analytic two-bone solve.

    Returns flat ndarray [mid_new_x,y,z, qa_x,y,z,w, qb_x,y,z,w].
    a, b, c, target, pole must be (3,) float arrays.
    """
    cdef double[:, ::1] A = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    cdef double[:, ::1] B = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    cdef double[:, ::1] C = np.asarray(c, dtype=np.float64).reshape(-1, 3)
    cdef double[:, ::1] TT = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    cdef double[:, ::1] PP
    cdef double _a[3], _b[3], _c[3], _T[3], _pole[3]
    cdef int has_pole = 0
    cdef double _bnew[3], _qa[4], _qb[4]
    _a[0] = A[0,0]; _a[1] = A[0,1]; _a[2] = A[0,2]
    _b[0] = B[0,0]; _b[1] = B[0,1]; _b[2] = B[0,2]
    _c[0] = C[0,0]; _c[1] = C[0,1]; _c[2] = C[0,2]
    _T[0] = TT[0,0]; _T[1] = TT[0,1]; _T[2] = TT[0,2]
    if pole is not None:
        PP = np.asarray(pole, dtype=np.float64).reshape(-1, 3)
        _pole[0] = PP[0,0]; _pole[1] = PP[0,1]; _pole[2] = PP[0,2]
        has_pole = 1
    _two_bone_geometry(_a, _b, _c, _T, _pole, has_pole, stretch, bend, _bnew, _qa, _qb)
    np_res = np.empty(11, dtype=np.float64)
    np_res[0] = _bnew[0]; np_res[1] = _bnew[1]; np_res[2] = _bnew[2]
    np_res[3] = _qa[0]; np_res[4] = _qa[1]; np_res[5] = _qa[2]; np_res[6] = _qa[3]
    np_res[7] = _qb[0]; np_res[8] = _qb[1]; np_res[9] = _qb[2]; np_res[10] = _qb[3]
    return np_res


def fabrik_flat(points, target, int iterations=10, double tolerance=1e-5):
    """Iterative FABRIK solve. Returns ndarray (n, 3) of new joint positions."""
    cdef np.ndarray[double, ndim=2, mode="c"] P = np.asarray(points, dtype=np.float64).copy()
    cdef int n = P.shape[0]
    cdef double[:, ::1] Pv = P
    cdef double[::1] Tv = np.asarray(target, dtype=np.float64).reshape(-1)
    if n < 2 or Tv.shape[0] < 3:
        return P
    cdef np.ndarray[double, ndim=2, mode="c"] ORIG = P.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] LEN = np.empty(n - 1, dtype=np.float64)
    cdef double[:, ::1] Ov = ORIG
    cdef double[::1] Lv = LEN
    cdef int i
    for i in range(n - 1):
        Lv[i] = sqrt((Ov[i+1,0]-Ov[i,0])**2 + (Ov[i+1,1]-Ov[i,1])**2 + (Ov[i+1,2]-Ov[i,2])**2)
        if Lv[i] < _EPS:
            Lv[i] = _EPS
    cdef double T[3]
    T[0] = Tv[0]; T[1] = Tv[1]; T[2] = Tv[2]
    _fabrik_positions(&Pv[0,0], &Ov[0,0], n, &Lv[0], T, iterations, tolerance)
    return P


def batch_update_ik(list components, double dt):
    cdef Py_ssize_t i, n = len(components)
    cdef object c
    import sys as _sys
    for i in range(n):
        c = components[i]
        try:
            _update_one(c)
        except Exception as ex:
            _sys.stderr.write("IK batch error in %s: %s\n" % (type(c).__name__, ex))
            _sys.stderr.flush()