# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from libc.math cimport fabs, atan2, asin, sqrt, sin as csin, cos as ccos

cdef inline double _clamp(double v, double lo, double hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


cdef inline void _quat_to_euler_rad(double qx, double qy, double qz, double qw,
                                     double *rx, double *ry, double *rz):
    cdef double sinx = 2.0 * (qw * qx + qy * qz)
    cdef double cosx = 1.0 - 2.0 * (qx * qx + qy * qy)
    rx[0] = atan2(sinx, cosx)
    cdef double siny = 2.0 * (qw * qy - qz * qx)
    ry[0] = asin(_clamp(siny, -1.0, 1.0))
    cdef double sinz = 2.0 * (qw * qz + qx * qy)
    cdef double cosz = 1.0 - 2.0 * (qy * qy + qz * qz)
    rz[0] = atan2(sinz, cosz)


cdef inline void _euler_to_quat_rad(double rx, double ry, double rz,
                                     double *qx, double *qy, double *qz, double *qw):
    cdef double hx = rx * 0.5
    cdef double hy = ry * 0.5
    cdef double hz = rz * 0.5
    cdef double sx = csin(hx), cx = ccos(hx)
    cdef double sy = csin(hy), cy = ccos(hy)
    cdef double sz = csin(hz), cz = ccos(hz)
    qx[0] = sx * cy * cz - cx * sy * sz
    qy[0] = cx * sy * cz + sx * cy * sz
    qz[0] = cx * cy * sz - sx * sy * cz
    qw[0] = cx * cy * cz + sx * sy * sz


def batch_sync_ecs_to_physics(list items, object solver):
    """items: list of (entity_id, body_id, entity, rb, tr, is_2d)"""
    cdef Py_ssize_t i, n = len(items)
    cdef int body_id
    cdef bint is_2d
    cdef double ex, ey, ez
    cdef double fx, fy, fz
    cdef double rx, ry, rz, rw
    cdef double er_x, er_y, er_z
    for i in range(n):
        entity_id, body_id, entity, rb, tr, is_2d = items[i]
        if not entity._active:
            continue
        if rb.is_kinematic:
            p = tr._local_pos
            if is_2d:
                _quat_to_euler_rad(tr._local_rot._x, tr._local_rot._y,
                                   tr._local_rot._z, tr._local_rot._w,
                                   &er_x, &er_y, &er_z)
                solver.set_body_transform(body_id,
                    (p._x, p._y, 0.0),
                    (0.0, 0.0, er_z))
            else:
                _quat_to_euler_rad(tr._local_rot._x, tr._local_rot._y,
                                   tr._local_rot._z, tr._local_rot._w,
                                   &er_x, &er_y, &er_z)
                solver.set_body_transform(body_id,
                    (p._x, p._y, p._z), (er_x, er_y, er_z))
        fa = rb._force_accum
        if is_2d:
            if fa._x != 0.0 or fa._y != 0.0:
                solver.apply_force(body_id, (fa._x, fa._y, 0.0))
            if fabs(rb._torque_accum) > 1e-10:
                solver.apply_torque(body_id, (0.0, 0.0, rb._torque_accum))
        else:
            fx = fa._x; fy = fa._y; fz = fa._z
            if fx * fx + fy * fy + fz * fz > 1e-10:
                solver.apply_force(body_id, (fx, fy, fz))
            ta = rb._torque_accum
            rx = ta._x; ry = ta._y; rz = ta._z
            if rx * rx + ry * ry + rz * rz > 1e-10:
                solver.apply_torque(body_id, (rx, ry, rz))


def batch_sync_physics_to_ecs(list items, object solver):
    """items: list of (entity_id, body_id, entity, rb, tr, is_2d)"""
    from core.math_helpers import quat_from_euler_rad
    cdef Py_ssize_t i, n = len(items)
    cdef int body_id
    cdef bint is_2d
    cdef double qr_x, qr_y, qr_z, qr_w
    for i in range(n):
        entity_id, body_id, entity, rb, tr, is_2d = items[i]
        if not entity._active or rb.is_kinematic:
            continue
        pos, rot = solver.get_body_transform(body_id)
        vel = solver.get_velocity(body_id)
        ang_vel = solver.get_angular_velocity(body_id)
        tr._local_pos._x = pos[0]
        tr._local_pos._y = pos[1]
        tr._local_pos._z = 0.0 if is_2d else pos[2]
        if is_2d:
            q = quat_from_euler_rad(0.0, 0.0, rot[2])
        else:
            q = quat_from_euler_rad(rot[0], rot[1], rot[2])
        tr._local_rot._x = q[0]
        tr._local_rot._y = q[1]
        tr._local_rot._z = q[2]
        tr._local_rot._w = q[3]
        tr._dirty = True
        rb._velocity._x = vel[0]
        rb._velocity._y = vel[1]
        if not is_2d:
            rb._velocity._z = vel[2]
            rb._angular_velocity._x = ang_vel[0]
            rb._angular_velocity._y = ang_vel[1]
            rb._angular_velocity._z = ang_vel[2]
        else:
            rb._angular_velocity = ang_vel[2]
        rb._force_accum._x = 0.0
        rb._force_accum._y = 0.0
        if not is_2d:
            rb._force_accum._z = 0.0
            rb._torque_accum._x = 0.0
            rb._torque_accum._y = 0.0
            rb._torque_accum._z = 0.0
        else:
            rb._torque_accum = 0.0


def sync_read_to_ecs(object shared, list cache):
    """cache: list of (entity, rb, rb2d, tr, slot)"""
    cdef unsigned char[:] flags = shared._flags_nd
    cdef float[:, :] rdata = shared._rdata_nd
    cdef Py_ssize_t i, n = len(cache)
    cdef int slot
    cdef unsigned char fl
    cdef double hz, r0, r1, r2
    cdef double sr, cr, sp, cp, sy, cy
    cdef object entity, rb, rb2d, tr, lp, lq, vel, av, fa
    for i in range(n):
        entity, rb, rb2d, tr, slot = cache[i]
        fl = flags[slot]
        if not (fl & 1) or (fl & 4):
            continue
        if rb2d is not None:
            lp = tr._local_pos
            lp._x = rdata[slot, 0]
            lp._y = rdata[slot, 1]
            lp._z = 0.0
            hz = rdata[slot, 5] * 0.5
            lq = tr._local_rot
            lq._x = 0.0
            lq._y = 0.0
            lq._z = csin(hz)
            lq._w = ccos(hz)
            tr._dirty = True
            if not rb2d._velocity_dirty:
                vel = rb2d._velocity
                vel._x = rdata[slot, 6]
                vel._y = rdata[slot, 7]
                rb2d._angular_velocity = rdata[slot, 11]
            fa = rb2d._force_accum
            fa._x = 0.0
            fa._y = 0.0
            rb2d._torque_accum = 0.0
        elif rb is not None:
            lp = tr._local_pos
            lp._x = rdata[slot, 0]
            lp._y = rdata[slot, 1]
            lp._z = rdata[slot, 2]
            r0 = rdata[slot, 3]
            r1 = rdata[slot, 4]
            r2 = rdata[slot, 5]
            sr, cr = csin(r0 * 0.5), ccos(r0 * 0.5)
            sp, cp = csin(r1 * 0.5), ccos(r1 * 0.5)
            sy, cy = csin(r2 * 0.5), ccos(r2 * 0.5)
            lq = tr._local_rot
            lq._x = sr * cp * cy - cr * sp * sy
            lq._y = cr * sp * cy + sr * cp * sy
            lq._z = cr * cp * sy - sr * sp * cy
            lq._w = cr * cp * cy + sr * sp * sy
            tr._dirty = True
            if not rb._velocity_dirty:
                vel = rb._velocity
                vel._x = rdata[slot, 6]
                vel._y = rdata[slot, 7]
                vel._z = rdata[slot, 8]
                av = rb._angular_velocity
                av._x = rdata[slot, 9]
                av._y = rdata[slot, 10]
                av._z = rdata[slot, 11]


def sync_write_from_ecs(object shared, list cache):
    """cache: list of (entity, rb, rb2d, tr, slot)"""
    cdef unsigned char[:] flags = shared._flags_nd
    cdef float[:, :] edata = shared._edata_nd
    cdef float[:, :] fdata = shared._fdata_nd
    cdef Py_ssize_t i, n = len(cache)
    cdef int slot, max_slot = -1
    cdef double qx, qy, qz, qw, sz
    cdef object entity, rb, rb2d, tr, lp, q, fa, ta, vel, av
    for i in range(n):
        entity, rb, rb2d, tr, slot = cache[i]
        if not entity._active:
            continue
        lp = tr._local_pos
        if rb2d is not None:
            q = tr._local_rot
            sz = 2.0 * asin(_clamp(q._z, -1.0, 1.0))
            fa = rb2d._force_accum
            vel = rb2d._velocity
            edata[slot, 0] = lp._x
            edata[slot, 1] = lp._y
            edata[slot, 2] = 0.0
            edata[slot, 3] = 0.0
            edata[slot, 4] = 0.0
            edata[slot, 5] = sz
            edata[slot, 6] = vel._x
            edata[slot, 7] = vel._y
            edata[slot, 8] = 0.0
            edata[slot, 9] = 0.0
            edata[slot, 10] = 0.0
            edata[slot, 11] = rb2d._angular_velocity
            fdata[slot, 0] = fa._x
            fdata[slot, 1] = fa._y
            fdata[slot, 2] = 0.0
            fdata[slot, 3] = 0.0
            fdata[slot, 4] = 0.0
            fdata[slot, 5] = rb2d._torque_accum
            if rb2d.is_kinematic:
                flags[slot] = 15
            else:
                flags[slot] = 11 if rb2d._velocity_dirty else 9
                rb2d._velocity_dirty = False
        elif rb is not None:
            q = tr._local_rot
            qx = q._x
            qy = q._y
            qz = q._z
            qw = q._w
            fa = rb._force_accum
            ta = rb._torque_accum
            vel = rb._velocity
            av = rb._angular_velocity
            edata[slot, 0] = lp._x
            edata[slot, 1] = lp._y
            edata[slot, 2] = lp._z
            edata[slot, 3] = atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
            edata[slot, 4] = asin(_clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
            edata[slot, 5] = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            edata[slot, 6] = vel._x
            edata[slot, 7] = vel._y
            edata[slot, 8] = vel._z
            edata[slot, 9] = av._x
            edata[slot, 10] = av._y
            edata[slot, 11] = av._z
            fdata[slot, 0] = fa._x
            fdata[slot, 1] = fa._y
            fdata[slot, 2] = fa._z
            fdata[slot, 3] = ta._x
            fdata[slot, 4] = ta._y
            fdata[slot, 5] = ta._z
            fa._x = 0.0
            fa._y = 0.0
            fa._z = 0.0
            ta._x = 0.0
            ta._y = 0.0
            ta._z = 0.0
            if rb.is_kinematic:
                flags[slot] = 7
            else:
                flags[slot] = 3 if rb._velocity_dirty else 1
                rb._velocity_dirty = False
        if slot > max_slot:
            max_slot = slot
    return max_slot
