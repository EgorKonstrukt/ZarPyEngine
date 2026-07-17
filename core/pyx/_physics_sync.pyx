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
