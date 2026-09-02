# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from libc.math cimport fabs, atan2, asin, sqrt, sin as csin, cos as ccos
from core._math_vec cimport Vec2, Vec3, Quat

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
        if getattr(tr, "_physics_dirty", False):
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
            try:
                solver.activate(body_id)
            except Exception:
                pass
            tr._physics_dirty = False
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
    from core.math_helpers import quat_from_euler_rad
    cdef Py_ssize_t i, n = len(items)
    cdef int body_id
    cdef bint is_2d
    cdef double qr_x, qr_y, qr_z, qr_w
    for i in range(n):
        entity_id, body_id, entity, rb, tr, is_2d = items[i]
        if not entity._active or rb.is_kinematic or getattr(tr, "_physics_dirty", False):
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
    cdef unsigned char[:] flags = shared._flags_nd
    cdef float[:, :] rdata = shared._rdata_nd
    cdef Py_ssize_t i, n = len(cache)
    cdef int slot
    cdef unsigned char fl
    cdef object entity, rb, rb2d, tr, scn
    cdef Vec3 lp
    cdef Quat lq
    cdef Vec2 vel2, fa2
    cdef Vec3 vel3, av3
    for i in range(n):
        entity, rb, rb2d, tr, slot = cache[i]
        if getattr(tr, "_physics_dirty", False):
            continue
        fl = flags[slot]
        if not (fl & 1) or (fl & 4):
            continue
        if rb2d is not None:
            lp = tr._local_pos
            lq = tr._local_rot
            if fabs(lp._x - rdata[slot, 0]) < 1e-5 and fabs(lp._y - rdata[slot, 1]) < 1e-5 and fabs(lq._z - rdata[slot, 5]) < 1e-4 and fabs(lq._w - rdata[slot, 6]) < 1e-4:
                if not rb2d._velocity_dirty:
                    vel2 = rb2d._velocity
                    vel2._x = rdata[slot, 7]
                    vel2._y = rdata[slot, 8]
                    rb2d._angular_velocity = rdata[slot, 12]
                fa2 = rb2d._force_accum
                fa2._x = 0.0
                fa2._y = 0.0
                rb2d._torque_accum = 0.0
                continue
            lp._x = rdata[slot, 0]
            lp._y = rdata[slot, 1]
            lp._z = 0.0
            lq._x = 0.0
            lq._y = 0.0
            lq._z = rdata[slot, 5]
            lq._w = rdata[slot, 6]
            tr._dirty = True
            scn = entity._scene
            if scn is not None:
                scn._dirty_roots.add(tr)
                scn._spatial_dirty_entities.add(entity.id)
                scn._spatial_dirty = True
            if not rb2d._velocity_dirty:
                vel2 = rb2d._velocity
                vel2._x = rdata[slot, 7]
                vel2._y = rdata[slot, 8]
                rb2d._angular_velocity = rdata[slot, 12]
            fa2 = rb2d._force_accum
            fa2._x = 0.0
            fa2._y = 0.0
            rb2d._torque_accum = 0.0
        elif rb is not None:
            lp = tr._local_pos
            lq = tr._local_rot
            if fabs(lp._x - rdata[slot, 0]) < 1e-5 and fabs(lp._y - rdata[slot, 1]) < 1e-5 and fabs(lp._z - rdata[slot, 2]) < 1e-5 and fabs(lq._x - rdata[slot, 3]) < 1e-4 and fabs(lq._y - rdata[slot, 4]) < 1e-4 and fabs(lq._z - rdata[slot, 5]) < 1e-4 and fabs(lq._w - rdata[slot, 6]) < 1e-4:
                if not rb._velocity_dirty:
                    vel3 = rb._velocity
                    vel3._x = rdata[slot, 7]
                    vel3._y = rdata[slot, 8]
                    vel3._z = rdata[slot, 9]
                    av3 = rb._angular_velocity
                    av3._x = rdata[slot, 10]
                    av3._y = rdata[slot, 11]
                    av3._z = rdata[slot, 12]
                continue
            lp._x = rdata[slot, 0]
            lp._y = rdata[slot, 1]
            lp._z = rdata[slot, 2]
            lq._x = rdata[slot, 3]
            lq._y = rdata[slot, 4]
            lq._z = rdata[slot, 5]
            lq._w = rdata[slot, 6]
            tr._dirty = True
            scn = entity._scene
            if scn is not None:
                scn._dirty_roots.add(tr)
                scn._spatial_dirty_entities.add(entity.id)
                scn._spatial_dirty = True
            if not rb._velocity_dirty:
                vel3 = rb._velocity
                vel3._x = rdata[slot, 7]
                vel3._y = rdata[slot, 8]
                vel3._z = rdata[slot, 9]
                av3 = rb._angular_velocity
                av3._x = rdata[slot, 10]
                av3._y = rdata[slot, 11]
                av3._z = rdata[slot, 12]


def sync_write_from_ecs(object shared, list cache):
    cdef unsigned char[:] flags = shared._flags_nd
    cdef float[:, :] edata = shared._edata_nd
    cdef float[:, :] fdata = shared._fdata_nd
    cdef Py_ssize_t i, n = len(cache)
    cdef int slot, max_slot = -1
    cdef object entity, rb, rb2d, tr
    cdef Vec3 lp
    cdef Quat q
    cdef Vec2 vel2, fa2
    cdef Vec3 vel3, av3, fa3, ta3
    for i in range(n):
        entity, rb, rb2d, tr, slot = cache[i]
        if not entity._active:
            continue
        lp = tr._local_pos
        if rb2d is not None:
            fa2 = rb2d._force_accum
            vel2 = rb2d._velocity
            if getattr(tr, "_physics_dirty", False):
                q = tr._local_rot
                edata[slot, 0] = lp._x
                edata[slot, 1] = lp._y
                edata[slot, 2] = 0.0
                edata[slot, 3] = 0.0
                edata[slot, 4] = 0.0
                edata[slot, 5] = q._z
                edata[slot, 6] = q._w
                edata[slot, 7] = vel2._x
                edata[slot, 8] = vel2._y
                edata[slot, 9] = 0.0
                edata[slot, 10] = 0.0
                edata[slot, 11] = 0.0
                edata[slot, 12] = rb2d._angular_velocity
                fdata[slot, 0] = fa2._x
                fdata[slot, 1] = fa2._y
                fdata[slot, 2] = 0.0
                fdata[slot, 3] = 0.0
                fdata[slot, 4] = 0.0
                fdata[slot, 5] = rb2d._torque_accum
                flags[slot] = 15
                tr._physics_dirty = False
            elif rb2d.is_kinematic:
                q = tr._local_rot
                edata[slot, 0] = lp._x
                edata[slot, 1] = lp._y
                edata[slot, 2] = 0.0
                edata[slot, 3] = 0.0
                edata[slot, 4] = 0.0
                edata[slot, 5] = q._z
                edata[slot, 6] = q._w
                edata[slot, 7] = vel2._x
                edata[slot, 8] = vel2._y
                edata[slot, 9] = 0.0
                edata[slot, 10] = 0.0
                edata[slot, 11] = 0.0
                edata[slot, 12] = rb2d._angular_velocity
                fdata[slot, 0] = fa2._x
                fdata[slot, 1] = fa2._y
                fdata[slot, 2] = 0.0
                fdata[slot, 3] = 0.0
                fdata[slot, 4] = 0.0
                fdata[slot, 5] = rb2d._torque_accum
                flags[slot] = 15
            else:
                edata[slot, 0] = 0.0
                edata[slot, 1] = 0.0
                edata[slot, 2] = 0.0
                edata[slot, 3] = 0.0
                edata[slot, 4] = 0.0
                edata[slot, 5] = 0.0
                edata[slot, 6] = 0.0
                edata[slot, 7] = vel2._x
                edata[slot, 8] = vel2._y
                edata[slot, 9] = 0.0
                edata[slot, 10] = 0.0
                edata[slot, 11] = 0.0
                edata[slot, 12] = rb2d._angular_velocity
                fdata[slot, 0] = fa2._x
                fdata[slot, 1] = fa2._y
                fdata[slot, 2] = 0.0
                fdata[slot, 3] = 0.0
                fdata[slot, 4] = 0.0
                fdata[slot, 5] = rb2d._torque_accum
                flags[slot] = 11 if rb2d._velocity_dirty else 9
                rb2d._velocity_dirty = False
        elif rb is not None:
            fa3 = rb._force_accum
            ta3 = rb._torque_accum
            vel3 = rb._velocity
            av3 = rb._angular_velocity
            if getattr(tr, "_physics_dirty", False):
                q = tr._local_rot
                edata[slot, 0] = lp._x
                edata[slot, 1] = lp._y
                edata[slot, 2] = lp._z
                edata[slot, 3] = q._x
                edata[slot, 4] = q._y
                edata[slot, 5] = q._z
                edata[slot, 6] = q._w
                edata[slot, 7] = vel3._x
                edata[slot, 8] = vel3._y
                edata[slot, 9] = vel3._z
                edata[slot, 10] = av3._x
                edata[slot, 11] = av3._y
                edata[slot, 12] = av3._z
                fdata[slot, 0] = fa3._x
                fdata[slot, 1] = fa3._y
                fdata[slot, 2] = fa3._z
                fdata[slot, 3] = ta3._x
                fdata[slot, 4] = ta3._y
                fdata[slot, 5] = ta3._z
                fa3._x = 0.0
                fa3._y = 0.0
                fa3._z = 0.0
                ta3._x = 0.0
                ta3._y = 0.0
                ta3._z = 0.0
                flags[slot] = 7
                tr._physics_dirty = False
            elif rb.is_kinematic:
                q = tr._local_rot
                edata[slot, 0] = lp._x
                edata[slot, 1] = lp._y
                edata[slot, 2] = lp._z
                edata[slot, 3] = q._x
                edata[slot, 4] = q._y
                edata[slot, 5] = q._z
                edata[slot, 6] = q._w
                edata[slot, 7] = vel3._x
                edata[slot, 8] = vel3._y
                edata[slot, 9] = vel3._z
                edata[slot, 10] = av3._x
                edata[slot, 11] = av3._y
                edata[slot, 12] = av3._z
                fdata[slot, 0] = fa3._x
                fdata[slot, 1] = fa3._y
                fdata[slot, 2] = fa3._z
                fdata[slot, 3] = ta3._x
                fdata[slot, 4] = ta3._y
                fdata[slot, 5] = ta3._z
                fa3._x = 0.0
                fa3._y = 0.0
                fa3._z = 0.0
                ta3._x = 0.0
                ta3._y = 0.0
                ta3._z = 0.0
                flags[slot] = 7
            else:
                edata[slot, 0] = 0.0
                edata[slot, 1] = 0.0
                edata[slot, 2] = 0.0
                edata[slot, 3] = 0.0
                edata[slot, 4] = 0.0
                edata[slot, 5] = 0.0
                edata[slot, 6] = 0.0
                edata[slot, 7] = vel3._x
                edata[slot, 8] = vel3._y
                edata[slot, 9] = vel3._z
                edata[slot, 10] = av3._x
                edata[slot, 11] = av3._y
                edata[slot, 12] = av3._z
                fdata[slot, 0] = fa3._x
                fdata[slot, 1] = fa3._y
                fdata[slot, 2] = fa3._z
                fdata[slot, 3] = ta3._x
                fdata[slot, 4] = ta3._y
                fdata[slot, 5] = ta3._z
                fa3._x = 0.0
                fa3._y = 0.0
                fa3._z = 0.0
                ta3._x = 0.0
                ta3._y = 0.0
                ta3._z = 0.0
                flags[slot] = 3 if rb._velocity_dirty else 1
                rb._velocity_dirty = False
        if slot > max_slot:
            max_slot = slot
    return max_slot
