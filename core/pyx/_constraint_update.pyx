# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from libc.math cimport fabs, sqrt, atan2, asin, sin, cos, acos, fmin, fmax

cdef inline double _clamp(double v, double lo, double hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


cdef inline double _vec3_length(double x, double y, double z):
    return sqrt(x * x + y * y + z * z)


cdef inline double _vec3_normalize_inplace(double *x, double *y, double *z):
    cdef double n = sqrt(x[0] * x[0] + y[0] * y[0] + z[0] * z[0])
    cdef double inv = 0.0
    if n > 1e-10:
        inv = 1.0 / n
        x[0] *= inv
        y[0] *= inv
        z[0] *= inv
    return n


cdef inline void _quat_normalize_inplace(double *x, double *y, double *z, double *w):
    cdef double n = sqrt(x[0] * x[0] + y[0] * y[0] + z[0] * z[0] + w[0] * w[0])
    cdef double inv = 0.0
    if n > 1e-10:
        inv = 1.0 / n
        x[0] *= inv
        y[0] *= inv
        z[0] *= inv
        w[0] *= inv


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
    cdef double sx = sin(hx), cx = cos(hx)
    cdef double sy = sin(hy), cy = cos(hy)
    cdef double sz = sin(hz), cz = cos(hz)
    qx[0] = sx * cy * cz - cx * sy * sz
    qy[0] = cx * sy * cz + sx * cy * sz
    qz[0] = cx * cy * sz - sx * sy * cz
    qw[0] = cx * cy * cz + sx * sy * sz


cdef inline void _quat_mul(double ax, double ay, double az, double aw,
                            double bx, double by, double bz, double bw,
                            double *rx, double *ry, double *rz, double *rw):
    rx[0] = aw * bx + ax * bw + ay * bz - az * by
    ry[0] = aw * by - ax * bz + ay * bw + az * bx
    rz[0] = aw * bz + ax * by - ay * bx + az * bw
    rw[0] = aw * bw - ax * bx - ay * by - az * bz


cdef inline void _quat_conjugate(double x, double y, double z, double w,
                                  double *rx, double *ry, double *rz, double *rw):
    rx[0] = -x
    ry[0] = -y
    rz[0] = -z
    rw[0] = w


cdef inline void _quat_slerp(double ax, double ay, double az, double aw,
                              double bx, double by, double bz, double bw,
                              double t, double *rx, double *ry, double *rz, double *rw):
    cdef double dot = ax * bx + ay * by + az * bz + aw * bw
    cdef double s0, s1, theta, sin_theta
    if dot < 0.0:
        bx = -bx; by = -by; bz = -bz; bw = -bw; dot = -dot
    if dot > 0.9995:
        rx[0] = ax + t * (bx - ax)
        ry[0] = ay + t * (by - ay)
        rz[0] = az + t * (bz - az)
        rw[0] = aw + t * (bw - aw)
        _quat_normalize_inplace(rx, ry, rz, rw)
        return
    theta = acos(dot)
    sin_theta = sin(theta)
    if fabs(sin_theta) < 1e-10:
        rx[0] = ax; ry[0] = ay; rz[0] = az; rw[0] = aw
        return
    cdef double inv_sin = 1.0 / sin_theta
    cdef double theta_t = theta * t
    s0 = (cos(theta_t) - dot * sin(theta_t)) * inv_sin
    s1 = sin(theta_t) * inv_sin
    rx[0] = s0 * ax + s1 * bx
    ry[0] = s0 * ay + s1 * by
    rz[0] = s0 * az + s1 * bz
    rw[0] = s0 * aw + s1 * bw
    _quat_normalize_inplace(rx, ry, rz, rw)


cdef _get_source_transform(object constraint, dict src_data):
    scene = constraint._entity._scene if constraint._entity is not None else None
    if scene is None:
        return None
    entity_id = src_data.get("entity_id")
    if not entity_id:
        return None
    entity = scene.get_entity(entity_id)
    if entity is None:
        return None
    return entity.transform


cdef _compute_weighted_position(object constraint):
    cdef list sources = constraint.sources
    cdef list valid = []
    cdef dict s
    cdef Py_ssize_t i
    for i in range(len(sources)):
        s = sources[i]
        if s.get("weight", 0.0) > 1e-6:
            valid.append(s)
    if not valid:
        return None
    cdef double total_weight = 0.0
    for i in range(len(valid)):
        total_weight += (<dict>valid[i]).get("weight", 0.0)
    if total_weight < 1e-8:
        return None
    cdef double px = 0.0, py = 0.0, pz = 0.0
    cdef double w, inv_w
    cdef object st
    for i in range(len(valid)):
        s = <dict>valid[i]
        w = s.get("weight", 0.0)
        st = _get_source_transform(constraint, s)
        if st is None:
            continue
        wp = st.position
        inv_w = w / total_weight
        px += wp._x * inv_w
        py += wp._y * inv_w
        pz += wp._z * inv_w
    from core.maths.math3d import Vec3
    return Vec3._make(px, py, pz)


cdef _compute_weighted_rotation(object constraint):
    cdef list sources = constraint.sources
    cdef list valid = []
    cdef dict s
    cdef Py_ssize_t i
    for i in range(len(sources)):
        s = sources[i]
        if s.get("weight", 0.0) > 1e-6:
            valid.append(s)
    if not valid:
        return None
    cdef object st
    if len(valid) == 1:
        st = _get_source_transform(constraint, <dict>valid[0])
        if st is not None:
            return st.local_rotation
        return None
    cdef double total_weight = 0.0
    for i in range(len(valid)):
        total_weight += (<dict>valid[i]).get("weight", 0.0)
    if total_weight < 1e-8:
        return None
    cdef double sx = 0.0, sy = 0.0, sz = 0.0, sw = 0.0
    cdef double w, inv_w
    for i in range(len(valid)):
        s = <dict>valid[i]
        w = s.get("weight", 0.0)
        st = _get_source_transform(constraint, s)
        if st is None:
            continue
        q = st.local_rotation
        inv_w = w / total_weight
        sx += q._x * inv_w
        sy += q._y * inv_w
        sz += q._z * inv_w
        sw += q._w * inv_w
    _quat_normalize_inplace(&sx, &sy, &sz, &sw)
    from core.maths.math3d import Quat
    return Quat._make(sx, sy, sz, sw)


cdef _compute_weighted_scale(object constraint):
    cdef list sources = constraint.sources
    cdef list valid = []
    cdef dict s
    cdef Py_ssize_t i
    for i in range(len(sources)):
        s = sources[i]
        if s.get("weight", 0.0) > 1e-6:
            valid.append(s)
    if not valid:
        return None
    cdef double total_weight = 0.0
    for i in range(len(valid)):
        total_weight += (<dict>valid[i]).get("weight", 0.0)
    if total_weight < 1e-8:
        return None
    cdef double sx = 0.0, sy = 0.0, sz = 0.0
    cdef double w, inv_w
    cdef object st
    for i in range(len(valid)):
        s = <dict>valid[i]
        w = s.get("weight", 0.0)
        st = _get_source_transform(constraint, s)
        if st is None:
            continue
        sc = st.local_scale
        inv_w = w / total_weight
        sx += sc._x * inv_w
        sy += sc._y * inv_w
        sz += sc._z * inv_w
    from core.maths.math3d import Vec3
    return Vec3._make(sx, sy, sz)


cdef void _update_position_constraint(object c, double dt):
    if not c.is_active:
        return
    t = c.transform
    if t is None:
        return
    target = _compute_weighted_position(c)
    if target is None:
        return
    current = t.position
    cdef double ox = c._offset._x, oy = c._offset._y, oz = c._offset._z
    cdef double wx = c.weight_x, wy = c.weight_y, wz = c.weight_z
    cdef double rx = current._x + (target._x + ox - current._x) * wx
    cdef double ry = current._y + (target._y + oy - current._y) * wy
    cdef double rz = current._z + (target._z + oz - current._z) * wz
    from core.maths.math3d import Vec3
    t.position = Vec3._make(rx, ry, rz)


cdef void _update_rotation_constraint(object c, double dt):
    if not c.is_active:
        return
    t = c.transform
    if t is None:
        return
    target_quat = _compute_weighted_rotation(c)
    if target_quat is None:
        return
    cdef double cx, cy, cz, cw
    cur_q = t.local_rotation
    cx = cur_q._x; cy = cur_q._y; cz = cur_q._z; cw = cur_q._w
    cdef double ox = c._offset_rotation._x, oy = c._offset_rotation._y
    cdef double oz = c._offset_rotation._z, ow = c._offset_rotation._w
    cdef double tx, ty, tz, tw
    _quat_mul(target_quat._x, target_quat._y, target_quat._z, target_quat._w,
              ox, oy, oz, ow, &tx, &ty, &tz, &tw)
    cdef double te_x, te_y, te_z
    _quat_to_euler_rad(tx, ty, tz, tw, &te_x, &te_y, &te_z)
    cdef double ce_x, ce_y, ce_z
    _quat_to_euler_rad(cx, cy, cz, cw, &ce_x, &ce_y, &ce_z)
    cdef double wx = c.weight_x, wy = c.weight_y, wz = c.weight_z
    cdef double nx = ce_x + (te_x - ce_x) * wx
    cdef double ny = ce_y + (te_y - ce_y) * wy
    cdef double nz = ce_z + (te_z - ce_z) * wz
    from core.maths.math3d import Vec3
    t.local_euler_angles = Vec3._make(nx, ny, nz)


cdef void _update_scale_constraint(object c, double dt):
    if not c.is_active:
        return
    t = c.transform
    if t is None:
        return
    target = _compute_weighted_scale(c)
    if target is None:
        return
    current = t.local_scale
    cdef double ox = c._offset_scale._x, oy = c._offset_scale._y, oz = c._offset_scale._z
    cdef double wx = c.weight_x, wy = c.weight_y, wz = c.weight_z
    cdef double ex = target._x * ox, ey = target._y * oy, ez = target._z * oz
    cdef double rx = current._x + (ex - current._x) * wx
    cdef double ry = current._y + (ey - current._y) * wy
    cdef double rz = current._z + (ez - current._z) * wz
    from core.maths.math3d import Vec3
    t.local_scale = Vec3._make(rx, ry, rz)


cdef void _update_parent_constraint(object c, double dt):
    cdef double cx, cy, cz, cw
    cdef double rox, roy, roz, row
    cdef double ttx, tty, ttz, ttw
    cdef double te_x, te_y, te_z
    cdef double ce_x, ce_y, ce_z
    cdef double nx, ny, nz
    cdef double sox, soy, soz
    cdef double ex, ey, ez
    cdef double snx, sny, snz
    cdef double rx, ry, rz
    if not c.is_active:
        return
    t = c.transform
    if t is None:
        return
    from core.maths.math3d import Vec3

    pos_target = _compute_weighted_position(c)
    if pos_target is not None:
        current = t.position
        rx = current._x
        ry = current._y
        rz = current._z
        if c.constrain_position_x:
            rx = pos_target._x + c._position_offset._x
        if c.constrain_position_y:
            ry = pos_target._y + c._position_offset._y
        if c.constrain_position_z:
            rz = pos_target._z + c._position_offset._z
        t.position = Vec3._make(rx, ry, rz)

    rot_target = _compute_weighted_rotation(c)
    if rot_target is not None:
        cur_q = t.local_rotation
        cx = cur_q._x; cy = cur_q._y; cz = cur_q._z; cw = cur_q._w
        rox = c._rotation_offset._x; roy = c._rotation_offset._y
        roz = c._rotation_offset._z; row = c._rotation_offset._w
        _quat_mul(rot_target._x, rot_target._y, rot_target._z, rot_target._w,
                  rox, roy, roz, row, &ttx, &tty, &ttz, &ttw)
        _quat_to_euler_rad(ttx, tty, ttz, ttw, &te_x, &te_y, &te_z)
        _quat_to_euler_rad(cx, cy, cz, cw, &ce_x, &ce_y, &ce_z)
        nx = ce_x if not c.constrain_rotation_x else te_x
        ny = ce_y if not c.constrain_rotation_y else te_y
        nz = ce_z if not c.constrain_rotation_z else te_z
        t.local_euler_angles = Vec3._make(nx, ny, nz)

    scale_target = _compute_weighted_scale(c)
    if scale_target is not None:
        current_scale = t.local_scale
        sox = c._scale_offset._x; soy = c._scale_offset._y; soz = c._scale_offset._z
        ex = scale_target._x * sox
        ey = scale_target._y * soy
        ez = scale_target._z * soz
        snx = current_scale._x if not c.constrain_scale_x else ex
        sny = current_scale._y if not c.constrain_scale_y else ey
        snz = current_scale._z if not c.constrain_scale_z else ez
        t.local_scale = Vec3._make(snx, sny, snz)


cdef void _update_move_towards_constraint(object c, double dt):
    if not c.is_active:
        return
    t = c.transform
    if t is None:
        return
    target_pos = _compute_weighted_position(c)
    if target_pos is None:
        return
    current = t.position
    cdef double etx = target_pos._x
    cdef double ety = target_pos._y
    cdef double etz = target_pos._z
    if c.maintain_offset:
        etx += c._offset._x
        ety += c._offset._y
        etz += c._offset._z
    cdef double dx = etx - current._x
    cdef double dy = ety - current._y
    cdef double dz = etz - current._z
    cdef double dist = _vec3_length(dx, dy, dz)
    if dist < 1e-6:
        return
    cdef double move_dist = c.speed * dt
    if move_dist > dist:
        move_dist = dist
    cdef double inv_dist = move_dist / dist
    from core.maths.math3d import Vec3
    t.position = Vec3._make(current._x + dx * inv_dist,
                            current._y + dy * inv_dist,
                            current._z + dz * inv_dist)


cdef void _update_rotate_towards_constraint(object c, double dt):
    if not c.is_active:
        return
    t = c.transform
    if t is None:
        return
    target_quat = _compute_weighted_rotation(c)
    if target_quat is None:
        return
    cur_q = t.local_rotation
    cdef double max_angle = c.speed * dt / 180.0
    if max_angle > 1.0:
        max_angle = 1.0
    cdef double rx, ry, rz, rw
    _quat_slerp(cur_q._x, cur_q._y, cur_q._z, cur_q._w,
                target_quat._x, target_quat._y, target_quat._z, target_quat._w,
                max_angle, &rx, &ry, &rz, &rw)
    from core.maths.math3d import Quat
    t.local_rotation = Quat._make(rx, ry, rz, rw)


cdef void _update_scale_to_constraint(object c, double dt):
    if not c.is_active:
        return
    t = c.transform
    if t is None:
        return
    target = _compute_weighted_scale(c)
    if target is None:
        return
    cdef double factor = c.scale_factor
    cdef double lo = c.min_scale
    cdef double hi = c.max_scale
    cdef double nx = _clamp(target._x * factor, lo, hi)
    cdef double ny = _clamp(target._y * factor, lo, hi)
    cdef double nz = _clamp(target._z * factor, lo, hi)
    from core.maths.math3d import Vec3
    t.local_scale = Vec3._make(nx, ny, nz)


cdef void _update_aim_constraint(object c, double dt):
    cdef double inv_eupl = 0.0
    cdef double dx, dy, dz, dl, inv_dl
    cdef double upx, upy, upz, wu, eupx, eupy, eupz, eupl
    cdef double rx, ry, rz, rw
    cdef double ce_x, ce_y, ce_z
    cdef double te_x, te_y, te_z
    cdef double nx, ny, nz, aim_w
    if not c.is_active:
        return
    t = c.transform
    if t is None:
        return
    target_pos = _compute_weighted_position(c)
    if target_pos is None:
        return
    current = t.position
    dx = target_pos._x - current._x
    dy = target_pos._y - current._y
    dz = target_pos._z - current._z
    dl = _vec3_length(dx, dy, dz)
    if dl < 1e-8:
        return
    inv_dl = 1.0 / dl
    dx *= inv_dl; dy *= inv_dl; dz *= inv_dl

    upx = 0.0; upy = 1.0; upz = 0.0
    wu = c.world_up_weight
    eupx = upx * wu + current._x * (1.0 - wu)
    eupy = upy * wu + current._y * (1.0 - wu)
    eupz = upz * wu + current._z * (1.0 - wu)
    eupl = _vec3_length(eupx, eupy, eupz)
    if eupl < 1e-6:
        eupx = upx; eupy = upy; eupz = upz
    else:
        inv_eupl = 1.0 / eupl
        eupx *= inv_eupl; eupy *= inv_eupl; eupz *= inv_eupl

    _look_rotation(dx, dy, dz, eupx, eupy, eupz, &rx, &ry, &rz, &rw)

    cur_q = t.local_rotation
    _quat_to_euler_rad(cur_q._x, cur_q._y, cur_q._z, cur_q._w, &ce_x, &ce_y, &ce_z)
    _quat_to_euler_rad(rx, ry, rz, rw, &te_x, &te_y, &te_z)

    nx = ce_x if not c.local_euler_axis_x else te_x
    ny = ce_y if not c.local_euler_axis_y else te_y
    nz = ce_z if not c.local_euler_axis_z else te_z

    aim_w = c.aim_position_weight
    nx = ce_x + (nx - ce_x) * aim_w
    ny = ce_y + (ny - ce_y) * aim_w
    nz = ce_z + (nz - ce_z) * aim_w

    from core.maths.math3d import Vec3
    t.local_euler_angles = Vec3._make(nx, ny, nz)


cdef void _update_look_at_constraint(object c, double dt):
    cdef double inv_eupl = 0.0
    cdef double dx, dy, dz, dl, inv_dl
    cdef double upx, upy, upz, wu, eupx, eupy, eupz, eupl
    cdef double rx, ry, rz, rw
    cdef double ce_x, ce_y, ce_z
    cdef double te_x, te_y, te_z
    cdef double w, nx, ny, nz
    if not c.is_active:
        return
    t = c.transform
    if t is None:
        return
    target_pos = _compute_weighted_position(c)
    if target_pos is None:
        return
    current = t.position
    dx = target_pos._x - current._x
    dy = target_pos._y - current._y
    dz = target_pos._z - current._z
    dl = _vec3_length(dx, dy, dz)
    if dl < 1e-8:
        return
    inv_dl = 1.0 / dl
    dx *= inv_dl; dy *= inv_dl; dz *= inv_dl

    upx = 0.0; upy = 1.0; upz = 0.0
    wu = c.world_up_weight
    eupx = upx * wu + current._x * (1.0 - wu)
    eupy = upy * wu + current._y * (1.0 - wu)
    eupz = upz * wu + current._z * (1.0 - wu)
    eupl = _vec3_length(eupx, eupy, eupz)
    if eupl < 1e-6:
        eupx = upx; eupy = upy; eupz = upz
    else:
        inv_eupl = 1.0 / eupl
        eupx *= inv_eupl; eupy *= inv_eupl; eupz *= inv_eupl

    _look_rotation(dx, dy, dz, eupx, eupy, eupz, &rx, &ry, &rz, &rw)

    cur_q = t.local_rotation
    _quat_to_euler_rad(cur_q._x, cur_q._y, cur_q._z, cur_q._w, &ce_x, &ce_y, &ce_z)
    _quat_to_euler_rad(rx, ry, rz, rw, &te_x, &te_y, &te_z)

    w = c.look_at_weight
    nx = ce_x + (te_x - ce_x) * w
    ny = ce_y + (te_y - ce_y) * w
    nz = ce_z + (te_z - ce_z) * w

    from core.maths.math3d import Vec3
    t.local_euler_angles = Vec3._make(nx, ny, nz)


cdef void _update_follow_transform_constraint(object c, double dt):
    cdef double dx, dy, dz, dist, move_dist, inv_dist
    cdef double max_angle, frx, fry, frz, frw
    if not c.is_active:
        return
    t = c.transform
    if t is None:
        return
    from core.maths.math3d import Vec3, Quat

    if c.follow_position:
        target_pos = _compute_weighted_position(c)
        if target_pos is not None:
            current = t.position
            dx = target_pos._x - current._x
            dy = target_pos._y - current._y
            dz = target_pos._z - current._z
            dist = _vec3_length(dx, dy, dz)
            if dist > 1e-6:
                move_dist = c.position_speed * dt
                if move_dist > dist:
                    move_dist = dist
                inv_dist = move_dist / dist
                t.position = Vec3._make(current._x + dx * inv_dist,
                                        current._y + dy * inv_dist,
                                        current._z + dz * inv_dist)

    if c.follow_rotation:
        target_quat = _compute_weighted_rotation(c)
        if target_quat is not None:
            cur_q = t.local_rotation
            max_angle = c.rotation_speed * dt / 180.0
            if max_angle > 1.0:
                max_angle = 1.0
            _quat_slerp(cur_q._x, cur_q._y, cur_q._z, cur_q._w,
                        target_quat._x, target_quat._y, target_quat._z, target_quat._w,
                        max_angle, &frx, &fry, &frz, &frw)
            t.local_rotation = Quat._make(frx, fry, frz, frw)


cdef inline void _look_rotation(double fx, double fy, double fz,
                                 double ux, double uy, double uz,
                                 double *rx, double *ry, double *rz, double *rw):
    cdef double rx1 = uy * fz - uz * fy
    cdef double ry1 = uz * fx - ux * fz
    cdef double rz1 = ux * fy - uy * fx
    cdef double rl = _vec3_length(rx1, ry1, rz1)
    cdef double inv_rl = 0.0
    cdef double ux2 = 0.0, uy2 = 0.0, uz2 = 0.0
    cdef double m00 = 0.0, m01 = 0.0, m02 = 0.0
    cdef double m10 = 0.0, m11 = 0.0, m12 = 0.0
    cdef double m20 = 0.0, m21 = 0.0, m22 = 0.0
    cdef double trace = 0.0
    cdef double s = 0.0
    if rl < 1e-6:
        rx[0] = 0.0; ry[0] = 0.0; rz[0] = 0.0; rw[0] = 1.0
        return
    inv_rl = 1.0 / rl
    rx1 *= inv_rl; ry1 *= inv_rl; rz1 *= inv_rl
    ux2 = fy * rz1 - fz * ry1
    uy2 = fz * rx1 - fx * rz1
    uz2 = fx * ry1 - fy * rx1

    m00 = rx1; m01 = ux2; m02 = -fx
    m10 = ry1; m11 = uy2; m12 = -fy
    m20 = rz1; m21 = uz2; m22 = -fz
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / sqrt(trace + 1.0)
        rw[0] = 0.25 / s
        rx[0] = (m21 - m12) * s
        ry[0] = (m02 - m20) * s
        rz[0] = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * sqrt(1.0 + m00 - m11 - m22)
        rw[0] = (m21 - m12) / s
        rx[0] = 0.25 * s
        ry[0] = (m01 + m10) / s
        rz[0] = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * sqrt(1.0 + m11 - m00 - m22)
        rw[0] = (m02 - m20) / s
        rx[0] = (m01 + m10) / s
        ry[0] = 0.25 * s
        rz[0] = (m12 + m21) / s
    else:
        s = 2.0 * sqrt(1.0 + m22 - m00 - m11)
        rw[0] = (m10 - m01) / s
        rx[0] = (m02 + m20) / s
        ry[0] = (m12 + m21) / s
        rz[0] = 0.25 * s
    _quat_normalize_inplace(rx, ry, rz, rw)


cdef void _update_constraint(object c, str tname, double dt):
    if tname == "PositionConstraint":
        _update_position_constraint(c, dt)
    elif tname == "RotationConstraint":
        _update_rotation_constraint(c, dt)
    elif tname == "ScaleConstraint":
        _update_scale_constraint(c, dt)
    elif tname == "ParentConstraint":
        _update_parent_constraint(c, dt)
    elif tname == "MoveTowardsConstraint":
        _update_move_towards_constraint(c, dt)
    elif tname == "RotateTowardsConstraint":
        _update_rotate_towards_constraint(c, dt)
    elif tname == "ScaleToConstraint":
        _update_scale_to_constraint(c, dt)
    elif tname == "AimConstraint":
        _update_aim_constraint(c, dt)
    elif tname == "LookAtConstraint":
        _update_look_at_constraint(c, dt)
    elif tname == "FollowTransformConstraint":
        _update_follow_transform_constraint(c, dt)


def batch_update_constraints(list constraints, double dt):
    cdef Py_ssize_t i, n = len(constraints)
    cdef object c
    cdef str tname
    for i in range(n):
        c = constraints[i]
        tname = type(c).__name__
        _update_constraint(c, tname, dt)
