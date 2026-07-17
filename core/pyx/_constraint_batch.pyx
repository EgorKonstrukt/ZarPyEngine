# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, acos, sqrt

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef inline void _q_normalize(DTYPE_t x, DTYPE_t y, DTYPE_t z, DTYPE_t w,
                               DTYPE_t* rx, DTYPE_t* ry, DTYPE_t* rz, DTYPE_t* rw) noexcept nogil:
    cdef DTYPE_t len_sq = x*x + y*y + z*z + w*w
    cdef DTYPE_t inv_len
    if len_sq < 1e-30:
        rx[0] = 0.0; ry[0] = 0.0; rz[0] = 0.0; rw[0] = 1.0
        return
    inv_len = 1.0 / sqrt(len_sq)
    rx[0] = x * inv_len; ry[0] = y * inv_len
    rz[0] = z * inv_len; rw[0] = w * inv_len

cdef inline void _q_slerp(
    DTYPE_t ax, DTYPE_t ay, DTYPE_t az, DTYPE_t aw,
    DTYPE_t bx, DTYPE_t by, DTYPE_t bz, DTYPE_t bw,
    DTYPE_t t,
    DTYPE_t* rx, DTYPE_t* ry, DTYPE_t* rz, DTYPE_t* rw,
) noexcept nogil:
    cdef DTYPE_t dot = ax*bx + ay*by + az*bz + aw*bw
    cdef DTYPE_t sign = 1.0
    cdef DTYPE_t angle, sin_angle, wa, wb, len_val
    if dot < 0.0:
        dot = -dot
        sign = -1.0
    if dot > 0.9995:
        wa = 1.0 - t
        wb = t * sign
        rx[0] = wa*ax + wb*bx
        ry[0] = wa*ay + wb*by
        rz[0] = wa*az + wb*bz
        rw[0] = wa*aw + wb*bw
        len_val = sqrt(rx[0]*rx[0] + ry[0]*ry[0] + rz[0]*rz[0] + rw[0]*rw[0])
        if len_val > 1e-30:
            rx[0] /= len_val; ry[0] /= len_val; rz[0] /= len_val; rw[0] /= len_val
        return
    angle = acos(dot if dot <= 1.0 else 1.0)
    sin_angle = sin(angle)
    if sin_angle < 1e-10:
        wa = 1.0 - t
        wb = t * sign
        rx[0] = wa*ax + wb*bx
        ry[0] = wa*ay + wb*by
        rz[0] = wa*az + wb*bz
        rw[0] = wa*aw + wb*bw
        return
    wa = sin((1.0 - t) * angle) / sin_angle
    wb = sin(t * angle) / sin_angle * sign
    rx[0] = wa*ax + wb*bx
    ry[0] = wa*ay + wb*by
    rz[0] = wa*az + wb*bz
    rw[0] = wa*aw + wb*bw


def batch_weighted_positions(
    np.ndarray[DTYPE_t, ndim=1] cur_x,
    np.ndarray[DTYPE_t, ndim=1] cur_y,
    np.ndarray[DTYPE_t, ndim=1] cur_z,
    np.ndarray[DTYPE_t, ndim=1] target_x,
    np.ndarray[DTYPE_t, ndim=1] target_y,
    np.ndarray[DTYPE_t, ndim=1] target_z,
    np.ndarray[DTYPE_t, ndim=1] offset_x,
    np.ndarray[DTYPE_t, ndim=1] offset_y,
    np.ndarray[DTYPE_t, ndim=1] offset_z,
    np.ndarray[DTYPE_t, ndim=1] weight_x,
    np.ndarray[DTYPE_t, ndim=1] weight_y,
    np.ndarray[DTYPE_t, ndim=1] weight_z,
    np.ndarray[np.int8_t, ndim=1] has_target,
):
    cdef int n = cur_x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out_x = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_y = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_z = np.empty(n, dtype=DTYPE)
    cdef int i
    cdef DTYPE_t tx, ty, tz
    for i in range(n):
        if has_target[i]:
            tx = target_x[i] + offset_x[i]
            ty = target_y[i] + offset_y[i]
            tz = target_z[i] + offset_z[i]
            out_x[i] = cur_x[i] + (tx - cur_x[i]) * weight_x[i]
            out_y[i] = cur_y[i] + (ty - cur_y[i]) * weight_y[i]
            out_z[i] = cur_z[i] + (tz - cur_z[i]) * weight_z[i]
        else:
            out_x[i] = cur_x[i]
            out_y[i] = cur_y[i]
            out_z[i] = cur_z[i]
    return out_x, out_y, out_z

def batch_weighted_scales(
    np.ndarray[DTYPE_t, ndim=1] cur_x,
    np.ndarray[DTYPE_t, ndim=1] cur_y,
    np.ndarray[DTYPE_t, ndim=1] cur_z,
    np.ndarray[DTYPE_t, ndim=1] target_x,
    np.ndarray[DTYPE_t, ndim=1] target_y,
    np.ndarray[DTYPE_t, ndim=1] target_z,
    np.ndarray[DTYPE_t, ndim=1] offset_x,
    np.ndarray[DTYPE_t, ndim=1] offset_y,
    np.ndarray[DTYPE_t, ndim=1] offset_z,
    np.ndarray[DTYPE_t, ndim=1] weight_x,
    np.ndarray[DTYPE_t, ndim=1] weight_y,
    np.ndarray[DTYPE_t, ndim=1] weight_z,
    np.ndarray[np.int8_t, ndim=1] has_target,
):
    cdef int n = cur_x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out_x = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_y = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_z = np.empty(n, dtype=DTYPE)
    cdef int i
    cdef DTYPE_t tx, ty, tz
    for i in range(n):
        if has_target[i]:
            tx = target_x[i] * offset_x[i]
            ty = target_y[i] * offset_y[i]
            tz = target_z[i] * offset_z[i]
            out_x[i] = cur_x[i] + (tx - cur_x[i]) * weight_x[i]
            out_y[i] = cur_y[i] + (ty - cur_y[i]) * weight_y[i]
            out_z[i] = cur_z[i] + (tz - cur_z[i]) * weight_z[i]
        else:
            out_x[i] = cur_x[i]
            out_y[i] = cur_y[i]
            out_z[i] = cur_z[i]
    return out_x, out_y, out_z

def batch_weighted_euler_angles(
    np.ndarray[DTYPE_t, ndim=1] cur_x,
    np.ndarray[DTYPE_t, ndim=1] cur_y,
    np.ndarray[DTYPE_t, ndim=1] cur_z,
    np.ndarray[DTYPE_t, ndim=1] target_x,
    np.ndarray[DTYPE_t, ndim=1] target_y,
    np.ndarray[DTYPE_t, ndim=1] target_z,
    np.ndarray[DTYPE_t, ndim=1] weight_x,
    np.ndarray[DTYPE_t, ndim=1] weight_y,
    np.ndarray[DTYPE_t, ndim=1] weight_z,
    np.ndarray[np.int8_t, ndim=1] has_target,
):
    cdef int n = cur_x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out_x = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_y = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_z = np.empty(n, dtype=DTYPE)
    cdef int i
    for i in range(n):
        if has_target[i]:
            out_x[i] = cur_x[i] + (target_x[i] - cur_x[i]) * weight_x[i]
            out_y[i] = cur_y[i] + (target_y[i] - cur_y[i]) * weight_y[i]
            out_z[i] = cur_z[i] + (target_z[i] - cur_z[i]) * weight_z[i]
        else:
            out_x[i] = cur_x[i]
            out_y[i] = cur_y[i]
            out_z[i] = cur_z[i]
    return out_x, out_y, out_z

def batch_follow_positions(
    np.ndarray[DTYPE_t, ndim=1] cur_x,
    np.ndarray[DTYPE_t, ndim=1] cur_y,
    np.ndarray[DTYPE_t, ndim=1] cur_z,
    np.ndarray[DTYPE_t, ndim=1] target_x,
    np.ndarray[DTYPE_t, ndim=1] target_y,
    np.ndarray[DTYPE_t, ndim=1] target_z,
    np.ndarray[DTYPE_t, ndim=1] speeds,
    np.ndarray[DTYPE_t, ndim=1] dts,
):
    cdef int n = cur_x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out_x = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_y = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_z = np.empty(n, dtype=DTYPE)
    cdef int i
    cdef DTYPE_t dx, dy, dz, dist, move_dist, inv_dist
    for i in range(n):
        dx = target_x[i] - cur_x[i]
        dy = target_y[i] - cur_y[i]
        dz = target_z[i] - cur_z[i]
        dist = (dx*dx + dy*dy + dz*dz)**0.5
        if dist > 1e-6:
            move_dist = speeds[i] * dts[i]
            if move_dist > dist:
                move_dist = dist
            inv_dist = move_dist / dist
            out_x[i] = cur_x[i] + dx * inv_dist
            out_y[i] = cur_y[i] + dy * inv_dist
            out_z[i] = cur_z[i] + dz * inv_dist
        else:
            out_x[i] = cur_x[i]
            out_y[i] = cur_y[i]
            out_z[i] = cur_z[i]
    return out_x, out_y, out_z

def batch_follow_rotations(
    np.ndarray[DTYPE_t, ndim=1] cur_rx,
    np.ndarray[DTYPE_t, ndim=1] cur_ry,
    np.ndarray[DTYPE_t, ndim=1] cur_rz,
    np.ndarray[DTYPE_t, ndim=1] cur_rw,
    np.ndarray[DTYPE_t, ndim=1] target_rx,
    np.ndarray[DTYPE_t, ndim=1] target_ry,
    np.ndarray[DTYPE_t, ndim=1] target_rz,
    np.ndarray[DTYPE_t, ndim=1] target_rw,
    np.ndarray[DTYPE_t, ndim=1] speeds,
    np.ndarray[DTYPE_t, ndim=1] dts,
):
    cdef int n = cur_rx.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out_rx = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_ry = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_rz = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_rw = np.empty(n, dtype=DTYPE)
    cdef int i
    cdef DTYPE_t max_angle, blend, rx, ry, rz, rw
    for i in range(n):
        max_angle = speeds[i] * dts[i] / 180.0
        if max_angle > 1.0:
            max_angle = 1.0
        _q_slerp(
            cur_rx[i], cur_ry[i], cur_rz[i], cur_rw[i],
            target_rx[i], target_ry[i], target_rz[i], target_rw[i],
            max_angle,
            &rx, &ry, &rz, &rw,
        )
        out_rx[i] = rx; out_ry[i] = ry; out_rz[i] = rz; out_rw[i] = rw
    return out_rx, out_ry, out_rz, out_rw

def batch_rotate_towards(
    np.ndarray[DTYPE_t, ndim=1] cur_rx,
    np.ndarray[DTYPE_t, ndim=1] cur_ry,
    np.ndarray[DTYPE_t, ndim=1] cur_rz,
    np.ndarray[DTYPE_t, ndim=1] cur_rw,
    np.ndarray[DTYPE_t, ndim=1] target_rx,
    np.ndarray[DTYPE_t, ndim=1] target_ry,
    np.ndarray[DTYPE_t, ndim=1] target_rz,
    np.ndarray[DTYPE_t, ndim=1] target_rw,
    np.ndarray[DTYPE_t, ndim=1] speeds,
    np.ndarray[DTYPE_t, ndim=1] dts,
):
    cdef int n = cur_rx.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out_rx = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_ry = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_rz = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_rw = np.empty(n, dtype=DTYPE)
    cdef int i
    cdef DTYPE_t max_angle, blend, rx, ry, rz, rw
    for i in range(n):
        max_angle = speeds[i] * dts[i] / 180.0
        if max_angle > 1.0:
            max_angle = 1.0
        _q_slerp(
            cur_rx[i], cur_ry[i], cur_rz[i], cur_rw[i],
            target_rx[i], target_ry[i], target_rz[i], target_rw[i],
            max_angle,
            &rx, &ry, &rz, &rw,
        )
        out_rx[i] = rx; out_ry[i] = ry; out_rz[i] = rz; out_rw[i] = rw
    return out_rx, out_ry, out_rz, out_rw

def batch_scale_to(
    np.ndarray[DTYPE_t, ndim=1] target_x,
    np.ndarray[DTYPE_t, ndim=1] target_y,
    np.ndarray[DTYPE_t, ndim=1] target_z,
    np.ndarray[DTYPE_t, ndim=1] factors,
    np.ndarray[DTYPE_t, ndim=1] min_scales,
    np.ndarray[DTYPE_t, ndim=1] max_scales,
):
    cdef int n = target_x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out_x = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_y = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out_z = np.empty(n, dtype=DTYPE)
    cdef int i
    cdef DTYPE_t v
    for i in range(n):
        v = target_x[i] * factors[i]
        out_x[i] = min_scales[i] if v < min_scales[i] else (max_scales[i] if v > max_scales[i] else v)
        v = target_y[i] * factors[i]
        out_y[i] = min_scales[i] if v < min_scales[i] else (max_scales[i] if v > max_scales[i] else v)
        v = target_z[i] * factors[i]
        out_z[i] = min_scales[i] if v < min_scales[i] else (max_scales[i] if v > max_scales[i] else v)
    return out_x, out_y, out_z
