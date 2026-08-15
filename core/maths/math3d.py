# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import numpy as np

from core._math_vec import Vec2, Vec3, Vec4, Quat

__all__ = ["Vec2", "Vec3", "Vec4", "Quat", "Mat4", "FLOAT_TYPE", "FLT_EPSILON"]

FLOAT_TYPE = np.float64


def _get_mh():
    from core import math_helpers as _mh
    return _mh


class Mat4:
    __slots__ = ("_d",)
    def __init__(self, data: np.ndarray = None):
        if data is not None:
            self._d = np.array(data, dtype=FLOAT_TYPE)
        else:
            self._d = np.eye(4, dtype=FLOAT_TYPE)

    @staticmethod
    def identity() -> Mat4: return Mat4()

    @staticmethod
    def translation(v: Vec3) -> Mat4:
        m = Mat4(); m._d[3,0] = v.x; m._d[3,1] = v.y; m._d[3,2] = v.z; return m

    @staticmethod
    def scale(v: Vec3) -> Mat4:
        m = Mat4(); m._d[0,0] = v.x; m._d[1,1] = v.y; m._d[2,2] = v.z; return m

    @staticmethod
    def perspective(fov_deg: float, aspect: float, near: float, far: float) -> Mat4:
        f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
        m = np.zeros((4,4), dtype=FLOAT_TYPE)
        m[0,0] = f / aspect; m[1,1] = f
        m[2,2] = (far+near)/(near-far); m[2,3] = -1.0
        m[3,2] = (2*far*near)/(near-far)
        return Mat4(m)

    @staticmethod
    def orthographic(left, right, bottom, top, near, far) -> Mat4:
        m = np.zeros((4,4), dtype=FLOAT_TYPE)
        m[0,0] = 2/(right-left); m[3,0] = -(right+left)/(right-left)
        m[1,1] = 2/(top-bottom); m[3,1] = -(top+bottom)/(top-bottom)
        m[2,2] = -2/(far-near);  m[3,2] = -(far+near)/(far-near)
        m[3,3] = 1
        return Mat4(m)

    @staticmethod
    def look_at(eye: Vec3, center: Vec3, up: Vec3) -> Mat4:
        f = (center - eye).normalized()
        r = f.cross(up).normalized()
        u = r.cross(f)
        m = np.eye(4, dtype=FLOAT_TYPE)
        m[0,0]=r.x; m[1,0]=r.y; m[2,0]=r.z
        m[0,1]=u.x; m[1,1]=u.y; m[2,1]=u.z
        m[0,2]=-f.x; m[1,2]=-f.y; m[2,2]=-f.z
        m[3,0]=-r.dot(eye); m[3,1]=-u.dot(eye); m[3,2]=f.dot(eye)
        return Mat4(m)

    def __mul__(self, o): return Mat4(self._d @ o._d)
    def __matmul__(self, o): return self.__mul__(o)
    def transposed(self): return Mat4(self._d.T)
    def inverted(self): return Mat4(np.linalg.inv(self._d))
    def to_array(self): return self._d.copy()
    def to_f32(self):
        return self._d.T.astype(np.float32).flatten(order='F')

    @staticmethod
    def batch_to_f32(matrices: list[Mat4]) -> np.ndarray:
        n = len(matrices)
        if n == 0:
            return np.zeros((0, 16), dtype=np.float32)
        from core.math_helpers import batch_matrices_to_f32
        stacked = np.array([m._d for m in matrices])
        return batch_matrices_to_f32(stacked)
    def to_list(self): return self._d.tolist()
    def get_translation(self): return Vec3(float(self._d[3,0]), float(self._d[3,1]), float(self._d[3,2]))
    def decompose(self) -> tuple[Vec3, Quat, Vec3]:
        pos = Vec3(float(self._d[3,0]), float(self._d[3,1]), float(self._d[3,2]))
        sx = float(np.linalg.norm(self._d[0,:3]))
        sy = float(np.linalg.norm(self._d[1,:3]))
        sz = float(np.linalg.norm(self._d[2,:3]))
        scale = Vec3(sx, sy, sz)
        rm = np.array(self._d[:3,:3], dtype=FLOAT_TYPE)
        if sx > 1e-10: rm[0,:] /= sx
        if sy > 1e-10: rm[1,:] /= sy
        if sz > 1e-10: rm[2,:] /= sz
        rot = Quat._from_rotation_matrix3(rm.T)
        return pos, rot, scale


FLT_EPSILON = 1.1920928955078125e-07


def _get_core_batch():
    try:
        from core import _core_batch as _cb
        return _cb
    except ImportError:
        return None

_CB = None


def _ensure_cb():
    global _CB
    if _CB is None:
        _CB = _get_core_batch()
    return _CB


def _to_f64_array(v):
    if hasattr(v, '__len__') and not isinstance(v, np.ndarray):
        return np.asarray(v, dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def mat4_mul_flat(l, r):
    cb = _ensure_cb()
    if cb is not None:
        return list(cb.mat4_mul_flat(_to_f64_array(l), _to_f64_array(r)))
    out = [0.0] * 16
    out[0] = l[0] * r[0] + l[1] * r[4] + l[2] * r[8] + l[3] * r[12]
    out[1] = l[0] * r[1] + l[1] * r[5] + l[2] * r[9] + l[3] * r[13]
    out[2] = l[0] * r[2] + l[1] * r[6] + l[2] * r[10] + l[3] * r[14]
    out[3] = l[0] * r[3] + l[1] * r[7] + l[2] * r[11] + l[3] * r[15]
    out[4] = l[4] * r[0] + l[5] * r[4] + l[6] * r[8] + l[7] * r[12]
    out[5] = l[4] * r[1] + l[5] * r[5] + l[6] * r[9] + l[7] * r[13]
    out[6] = l[4] * r[2] + l[5] * r[6] + l[6] * r[10] + l[7] * r[14]
    out[7] = l[4] * r[3] + l[5] * r[7] + l[6] * r[11] + l[7] * r[15]
    out[8] = l[8] * r[0] + l[9] * r[4] + l[10] * r[8] + l[11] * r[12]
    out[9] = l[8] * r[1] + l[9] * r[5] + l[10] * r[9] + l[11] * r[13]
    out[10] = l[8] * r[2] + l[9] * r[6] + l[10] * r[10] + l[11] * r[14]
    out[11] = l[8] * r[3] + l[9] * r[7] + l[10] * r[11] + l[11] * r[15]
    out[12] = l[12] * r[0] + l[13] * r[4] + l[14] * r[8] + l[15] * r[12]
    out[13] = l[12] * r[1] + l[13] * r[5] + l[14] * r[9] + l[15] * r[13]
    out[14] = l[12] * r[2] + l[13] * r[6] + l[14] * r[10] + l[15] * r[14]
    out[15] = l[12] * r[3] + l[13] * r[7] + l[14] * r[11] + l[15] * r[15]
    return out


def mat4_mul_vec_flat(m, x, y, z, w=1.0):
    cb = _ensure_cb()
    if cb is not None:
        return cb.mat4_mul_vec_flat(_to_f64_array(m), x, y, z, w)
    return (
        m[0] * x + m[4] * y + m[8] * z + m[12] * w,
        m[1] * x + m[5] * y + m[9] * z + m[13] * w,
        m[2] * x + m[6] * y + m[10] * z + m[14] * w,
        m[3] * x + m[7] * y + m[11] * z + m[15] * w,
    )


def dot3_flat(ax, ay, az, bx, by, bz):
    cb = _ensure_cb()
    if cb is not None:
        return cb.dot3_flat(ax, ay, az, bx, by, bz)
    return ax * bx + ay * by + az * bz


def cross3_flat(ax, ay, az, bx, by, bz):
    cb = _ensure_cb()
    if cb is not None:
        return cb.cross3_flat(ax, ay, az, bx, by, bz)
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    )


def normalize3_flat(x, y, z):
    cb = _ensure_cb()
    if cb is not None:
        return cb.normalize3_flat(x, y, z)
    l = math.sqrt(x * x + y * y + z * z)
    if l > 1e-10:
        inv = 1.0 / l
        return (x * inv, y * inv, z * inv)
    return (0.0, 0.0, 0.0)


def rotate_around_axis_flat(vx, vy, vz, ax, ay, az, angle):
    cb = _ensure_cb()
    if cb is not None:
        return cb.rotate_around_axis_flat(vx, vy, vz, ax, ay, az, angle)
    n = normalize3_flat(ax, ay, az)
    c = math.cos(angle)
    s = math.sin(angle)
    d = dot3_flat(n[0], n[1], n[2], vx, vy, vz)
    return (
        vx * c + (n[1] * vz - n[2] * vy) * s + n[0] * d * (1.0 - c),
        vy * c + (n[2] * vx - n[0] * vz) * s + n[1] * d * (1.0 - c),
        vz * c + (n[0] * vy - n[1] * vx) * s + n[2] * d * (1.0 - c),
    )


def axis_vector_flat(axis: int, sign: float = 1.0):
    cb = _ensure_cb()
    if cb is not None:
        return cb.axis_vector_flat(axis, sign)
    if axis == 0:
        return (sign, 0.0, 0.0)
    if axis == 1:
        return (0.0, sign, 0.0)
    return (0.0, 0.0, sign)


def coordinate_system_axes(cs: int):
    cb = _ensure_cb()
    if cb is not None:
        return cb.coordinate_system_axes(cs)
    if cs == 1:
        return (1, 2, 0)
    if cs == 2:
        return (2, 0, 1)
    if cs == 3:
        return (0, 2, 1)
    if cs == 4:
        return (1, 0, 2)
    if cs == 5:
        return (2, 1, 0)
    return (0, 1, 2)


def is_right_handed_cs(cs: int) -> bool:
    cb = _ensure_cb()
    if cb is not None:
        return cb.is_right_handed_cs(cs)
    return cs in (0, 1, 2)


def look_at_flat(eye, at, up, cs: int):
    cb = _ensure_cb()
    if cb is not None:
        return list(cb.look_at_flat(
            eye[0], eye[1], eye[2], at[0], at[1], at[2],
            up[0], up[1], up[2], int(cs),
        ))
    right_handed = is_right_handed_cs(cs)
    fx, fy, fz = normalize3_flat(at[0] - eye[0], at[1] - eye[1], at[2] - eye[2])
    if right_handed:
        rx, ry, rz = cross3_flat(fx, fy, fz, up[0], up[1], up[2])
    else:
        rx, ry, rz = cross3_flat(up[0], up[1], up[2], fx, fy, fz)
    rx, ry, rz = normalize3_flat(rx, ry, rz)
    if right_handed:
        ux, uy, uz = cross3_flat(rx, ry, rz, fx, fy, fz)
    else:
        ux, uy, uz = cross3_flat(fx, fy, fz, rx, ry, rz)
    z_sign = -1.0 if right_handed else 1.0
    m = [0.0] * 16
    m[0] = rx; m[1] = ux; m[2] = z_sign * fx; m[3] = 0.0
    m[4] = ry; m[5] = uy; m[6] = z_sign * fy; m[7] = 0.0
    m[8] = rz; m[9] = uz; m[10] = z_sign * fz; m[11] = 0.0
    m[12] = -dot3_flat(rx, ry, rz, eye[0], eye[1], eye[2])
    m[13] = -dot3_flat(ux, uy, uz, eye[0], eye[1], eye[2])
    m[14] = -z_sign * dot3_flat(fx, fy, fz, eye[0], eye[1], eye[2])
    m[15] = 1.0
    return m


def point_in_circle(cx, cy, radius, px, py) -> bool:
    cb = _ensure_cb()
    if cb is not None:
        return cb.point_in_circle(cx, cy, radius, px, py)
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy <= radius * radius


def mat4_invert_flat(m):
    cb = _ensure_cb()
    if cb is not None:
        return list(cb.mat4_invert_flat(_to_f64_array(m)))
    out = [0.0] * 16
    out[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10]
    out[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10]
    out[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9]
    out[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9]
    out[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10]
    out[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10]
    out[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9]
    out[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9]
    out[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6]
    out[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6]
    out[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5]
    out[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5]
    out[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6]
    out[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6]
    out[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5]
    out[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5]
    det = m[0]*out[0] + m[1]*out[4] + m[2]*out[8] + m[3]*out[12]
    det = 1.0 / det
    return [v * det for v in out]
