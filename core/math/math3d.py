# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import numpy as np
from typing import Union

FLOAT_TYPE = np.float64


def _get_mh():
    from core import math_helpers as _mh
    return _mh


class Vec2:
    __slots__ = ("_x", "_y")
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self._x = float(x)
        self._y = float(y)

    @classmethod
    def from_array(cls, a) -> Vec2:
        return cls(float(a[0]), float(a[1]))

    @staticmethod
    def _make(x: float, y: float) -> Vec2:
        v = object.__new__(Vec2)
        v._x = x; v._y = y
        return v

    @property
    def _d(self):
        return (self._x, self._y)

    @property
    def x(self) -> float: return self._x
    @x.setter
    def x(self, v: float): self._x = float(v)
    @property
    def y(self) -> float: return self._y
    @y.setter
    def y(self, v: float): self._y = float(v)

    def __add__(self, o): return Vec2._make(self._x + o._x, self._y + o._y)
    def __sub__(self, o): return Vec2._make(self._x - o._x, self._y - o._y)
    def __mul__(self, s): return Vec2._make(self._x * s, self._y * s)
    def __rmul__(self, s): return self.__mul__(s)
    def __truediv__(self, s): return Vec2._make(self._x / s, self._y / s)
    def __neg__(self): return Vec2._make(-self._x, -self._y)
    def __repr__(self): return f"Vec2({self._x:.4f}, {self._y:.4f})"
    def __eq__(self, o):
        return isinstance(o, Vec2) and abs(self._x-o._x) < 1e-8 and abs(self._y-o._y) < 1e-8
    def __hash__(self): return hash((self._x, self._y))

    def dot(self, o): return self._x * o._x + self._y * o._y
    def length(self): return math.sqrt(self._x*self._x + self._y*self._y)
    def normalized(self):
        l = math.sqrt(self._x*self._x + self._y*self._y)
        return Vec2._make(self._x/l, self._y/l) if l > 1e-10 else Vec2()
    def to_list(self): return [self._x, self._y]

    @staticmethod
    def zero(): v = object.__new__(Vec2); v._x = 0.0; v._y = 0.0; return v
    @staticmethod
    def one(): v = object.__new__(Vec2); v._x = 1.0; v._y = 1.0; return v


class Vec3:
    __slots__ = ("_x", "_y", "_z")
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)

    @classmethod
    def from_array(cls, a) -> Vec3:
        return cls(float(a[0]), float(a[1]), float(a[2]))

    @staticmethod
    def _make(x: float, y: float, z: float) -> Vec3:
        v = object.__new__(Vec3)
        v._x = x; v._y = y; v._z = z
        return v

    @property
    def _d(self):
        return (self._x, self._y, self._z)

    @property
    def x(self) -> float: return self._x
    @x.setter
    def x(self, v: float): self._x = float(v)
    @property
    def y(self) -> float: return self._y
    @y.setter
    def y(self, v: float): self._y = float(v)
    @property
    def z(self) -> float: return self._z
    @z.setter
    def z(self, v: float): self._z = float(v)

    def __getitem__(self, i: int) -> float:
        if i == 0: return self._x
        if i == 1: return self._y
        if i == 2: return self._z
        raise IndexError(i)

    def __iter__(self):
        yield self._x; yield self._y; yield self._z

    def __add__(self, o): return Vec3._make(self._x + o._x, self._y + o._y, self._z + o._z)
    def __sub__(self, o): return Vec3._make(self._x - o._x, self._y - o._y, self._z - o._z)
    def __mul__(self, s): return Vec3._make(self._x * s, self._y * s, self._z * s)
    def __rmul__(self, s): return self.__mul__(s)
    def __truediv__(self, s): return Vec3._make(self._x / s, self._y / s, self._z / s)
    def __neg__(self): return Vec3._make(-self._x, -self._y, -self._z)
    def __repr__(self): return f"Vec3({self._x:.4f}, {self._y:.4f}, {self._z:.4f})"
    def __eq__(self, o):
        if not isinstance(o, Vec3): return False
        return abs(self._x-o._x) < 1e-8 and abs(self._y-o._y) < 1e-8 and abs(self._z-o._z) < 1e-8
    def __hash__(self): return hash((self._x, self._y, self._z))

    def dot(self, o): return self._x*o._x + self._y*o._y + self._z*o._z
    def cross(self, o):
        return Vec3._make(
            self._y*o._z - self._z*o._y,
            self._z*o._x - self._x*o._z,
            self._x*o._y - self._y*o._x
        )
    def length(self): return math.sqrt(self._x*self._x + self._y*self._y + self._z*self._z)
    def length_sq(self): return self._x*self._x + self._y*self._y + self._z*self._z
    def normalized(self):
        l = math.sqrt(self._x*self._x + self._y*self._y + self._z*self._z)
        return Vec3._make(self._x/l, self._y/l, self._z/l) if l > 1e-10 else Vec3()
    def lerp(self, o, t):
        return Vec3._make(
            self._x + (o._x - self._x)*t,
            self._y + (o._y - self._y)*t,
            self._z + (o._z - self._z)*t
        )
    def to_array(self): return np.array([self._x, self._y, self._z], dtype=FLOAT_TYPE)
    def to_list(self): return [self._x, self._y, self._z]

    @staticmethod
    def zero(): v = object.__new__(Vec3); v._x = 0.0; v._y = 0.0; v._z = 0.0; return v
    @staticmethod
    def one(): v = object.__new__(Vec3); v._x = 1.0; v._y = 1.0; v._z = 1.0; return v
    @staticmethod
    def up(): v = object.__new__(Vec3); v._x = 0.0; v._y = 1.0; v._z = 0.0; return v
    @staticmethod
    def forward(): v = object.__new__(Vec3); v._x = 0.0; v._y = 0.0; v._z = -1.0; return v
    @staticmethod
    def right(): v = object.__new__(Vec3); v._x = 1.0; v._y = 0.0; v._z = 0.0; return v


class Vec4:
    __slots__ = ("_x", "_y", "_z", "_w")
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self._x = float(x); self._y = float(y); self._z = float(z); self._w = float(w)

    @classmethod
    def from_array(cls, a) -> Vec4:
        return cls(float(a[0]), float(a[1]), float(a[2]), float(a[3]))

    @staticmethod
    def _make(x: float, y: float, z: float, w: float) -> Vec4:
        v = object.__new__(Vec4)
        v._x = x; v._y = y; v._z = z; v._w = w
        return v

    @property
    def _d(self): return (self._x, self._y, self._z, self._w)

    @property
    def x(self): return self._x
    @property
    def y(self): return self._y
    @property
    def z(self): return self._z
    @property
    def w(self): return self._w
    def to_list(self): return [self._x, self._y, self._z, self._w]


class Quat:
    __slots__ = ("_x", "_y", "_z", "_w")
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self._x = float(x); self._y = float(y); self._z = float(z); self._w = float(w)

    @classmethod
    def from_array(cls, a) -> Quat:
        return cls(float(a[0]), float(a[1]), float(a[2]), float(a[3]))

    @staticmethod
    def _make(x: float, y: float, z: float, w: float) -> Quat:
        q = object.__new__(Quat)
        q._x = x; q._y = y; q._z = z; q._w = w
        return q

    @property
    def _d(self): return (self._x, self._y, self._z, self._w)

    @classmethod
    def identity(cls): return cls(0, 0, 0, 1)

    @classmethod
    def from_euler(cls, x, y, z):
        x, y, z, w = _get_mh().quat_from_euler(x, y, z)
        return Quat._make(x, y, z, w)

    @classmethod
    def from_axis_angle(cls, axis: Vec3, angle_deg: float):
        ax = axis.normalized()
        a = math.radians(angle_deg) * 0.5
        s = math.sin(a)
        return Quat._make(ax._x * s, ax._y * s, ax._z * s, math.cos(a))

    @classmethod
    def look_rotation(cls, forward: Vec3, up: Vec3 = None) -> Quat:
        if up is None: up = Vec3.up()
        f = forward.normalized()
        r = up.cross(f).normalized()
        u = f.cross(r)
        m = np.eye(3, dtype=FLOAT_TYPE)
        m[0] = np.array([r._x, r._y, r._z])
        m[1] = np.array([u._x, u._y, u._z])
        m[2] = np.array([f._x, f._y, f._z])
        return cls._from_rotation_matrix3(m)

    @classmethod
    def _from_rotation_matrix3(cls, m: np.ndarray) -> Quat:
        t = m[0,0] + m[1,1] + m[2,2]
        if t > 0:
            s = 0.5 / math.sqrt(t + 1.0)
            return Quat._make((m[2,1]-m[1,2])*s, (m[0,2]-m[2,0])*s, (m[1,0]-m[0,1])*s, 0.25/s)
        elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
            s = 2.0*math.sqrt(1.0+m[0,0]-m[1,1]-m[2,2])
            return Quat._make(0.25*s, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s, (m[2,1]-m[1,2])/s)
        elif m[1,1] > m[2,2]:
            s = 2.0*math.sqrt(1.0+m[1,1]-m[0,0]-m[2,2])
            return Quat._make((m[0,1]+m[1,0])/s, 0.25*s, (m[1,2]+m[2,1])/s, (m[0,2]-m[2,0])/s)
        else:
            s = 2.0*math.sqrt(1.0+m[2,2]-m[0,0]-m[1,1])
            return Quat._make((m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, 0.25*s, (m[1,0]-m[0,1])/s)

    @property
    def x(self): return self._x
    @property
    def y(self): return self._y
    @property
    def z(self): return self._z
    @property
    def w(self): return self._w

    def __mul__(self, o):
        _mh = _get_mh()
        x, y, z, w = _mh.quat_mul(self._x, self._y, self._z, self._w, o._x, o._y, o._z, o._w)
        return Quat._make(x, y, z, w)

    def conjugate(self):
        x, y, z, w = _get_mh().quat_conjugate(self._x, self._y, self._z, self._w)
        return Quat._make(x, y, z, w)

    def normalized(self):
        x, y, z, w = _get_mh().quat_normalize(self._x, self._y, self._z, self._w)
        return Quat._make(x, y, z, w)

    def rotate_vec3(self, v: Vec3) -> Vec3:
        x, y, z = _get_mh().quat_rotate_vec3(self._x, self._y, self._z, self._w, v._x, v._y, v._z)
        return Vec3(x, y, z)

    def to_euler(self) -> Vec3:
        x, y, z = _get_mh().quat_to_euler(self._x, self._y, self._z, self._w)
        return Vec3(x, y, z)

    def to_matrix4(self) -> Mat4:
        n = self.normalized()
        x, y, z, w = n._x, n._y, n._z, n._w
        m = np.eye(4, dtype=FLOAT_TYPE)
        m[0,0]=1-2*y*y-2*z*z; m[0,1]=2*x*y+2*w*z; m[0,2]=2*x*z-2*w*y
        m[1,0]=2*x*y-2*w*z;   m[1,1]=1-2*x*x-2*z*z; m[1,2]=2*y*z+2*w*x
        m[2,0]=2*x*z+2*w*y;   m[2,1]=2*y*z-2*w*x;   m[2,2]=1-2*x*x-2*y*y
        return Mat4(m)

    def slerp(self, o, t):
        x, y, z, w = _get_mh().quat_slerp(self._x, self._y, self._z, self._w, o._x, o._y, o._z, o._w, t)
        n = math.sqrt(x*x + y*y + z*z + w*w)
        if n > 1e-10:
            inv = 1.0 / n
            return Quat._make(x*inv, y*inv, z*inv, w*inv)
        return Quat.identity()

    def to_list(self): return [self._x, self._y, self._z, self._w]


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
        sx = float(np.linalg.norm(self._d[:3,0]))
        sy = float(np.linalg.norm(self._d[:3,1]))
        sz = float(np.linalg.norm(self._d[:3,2]))
        scale = Vec3(sx, sy, sz)
        rm = np.array(self._d[:3,:3], dtype=FLOAT_TYPE)
        if sx > 1e-10: rm[:,0] /= sx
        if sy > 1e-10: rm[:,1] /= sy
        if sz > 1e-10: rm[:,2] /= sz
        rot = Quat._from_rotation_matrix3(rm.T)
        return pos, rot, scale
