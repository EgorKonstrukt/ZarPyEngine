# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import math
import numpy as np


def _get_mh():
    from core import math_helpers as _mh
    return _mh


cdef class Vec2:
    def __init__(self, double x=0.0, double y=0.0):
        self._x = x
        self._y = y

    @classmethod
    def from_array(cls, a):
        return cls(float(a[0]), float(a[1]))

    @staticmethod
    def _make(x, y):
        cdef Vec2 v = Vec2.__new__(Vec2)
        v._x = x
        v._y = y
        return v

    @property
    def _d(self):
        return (self._x, self._y)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        self._x = float(v)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, v):
        self._y = float(v)

    def __add__(self, o):
        return Vec2._make(self._x + o._x, self._y + o._y)

    def __sub__(self, o):
        return Vec2._make(self._x - o._x, self._y - o._y)

    def __mul__(self, s):
        return Vec2._make(self._x * s, self._y * s)

    def __rmul__(self, s):
        return self.__mul__(s)

    def __truediv__(self, s):
        return Vec2._make(self._x / s, self._y / s)

    def __neg__(self):
        return Vec2._make(-self._x, -self._y)

    def __repr__(self):
        return f"Vec2({self._x:.4f}, {self._y:.4f})"

    def __eq__(self, o):
        return isinstance(o, Vec2) and abs(self._x - o._x) < 1e-8 and abs(self._y - o._y) < 1e-8

    def __hash__(self):
        return hash((self._x, self._y))

    def dot(self, o):
        return self._x * o._x + self._y * o._y

    def length(self):
        return math.sqrt(self._x * self._x + self._y * self._y)

    def normalized(self):
        l = math.sqrt(self._x * self._x + self._y * self._y)
        return Vec2._make(self._x / l, self._y / l) if l > 1e-10 else Vec2()

    def to_list(self):
        return [self._x, self._y]

    @staticmethod
    def zero():
        cdef Vec2 v = Vec2.__new__(Vec2)
        v._x = 0.0
        v._y = 0.0
        return v

    @staticmethod
    def one():
        cdef Vec2 v = Vec2.__new__(Vec2)
        v._x = 1.0
        v._y = 1.0
        return v

    def __reduce__(self):
        return (type(self), (self._x, self._y))


cdef class Vec3:
    def __init__(self, double x=0.0, double y=0.0, double z=0.0):
        self._x = x
        self._y = y
        self._z = z

    @classmethod
    def from_array(cls, a):
        return cls(float(a[0]), float(a[1]), float(a[2]))

    @staticmethod
    def _make(x, y, z):
        cdef Vec3 v = Vec3.__new__(Vec3)
        v._x = x
        v._y = y
        v._z = z
        return v

    @property
    def _d(self):
        return (self._x, self._y, self._z)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        self._x = float(v)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, v):
        self._y = float(v)

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, v):
        self._z = float(v)

    def __getitem__(self, int i):
        if i == 0:
            return self._x
        if i == 1:
            return self._y
        if i == 2:
            return self._z
        raise IndexError(i)

    def __iter__(self):
        yield self._x
        yield self._y
        yield self._z

    def __add__(self, o):
        return Vec3._make(self._x + o._x, self._y + o._y, self._z + o._z)

    def __sub__(self, o):
        return Vec3._make(self._x - o._x, self._y - o._y, self._z - o._z)

    def __mul__(self, s):
        return Vec3._make(self._x * s, self._y * s, self._z * s)

    def __rmul__(self, s):
        return self.__mul__(s)

    def __truediv__(self, s):
        return Vec3._make(self._x / s, self._y / s, self._z / s)

    def __neg__(self):
        return Vec3._make(-self._x, -self._y, -self._z)

    def __repr__(self):
        return f"Vec3({self._x:.4f}, {self._y:.4f}, {self._z:.4f})"

    def __eq__(self, o):
        if not isinstance(o, Vec3):
            return False
        return abs(self._x - o._x) < 1e-8 and abs(self._y - o._y) < 1e-8 and abs(self._z - o._z) < 1e-8

    def __hash__(self):
        return hash((self._x, self._y, self._z))

    def dot(self, o):
        return self._x * o._x + self._y * o._y + self._z * o._z

    def cross(self, o):
        return Vec3._make(
            self._y * o._z - self._z * o._y,
            self._z * o._x - self._x * o._z,
            self._x * o._y - self._y * o._x,
        )

    def length(self):
        return math.sqrt(self._x * self._x + self._y * self._y + self._z * self._z)

    def length_sq(self):
        return self._x * self._x + self._y * self._y + self._z * self._z

    def distance_to(self, o):
        cdef double dx = self._x - o._x
        cdef double dy = self._y - o._y
        cdef double dz = self._z - o._z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def distance_sq_to(self, o):
        cdef double dx = self._x - o._x
        cdef double dy = self._y - o._y
        cdef double dz = self._z - o._z
        return dx * dx + dy * dy + dz * dz

    def normalized(self):
        l = math.sqrt(self._x * self._x + self._y * self._y + self._z * self._z)
        return Vec3._make(self._x / l, self._y / l, self._z / l) if l > 1e-10 else Vec3()

    def lerp(self, o, t):
        return Vec3._make(
            self._x + (o._x - self._x) * t,
            self._y + (o._y - self._y) * t,
            self._z + (o._z - self._z) * t,
        )

    def to_array(self):
        return np.array([self._x, self._y, self._z], dtype=np.float64)

    def to_list(self):
        return [self._x, self._y, self._z]

    @staticmethod
    def zero():
        cdef Vec3 v = Vec3.__new__(Vec3)
        v._x = 0.0
        v._y = 0.0
        v._z = 0.0
        return v

    @staticmethod
    def one():
        cdef Vec3 v = Vec3.__new__(Vec3)
        v._x = 1.0
        v._y = 1.0
        v._z = 1.0
        return v

    @staticmethod
    def up():
        cdef Vec3 v = Vec3.__new__(Vec3)
        v._x = 0.0
        v._y = 1.0
        v._z = 0.0
        return v

    @staticmethod
    def forward():
        cdef Vec3 v = Vec3.__new__(Vec3)
        v._x = 0.0
        v._y = 0.0
        v._z = -1.0
        return v

    @staticmethod
    def right():
        cdef Vec3 v = Vec3.__new__(Vec3)
        v._x = 1.0
        v._y = 0.0
        v._z = 0.0
        return v

    def __reduce__(self):
        return (type(self), (self._x, self._y, self._z))


cdef class Vec4:
    def __init__(self, double x=0.0, double y=0.0, double z=0.0, double w=1.0):
        self._x = x
        self._y = y
        self._z = z
        self._w = w

    @classmethod
    def from_array(cls, a):
        return cls(float(a[0]), float(a[1]), float(a[2]), float(a[3]))

    @staticmethod
    def _make(x, y, z, w):
        cdef Vec4 v = Vec4.__new__(Vec4)
        v._x = x
        v._y = y
        v._z = z
        v._w = w
        return v

    @property
    def _d(self):
        return (self._x, self._y, self._z, self._w)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def w(self):
        return self._w

    def to_list(self):
        return [self._x, self._y, self._z, self._w]

    def __reduce__(self):
        return (type(self), (self._x, self._y, self._z, self._w))


cdef class Quat:
    def __init__(self, double x=0.0, double y=0.0, double z=0.0, double w=1.0):
        self._x = x
        self._y = y
        self._z = z
        self._w = w

    @classmethod
    def from_array(cls, a):
        return cls(float(a[0]), float(a[1]), float(a[2]), float(a[3]))

    @staticmethod
    def _make(x, y, z, w):
        cdef Quat q = Quat.__new__(Quat)
        q._x = x
        q._y = y
        q._z = z
        q._w = w
        return q

    @property
    def _d(self):
        return (self._x, self._y, self._z, self._w)

    @classmethod
    def identity(cls):
        return cls(0, 0, 0, 1)

    @classmethod
    def from_euler(cls, x, y, z):
        x, y, z, w = _get_mh().quat_from_euler(x, y, z)
        return Quat._make(x, y, z, w)

    @classmethod
    def from_axis_angle(cls, axis, angle_deg):
        ax = axis.normalized()
        a = math.radians(angle_deg) * 0.5
        s = math.sin(a)
        return Quat._make(ax._x * s, ax._y * s, ax._z * s, math.cos(a))

    @classmethod
    def look_rotation(cls, forward, up=None):
        if up is None:
            up = Vec3.up()
        f = forward.normalized()
        r = up.cross(f).normalized()
        u = f.cross(r)
        m = np.eye(3, dtype=np.float64)
        m[0] = np.array([r._x, r._y, r._z])
        m[1] = np.array([u._x, u._y, u._z])
        m[2] = np.array([f._x, f._y, f._z])
        return cls._from_rotation_matrix3(m)

    @classmethod
    def _from_rotation_matrix3(cls, m):
        t = m[0, 0] + m[1, 1] + m[2, 2]
        if t > 0:
            s = 0.5 / math.sqrt(t + 1.0)
            return Quat._make((m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s, 0.25 / s)
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            return Quat._make(0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            return Quat._make((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            return Quat._make((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def w(self):
        return self._w

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

    def rotate_vec3(self, v):
        x, y, z = _get_mh().quat_rotate_vec3(self._x, self._y, self._z, self._w, v._x, v._y, v._z)
        return Vec3(x, y, z)

    def to_euler(self):
        x, y, z = _get_mh().quat_to_euler(self._x, self._y, self._z, self._w)
        return Vec3(x, y, z)

    def to_matrix4(self):
        from core.maths.math3d import Mat4
        n = self.normalized()
        x, y, z, w = n._x, n._y, n._z, n._w
        m = np.eye(4, dtype=np.float64)
        m[0, 0] = 1 - 2 * y * y - 2 * z * z
        m[0, 1] = 2 * x * y + 2 * w * z
        m[0, 2] = 2 * x * z - 2 * w * y
        m[1, 0] = 2 * x * y - 2 * w * z
        m[1, 1] = 1 - 2 * x * x - 2 * z * z
        m[1, 2] = 2 * y * z + 2 * w * x
        m[2, 0] = 2 * x * z + 2 * w * y
        m[2, 1] = 2 * y * z - 2 * w * x
        m[2, 2] = 1 - 2 * x * x - 2 * y * y
        return Mat4(m)

    def slerp(self, o, t):
        x, y, z, w = _get_mh().quat_slerp(self._x, self._y, self._z, self._w, o._x, o._y, o._z, o._w, t)
        n = math.sqrt(x * x + y * y + z * z + w * w)
        if n > 1e-10:
            inv = 1.0 / n
            return Quat._make(x * inv, y * inv, z * inv, w * inv)
        return Quat.identity()

    def to_list(self):
        return [self._x, self._y, self._z, self._w]

    def __reduce__(self):
        return (type(self), (self._x, self._y, self._z, self._w))
