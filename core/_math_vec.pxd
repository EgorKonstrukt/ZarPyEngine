# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

cdef class Vec2:
    cdef public double _x
    cdef public double _y

cdef class Vec3:
    cdef public double _x
    cdef public double _y
    cdef public double _z

cdef class Vec4:
    cdef public double _x
    cdef public double _y
    cdef public double _z
    cdef public double _w

cdef class Quat:
    cdef public double _x
    cdef public double _y
    cdef public double _z
    cdef public double _w
