# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from multiprocessing import shared_memory
import numpy as np
from typing import Optional
import uuid

MAX_ENTITIES = 4096

_DTYPE_FLAGS = np.uint8
_DTYPE_BODY = np.int32
_DTYPE_F32 = np.float32

_EDATA_COLS = 14
_RDATA_COLS = 14

_SZ_FLAGS = MAX_ENTITIES
_SZ_BODY_MAP = MAX_ENTITIES * _DTYPE_BODY().itemsize
_SZ_EDATA = MAX_ENTITIES * _EDATA_COLS * _DTYPE_F32().itemsize
_SZ_FDATA = MAX_ENTITIES * 6 * _DTYPE_F32().itemsize
_SZ_RDATA = MAX_ENTITIES * _RDATA_COLS * _DTYPE_F32().itemsize

_OFF_RESULT_VER = 0
_OFF_NUM = 8
_OFF_FLAGS = 12
_OFF_BODY_MAP = _OFF_FLAGS + _SZ_FLAGS
_OFF_EDATA = _OFF_BODY_MAP + _SZ_BODY_MAP
_OFF_FDATA = _OFF_EDATA + _SZ_EDATA
_OFF_RDATA = _OFF_FDATA + _SZ_FDATA
TOTAL_SIZE = _OFF_RDATA + _SZ_RDATA


class SharedPhysicsBuffer:
    def __init__(self, name: str = ""):
        self._name = name or f"physics_{uuid.uuid4().hex[:8]}"
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._own = False

    @property
    def name(self) -> str:
        return self._name

    def create(self) -> "SharedPhysicsBuffer":
        self._shm = shared_memory.SharedMemory(name=self._name, create=True, size=TOTAL_SIZE)
        self._own = True
        self._map_arrays()
        return self

    def attach(self, name: str = "") -> "SharedPhysicsBuffer":
        if name:
            self._name = name
        self._shm = shared_memory.SharedMemory(name=self._name)
        self._map_arrays()
        return self

    def _map_arrays(self):
        b = self._shm.buf
        off = _OFF_FLAGS
        self._flags_nd = np.ndarray((MAX_ENTITIES,), dtype=_DTYPE_FLAGS, buffer=b[off:off+_SZ_FLAGS])
        off = _OFF_BODY_MAP
        self._body_map_nd = np.ndarray((MAX_ENTITIES,), dtype=_DTYPE_BODY, buffer=b[off:off+_SZ_BODY_MAP])
        off = _OFF_EDATA
        self._edata_nd = np.ndarray((MAX_ENTITIES, _EDATA_COLS), dtype=_DTYPE_F32, buffer=b[off:off+_SZ_EDATA])
        off = _OFF_FDATA
        self._fdata_nd = np.ndarray((MAX_ENTITIES, 6), dtype=_DTYPE_F32, buffer=b[off:off+_SZ_FDATA])
        off = _OFF_RDATA
        self._rdata_nd = np.ndarray((MAX_ENTITIES, _RDATA_COLS), dtype=_DTYPE_F32, buffer=b[off:off+_SZ_RDATA])

    def close(self):
        if self._shm:
            self._shm.close()

    def unlink(self):
        if self._shm:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    # --- result_version (uint64 at offset 0) ---
    def get_result_version(self) -> int:
        v = memoryview(self._shm.buf[0:8]).cast('Q')
        return int(v[0])

    def set_result_version(self, v: int):
        memoryview(self._shm.buf[0:8]).cast('Q')[0] = v

    # --- num_entities (int32 at offset 8) ---
    def get_num_entities(self) -> int:
        v = memoryview(self._shm.buf[_OFF_NUM:_OFF_NUM+4]).cast('i')
        return int(v[0])

    def set_num_entities(self, n: int):
        memoryview(self._shm.buf[_OFF_NUM:_OFF_NUM+4]).cast('i')[0] = n

    # --- flags ---
    def set_active(self, slot: int, active: bool = True):
        f = self._flags_nd[slot]
        if active:
            self._flags_nd[slot] = f | 1
        else:
            self._flags_nd[slot] = f & 0xFE

    def set_dirty(self, slot: int, dirty: bool = True):
        f = self._flags_nd[slot]
        if dirty:
            self._flags_nd[slot] = f | 2
        else:
            self._flags_nd[slot] = f & 0xFD

    def set_kinematic(self, slot: int, v: bool):
        f = self._flags_nd[slot]
        self._flags_nd[slot] = (f | 4) if v else (f & 0xFB)

    def set_2d(self, slot: int, v: bool):
        f = self._flags_nd[slot]
        self._flags_nd[slot] = (f | 8) if v else (f & 0xF7)

    def get_flags(self, slot: int) -> int:
        return int(self._flags_nd[slot])

    # --- body_map ---
    def set_body_id(self, slot: int, body_id: int):
        self._body_map_nd[slot] = body_id

    def get_body_id(self, slot: int) -> int:
        return int(self._body_map_nd[slot])

    def write_entity_data(self, slot: int, pos, rot, vel, ang_vel):
        row = self._edata_nd[slot]
        row[0:3] = pos
        if len(rot) == 4:
            row[3:7] = rot
            row[7:10] = vel
            row[10:13] = ang_vel
        else:
            import math
            rx, ry, rz = rot
            hx, hy, hz = rx * 0.5, ry * 0.5, rz * 0.5
            sx, cx = math.sin(hx), math.cos(hx)
            sy, cy = math.sin(hy), math.cos(hy)
            sz, cz = math.sin(hz), math.cos(hz)
            row[3] = sx * cy * cz - cx * sy * sz
            row[4] = cx * sy * cz + sx * cy * sz
            row[5] = cx * cy * sz - sx * sy * cz
            row[6] = cx * cy * cz + sx * sy * sz
            row[7:10] = vel
            row[10:13] = ang_vel

    def read_entity_data(self, slot: int):
        row = self._edata_nd[slot]
        return (
            (float(row[0]), float(row[1]), float(row[2])),
            (float(row[3]), float(row[4]), float(row[5]), float(row[6])),
            (float(row[7]), float(row[8]), float(row[9])),
            (float(row[10]), float(row[11]), float(row[12])),
        )

    # --- force_data ---
    def write_force_data(self, slot: int, force, torque):
        row = self._fdata_nd[slot]
        row[0:3] = force
        row[3:6] = torque

    def read_force_data(self, slot: int):
        row = self._fdata_nd[slot]
        return (
            (float(row[0]), float(row[1]), float(row[2])),
            (float(row[3]), float(row[4]), float(row[5])),
        )

    def write_result(self, slot: int, pos, rot, vel, ang_vel):
        row = self._rdata_nd[slot]
        row[0:3] = pos
        if len(rot) == 4:
            row[3:7] = rot
            row[7:10] = vel
            row[10:13] = ang_vel
        else:
            import math
            rx, ry, rz = rot
            hx, hy, hz = rx * 0.5, ry * 0.5, rz * 0.5
            sx, cx = math.sin(hx), math.cos(hx)
            sy, cy = math.sin(hy), math.cos(hy)
            sz, cz = math.sin(hz), math.cos(hz)
            row[3] = sx * cy * cz - cx * sy * sz
            row[4] = cx * sy * cz + sx * cy * sz
            row[5] = cx * cy * sz - sx * sy * cz
            row[6] = cx * cy * cz + sx * sy * sz
            row[7:10] = vel
            row[10:13] = ang_vel

    def read_result(self, slot: int):
        row = self._rdata_nd[slot]
        return (
            (float(row[0]), float(row[1]), float(row[2])),
            (float(row[3]), float(row[4]), float(row[5]), float(row[6])),
            (float(row[7]), float(row[8]), float(row[9])),
            (float(row[10]), float(row[11]), float(row[12])),
        )
