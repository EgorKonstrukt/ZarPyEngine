# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from multiprocessing import shared_memory
import numpy as np
from typing import Optional
import uuid

MAX_SOFT_BODIES = 16
MAX_SOFT_VERTS = 32768

_COM_COLS = 10

_SZ_COUNTS = MAX_SOFT_BODIES * np.dtype(np.int32).itemsize
_SZ_COM = MAX_SOFT_BODIES * _COM_COLS * np.dtype(np.float32).itemsize
_SZ_VERTS = MAX_SOFT_BODIES * MAX_SOFT_VERTS * 3 * np.dtype(np.float32).itemsize

_OFF_COUNTS = 0
_OFF_COM = _OFF_COUNTS + _SZ_COUNTS
_OFF_VERTS = _OFF_COM + _SZ_COM
TOTAL_SIZE = _OFF_VERTS + _SZ_VERTS


class SoftSharedBuffer:
    def __init__(self, name: str = ""):
        self._name = name or f"softphysics_{uuid.uuid4().hex[:8]}"
        self._shm: Optional[shared_memory.SharedMemory] = None

    @property
    def name(self) -> str:
        return self._name

    def create(self) -> "SoftSharedBuffer":
        self._shm = shared_memory.SharedMemory(name=self._name, create=True, size=TOTAL_SIZE)
        self._map_arrays()
        self._counts_nd[:] = 0
        return self

    def attach(self, name: str = "") -> "SoftSharedBuffer":
        if name:
            self._name = name
        self._shm = shared_memory.SharedMemory(name=self._name)
        self._map_arrays()
        return self

    def _map_arrays(self):
        b = self._shm.buf
        self._counts_nd = np.ndarray((MAX_SOFT_BODIES,), dtype=np.int32, buffer=b[_OFF_COUNTS:_OFF_COUNTS + _SZ_COUNTS])
        self._com_nd = np.ndarray((MAX_SOFT_BODIES, _COM_COLS), dtype=np.float32, buffer=b[_OFF_COM:_OFF_COM + _SZ_COM])
        self._verts_nd = np.ndarray((MAX_SOFT_BODIES * MAX_SOFT_VERTS, 3), dtype=np.float32, buffer=b[_OFF_VERTS:_OFF_VERTS + _SZ_VERTS])

    def close(self):
        if self._shm:
            self._shm.close()

    def unlink(self):
        if self._shm:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    def write_soft(self, slot: int, verts, com_pos, com_quat, com_vel) -> int:
        n = int(len(verts)) if verts is not None else 0
        if n <= 0 or n > MAX_SOFT_VERTS:
            self._counts_nd[slot] = -n
            return n
        base = slot * MAX_SOFT_VERTS
        self._verts_nd[base:base + n] = np.ascontiguousarray(verts, dtype=np.float32).reshape(-1, 3)
        row = self._com_nd[slot]
        row[0:3] = com_pos
        row[3:7] = com_quat
        row[7:10] = com_vel
        self._counts_nd[slot] = n
        return n

    def clear_slot(self, slot: int):
        self._counts_nd[slot] = 0

    def read_count(self, slot: int) -> int:
        return int(self._counts_nd[slot])

    def read_soft(self, slot: int):
        n = int(self._counts_nd[slot])
        if n <= 0:
            return None
        base = slot * MAX_SOFT_VERTS
        row = self._com_nd[slot]
        return (
            np.array(self._verts_nd[base:base + n], dtype=np.float32, copy=True),
            (float(row[0]), float(row[1]), float(row[2])),
            (float(row[3]), float(row[4]), float(row[5]), float(row[6])),
            (float(row[7]), float(row[8]), float(row[9])),
        )
