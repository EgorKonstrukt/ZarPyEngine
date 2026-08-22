# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun
from __future__ import annotations
import os
import numpy as np
import moderngl
from typing import Optional

class GpuTransform:
    __slots__ = ('_ctx', '_prog', '_pos_buf', '_rot_buf', '_scale_buf', '_parent_buf', '_out_buf', '_cap', '_ready')
    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._prog: Optional[moderngl.ComputeShader] = None
        self._pos_buf: Optional[moderngl.Buffer] = None
        self._rot_buf: Optional[moderngl.Buffer] = None
        self._scale_buf: Optional[moderngl.Buffer] = None
        self._parent_buf: Optional[moderngl.Buffer] = None
        self._out_buf: Optional[moderngl.Buffer] = None
        self._cap: int = 0
        self._ready: bool = False
        self._try_compile()
    def _try_compile(self):
        try:
            from core.renderer.mesh_data import SHADER_DIR
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shaders", "transform.comp")
            if not os.path.exists(path):
                path = os.path.join(SHADER_DIR, "transform.comp")
                if not os.path.exists(path):
                    path = os.path.join(os.path.dirname(__file__), "..", "shaders", "transform.comp")
            if os.path.exists(path):
                with open(path) as f:
                    src = f.read()
                self._prog = self._ctx.compute_shader(src)
                self._ready = self._prog is not None
        except Exception:
            self._ready = False
    def is_ready(self) -> bool:
        return self._ready and self._prog is not None
    def ensure_capacity(self, n: int):
        if n <= self._cap:
            return
        new_cap = max(n, max(256, self._cap * 2)) if self._cap else n
        for b in (self._pos_buf, self._rot_buf, self._scale_buf, self._parent_buf, self._out_buf):
            if b is not None:
                try: b.release()
                except: pass
        self._pos_buf = self._ctx.buffer(reserve=new_cap * 16)
        self._rot_buf = self._ctx.buffer(reserve=new_cap * 16)
        self._scale_buf = self._ctx.buffer(reserve=new_cap * 16)
        self._parent_buf = self._ctx.buffer(reserve=new_cap * 4)
        self._out_buf = self._ctx.buffer(reserve=new_cap * 64)
        self._cap = new_cap
    def upload_and_dispatch(self, pos: np.ndarray, rot: np.ndarray, scale: np.ndarray, parent_idx: np.ndarray) -> Optional[moderngl.Buffer]:
        n = pos.shape[0]
        if n == 0 or not self.is_ready():
            return None
        self.ensure_capacity(n)
        self._pos_buf.write(pos.astype(np.float32).tobytes())
        self._rot_buf.write(rot.astype(np.float32).tobytes())
        self._scale_buf.write(scale.astype(np.float32).tobytes())
        self._parent_buf.write(parent_idx.astype(np.int32).tobytes())
        self._pos_buf.bind_to_storage_buffer(0)
        self._rot_buf.bind_to_storage_buffer(1)
        self._scale_buf.bind_to_storage_buffer(2)
        self._parent_buf.bind_to_storage_buffer(3)
        self._out_buf.bind_to_storage_buffer(4)
        groups = (n + 63)//64
        try:
            self._prog.run(groups, 1, 1)
            self._ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)
            return self._out_buf
        except Exception:
            return None
    def read_result(self, n: int) -> Optional[np.ndarray]:
        if self._out_buf is None:
            return None
        try:
            data = self._out_buf.read(n * 64)
            return np.frombuffer(data, dtype=np.float32).reshape(n, 4, 4)
        except Exception:
            return None
    def release(self):
        for b in (self._pos_buf, self._rot_buf, self._scale_buf, self._parent_buf, self._out_buf):
            if b is not None:
                try: b.release()
                except: pass
        self._pos_buf = self._rot_buf = self._scale_buf = self._parent_buf = self._out_buf = None
        if self._prog is not None:
            try: self._prog.release()
            except: pass
            self._prog = None
        self._cap = 0
        self._ready = False
