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

class GpuSkinning:
    __slots__ = ('_ctx', '_prog', '_cap', '_in_pos', '_in_nrm', '_bone_mats', '_bone_idx', '_bone_wgt', '_out_pos', '_out_nrm', '_ready')
    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._prog: Optional[moderngl.ComputeShader] = None
        self._cap: int = 0
        self._in_pos: Optional[moderngl.Buffer] = None
        self._in_nrm: Optional[moderngl.Buffer] = None
        self._bone_mats: Optional[moderngl.Buffer] = None
        self._bone_idx: Optional[moderngl.Buffer] = None
        self._bone_wgt: Optional[moderngl.Buffer] = None
        self._out_pos: Optional[moderngl.Buffer] = None
        self._out_nrm: Optional[moderngl.Buffer] = None
        self._ready: bool = False
        self._try_compile()
    def _try_compile(self):
        try:
            p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shaders", "skinning.comp")
            if not os.path.exists(p):
                from core.renderer.mesh_data import SHADER_DIR
                p = os.path.join(SHADER_DIR, "skinning.comp")
            if os.path.exists(p):
                with open(p) as f:
                    src = f.read()
                self._prog = self._ctx.compute_shader(src)
                self._ready = self._prog is not None
        except Exception:
            self._ready = False
    def is_ready(self) -> bool:
        return self._ready and self._prog is not None
    def ensure(self, nverts: int, nbones: int):
        if nverts <= self._cap:
            return
        new_cap = max(nverts, max(1024, self._cap * 2)) if self._cap else nverts
        for b in (self._in_pos, self._in_nrm, self._bone_idx, self._bone_wgt, self._out_pos, self._out_nrm):
            if b is not None:
                try: b.release()
                except: pass
        self._in_pos = self._ctx.buffer(reserve=new_cap * 16)
        self._in_nrm = self._ctx.buffer(reserve=new_cap * 16)
        self._bone_idx = self._ctx.buffer(reserve=new_cap * 16)
        self._bone_wgt = self._ctx.buffer(reserve=new_cap * 16)
        self._out_pos = self._ctx.buffer(reserve=new_cap * 16)
        self._out_nrm = self._ctx.buffer(reserve=new_cap * 16)
        if self._bone_mats is None or self._bone_mats.size < nbones * 64:
            if self._bone_mats is not None:
                try: self._bone_mats.release()
                except: pass
            self._bone_mats = self._ctx.buffer(reserve=max(64, nbones * 64))
        self._cap = new_cap
    def dispatch(self, verts: np.ndarray, normals: np.ndarray, bone_mats: np.ndarray, bone_idx: np.ndarray, bone_wgt: np.ndarray):
        n = verts.shape[0]
        if n == 0 or not self.is_ready():
            return None, None
        self.ensure(n, bone_mats.shape[0])
        self._in_pos.write(verts.astype(np.float32).tobytes())
        self._in_nrm.write(normals.astype(np.float32).tobytes())
        self._bone_mats.write(bone_mats.astype(np.float32).tobytes())
        self._bone_idx.write(bone_idx.astype(np.int32).tobytes())
        self._bone_wgt.write(bone_wgt.astype(np.float32).tobytes())
        self._in_pos.bind_to_storage_buffer(0)
        self._in_nrm.bind_to_storage_buffer(0)
        self._bone_mats.bind_to_storage_buffer(1)
        self._bone_idx.bind_to_storage_buffer(2)
        self._bone_wgt.bind_to_storage_buffer(3)
        self._out_pos.bind_to_storage_buffer(4)
        groups = (n + 63)//64
        try:
            self._prog.run(groups, 1, 1)
            self._ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)
            return self._out_pos, self._out_nrm
        except Exception:
            return None, None
    def release(self):
        for b in (self._in_pos, self._in_nrm, self._bone_mats, self._bone_idx, self._bone_wgt, self._out_pos, self._out_nrm):
            if b is not None:
                try: b.release()
                except: pass
        if self._prog is not None:
            try: self._prog.release()
            except: pass
        self._cap = 0
        self._ready = False
