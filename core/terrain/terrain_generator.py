# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import math
import numpy as np
import moderngl
from dataclasses import dataclass, field
from typing import Optional
from core.foundation.logger import Logger
from core.shaders.compute_shader import compile_compute_shader

_SHADER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shaders", "terrain_gen.compute")

_DEFAULTS = {
    "resolution": 512,
    "octaves": 9,
    "baseFrequency": 0.010,
    "lacunarity": 2.03,
    "persistence": 0.5,
    "heightScale": 120.0,
    "offset": 0.0,
    "warpeness": 0.6,
    "warpFrequency": 0.018,
    "warpIterations": 3.0,
    "ridge": 0.55,
    "ridgePower": 4.0,
    "ridgeSharpness": 1.0,
    "billow": 0.0,
    "billowPower": 2.0,
    "continentMask": 0.6,
    "continentScale": 0.0025,
    "continentFalloff": 1.4,
    "detail": 0.35,
    "detailFrequency": 6.0,
    "plateau": 0.0,
    "plateauLevel": 0.5,
    "strata": 0.0,
    "strataScale": 12.0,
    "slopeMask": 0.0,
    "slopeMin": 0.35,
    "thermalErosion": 0.35,
    "hydraulicErosion": 0.5,
    "erosionIterations": 16.0,
    "talus": 0.04,
    "sedimentCapacity": 4.0,
    "erosionStrength": 0.3,
    "peakSmoothing": 0.15,
    "valleyDepth": 0.0,
    "noiseSeed": 1337.0,
    "riverStrength": 0.0,
    "terrace": 0.0,
    "terraceSteps": 8.0,
    "dune": 0.0,
    "duneDir": 0.0,
    "fractalTwist": 0.0,
    "sharpen": 0.0,
    "heightBias": 0.0,
    "normalizeMin": -1.0,
    "normalizeMax": 1.0,
    "flipY": 0,
}

_FLOAT_KEYS = {
    "baseFrequency", "lacunarity", "persistence", "heightScale", "offset", "warpeness",
    "warpFrequency", "warpIterations", "ridge", "ridgePower", "ridgeSharpness", "billow",
    "billowPower", "continentMask", "continentScale", "continentFalloff", "detail",
    "detailFrequency", "plateau", "plateauLevel", "strata", "strataScale", "slopeMask",
    "slopeMin", "thermalErosion", "hydraulicErosion", "erosionIterations", "talus",
    "sedimentCapacity", "erosionStrength", "peakSmoothing", "valleyDepth", "noiseSeed",
    "riverStrength", "terrace", "terraceSteps", "dune", "duneDir", "fractalTwist",
    "sharpen", "heightBias", "normalizeMin", "normalizeMax",
}

_INT_KEYS = {"resolution", "octaves", "flipY"}


@dataclass
class TerrainSettings:
    data: dict = field(default_factory=lambda: dict(_DEFAULTS))

    def get(self, key: str, default=None):
        if key in self.data:
            return self.data[key]
        return default if default is not None else _DEFAULTS.get(key)

    def set(self, key: str, value):
        self.data[key] = value

    def copy(self) -> "TerrainSettings":
        return TerrainSettings(dict(self.data))

    def to_dict(self) -> dict:
        return dict(self.data)

    @classmethod
    def from_dict(cls, d: dict) -> "TerrainSettings":
        merged = dict(_DEFAULTS)
        merged.update(d or {})
        return cls(merged)


class TerrainGenerator:
    def __init__(self):
        self._ctx: Optional[moderngl.Context] = None
        self._owns_ctx: bool = False
        self._program: Optional[moderngl.ComputeShader] = None
        self._height_buf: Optional[moderngl.Buffer] = None
        self._last_res = 0
        self._float_uniforms: list[str] = []
        self._int_uniforms: list[str] = []

    def _ensure_ctx(self) -> bool:
        if self._ctx is not None:
            return True
        ctx = self._try_engine_context()
        if ctx is not None:
            self._ctx = ctx
            self._owns_ctx = False
            try:
                self._ctx.pixel_alignment = 1
            except Exception:
                pass
            return True
        try:
            self._ctx = moderngl.create_standalone_context(require=460)
            self._ctx.pixel_alignment = 1
            self._owns_ctx = True
        except Exception as e:
            Logger.error(f"TerrainGenerator: cannot create GL context: {e}", e)
            return False
        return True

    def _try_engine_context(self) -> Optional[moderngl.Context]:
        try:
            from core.engine.engine import Engine
            eng = Engine.instance()
            if eng is None:
                return None
            vp = getattr(eng, "viewport", None)
            if vp is None:
                return None
            renderer = getattr(vp, "renderer", None)
            if renderer is not None and getattr(renderer, "_ctx", None) is not None:
                return renderer._ctx
            ctx = getattr(vp, "_ctx", None)
            if ctx is not None:
                return ctx
        except Exception:
            pass
        return None

    def _load_program(self):
        if self._program is not None:
            return
        if not os.path.exists(_SHADER_PATH):
            Logger.error(f"TerrainGenerator: shader not found: {_SHADER_PATH}")
            return
        with open(_SHADER_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        start = content.find("GLSLPROGRAM")
        end = content.find("ENDGLSL", start)
        if start < 0 or end < 0:
            Logger.error("TerrainGenerator: invalid shader file")
            return
        source = content[start + len("GLSLPROGRAM"):end].strip()
        self._program = compile_compute_shader(self._ctx, source, _SHADER_PATH)

    def _cache_uniform_names(self):
        if self._float_uniforms or self._int_uniforms:
            return
        prog = self._program
        if prog is None:
            return
        float_names = set("u_" + k for k in _FLOAT_KEYS)
        int_names = set("u_" + k for k in _INT_KEYS)
        for name in prog:
            if name in float_names:
                self._float_uniforms.append(name)
            elif name in int_names:
                self._int_uniforms.append(name)

    def generate_heightfield(self, settings: TerrainSettings) -> Optional[np.ndarray]:
        if not self._ensure_ctx():
            return None
        self._load_program()
        if self._program is None:
            return None
        res = int(settings.get("resolution"))
        res = max(16, min(2048, res))
        res = (res // 16) * 16
        if res < 16:
            res = 16
        n = res * res
        if self._height_buf is None or self._last_res != res:
            self._height_buf = self._ctx.buffer(reserve=n * 4)
            self._last_res = res
        self._cache_uniform_names()
        d = settings.data
        for name in self._float_uniforms:
            try:
                self._program[name].value = float(d.get(name[2:], _DEFAULTS[name[2:]]))
            except Exception:
                pass
        for name in self._int_uniforms:
            try:
                self._program[name].value = int(d.get(name[2:], _DEFAULTS[name[2:]]))
            except Exception:
                pass
        self._height_buf.bind_to_storage_buffer(0)
        groups = (res + 15) // 16
        self._program.run(groups, groups, 1)
        self._ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)
        raw = self._height_buf.read()
        arr = np.frombuffer(raw, dtype=np.float32).reshape(res, res).copy()
        return arr

    def generate_mesh(self, settings: TerrainSettings, size: float = 1000.0) -> Optional[dict]:
        hf = self.generate_heightfield(settings)
        if hf is None:
            return None
        return self.mesh_from_heightfield(hf, size)

    def mesh_from_heightfield(self, hf: np.ndarray, size: float = 1000.0) -> Optional[dict]:
        if hf is None:
            return None
        res = hf.shape[0]
        step = size / (res - 1)
        half = size * 0.5
        xs = (np.arange(res) * step - half).astype(np.float32)
        zs = (np.arange(res) * step - half).astype(np.float32)
        gx, gz = np.meshgrid(xs, zs)
        verts = np.empty((res * res, 3), dtype=np.float32)
        verts[:, 0] = gx.ravel()
        verts[:, 1] = hf.ravel().astype(np.float32)
        verts[:, 2] = gz.ravel()
        uvs = np.empty((res * res, 2), dtype=np.float32)
        u = (np.arange(res) / (res - 1)).astype(np.float32)
        uvs[:, 0] = np.tile(u, res)
        uvs[:, 1] = np.repeat(u, res)
        rows = res - 1
        zz, xx = np.meshgrid(np.arange(rows), np.arange(rows), indexing='ij')
        a = (zz * res + xx).ravel()
        b = (zz * res + xx + 1).ravel()
        c = ((zz + 1) * res + xx + 1).ravel()
        d = ((zz + 1) * res + xx).ravel()
        indices = np.empty(len(a) * 6, dtype=np.uint32)
        indices[0::6] = a
        indices[1::6] = c
        indices[2::6] = b
        indices[3::6] = a
        indices[4::6] = d
        indices[5::6] = c
        normals = self._compute_normals(verts, indices, res)
        return {
            "vertices": verts.ravel(),
            "normals": normals.ravel(),
            "uvs": uvs.ravel(),
            "indices": indices,
            "heightfield": hf,
            "size": size,
            "resolution": res,
        }

    def _compute_normals(self, verts: np.ndarray, indices: np.ndarray, res: int) -> np.ndarray:
        v = verts.reshape(-1, 3)
        tris = v[indices.reshape(-1, 3)]
        p0 = tris[:, 0]
        p1 = tris[:, 1]
        p2 = tris[:, 2]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        ln[ln < 1e-8] = 1.0
        nrm /= ln
        normals = np.zeros((len(v), 3), dtype=np.float32)
        np.add.at(normals, indices, np.repeat(nrm, 3, axis=0))
        ln2 = np.linalg.norm(normals, axis=1, keepdims=True)
        ln2[ln2 < 1e-8] = 1.0
        normals /= ln2
        return normals

    def release(self):
        if not self._owns_ctx:
            self._height_buf = None
            self._program = None
            self._ctx = None
            return
        if self._height_buf is not None:
            self._height_buf.release()
            self._height_buf = None
        if self._program is not None:
            self._program.release()
            self._program = None
        if self._ctx is not None:
            self._ctx.release()
            self._ctx = None


_generator: Optional[TerrainGenerator] = None


def get_generator() -> TerrainGenerator:
    global _generator
    if _generator is None:
        _generator = TerrainGenerator()
    return _generator


def build_terrain_mesh(settings: TerrainSettings, size: float = 1000.0) -> Optional[dict]:
    return get_generator().generate_mesh(settings, size)
