# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
import moderngl
from typing import Optional, Dict
from core.foundation.logger import Logger

_ctx: Optional[moderngl.Context] = None
_program: Optional[moderngl.ComputeShader] = None
_buf: Optional[moderngl.Buffer] = None
_last_res: int = 0
_last_source: str = ""

_preview_program: Optional[moderngl.ComputeShader] = None
_preview_buf: Optional[moderngl.Buffer] = None
_preview_last_res: int = 0
_preview_last_source: str = ""


def _ensure_ctx() -> bool:
    global _ctx
    if _ctx is not None:
        return True
    try:
        from core.engine.engine import Engine
        eng = Engine.instance()
        if eng is not None:
            vp = getattr(eng, "viewport", None)
            if vp is not None:
                renderer = getattr(vp, "renderer", None)
                if renderer is not None and getattr(renderer, "_ctx", None) is not None:
                    _ctx = renderer._ctx
                    try:
                        _ctx.pixel_alignment = 1
                    except Exception:
                        pass
                    return True
    except Exception:
        pass
    try:
        _ctx = moderngl.create_standalone_context(require=460)
        _ctx.pixel_alignment = 1
        return True
    except Exception as e:
        Logger.error(f"TerrainNodeGPU: cannot create GL context: {e}", e)
        return False


def run_shader(source: str, resolution: int, uniforms: Optional[Dict[str, float]] = None) -> Optional[np.ndarray]:
    global _program, _buf, _last_res, _last_source

    if not _ensure_ctx():
        return None

    resolution = max(16, min(2048, resolution))
    resolution = (resolution // 16) * 16
    if resolution < 16:
        resolution = 16

    if source != _last_source:
        if _program is not None:
            try:
                _program.release()
            except Exception:
                pass
            _program = None
        try:
            _program = _ctx.compute_shader(source)
            _last_source = source
        except Exception as e:
            Logger.error(f"TerrainNodeGPU: shader compile error: {e}", e)
            _program = None
            return None

    if _program is None:
        return None

    n = resolution * resolution
    if _buf is None or _last_res != resolution:
        if _buf is not None:
            try:
                _buf.release()
            except Exception:
                pass
        _buf = _ctx.buffer(reserve=n * 4)
        _last_res = resolution

    _buf.bind_to_storage_buffer(0)

    try:
        _program["u_resolution"].value = int(resolution)
    except Exception as e:
        Logger.error(f"TerrainNodeGPU: failed to set u_resolution: {e}", e)

    if uniforms:
        for name, value in uniforms.items():
            try:
                if isinstance(value, str):
                    value = float(value)
                _program[name].value = value
            except Exception as e:
                Logger.error(f"TerrainNodeGPU: failed to set uniform '{name}'={value!r}: {e}", e)

    groups = (resolution + 15) // 16
    _program.run(groups, groups, 1)
    _ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)

    raw = _buf.read()
    return np.frombuffer(raw, dtype=np.float32).reshape(resolution, resolution).copy()


def run_preview_shader(source: str, resolution: int, uniforms: Optional[Dict[str, float]] = None) -> Optional[np.ndarray]:
    global _preview_program, _preview_buf, _preview_last_res, _preview_last_source

    if not _ensure_ctx():
        return None

    resolution = max(16, min(128, resolution))
    resolution = (resolution // 16) * 16
    if resolution < 16:
        resolution = 16

    if source != _preview_last_source:
        if _preview_program is not None:
            try:
                _preview_program.release()
            except Exception:
                pass
            _preview_program = None
        try:
            _preview_program = _ctx.compute_shader(source)
            _preview_last_source = source
        except Exception as e:
            Logger.warning(f"TerrainNodeGPU: preview compile error: {e}", e)
            _preview_program = None
            return None

    if _preview_program is None:
        return None

    n = resolution * resolution
    if _preview_buf is None or _preview_last_res != resolution:
        if _preview_buf is not None:
            try:
                _preview_buf.release()
            except Exception:
                pass
        _preview_buf = _ctx.buffer(reserve=n * 4)
        _preview_last_res = resolution

    _preview_buf.bind_to_storage_buffer(0)

    try:
        _preview_program["u_resolution"].value = int(resolution)
    except Exception:
        pass

    if uniforms:
        for name, value in uniforms.items():
            try:
                if isinstance(value, str):
                    value = float(value)
                _preview_program[name].value = value
            except Exception:
                pass

    groups = (resolution + 15) // 16
    _preview_program.run(groups, groups, 1)
    _ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)

    raw = _preview_buf.read()
    return np.frombuffer(raw, dtype=np.float32).reshape(resolution, resolution).copy()


def release():
    global _ctx, _program, _buf, _last_res, _last_source
    global _preview_program, _preview_buf, _preview_last_res, _preview_last_source
    if _buf is not None:
        try:
            _buf.release()
        except Exception:
            pass
        _buf = None
    if _preview_buf is not None:
        try:
            _preview_buf.release()
        except Exception:
            pass
        _preview_buf = None
    if _program is not None:
        try:
            _program.release()
        except Exception:
            pass
        _program = None
    if _preview_program is not None:
        try:
            _preview_program.release()
        except Exception:
            pass
        _preview_program = None
    _ctx = None
    _last_res = 0
    _last_source = ""
    _preview_last_res = 0
    _preview_last_source = ""


def clear_cache():
    global _program, _last_source
    if _program is not None:
        try:
            _program.release()
        except Exception:
            pass
        _program = None
    _last_source = ""
