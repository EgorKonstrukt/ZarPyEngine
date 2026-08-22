# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

_NUM_WORKERS = min(8, max(2, (os.cpu_count() or 4)))

_general_pool: ThreadPoolExecutor | None = None
_plugin_pool: ThreadPoolExecutor | None = None
_audio_pool: ThreadPoolExecutor | None = None
_asset_pool: ThreadPoolExecutor | None = None
_bvh_pool: ProcessPoolExecutor | None = None
_bvh_parallel_pool: ThreadPoolExecutor | None = None
_mesh_import_pool: ProcessPoolExecutor | None = None


def _get_or_create(name: str, max_workers: int | None = None) -> ThreadPoolExecutor:
    global _general_pool, _plugin_pool, _audio_pool, _asset_pool, _bvh_parallel_pool
    if name == "general":
        p = _general_pool
        if p is None or getattr(p, "_shutdown", False):
            p = ThreadPoolExecutor(max_workers=max_workers or _NUM_WORKERS, thread_name_prefix=name)
            globals()["_general_pool"] = p
        return p
    if name == "plugin":
        p = _plugin_pool
        if p is None or getattr(p, "_shutdown", False):
            p = ThreadPoolExecutor(max_workers=max_workers or _NUM_WORKERS, thread_name_prefix=name)
            globals()["_plugin_pool"] = p
        return p
    if name == "audio":
        p = _audio_pool
        if p is None or getattr(p, "_shutdown", False):
            p = ThreadPoolExecutor(max_workers=max_workers or 2, thread_name_prefix=name)
            globals()["_audio_pool"] = p
        return p
    if name == "asset":
        p = _asset_pool
        if p is None or getattr(p, "_shutdown", False):
            p = ThreadPoolExecutor(max_workers=max_workers or _NUM_WORKERS, thread_name_prefix=name)
            globals()["_asset_pool"] = p
        return p
    p = _bvh_parallel_pool
    if p is None or getattr(p, "_shutdown", False):
        p = ThreadPoolExecutor(max_workers=max_workers or min(8, max(4, (os.cpu_count() or 4))), thread_name_prefix=name)
        globals()["_bvh_parallel_pool"] = p
    return p


def set_global(name: str, val):
    globals()[name] = val


def general() -> ThreadPoolExecutor:
    return _get_or_create("general")


def plugin() -> ThreadPoolExecutor:
    return _get_or_create("plugin")


def audio() -> ThreadPoolExecutor:
    return _get_or_create("audio", max_workers=2)


def asset() -> ThreadPoolExecutor:
    return _get_or_create("asset")


def bvh() -> ProcessPoolExecutor:
    global _bvh_pool
    if _bvh_pool is None or getattr(_bvh_pool, "_shutdown", False):
        _bvh_pool = ProcessPoolExecutor(
            max_workers=min(4, (os.cpu_count() or 2))
        )
    return _bvh_pool


def bvh_parallel() -> ThreadPoolExecutor:
    return _get_or_create("bvh_parallel", max_workers=min(8, max(4, os.cpu_count() or 4)))


def mesh_import() -> ProcessPoolExecutor:
    global _mesh_import_pool
    if _mesh_import_pool is None or getattr(_mesh_import_pool, "_shutdown", False):
        _mesh_import_pool = ProcessPoolExecutor(
            max_workers=min(4, (os.cpu_count() or 2))
        )
    return _mesh_import_pool
