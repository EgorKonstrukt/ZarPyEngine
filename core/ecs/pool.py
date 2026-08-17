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
    pools = {
        "general": lambda: _general_pool,
        "plugin": lambda: _plugin_pool,
        "audio": lambda: _audio_pool,
        "asset": lambda: _asset_pool,
        "bvh_parallel": lambda: _bvh_parallel_pool,
    }
    setters = {
        "general": lambda v: set_global("_general_pool", v),
        "plugin": lambda v: set_global("_plugin_pool", v),
        "audio": lambda v: set_global("_audio_pool", v),
        "asset": lambda v: set_global("_asset_pool", v),
        "bvh_parallel": lambda v: set_global("_bvh_parallel_pool", v),
    }
    p = pools[name]()
    if p is None or p._shutdown:
        p = ThreadPoolExecutor(max_workers=max_workers or _NUM_WORKERS, thread_name_prefix=name)
        setters[name](p)
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
