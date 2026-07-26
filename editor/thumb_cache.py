# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file, You
# can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

"""On-disk thumbnail cache.

Thumbnails are cached as PNG files under ``<project_root>/cache/thumbs``. A
content/size key derived from an xxhash of the file path, its mtime and its
byte size is used so that a thumbnail is only regenerated when the source
asset actually changes.
"""

from __future__ import annotations

import os
import xxhash


def cache_root_for_project(project_root: str) -> str:
    return os.path.join(os.path.abspath(project_root), "cache", "thumbs")


def thumb_disk_key(path: str, size: int, mtime: float, fsize: int,
                    mode: str = "metadata") -> str:
    h = xxhash.xxh64()
    h.update(os.path.abspath(path).encode("utf-8", "surrogateescape"))
    h.update(b"|")
    h.update(str(size).encode())
    h.update(b"|")
    if mode == "content":
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1048576)
                    if not chunk:
                        break
                    h.update(chunk)
        except OSError:
            h.update(str(mtime).encode())
            h.update(b"|")
            h.update(str(fsize).encode())
    else:
        h.update(str(mtime).encode())
        h.update(b"|")
        h.update(str(fsize).encode())
    return h.hexdigest()


def thumb_disk_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, key[:2], key + ".png")


def load_thumb_disk(cache_dir: str, key: str):
    if not cache_dir:
        return None
    p = thumb_disk_path(cache_dir, key)
    try:
        if os.path.isfile(p):
            with open(p, "rb") as f:
                return f.read()
    except Exception:
        return None
    return None


def save_thumb_disk(cache_dir: str, key: str, png_bytes: bytes):
    if not cache_dir or not png_bytes:
        return
    p = thumb_disk_path(cache_dir, key)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(png_bytes)
    except Exception:
        pass
