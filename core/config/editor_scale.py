# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations


def _get_scale() -> float:
    try:
        from core.config.config import get_global_config
        cfg = get_global_config()
        return cfg.get("editor.ui_scale", 100) / 100.0
    except Exception:
        return 1.0


def scale(val) -> int:
    return int(val * _get_scale())


def scale_xy(w, h) -> tuple[int, int]:
    s = _get_scale()
    return int(w * s), int(h * s)
