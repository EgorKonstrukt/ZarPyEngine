# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from PyQt6.QtWidgets import QStatusBar, QLabel
from PyQt6.QtCore import QTimer


def _get_gpu_name(mw) -> str:
    try:
        vp = getattr(mw, '_viewport', None)
        ctx = getattr(vp, '_ctx', None)
        if ctx is not None:
            info = ctx.info
            return info.get("GL_RENDERER", "Unknown")
    except Exception:
        pass
    return "Unknown"


def setup_statusbar(mw):
    mw._statusbar = QStatusBar(mw)
    mw.setStatusBar(mw._statusbar)
    mw._statusbar.setStyleSheet("QStatusBar::item { border: none; }")
    gpu_name = _get_gpu_name(mw)
    mw._status_gpu_name_lbl = QLabel(gpu_name)
    mw._status_gpu_name_lbl.setStyleSheet("padding: 0 4px; font-weight: bold;")
    mw._statusbar.addPermanentWidget(mw._status_gpu_name_lbl)
    mw._status_scene_lbl = QLabel("No scene")
    mw._statusbar.addPermanentWidget(mw._status_scene_lbl)
    mw._status_mode_lbl = QLabel("Edit Mode")
    mw._statusbar.addPermanentWidget(mw._status_mode_lbl)
    mw._status_fps_lbl = QLabel("FPS: 0 | TPS: 0")
    mw._statusbar.addPermanentWidget(mw._status_fps_lbl)
    mw._status_cpu_lbl = QLabel("CPU: --%")
    mw._status_cpu_lbl.setStyleSheet("padding: 0 4px;")
    mw._statusbar.addPermanentWidget(mw._status_cpu_lbl)
    mw._status_gpu_lbl = QLabel("GPU: --%")
    mw._status_gpu_lbl.setStyleSheet("padding: 0 4px;")
    mw._statusbar.addPermanentWidget(mw._status_gpu_lbl)
    mw._status_ram_lbl = QLabel("RAM: -- MB")
    mw._status_ram_lbl.setStyleSheet("padding: 0 4px;")
    mw._statusbar.addPermanentWidget(mw._status_ram_lbl)
    mw._status_vram_lbl = QLabel("VRAM: -- MB")
    mw._status_vram_lbl.setStyleSheet("padding: 0 4px;")
    mw._statusbar.addPermanentWidget(mw._status_vram_lbl)
    mw._fps_timer = QTimer(mw)
    mw._fps_timer.timeout.connect(mw._update_status)
    mw._fps_timer.start(1000)
