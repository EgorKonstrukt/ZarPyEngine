# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was distributed with this file, You
# can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

"""Subprocess worker for mesh thumbnail generation.

This module is executed inside separate processes via a multiprocessing.Pool
so that the heavy assimp import (``load_mesh``) and the numpy/QPainter
rendering never touch the editor's main-process GIL. Each worker renders the
mesh to a QPixmap using an offscreen Qt platform and returns the result as PNG
bytes, which are cheap to pickle back to the main process.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import math
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import (
    QPixmap, QPainter, QColor, QPen, QBrush, QPainterPath,
)
from PyQt6.QtCore import Qt, QBuffer, QIODevice

from core.assets.asset_importer import load_mesh

_app = None


def _ensure_app() -> QApplication:
    global _app
    if _app is None:
        argv = sys.argv if sys.argv else ["mesh_thumb_worker"]
        _app = QApplication(argv)
    return _app


def _render_mesh_ortho(verts_flat: np.ndarray, idx: np.ndarray, size: int) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    if len(verts_flat) < 3 or len(idx) < 3:
        return pm
    pts = verts_flat.reshape(-1, 3).copy()
    rot_y = math.radians(-45)
    rot_x = math.radians(30)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)
    for i in range(len(pts)):
        x, y, z = pts[i]
        rx = x * cos_y - z * sin_y
        rz = x * sin_y + z * cos_y
        ry = y * cos_x - rz * sin_x
        rz = y * sin_x + rz * cos_x
        pts[i] = [rx, ry, rz]
    proj = pts[:, :2].copy()
    cx, cy = proj.mean(axis=0)
    proj -= [cx, cy]
    max_ext = np.abs(proj).max()
    if max_ext < 1e-8:
        return pm
    margin = max(4, size // 16)
    s = (size - 2 * margin) / (2 * max_ext)
    proj *= s
    proj += [size // 2, size // 2]
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    tri_color = QColor(100, 160, 220, 40)
    wire_color = QColor(180, 210, 240, 200)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(tri_color))
    path = QPainterPath()
    for i in range(0, len(idx), 3):
        if i + 2 >= len(idx):
            break
        i0, i1, i2 = int(idx[i]), int(idx[i + 1]), int(idx[i + 2])
        if i0 >= len(proj) or i1 >= len(proj) or i2 >= len(proj):
            continue
        x0, y0 = proj[i0]
        x1, y1 = proj[i1]
        x2, y2 = proj[i2]
        path.moveTo(x0, y0)
        path.lineTo(x1, y1)
        path.lineTo(x2, y2)
        path.closeSubpath()
    p.drawPath(path)
    p.setPen(QPen(wire_color, 1))
    p.setBrush(Qt.BrushStyle.NoBrush)
    for i in range(0, len(idx), 3):
        if i + 2 >= len(idx):
            break
        i0, i1, i2 = int(idx[i]), int(idx[i + 1]), int(idx[i + 2])
        if i0 >= len(proj) or i1 >= len(proj) or i2 >= len(proj):
            continue
        x0, y0 = proj[i0]
        x1, y1 = proj[i1]
        x2, y2 = proj[i2]
        p.drawLine(int(x0), int(y0), int(x1), int(y1))
        p.drawLine(int(x1), int(y1), int(x2), int(y2))
        p.drawLine(int(x2), int(y2), int(x0), int(y0))
    p.end()
    return pm


def render_mesh_png(path: str, size: int):
    """Load a mesh, render it offscreen and return PNG bytes (or None)."""
    _ensure_app()
    try:
        data = load_mesh(path)
    except Exception:
        return None
    if data is None or len(getattr(data, "vertices", [])) < 3 or len(getattr(data, "indices", [])) < 3:
        return None
    pm = _render_mesh_ortho(data.vertices, data.indices, size)
    if pm.isNull():
        return None
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    if not pm.save(buf, "PNG"):
        return None
    return bytes(buf.data())
