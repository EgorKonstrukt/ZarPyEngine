# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPen

from core.math.math3d import Vec3
from core.renderer.render_stats import (
    _SPIKE_LOG,
    build_stats_rows,
    collect_render_stats,
    compute_frame_metrics,
    draw_stats_panel,
    log_spike,
)


def draw_stats_overlay(vp, painter):
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    font = QFont("Consolas", 9)
    font.setStyleStrategy(QFont.StyleStrategy.ForceOutline)
    painter.setFont(font)

    paint_dt = getattr(vp, '_paint_dt', 0.016)
    if not hasattr(vp, '_frame_times_ms'):
        vp._frame_times_ms = []
    if paint_dt > 0:
        vp._frame_times_ms.append(paint_dt * 1000.0)
        if len(vp._frame_times_ms) > 300:
            vp._frame_times_ms.pop(0)

    if vp._frame_times_ms and vp._frame_times_ms[-1] > 33.0:
        prof = getattr(vp._engine, '_profiler', None)
        if prof is not None and getattr(prof, 'enabled', False):
            log_spike(vp._frame_times_ms[-1], prof)

    m = compute_frame_metrics(vp._frame_times_ms)
    live_fps = getattr(vp, '_fps', 0.0) or 0.0
    if live_fps > 0:
        m['fps'] = live_fps
        m['avg_fps'] = live_fps
    st = collect_render_stats(vp._engine, vp._renderer)
    fw, fh = vp._get_physical_dims()
    timings = {
        'cpu_ms': paint_dt * 1000.0,
        'render_ms': getattr(vp, '_last_render_ms', 0.0) or 0.0,
        'gizmo_ms': getattr(vp, '_last_gizmo_ms', 0.0) or 0.0,
        'overlay_ms': getattr(vp, '_last_overlay_ms', 0.0) or 0.0,
        'paint_ms': getattr(vp, '_last_paint_full_ms', 0.0) or 0.0,
        'res': f"{fw}x{fh}",
    }
    rows = build_stats_rows(m, st, timings)
    draw_stats_panel(painter, rows, vp._frame_times_ms, _SPIKE_LOG)
    painter.restore()


def draw_delta_label(vp, painter):
    dt = vp._gizmo.delta_text
    if not dt or not vp._gizmo.show_delta_label:
        return
    painter.save()
    f = QFont("Segoe UI", 12, QFont.Weight.Bold)
    f.setStyleStrategy(QFont.StyleStrategy.ForceOutline)
    painter.setFont(f)
    mx, my = vp._last_mouse_pos
    fm = QFontMetrics(painter.font())
    tw = fm.horizontalAdvance(dt) + 16
    th = fm.height() + 6
    rect = QRect(mx + 12, my - th - 8, tw, th)
    painter.setPen(QPen(QColor(255, 170, 0, 220), 1))
    painter.setBrush(QBrush(QColor(30, 30, 30, 200)))
    painter.drawRoundedRect(rect, 4, 4)
    painter.setPen(QColor(255, 170, 0, 255))
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, dt)
    painter.restore()


def get_grid_label_positions(vp):
    w, h = vp.width(), vp.height()
    if w <= 0 or h <= 0:
        return [], []
    cam_pos = vp._cam.position
    step = vp._renderer._compute_grid_size(cam_pos) if vp._renderer else 10.0
    inv_vp = (vp._cam.get_view_matrix() * vp._cam.get_projection_matrix(w / max(1, h))).inverted()
    corners_ndc = [
        np.array([-1.0, -1.0, -1.0, 1.0]),
        np.array([1.0, -1.0, -1.0, 1.0]),
        np.array([1.0, 1.0, -1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0, 1.0]),
    ]
    ground_points = []
    for ndc in corners_ndc:
        near_w = ndc @ inv_vp._d
        near_w /= near_w[3]
        far_ndc = np.array([ndc[0], ndc[1], 1.0, 1.0])
        far_w = far_ndc @ inv_vp._d
        far_w /= far_w[3]
        dx = far_w[0] - near_w[0]
        dy = far_w[1] - near_w[1]
        dz = far_w[2] - near_w[2]
        if abs(dy) < 1e-8:
            continue
        t = -near_w[1] / dy
        if t > 0:
            gx = near_w[0] + dx * t
            gz = near_w[2] + dz * t
            ground_points.append((gx, gz))
    if len(ground_points) < 3:
        return [], []
    all_x = [p[0] for p in ground_points]
    all_z = [p[1] for p in ground_points]
    min_x, max_x = min(all_x), max(all_x)
    min_z, max_z = min(all_z), max(all_z)
    margin = step * 2.0
    start_x = int((min_x - margin) / step) * step
    end_x = int((max_x + margin) / step) * step
    start_z = int((min_z - margin) / step) * step
    end_z = int((max_z + margin) / step) * step
    x_labels = []
    z_labels = []
    MAX_ITERATIONS = 1000
    x_step = int(step)
    x_count = max(1, int((end_x - start_x) / x_step))
    if x_count > MAX_ITERATIONS:
        x_step = max(1, int(x_step * x_count / MAX_ITERATIONS))
    z_step = int(step)
    z_count = max(1, int((end_z - start_z) / z_step))
    if z_count > MAX_ITERATIONS:
        z_step = max(1, int(z_step * z_count / MAX_ITERATIONS))
    for val in range(int(start_x), int(end_x) + x_step, x_step):
        clip = inv_vp._d @ np.array([float(val), 0.0, cam_pos.z, 1.0])
        if abs(clip[3]) < 1e-6:
            continue
        ndc = clip[:3] / clip[3]
        sx = (ndc[0] + 1.0) * 0.5 * w
        sy = (1.0 - ndc[1]) * 0.5 * h
        if ndc[2] < -1 or ndc[2] > 1:
            continue
        if h * 0.75 <= sy <= h and 0 <= sx <= w:
            x_labels.append((sx, sy, val))
    for val in range(int(start_z), int(end_z) + z_step, z_step):
        clip = inv_vp._d @ np.array([cam_pos.x, 0.0, float(val), 1.0])
        if abs(clip[3]) < 1e-6:
            continue
        ndc = clip[:3] / clip[3]
        sx = (ndc[0] + 1.0) * 0.5 * w
        sy = (1.0 - ndc[1]) * 0.5 * h
        if ndc[2] < -1 or ndc[2] > 1:
            continue
        if sy <= h * 0.75 and 0 <= sx <= w:
            z_labels.append((sx, sy, val))
    return x_labels[:20], z_labels[:20]


def draw_grid_labels(vp, painter):
    x_labels, z_labels = get_grid_label_positions(vp)
    if not x_labels and not z_labels:
        return
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont("Segoe UI", 9)
    font.setStyleStrategy(QFont.StyleStrategy.ForceOutline)
    painter.setFont(font)
    margin = 4
    bg_color = QColor(30, 30, 30, 200)
    text_color = QColor(180, 180, 180)
    pen = QPen(QColor(100, 100, 100))
    for sx, sy, val in x_labels:
        text = str(val)
        rect = QRect(int(sx - 20), int(sy) - 7, 40, 16)
        painter.fillRect(rect, bg_color)
        painter.setPen(pen)
        painter.drawRect(rect)
        painter.setPen(text_color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
    for sx, sy, val in z_labels:
        text = str(val)
        rect = QRect(int(sx - 20), int(sy) - 7, 40, 16)
        painter.fillRect(rect, bg_color)
        painter.setPen(pen)
        painter.drawRect(rect)
        painter.setPen(text_color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
    painter.restore()
