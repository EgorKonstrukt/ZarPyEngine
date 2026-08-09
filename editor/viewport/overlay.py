# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPen

from core.maths.math3d import Vec3
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


def draw_audio_viz_header(vp, painter):
    if not getattr(vp, '_audio_viz_enabled', False):
        return
    an = getattr(vp, '_audio_analyzer', None)
    av = getattr(vp, '_audio_viz', None)
    if an is None or av is None:
        return
    try:
        fw, fh = vp._get_physical_dims()
        dpr = vp.devicePixelRatio() or 1.0
        x, y, w, h = av.panel_rect(fw, fh)
        lx, ly = x / dpr, y / dpr
        lw = w / dpr
        painter.save()
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        font.setStyleStrategy(QFont.StyleStrategy.ForceOutline)
        painter.setFont(font)
        fm = QFontMetrics(font)
        bar_w = 90
        pad = 6
        head_h = fm.height() + 8
        rect = QRect(int(lx) + 4, int(ly) + 4, int(lw) - 8, head_h)
        painter.setPen(QPen(QColor(0, 0, 0, 60), 1))
        painter.setBrush(QColor(12, 14, 16, 150))
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QColor(200, 220, 235, 255))
        painter.drawText(rect.left() + pad, rect.center().y() + fm.ascent() * 0.35, "Audio Viz")
        info = f"{an.sample_rate // 1000}kHz"
        try:
            lvl_db = getattr(an, "level_db", None)
            db_txt = f"{lvl_db:+.1f}dB" if lvl_db is not None else ""
            rms_db = getattr(an, "rms_db", None)
            rms_txt = f"{rms_db:+.1f}" if rms_db is not None else ""
            pf = float(getattr(an, "peak_freq", 0.0) or 0.0)
            pf_txt = f"{pf / 1000.0:.1f}k" if pf >= 1000 else f"{pf:.0f}Hz"
            info += f" | src:{an.active} pk:{db_txt} rms:{rms_txt}dB f:{pf_txt}"
        except Exception:
            pass
        inf_w = fm.horizontalAdvance(info) + 8
        bar_x = rect.right() - bar_w - inf_w
        bar_y = rect.center().y() - 3
        lvl = max(0.0, min(1.0, float(getattr(an, 'level', 0.0) or 0.0)))
        painter.fillRect(QRect(bar_x, bar_y, bar_w, 6), QColor(40, 40, 40, 220))
        if lvl > 0.001:
            bw = int(bar_w * lvl)
            if lvl > 0.9:
                lcol = QColor(255, 60, 50, 255)
            elif lvl > 0.7:
                lcol = QColor(255, 200, 40, 255)
            else:
                lcol = QColor(40, 220, 120, 255)
            painter.fillRect(QRect(bar_x, bar_y, bw, 6), lcol)
        painter.setPen(QColor(170, 180, 190, 255))
        painter.drawText(QRect(bar_x + bar_w + 4, bar_y - 2, inf_w, 12),
                         Qt.AlignmentFlag.AlignVCenter, info)
        painter.restore()
    except Exception:
        pass


def draw_audio_freq_labels(vp, painter):
    if not getattr(vp, "_audio_viz_enabled", False):
        return
    an = getattr(vp, "_audio_analyzer", None)
    av = getattr(vp, "_audio_viz", None)
    if an is None or av is None:
        return
    try:
        fw, fh = vp._get_physical_dims()
        dpr = vp.devicePixelRatio() or 1.0
        x, y, w, h = av.panel_rect(fw, fh)
        lx, ly = x / dpr, y / dpr
        lw = w / dpr
        lh = h / dpr
        sr = getattr(an, "sample_rate", 0) or 48000
        nyquist = sr * 0.5
        if nyquist <= 20.0:
            return
        lmin = math.log10(20.0)
        lmax = math.log10(nyquist)
        spec_bottom = ly + lh * 0.52
        painter.save()
        font = QFont("Consolas", 7)
        font.setStyleStrategy(QFont.StyleStrategy.ForceOutline)
        painter.setFont(font)
        fm = QFontMetrics(font)
        ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        for f in ticks:
            if f < 20.0 or f > nyquist:
                continue
            t = (math.log10(f) - lmin) / (lmax - lmin)
            px = lx + t * lw
            if f >= 1000:
                label = f"{f // 1000}k"
            else:
                label = str(int(f))
            tw = fm.horizontalAdvance(label)
            rect = QRect(int(px - tw / 2 - 2), int(spec_bottom + 2), int(tw) + 4, fm.height())
            painter.setPen(QColor(160, 180, 200, 220))
            painter.drawLine(int(px), int(spec_bottom - 4), int(px), int(spec_bottom + 1))
            painter.fillRect(rect, QColor(0, 0, 0, 110))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
        painter.restore()
    except Exception:
        pass


def draw_audio_db_labels(vp, painter):
    if not getattr(vp, "_audio_viz_enabled", False):
        return
    an = getattr(vp, "_audio_analyzer", None)
    av = getattr(vp, "_audio_viz", None)
    if an is None or av is None:
        return
    try:
        fw, fh = vp._get_physical_dims()
        dpr = vp.devicePixelRatio() or 1.0
        x, y, w, h = av.panel_rect(fw, fh)
        lx, ly = x / dpr, y / dpr
        lw = w / dpr
        lh = h / dpr
        top_db = float(getattr(an, "spec_top_db", 0.0) or 0.0)
        floor_db = float(getattr(an, "spec_floor_db", -60.0) or -60.0)
        if floor_db >= top_db:
            return
        span = max(top_db - floor_db, 1e-3)
        spec_bottom = ly + lh * 0.52
        painter.save()
        font = QFont("Consolas", 7)
        font.setStyleStrategy(QFont.StyleStrategy.ForceOutline)
        painter.setFont(font)
        fm = QFontMetrics(font)
        db0 = int(math.floor(floor_db / 10.0) * 10.0)
        for db in range(db0, int(math.ceil(top_db)) + 1, 10):
            if db < floor_db - 0.01:
                continue
            v = (db - floor_db) / span
            py = spec_bottom - 2.0 - v * (spec_bottom - ly - 4.0)
            if py < ly + 1.0 or py > spec_bottom - 1.0:
                continue
            label = str(int(db))
            tw = fm.horizontalAdvance(label)
            txt_x = lx + 3.0
            txt_y = py - fm.height() * 0.5
            painter.setPen(QColor(120, 150, 170, 90))
            painter.drawLine(int(lx + 30), int(py), int(lx + lw - 2), int(py))
            painter.fillRect(QRect(int(txt_x), int(txt_y), int(tw) + 4, fm.height()),
                             QColor(0, 0, 0, 110))
            painter.setPen(QColor(150, 175, 195, 210))
            painter.drawText(QRect(int(txt_x + 2), int(txt_y), int(tw), fm.height()),
                             Qt.AlignmentFlag.AlignVCenter, label)
        painter.restore()
    except Exception:
        pass


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
