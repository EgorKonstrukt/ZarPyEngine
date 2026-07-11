# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import gc as _gc
import math
import os
import time

import numpy as np
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPen

from core.math.math3d import Vec3

try:
    import psutil as _psutil
    _psutil_available = True
except Exception:
    _psutil_available = False

try:
    import ctypes as _ctypes

    class _PMC(_ctypes.Structure):
        _fields_ = [
            ('cb', _ctypes.c_uint32),
            ('PageFaultCount', _ctypes.c_uint32),
            ('PeakWorkingSetSize', _ctypes.c_size_t),
            ('WorkingSetSize', _ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage', _ctypes.c_size_t),
            ('QuotaPagedPoolUsage', _ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage', _ctypes.c_size_t),
            ('QuotaNonPagedPoolUsage', _ctypes.c_size_t),
            ('PagefileUsage', _ctypes.c_size_t),
            ('PeakPagefileUsage', _ctypes.c_size_t),
        ]

    _psapi = _ctypes.windll.psapi
    _psapi.GetProcessMemoryInfo.argtypes = [_ctypes.c_void_p, _ctypes.c_void_p, _ctypes.c_uint32]
    _psapi.GetProcessMemoryInfo.restype = _ctypes.c_int
    _ctypes_available = True
except Exception:
    _ctypes_available = False
    _PMC = None
    _psapi = None


def _get_ram_mb() -> float:
    if _psutil_available:
        try:
            return _psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            pass
    try:
        import resource as _resource
        return _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        pass
    if _ctypes_available:
        try:
            pmc = _PMC()
            pmc.cb = _ctypes.sizeof(_PMC)
            h = _ctypes.c_void_p(-1)
            if _psapi.GetProcessMemoryInfo(h, _ctypes.byref(pmc), _ctypes.sizeof(pmc)):
                return pmc.WorkingSetSize / (1024 * 1024)
        except Exception:
            pass
    return 0.0


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


_vram_cache = {'used_mb': 0.0, 'total_mb': 0.0, 'last_query': 0.0}


try:
    import ctypes as _nvml_ctypes

    class _NVMLMem(_nvml_ctypes.Structure):
        _fields_ = [
            ('total', _nvml_ctypes.c_ulonglong),
            ('free', _nvml_ctypes.c_ulonglong),
            ('used', _nvml_ctypes.c_ulonglong),
        ]

    _nvml_handle = _nvml_ctypes.WinDLL('nvml.dll')
    _nvml_handle.nvmlInit()
    _nvml_dev_handle = _nvml_ctypes.c_void_p()
    _nvml_handle.nvmlDeviceGetHandleByIndex(0, _nvml_ctypes.byref(_nvml_dev_handle))
    _nvml_available = True
except Exception:
    _nvml_available = False
    _nvml_dev_handle = None
    _NVMLMem = None


def _get_vram_mb() -> tuple[float, float]:
    global _vram_cache
    now = time.time()
    if now - _vram_cache['last_query'] < 2.0:
        return _vram_cache['used_mb'], _vram_cache['total_mb']

    used_mb = total_mb = 0.0
    if _nvml_available and _nvml_dev_handle is not None:
        try:
            mem = _NVMLMem()
            _nvml_handle.nvmlDeviceGetMemoryInfo(
                _nvml_dev_handle, _nvml_ctypes.byref(mem))
            total_mb = mem.total / (1024 * 1024)
            used_mb = mem.used / (1024 * 1024)
        except Exception:
            pass

    _vram_cache = {'used_mb': used_mb, 'total_mb': total_mb, 'last_query': now}
    return used_mb, total_mb


_SPIKE_LOG: list[tuple[float, dict[str, float]]] = []
_last_spike_cumulative: dict[str, float] = {}


def _log_spike(frame_time_ms: float, prof):
    per_frame: dict[str, float] = {}
    cf = prof._current_frame
    if cf is not None and cf.flat_data:
        per_frame = dict(cf.flat_data)
    else:
        frames = prof.frames
        if frames:
            per_frame = dict(frames[-1].flat_data)

    if not per_frame:
        global _last_spike_cumulative
        cur = dict(prof.data)
        if _last_spike_cumulative:
            for k, v in cur.items():
                prev = _last_spike_cumulative.get(k, 0.0)
                if isinstance(v, (int, float)) and isinstance(prev, (int, float)):
                    delta = v - prev
                    if delta > 0.01:
                        per_frame[k] = delta
        _last_spike_cumulative = cur

    _SPIKE_LOG.append((frame_time_ms, per_frame))
    if len(_SPIKE_LOG) > 20:
        _SPIKE_LOG.pop(0)


def _draw_val(painter, label, val, fm, cx, cy, line_height):
    color_map = {
        "FPS": QColor(100, 220, 100),
        "1%": QColor(255, 200, 100),
        "0.1%": QColor(255, 150, 100),
        "CPU": QColor(100, 200, 255),
        "GPU": QColor(100, 200, 255),
        "RAM": QColor(180, 255, 180),
        "VRAM": QColor(255, 180, 255),
        "GC": QColor(180, 180, 255),
        "TPS": QColor(180, 255, 180),
        "DSP": QColor(255, 255, 180),
        "Sounds": QColor(180, 255, 255),
        "Entities": QColor(255, 180, 180),
        "Draw Calls": QColor(200, 200, 200),
        "Tris": QColor(200, 220, 255),
        "Verts": QColor(200, 220, 255),
        "Batches": QColor(200, 200, 200),
        "Instanced": QColor(200, 200, 200),
        "Gizmo Draws": QColor(200, 220, 255),
        "GLines": QColor(200, 220, 255),
        "Res": QColor(180, 200, 255),
        "Culled": QColor(255, 200, 150),
        "TS": QColor(180, 255, 180),
    }
    c = color_map.get(label, QColor(255, 255, 255))
    painter.setPen(c)
    painter.drawText(QRect(cx, cy, fm.horizontalAdvance(val), line_height),
                     Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, val)


def draw_stats_overlay(vp, painter):
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    font = QFont("Consolas", 9)
    font.setStyleStrategy(QFont.StyleStrategy.ForceOutline)
    painter.setFont(font)

    fps = vp._fps if vp._fps > 0 else 0.0
    paint_dt = getattr(vp, '_paint_dt', 0.016)

    if not hasattr(vp, '_frame_times_ms'):
        vp._frame_times_ms = []

    if paint_dt > 0:
        vp._frame_times_ms.append(paint_dt * 1000.0)
        if len(vp._frame_times_ms) > 300:
            vp._frame_times_ms.pop(0)

    if vp._frame_times_ms:
        current_ft = vp._frame_times_ms[-1]
        if current_ft > 33.0:
            prof = getattr(vp._engine, '_profiler', None)
            if prof and prof.enabled:
                _log_spike(current_ft, prof)

    sorted_ft = sorted(vp._frame_times_ms)
    n = len(sorted_ft)

    p1_count = max(1, int(n * 0.01))
    p01_count = max(1, int(n * 0.001))
    p1_low = sum(sorted_ft[-p1_count:]) / p1_count if sorted_ft else 0.0
    p01_low = sum(sorted_ft[-p01_count:]) / p01_count if sorted_ft else 0.0

    cpu_ms = paint_dt * 1000.0

    tps = vp._engine.tps if hasattr(vp._engine, 'tps') else 0.0

    ram_mb = _get_ram_mb()
    vram_used, vram_total = _get_vram_mb()

    gc_gen0, gc_gen1, gc_gen2 = _gc.get_count()

    dsp_load = 0.0
    active_sounds = 0
    total_sounds = 0
    try:
        from core.audio.audio_system import AudioSourceManager
        mgr = AudioSourceManager.instance()
        if mgr:
            dsp_load = mgr.get_dsp_load()
            active_sounds = mgr.get_active_sound_count()
            total_sounds = mgr.get_total_sound_count()
    except Exception:
        pass

    entities = len(vp._engine.scene.get_all_entities()) if vp._engine.scene else 0
    triangles = vp._renderer._triangles_drawn if hasattr(vp._renderer, '_triangles_drawn') else 0
    vertices = vp._renderer._vertices_drawn if hasattr(vp._renderer, '_vertices_drawn') else 0
    draw_calls = vp._renderer._draw_calls if hasattr(vp._renderer, '_draw_calls') else 0

    gpu_ms = vp._last_render_ms if hasattr(vp, '_last_render_ms') else 0.0

    fw, fh = vp._get_physical_dims()

    culled_visible = vp._renderer._culled_visible if hasattr(vp._renderer, '_culled_visible') else 0
    culled_total = vp._renderer._culled_total if hasattr(vp._renderer, '_culled_total') else 0
    culled_str = f"{culled_visible}/{culled_total}"

    time_scale = vp._engine.time_scale if hasattr(vp._engine, 'time_scale') else 1.0

    batches = 0
    instanced = 0
    if hasattr(vp._renderer, '_batcher') and vp._renderer._batcher:
        batches = vp._renderer._batcher.batches
        instanced = vp._renderer._batcher.instanced

    gizmo_lines = vp._renderer._gizmo._stat_lines if vp._renderer._gizmo else 0
    gizmo_draws = vp._renderer._gizmo._stat_draws if vp._renderer._gizmo else 0

    particles = vp._renderer._particle_count if hasattr(vp._renderer, '_particle_count') else 0

    stats_lines = [
        f"FPS: {fps:.1f}  |  1%: {1000.0/max(p1_low,0.1):.1f}  |  0.1%: {1000.0/max(p01_low,0.1):.1f}  |  CPU: {cpu_ms:.1f}ms  |  GPU: {gpu_ms:.1f}ms  |  Res: {fw}x{fh}",
        f"RAM: {ram_mb:.0f} MB  |  VRAM: {vram_used:.0f}/{vram_total:.0f} MB  |  GC: {gc_gen0}/{gc_gen1}/{gc_gen2}  |  TPS: {tps:.0f}  |  TS: {time_scale:.2f}",
        f"DSP: {dsp_load:.0f}%  |  Sounds: {active_sounds}/{total_sounds}",
        f"Entities: {entities}  |  Culled: {culled_str}  |  Draw Calls: {draw_calls}  |  Tris: {_fmt_count(triangles)}  |  Verts: {_fmt_count(vertices)}",
        f"Particles: {_fmt_count(particles)}  |  Batches: {batches}  |  Instanced: {instanced}  |  Gizmo Draws: {gizmo_draws}  |  GLines: {_fmt_count(gizmo_lines)}",
    ]

    text_color = QColor(255, 255, 255)
    label_color = QColor(160, 160, 160)
    bg_color = QColor(0, 0, 0, 160)
    border_color = QColor(80, 80, 80, 200)
    padding = 6
    line_height = 15
    total_h = len(stats_lines) * line_height + padding * 2

    fm = QFontMetrics(font)
    max_w = max(fm.horizontalAdvance(line) for line in stats_lines) + padding * 2
    max_w = max(max_w, 460)

    x = 8
    y = 35
    rect = QRect(x, y, int(max_w), total_h)
    painter.fillRect(rect, bg_color)
    painter.setPen(QPen(border_color, 1))
    painter.drawRect(rect)

    painter.setFont(font)
    for i, line in enumerate(stats_lines):
        cx = x + padding
        cy = y + padding + i * line_height
        segments = line.split("  |  ")
        for idx, seg in enumerate(segments):
            seg = seg.strip()
            if not seg:
                continue
            if ":" in seg:
                lab, val = seg.split(":", 1)
                lab = lab.strip() + ": "
                val = val.strip()
                painter.setPen(label_color)
                painter.drawText(QRect(cx, cy, fm.horizontalAdvance(lab), line_height),
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, lab)
                cx += fm.horizontalAdvance(lab)
                _draw_val(painter, lab.rstrip(": "), val, fm, cx, cy, line_height)
                cx += fm.horizontalAdvance(val)
            else:
                painter.setPen(text_color)
                painter.drawText(QRect(cx, cy, fm.horizontalAdvance(seg), line_height),
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, seg)
                cx += fm.horizontalAdvance(seg)
            if idx < len(segments) - 1:
                painter.setPen(QColor(100, 100, 100))
                sep = " | "
                painter.drawText(QRect(cx, cy, fm.horizontalAdvance(sep), line_height),
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, sep)
                cx += fm.horizontalAdvance(sep)

    # в”Ђв”Ђ frame time bar chart в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
    chart_y = y + total_h + 6
    chart_h = 30
    chart_rect = QRect(x, chart_y, int(max_w), chart_h)
    painter.fillRect(chart_rect, bg_color)
    painter.setPen(QPen(border_color, 1))
    painter.drawRect(chart_rect)
    ft_list = vp._frame_times_ms
    n_bars = min(len(ft_list), chart_rect.width() - 4)
    if n_bars > 1:
        bar_w = (chart_rect.width() - 4) / n_bars
        max_ft = max(max(ft_list[-n_bars:]) * 1.1, 16.0)
        for bi in range(n_bars):
            ft_val = ft_list[-n_bars + bi]
            bh = max(1, int((ft_val / max_ft) * (chart_h - 4)))
            bar_x = chart_rect.x() + 2 + int(bar_w * bi)
            bar_y = chart_rect.bottom() - 2 - bh
            if ft_val > 33.0:
                painter.setBrush(QBrush(QColor(255, 80, 80, 180)))
            elif ft_val > 16.0:
                painter.setBrush(QBrush(QColor(255, 200, 80, 160)))
            else:
                painter.setBrush(QBrush(QColor(80, 200, 80, 140)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(QRect(int(bar_x), bar_y, max(1, int(bar_w)), bh))
        # reference lines
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        ref_y = chart_rect.bottom() - 2 - int((16.0 / max_ft) * (chart_h - 4))
        if ref_y > chart_rect.y() + 2:
            painter.drawLine(chart_rect.x() + 2, ref_y, chart_rect.right() - 2, ref_y)

    # в”Ђв”Ђ last spike detail в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
    if _SPIKE_LOG:
        spike_ft, spike_prof = _SPIKE_LOG[-1]
        spike_lines = [f"Spike: {spike_ft:.0f}ms  frame"]
        sorted_spike = sorted(spike_prof.items(), key=lambda kv: -kv[1])[:3]
        for sp_name, sp_val in sorted_spike:
            spike_lines.append(f"  {sp_name}: {sp_val:.1f}ms")
        spike_y = chart_y + chart_h + 6
        spike_h = len(spike_lines) * line_height + padding * 2
        spike_rect = QRect(x, spike_y, int(max_w), spike_h)
        painter.fillRect(spike_rect, QColor(60, 20, 20, 200))
        painter.setPen(QPen(QColor(255, 80, 80, 200), 1))
        painter.drawRect(spike_rect)
        painter.setFont(font)
        for si, sline in enumerate(spike_lines):
            sx = x + padding
            sy = spike_y + padding + si * line_height
            painter.setPen(QColor(255, 200, 200))
            painter.drawText(QRect(sx, sy, int(max_w), line_height),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, sline)


def draw_delta_label(vp, painter):
    dt = vp._gizmo.delta_text
    if not dt or not vp._gizmo.show_delta_label:
        return
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


def draw_grid_labels(vp, painter=None):
    x_labels, z_labels = get_grid_label_positions(vp)
    if not x_labels and not z_labels:
        return
    own_painter = False
    if painter is None:
        painter = QPainter(vp)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        own_painter = True
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
    if own_painter:
        painter.end()
