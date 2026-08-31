# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import gc as _gc
import subprocess
import threading
import time

import numpy as _np

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPen

_STATS_FONT = QFont("Consolas", 9)
_STATS_FONT.setStyleStrategy(QFont.StyleStrategy.ForceOutline)
_STATS_FM_CACHE: tuple = (None, None)
_RAM_CACHE = (0.0, 0.0)

VALUE_COLORS = {
    "FPS": QColor(100, 220, 100),
    "Avg": QColor(140, 230, 140),
    "1%": QColor(255, 200, 100),
    "0.1%": QColor(255, 150, 100),
    "Min": QColor(255, 120, 120),
    "CPU": QColor(100, 200, 255),
    "GPU": QColor(100, 200, 255),
    "Render": QColor(100, 200, 255),
    "Gizmos": QColor(160, 220, 255),
    "Overlay": QColor(190, 230, 255),
    "Paint": QColor(210, 210, 210),
    "RAM": QColor(180, 255, 180),
    "VRAM": QColor(255, 180, 255),
    "GC": QColor(180, 180, 255),
    "TPS": QColor(180, 255, 180),
    "TS": QColor(180, 255, 180),
    "DSP": QColor(255, 255, 180),
    "Sounds": QColor(180, 255, 255),
    "Entities": QColor(255, 180, 180),
    "Draw": QColor(200, 200, 200),
    "Tris": QColor(200, 220, 255),
    "Verts": QColor(200, 220, 255),
    "Fill": QColor(200, 255, 220),
    "Cull": QColor(255, 200, 150),
    "Batches": QColor(200, 200, 200),
    "Inst": QColor(200, 200, 200),
    "Particles": QColor(255, 220, 180),
    "Draws": QColor(200, 220, 255),
    "Lines": QColor(200, 220, 255),
    "Mesh": QColor(200, 220, 255),
    "Upload": QColor(255, 255, 150),
    "Full": QColor(255, 180, 180),
    "Part": QColor(180, 255, 180),
    "Res": QColor(180, 200, 255),
    "GPU": QColor(200, 200, 200),
    "GL": QColor(200, 200, 200),
    "MRays": QColor(255, 200, 100),
}


def _fmt_count(n) -> str:
    try:
        n = int(n)
    except Exception:
        n = 0
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _fmt_bytes(b) -> str:
    try:
        b = int(b)
    except Exception:
        b = 0
    if b >= 1 << 20:
        return f"{b / (1 << 20):.1f}MB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):.1f}KB"
    return f"{b}B"


def _fmt_vram_mb(mb) -> str:
    try:
        mb = float(mb)
    except Exception:
        return "0MB"
    if mb >= 2048:
        return f"{mb / 1024.0:.1f}GB"
    return f"{mb:.0f}MB"


def _short_gpu(name) -> str:
    name = str(name).split("/")[0].strip()
    for prefix in ("NVIDIA GeForce ", "AMD Radeon ", "Intel "):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name or "?"


_RAM_CACHE_VAL = 0.0
_RAM_CACHE_T = 0.0

def _get_ram_mb() -> float:
    global _RAM_CACHE_VAL, _RAM_CACHE_T
    now = time.time()
    if now - _RAM_CACHE_T < 1.0 and _RAM_CACHE_VAL > 0:
        return _RAM_CACHE_VAL
    try:
        import psutil as _psutil
        v = _psutil.Process().memory_info().rss / (1024 * 1024)
        _RAM_CACHE_VAL = v
        _RAM_CACHE_T = now
        return v
    except Exception:
        pass
    try:
        import resource as _resource
        v = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024
        _RAM_CACHE_VAL = v
        _RAM_CACHE_T = now
        return v
    except Exception:
        pass
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
        pmc = _PMC()
        pmc.cb = _ctypes.sizeof(_PMC)
        h = _ctypes.c_void_p(-1)
        if _psapi.GetProcessMemoryInfo(h, _ctypes.byref(pmc), _ctypes.sizeof(pmc)):
            v = pmc.WorkingSetSize / (1024 * 1024)
            _RAM_CACHE_VAL = v
            _RAM_CACHE_T = now
            return v
    except Exception:
        pass
    return _RAM_CACHE_VAL if _RAM_CACHE_VAL else 0.0


def _query_vram_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        line = out.stdout.strip().splitlines()[0]
        used_s, total_s = line.split(",")
        used = float(used_s.strip().split()[0])
        total = float(total_s.strip().split()[0])
        if total > 0:
            return used, total
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=2,
        )
        used = total = 0.0
        for ln in out.stdout.splitlines():
            ln = ln.strip()
            if "Used memory" in ln:
                used = float(ln.split(":")[1].strip().split()[0])
            elif "Total memory" in ln:
                total = float(ln.split(":")[1].strip().split()[0])
        if total > 0:
            return used, total
    except Exception:
        pass
    return 0.0, 0.0


_expensive = {}
_expensive_t = 0.0
_vram = (0.0, 0.0)
_vram_t = 0.0
_vram_busy = False
_frame_metrics_cache: dict = {"key": None, "val": None}


def _refresh_vram_async():
    global _vram_busy
    if _vram_busy:
        return
    _vram_busy = True

    def worker():
        global _vram, _vram_t, _vram_busy
        try:
            _vram = _query_vram_mb()
        except Exception:
            pass
        _vram_t = time.time()
        _vram_busy = False

    try:
        threading.Thread(target=worker, daemon=True).start()
    except Exception:
        _vram_busy = False


def collect_expensive_stats() -> dict:
    global _expensive_t
    now = time.time()
    if _expensive and (now - _expensive_t) < 10.0:
        return _expensive
    _expensive_t = now
    if (now - _vram_t) > 10.0:
        _refresh_vram_async()
    vram_used, vram_total = _vram
    try:
        gc0, gc1, gc2 = _gc.get_count()
    except Exception:
        gc0 = gc1 = gc2 = 0
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
    _expensive.update({
        'ram_mb': _get_ram_mb(),
        'vram_used': vram_used,
        'vram_total': vram_total,
        'gc0': gc0, 'gc1': gc1, 'gc2': gc2,
        'dsp_load': dsp_load,
        'active_sounds': active_sounds,
        'total_sounds': total_sounds,
    })
    return _expensive


def collect_render_stats(engine, renderer) -> dict:
    st = {}
    for out_key, attr in (
            ('draw_calls', '_draw_calls'),
            ('triangles', '_triangles_drawn'),
            ('vertices', '_vertices_drawn'),
            ('particles', '_particle_count'),
            ('culled_visible', '_culled_visible'),
            ('culled_total', '_culled_total')):
        st[out_key] = getattr(renderer, attr, 0) or 0
    batcher = getattr(renderer, '_batcher', None)
    st['batches'] = batcher.batches if batcher is not None else 0
    st['instanced'] = batcher.instanced if batcher is not None else 0
    gizmo = getattr(renderer, '_gizmo', None)
    gizmo_keys = ('gizmo_draws', 'gizmo_lines', 'gizmo_instances', 'gizmo_mesh_verts',
                  'gizmo_upload_bytes', 'gizmo_upload_full', 'gizmo_upload_partial')
    if gizmo is not None:
        for gk, attr in zip(gizmo_keys, ('_stat_draws', '_stat_lines', '_stat_instances',
                                         '_stat_mesh_verts', '_stat_upload_bytes',
                                         '_stat_upload_full', '_stat_upload_partial')):
            st[gk] = getattr(gizmo, attr, 0) or 0
    else:
        for gk in gizmo_keys:
            st[gk] = 0
    st['tps'] = getattr(engine, 'tps', 0.0) or 0.0
    st['time_scale'] = getattr(engine, 'time_scale', 1.0) or 1.0
    st['entities'] = 0
    scene = getattr(engine, 'scene', None)
    if scene is not None:
        try:
            st['entities'] = len(scene.get_all_entities())
        except Exception:
            pass
    ctx = getattr(renderer, '_ctx', None)
    st['gl_renderer'] = ''
    st['gl_version'] = ''
    if ctx is not None:
        info = getattr(ctx, 'info', None) or {}
        st['gl_renderer'] = str(info.get('GL_RENDERER', ''))
        st['gl_version'] = str(info.get('GL_VERSION', ''))
    st['rt_rays_per_frame'] = getattr(renderer, '_rt_rays_per_frame', 0)
    st.update(collect_expensive_stats())
    return st


def compute_frame_metrics(frame_times_ms) -> dict:
    if not frame_times_ms:
        return {
            'fps': 0.0, 'avg_fps': 0.0, 'max_fps': 0.0, 'min_fps': 0.0,
            'p1_fps': 0.0, 'p01_fps': 0.0, 'frame_ms': 0.0, 'avg_ms': 0.0,
            'p1_ms': 0.0, 'p01_ms': 0.0,
        }
    n = len(frame_times_ms)
    key = (n, frame_times_ms[-1] if n else 0, round(sum(frame_times_ms) * 0.1))
    cached = _frame_metrics_cache.get("key")
    if cached == key and _frame_metrics_cache.get("val") is not None:
        return _frame_metrics_cache["val"]
    try:
        arr = _np.asarray(frame_times_ms, dtype=_np.float32)
        s = _np.sort(arr)
        avg_ms = float(_np.mean(s))
        p1_c = max(1, int(n * 0.01))
        p01_c = max(1, int(n * 0.001))
        p1_ms = float(_np.mean(s[-p1_c:]))
        p01_ms = float(_np.mean(s[-p01_c:]))
        max_ms = float(s[0])
        min_ms = float(s[-1])
    except Exception:
        s = sorted(frame_times_ms)
        avg_ms = sum(s) / n
        p1_c = max(1, int(n * 0.01))
        p01_c = max(1, int(n * 0.001))
        p1_ms = sum(s[-p1_c:]) / p1_c
        p01_ms = sum(s[-p01_c:]) / p01_c
        max_ms = s[0]
        min_ms = s[-1]
    val = {
        'fps': 1000.0 / max(avg_ms, 0.1),
        'avg_fps': 1000.0 / max(avg_ms, 0.1),
        'max_fps': 1000.0 / max(max_ms, 0.1),
        'min_fps': 1000.0 / max(min_ms, 0.1),
        'p1_fps': 1000.0 / max(p1_ms, 0.1),
        'p01_fps': 1000.0 / max(p01_ms, 0.1),
        'frame_ms': frame_times_ms[-1],
        'avg_ms': avg_ms,
        'p1_ms': p1_ms,
        'p01_ms': p01_ms,
    }
    _frame_metrics_cache["key"] = key
    _frame_metrics_cache["val"] = val
    return val


def build_stats_rows(m: dict, st: dict, timings: dict) -> list:
    cull_total = st['culled_total']
    cull_str = f"{st['culled_visible']}/{cull_total}"
    if cull_total > 0:
        cull_str += f" ({100.0 * st['culled_visible'] / cull_total:.0f}%)"
    fill_mts = st['triangles'] * m['fps'] / 1e6
    frame_kvs = [
        ("FPS", f"{m['fps']:.1f}", "FPS"),
        ("1%", f"{m['p1_fps']:.1f}", "1%"),
        ("0.1%", f"{m['p01_fps']:.1f}", "0.1%"),
        ("Min", f"{m['min_fps']:.1f}", "Min"),
        ("CPU", f"{timings['cpu_ms']:.1f}ms", "CPU"),
        ("GPU", f"{timings['render_ms']:.1f}ms", "GPU"),
    ]
    if st['rt_rays_per_frame'] > 0:
        rt_mrays = st['rt_rays_per_frame'] * m['fps'] / 1e6
        frame_kvs.append(("MRays", f"{rt_mrays:.1f}MR/s", "MRays"))
    return [
        ("h", "Frame"),
        ("kv", frame_kvs),
        ("h", "Timing"),
        ("kv", [
            ("Render", f"{timings['render_ms']:.2f}ms", "Render"),
            ("Gizmos", f"{timings['gizmo_ms']:.2f}ms", "Gizmos"),
            ("Overlay", f"{timings['overlay_ms']:.2f}ms", "Overlay"),
            ("Paint", f"{timings['paint_ms']:.2f}ms", "Paint"),
        ]),
        ("h", "Scene"),
        ("kv", [
            ("Draw", f"{st['draw_calls']}", "Draw"),
            ("Tris", f"{_fmt_count(st['triangles'])}", "Tris"),
            ("Verts", f"{_fmt_count(st['vertices'])}", "Verts"),
            ("Fill", f"{fill_mts:.0f}MT/s", "Fill"),
            ("Cull", cull_str, "Cull"),
        ]),
        ("h", "Batches"),
        ("kv", [
            ("Batches", f"{st['batches']}", "Batches"),
            ("Inst", f"{_fmt_count(st['instanced'])}", "Inst"),
            ("Particles", f"{_fmt_count(st['particles'])}", "Particles"),
        ]),
        ("h", "Gizmo"),
        ("kv", [
            ("Draws", f"{st['gizmo_draws']}", "Draws"),
            ("Lines", f"{_fmt_count(st['gizmo_lines'])}", "Lines"),
            ("Inst", f"{_fmt_count(st['gizmo_instances'])}", "Inst"),
            ("Mesh", f"{_fmt_count(st['gizmo_mesh_verts'])}", "Mesh"),
            ("Upload", f"{_fmt_bytes(st['gizmo_upload_bytes'])}", "Upload"),
            ("Full", f"{st['gizmo_upload_full']}", "Full"),
            ("Part", f"{st['gizmo_upload_partial']}", "Part"),
        ]),
        ("h", "Memory"),
        ("kv", [
            ("RAM", f"{st['ram_mb']:.0f}MB", "RAM"),
            ("VRAM", f"{_fmt_vram_mb(st['vram_used'])}/{_fmt_vram_mb(st['vram_total'])}", "VRAM"),
            ("GC", f"{st['gc0']}/{st['gc1']}/{st['gc2']}", "GC"),
        ]),
        ("h", "Engine"),
        ("kv", [
            ("TPS", f"{st['tps']:.0f}", "TPS"),
            ("TS", f"{st['time_scale']:.2f}", "TS"),
            ("Entities", f"{st['entities']}", "Entities"),
            ("DSP", f"{st['dsp_load']:.0f}%", "DSP"),
            ("Sounds", f"{st['active_sounds']}/{st['total_sounds']}", "Sounds"),
        ]),
        ("h", "Device"),
        ("kv", [
            ("GPU", _short_gpu(st['gl_renderer']), "GPU"),
            ("GL", st['gl_version'].split()[0] if st['gl_version'] else "?", "GL"),
            ("Res", timings['res'], "Res"),
        ]),
    ]


def draw_stats_panel(painter, rows: list, frame_times_ms, spike_log: list):
    fnt = painter.font()
    if fnt.family() != _STATS_FONT.family() or fnt.pointSize() != _STATS_FONT.pointSize():
        painter.setFont(_STATS_FONT)
    fm = QFontMetrics(_STATS_FONT)
    padding = 6
    line_h = 15
    sections = []
    cur = None
    for row in rows:
        if row[0] == "h":
            cur = [row[1], []]
            sections.append(cur)
        else:
            if cur is not None:
                cur[1].extend(row[1])
    seg_widths = []
    for hdr, kvs in sections:
        seg = f"{hdr}: " + "  |  ".join(f"{lab}: {val}" for lab, val, _ in kvs)
        seg_widths.append(fm.horizontalAdvance(seg))
    max_w = max(seg_widths, default=0) + padding * 2
    max_w = max(max_w, 480)
    view_w = painter.device().width()
    if max_w > view_w - 16:
        max_w = max(view_w - 16, 120)
    x = 8
    y = 35
    total_h = len(sections) * line_h + padding * 2
    view_h = painter.device().height()
    show_chart = True
    show_spike = bool(spike_log)
    while (y + total_h + (36 if show_chart else 0) + (72 if show_spike else 0)) > view_h:
        if show_spike:
            show_spike = False
        elif show_chart:
            show_chart = False
        else:
            break
    bg = QColor(0, 0, 0, 160)
    border = QColor(80, 80, 80, 200)
    rect = QRect(x, y, int(max_w), total_h)
    painter.fillRect(rect, bg)
    painter.setPen(QPen(border, 1))
    painter.drawRect(rect)
    label_color = QColor(160, 160, 160)
    header_color = QColor(120, 180, 255)
    text_color = QColor(255, 255, 255)
    for i, (hdr, kvs) in enumerate(sections):
        cy = y + padding + i * line_h
        cx = x + padding
        hs = hdr + ": "
        painter.setPen(header_color)
        painter.drawText(QRect(cx, cy, fm.horizontalAdvance(hs), line_h),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, hs)
        cx += fm.horizontalAdvance(hs)
        for ki, (lab, val, key) in enumerate(kvs):
            lab_s = lab + ": "
            painter.setPen(label_color)
            painter.drawText(QRect(cx, cy, fm.horizontalAdvance(lab_s), line_h),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, lab_s)
            cx += fm.horizontalAdvance(lab_s)
            painter.setPen(VALUE_COLORS.get(key, text_color))
            painter.drawText(QRect(cx, cy, fm.horizontalAdvance(val), line_h),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, val)
            cx += fm.horizontalAdvance(val)
            if ki < len(kvs) - 1:
                painter.setPen(QColor(100, 100, 100))
                sep = " | "
                painter.drawText(QRect(cx, cy, fm.horizontalAdvance(sep), line_h),
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, sep)
                cx += fm.horizontalAdvance(sep)
        if i < len(sections) - 1:
            painter.setPen(QPen(QColor(255, 255, 255, 24), 1))
            painter.drawLine(x + padding, cy + line_h - 3, int(x + max_w - padding), cy + line_h - 3)
    if show_chart:
        chart_bottom = draw_frame_chart(painter, frame_times_ms, x, y + total_h + 6, int(max_w))
        if show_spike:
            draw_spike_box(painter, spike_log[-1], x, chart_bottom + 6, int(max_w))


def draw_frame_chart(painter, ft_list, x: int, y: int, w: int) -> int:
    chart_h = 30
    bg = QColor(0, 0, 0, 160)
    border = QColor(80, 80, 80, 200)
    chart_rect = QRect(x, y, w, chart_h)
    painter.fillRect(chart_rect, bg)
    painter.setPen(QPen(border, 1))
    painter.drawRect(chart_rect)
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
                color = QColor(255, 80, 80, 180)
            elif ft_val > 16.0:
                color = QColor(255, 200, 80, 160)
            else:
                color = QColor(80, 200, 80, 140)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(QRect(int(bar_x), bar_y, max(1, int(bar_w)), bh))
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        ref_y = chart_rect.bottom() - 2 - int((16.0 / max_ft) * (chart_h - 4))
        if ref_y > chart_rect.y() + 2:
            painter.drawLine(chart_rect.x() + 2, ref_y, chart_rect.right() - 2, ref_y)
    return chart_rect.bottom()


def draw_spike_box(painter, spike, x: int, y: int, w: int):
    line_h = 15
    padding = 6
    spike_ft, spike_prof = spike
    lines = [f"Spike: {spike_ft:.0f}ms  frame"]
    sorted_spike = sorted(spike_prof.items(), key=lambda kv: -kv[1])[:3]
    for sp_name, sp_val in sorted_spike:
        lines.append(f"  {sp_name}: {sp_val:.1f}ms")
    spike_h = len(lines) * line_h + padding * 2
    rect = QRect(x, y, w, spike_h)
    painter.fillRect(rect, QColor(60, 20, 20, 200))
    painter.setPen(QPen(QColor(255, 80, 80, 200), 1))
    painter.drawRect(rect)
    for si, sline in enumerate(lines):
        sy = y + padding + si * line_h
        painter.setPen(QColor(255, 200, 200))
        painter.drawText(QRect(x + padding, sy, w, line_h),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, sline)


_SPIKE_LOG: list = []
_last_spike_cumulative: dict = {}


def log_spike(frame_time_ms: float, prof):
    per_frame: dict = {}
    cf = getattr(prof, '_current_frame', None) if prof else None
    if cf is not None and cf.flat_data:
        per_frame = dict(cf.flat_data)
    else:
        frames = getattr(prof, 'frames', None)
        if frames:
            per_frame = dict(frames[-1].flat_data)
    if not per_frame:
        global _last_spike_cumulative
        cur = dict(prof.data) if prof else {}
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
