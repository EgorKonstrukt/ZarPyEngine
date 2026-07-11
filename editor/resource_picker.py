# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import json
import math
import wave
import struct
import threading
import numpy as np
from typing import Optional, Callable
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit,
                              QListWidget, QListWidgetItem, QLabel,
                              QPushButton, QWidget, QSplitter, QFileDialog,
                              QListWidgetItem, QAbstractItemView, QListView)
from PyQt6.QtCore import Qt, QSize, QRect, QRectF, QPoint, QThread, pyqtSignal, QTimer, QMutex, QCoreApplication
from PyQt6.QtGui import (QFont, QPixmap, QPainter, QPainterPath, QColor, QPen, QBrush,
                         QFontMetrics, QLinearGradient, QRadialGradient, QIcon, QPalette,
                         QImageReader, QPolygonF, QImage)

_thumbnail_cache: dict[str, QPixmap] = {}
_thumbnail_mutex = QMutex()

from editor.constants import THUMB_SIZE, PREVIEW_SIZE
from core.config.editor_scale import scale, scale_xy

_placeholder_icon: Optional[QIcon] = None


def _get_placeholder_icon() -> QIcon:
    global _placeholder_icon
    if _placeholder_icon is None:
        _placeholder_icon = QIcon(_draw_file_icon(THUMB_SIZE))
    return _placeholder_icon


EXTENSION_FILTERS = {
    "Models (*.obj *.fbx *.stl *.gltf *.glb *.usdz)": (".obj", ".fbx", ".stl", ".gltf", ".glb", ".usdz"),
    "Audio (*.wav *.mp3 *.ogg)": (".wav", ".mp3", ".ogg"),
    "Python Scripts (*.py)": (".py",),
    "Fonts (*.ttf *.otf)": (".ttf", ".otf"),
    "Animation Clips (*.animclip)": (".animclip",),
    "Animator Controllers (*.animcontroller)": (".animcontroller",),
    "All Files (*)": (),
}

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _make_icon_bg(base_color: QColor, size: int) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    grad = QLinearGradient(0, 0, size, size)
    grad.setColorAt(0.0, base_color.lighter(140))
    grad.setColorAt(1.0, base_color)
    p.setBrush(QBrush(grad))
    pen = QPen(base_color.darker(130), 1)
    p.setPen(pen)
    r = size * 0.15
    p.drawRoundedRect(QRect(1, 1, size - 2, size - 2), r, r)
    p.end()
    return pm

def _draw_text_centered(painter: QPainter, text: str, rect: QRect, color: QColor):
    painter.setPen(color)
    f = painter.font()
    f.setPixelSize(int(rect.height() * 0.45))
    f.setBold(True)
    painter.setFont(f)
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

_mesh_thumb_cache: dict[str, QPixmap] = {}
_mesh_thumb_mutex = QMutex()


class _MeshDataLoader(QThread):
    mesh_data_ready = pyqtSignal(str, int, object, object)

    def __init__(self):
        super().__init__()
        self._queue: list[tuple[str, int]] = []
        self._cancelled = False
        self._lock = QMutex()

    def cancel(self):
        self._cancelled = True

    def enqueue(self, path: str, size: int):
        self._lock.lock()
        if (path, size) not in self._queue:
            self._queue.append((path, size))
        self._lock.unlock()

    def run(self):
        from core.assets.asset_importer import load_mesh
        while not self._cancelled:
            self._lock.lock()
            work = self._queue.pop(0) if self._queue else None
            self._lock.unlock()
            if work is None:
                self.msleep(20)
                continue
            path, size = work
            cache_key = f"mesh:{path}:{size}"
            _mesh_thumb_mutex.lock()
            if cache_key in _mesh_thumb_cache:
                _mesh_thumb_mutex.unlock()
                continue
            _mesh_thumb_mutex.unlock()
            try:
                data = load_mesh(path)
                if data is None or len(data.vertices) < 3 or len(data.indices) < 3:
                    continue
                self.mesh_data_ready.emit(path, size, data.vertices, data.indices)
            except Exception:
                continue


_mesh_loader: Optional[_MeshDataLoader] = None


def _get_mesh_loader() -> _MeshDataLoader:
    global _mesh_loader
    if _mesh_loader is None:
        _mesh_loader = _MeshDataLoader()
        _mesh_loader.start()
    return _mesh_loader


def _render_mesh_ortho(verts_flat: np.ndarray, idx: np.ndarray, size: int) -> Optional[QPixmap]:
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


def _render_mesh_for_cache(path: str, size: int, verts: np.ndarray, idx: np.ndarray) -> Optional[QPixmap]:
    cache_key = f"mesh:{path}:{size}"
    _mesh_thumb_mutex.lock()
    if cache_key in _mesh_thumb_cache:
        pm = _mesh_thumb_cache[cache_key]
        _mesh_thumb_mutex.unlock()
        return pm
    _mesh_thumb_mutex.unlock()
    pm = _render_mesh_ortho(verts, idx, size)
    if pm:
        _mesh_thumb_mutex.lock()
        _mesh_thumb_cache[cache_key] = pm
        _mesh_thumb_mutex.unlock()
    return pm


def _get_thumbnail(path: str, size: int) -> QPixmap:
    cache_key = f"thumb:{path}:{size}"
    _thumbnail_mutex.lock()
    if cache_key in _thumbnail_cache:
        cached = _thumbnail_cache[cache_key]
        _thumbnail_mutex.unlock()
        return cached
    _thumbnail_mutex.unlock()
    pm = _get_thumbnail_raw(path, size)
    if pm:
        _thumbnail_mutex.lock()
        _thumbnail_cache[cache_key] = pm
        _thumbnail_mutex.unlock()
    return pm


def _draw_mesh_icon(size: int) -> QPixmap:
    pm = _make_icon_bg(QColor(70, 130, 200), size)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(255, 255, 255, 200), 2))
    cx, cy = size // 2, size // 2
    r = size * 0.28
    for i in range(3):
        a1 = math.radians(i * 120 - 90)
        a2 = math.radians((i + 1) * 120 - 90)
        x1 = cx + r * math.cos(a1); y1 = cy + r * math.sin(a1)
        x2 = cx + r * math.cos(a2); y2 = cy + r * math.sin(a2)
        p.drawLine(int(x1), int(y1), int(x2), int(y2))
    p.end()
    return pm

def _read_audio_data(path: str, num_samples: int = 128) -> Optional[list]:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".wav":
            with wave.open(path, "rb") as wf:
                nframes = wf.getnframes()
                sampwidth = wf.getsampwidth()
                nchannels = wf.getnchannels()
                raw = wf.readframes(nframes)
                if sampwidth == 1:
                    fmt = f"<{len(raw)}B"
                    data = struct.unpack(fmt, raw)
                    data = [v - 128 for v in data]
                elif sampwidth == 2:
                    fmt = f"<{len(raw) // 2}h"
                    data = struct.unpack(fmt, raw)
                else:
                    return None
                step = max(1, len(data) // nchannels // num_samples)
        else:
            try:
                import subprocess, tempfile, struct
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "22050", "-f", "wav", tmp_path],
                               capture_output=True, timeout=10)
                with wave.open(tmp_path, "rb") as wf:
                    nframes = wf.getnframes()
                    raw = wf.readframes(min(nframes, 44100))
                    fmt_s = f"<{len(raw) // 2}h"
                    data = struct.unpack(fmt_s, raw)
                    step = max(1, len(data) // num_samples)
                os.unlink(tmp_path)
            except Exception:
                return None
        peaks = []
        for i in range(0, len(data), step):
            chunk = abs(data[i])
            if chunk > 32767: chunk = 32767
            peaks.append(chunk)
        max_val = max(peaks) if peaks else 1
        return [p / max_val for p in peaks]
    except Exception:
        return None

def _draw_audio_icon(size: int) -> QPixmap:
    return _make_icon_bg(QColor(80, 180, 80), size)

def _draw_audio_waveform(path: str, size: int) -> QPixmap:
    peaks = _read_audio_data(path, size // 2)
    if peaks is None:
        return _draw_audio_icon(size)
    pm = _make_icon_bg(QColor(40, 90, 40), size)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    margin = 4
    w = size - 2 * margin
    h = size - 2 * margin
    mid_y = size // 2
    bar_w = max(1, w // len(peaks))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(100, 220, 100, 220)))
    for i, peak in enumerate(peaks):
        bar_h = max(1, int(peak * h * 0.45))
        x = margin + i * bar_w
        p.drawRect(x, mid_y - bar_h, bar_w, bar_h * 2)
    p.end()
    return pm

def _draw_script_icon(size: int) -> QPixmap:
    pm = _make_icon_bg(QColor(200, 140, 50), size)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(255, 255, 255, 220), 1))
    lines = [(0.25, 0.3, 0.75, 0.3), (0.25, 0.5, 0.75, 0.5), (0.25, 0.7, 0.6, 0.7)]
    for x1, y1, x2, y2 in lines:
        p.drawLine(int(size * x1), int(size * y1), int(size * x2), int(size * y2))
    p.end()
    return pm

def _draw_image_icon(size: int) -> QPixmap:
    pm = _make_icon_bg(QColor(100, 150, 200), size)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(255, 255, 255, 180), 2))
    m = size * 0.2
    p.drawRect(int(m), int(m), int(size - 2 * m), int(size - 2 * m))
    p.drawLine(int(m * 1.3), int(size - m * 1.2), int(size * 0.45), int(size * 0.5))
    p.drawLine(int(size * 0.45), int(size * 0.5), int(size * 0.65), int(size * 0.65))
    p.drawLine(int(size * 0.65), int(size * 0.65), int(size - m * 1.3), int(size * 0.35))
    p.end()
    return pm

def _draw_scene_icon(size: int) -> QPixmap:
    pm = _make_icon_bg(QColor(60, 60, 100), size)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(255, 255, 255, 180), 2))
    cx, cy = size // 2, size // 2
    r = size * 0.3
    p.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))
    p.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
    p.drawLine(int(cx), int(cy - r), int(cx), int(cy + r))
    p.end()
    return pm

def _draw_prefab_icon(size: int) -> QPixmap:
    pm = _make_icon_bg(QColor(60, 160, 180), size)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(255, 255, 255, 200), 2))
    m = size * 0.25
    p.drawRect(int(m), int(m), int(size - 2 * m), int(size - 2 * m))
    p.drawLine(int(size * 0.3), int(size * 0.5), int(size * 0.7), int(size * 0.5))
    p.drawLine(int(size * 0.5), int(size * 0.3), int(size * 0.5), int(size * 0.7))
    p.end()
    return pm

def _draw_material_icon(size: int) -> QPixmap:
    return _draw_material_sphere(size, QColor(160, 80, 180))

def _draw_material_sphere(size: int, color: QColor, texture_path: str = "") -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    cx, cy = size // 2, size // 2
    r = size * 0.37
    rect = QRectF(cx - r, cy - r, r * 2, r * 2)
    path = QPainterPath()
    path.addEllipse(rect)
    p.setClipPath(path)
    if texture_path and os.path.exists(texture_path):
        tex_img = QImage(texture_path)
        if not tex_img.isNull():
            tex_pm = QPixmap.fromImage(tex_img.scaled(size, size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            tw, th = tex_pm.width(), tex_pm.height()
            p.drawPixmap(int(cx - tw / 2), int(cy - th / 2), tex_pm)
    fill = QRadialGradient(cx - r * 0.3, cy - r * 0.3, r * 1.2)
    fill.setColorAt(0.0, QColor(255, 255, 255, 180))
    fill.setColorAt(0.3, QColor(255, 255, 255, 80))
    fill.setColorAt(0.7, QColor(0, 0, 0, 40))
    fill.setColorAt(1.0, QColor(0, 0, 0, 160))
    p.setBrush(QBrush(fill))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(rect)
    spec = QRadialGradient(cx - r * 0.25, cy - r * 0.25, r * 0.5)
    spec.setColorAt(0.0, QColor(255, 255, 255, 120))
    spec.setColorAt(0.5, QColor(255, 255, 255, 20))
    spec.setColorAt(1.0, QColor(255, 255, 255, 0))
    p.setBrush(QBrush(spec))
    p.drawEllipse(rect)
    p.setClipping(False)
    p.setPen(QPen(color.darker(120), 1))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(rect)
    p.end()
    return pm

def _draw_shader_icon(size: int) -> QPixmap:
    pm = _make_icon_bg(QColor(100, 100, 120), size)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(255, 255, 255, 200), 2))
    m = size * 0.2
    p.drawRect(int(m), int(m), int(size - 2 * m), int(size - 2 * m))
    p.drawLine(int(m * 1.2), int(size * 0.45), int(size - m * 1.2), int(size * 0.45))
    p.drawLine(int(size * 0.55), int(m * 1.2), int(size * 0.55), int(size * 0.4))
    p.drawLine(int(size * 0.45), int(size * 0.6), int(size * 0.45), int(size - m * 1.2))
    p.end()
    return pm

def _draw_file_icon(size: int) -> QPixmap:
    pm = _make_icon_bg(QColor(140, 140, 150), size)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(255, 255, 255, 180), 2))
    m = size * 0.2
    p.drawRect(int(m), int(m), int(size - 2 * m), int(size - 2 * m))
    p.end()
    return pm

def _draw_folder_icon(size: int) -> QPixmap:
    pm = _make_icon_bg(QColor(200, 180, 60), size)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(255, 255, 255, 160), 1))
    m = size * 0.2
    tab_w = size * 0.25
    tab_h = size * 0.1
    body = QRect(int(m), int(m + tab_h), int(size - 2 * m), int(size - m * 2 - tab_h))
    p.drawRect(body)
    p.drawLine(int(m), int(m + tab_h), int(m + tab_w), int(m + tab_h))
    p.drawLine(int(m + tab_w), int(m + tab_h), int(m + tab_w), int(m))
    p.drawLine(int(m + tab_w), int(m), int(body.right()), int(m))
    p.end()
    return pm

def _draw_font_thumbnail(path: str, size: int) -> QPixmap:
    pm = _make_icon_bg(QColor(70, 120, 200), size)
    try:
        from PIL import Image, ImageDraw, ImageFont
        fs = max(int(size * 0.45), 8)
        font = ImageFont.truetype(path, fs)
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        bbox = font.getbbox("Aa")
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (size - tw) // 2 - bbox[0]
        y = (size - th) // 2 - bbox[1]
        draw.text((x, y), "Aa", font=font, fill=(255, 255, 255, 255))
        arr = np.array(img)
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], 4 * arr.shape[1], QImage.Format.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)
        painter = QPainter(pm)
        painter.drawPixmap(0, 0, pix)
        painter.end()
    except Exception:
        painter = QPainter(pm)
        _draw_text_centered(painter, "Aa", QRect(0, 0, size, size), QColor(255, 255, 255))
        painter.end()
    return pm

def _get_thumbnail_raw(path: str, size: int) -> QPixmap:
    if os.path.isdir(path):
        return _draw_folder_icon(size)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg"):
        reader = QImageReader(path)
        if reader.canRead():
            img = reader.read()
            if img:
                pm = QPixmap.fromImage(img)
                return pm.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
    if ext in (".obj", ".fbx", ".stl", ".usdz", ".gltf", ".glb"):
        cache_key = f"mesh:{path}:{size}"
        _mesh_thumb_mutex.lock()
        cached = _mesh_thumb_cache.get(cache_key)
        _mesh_thumb_mutex.unlock()
        if cached is not None:
            return cached
        _get_mesh_loader().enqueue(path, size)
        return _draw_mesh_icon(size)
    if ext == ".wav":
        return _draw_audio_waveform(path, size)
    if ext in (".mp3", ".ogg"):
        return _draw_audio_icon(size)
    if ext == ".py":
        return _draw_script_icon(size)
    if ext == ".zpes":
        return _draw_scene_icon(size)
    if ext == ".zpep":
        return _draw_prefab_icon(size)
    if ext == ".mat":
        pm = _get_material_thumbnail(path, size)
        if pm: return pm
        return _draw_material_icon(size)
    if ext in (".vert", ".frag"):
        return _draw_shader_icon(size)
    if ext == ".animclip":
        return _draw_file_icon(size)
    if ext == ".animcontroller":
        return _draw_file_icon(size)
    if ext in (".ttf", ".otf"):
        return _draw_font_thumbnail(path, size)
    return _draw_file_icon(size)

def _get_material_thumbnail(path: str, size: int) -> Optional[QPixmap]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        props = data.get("properties", {})
        color = props.get("_BaseColor", props.get("albedo_color", [1.0, 1.0, 1.0, 1.0]))
        tex_path = props.get("_BaseMap", props.get("albedo_texture", ""))
        mat_dir = os.path.dirname(os.path.abspath(path))
        if tex_path and not os.path.isabs(tex_path):
            tex_path = os.path.normpath(os.path.join(mat_dir, tex_path))
        r = _clamp(int(color[0] * 255), 0, 255)
        g = _clamp(int(color[1] * 255), 0, 255)
        b = _clamp(int(color[2] * 255), 0, 255)
        return _draw_material_sphere(size, QColor(r, g, b), tex_path)
    except Exception:
        return None

def _render_material_thumbnail(size: int, albedo: list[float], metallic: float = 0.0,
                                smoothness: float = 0.5, emission: Optional[list[float]] = None,
                                emit_intensity: float = 0.0,
                                texture_path: Optional[str] = None) -> Optional[QPixmap]:
    return None

def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024: return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024: return f"{size_bytes / 1024:.1f} KB"
    else: return f"{size_bytes / (1024 * 1024):.1f} MB"

class _PopulateWorker(QThread):
    batch_ready = pyqtSignal(object)

    def __init__(self, project_root: str, extensions: tuple, filter_text: str):
        super().__init__()
        self._project_root = project_root
        self._extensions = extensions
        self._filter_text = filter_text
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        _skip_dirs = {"build", "build_output", ".git", ".idea", "__pycache__",
                       "node_modules", ".venv", "venv", "tools", ".pytest_cache",
                       "physics_solvers", "thirdparty", "lib"}
        search_root = self._project_root
        assets_dir = os.path.join(self._project_root, "assets")
        if os.path.isdir(assets_dir):
            search_root = assets_dir
        batch: list = []
        for root, dirs, files in os.walk(search_root):
            if self._cancelled:
                return
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _skip_dirs]
            for f in files:
                if self._cancelled:
                    return
                if f.startswith("."):
                    continue
                ext = os.path.splitext(f)[1].lower()
                if self._extensions and ext not in self._extensions:
                    continue
                if self._filter_text and self._filter_text.lower() not in f.lower():
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, self._project_root)
                try:
                    file_size = os.path.getsize(full_path)
                except OSError:
                    file_size = 0
                batch.append((full_path, f, rel_path, file_size))
                if len(batch) >= 100:
                    self.batch_ready.emit(batch)
                    batch = []
        if batch:
            self.batch_ready.emit(batch)


class _ThumbnailLoader(QThread):
    thumbnail_loaded = pyqtSignal(object, object)

    def __init__(self, items: list[tuple], thumb_size: int):
        super().__init__()
        self._items = items
        self._thumb_size = thumb_size
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        batch: list[tuple[int, str]] = []
        for idx, (full_path, filename, rel_path, file_size) in enumerate(self._items):
            if self._cancelled:
                return
            cache_key = f"thumb:{full_path}:{self._thumb_size}"
            _thumbnail_mutex.lock()
            exists = cache_key in _thumbnail_cache
            _thumbnail_mutex.unlock()
            if not exists:
                pm = _get_thumbnail_raw(full_path, self._thumb_size)
                if pm:
                    _thumbnail_mutex.lock()
                    _thumbnail_cache[cache_key] = pm
                    _thumbnail_mutex.unlock()
            batch.append((idx, full_path))
            if len(batch) >= 32:
                self.thumbnail_loaded.emit(batch, None)
                batch.clear()
        if batch:
            self.thumbnail_loaded.emit(batch, None)


class ResourcePickerDialog(QDialog):
    def __init__(self, title: str, filter_str: str, project_root: str = ".", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(640, 480)
        self.resize(760, 560)
        self._filter_str = filter_str
        self._project_root = os.path.abspath(project_root)
        self._selected_path: Optional[str] = None
        self._extensions = self._parse_extensions(filter_str)
        self._search_text = ""
        self._setup_ui()
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(250)
        self._search_timer.timeout.connect(self._do_search)
        self._start_populate("")

    def _parse_extensions(self, filter_str: str) -> tuple:
        for key, exts in EXTENSION_FILTERS.items():
            if filter_str.lower() in key.lower():
                return exts
        return EXTENSION_FILTERS.get(filter_str, ())

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        search_layout = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search assets...")
        self._search.textChanged.connect(self._on_search)
        search_layout.addWidget(self._search, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        search_layout.addWidget(browse_btn)
        layout.addLayout(search_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        list_container = QWidget()
        list_layout = QVBoxLayout(list_container)
        list_layout.setContentsMargins(0, 0, 0, 0)
        self._list = QListWidget()
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self._list.setGridSize(QSize(THUMB_SIZE + 24, THUMB_SIZE + 44))
        self._list.setWordWrap(True)
        self._list.setSpacing(4)
        self._list.setUniformItemSizes(True)
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setLayoutMode(QListView.LayoutMode.Batched)
        self._list.setBatchSize(50)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.itemDoubleClicked.connect(self._accept_selection)
        list_layout.addWidget(self._list)
        splitter.addWidget(list_container)

        preview_container = QWidget()
        preview_container.setFixedWidth(220)
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(8, 0, 0, 0)
        preview_layout.setSpacing(6)

        self._preview_icon = QLabel()
        self._preview_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_icon.setFixedSize(*scale_xy(PREVIEW_SIZE + 20, PREVIEW_SIZE + 20))
        self._preview_icon.setStyleSheet("background: #2a2a2a; border: 1px solid #444; border-radius: 4px;")
        preview_layout.addWidget(self._preview_icon)

        self._preview_name = QLabel()
        self._preview_name.setWordWrap(True)
        self._preview_name.setStyleSheet("font-weight: bold; font-size: 12px; color: #ddd;")
        preview_layout.addWidget(self._preview_name)

        self._preview_info = QLabel()
        self._preview_info.setWordWrap(True)
        self._preview_info.setStyleSheet("font-size: 11px; color: #999;")
        preview_layout.addWidget(self._preview_info)

        preview_layout.addStretch()

        self._select_btn = QPushButton("Select")
        self._select_btn.clicked.connect(self._accept_selection)
        self._select_btn.setEnabled(False)
        preview_layout.addWidget(self._select_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        preview_layout.addWidget(cancel_btn)

        splitter.addWidget(preview_container)
        splitter.setSizes([500, 220])
        layout.addWidget(splitter, 1)

        self._update_preview(None)

    def _start_populate(self, filter_text: str):
        if hasattr(self, '_worker') and self._worker.isRunning():
            self._worker.cancel()
            self._worker.quit()
            self._worker.wait()
        if hasattr(self, '_loader') and self._loader.isRunning():
            self._loader.cancel()
            self._loader.quit()
            self._loader.wait()
        self._search_text = filter_text
        self._list.setUpdatesEnabled(False)
        self._list.clear()
        self._list.setUpdatesEnabled(True)
        self._pending_items: list[tuple[str, str, str, int]] = []
        self._all_items: list[tuple[str, str, str, int]] = []
        self._item_paths: dict[int, str] = {}
        self._batch_queue: list = []
        self._processing_batch = False
        self._placeholder = _get_placeholder_icon()
        self._thumb_queue: list = []
        self._processing_thumbs = False
        self._worker = _PopulateWorker(
            self._project_root,
            self._extensions,
            filter_text,
        )
        self._worker.batch_ready.connect(self._on_batch_ready)
        self._worker.start()

    def _on_batch_ready(self, batch: list):
        self._batch_queue.append(batch)
        if not self._processing_batch:
            self._processing_batch = True
            QTimer.singleShot(0, self._process_next_batch)

    def _process_next_batch(self):
        if not self._batch_queue:
            self._processing_batch = False
            self._list.setUpdatesEnabled(True)
            self._pending_items = self._all_items
            if self._all_items:
                self._start_thumbnail_loader()
            return
        batch = self._batch_queue.pop(0)
        placeholder = self._placeholder
        self._list.setUpdatesEnabled(False)
        for full_path, filename, rel_path, file_size in batch:
            item = QListWidgetItem(QIcon(placeholder), os.path.splitext(filename)[0])
            item.setData(Qt.ItemDataRole.UserRole, full_path)
            item.setToolTip(f"{rel_path}\n{_format_size(file_size)}")
            self._list.addItem(item)
            self._item_paths[self._list.count() - 1] = full_path
        self._list.setUpdatesEnabled(True)
        self._all_items.extend(batch)
        QTimer.singleShot(0, self._process_next_batch)

    def _start_thumbnail_loader(self):
        items = self._pending_items
        if not items:
            return
        if hasattr(self, '_loader') and self._loader.isRunning():
            self._loader.cancel()
            self._loader.quit()
            self._loader.wait()
        self._pending_items = []
        self._loader = _ThumbnailLoader(items, THUMB_SIZE)
        self._loader.thumbnail_loaded.connect(self._on_thumbnail_loaded)
        self._loader.start()

    def _on_thumbnail_loaded(self, batch, _):
        self._thumb_queue.extend(batch)
        if not self._processing_thumbs:
            self._processing_thumbs = True
            QTimer.singleShot(0, self._process_thumb_batch)

    def _process_thumb_batch(self):
        if not self._thumb_queue:
            self._processing_thumbs = False
            return
        count = 0
        while self._thumb_queue and count < 8:
            idx, path = self._thumb_queue.pop(0)
            if idx < self._list.count():
                cache_key = f"thumb:{path}:{THUMB_SIZE}"
                _thumbnail_mutex.lock()
                pm = _thumbnail_cache.get(cache_key)
                _thumbnail_mutex.unlock()
                if pm:
                    item = self._list.item(idx)
                    if item:
                        item.setIcon(QIcon(pm))
            count += 1
        QTimer.singleShot(8, self._process_thumb_batch)

    def _on_search(self, text: str):
        self._search_timer.start()

    def _do_search(self):
        self._start_populate(self._search.text())

    def _on_selection_changed(self):
        items = self._list.selectedItems()
        if items:
            path = items[0].data(Qt.ItemDataRole.UserRole)
            self._update_preview(path)
            self._select_btn.setEnabled(True)
        else:
            self._update_preview(None)
            self._select_btn.setEnabled(False)

    def _update_preview(self, path: Optional[str]):
        if not path or not os.path.isfile(path):
            self._preview_icon.clear()
            self._preview_icon.setText("No selection")
            self._preview_icon.setStyleSheet("background: #2a2a2a; border: 1px solid #444; border-radius: 4px; color: #666;")
            self._preview_name.setText("")
            self._preview_info.setText("")
            return

        self._preview_icon.setStyleSheet("background: #2a2a2a; border: 1px solid #444; border-radius: 4px;")
        thumb = _get_thumbnail(path, PREVIEW_SIZE)
        self._preview_icon.setPixmap(thumb)

        name = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()
        size_str = _format_size(os.path.getsize(path))

        type_map = {
            ".obj": "3D Model", ".fbx": "3D Model", ".stl": "3D Model", ".usdz": "3D Model", ".gltf": "3D Model", ".glb": "3D Model",
            ".wav": "Audio", ".mp3": "Audio", ".ogg": "Audio",
            ".py": "Python Script",
            ".png": "Image", ".jpg": "Image", ".jpeg": "Image",
            ".zpes": "Scene", ".zpep": "Prefab", ".mat": "Material",
            ".vert": "Vertex Shader", ".frag": "Fragment Shader",
            ".animclip": "Animation Clip", ".animcontroller": "Animator Controller",
        }
        type_name = type_map.get(ext, "File")

        info_lines = [f"Type: {type_name}", f"Size: {size_str}"]
        if ext in (".png", ".jpg", ".jpeg"):
            reader = QImageReader(path)
            if reader.canRead():
                sz = reader.size()
                info_lines.append(f"Dimensions: {sz.width()}x{sz.height()}")
        info_lines.append(f"Path: {os.path.relpath(path, self._project_root)}")
        self._preview_name.setText(name)
        self._preview_info.setText("\n".join(info_lines))

    def closeEvent(self, event):
        if hasattr(self, '_worker') and self._worker.isRunning():
            self._worker.cancel()
            self._worker.quit()
            self._worker.wait()
        if hasattr(self, '_loader') and self._loader.isRunning():
            self._loader.cancel()
            self._loader.quit()
            self._loader.wait()
        super().closeEvent(event)

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Resource", "", self._filter_str)
        if path:
            self._selected_path = path
            self.accept()

    def _accept_selection(self):
        items = self._list.selectedItems()
        if items:
            self._selected_path = items[0].data(Qt.ItemDataRole.UserRole)
            self.accept()
        elif self._list.count() == 0 and self._search.text():
            self._on_browse()
        else:
            self.reject()

    def selected_path(self) -> Optional[str]:
        return self._selected_path

def pick_resource(parent, title: str, filter_str: str, current_path: str = "",
                  project_root: str = "") -> Optional[str]:
    if not project_root:
        from core.engine.engine import Engine
        eng = Engine.instance()
        if eng is not None:
            project_root = eng.project_root
        else:
            project_root = os.getcwd()
    dlg = ResourcePickerDialog(title, filter_str, project_root, parent)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        return dlg.selected_path()
    return None
