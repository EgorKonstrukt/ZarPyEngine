# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import json
import math
import re
import wave
import struct
import threading
import numpy as np
from typing import Optional, Callable
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit,
                              QListWidget, QListWidgetItem, QLabel,
                              QPushButton, QWidget, QSplitter, QFileDialog,
                              QListWidgetItem, QAbstractItemView, QListView)
from PyQt6.QtCore import (Qt, QSize, QRect, QRectF, QPoint, QThread, pyqtSignal,
                          QTimer, QMutex, QCoreApplication, QMetaObject, pyqtSlot,
                          QObject, Q_ARG)
from PyQt6.QtGui import (QFont, QPixmap, QPainter, QPainterPath, QColor, QPen, QBrush,
                         QFontMetrics, QLinearGradient, QRadialGradient, QIcon, QPalette,
                         QImageReader, QPolygonF, QImage)

_thumbnail_cache: dict[str, QPixmap] = {}
_thumbnail_mutex = QMutex()
_placeholder_cache: dict[str, QPixmap] = {}
_icon_cache: dict[str, QIcon] = {}
_icon_mutex = QMutex()

from editor.constants import THUMB_SIZE, PREVIEW_SIZE
from core.config.editor_scale import scale, scale_xy
from editor.thumb_cache import cache_root_for_project, thumb_disk_key, save_thumb_disk, load_thumb_disk

_placeholder_icon: Optional[QIcon] = None

_MAX_PROCESS_INFLIGHT = 12


def _find_engine_root() -> Optional[str]:
    cur = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.basename(cur) == "editor":
            return os.path.dirname(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def _get_placeholder_icon() -> QIcon:
    global _placeholder_icon
    if _placeholder_icon is None:
        _placeholder_icon = QIcon(_draw_file_icon(THUMB_SIZE))
    return _placeholder_icon


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

_PROCESS_EXTS = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff", ".webp", ".hdr",
    ".obj", ".fbx", ".stl", ".usdz", ".gltf", ".glb",
    ".wav", ".mp3", ".ogg", ".flac", ".aiff", ".m4a",
    ".ttf", ".otf", ".ttc", ".otc", ".woff", ".woff2",
    ".mat", ".zpem",
)


def _is_process_thumb(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _PROCESS_EXTS


class _ThumbnailProcessService(QObject):
    """Generates ALL heavy thumbnails in separate processes (no GIL contention).

    Image decoding, mesh import (assimp), audio reading / ffmpeg, font
    rasterization and material (OpenGL) previews all run inside a
    multiprocessing.Pool of worker processes. Each worker returns PNG bytes
    which are decoded on the GUI thread and cached as a QPixmap.

    The pool is created lazily on the first request and torn down after a
    period of inactivity so no Python worker processes linger when the editor
    is idle. Results are also persisted to an on-disk cache (keyed by an
    xxhash of path + mtime + size) under ``<project>/cache/thumbs``.
    """

    thumbnail_ready = pyqtSignal(str, int, object)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._pool = None
        self._inflight: set[tuple[str, int]] = set()
        self._lock = QMutex()
        self._cache_dir: Optional[str] = None
        self._idle_timer: Optional[QTimer] = None

    def set_cache_dir(self, cache_dir: Optional[str]):
        self._cache_dir = cache_dir

    def _ensure_pool(self):
        if self._pool is not None:
            return
        try:
            import multiprocessing as mp
            nproc = max(1, min(4, (mp.cpu_count() or 2)))
            self._pool = mp.Pool(
                processes=nproc,
                initializer=_thumb_worker_init,
            )
        except Exception:
            self._pool = None

    def _arm_idle(self):
        # Queue the timer start onto the owning (main) thread so QTimer
        # operations are always performed on the correct thread.
        if QThread.currentThread() != self.thread():
            QMetaObject.invokeMethod(
                self, "_arm_idle_main", Qt.ConnectionType.QueuedConnection,
            )
            return
        self._arm_idle_main()

    @pyqtSlot()
    def _arm_idle_main(self):
        if self._idle_timer is None:
            self._idle_timer = QTimer(self)
            self._idle_timer.setSingleShot(True)
            self._idle_timer.timeout.connect(self._on_idle)
        self._idle_timer.start(30000)

    @pyqtSlot()
    def _on_idle(self):
        self._lock.lock()
        has_inflight = len(self._inflight) > 0
        self._lock.unlock()
        if has_inflight:
            # Thumbnails still being processed — re-arm and wait.
            self._arm_idle_main()
            return
        self.shutdown()

    def enqueue(self, path: str, size: int):
        cache_key = f"thumb:{path}:{size}"
        _thumbnail_mutex.lock()
        cached = cache_key in _thumbnail_cache
        _thumbnail_mutex.unlock()
        if cached:
            return
        # On-disk cache hit: decode locally, no subprocess.
        if self._cache_dir:
            try:
                st = os.stat(path)
                dkey = thumb_disk_key(path, size, st.st_mtime, st.st_size,
                                       mode=_thumb_cache_mode())
                png = load_thumb_disk(self._cache_dir, dkey)
                if png:
                    pm = QPixmap()
                    if pm.loadFromData(png, "PNG") and not pm.isNull():
                        _thumbnail_mutex.lock()
                        _thumbnail_cache[cache_key] = pm
                        _thumbnail_mutex.unlock()
                        self.thumbnail_ready.emit(path, size, pm)
                        return
            except OSError:
                pass
        self._lock.lock()
        if (path, size) in self._inflight:
            self._lock.unlock()
            return
        self._inflight.add((path, size))
        self._lock.unlock()
        self._arm_idle()
        self._ensure_pool()
        if self._pool is None:
            return
        try:
            self._pool.apply_async(
                _thumb_render_wrap,
                (path, size, self._cache_dir, _thumb_cache_mode(),
                 _mesh_preview_settings()),
                callback=self._on_pool_result,
                error_callback=self._on_pool_error,
            )
        except Exception:
            self._lock.lock()
            self._inflight.discard((path, size))
            self._lock.unlock()

    def _on_pool_result(self, payload):
        QMetaObject.invokeMethod(
            self, "_deliver", Qt.ConnectionType.QueuedConnection,
            Q_ARG(object, payload),
        )

    def _on_pool_error(self, exc):
        pass

    @pyqtSlot(object)
    def _deliver(self, payload):
        if not isinstance(payload, tuple) or len(payload) != 3:
            return
        path, size, png_bytes = payload
        self._lock.lock()
        self._inflight.discard((path, size))
        idle = len(self._inflight) == 0
        self._lock.unlock()
        pm = None
        if png_bytes:
            pm = QPixmap()
            if not pm.loadFromData(png_bytes, "PNG"):
                pm = None
        if pm is not None:
            cache_key = f"thumb:{path}:{size}"
            _thumbnail_mutex.lock()
            _thumbnail_cache[cache_key] = pm
            _thumbnail_mutex.unlock()
        if idle:
            self._arm_idle()
        self.thumbnail_ready.emit(path, size, pm)

    def shutdown(self):
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.terminate()
            except Exception:
                pass
            self._pool = None
        self._lock.lock()
        self._inflight.clear()
        self._lock.unlock()

    def invalidate_thumbnails(self):
        _thumbnail_mutex.lock()
        _thumbnail_cache.clear()
        _thumbnail_mutex.unlock()
        _icon_mutex.lock()
        _icon_cache.clear()
        _icon_mutex.unlock()
        if self._cache_dir:
            import shutil
            try:
                shutil.rmtree(self._cache_dir, ignore_errors=True)
            except Exception:
                pass


def _thumb_worker_init():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


_thumb_service: Optional[_ThumbnailProcessService] = None


def _thumb_render_wrap(path: str, size: int, cache_dir: Optional[str],
                        mode: str = "metadata",
                        mesh_preview: Optional[dict] = None):
    from editor.thumb_worker import render_thumbnail
    png = render_thumbnail(path, size, cache_dir, mesh_preview=mesh_preview)
    if png and cache_dir:
        try:
            st = os.stat(path)
            dkey = thumb_disk_key(path, size, st.st_mtime, st.st_size,
                                   mode=mode)
            save_thumb_disk(cache_dir, dkey, png)
        except OSError:
            pass
    return (path, size, png)


def _get_thumb_service() -> _ThumbnailProcessService:
    global _thumb_service
    if _thumb_service is None:
        _thumb_service = _ThumbnailProcessService()
    return _thumb_service


def _get_mesh_loader() -> _ThumbnailProcessService:
    return _get_thumb_service()


def _thumb_cache_mode() -> str:
    try:
        from core.config.config import get_global_config
        return get_global_config().get("editor.thumb_cache_mode", "metadata")
    except Exception:
        return "metadata"


def _thumb_resolution() -> int:
    try:
        from core.config.config import get_global_config
        return get_global_config().get("editor.thumb_resolution", 512)
    except Exception:
        return 512


def _mesh_preview_settings() -> dict:
    try:
        from core.config.config import get_global_config
        return get_global_config().get("mesh_preview", {})
    except Exception:
        return {}


def _get_thumbnail(path: str, size: int) -> QPixmap:
    cache_key = f"thumb:{path}:{size}"
    _thumbnail_mutex.lock()
    if cache_key in _thumbnail_cache:
        cached = _thumbnail_cache[cache_key]
        _thumbnail_mutex.unlock()
        return cached
    _thumbnail_mutex.unlock()
    svc = _get_thumb_service()
    if svc._cache_dir:
        try:
            st = os.stat(path)
            dkey = thumb_disk_key(path, size, st.st_mtime, st.st_size,
                                   mode=_thumb_cache_mode())
            png = load_thumb_disk(svc._cache_dir, dkey)
            if png:
                pm = QPixmap()
                if pm.loadFromData(png, "PNG") and not pm.isNull():
                    _thumbnail_mutex.lock()
                    _thumbnail_cache[cache_key] = pm
                    _thumbnail_mutex.unlock()
                    return pm
        except OSError:
            pass
    pm = _get_thumbnail_raw(path, size)
    if pm:
        _thumbnail_mutex.lock()
        _thumbnail_cache[cache_key] = pm
        _thumbnail_mutex.unlock()
    return pm


def _get_cached_icon(path: str, size: int) -> QIcon:
    cache_key = f"icon:{path}:{size}"
    _icon_mutex.lock()
    icon = _icon_cache.get(cache_key)
    _icon_mutex.unlock()
    if icon is not None:
        return icon
    pm = _get_thumbnail(path, size)
    if pm is None:
        return QIcon()
    icon = QIcon(pm)
    _icon_mutex.lock()
    _icon_cache[cache_key] = icon
    _icon_mutex.unlock()
    return icon


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

def _draw_audio_icon(size: int) -> QPixmap:
    return _make_icon_bg(QColor(80, 180, 80), size)

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

def _draw_font_placeholder(size: int) -> QPixmap:
    pm = _make_icon_bg(QColor(70, 120, 200), size)
    p = QPainter(pm)
    _draw_text_centered(p, "Aa", QRect(0, 0, size, size), QColor(255, 255, 255))
    p.end()
    return pm

def _get_thumbnail_raw(path: str, size: int, enqueue_mesh: bool = True) -> QPixmap:
    if os.path.isdir(path):
        return _draw_folder_icon(size)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff",
               ".webp", ".hdr"):
        if _is_process_thumb(path):
            cache_key = f"thumb:{path}:{size}"
            _thumbnail_mutex.lock()
            cached = _thumbnail_cache.get(cache_key)
            _thumbnail_mutex.unlock()
            if cached is not None:
                return cached
            _get_thumb_service().enqueue(path, size)
        return _draw_image_icon(size)
    if ext in (".obj", ".fbx", ".stl", ".usdz", ".gltf", ".glb"):
        if _is_process_thumb(path):
            cache_key = f"thumb:{path}:{size}"
            _thumbnail_mutex.lock()
            cached = _thumbnail_cache.get(cache_key)
            _thumbnail_mutex.unlock()
            if cached is not None:
                return cached
            if enqueue_mesh:
                _get_thumb_service().enqueue(path, size)
        return _draw_mesh_icon(size)
    if ext in (".wav", ".mp3", ".ogg", ".flac", ".aiff", ".m4a"):
        if _is_process_thumb(path):
            cache_key = f"thumb:{path}:{size}"
            _thumbnail_mutex.lock()
            cached = _thumbnail_cache.get(cache_key)
            _thumbnail_mutex.unlock()
            if cached is not None:
                return cached
            _get_thumb_service().enqueue(path, size)
        return _draw_audio_icon(size)
    if ext == ".py":
        return _draw_script_icon(size)
    if ext == ".zpes":
        return _draw_scene_icon(size)
    if ext == ".zpep":
        return _draw_prefab_icon(size)
    if ext in (".mat", ".zpem"):
        if _is_process_thumb(path):
            cache_key = f"thumb:{path}:{size}"
            _thumbnail_mutex.lock()
            cached = _thumbnail_cache.get(cache_key)
            _thumbnail_mutex.unlock()
            if cached is not None:
                return cached
            _get_thumb_service().enqueue(path, size)
        return _draw_material_icon(size)
    if ext in (".shader", ".vert", ".frag", ".compute"):
        return _draw_shader_icon(size)
    if ext in (".animclip", ".animcontroller"):
        return _draw_file_icon(size)
    if ext in (".ttf", ".otf", ".ttc", ".otc", ".woff", ".woff2"):
        if _is_process_thumb(path):
            cache_key = f"thumb:{path}:{size}"
            _thumbnail_mutex.lock()
            cached = _thumbnail_cache.get(cache_key)
            _thumbnail_mutex.unlock()
            if cached is not None:
                return cached
            _get_thumb_service().enqueue(path, size)
        return _draw_font_placeholder(size)
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
        engine_root = _find_engine_root()
        extra_dirs: list[str] = []
        for sub in ("core/shaders", "editor/shaders"):
            for base in (self._project_root, engine_root):
                if not base:
                    continue
                d = os.path.normpath(os.path.join(base, sub))
                if os.path.isdir(d) and not d.startswith(os.path.normpath(search_root)):
                    if d not in extra_dirs:
                        extra_dirs.append(d)
        search_dirs = [search_root] + extra_dirs
        batch: list = []
        seen_paths: set[str] = set()
        for search_dir in search_dirs:
            for root, dirs, files in os.walk(search_dir):
                if self._cancelled:
                    return
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _skip_dirs]
                for f in files:
                    if self._cancelled:
                        return
                    if f.startswith("."):
                        continue
                    if f.endswith(".import"):
                        continue
                    ext = os.path.splitext(f)[1].lower()
                    if self._extensions and ext not in self._extensions:
                        continue
                    if self._filter_text and self._filter_text.lower() not in f.lower():
                        continue
                    full_path = os.path.normpath(os.path.join(root, f))
                    if full_path in seen_paths:
                        continue
                    seen_paths.add(full_path)
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


def _draw_vector_placeholder(path: str, size: int) -> Optional[QPixmap]:
    if os.path.isdir(path):
        return _draw_folder_icon(size)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff",
               ".webp", ".hdr"):
        return _draw_image_icon(size)
    if ext in (".obj", ".fbx", ".stl", ".usdz", ".gltf", ".glb"):
        return _draw_mesh_icon(size)
    if ext in (".wav", ".mp3", ".ogg", ".flac", ".aiff", ".m4a"):
        return _draw_audio_icon(size)
    if ext == ".py":
        return _draw_script_icon(size)
    if ext == ".zpes":
        return _draw_scene_icon(size)
    if ext == ".zpep":
        return _draw_prefab_icon(size)
    if ext in (".mat", ".zpem"):
        return _draw_material_icon(size)
    if ext in (".shader", ".vert", ".frag", ".compute"):
        return _draw_shader_icon(size)
    if ext in (".animclip", ".animcontroller"):
        return _draw_file_icon(size)
    if ext in (".ttf", ".otf", ".ttc", ".otc", ".woff", ".woff2"):
        return _draw_font_placeholder(size)
    return _draw_file_icon(size)


class _ThumbnailLoader(QThread):
    thumbnail_loaded = pyqtSignal(object, object)

    def __init__(self, thumb_size: int):
        super().__init__()
        self._thumb_size = thumb_size
        self._queue: list[tuple[int, str]] = []
        self._queue_mutex = QMutex()
        self._cancelled = False
        self._running = False

    def cancel(self):
        self._cancelled = True

    def enqueue(self, items: list[tuple[int, str]]):
        self._queue_mutex.lock()
        self._queue.extend(items)
        self._queue_mutex.unlock()

    def pending_count(self) -> int:
        self._queue_mutex.lock()
        n = len(self._queue)
        self._queue_mutex.unlock()
        return n

    def run(self):
        self._running = True
        batch: list[tuple[int, str]] = []
        while not self._cancelled:
            self._queue_mutex.lock()
            if not self._queue:
                self._queue_mutex.unlock()
                break
            idx, full_path = self._queue.pop(0)
            self._queue_mutex.unlock()
            cache_key = f"thumb:{full_path}:{self._thumb_size}"
            _thumbnail_mutex.lock()
            exists = cache_key in _placeholder_cache
            _thumbnail_mutex.unlock()
            if not exists:
                pm = _draw_vector_placeholder(full_path, self._thumb_size)
                if pm:
                    _thumbnail_mutex.lock()
                    _placeholder_cache[cache_key] = pm
                    _thumbnail_mutex.unlock()
            batch.append((idx, full_path))
            if len(batch) >= 32:
                self.thumbnail_loaded.emit(batch, None)
                batch.clear()
        if batch:
            self.thumbnail_loaded.emit(batch, None)
        self._running = False


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
        m = re.search(r'\(([^)]+)\)', filter_str)
        if not m:
            return ()
        raw = m.group(1).strip()
        if raw == "*" or not raw:
            return ()
        parts = raw.split()
        exts = tuple(p if p.startswith(".") else p[1:] for p in parts if p != "*")
        return exts

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
        self._thumb_scroll_timer = QTimer(self)
        self._thumb_scroll_timer.setSingleShot(True)
        self._thumb_scroll_timer.setInterval(60)
        self._thumb_scroll_timer.timeout.connect(self._schedule_thumbnails)
        self._list.verticalScrollBar().valueChanged.connect(
            lambda _: self._thumb_scroll_timer.start())
        self._list.horizontalScrollBar().valueChanged.connect(
            lambda _: self._thumb_scroll_timer.start())
        self._mesh_pending = 0
        self._mesh_loaded: set[int] = set()
        _get_thumb_service().thumbnail_ready.connect(
            self._on_thumb_ready, Qt.ConnectionType.QueuedConnection)
        _get_thumb_service().thumbnail_ready.connect(
            self._on_thumb_ready_preview, Qt.ConnectionType.QueuedConnection)
        self._preview_path: Optional[str] = None
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
        _get_thumb_service().set_cache_dir(cache_root_for_project(self._project_root))
        self._list.setUpdatesEnabled(False)
        self._list.clear()
        self._list.setUpdatesEnabled(True)
        self._all_items: list[tuple[str, str, str, int]] = []
        self._item_paths: dict[int, str] = {}
        self._batch_queue: list = []
        self._processing_batch = False
        self._placeholder = _get_placeholder_icon()
        self._thumb_queue: list = []
        self._processing_thumbs = False
        self._mesh_pending = 0
        self._mesh_loaded: set[int] = set()
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
        if not hasattr(self, '_loader') or not self._loader.isRunning():
            if hasattr(self, '_loader'):
                self._loader.cancel()
                self._loader.wait()
            self._loader = _ThumbnailLoader(THUMB_SIZE)
            self._loader.thumbnail_loaded.connect(self._on_thumbnail_loaded)
            self._loader.start()
        self._loaded_thumbs = set()
        self._schedule_thumbnails(force=True)

    def _visible_item_range(self) -> tuple[int, int]:
        count = self._list.count()
        if count == 0:
            return 0, -1
        vp = self._list.viewport()
        rect = vp.rect()
        top_index = self._list.indexAt(rect.topLeft())
        bottom_index = self._list.indexAt(QPoint(rect.left(), rect.bottom() - 2))
        first = top_index.row() if top_index.isValid() else 0
        last = bottom_index.row() if bottom_index.isValid() else count - 1
        if last < first:
            last = first
        buffer = 40
        return max(0, first - buffer), min(count - 1, last + buffer)

    def _schedule_thumbnails(self, force: bool = False):
        if not hasattr(self, '_loader'):
            self._start_thumbnail_loader()
            return
        if not force and not self._loader.isRunning():
            return
        if not hasattr(self, '_loaded_thumbs'):
            self._loaded_thumbs = set()
        first, last = self._visible_item_range()
        if last < first:
            return
        # strictly visible items (no buffer) get expensive mesh previews
        vis_first, vis_last = self._visible_item_range_strict()
        to_load: list[tuple[int, str]] = []
        for idx in range(first, last + 1):
            if idx in self._loaded_thumbs:
                continue
            path = self._item_paths.get(idx)
            if not path:
                continue
            cache_key = f"thumb:{path}:{_thumb_resolution()}"
            _thumbnail_mutex.lock()
            cached = cache_key in _thumbnail_cache
            _thumbnail_mutex.unlock()
            self._loaded_thumbs.add(idx)
            if cached:
                self._thumb_queue.append((idx, path))
            else:
                to_load.append((idx, path))
            # Enqueue expensive (process) thumbnails for the whole visible
            # window plus buffer, not just strictly-visible items, so items
            # that become visible on scroll already have real thumbnails.
            self._maybe_enqueue_process_thumb(idx, path)
        if to_load:
            self._loader.enqueue(to_load)
        if self._thumb_queue and not self._processing_thumbs:
            self._processing_thumbs = True
            QTimer.singleShot(0, self._process_thumb_batch)

    def _visible_item_range_strict(self) -> tuple[int, int]:
        count = self._list.count()
        if count == 0:
            return 0, -1
        vp = self._list.viewport()
        rect = vp.rect()
        top_index = self._list.indexAt(rect.topLeft())
        bottom_index = self._list.indexAt(QPoint(rect.left(), rect.bottom() - 2))
        first = top_index.row() if top_index.isValid() else 0
        last = bottom_index.row() if bottom_index.isValid() else count - 1
        return max(0, first), min(count - 1, last)

    def _maybe_enqueue_process_thumb(self, idx: int, path: str):
        if self._mesh_pending >= _MAX_PROCESS_INFLIGHT:
            return
        if idx in self._mesh_loaded:
            return
        if not _is_process_thumb(path):
            return
        cache_key = f"thumb:{path}:{_thumb_resolution()}"
        _thumbnail_mutex.lock()
        cached = cache_key in _thumbnail_cache
        _thumbnail_mutex.unlock()
        if cached:
            self._mesh_loaded.add(idx)
            self._thumb_queue.append((idx, path))
            if self._thumb_queue and not self._processing_thumbs:
                self._processing_thumbs = True
                QTimer.singleShot(0, self._process_thumb_batch)
            return
        self._mesh_loaded.add(idx)
        self._mesh_pending += 1
        _get_thumb_service().enqueue(path, _thumb_resolution())

    def _on_thumb_ready(self, path: str, size: int, pixmap):
        if size != _thumb_resolution():
            return
        self._mesh_pending = max(0, self._mesh_pending - 1)
        if pixmap is None or pixmap.isNull():
            return
        icon = QIcon(pixmap)
        # Refresh the icon cache so _process_thumb_batch / _get_cached_icon
        # never re-apply a stale placeholder over the real thumbnail.
        icon_key = f"icon:{path}:{size}"
        _icon_mutex.lock()
        _icon_cache[icon_key] = icon
        _icon_mutex.unlock()
        for idx, p in self._item_paths.items():
            if p == path and idx < self._list.count():
                item = self._list.item(idx)
                if item:
                    item.setIcon(icon)

    def _on_thumb_ready_preview(self, path: str, size: int, pixmap):
        if size != PREVIEW_SIZE:
            return
        if self._preview_path != path:
            return
        if pixmap is None or pixmap.isNull():
            return
        self._preview_icon.setPixmap(pixmap)

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
        while self._thumb_queue and count < 60:
            idx, path = self._thumb_queue.pop(0)
            if idx < self._list.count():
                pm = None
                cache_key = f"thumb:{path}:{_thumb_resolution()}"
                _thumbnail_mutex.lock()
                pm = _thumbnail_cache.get(cache_key)
                _thumbnail_mutex.unlock()
                if pm is None:
                    ph_key = f"thumb:{path}:{THUMB_SIZE}"
                    _thumbnail_mutex.lock()
                    pm = _placeholder_cache.get(ph_key)
                    _thumbnail_mutex.unlock()
                item = self._list.item(idx)
                if item and pm is not None and not pm.isNull():
                    item.setIcon(QIcon(pm))
            count += 1
        QTimer.singleShot(2, self._process_thumb_batch)

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
        self._preview_path = path
        thumb = _get_thumbnail(path, PREVIEW_SIZE)
        if thumb.isNull():
            pm = _get_placeholder_icon()
            self._preview_icon.setPixmap(pm)
        else:
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
            ".shader": "Shader", ".compute": "Compute Shader",
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
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QCursor
    dlg = ResourcePickerDialog(title, filter_str, project_root, None)
    dlg.adjustSize()
    screen = QApplication.primaryScreen().availableGeometry()
    dw, dh = dlg.width(), dlg.height()
    cursor_pos = QCursor.pos()
    x = cursor_pos.x() - dw // 2
    y = cursor_pos.y() + 4
    if y + dh > screen.bottom():
        y = cursor_pos.y() - dh - 4
    if y < screen.top():
        y = screen.top()
    if x + dw > screen.right():
        x = screen.right() - dw
    if x < screen.left():
        x = screen.left()
    dlg.move(x, y)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        return dlg.selected_path()
    return None
