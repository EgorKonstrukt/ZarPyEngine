# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was distributed with this file, You
# can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

"""Subprocess workers for ALL heavy thumbnail generation.

Every expensive thumbnail kind (images/textures, 3D meshes, audio waveforms,
fonts) is rendered here inside a separate process via a multiprocessing.Pool.
That keeps ``load_mesh`` (assimp), image decoding, audio reading / ffmpeg and
font rasterization completely off the editor's main-process GIL. Each worker
returns the result as PNG bytes which are cheap to pickle back to the GUI.
"""

from __future__ import annotations

import os
import sys
import math
import struct
import wave
import io

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import (
    QPixmap, QPainter, QColor, QPen, QBrush, QPainterPath,
    QImage, QImageReader, QFont, QFontMetrics,
)
from PyQt6.QtCore import Qt, QBuffer, QIODevice, QSize, QRect, QRectF

from core.assets.asset_importer import load_mesh

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff", ".webp", ".hdr")
MESH_EXTS = (".obj", ".fbx", ".stl", ".usdz", ".gltf", ".glb")
AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".aiff", ".m4a")
FONT_EXTS = (".ttf", ".otf", ".ttc", ".otc", ".woff", ".woff2")
MATERIAL_EXTS = (".mat", ".zpem")

from editor import thumb_cache as _thumb_cache

_app = None


def _ensure_app() -> QApplication:
    global _app
    if _app is None:
        argv = sys.argv if sys.argv else ["thumb_worker"]
        _app = QApplication(argv)
    return _app


def _pm_to_png(pm: QPixmap):
    if pm is None or pm.isNull():
        return None
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    if not pm.save(buf, "PNG"):
        return None
    return bytes(buf.data())



def _render_image(path: str, size: int):
    _ensure_app()
    reader = QImageReader(path)
    if not reader.canRead():
        return None
    src = reader.size()
    if not src.isEmpty():
        scale = size / max(src.width(), src.height())
        reader.setScaledSize(QSize(max(1, int(src.width() * scale)),
                                   max(1, int(src.height() * scale))))
    img = reader.read()
    if img.isNull():
        return None
    pm = QPixmap.fromImage(img)
    if pm.isNull():
        return None
    return _pm_to_png(pm)



def _render_mesh_ortho(verts_flat: np.ndarray, idx: np.ndarray, size: int,
                        settings: Optional[dict] = None) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    if len(verts_flat) < 3 or len(idx) < 3:
        return pm
    cfg = settings or {}
    rot_y_deg = cfg.get("camera_rot_y", -45.0)
    rot_x_deg = cfg.get("camera_rot_x", 30.0)
    bg = cfg.get("bg", [0.0, 0.0, 0.0, 0.0])
    tri = cfg.get("tri", [0.39, 0.63, 0.86, 0.16])
    wire = cfg.get("wire", [0.71, 0.82, 0.94, 0.78])
    wire_w = cfg.get("wire_width", 1.0)
    bg_r, bg_g, bg_b, bg_a = (bg + [0.0, 0.0, 0.0, 0.0])[:4]
    tri_r, tri_g, tri_b, tri_a = (tri + [0.0, 0.0, 0.0, 0.0])[:4]
    wire_r, wire_g, wire_b, wire_a = (wire + [0.0, 0.0, 0.0, 0.0])[:4]
    if bg_a > 0:
        pm.fill(QColor(int(bg_r * 255), int(bg_g * 255),
                        int(bg_b * 255), int(bg_a * 255)))
    pts = verts_flat.reshape(-1, 3).copy()
    rot_y = math.radians(rot_y_deg)
    rot_x = math.radians(rot_x_deg)
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
    tri_color = QColor(int(tri_r * 255), int(tri_g * 255),
                       int(tri_b * 255), int(tri_a * 255))
    wire_color = QColor(int(wire_r * 255), int(wire_g * 255),
                        int(wire_b * 255), int(wire_a * 255))
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
    p.setPen(QPen(wire_color, wire_w))
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


def _render_mesh(path: str, size: int, settings: Optional[dict] = None):
    _ensure_app()
    try:
        data = load_mesh(path)
    except Exception:
        return None
    if data is None or len(getattr(data, "vertices", [])) < 3 or len(getattr(data, "indices", [])) < 3:
        return None
    pm = _render_mesh_ortho(data.vertices, data.indices, size, settings=settings)
    return _pm_to_png(pm)



def _load_audio_mono(path: str, max_samples: int = 500000):
    ext = os.path.splitext(path)[1].lower()
    data = None
    if ext == ".wav":
        try:
            with wave.open(path, "rb") as wf:
                nframes = wf.getnframes()
                sw = wf.getsampwidth()
                nch = wf.getnchannels()
                raw = wf.readframes(nframes)
            if sw == 1:
                d = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
            elif sw == 2:
                d = np.frombuffer(raw, dtype="<i2").astype(np.float32)
            elif sw == 4:
                d = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 65536.0
            else:
                return None
            if nch > 1:
                d = d.reshape(-1, nch).mean(axis=1)
            maxv = np.max(np.abs(d)) if d.size else 1.0
            if maxv > 0:
                d = d / maxv
            data = d
        except Exception:
            return None
    else:
        try:
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "22050",
                 "-f", "wav", tmp_path],
                capture_output=True, timeout=20,
            )
            with wave.open(tmp_path, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            d = np.frombuffer(raw, dtype="<i2").astype(np.float32)
            maxv = np.max(np.abs(d)) if d.size else 1.0
            if maxv > 0:
                d = d / maxv
            data = d
            os.unlink(tmp_path)
        except Exception:
            return None
    if data is None or data.size == 0:
        return None
    if data.size > max_samples:
        step = data.size // max_samples
        idx = np.arange(0, data.size, step)[:max_samples]
        data = data[idx]
    return data


def _render_audio(path: str, size: int):
    _ensure_app()
    samples = _load_audio_mono(path)
    if samples is None or samples.size == 0:
        return None
    pm = QPixmap(size, size)
    pm.fill(QColor(34, 38, 46))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    mid = size / 2.0
    amp = (size / 2.0) * 0.92
    cols = size
    n = samples.size
    step = max(1, n // cols)

    # center line (faint)
    p.setPen(QPen(QColor(255, 255, 255, 30), 1))
    p.drawLine(0, int(mid), size, int(mid))

    top = QPainterPath()
    bot = QPainterPath()
    for x in range(cols):
        seg = samples[x * step:(x + 1) * step]
        if seg.size == 0:
            continue
        mx = float(np.max(np.abs(seg)))
        y = mid - mx * amp
        if x == 0:
            top.moveTo(x, y)
            bot.moveTo(x, size - y)
        else:
            top.lineTo(x, y)
            bot.lineTo(x, size - y)
    env = QPainterPath(top)
    # build closed filled shape: top L->R then bottom R->L
    closed = QPainterPath(top)
    # reverse bottom points
    bx = list(range(cols - 1, -1, -1))
    for x in bx:
        seg = samples[x * step:(x + 1) * step]
        if seg.size == 0:
            continue
        mx = float(np.max(np.abs(seg)))
        y = mid - mx * amp
        closed.lineTo(x, size - y)
    closed.closeSubpath()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(70, 180, 150, 220)))
    p.drawPath(closed)

    # bright center core
    core = QPainterPath()
    for x in range(cols):
        seg = samples[x * step:(x + 1) * step]
        if seg.size == 0:
            continue
        mx = float(np.max(np.abs(seg))) * 0.55
        y = mid - mx * amp
        if x == 0:
            core.moveTo(x, y)
        else:
            core.lineTo(x, y)
    closed_core = QPainterPath(core)
    for x in bx:
        seg = samples[x * step:(x + 1) * step]
        if seg.size == 0:
            continue
        mx = float(np.max(np.abs(seg))) * 0.55
        y = mid - mx * amp
        closed_core.lineTo(x, size - y)
    closed_core.closeSubpath()
    p.setBrush(QBrush(QColor(140, 230, 200, 255)))
    p.drawPath(closed_core)
    p.end()
    return _pm_to_png(pm)


def _render_font(path: str, size: int):
    _ensure_app()
    try:
        from PIL import Image, ImageDraw, ImageFont
        fs = max(int(size * 0.5), 8)
        font = ImageFont.truetype(path, fs)
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        bbox = font.getbbox("Aa")
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (size - tw) // 2 - bbox[0]
        y = (size - th) // 2 - bbox[1]
        draw.text((x, y), "Aa", font=font, fill=(235, 235, 235, 255))
        arr = np.array(img)
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], 4 * arr.shape[1],
                      QImage.Format.Format_RGBA8888)
        pm = QPixmap.fromImage(qimg)
        if pm.isNull():
            return None
        return _pm_to_png(pm)
    except Exception:
        return None


def _find_project_root(path: str):
    cur = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.isdir(os.path.join(cur, "assets")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _render_material(path: str, size: int):
    _ensure_app()
    try:
        from core.assets.material import Material
        project_root = _find_project_root(path)
        mat = Material.load(path, project_root)
        props = mat.properties if mat is not None else {}
        albedo = props.get("_BaseColor", (0.8, 0.8, 0.8))
        metallic = float(props.get("_Metallic", 0.0))
        smoothness = float(props.get("_Smoothness", 0.5))
        emission = props.get("_EmissionColor", (0.0, 0.0, 0.0))
        emit_intensity = float(props.get("_EmissionIntensity", 0.0))
        tex_path = props.get("_BaseMap") or props.get("albedo_texture", None)
        from editor.gl_offscreen import render_sphere
        pm = render_sphere(
            size, size,
            albedo=albedo,
            metallic=metallic,
            smoothness=smoothness,
            emission=emission,
            emit_intensity=emit_intensity,
            tex_path=tex_path,
        )
        if pm is not None and not pm.isNull():
            return _pm_to_png(pm)
    except Exception:
        pass
    # Fallback: simple vector material swatch.
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QBrush(QColor(120, 120, 140)))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(2, 2, size - 4, size - 4), 6, 6)
    p.end()
    return _pm_to_png(pm)


def render_thumbnail(path: str, size: int, cache_dir: Optional[str] = None,
                     mesh_preview: Optional[dict] = None):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return _render_image(path, size)
    if ext in MESH_EXTS:
        return _render_mesh(path, size, settings=mesh_preview)
    if ext in AUDIO_EXTS:
        return _render_audio(path, size)
    if ext in FONT_EXTS:
        return _render_font(path, size)
    if ext in MATERIAL_EXTS:
        return _render_material(path, size)
    return None
