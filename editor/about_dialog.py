# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os
import platform
import random
import sys
import time
import datetime
import json

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QTextEdit, QApplication)
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor, QPen, QKeyEvent
from PyQt6.QtCore import Qt, QRectF, QTimer

from PyQt6.QtSvg import QSvgRenderer

from core.config.constants import APP_VERSION, APP_VERSION_DISPLAY

MPL_URL = "https://mozilla.org/MPL/2.0/"
LOGO_W = 256
_EASTER_CLICKS_NEEDED = 10


def _read_engine_optimization() -> dict:
    try:
        from pathlib import Path
        settings = Path.home() / ".zarin" / "settings.json"
        if settings.exists():
            with open(settings) as f:
                data = json.load(f)
            return data.get("engine", {})
    except Exception:
        pass
    return {}


def _render_logo(target_w: int) -> QPixmap | None:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "zarin_logo.svg")
    renderer = QSvgRenderer(path)
    if not renderer.isValid():
        return None
    vb = renderer.viewBoxF()
    if vb.isEmpty():
        vb = QRectF(0, 0, 512, 512)
    target_h = round(target_w * vb.height() / vb.width())
    pm = QPixmap(target_w, target_h)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(p, QRectF(0, 0, target_w, target_h))
    p.end()
    return pm


def _render_icon(size: int) -> QPixmap | None:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "zarin_icon.svg")
    renderer = QSvgRenderer(path)
    if not renderer.isValid():
        return None
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(p, QRectF(0, 0, size, size))
    p.end()
    return pm


_COLS = 20
_ROWS = 15
_CELL = 22
_TICK_MS = 110

_DIR = {
    Qt.Key.Key_Up: (0, -1),
    Qt.Key.Key_Down: (0, 1),
    Qt.Key.Key_Left: (-1, 0),
    Qt.Key.Key_Right: (1, 0),
}
_OPPOSITE = {
    Qt.Key.Key_Up: Qt.Key.Key_Down,
    Qt.Key.Key_Down: Qt.Key.Key_Up,
    Qt.Key.Key_Left: Qt.Key.Key_Right,
    Qt.Key.Key_Right: Qt.Key.Key_Left,
}

_BG = QColor(24, 24, 28)
_GRID = QColor(30, 30, 36)
_SNAKE_HEAD = QColor(70, 180, 100)
_SNAKE_BODY = QColor(50, 140, 75)
_FOOD = QColor(220, 60, 60)
_TEXT = QColor(200, 200, 210)


class _SnakeGame(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zarin Snake")
        self.setFixedSize(_COLS * _CELL + 2, _ROWS * _CELL + 2)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._head_pm = _render_icon(_CELL)
        self._reset()

    def _reset(self):
        mid_x, mid_y = _COLS // 2, _ROWS // 2
        self._snake = [(mid_x - i, mid_y) for i in range(4)]
        self._dir = (1, 0)
        self._next_dir = (1, 0)
        self._alive = True
        self._score = 0
        self._spawn_food()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(_TICK_MS)

    def _spawn_food(self):
        while True:
            pos = (random.randint(0, _COLS - 1), random.randint(0, _ROWS - 1))
            if pos not in self._snake:
                self._food = pos
                return

    def keyPressEvent(self, e: QKeyEvent):
        nd = _DIR.get(e.key())
        if nd is None:
            return
        dx, dy = self._dir
        if nd != (-dx, -dy):
            self._next_dir = nd

    def _tick(self):
        if not self._alive:
            return
        self._dir = self._next_dir
        hx, hy = self._snake[0]
        dx, dy = self._dir
        nx, ny = hx + dx, hy + dy

        if nx < 0 or nx >= _COLS or ny < 0 or ny >= _ROWS:
            return self._die()
        if (nx, ny) in self._snake:
            return self._die()

        self._snake.insert(0, (nx, ny))
        if (nx, ny) == self._food:
            self._score += 1
            self._spawn_food()
        else:
            self._snake.pop()
        self.update()

    def _die(self):
        self._alive = False
        self._timer.stop()
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 0, self.width(), self.height(), _BG)

        for x in range(_COLS + 1):
            p.setPen(_GRID)
            p.drawLine(x * _CELL, 0, x * _CELL, _ROWS * _CELL)
        for y in range(_ROWS + 1):
            p.setPen(_GRID)
            p.drawLine(0, y * _CELL, _COLS * _CELL, y * _CELL)

        fx, fy = self._food
        p.setBrush(_FOOD)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(fx * _CELL + 2, fy * _CELL + 2, _CELL - 4, _CELL - 4)

        for i, (sx, sy) in enumerate(self._snake):
            if i == 0 and self._head_pm is not None:
                p.drawPixmap(sx * _CELL, sy * _CELL, self._head_pm)
            else:
                c = _SNAKE_BODY
                p.setBrush(c)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(sx * _CELL + 1, sy * _CELL + 1, _CELL - 2, _CELL - 2, 4, 4)

        if not self._alive:
            p.setPen(QPen(_TEXT, 1))
            p.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       f"Game Over!\nScore: {self._score}\n\nPress Space to restart")
        p.end()

    def keyReleaseEvent(self, e: QKeyEvent):
        if not self._alive and e.key() == Qt.Key.Key_Space:
            self._reset()

    def closeEvent(self, e):
        self._timer.stop()
        super().closeEvent(e)


def _collect_system_info(parent=None) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("Zarin Engine — System Info")
    lines.append("=" * 60)
    lines.append("")

    lines.append("[Application]")
    lines.append(f"  Version:       {APP_VERSION_DISPLAY}")
    lines.append(f"  Built:         {time.ctime(os.path.getmtime(os.path.abspath(__file__)))}")
    lines.append("")

    lines.append("[Platform]")
    lines.append(f"  OS:            {platform.system()} {platform.release()}")
    lines.append(f"  OS Version:    {platform.version()}")
    lines.append(f"  Machine:       {platform.machine()}")
    lines.append(f"  Processor:     {platform.processor() or 'N/A'}")
    lines.append(f"  Node:          {platform.node()}")
    lines.append("")

    lines.append("[CPU]")
    lines.append(f"  Cores:         {os.cpu_count() or 'N/A'}")
    lines.append("")

    lines.append("[Python]")
    lines.append(f"  Version:       {sys.version.split()[0]}")
    lines.append(f"  Build:         {sys.version.split('(', 1)[1].rstrip(')') if '(' in sys.version else sys.version.split()[1] if len(sys.version.split()) > 1 else 'N/A'}")
    lines.append(f"  Implementation:{sys.version.split('[')[1].split(']')[0] if '[' in sys.version else 'CPython'}")
    gil_enabled = getattr(sys, '_is_gil_enabled', lambda: True)()
    lines.append(f"  GIL:           {'Enabled' if gil_enabled else 'Disabled (nogil)'}")
    jit_enabled = getattr(sys, '_jit_enabled', False)
    lines.append(f"  JIT:           {'Active' if jit_enabled else 'Off'}")
    lines.append(f"  Executable:    {sys.executable}")
    lines.append(f"  Prefix:        {sys.prefix}")
    lines.append("")
    lines.append("[Interpreter Flags]")
    fl = sys.flags
    lines.append(f"  debug:                  {fl.debug}")
    lines.append(f"  inspect:                {fl.inspect}")
    lines.append(f"  interactive:            {fl.interactive}")
    lines.append(f"  optimize:               {fl.optimize}")
    lines.append(f"  dont_write_bytecode:    {fl.dont_write_bytecode}")
    lines.append(f"  no_user_site:           {fl.no_user_site}")
    lines.append(f"  no_site:                {fl.no_site}")
    lines.append(f"  ignore_environment:     {fl.ignore_environment}")
    lines.append(f"  verbose:                {fl.verbose}")
    lines.append(f"  bytes_warning:          {fl.bytes_warning}")
    lines.append(f"  quiet:                  {fl.quiet}")
    lines.append(f"  hash_randomization:     {fl.hash_randomization}")
    lines.append(f"  isolated:               {getattr(fl, 'isolated', 'N/A')}")
    lines.append(f"  dev_mode:               {getattr(fl, 'dev_mode', 'N/A')}")
    lines.append(f"  utf8_mode:              {getattr(fl, 'utf8_mode', 'N/A')}")
    lines.append(f"  warn_default_encoding:  {getattr(fl, 'warn_default_encoding', 'N/A')}")
    lines.append(f"  safe_path:              {getattr(fl, 'safe_path', 'N/A')}")
    lines.append(f"  int_max_str_digits:     {getattr(fl, 'int_max_str_digits', 'N/A')}")
    lines.append("")

    lines.append("[Optimization Config]")
    _eng_cfg = _read_engine_optimization()
    lines.append(f"  python_jit:             {_eng_cfg.get('python_jit', False)}")
    lines.append(f"  python_optimize:        {_eng_cfg.get('python_optimize', 0)}")
    lines.append(f"  python_unbuffered:      {_eng_cfg.get('python_unbuffered', False)}")
    lines.append(f"  python_no_bytecode:     {_eng_cfg.get('python_no_bytecode', False)}")
    lines.append(f"  PYTHON_JIT env:         {os.environ.get('PYTHON_JIT', '(unset)')}")
    lines.append(f"  PYTHONUNBUFFERED env:   {os.environ.get('PYTHONUNBUFFERED', '(unset)')}")
    lines.append(f"  PYTHONDONTWRITEBYTECODE env: {os.environ.get('PYTHONDONTWRITEBYTECODE', '(unset)')}")
    lines.append("")

    gl_info = {}
    gl_exts = []
    try:
        if parent is not None:
            vp = getattr(parent, '_viewport', None)
            if vp is not None:
                gl_info = getattr(vp, '_gl_info_cache', {}) or {}
                gl_exts = getattr(vp, '_gl_extensions_cache', []) or []
    except Exception:
        pass

    lines.append("[OpenGL]")
    lines.append(f"  Renderer:      {gl_info.get('GL_RENDERER', 'Unknown')}")
    lines.append(f"  Vendor:        {gl_info.get('GL_VENDOR', 'Unknown')}")
    lines.append(f"  Version:       {gl_info.get('GL_VERSION', 'Unknown')}")
    major = gl_info.get("GL_MAJOR_VERSION", "?")
    minor = gl_info.get("GL_MINOR_VERSION", "?")
    lines.append(f"  Profile:       Core {major}.{minor}")
    lines.append(f"  Double Buf:    {gl_info.get('GL_DOUBLEBUFFER', '?')}")
    lines.append(f"  Max Texture:   {gl_info.get('GL_MAX_TEXTURE_SIZE', '?')}")
    lines.append(f"  Max Viewport:  {gl_info.get('GL_MAX_VIEWPORT_DIMS', '?')}")
    lines.append(f"  Max Renderbuf: {gl_info.get('GL_MAX_RENDERBUFFER_SIZE', '?')}")
    lines.append(f"  Max Samples:   {gl_info.get('GL_MAX_SAMPLES', '?')}")
    lines.append(f"  Max Draw Bufs: {gl_info.get('GL_MAX_DRAW_BUFFERS', '?')}")
    lines.append(f"  Max Tex Units: {gl_info.get('GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS', '?')}")
    lines.append(f"  Max Vert Atts: {gl_info.get('GL_MAX_VERTEX_ATTRIBS', '?')}")
    lines.append(f"  Max UBO Bind:  {gl_info.get('GL_MAX_UNIFORM_BUFFER_BINDINGS', '?')}")
    lines.append(f"  Uniform Align: {gl_info.get('GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT', '?')}")
    lines.append("")

    if gl_exts:
        lines.append("[OpenGL Extensions]")
        for ext in sorted(gl_exts):
            lines.append(f"  {ext}")
        lines.append("")

    lines.append("[Display]")
    try:
        app = QApplication.instance()
        if app is not None:
            screens = app.screens()
            lines.append(f"  Monitors:      {len(screens)}")
            for i, scr in enumerate(screens):
                g = scr.geometry()
                dpr = scr.devicePixelRatio()
                dpi = scr.logicalDotsPerInch()
                lines.append(f"    [{i}] {scr.name()}: {g.width()}x{g.height()} @ {dpr}x  ({dpi:.0f} DPI)")
            primary = app.primaryScreen()
            if primary:
                g = primary.geometry()
                lines.append(f"  Primary:       {primary.name()} ({g.width()}x{g.height()})")
        else:
            lines.append("  (QApplication not available)")
    except Exception as e:
        lines.append(f"  (unavailable: {e})")
    lines.append("")

    lines.append("[Locale / Timezone]")
    try:
        lines.append(f"  Locale:        {platform.locale()}")
    except Exception:
        lines.append(f"  Locale:        N/A")
    try:
        lines.append(f"  Timezone:      {datetime.datetime.now().astimezone().tzinfo}")
    except Exception:
        lines.append(f"  Timezone:      N/A")
    lines.append("")

    engine = None
    try:
        if parent is not None:
            engine = getattr(parent, '_engine', None)
    except Exception:
        pass

    scene = None
    entity_count = 0
    scene_name = "None"
    if engine is not None:
        scene = getattr(engine, 'scene', None)
        if scene is not None:
            try:
                entity_count = len(scene.get_all_entities())
            except Exception:
                pass
            scene_name = getattr(scene, 'name', 'Unnamed')

    lines.append("[Engine]")
    lines.append(f"  Scene:         {scene_name}")
    lines.append(f"  Entities:      {entity_count}")

    render_mode = "N/A"
    try:
        if parent is not None:
            vp = getattr(parent, '_viewport', None)
            r = getattr(vp, '_renderer', None) if vp is not None else None
            if r is not None:
                rm = r.render_mode
                render_mode = rm.value if hasattr(rm, 'value') else str(rm)
    except Exception:
        pass
    lines.append(f"  Render Mode:   {render_mode}")

    play_mode = False
    try:
        if engine is not None:
            play_mode = getattr(engine, 'play_mode', False)
    except Exception:
        pass
    lines.append(f"  Play Mode:     {play_mode}")

    time_scale = 1.0
    try:
        if engine is not None:
            time_scale = getattr(engine, 'time_scale', 1.0)
    except Exception:
        pass
    lines.append(f"  Time Scale:    {time_scale}")
    lines.append("")

    try:
        from core.ecs.ecs import ComponentRegistry
        registered = list(ComponentRegistry._registry.keys())
        lines.append("[ECS Components]")
        lines.append(f"  Registered:    {len(registered)}")
        for name in sorted(registered):
            lines.append(f"    - {name}")
        lines.append("")
    except Exception:
        pass

    lines.append("[Plugins]")
    lines.append(f"  Count:         {AboutDialog._count_plugins()}")
    plugins_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plugins")
    try:
        for d in sorted(os.listdir(plugins_dir)):
            full = os.path.join(plugins_dir, d)
            if os.path.isdir(full) and not d.startswith("_"):
                lines.append(f"    - {d}")
    except FileNotFoundError:
        lines.append("    (none found)")
    lines.append("")

    lines.append("[Python Packages]")
    try:
        from importlib.metadata import distributions
        for dist in sorted(distributions(), key=lambda d: d.metadata["Name"].lower()):
            lines.append(f"  {dist.metadata['Name']:30s} {dist.version}")
    except Exception:
        lines.append("  (unavailable)")
    lines.append("")

    lines.append("[Environment]")
    lines.append(f"  PYTHONPATH:    {os.environ.get('PYTHONPATH', '(not set)')}")
    path_val = os.environ.get("PATH", "")
    if len(path_val) > 120:
        path_val = path_val[:120] + "..."
    lines.append(f"  PATH:          {path_val}")
    lines.append(f"  TEMP:          {os.environ.get('TEMP', 'N/A')}")
    lines.append(f"  HOME:          {os.environ.get('HOME') or os.environ.get('USERPROFILE', 'N/A')}")
    lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


class _SystemInfoDialog(QDialog):
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Info")
        self.setFixedSize(640, 520)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        lbl = QLabel("Full system information for debugging:")
        root.addWidget(lbl)

        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(text)
        te.setFont(QFont("Consolas", 9))
        te.setStyleSheet("QTextEdit { background: #1e1e1e; color: #d4d4d4; border: 1px solid #555; }")
        te.setTabStopDistance(te.fontMetrics().horizontalAdvance(" ") * 4)
        root.addWidget(te)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.setFixedWidth(150)
        copy_btn.clicked.connect(lambda: (QApplication.clipboard().setText(text), copy_btn.setText("Copied!"),
                                          QTimer.singleShot(1500, lambda: copy_btn.setText("Copy to Clipboard"))))
        btn_row.addWidget(copy_btn)
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Zarin Engine")
        self.setFixedWidth(440)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self._logo_clicks = 0
        self._gpu_name = self._detect_gpu(parent)
        self._setup_ui()

    @staticmethod
    def _detect_gpu(parent) -> str:
        try:
            vp = getattr(parent, '_viewport', None)
            if vp is not None:
                info = getattr(vp, '_gl_info_cache', None)
                if info:
                    return info.get("GL_RENDERER", "Unknown")
        except Exception:
            pass
        return "Unknown"

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(20, 20, 20, 16)

        logo_pm = _render_logo(LOGO_W)
        if logo_pm is not None:
            logo_lbl = QLabel()
            logo_lbl.setPixmap(logo_pm)
            logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
            logo_lbl.mousePressEvent = self._on_logo_click
            root.addWidget(logo_lbl)

        ver = QLabel(APP_VERSION_DISPLAY)
        ver.setFont(QFont("Segoe UI", 9))
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(ver)

        root.addWidget(self._hline())

        desc = QLabel(
            "A high-performance 64-bit ECS 3D engine with\n"
            "plugin-based architecture, real-time rendering,\n"
            "and a full-featured editor."
        )
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(desc)

        root.addWidget(self._hline())

        gil_enabled = getattr(sys, '_is_gil_enabled', lambda: True)()
        gil_status = "No (nogil)" if not gil_enabled else "Yes"
        jit_enabled = getattr(sys, '_jit_enabled', False)
        jit_status = "Active" if jit_enabled else "Off"
        flags = sys.flags
        debug_flag = "on" if flags.debug else "off"
        optimize_flag = flags.optimize
        dev_mode = getattr(flags, 'dev_mode', False)
        isolated = getattr(flags, 'isolated', False)
        utf8 = getattr(flags, 'utf8_mode', False)
        eng_cfg = _read_engine_optimization()
        jit_cfg = bool(eng_cfg.get("python_jit", False))
        opt_cfg = int(eng_cfg.get("python_optimize", 0))
        unbuf_cfg = bool(eng_cfg.get("python_unbuffered", False))
        nobc_cfg = bool(eng_cfg.get("python_no_bytecode", False))
        info = QLabel(
            f"<b>Tech Stack:</b> Python 3, ModernGL, PyQt6, NumPy, Bullet3 / PhysX<br>"
            f"<b>Plugins:</b> {self._count_plugins()} loaded<br>"
            f"<b>System:</b> {platform.system()} {platform.machine()}<br>"
            f"<b>Python:</b> {sys.version.split()[0]}<br>"
            f"<b>JIT:</b> {jit_status} (cfg: {'on' if jit_cfg else 'off'})<br>"
            f"<b>GIL:</b> {gil_status}<br>"
            f"<b>Debug:</b> {debug_flag} &nbsp; <b>Optimize:</b> {optimize_flag} (cfg: {opt_cfg})<br>"
            f"<b>Unbuffered:</b> {'on' if unbuf_cfg else 'off'} &nbsp; "
            f"<b>No-Bytecode:</b> {'on' if nobc_cfg else 'off'}<br>"
            f"<b>Dev:</b> {dev_mode} &nbsp; <b>Isolated:</b> {isolated} &nbsp; <b>UTF-8:</b> {utf8}<br>"
            f"<b>Interpreter:</b> {os.path.basename(sys.executable)}<br>"
            f"<b>Renderer:</b> OpenGL 4.6 Core Profile<br>"
            f"<b>GPU:</b> {self._gpu_name}"
        )
        info.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(info)

        root.addWidget(self._hline())

        copy_lbl = QLabel("Copyright © 2026 Zarrakun")
        copy_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(copy_lbl)

        lic_lbl = QLabel(
            f'Licensed under the <a href="{MPL_URL}">Mozilla Public License 2.0</a>'
        )
        lic_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lic_lbl.setOpenExternalLinks(True)
        root.addWidget(lic_lbl)

        root.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        info_btn = QPushButton("System Info")
        info_btn.setFixedWidth(110)
        info_btn.clicked.connect(self._open_system_info)
        btn_row.addWidget(info_btn)
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _on_logo_click(self, _event):
        self._logo_clicks += 1
        if self._logo_clicks >= _EASTER_CLICKS_NEEDED:
            self._logo_clicks = 0
            dlg = _SnakeGame(self)
            dlg.exec()

    def _open_system_info(self):
        text = _collect_system_info(self.parent())
        dlg = _SystemInfoDialog(text, self)
        dlg.exec()

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    @staticmethod
    def _count_plugins() -> int:
        plugins_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plugins")
        try:
            return len([d for d in os.listdir(plugins_dir)
                        if os.path.isdir(os.path.join(plugins_dir, d)) and not d.startswith("_")])
        except FileNotFoundError:
            return 0
