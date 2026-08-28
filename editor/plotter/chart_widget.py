# MIT License
#
# Copyright (c) 2026 Zarrakun
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations
import csv
import time
from typing import List, Optional, Tuple, Union
from PyQt6.QtWidgets import (QWidget, QSizePolicy, QFileDialog, QToolButton,
                              QHBoxLayout, QVBoxLayout, QStyle, QMenu)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QTimer, QEasingCurve
from PyQt6.QtGui import QColor, QFont, QPen, QPixmap
from .canvas_backend import make_canvas as _make_canvas
from .items import _LineItem, _ScatterItem, _FitItem, _InfLine, _FunctionItem, _RulerItem
from .sidebar import SidebarLabel
from .i18n import tr
from .math_utils import get_fit_modes, get_fit_mode
from .palette import next_line_color, next_scatter_color

_AnyItem = Union[_LineItem, _ScatterItem]
_AUTOFIT_DEBOUNCE_MS = 30
_ANIM_FRAME_MS = 16
_RAPID_THRESHOLD_S = 0.25


class ChartWidget(QWidget):
    def __init__(self, parent=None, *,
                 show_toolbar: bool = True,
                 show_legend: bool = False,
                 show_sidebar: bool = False,
                 font: Optional[QFont] = None,
                 anim_duration: int = 450,
                 anim_easing: QEasingCurve.Type = QEasingCurve.Type.OutQuint,
                 threaded_fit: bool = False,
                 grid_px_x: int = 80,
                 grid_px_y: int = 60,
                 origin_axes: bool = False,):
        super().__init__(parent)
        self._lines: List[_LineItem] = []
        self._scatters: List[_ScatterItem] = []
        self._lines_r2: List[_LineItem] = []
        self._scatters_r2: List[_ScatterItem] = []
        self._fits: List[_FitItem] = []
        self._inflines: List[_InfLine] = []
        self._functions: List[_FunctionItem] = []
        self._rulers: List[_RulerItem] = []
        self._viewport_changed_callbacks = []
        self._last_emitted_viewport: Optional[tuple] = None
        self._label_left = ""
        self._label_right = ""
        self._label_bottom = ""
        self._label_title = ""
        self._title_font: Optional[QFont] = None
        self._font = font or QFont("Arial", 8)
        self._vx0 = 0.0; self._vx1 = 1.0
        self._vy0 = 0.0; self._vy1 = 1.0
        self._vy0_r = 0.0; self._vy1_r = 1.0
        self._log_x = False
        self._log_y = False
        self._show_legend = show_legend
        self._active_fit_key: Optional[str] = None
        self._autofit_enabled = True
        self._threaded_fit = threaded_fit
        self._grid_px_x = max(20, grid_px_x)
        self._grid_px_y = max(20, grid_px_y)
        self._bounds_dirty = True
        self._bounds_cache: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
        self._bounds_r2_cache: Tuple[float, float] = (0.0, 1.0)
        self._last_autofit_t = 0.0
        self._canvas = _make_canvas(self)
        if self._canvas is None:
            raise RuntimeError(
                "make_canvas() returned None - check that editor/plotter/canvas.py "
                "exists and that _PlotCanvas can be imported from it."
            )
        self._canvas._show_origin_axes = origin_axes
        self._toolbar_layout = self._build_toolbar()
        self._toolbar_widget = QWidget(self)
        self._toolbar_widget.setLayout(self._toolbar_layout)
        self._sidebar = SidebarLabel(self) if show_sidebar else None
        self._toolbar_widget.setVisible(show_toolbar)
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._step_animation)
        self._anim_dur = anim_duration
        self._anim_easing = QEasingCurve(anim_easing)
        self._anim_start: Optional[Tuple] = None
        self._anim_target: Optional[Tuple] = None
        self._anim_elapsed = 0
        self._viewport_notify_timer = QTimer(self)
        self._viewport_notify_timer.setSingleShot(True)
        self._viewport_notify_timer.timeout.connect(self._emit_viewport_changed)
        self._autofit_timer = QTimer(self)
        self._autofit_timer.setSingleShot(True)
        self._autofit_timer.timeout.connect(self._deferred_autofit)
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        if self._sidebar:
            content.addWidget(self._sidebar)
        content.addWidget(self._canvas, 1)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._toolbar_widget)
        root.addLayout(content, 1)
        self.setMinimumSize(200, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _make_btn(self, icon_sp: str, lbl_key: str, tip_key: str, slot) -> QToolButton:
        btn = QToolButton(self)
        btn.setText(tr(lbl_key))
        btn.setToolTip(tr(tip_key))
        btn.setIcon(self.style().standardIcon(getattr(QStyle.StandardPixmap, icon_sp)))
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        btn.setFixedHeight(24)
        btn.clicked.connect(slot)
        return btn

    def _make_toggle_btn(self, lbl_key: str, tip_key: str,
                         default_checked: bool, slot) -> QToolButton:
        btn = QToolButton(self)
        btn.setText(tr(lbl_key))
        btn.setToolTip(tr(tip_key))
        btn.setCheckable(True)
        btn.setChecked(default_checked)
        btn.setFixedHeight(24)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        btn.toggled.connect(slot)
        return btn

    def _build_toolbar(self) -> QHBoxLayout:
        self._btn_autofit_toggle = self._make_toggle_btn(
            "chart_widget.btn_autofit_toggle",
            "chart_widget.btn_autofit_toggle_tip",
            True,
            self.setAutofitEnabled,
        )
        self._btn_latest_toggle = self._make_toggle_btn(
            "chart_widget.btn_latest",
            "chart_widget.btn_latest_tip",
            False,
            self.setLatestPointVisible,
        )
        self._btn_origin_axes = self._make_toggle_btn(
            "chart_widget.btn_origin_axes",
            "chart_widget.btn_origin_axes_tip",
            False,
            self.setOriginAxesVisible,
        )
        self._btn_crosshair = self._make_toggle_btn(
            "chart_widget.btn_crosshair",
            "chart_widget.btn_crosshair_tip",
            True,
            self.setCrosshairVisible,
        )
        self._btn_fit = self._make_btn("SP_FileDialogContentsView",
                                       "chart_widget.btn_fit",
                                       "chart_widget.btn_fit_tip",
                                       self.autofit)
        self._btn_stats = self._make_btn("SP_FileDialogInfoView",
                                         "chart_widget.btn_analytics",
                                         "chart_widget.btn_analytics_tip",
                                         self._canvas.toggleAnalytics)
        self._btn_csv = self._make_btn("SP_DialogSaveButton",
                                       "chart_widget.btn_csv",
                                       "chart_widget.btn_csv_tip",
                                       self.exportCsv)
        self._btn_img = self._make_btn("SP_DialogSaveButton",
                                       "chart_widget.btn_img",
                                       "chart_widget.btn_img_tip",
                                       self.exportImage)
        self._btn_fit_mode = QToolButton(self)
        self._btn_fit_mode.setToolTip(tr("chart_widget.btn_fit_mode_tip"))
        self._btn_fit_mode.setFixedHeight(24)
        self._btn_fit_mode.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._btn_fit_mode.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._fit_menu = QMenu(self._btn_fit_mode)
        none_act = QAction("None", self)
        none_act.triggered.connect(lambda: self._on_fit_mode_selected(None))
        self._fit_menu.addAction(none_act)
        for mode in get_fit_modes():
            act = QAction(mode.label, self)
            act.setData(mode.key)
            act.triggered.connect(lambda checked, k=mode.key: self._on_fit_mode_selected(k))
            self._fit_menu.addAction(act)
        self._btn_fit_mode.setMenu(self._fit_menu)
        self._update_fit_mode_label()
        tb = QHBoxLayout()
        tb.setContentsMargins(2, 2, 2, 2)
        tb.setSpacing(2)
        tb.addStretch()
        for w in (self._btn_autofit_toggle, self._btn_latest_toggle,
                  self._btn_origin_axes, self._btn_crosshair,
                  self._btn_fit, self._btn_stats, self._btn_fit_mode,
                  self._btn_csv, self._btn_img):
            tb.addWidget(w)
        return tb

    def _on_fit_mode_selected(self, key: Optional[str]):
        self._active_fit_key = key
        self._update_fit_mode_label()
        for fit in self._fits:
            if key is not None:
                fit.setModeKey(key)
        self._canvas.update()

    def _update_fit_mode_label(self):
        if self._active_fit_key is None:
            self._btn_fit_mode.setText("Approx")
            return
        mode = get_fit_mode(self._active_fit_key)
        lbl = mode.label if mode else self._active_fit_key
        self._btn_fit_mode.setText(f"{lbl} ")

    def setToolbarVisible(self, visible: bool):
        self._toolbar_widget.setVisible(visible)

    def setSidebarVisible(self, visible: bool):
        if self._sidebar:
            self._sidebar.setVisible(visible)

    def sidebar(self) -> Optional[SidebarLabel]:
        return self._sidebar

    def setLabel(self, side: str, text: str):
        if side == "left": self._label_left = text
        elif side == "bottom": self._label_bottom = text
        elif side == "right": self._label_right = text
        self._canvas.update()

    def setTitle(self, text: str, font: Optional[QFont] = None):
        self._label_title = text
        self._title_font = font
        self._canvas.update()

    def setFont(self, font: QFont):
        self._font = font
        self._canvas.update()

    def setLegendVisible(self, visible: bool):
        self._show_legend = visible
        self._canvas.update()

    def setAutofitEnabled(self, enabled: bool):
        self._autofit_enabled = enabled

    def setLatestPointVisible(self, visible: bool):
        self._canvas._show_latest = visible
        self._canvas.update()

    def setOriginAxesVisible(self, visible: bool):
        self._canvas._show_origin_axes = visible
        self._canvas.update()

    def setCrosshairVisible(self, visible: bool):
        self._canvas._crosshair_enabled = visible
        self._canvas.update()

    def setThreadedFit(self, threaded: bool):
        self._threaded_fit = threaded

    def setGridDensity(self, px_x: int, px_y: int):
        self._grid_px_x = max(20, px_x)
        self._grid_px_y = max(20, px_y)
        self._canvas.update()

    def setLogScale(self, x: Optional[bool] = None, y: Optional[bool] = None):
        if x is not None:
            self._log_x = x
        if y is not None:
            self._log_y = y
        self._canvas.update()

    def setZoomLock(self, lock: str):
        self._canvas.setZoomLock(lock)

    def setRangeSelection(self, x_lo: float, x_hi: float):
        self._canvas._range_sel_x = (x_lo, x_hi)
        self._canvas.update()

    def clearRangeSelection(self):
        self._canvas.clearRangeSelection()

    def setSnapToData(self, enabled: bool):
        self._canvas._snap_to_data = enabled
        self._canvas.update()

    def setShowDataDots(self, enabled: bool):
        self._canvas._show_dots = enabled
        self._canvas.update()

    def setShowTangent(self, enabled: bool):
        self._canvas._show_tangent = enabled
        self._canvas.update()

    def plot(self, color: Optional[str] = None, width: int = 2,
             label: str = "", dashed: bool = False,
             right_axis: bool = False) -> _LineItem:
        c = color or next_line_color()
        pen = QPen(QColor(c), width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        item = _LineItem(self, pen, label)
        if right_axis:
            self._lines_r2.append(item)
        else:
            self._lines.append(item)
        return item

    def addScatter(self, size: int = 10, color: Optional[str] = None,
                   label: str = "", right_axis: bool = False) -> _ScatterItem:
        c = color or next_scatter_color()
        item = _ScatterItem(self, size, QColor(c), label)
        if right_axis:
            self._scatters_r2.append(item)
        else:
            self._scatters.append(item)
        return item

    def addFit(self, source: _AnyItem, mode_key: Optional[str] = None,
               color: Optional[str] = None, width: int = 2,
               dashed: bool = True, label: str = "",
               show_formula: bool = False) -> _FitItem:
        key = mode_key or self._active_fit_key or "linear"
        c = color or next_line_color()
        pen = QPen(QColor(c), width,
                   Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        fit = _FitItem(self, source, key, pen, label)
        fit.show_formula = show_formula
        self._fits.append(fit)
        if self._active_fit_key is None:
            self._active_fit_key = key
            self._update_fit_mode_label()
        self._canvas.update()
        return fit

    def addLine(self, y: Optional[float] = None, x: Optional[float] = None,
                color: str = "#f39c12", width: int = 1,
                dashed: bool = True) -> _InfLine:
        horiz = y is not None
        val = y if horiz else (x if x is not None else 0.0)
        pen = QPen(QColor(color), width,
                   Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        ln = _InfLine(self, horiz, val, pen)
        self._inflines.append(ln)
        return ln

    def addFunction(self, fn, color: Optional[str] = None, width: int = 2,
                    dashed: bool = False, label: str = "",
                    resolution: float = 1.5) -> _FunctionItem:
        c = color or next_line_color()
        pen = QPen(QColor(c), width,
                   Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        item = _FunctionItem(self, fn, pen, label, resolution)
        self._functions.append(item)
        self._canvas.update()
        return item

    def addRuler(self, x0: float = 0.0, y0: float = 0.0,
                 x1: float = 1.0, y1: float = 1.0,
                 color: str = "#e74c3c", width: int = 2,
                 handle_radius: int = 6) -> _RulerItem:
        pen = QPen(QColor(color), width)
        ruler = _RulerItem(self, pen, handle_radius)
        ruler.x0, ruler.y0, ruler.x1, ruler.y1 = x0, y0, x1, y1
        self._rulers.append(ruler)
        return ruler

    def onViewportChanged(self, callback) -> None:
        if callback not in self._viewport_changed_callbacks:
            self._viewport_changed_callbacks.append(callback)

    def removeViewportChangedCallback(self, callback) -> None:
        try:
            self._viewport_changed_callbacks.remove(callback)
        except ValueError:
            pass

    def _notify_viewport_changed(self):
        vp = (self._vx0, self._vx1, self._vy0, self._vy1)
        if vp == self._last_emitted_viewport:
            return
        self._last_emitted_viewport = vp
        for cb in self._viewport_changed_callbacks:
            try:
                cb(self._vx0, self._vx1, self._vy0, self._vy1)
            except Exception:
                pass

    def _schedule_viewport_changed(self):
        if self._viewport_notify_timer.isActive():
            return
        self._viewport_notify_timer.start(0)

    def _emit_viewport_changed(self):
        self._notify_viewport_changed()

    @property
    def viewport(self):
        class _VP:
            __slots__ = ("x0", "x1", "y0", "y1")
            def __init__(self, x0, x1, y0, y1):
                self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
            def __repr__(self):
                return (f"Viewport(x0={self.x0}, x1={self.x1}, "
                        f"y0={self.y0}, y1={self.y1})")
        return _VP(self._vx0, self._vx1, self._vy0, self._vy1)

    def setViewport(self, x0: float, x1: float, y0: float, y1: float,
                    animated: bool = False):
        self._autofit_timer.stop()
        if animated:
            self._anim_start = (self._vx0, self._vx1, self._vy0, self._vy1)
            self._anim_target = (x0, x1, y0, y1)
            self._anim_elapsed = 0
            if self._anim_start != self._anim_target:
                self._anim_timer.start(_ANIM_FRAME_MS)
                return
        self._vx0, self._vx1, self._vy0, self._vy1 = x0, x1, y0, y1
        self._canvas.update()

    def removeItem(self, item):
        for lst in (self._lines, self._scatters, self._fits, self._inflines,
                    self._lines_r2, self._scatters_r2,
                    self._functions, self._rulers):
            if item in lst:
                lst.remove(item)
        self._bounds_dirty = True
        self._canvas.update()

    def clearAll(self):
        self._lines.clear()
        self._scatters.clear()
        self._fits.clear()
        self._inflines.clear()
        self._lines_r2.clear()
        self._scatters_r2.clear()
        self._functions.clear()
        self._rulers.clear()
        self._bounds_dirty = True
        self._canvas.update()

    def _data_bounds(self) -> Tuple[float, float, float, float]:
        if not self._bounds_dirty:
            return self._bounds_cache
        items_main = [it for it in self._lines + self._scatters if it.xs]
        if not items_main:
            self._bounds_cache = (0.0, 1.0, 0.0, 1.0)
        else:
            x0 = min(min(it.xs) for it in items_main)
            x1 = max(max(it.xs) for it in items_main)
            y0 = min(min(it.ys) for it in items_main)
            y1 = max(max(it.ys) for it in items_main)
            for ln in self._inflines:
                if ln.visible:
                    if ln.horizontal:
                        y0 = min(y0, ln.value)
                        y1 = max(y1, ln.value)
                    else:
                        x0 = min(x0, ln.value)
                        x1 = max(x1, ln.value)
            if x0 == x1:
                x0 -= 1.0; x1 += 1.0
            if y0 == y1:
                y0 -= 1.0; y1 += 1.0
            px = (x1 - x0) * 0.05
            py = (y1 - y0) * 0.08
            self._bounds_cache = (x0 - px, x1 + px, y0 - py, y1 + py)
        items_r2 = [it for it in self._lines_r2 + self._scatters_r2 if it.xs]
        if items_r2:
            y0r = min(min(it.ys) for it in items_r2)
            y1r = max(max(it.ys) for it in items_r2)
            if y0r == y1r:
                y0r -= 1.0; y1r += 1.0
            py = (y1r - y0r) * 0.08
            self._bounds_r2_cache = (y0r - py, y1r + py)
        else:
            self._bounds_r2_cache = (0.0, 1.0)
        self._bounds_dirty = False
        return self._bounds_cache

    def _schedule_autofit(self):
        self._bounds_dirty = True
        if self._autofit_enabled:
            self._autofit_timer.start(_AUTOFIT_DEBOUNCE_MS)

    def _deferred_autofit(self):
        if self._autofit_enabled:
            self._run_autofit(animated=False)

    def autofit(self):
        self._autofit_timer.stop()
        self._bounds_dirty = True
        self._run_autofit(animated=True)

    def _run_autofit(self, animated: bool = True):
        tgt = self._data_bounds()
        self._vy0_r, self._vy1_r = self._bounds_r2_cache
        now = time.monotonic()
        rapid = (now - self._last_autofit_t) < _RAPID_THRESHOLD_S
        self._last_autofit_t = now
        if not animated or rapid:
            self._anim_timer.stop()
            self._vx0, self._vx1, self._vy0, self._vy1 = tgt
            self._canvas.update()
            return
        self._anim_start = (self._vx0, self._vx1, self._vy0, self._vy1)
        self._anim_target = tgt
        self._anim_elapsed = 0
        if self._anim_start != self._anim_target:
            self._anim_timer.start(_ANIM_FRAME_MS)
        else:
            self._vx0, self._vx1, self._vy0, self._vy1 = tgt
            self._canvas.update()

    def _step_animation(self):
        self._anim_elapsed += _ANIM_FRAME_MS
        p = min(1.0, self._anim_elapsed / self._anim_dur)
        f = self._anim_easing.valueForProgress(p)
        bounds = [s + (t - s) * f for s, t in zip(self._anim_start, self._anim_target)]
        self._vx0, self._vx1, self._vy0, self._vy1 = bounds
        self._canvas.update()
        if p >= 1.0:
            self._anim_timer.stop()

    def update(self):
        self._canvas.update()
        super().update()

    def refreshFitMenu(self):
        self._fit_menu.clear()
        none_act = QAction("None", self)
        none_act.triggered.connect(lambda: self._on_fit_mode_selected(None))
        self._fit_menu.addAction(none_act)
        for mode in get_fit_modes():
            act = QAction(mode.label, self)
            act.setData(mode.key)
            act.triggered.connect(lambda checked, k=mode.key: self._on_fit_mode_selected(k))
            self._fit_menu.addAction(act)

    def exportCsv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, tr("chart_widget.csv_title"), "chart_data.csv",
            tr("chart_widget.csv_filter"))
        if not path:
            return
        series = []
        for i, it in enumerate(self._lines + self._lines_r2):
            if it.xs:
                n = it.label or f"line{i}"
                series.append((f"{n}_x", f"{n}_y", it.xs, it.ys))
        for i, it in enumerate(self._scatters + self._scatters_r2):
            if it.xs:
                n = it.label or f"scatter{i}"
                series.append((f"{n}_x", f"{n}_y", it.xs, it.ys))
        if not series:
            return
        max_rows = max(len(s[2]) for s in series)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([col for s in series for col in (s[0], s[1])])
            for row in range(max_rows):
                w.writerow([
                    v for s in series
                    for v in (s[2][row] if row < len(s[2]) else "",
                              s[3][row] if row < len(s[3]) else "")
                ])

    def exportImage(self):
        path, _ = QFileDialog.getSaveFileName(
            self, tr("chart_widget.img_title"), "chart.png",
            tr("chart_widget.img_filter"))
        if not path:
            return
        self.grab().save(path)

    def grabImage(self) -> QPixmap:
        return self.grab()

    @property
    def vx0(self): return self._vx0
    @property
    def vx1(self): return self._vx1
    @property
    def vy0(self): return self._vy0
    @property
    def vy1(self): return self._vy1
    @property
    def vy0_r(self): return self._vy0_r
    @property
    def vy1_r(self): return self._vy1_r
    @property
    def log_x(self): return self._log_x
    @property
    def log_y(self): return self._log_y
    @property
    def font(self): return self._font
    @property
    def fits(self): return self._fits
    @property
    def lines(self): return self._lines
    @property
    def lines_r2(self): return self._lines_r2
    @property
    def scatters(self): return self._scatters
    @property
    def scatters_r2(self): return self._scatters_r2
    @property
    def inflines(self): return self._inflines
    @property
    def show_legend(self): return self._show_legend
    @property
    def label_left(self): return self._label_left
    @property
    def label_right(self): return self._label_right
    @property
    def label_bottom(self): return self._label_bottom
    @property
    def label_title(self): return self._label_title
    @property
    def title_font(self): return self._title_font
    @property
    def grid_px_x(self): return self._grid_px_x
    @property
    def grid_px_y(self): return self._grid_px_y
    @property
    def functions(self): return self._functions
    @property
    def rulers(self): return self._rulers