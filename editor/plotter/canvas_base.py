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
import math
import statistics
from bisect import bisect_left
from typing import List, Optional, Tuple, TYPE_CHECKING
from PyQt6.QtCore import Qt, QRect, QRectF, QPointF
from PyQt6.QtGui import (QPen, QColor, QBrush, QFont, QFontMetrics,
                         QPixmap, QWheelEvent, QMouseEvent)
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QMenu
from .math_utils import nice_ticks, nice_log_ticks, to_log, decimated, fmt, get_fit_modes
from .items import _LineItem, _FitItem, _InfLine, _FunctionItem
from .i18n import tr

if TYPE_CHECKING:
    from .chart_widget import ChartWidget

_ML, _MT, _MR, _MB = 58, 14, 20, 40
_MT_TITLE = 26
_MR2 = 52
_ZOOM_FACTOR = 1.15
_ZOOM_MIN_SPAN = 1e-10
_ZOOM_MAX_SPAN = 1e15
_SNAP_RADIUS_PX = 40
_TANGENT_HALF_FRAC = 0.18
_ANALYTICS_PAD = 8
_ANALYTICS_ROW_H = 17
_ANALYTICS_MAX_SERIES = 8
_TOOLTIP_MARGIN = 14
_SNAP_DOT_R = 5.0
_LEGEND_PAD = 8
_LEGEND_SWATCH = 12
_LATEST_TAG_PAD = 3
_LATEST_TAG_ROUND = 3
_COORD_CLAMP = 1e6
_DECIMATE_THRESHOLD = 2000
_INFLINE_HIT_PX = 6
_RUBBERBAND_MIN_PX = 4
_RANGE_SEL_ALPHA = 30
_SCREEN_Y_CLAMP = 32768.0
_GRID_PRESETS_X = [
    ("chart_widget.ctx_sparse", 120),
    ("chart_widget.ctx_normal", 80),
    ("chart_widget.ctx_dense", 50),
]
_GRID_PRESETS_Y = [
    ("chart_widget.ctx_sparse", 100),
    ("chart_widget.ctx_normal", 60),
    ("chart_widget.ctx_dense", 40),
]
_ORIGIN_AXIS_ALPHA = 160
_ORIGIN_TICK_LEN = 4
_ORIGIN_TICK_STEP_PX = 60


def _fmt_axis(v: float) -> str:
    if not math.isfinite(v):
        return str(v)
    if v == 0.0:
        return "0"
    abs_v = abs(v)
    if abs_v >= 1e15:
        return f"{v:.6g}"
    if abs_v >= 1:
        if abs_v >= 1000:
            return f"{v:.0f}"
        return f"{v:.6g}"
    for digits in range(1, 10):
        s = f"{v:.{digits}f}".rstrip("0").rstrip(".")
        if float(s) != 0.0:
            return s
    return fmt(v)


def _skewness(data: List[float]) -> Optional[float]:
    n = len(data)
    if n < 3:
        return None
    mean = statistics.mean(data)
    std = statistics.stdev(data)
    if std == 0:
        return 0.0
    return sum(((x - mean) / std) ** 3 for x in data) * n / ((n - 1) * (n - 2))


class CanvasBase:
    _chart: "ChartWidget"

    def _init_base_state(self):
        self._pan_start: Optional[QPointF] = None
        self._pan_vx0 = 0.0
        self._pan_vy0 = 0.0
        self._mouse_pos: Optional[QPointF] = None
        self._show_analytics = False
        self._show_latest = False
        self._show_origin_axes = False
        self._show_dots = False
        self._show_tangent = True
        self._rb_start: Optional[QPointF] = None
        self._rb_end: Optional[QPointF] = None
        self._rb_active = False
        self._dragging_infline: Optional[_InfLine] = None
        self._zoom_lock: str = "both"
        self._range_sel_x: Optional[Tuple[float, float]] = None
        self._nearest_cache_key: Optional[tuple] = None
        self._nearest_cache_result = None
        self._dragging_ruler_pt = None
        self._snap_to_data = True
        self._crosshair_enabled = True

    def toggleAnalytics(self):
        self._show_analytics = not self._show_analytics
        self.update()

    def toggleLatestPoint(self):
        self._show_latest = not self._show_latest
        self.update()

    def toggleOriginAxes(self):
        self._show_origin_axes = not self._show_origin_axes
        self.update()

    def toggleCrosshair(self):
        self._crosshair_enabled = not self._crosshair_enabled
        self.update()

    def _set_analytics(self, v: bool):
        self._show_analytics = v
        self.update()

    def _set_latest(self, v: bool):
        self._show_latest = v
        self.update()

    def _set_origin_axes(self, v: bool):
        self._show_origin_axes = v
        self.update()

    def _set_crosshair(self, v: bool):
        self._crosshair_enabled = v
        self.update()

    def _set_show_dots(self, v: bool):
        self._show_dots = v
        self.update()

    def _set_show_tangent(self, v: bool):
        self._show_tangent = v
        self.update()

    def _set_snap_to_data(self, v: bool):
        self._snap_to_data = v
        self.update()

    def setZoomLock(self, lock: str):
        self._zoom_lock = lock

    def clearRangeSelection(self):
        self._range_sel_x = None
        self.update()

    def _has_right_axis(self) -> bool:
        return bool(self._chart.lines_r2 or self._chart.scatters_r2)

    def _plot_rect(self) -> QRect:
        ml = _ML + (4 if self._chart.label_left else 0)
        mr = (_MR2 if self._has_right_axis() else _MR) + (4 if self._chart.label_right else 0)
        mt = (_MT_TITLE if self._chart.label_title else _MT)
        return QRect(ml, mt, max(1, self.width() - ml - mr), max(1, self.height() - mt - _MB))

    def _to_pt(self, xv, yv, x0, dx, y0, dy, pr) -> QPointF:
        px = pr.left() + (xv - x0) / dx * pr.width()
        py = pr.bottom() - (yv - y0) / dy * pr.height()
        return QPointF(
            max(-_COORD_CLAMP, min(_COORD_CLAMP, px)),
            max(-_COORD_CLAMP, min(_COORD_CLAMP, py)),
        )

    def _to_pt_log(self, xv, yv, x0, dx, y0, dy, pr, log_x: bool, log_y: bool) -> QPointF:
        lx = to_log(xv) if log_x else xv
        ly = to_log(yv) if log_y else yv
        return self._to_pt(lx, ly, x0, dx, y0, dy, pr)

    def _screen_to_data(self, sx, sy, x0, dx, y0, dy, pr, log_x, log_y) -> Tuple[float, float]:
        lx = x0 + (sx - pr.left()) / pr.width() * dx
        ly = y0 + (pr.bottom() - sy) / pr.height() * dy
        xv = 10 ** lx if log_x else lx
        yv = 10 ** ly if log_y else ly
        return xv, yv

    def _view_params(self, pr):
        c = self._chart
        x0 = to_log(c.vx0) if c.log_x else c.vx0
        x1 = to_log(c.vx1) if c.log_x else c.vx1
        y0 = to_log(c.vy0) if c.log_y else c.vy0
        y1 = to_log(c.vy1) if c.log_y else c.vy1
        dx = x1 - x0 if abs(x1 - x0) > 1e-300 else 1.0
        dy = y1 - y0 if abs(y1 - y0) > 1e-300 else 1.0
        return x0, x1, y0, y1, dx, dy

    def _nearest_on_segments(self, mouse, xs, ys, pr, x0, dx, y0, dy, log_x, log_y):
        if not xs:
            return None
        if len(xs) == 1:
            pt = self._to_pt_log(xs[0], ys[0], x0, dx, y0, dy, pr, log_x, log_y)
            return (xs[0], ys[0], math.hypot(pt.x() - mouse.x(), pt.y() - mouse.y()))
        best_d = float("inf")
        best_xi, best_yi = xs[0], ys[0]
        for i in range(len(xs) - 1):
            pt0 = self._to_pt_log(xs[i], ys[i], x0, dx, y0, dy, pr, log_x, log_y)
            pt1 = self._to_pt_log(xs[i + 1], ys[i + 1], x0, dx, y0, dy, pr, log_x, log_y)
            seg_x = pt1.x() - pt0.x()
            seg_y = pt1.y() - pt0.y()
            seg_len2 = seg_x * seg_x + seg_y * seg_y
            t = 0.0 if seg_len2 < 1e-10 else max(0.0, min(1.0,
                ((mouse.x() - pt0.x()) * seg_x + (mouse.y() - pt0.y()) * seg_y) / seg_len2))
            nx = pt0.x() + t * seg_x
            ny = pt0.y() + t * seg_y
            d = math.hypot(nx - mouse.x(), ny - mouse.y())
            if d < best_d:
                best_d = d
                best_xi = xs[i] + t * (xs[i + 1] - xs[i])
                best_yi = ys[i] + t * (ys[i + 1] - ys[i])
        return (best_xi, best_yi, best_d)

    def _find_nearest(self, mouse, pr, x0, dx, y0, dy):
        if not self._snap_to_data:
            return None
        key = (int(mouse.x()), int(mouse.y()), x0, dx, y0, dy)
        if self._nearest_cache_key == key:
            return self._nearest_cache_result
        c = self._chart
        log_x, log_y = c.log_x, c.log_y
        best_d = float("inf")
        best = None
        for item in c.scatters:
            if not item.visible:
                continue
            for xi, yi in zip(item.xs, item.ys):
                pt = self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, log_x, log_y)
                d = math.hypot(pt.x() - mouse.x(), pt.y() - mouse.y())
                if d < best_d:
                    best_d = d
                    best = (xi, yi, d, item)
        for item in c.lines:
            if not item.visible or not item.xs:
                continue
            dxs, dys = decimated(item.xs, item.ys, _DECIMATE_THRESHOLD)
            r = self._nearest_on_segments(mouse, dxs, dys, pr, x0, dx, y0, dy, log_x, log_y)
            if r and r[2] < best_d:
                best_d = r[2]
                best = (r[0], r[1], r[2], item)
        for item in c.fits:
            if not item.visible or not item.xs:
                continue
            r = self._nearest_on_segments(mouse, item.xs, item.ys, pr, x0, dx, y0, dy, log_x, log_y)
            if r and r[2] < best_d:
                best_d = r[2]
                best = (r[0], r[1], r[2], item)
        for item in c.functions:
            if not item.visible or not item.xs:
                continue
            fn_xs_f = [item.xs[i] for i in range(len(item.xs)) if item.ys[i] is not None]
            fn_ys_f = [item.ys[i] for i in range(len(item.ys)) if item.ys[i] is not None]
            if not fn_xs_f:
                continue
            r = self._nearest_on_segments(mouse, fn_xs_f, fn_ys_f, pr, x0, dx, y0, dy, log_x, log_y)
            if r and r[2] < best_d:
                best_d = r[2]
                best = (r[0], r[1], r[2], item)
        result = best if best and best[2] <= _SNAP_RADIUS_PX else None
        self._nearest_cache_key = key
        self._nearest_cache_result = result
        return result

    def _find_ruler_handle_at(self, pos, pr, x0, dx, y0, dy):
        c = self._chart
        log_x, log_y = c.log_x, c.log_y
        for ruler in c.rulers:
            if not ruler.visible or not ruler.draggable:
                continue
            for idx, (xv, yv) in enumerate(((ruler.x0, ruler.y0), (ruler.x1, ruler.y1))):
                pt = self._to_pt_log(xv, yv, x0, dx, y0, dy, pr, log_x, log_y)
                if math.hypot(pt.x() - pos.x(), pt.y() - pos.y()) <= ruler.handle_radius + 4:
                    return (ruler, idx)
        return None

    def _find_infline_at(self, pos, pr, x0, dx, y0, dy):
        c = self._chart
        for ln in c.inflines:
            if not ln.visible or not ln.draggable:
                continue
            if ln.horizontal:
                sy = pr.bottom() - (ln.value - y0) / dy * pr.height()
                if abs(pos.y() - sy) <= _INFLINE_HIT_PX and pr.left() <= pos.x() <= pr.right():
                    return ln
            else:
                sx = pr.left() + (ln.value - x0) / dx * pr.width()
                if abs(pos.x() - sx) <= _INFLINE_HIT_PX and pr.top() <= pos.y() <= pr.bottom():
                    return ln
        return None

    def _find_scatter_point_at(self, pos, pr, x0, dx, y0, dy):
        c = self._chart
        log_x, log_y = c.log_x, c.log_y
        for item in c.scatters:
            if not item.visible:
                continue
            for i, (xi, yi) in enumerate(zip(item.xs, item.ys)):
                pt = self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, log_x, log_y)
                if math.hypot(pt.x() - pos.x(), pt.y() - pos.y()) <= item.size:
                    return (item, i)
        return None

    def _tangent_slope(self, item, xi):
        if isinstance(item, _FunctionItem):
            pairs = [(x, y) for x, y in zip(item.xs, item.ys) if y is not None]
            if len(pairs) < 2:
                return None
            xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        elif isinstance(item, (_LineItem, _FitItem)) and len(item.xs) >= 2:
            xs, ys = item.xs, item.ys
        else:
            return None
        n = len(xs)
        pos = bisect_left(xs, xi)
        if pos >= n:
            pos = n - 1
        elif pos > 0 and abs(xs[pos - 1] - xi) < abs(xs[pos] - xi):
            pos -= 1
        if pos == 0:
            ddx, ddy = xs[1] - xs[0], ys[1] - ys[0]
        elif pos == n - 1:
            ddx, ddy = xs[-1] - xs[-2], ys[-1] - ys[-2]
        else:
            ddx, ddy = xs[pos + 1] - xs[pos - 1], ys[pos + 1] - ys[pos - 1]
        return ddy / ddx if abs(ddx) > 1e-15 else None

    def _paint_origin_axes(self, p, pr, x0, dx, y0, dy, fg):
        c = self._chart
        log_x, log_y = c.log_x, c.log_y
        fm = QFontMetrics(c.font)
        axis_col = QColor(fg)
        axis_col.setAlpha(_ORIGIN_AXIS_ALPHA)
        tick_col = QColor(fg)
        tick_col.setAlpha(120)
        lbl_col = QColor(fg)
        lbl_col.setAlpha(180)
        p.save()
        p.setClipRect(pr)
        origin_x_screen = pr.left() + (0.0 - x0) / dx * pr.width() if not log_x else None
        origin_y_screen = pr.bottom() - (0.0 - y0) / dy * pr.height() if not log_y else None
        if origin_x_screen is not None and pr.left() <= origin_x_screen <= pr.right():
            sx = int(origin_x_screen)
            p.setPen(QPen(axis_col, 1.5))
            p.drawLine(sx, pr.top(), sx, pr.bottom())
            n_y = max(2, pr.height() // max(1, c.grid_px_y))
            yt = nice_ticks(y0, y0 + dy, n_y) if not log_y else nice_log_ticks(10 ** y0, 10 ** (y0 + dy))
            p.setPen(QPen(tick_col, 1))
            p.setFont(c.font)
            for tv in yt:
                ty_log = to_log(tv) if log_y else tv
                sy = int(pr.bottom() - (ty_log - y0) / dy * pr.height())
                if pr.top() <= sy <= pr.bottom() and abs(tv) > 1e-12:
                    p.drawLine(sx - _ORIGIN_TICK_LEN, sy, sx + _ORIGIN_TICK_LEN, sy)
                    lbl = _fmt_axis(tv)
                    lw = fm.horizontalAdvance(lbl)
                    lx_pos = sx + _ORIGIN_TICK_LEN + 2
                    if lx_pos + lw > pr.right():
                        lx_pos = sx - lw - _ORIGIN_TICK_LEN - 2
                    p.setPen(lbl_col)
                    p.drawText(lx_pos, sy + fm.ascent() // 2, lbl)
                    p.setPen(QPen(tick_col, 1))
            p.setPen(lbl_col)
            lbl_y = tr("chart_widget.origin_y_label")
            p.drawText(sx + 4, pr.top() + fm.height(), lbl_y)
        if origin_y_screen is not None and pr.top() <= origin_y_screen <= pr.bottom():
            sy = int(origin_y_screen)
            p.setPen(QPen(axis_col, 1.5))
            p.drawLine(pr.left(), sy, pr.right(), sy)
            n_x = max(2, pr.width() // max(1, c.grid_px_x))
            xt = nice_ticks(x0, x0 + dx, n_x) if not log_x else nice_log_ticks(10 ** x0, 10 ** (x0 + dx))
            p.setPen(QPen(tick_col, 1))
            p.setFont(c.font)
            for tv in xt:
                tx_log = to_log(tv) if log_x else tv
                sx_t = int(pr.left() + (tx_log - x0) / dx * pr.width())
                if pr.left() <= sx_t <= pr.right() and abs(tv) > 1e-12:
                    p.drawLine(sx_t, sy - _ORIGIN_TICK_LEN, sx_t, sy + _ORIGIN_TICK_LEN)
                    lbl = _fmt_axis(tv)
                    lw = fm.horizontalAdvance(lbl)
                    ly_pos = sy + fm.height() + 2
                    if ly_pos + fm.height() > pr.bottom():
                        ly_pos = sy - 4
                    p.setPen(lbl_col)
                    p.drawText(sx_t - lw // 2, ly_pos, lbl)
                    p.setPen(QPen(tick_col, 1))
            p.setPen(lbl_col)
            lbl_x = tr("chart_widget.origin_x_label")
            p.drawText(pr.right() - fm.horizontalAdvance(lbl_x) - 4, sy - 4, lbl_x)
        if (origin_x_screen is not None and pr.left() <= origin_x_screen <= pr.right() and
                origin_y_screen is not None and pr.top() <= origin_y_screen <= pr.bottom()):
            dot_col = QColor(fg)
            dot_col.setAlpha(200)
            p.setPen(QPen(dot_col, 1))
            p.setBrush(QBrush(dot_col))
            p.drawEllipse(QPointF(origin_x_screen, origin_y_screen), 3.0, 3.0)
        p.restore()

    def _paint_data_dots(self, p, pr, x0, dx, y0, dy):
        c = self._chart
        log_x, log_y = c.log_x, c.log_y
        p.save()
        p.setClipRect(pr)
        for item in c.lines:
            if not item.visible or not item.xs:
                continue
            col = item.pen.color()
            p.setPen(QPen(col.darker(140), 1))
            p.setBrush(QBrush(col))
            for xi, yi in zip(item.xs, item.ys):
                pt = self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, log_x, log_y)
                if (pr.left() - 4 <= pt.x() <= pr.right() + 4 and
                        pr.top() - 4 <= pt.y() <= pr.bottom() + 4):
                    p.drawEllipse(pt, 2.5, 2.5)
        p.restore()

    def _paint_crosshair_lines(self, p, pr, snap, fg):
        ch_col = QColor(fg)
        ch_col.setAlpha(80)
        p.setPen(QPen(ch_col, 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(int(snap.x()), pr.top(), int(snap.x()), pr.bottom())
        p.drawLine(pr.left(), int(snap.y()), pr.right(), int(snap.y()))

    def _paint_snap_dot(self, p, snap, fg):
        dot_col = QColor(fg)
        dot_col.setAlpha(220)
        p.setPen(QPen(dot_col, 1))
        p.setBrush(QBrush(dot_col))
        p.drawEllipse(snap, _SNAP_DOT_R, _SNAP_DOT_R)

    def _paint_crosshair_axis_labels(self, p, pr, xi, yi, snap, fm, fg, bg):
        pad = 4
        th = fm.height() + pad
        rnd = 3
        bg_ = QColor(bg)
        bg_.setAlpha(215)
        br_ = QColor(fg)
        br_.setAlpha(100)
        lbl_col = QColor(fg)
        lbl_col.setAlpha(220)
        x_lbl = _fmt_axis(xi)
        xlw = fm.horizontalAdvance(x_lbl)
        x_tw = xlw + pad * 2
        xtx = int(snap.x()) - x_tw // 2
        xty = pr.bottom() + 2
        if pr.left() <= snap.x() <= pr.right():
            p.setBrush(QBrush(bg_))
            p.setPen(QPen(br_, 1))
            p.drawRoundedRect(xtx, xty, x_tw, th, rnd, rnd)
            p.setPen(lbl_col)
            p.drawText(xtx + pad, xty + fm.ascent() + pad // 2, x_lbl)
        y_lbl = _fmt_axis(yi)
        ylw = fm.horizontalAdvance(y_lbl)
        y_tw = ylw + pad * 2
        ytx = pr.left() - y_tw - 4
        yty = int(snap.y()) - th // 2
        if pr.top() <= snap.y() <= pr.bottom():
            p.setBrush(QBrush(bg_))
            p.setPen(QPen(br_, 1))
            p.drawRoundedRect(ytx, yty, y_tw, th, rnd, rnd)
            p.setPen(lbl_col)
            p.drawText(ytx + pad, yty + fm.ascent() + pad // 2, y_lbl)

    def _paint_crosshair_overlay(self, p, pr, x0, dx, y0, dy, fm, fg, bg):
        mp = self._mouse_pos
        c = self._chart
        nearest = self._find_nearest(mp, pr, x0, dx, y0, dy)
        if nearest is None:
            return
        xi, yi, _, item = nearest
        snap = self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, c.log_x, c.log_y)
        self._paint_crosshair_lines(p, pr, snap, fg)
        self._paint_snap_dot(p, snap, fg)
        if self._show_tangent:
            slope = self._tangent_slope(item, xi)
            if slope is not None:
                half = (c.vx1 - c.vx0) * _TANGENT_HALF_FRAC
                tp0 = self._to_pt_log(xi - half, yi - slope * half, x0, dx, y0, dy, pr, c.log_x, c.log_y)
                tp1 = self._to_pt_log(xi + half, yi + slope * half, x0, dx, y0, dy, pr, c.log_x, c.log_y)
                tg_col = QColor(fg)
                tg_col.setAlpha(140)
                p.setPen(QPen(tg_col, 1, Qt.PenStyle.DotLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawLine(tp0, tp1)
        self._paint_tooltip(p, pr, xi, yi, snap, fg, bg, fm, item)
        self._paint_crosshair_axis_labels(p, pr, xi, yi, snap, fm, fg, bg)

    def _paint_tooltip(self, p, pr, xi, yi, snap, fg, bg, fm, item=None):
        p.setFont(self._chart.font)
        slope = None
        if self._show_tangent and item is not None:
            slope = self._tangent_slope(item, xi)
        series_lbl = ""
        if item is not None and hasattr(item, "label") and item.label:
            series_lbl = item.label
        lx = f"{tr('chart_widget.tooltip_x')}: {fmt(xi)}"
        ly = f"{tr('chart_widget.tooltip_y')}: {fmt(yi)}"
        lines = [lx, ly]
        if slope is not None:
            lines.append(f"{tr('chart_widget.tooltip_slope')}: {fmt(slope)}")
        if series_lbl:
            lines.insert(0, series_lbl)
        tw = max(fm.horizontalAdvance(l) for l in lines) + 16
        th = fm.height() * len(lines) + 12
        tx = int(snap.x()) + _TOOLTIP_MARGIN
        ty = int(snap.y()) - th - 4
        if tx + tw > pr.right():
            tx = int(snap.x()) - tw - _TOOLTIP_MARGIN
        if ty < pr.top():
            ty = int(snap.y()) + 8
        bg_ = QColor(bg)
        bg_.setAlpha(215)
        br_ = QColor(fg)
        br_.setAlpha(100)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(br_, 1))
        p.drawRoundedRect(tx, ty, tw, th, 4, 4)
        p.setPen(fg)
        for i, line in enumerate(lines):
            p.drawText(tx + 8, ty + fm.ascent() + 4 + i * fm.height(), line)

    def _paint_analytics(self, p, pr, fg, bg, fm):
        from .math_utils import trapezoid_integral
        c = self._chart
        named = []
        for i, it in enumerate(c.lines):
            if it.xs and it.visible:
                named.append((it.label or tr("chart_widget.analytics_line", n=i + 1), it))
        for i, it in enumerate(c.scatters):
            if it.xs and it.visible:
                named.append((it.label or tr("chart_widget.analytics_scatter", n=i + 1), it))
        named = named[:_ANALYTICS_MAX_SERIES]
        if not named:
            return
        range_lo, range_hi = None, None
        if self._range_sel_x is not None:
            range_lo, range_hi = sorted(self._range_sel_x)
        row_keys = [
            "chart_widget.analytics_n",
            "chart_widget.analytics_xmin",
            "chart_widget.analytics_xmax",
            "chart_widget.analytics_ymin",
            "chart_widget.analytics_ymax",
            "chart_widget.analytics_mean",
            "chart_widget.analytics_median",
            "chart_widget.analytics_std",
            "chart_widget.analytics_range",
            "chart_widget.analytics_skew",
            "chart_widget.analytics_integral",
        ]
        row_lbls = [tr(k) for k in row_keys]
        table: List[List[str]] = []
        for _, it in named:
            if range_lo is not None and range_hi is not None:
                pairs = [(x, y) for x, y in zip(it.xs, it.ys) if range_lo <= x <= range_hi]
                fxs = [pair[0] for pair in pairs]
                fys = [pair[1] for pair in pairs]
            else:
                fxs, fys = list(it.xs), list(it.ys)
            n = len(fxs)
            if n == 0:
                table.append(["-"] * len(row_keys))
                continue
            fys_finite = [v for v in fys if v is not None and math.isfinite(v)]
            n_fin = len(fys_finite)
            st = fmt(statistics.stdev(fys_finite)) if n_fin > 1 else "-"
            med = fmt(statistics.median(fys_finite)) if n_fin >= 1 else "-"
            rng = fmt(max(fys_finite) - min(fys_finite)) if n_fin >= 1 else "-"
            sk = _skewness(fys_finite)
            skew_s = fmt(sk) if sk is not None else "-"
            intg = fmt(trapezoid_integral(fxs, fys_finite))
            table.append([
                str(n),
                fmt(min(fxs)), fmt(max(fxs)),
                fmt(min(fys_finite)) if fys_finite else "-",
                fmt(max(fys_finite)) if fys_finite else "-",
                fmt(statistics.mean(fys_finite)) if n_fin >= 1 else "-",
                med, st, rng, skew_s, intg,
            ])
        lbl_w = max(fm.horizontalAdvance(l) for l in row_lbls) + 10
        val_w = max(fm.horizontalAdvance(v) for row in table for v in row) + 10
        hdr_h = fm.height() + 6
        rh = _ANALYTICS_ROW_H
        pad = _ANALYTICS_PAD
        n_ser = len(named)
        total_w = pad * 2 + lbl_w + val_w * n_ser
        total_h = pad * 2 + hdr_h + rh * len(row_lbls)
        ax = pr.right() - total_w - 4
        ay = pr.top() + 4
        bg_ = QColor(bg)
        bg_.setAlpha(210)
        brd_ = QColor(fg)
        brd_.setAlpha(70)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(brd_, 1))
        p.drawRoundedRect(ax, ay, total_w, total_h, 4, 4)
        bold_f = QFont(self._chart.font)
        bold_f.setBold(True)
        hdr_col = QColor(fg)
        hdr_col.setAlpha(220)
        lbl_col = QColor(fg)
        lbl_col.setAlpha(150)
        for ci, (name, _) in enumerate(named):
            cx = ax + pad + lbl_w + ci * val_w + val_w // 2
            p.setFont(bold_f)
            p.setPen(hdr_col)
            p.drawText(cx - fm.horizontalAdvance(name) // 2, ay + pad + fm.ascent(), name)
        p.setFont(self._chart.font)
        for ri, lbl in enumerate(row_lbls):
            ry = ay + pad + hdr_h + ri * rh
            sep_col = QColor(fg)
            sep_col.setAlpha(20)
            if ri % 2 == 0:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(sep_col))
                p.drawRect(ax + 1, ry - 1, total_w - 2, rh)
            p.setPen(lbl_col)
            p.drawText(ax + pad, ry + fm.ascent(), lbl)
            p.setPen(hdr_col)
            for ci, row in enumerate(table):
                val = row[ri]
                cx = ax + pad + lbl_w + ci * val_w
                vw = fm.horizontalAdvance(val)
                p.drawText(cx + (val_w - vw) // 2, ry + fm.ascent(), val)

    def _paint_legend(self, p, pr, fg, bg, fm):
        c = self._chart
        entries = []
        for i, it in enumerate(c.lines + c.lines_r2):
            if not it.visible:
                continue
            suffix = " (R)" if it in c.lines_r2 else ""
            entries.append((it.label or tr("chart_widget.legend_label", n=i + 1) + suffix,
                             it.pen.color(), False))
        offset = len(c.lines) + len(c.lines_r2)
        for i, it in enumerate(c.scatters + c.scatters_r2):
            if not it.visible:
                continue
            suffix = " (R)" if it in c.scatters_r2 else ""
            color = it.color if hasattr(it, "color") else QColor(255, 255, 255)
            entries.append((it.label or tr("chart_widget.legend_label",
                             n=offset + i + 1) + suffix, color, True))
        for i, fit in enumerate(c.fits):
            if not fit.visible:
                continue
            entries.append((fit.label or tr("chart_widget.analytics_fit", n=i + 1),
                             fit.pen.color(), False))
        for i, fn_item in enumerate(c.functions):
            if not fn_item.visible:
                continue
            entries.append((fn_item.label or f"f(x) {i + 1}", fn_item.pen.color(), False))
        if not entries:
            return
        pad = _LEGEND_PAD
        sw = _LEGEND_SWATCH
        row_h = max(fm.height(), sw) + 4
        max_w = max(fm.horizontalAdvance(e[0]) for e in entries) + sw + pad * 2 + 6
        total_h = pad * 2 + row_h * len(entries)
        lx = pr.right() - max_w - 4
        ly = pr.bottom() - total_h - 4
        bg_ = QColor(bg)
        bg_.setAlpha(200)
        brd_ = QColor(fg)
        brd_.setAlpha(60)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(brd_, 1))
        p.drawRoundedRect(lx, ly, max_w, total_h, 4, 4)
        p.setFont(self._chart.font)
        for i, (label, color, is_scatter) in enumerate(entries):
            ry = ly + pad + i * row_h + (row_h - sw) // 2
            if is_scatter:
                p.setPen(QPen(color.darker(150), 1))
                p.setBrush(QBrush(color))
                p.drawEllipse(lx + pad, ry, sw, sw)
            else:
                p.setPen(QPen(color, 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                mid_y = ry + sw // 2
                p.drawLine(lx + pad, mid_y, lx + pad + sw, mid_y)
            txt_col = QColor(fg)
            txt_col.setAlpha(210)
            p.setPen(txt_col)
            p.drawText(lx + pad + sw + 6, ry + fm.ascent(), label)

    def _paint_ruler(self, p, ruler, pr, x0, dx, y0, dy, fg, bg, fm):
        c = self._chart
        log_x, log_y = c.log_x, c.log_y
        pt0 = self._to_pt_log(ruler.x0, ruler.y0, x0, dx, y0, dy, pr, log_x, log_y)
        pt1 = self._to_pt_log(ruler.x1, ruler.y1, x0, dx, y0, dy, pr, log_x, log_y)
        p.setPen(ruler.pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(pt0, pt1)
        handle_col = QColor(ruler.pen.color())
        handle_col.setAlpha(200)
        p.setPen(QPen(handle_col, 1))
        p.setBrush(QBrush(handle_col))
        r = float(ruler.handle_radius)
        p.drawEllipse(pt0, r, r)
        p.drawEllipse(pt1, r, r)
        mid = QPointF((pt0.x() + pt1.x()) / 2.0, (pt0.y() + pt1.y()) / 2.0)
        angle_deg = ruler.angle_deg
        dist_lbl = tr(
            "chart_widget.ruler_label",
            d=fmt(ruler.distance),
            dx=fmt(ruler.dx),
            dy=fmt(ruler.dy),
            angle=f"{angle_deg:.1f}",
        )
        tw = fm.horizontalAdvance(dist_lbl)
        th = fm.height()
        pad = 4
        tx = int(mid.x()) - tw // 2
        ty = int(mid.y()) - th - 6
        bg_ = QColor(bg)
        bg_.setAlpha(210)
        br_ = QColor(fg)
        br_.setAlpha(80)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(br_, 1))
        p.drawRoundedRect(tx - pad, ty - pad // 2, tw + pad * 2, th + pad, 3, 3)
        p.setPen(fg)
        p.setFont(c.font)
        p.drawText(tx, ty + fm.ascent(), dist_lbl)

    def _draw_formula_tag(self, p, pt, formula, fg, bg, fm):
        pad = 4
        tw = fm.horizontalAdvance(formula) + pad * 2
        th = fm.height() + pad
        tx = int(pt.x()) + 6
        ty = int(pt.y()) - th - 4
        bg_ = QColor(bg)
        bg_.setAlpha(200)
        br_ = QColor(fg)
        br_.setAlpha(80)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(br_, 1))
        p.drawRoundedRect(tx, ty, tw, th, 3, 3)
        p.setPen(fg)
        p.drawText(tx + pad, ty + fm.ascent() + pad // 2, formula)

    def _paint_latest_points(self, p, pr, x0, dx, y0, dy, fm):
        from .palette import contrast_color
        c = self._chart
        entries = []
        for it in c.lines:
            if it.visible and it.xs:
                entries.append((it.xs[-1], it.ys[-1], it.pen.color()))
        for it in c.scatters:
            if it.visible and it.xs:
                entries.append((it.xs[-1], it.ys[-1], it.color))
        p.setFont(c.font)
        pad = _LATEST_TAG_PAD
        rnd = _LATEST_TAG_ROUND
        th = fm.height()
        p.save()
        p.setClipRect(pr)
        log_x, log_y = c.log_x, c.log_y
        for xi, yi, color in entries:
            lx = to_log(xi) if log_x else xi
            ly = to_log(yi) if log_y else yi
            sx = pr.left() + (lx - x0) / dx * pr.width()
            sy = pr.bottom() - (ly - y0) / dy * pr.height()
            line_col = QColor(color)
            line_col.setAlpha(180)
            p.setPen(QPen(line_col, 1.5, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            if pr.top() <= sy <= pr.bottom():
                p.drawLine(int(sx), int(sy), int(sx), pr.bottom())
            if pr.left() <= sx <= pr.right():
                p.drawLine(pr.left(), int(sy), int(sx), int(sy))
        p.restore()
        for xi, yi, color in entries:
            lx = to_log(xi) if log_x else xi
            ly = to_log(yi) if log_y else yi
            sx = pr.left() + (lx - x0) / dx * pr.width()
            sy = pr.bottom() - (ly - y0) / dy * pr.height()
            txt_col = contrast_color(color)
            x_lbl = _fmt_axis(xi)
            xlw = fm.horizontalAdvance(x_lbl)
            x_tw = xlw + pad * 2
            xtx = int(sx) - x_tw // 2
            xty = pr.bottom() + 2
            if pr.left() - x_tw <= sx <= pr.right() + x_tw:
                p.setBrush(QBrush(color))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(xtx, xty, x_tw, th, rnd, rnd)
                p.setPen(txt_col)
                p.drawText(xtx + pad, xty + fm.ascent(), x_lbl)
            y_lbl = _fmt_axis(yi)
            ylw = fm.horizontalAdvance(y_lbl)
            y_tw = ylw + pad * 2
            ytx = pr.left() - y_tw - 4
            yty = int(sy) - th // 2
            if pr.top() - th <= sy <= pr.bottom() + th:
                p.setBrush(QBrush(color))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(ytx, yty, y_tw, th, rnd, rnd)
                p.setPen(txt_col)
                p.drawText(ytx + pad, yty + fm.ascent(), y_lbl)

    def _copy_coords_to_clipboard(self):
        from PyQt6.QtWidgets import QApplication
        pr = self._plot_rect()
        if self._mouse_pos is None:
            return
        c = self._chart
        x0, x1, y0, y1, dx, dy = self._view_params(pr)
        xv, yv = self._screen_to_data(
            self._mouse_pos.x(), self._mouse_pos.y(),
            x0, dx, y0, dy, pr, c.log_x, c.log_y,
        )
        nearest = self._find_nearest(self._mouse_pos, pr, x0, dx, y0, dy)
        if nearest is not None:
            xv, yv = nearest[0], nearest[1]
        QApplication.clipboard().setText(f"{fmt(xv)}\t{fmt(yv)}")

    def _build_context_menu(self) -> QMenu:
        c = self._chart
        menu = QMenu(self)

        act_autofit = QAction(tr("chart_widget.btn_autofit_toggle"), menu)
        act_autofit.setCheckable(True)
        act_autofit.setChecked(c._autofit_enabled)
        act_autofit.triggered.connect(c.setAutofitEnabled)
        menu.addAction(act_autofit)

        act_latest = QAction(tr("chart_widget.btn_latest"), menu)
        act_latest.setCheckable(True)
        act_latest.setChecked(self._show_latest)
        act_latest.triggered.connect(self._set_latest)
        menu.addAction(act_latest)

        act_analytics = QAction(tr("chart_widget.btn_analytics"), menu)
        act_analytics.setCheckable(True)
        act_analytics.setChecked(self._show_analytics)
        act_analytics.triggered.connect(self._set_analytics)
        menu.addAction(act_analytics)

        act_legend = QAction(tr("chart_widget.ctx_legend"), menu)
        act_legend.setCheckable(True)
        act_legend.setChecked(c.show_legend)
        act_legend.triggered.connect(c.setLegendVisible)
        menu.addAction(act_legend)

        menu.addSeparator()

        display_menu = menu.addMenu(tr("chart_widget.ctx_display"))

        act_origin = QAction(tr("chart_widget.ctx_origin_axes"), display_menu)
        act_origin.setCheckable(True)
        act_origin.setChecked(self._show_origin_axes)
        act_origin.triggered.connect(self._set_origin_axes)
        display_menu.addAction(act_origin)

        act_crosshair = QAction(tr("chart_widget.ctx_crosshair"), display_menu)
        act_crosshair.setCheckable(True)
        act_crosshair.setChecked(self._crosshair_enabled)
        act_crosshair.triggered.connect(self._set_crosshair)
        display_menu.addAction(act_crosshair)

        act_tangent = QAction(tr("chart_widget.ctx_tangent"), display_menu)
        act_tangent.setCheckable(True)
        act_tangent.setChecked(self._show_tangent)
        act_tangent.triggered.connect(self._set_show_tangent)
        display_menu.addAction(act_tangent)

        act_snap = QAction(tr("chart_widget.ctx_snap_to_data"), display_menu)
        act_snap.setCheckable(True)
        act_snap.setChecked(self._snap_to_data)
        act_snap.triggered.connect(self._set_snap_to_data)
        display_menu.addAction(act_snap)

        act_dots = QAction(tr("chart_widget.ctx_show_dots"), display_menu)
        act_dots.setCheckable(True)
        act_dots.setChecked(self._show_dots)
        act_dots.triggered.connect(self._set_show_dots)
        display_menu.addAction(act_dots)

        menu.addSeparator()

        log_menu = menu.addMenu(tr("chart_widget.ctx_log_scale"))
        act_log_x = QAction("Log X", log_menu)
        act_log_x.setCheckable(True)
        act_log_x.setChecked(c.log_x)
        act_log_x.triggered.connect(lambda v: c.setLogScale(x=v))
        log_menu.addAction(act_log_x)
        act_log_y = QAction("Log Y", log_menu)
        act_log_y.setCheckable(True)
        act_log_y.setChecked(c.log_y)
        act_log_y.triggered.connect(lambda v: c.setLogScale(y=v))
        log_menu.addAction(act_log_y)

        lock_menu = menu.addMenu(tr("chart_widget.ctx_zoom_lock"))
        grp_lock = QActionGroup(lock_menu)
        grp_lock.setExclusive(True)
        for key, lbl in (
            ("both", tr("chart_widget.ctx_zoom_both")),
            ("x", tr("chart_widget.ctx_zoom_x")),
            ("y", tr("chart_widget.ctx_zoom_y")),
        ):
            act = QAction(lbl, lock_menu)
            act.setCheckable(True)
            act.setChecked(self._zoom_lock == key)
            act.triggered.connect(lambda checked, k=key: self.setZoomLock(k))
            lock_menu.addAction(act)
            grp_lock.addAction(act)

        menu.addSeparator()

        fit_menu = menu.addMenu(tr("chart_widget.ctx_approximation"))
        grp = QActionGroup(fit_menu)
        grp.setExclusive(True)
        none_act = QAction("None", fit_menu)
        none_act.setCheckable(True)
        none_act.setChecked(c._active_fit_key is None)
        none_act.triggered.connect(lambda: c._on_fit_mode_selected(None))
        fit_menu.addAction(none_act)
        grp.addAction(none_act)
        for mode in get_fit_modes():
            act = QAction(mode.label, fit_menu)
            act.setCheckable(True)
            act.setChecked(c._active_fit_key == mode.key)
            act.triggered.connect(lambda checked, k=mode.key: c._on_fit_mode_selected(k))
            fit_menu.addAction(act)
            grp.addAction(act)

        menu.addSeparator()

        grid_menu = menu.addMenu(tr("chart_widget.ctx_grid"))
        x_menu = grid_menu.addMenu(tr("chart_widget.ctx_grid_x"))
        grp_x = QActionGroup(x_menu)
        grp_x.setExclusive(True)
        for key, px in _GRID_PRESETS_X:
            act = QAction(tr(key), x_menu)
            act.setCheckable(True)
            act.setChecked(c.grid_px_x == px)
            act.triggered.connect(lambda checked, v=px: c.setGridDensity(v, c.grid_px_y))
            x_menu.addAction(act)
            grp_x.addAction(act)
        y_menu = grid_menu.addMenu(tr("chart_widget.ctx_grid_y"))
        grp_y = QActionGroup(y_menu)
        grp_y.setExclusive(True)
        for key, px in _GRID_PRESETS_Y:
            act = QAction(tr(key), y_menu)
            act.setCheckable(True)
            act.setChecked(c.grid_px_y == px)
            act.triggered.connect(lambda checked, v=px: c.setGridDensity(c.grid_px_x, v))
            y_menu.addAction(act)
            grp_y.addAction(act)

        menu.addSeparator()

        rulers_menu = menu.addMenu(tr("chart_widget.ctx_rulers"))
        act_add_ruler = QAction(tr("chart_widget.ctx_add_ruler"), rulers_menu)
        act_add_ruler.triggered.connect(self._add_ruler_at_view_center)
        rulers_menu.addAction(act_add_ruler)
        if c.rulers:
            act_clear_rulers = QAction(tr("chart_widget.ctx_clear_rulers"), rulers_menu)
            act_clear_rulers.triggered.connect(self._clear_all_rulers)
            rulers_menu.addAction(act_clear_rulers)

        menu.addSeparator()

        if self._range_sel_x is not None:
            menu.addAction(tr("chart_widget.ctx_clear_range"), self.clearRangeSelection)

        act_copy = QAction(tr("chart_widget.ctx_copy_coords"), menu)
        act_copy.triggered.connect(self._copy_coords_to_clipboard)
        menu.addAction(act_copy)

        menu.addAction(tr("chart_widget.ctx_export_csv"), c.exportCsv)
        menu.addAction(tr("chart_widget.ctx_export_img"), c.exportImage)
        menu.addSeparator()
        menu.addAction(tr("chart_widget.ctx_reset_view"), c.autofit)
        return menu

    def _add_ruler_at_view_center(self):
        c = self._chart
        cx = (c.vx0 + c.vx1) / 2.0
        cy = (c.vy0 + c.vy1) / 2.0
        qx = (c.vx1 - c.vx0) * 0.15
        qy = (c.vy1 - c.vy0) * 0.15
        ruler = c.addRuler(cx - qx, cy - qy, cx + qx, cy + qy)
        ruler.setVisible(True)

    def _clear_all_rulers(self):
        c = self._chart
        for ruler in list(c.rulers):
            c.removeItem(ruler)

    def wheelEvent(self, ev: QWheelEvent):
        pr = self._plot_rect()
        pos = ev.position().toPoint()
        if not pr.contains(pos):
            return
        c = self._chart
        x0, x1, y0, y1, dx, dy = self._view_params(pr)
        cx_log = x0 + (pos.x() - pr.left()) / pr.width() * dx
        cy_log = y0 + (pr.bottom() - pos.y()) / pr.height() * dy
        factor = 1.0 / _ZOOM_FACTOR if ev.angleDelta().y() > 0 else _ZOOM_FACTOR
        if self._zoom_lock in ("both", "x"):
            nvx0 = cx_log + (x0 - cx_log) * factor
            nvx1 = cx_log + (x1 - cx_log) * factor
            sx = abs(nvx1 - nvx0)
            if math.isfinite(sx) and _ZOOM_MIN_SPAN <= sx <= _ZOOM_MAX_SPAN:
                c._vx0 = 10 ** nvx0 if c.log_x else nvx0
                c._vx1 = 10 ** nvx1 if c.log_x else nvx1
        if self._zoom_lock in ("both", "y"):
            nvy0 = cy_log + (y0 - cy_log) * factor
            nvy1 = cy_log + (y1 - cy_log) * factor
            sy = abs(nvy1 - nvy0)
            if math.isfinite(sy) and _ZOOM_MIN_SPAN <= sy <= _ZOOM_MAX_SPAN:
                c._vy0 = 10 ** nvy0 if c.log_y else nvy0
                c._vy1 = 10 ** nvy1 if c.log_y else nvy1
        self.update()
        ev.accept()

    def mousePressEvent(self, ev: QMouseEvent):
        pr = self._plot_rect()
        c = self._chart
        x0, x1, y0, y1, dx, dy = self._view_params(pr)
        if ev.button() == Qt.MouseButton.LeftButton:
            hit_ruler = self._find_ruler_handle_at(ev.position(), pr, x0, dx, y0, dy)
            if hit_ruler:
                self._dragging_ruler_pt = hit_ruler
                self.setCursor(Qt.CursorShape.CrossCursor)
                ev.accept()
                return
            hit_infline = self._find_infline_at(ev.position(), pr, x0, dx, y0, dy)
            if hit_infline:
                self._dragging_infline = hit_infline
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                ev.accept()
                return
            hit_scatter = self._find_scatter_point_at(ev.position(), pr, x0, dx, y0, dy)
            if hit_scatter:
                item, idx = hit_scatter
                item.selectPoint(idx)
                item.point_clicked.emit(item.xs[idx], item.ys[idx], idx)
                ev.accept()
                return
            if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._rb_start = ev.position()
                self._rb_end = ev.position()
                self._rb_active = True
                ev.accept()
                return
            self._pan_start = ev.position()
            self._pan_vx0 = c.vx0
            self._pan_vy0 = c.vy0
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()
        elif ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
        elif ev.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = ev.position()
            self._pan_vx0 = c.vx0
            self._pan_vy0 = c.vy0
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()

    def mouseMoveEvent(self, ev: QMouseEvent):
        pr = self._plot_rect()
        pos = ev.position()
        self._mouse_pos = pos if pr.contains(pos.toPoint()) else None
        c = self._chart
        if self._dragging_ruler_pt is not None:
            x0, x1, y0, y1, dx, dy = self._view_params(pr)
            ruler, pt_idx = self._dragging_ruler_pt
            xv, yv = self._screen_to_data(pos.x(), pos.y(), x0, dx, y0, dy, pr, c.log_x, c.log_y)
            if pt_idx == 0:
                ruler.x0, ruler.y0 = xv, yv
            else:
                ruler.x1, ruler.y1 = xv, yv
            if ruler.changed is not None:
                ruler.changed()
            self.update()
            return
        if self._dragging_infline is not None:
            x0, x1, y0, y1, dx, dy = self._view_params(pr)
            ln = self._dragging_infline
            if ln.horizontal:
                ly = y0 + (pr.bottom() - pos.y()) / pr.height() * dy
                ln.value = 10 ** ly if c.log_y else ly
            else:
                lx = x0 + (pos.x() - pr.left()) / pr.width() * dx
                ln.value = 10 ** lx if c.log_x else lx
            self._nearest_cache_key = None
            self.update()
            return
        if self._rb_active and self._rb_start is not None:
            self._rb_end = pos
            self.update()
            return
        if self._pan_start is not None:
            x0, x1, y0, y1, dx, dy = self._view_params(pr)
            ddx = (pos.x() - self._pan_start.x()) / pr.width() * dx
            ddy = (pos.y() - self._pan_start.y()) / pr.height() * dy
            new_vx0 = (to_log(self._pan_vx0) if c.log_x else self._pan_vx0) - ddx
            new_vy0 = (to_log(self._pan_vy0) if c.log_y else self._pan_vy0) + ddy
            if c.log_x:
                c._vx0 = 10 ** new_vx0
                c._vx1 = 10 ** (new_vx0 + dx)
            else:
                c._vx0 = new_vx0
                c._vx1 = new_vx0 + dx
            if c.log_y:
                c._vy0 = 10 ** new_vy0
                c._vy1 = 10 ** (new_vy0 + dy)
            else:
                c._vy0 = new_vy0
                c._vy1 = new_vy0 + dy
            self._nearest_cache_key = None
            self.update()
            return
        if pr.contains(pos.toPoint()):
            x0, x1, y0, y1, dx, dy = self._view_params(pr)
            ln = self._find_infline_at(pos, pr, x0, dx, y0, dy)
            if ln:
                self.setCursor(Qt.CursorShape.SizeAllCursor if ln.horizontal else Qt.CursorShape.SizeHorCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if self._dragging_ruler_pt is not None:
            self._dragging_ruler_pt = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            ev.accept()
            return
        if self._dragging_infline is not None:
            self._dragging_infline = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            ev.accept()
            return
        if self._rb_active and self._rb_start and self._rb_end:
            rb = QRectF(self._rb_start, self._rb_end).normalized()
            if rb.width() >= _RUBBERBAND_MIN_PX and rb.height() >= _RUBBERBAND_MIN_PX:
                pr = self._plot_rect()
                x0, x1, y0, y1, dx, dy = self._view_params(pr)
                c = self._chart
                x_lo_v, _ = self._screen_to_data(rb.left(), rb.top(), x0, dx, y0, dy, pr, c.log_x, c.log_y)
                x_hi_v, _ = self._screen_to_data(rb.right(), rb.bottom(), x0, dx, y0, dy, pr, c.log_x, c.log_y)
                _, y_hi_v = self._screen_to_data(rb.left(), rb.top(), x0, dx, y0, dy, pr, c.log_x, c.log_y)
                _, y_lo_v = self._screen_to_data(rb.left(), rb.bottom(), x0, dx, y0, dy, pr, c.log_x, c.log_y)
                c._vx0, c._vx1 = min(x_lo_v, x_hi_v), max(x_lo_v, x_hi_v)
                c._vy0, c._vy1 = min(y_lo_v, y_hi_v), max(y_lo_v, y_hi_v)
            self._rb_active = False
            self._rb_start = None
            self._rb_end = None
            self.update()
            ev.accept()
            return
        self._pan_start = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, ev: QMouseEvent):
        pr = self._plot_rect()
        pos = ev.position().toPoint()
        if ev.button() == Qt.MouseButton.LeftButton and pr.contains(pos):
            c = self._chart
            x0, x1, y0, y1, dx, dy = self._view_params(pr)
            xv, _ = self._screen_to_data(pos.x(), pos.y(), x0, dx, y0, dy, pr, c.log_x, c.log_y)
            if self._range_sel_x is None:
                self._range_sel_x = (xv, xv)
            else:
                lo, hi = self._range_sel_x
                if abs(xv - lo) < abs(xv - hi):
                    self._range_sel_x = (xv, hi)
                else:
                    self._range_sel_x = (lo, xv)
            self.update()
        else:
            self._chart.autofit()

    def leaveEvent(self, ev):
        self._mouse_pos = None
        self.update()

    def grab_image(self) -> QPixmap:
        return self.grab()