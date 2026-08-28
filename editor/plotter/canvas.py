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
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import (QPainter, QPen, QBrush, QColor, QFont,
                         QFontMetrics, QPainterPath)
from .canvas_base import (CanvasBase, _fmt_axis,
                           _COORD_CLAMP, _DECIMATE_THRESHOLD, _SCREEN_Y_CLAMP,
                           _RANGE_SEL_ALPHA, _TANGENT_HALF_FRAC)
from .math_utils import nice_ticks, nice_log_ticks, to_log, decimated

if TYPE_CHECKING:
    from .chart_widget import ChartWidget

try:
    import numpy as _np
    _NP_CANVAS = True
except ImportError:
    _NP_CANVAS = False

_SENTINEL = 1e308


def _build_fn_path_fast(fn_xs, fn_ys,
                        x0: float, dx: float, y0: float, dy: float,
                        pr_left: float, pr_bottom: float,
                        pr_width: float, pr_height: float) -> QPainterPath:
    path = QPainterPath()
    n = len(fn_xs)
    if n < 2:
        return path
    if _NP_CANVAS:
        xs_arr = _np.asarray(fn_xs, dtype=_np.float64)
        none_mask = _np.array([v is None for v in fn_ys], dtype=bool)
        ys_raw = _np.array([0.0 if v is None else v for v in fn_ys], dtype=_np.float64)
        px_arr = pr_left + (xs_arr - x0) / dx * pr_width
        py_arr = pr_bottom - (ys_raw - y0) / dy * pr_height
        _np.clip(px_arr, -_COORD_CLAMP, _COORD_CLAMP, out=px_arr)
        _np.clip(py_arr, -_SCREEN_Y_CLAMP, _SCREEN_Y_CLAMP, out=py_arr)
        path.reserve(n + 4)
        started = False
        for i in range(n):
            if none_mask[i]:
                started = False
            else:
                if not started:
                    path.moveTo(float(px_arr[i]), float(py_arr[i]))
                    started = True
                else:
                    path.lineTo(float(px_arr[i]), float(py_arr[i]))
        return path
    started = False
    for xi, yi in zip(fn_xs, fn_ys):
        if yi is None:
            started = False
            continue
        px = pr_left + (xi - x0) / dx * pr_width
        py = pr_bottom - (yi - y0) / dy * pr_height
        if not started:
            path.moveTo(px, py)
            started = True
        else:
            path.lineTo(px, py)
    return path


class _PlotCanvas(CanvasBase, QWidget):
    def __init__(self, chart: "ChartWidget"):
        QWidget.__init__(self, chart)
        self._chart = chart
        self._init_base_state()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)

    def contextMenuEvent(self, ev):
        self._build_context_menu().exec(ev.globalPos())

    def _make_step_pts(self, xs, ys, x0, dx, y0, dy, pr, log_x, log_y):
        pts = []
        for i in range(len(xs)):
            pts.append(self._to_pt_log(xs[i], ys[i], x0, dx, y0, dy, pr, log_x, log_y))
            if i < len(xs) - 1:
                pts.append(self._to_pt_log(xs[i + 1], ys[i], x0, dx, y0, dy, pr, log_x, log_y))
        return pts

    def paintEvent(self, _):
        c = self._chart
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        bg = pal.window().color()
        fg = pal.windowText().color()
        ax_col = QColor(fg); ax_col.setAlpha(80)
        gr_col = QColor(fg); gr_col.setAlpha(30)
        lb_col = QColor(fg); lb_col.setAlpha(255)
        p.fillRect(self.rect(), bg)
        if c.label_title:
            tf = c.title_font or QFont(c.font.family(), c.font.pointSize() + 3, QFont.Weight.Bold)
            p.setFont(tf)
            tfm = QFontMetrics(tf)
            lw = tfm.horizontalAdvance(c.label_title)
            tx = (self.width() - lw) // 2
            ty = tfm.ascent() + 3
            title_col = QColor(fg)
            p.setPen(title_col)
            p.drawText(tx, ty, c.label_title)
            p.setFont(c.font)
        pr = self._plot_rect()
        x0, x1, y0, y1, dx, dy = self._view_params(pr)
        log_x, log_y = c.log_x, c.log_y
        has_r2 = self._has_right_axis()
        if not math.isfinite(dx) or abs(dx) < 1e-300:
            dx = 1.0
        if not math.isfinite(dy) or abs(dy) < 1e-300:
            dy = 1.0
        fm = QFontMetrics(c.font)
        p.setFont(c.font)
        n_x = max(2, pr.width() // max(1, c.grid_px_x))
        n_y = max(2, pr.height() // max(1, c.grid_px_y))
        if log_x:
            xt = nice_log_ticks(10 ** x0, 10 ** x1)
            xt_screen = [to_log(v) for v in xt]
        else:
            xt = nice_ticks(x0, x1, n_x)
            xt_screen = xt
        if log_y:
            yt = nice_log_ticks(10 ** y0, 10 ** y1)
            yt_screen = [to_log(v) for v in yt]
        else:
            yt = nice_ticks(y0, y1, n_y)
            yt_screen = yt
        p.setPen(QPen(gr_col, 1, Qt.PenStyle.DotLine))
        for tv in xt_screen:
            sx = int(pr.left() + (tv - x0) / dx * pr.width())
            p.drawLine(sx, pr.top(), sx, pr.bottom())
        for tv in yt_screen:
            sy = int(pr.bottom() - (tv - y0) / dy * pr.height())
            p.drawLine(pr.left(), sy, pr.right(), sy)
        if self._show_origin_axes:
            self._paint_origin_axes(p, pr, x0, dx, y0, dy, fg)
        p.setPen(QPen(ax_col, 1))
        p.drawRect(pr)
        p.setPen(lb_col)
        for tv, ts in zip(xt, xt_screen):
            sx = int(pr.left() + (ts - x0) / dx * pr.width())
            lbl = _fmt_axis(tv)
            lw = fm.horizontalAdvance(lbl)
            tick_len = 5
            p.setPen(QPen(ax_col, 1))
            p.drawLine(sx, pr.bottom(), sx, pr.bottom() + tick_len)
            p.setPen(lb_col)
            p.drawText(sx - lw // 2, pr.bottom() + fm.height() + 2, lbl)
        for tv, ts in zip(yt, yt_screen):
            sy = int(pr.bottom() - (ts - y0) / dy * pr.height())
            lbl = _fmt_axis(tv)
            lw = fm.horizontalAdvance(lbl)
            tick_len = 5
            p.setPen(QPen(ax_col, 1))
            p.drawLine(pr.left() - tick_len, sy, pr.left(), sy)
            p.setPen(lb_col)
            p.drawText(pr.left() - lw - 6, sy + fm.ascent() // 2, lbl)
        if has_r2:
            y0r = to_log(c.vy0_r) if c.log_y else c.vy0_r
            y1r = to_log(c.vy1_r) if c.log_y else c.vy1_r
            dyr = y1r - y0r if abs(y1r - y0r) > 1e-300 else 1.0
            n_yr = max(2, pr.height() // max(1, c.grid_px_y))
            if log_y:
                ytr = nice_log_ticks(10 ** y0r, 10 ** y1r)
                ytr_screen = [to_log(v) for v in ytr]
            else:
                ytr = nice_ticks(y0r, y1r, n_yr)
                ytr_screen = ytr
            r2_col = QColor(fg); r2_col.setAlpha(160)
            p.setPen(r2_col)
            for tv, ts in zip(ytr, ytr_screen):
                sy = int(pr.bottom() - (ts - y0r) / dyr * pr.height())
                lbl = _fmt_axis(tv)
                p.drawText(pr.right() + 6, sy + fm.ascent() // 2, lbl)
            if c.label_right:
                p.save()
                p.translate(self.width() - 8, pr.top() + pr.height() // 2)
                p.rotate(90)
                lw = fm.horizontalAdvance(c.label_right)
                p.drawText(-lw // 2, fm.ascent() // 2, c.label_right)
                p.restore()
        if c.label_bottom:
            lw = fm.horizontalAdvance(c.label_bottom)
            p.drawText(pr.left() + (pr.width() - lw) // 2, self.height() - 3, c.label_bottom)
        if c.label_left:
            p.save()
            p.translate(11, pr.top() + pr.height() // 2)
            p.rotate(-90)
            lw = fm.horizontalAdvance(c.label_left)
            p.drawText(-lw // 2, fm.ascent() // 2, c.label_left)
            p.restore()
        p.setClipRect(pr)
        if self._range_sel_x is not None:
            rx_lo, rx_hi = sorted(self._range_sel_x)
            lx_lo = to_log(rx_lo) if log_x else rx_lo
            lx_hi = to_log(rx_hi) if log_x else rx_hi
            sx_lo = max(pr.left(), int(pr.left() + (lx_lo - x0) / dx * pr.width()))
            sx_hi = min(pr.right(), int(pr.left() + (lx_hi - x0) / dx * pr.width()))
            if sx_hi > sx_lo:
                rsel_col = QColor(fg); rsel_col.setAlpha(_RANGE_SEL_ALPHA)
                p.fillRect(QRectF(sx_lo, pr.top(), sx_hi - sx_lo, pr.height()), rsel_col)
                sel_border = QColor(fg); sel_border.setAlpha(100)
                p.setPen(QPen(sel_border, 1, Qt.PenStyle.DashLine))
                p.drawLine(sx_lo, pr.top(), sx_lo, pr.bottom())
                p.drawLine(sx_hi, pr.top(), sx_hi, pr.bottom())
        for ln in c.inflines:
            if not ln.visible:
                continue
            p.setPen(ln.pen)
            if ln.horizontal:
                ly = to_log(ln.value) if log_y else ln.value
                sy = int(pr.bottom() - (ly - y0) / dy * pr.height())
                p.drawLine(pr.left(), sy, pr.right(), sy)
            else:
                lx = to_log(ln.value) if log_x else ln.value
                sx = int(pr.left() + (lx - x0) / dx * pr.width())
                p.drawLine(sx, pr.top(), sx, pr.bottom())
        x_lo = min(c.vx0, c.vx1)
        x_hi = max(c.vx0, c.vx1)
        _fn_margin = (x_hi - x_lo) * 0.05
        x_lo_fn = x_lo - _fn_margin
        x_hi_fn = x_hi + _fn_margin
        for fit in c.fits:
            if not fit.visible:
                continue
            fit._recompute(x_lo, x_hi, threaded=c._threaded_fit)
            if len(fit.xs) < 2:
                continue
            pts = [self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, log_x, log_y)
                   for xi, yi in zip(fit.xs, fit.ys)]
            path = QPainterPath()
            path.moveTo(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            mid_pt = pts[len(pts) // 2]
            mid_flat_x, mid_flat_y = mid_pt.x(), mid_pt.y()
            p.setPen(fit.pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            if fit.show_formula:
                formula = fit.getFormula()
                if formula:
                    self._draw_formula_tag(p, QPointF(mid_flat_x, mid_flat_y), formula, fg, bg, fm)
        for fn_item in c.functions:
            if not fn_item.visible:
                continue
            fn_xs, fn_ys = fn_item.evaluate(x_lo_fn, x_hi_fn, max(1, pr.width()), max(1, pr.height()))
            if len(fn_xs) < 2:
                continue
            path = _build_fn_path_fast(
                fn_xs, fn_ys, x0, dx, y0, dy,
                float(pr.left()), float(pr.bottom()),
                float(pr.width()), float(pr.height()),
            )
            if fn_item.fill_under and not path.isEmpty():
                fill_path = QPainterPath(path)
                baseline_y = pr.bottom() - (0.0 - y0) / dy * pr.height()
                baseline_y = max(float(pr.top()), min(float(pr.bottom()), baseline_y))
                last_pt = path.currentPosition()
                first_elem = path.elementAt(0)
                fill_path.lineTo(last_pt.x(), baseline_y)
                fill_path.lineTo(first_elem.x, baseline_y)
                fill_path.closeSubpath()
                fill_col = QColor(fn_item.pen.color())
                fill_col.setAlpha(fn_item.fill_alpha)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(fill_col))
                p.drawPath(fill_path)
            p.setPen(fn_item.pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        for item in c.lines:
            if not item.visible or not item.raw_visible or len(item.xs) < 2:
                continue
            dxs, dys = decimated(item.xs, item.ys, _DECIMATE_THRESHOLD)
            if item.step_mode:
                pts = self._make_step_pts(dxs, dys, x0, dx, y0, dy, pr, log_x, log_y)
            else:
                pts = [self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, log_x, log_y)
                       for xi, yi in zip(dxs, dys)]
            path = QPainterPath()
            path.moveTo(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            if item.fill_under:
                fill_path = QPainterPath(path)
                baseline_y = pr.bottom() - (0 - y0) / dy * pr.height()
                baseline_y = max(pr.top(), min(pr.bottom(), baseline_y))
                fill_path.lineTo(pts[-1].x(), baseline_y)
                fill_path.lineTo(pts[0].x(), baseline_y)
                fill_path.closeSubpath()
                fill_col = QColor(item.pen.color())
                fill_col.setAlpha(item.fill_alpha)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(fill_col))
                p.drawPath(fill_path)
            p.setPen(item.pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        for item in c.lines_r2:
            if not item.visible or not item.raw_visible or len(item.xs) < 2:
                continue
            y0r = to_log(c.vy0_r) if log_y else c.vy0_r
            y1r = to_log(c.vy1_r) if log_y else c.vy1_r
            dyr = y1r - y0r if abs(y1r - y0r) > 1e-300 else 1.0
            dxs, dys = decimated(item.xs, item.ys, _DECIMATE_THRESHOLD)
            pts = [self._to_pt_log(xi, yi, x0, dx, y0r, dyr, pr, log_x, log_y)
                   for xi, yi in zip(dxs, dys)]
            path = QPainterPath()
            path.moveTo(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            p.setPen(item.pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        if self._show_dots:
            self._paint_data_dots(p, pr, x0, dx, y0, dy)
        all_scatters = list(c.scatters) + list(c.scatters_r2)
        for item in all_scatters:
            if not item.visible or not item.raw_visible or not item.xs:
                continue
            is_r2 = item in c.scatters_r2
            y0_eff = (to_log(c.vy0_r) if log_y else c.vy0_r) if is_r2 else y0
            dy_eff = ((to_log(c.vy1_r) - to_log(c.vy0_r)) if log_y else (c.vy1_r - c.vy0_r)) if is_r2 else dy
            if is_r2 and abs(dy_eff) < 1e-300:
                dy_eff = 1.0
            dxs, dys = decimated(item.xs, item.ys, _DECIMATE_THRESHOLD)
            if item.error_ys is not None:
                err_pen = QPen(item.color.darker(150), 1)
                p.setPen(err_pen)
                cap_px = 4
                for i, (xi, yi) in enumerate(zip(dxs, dys)):
                    if i >= len(item.error_ys):
                        break
                    ey = item.error_ys[i]
                    pt_top = self._to_pt_log(xi, yi + ey, x0, dx, y0_eff, dy_eff, pr, log_x, log_y)
                    pt_bot = self._to_pt_log(xi, yi - ey, x0, dx, y0_eff, dy_eff, pr, log_x, log_y)
                    p.drawLine(pt_bot, pt_top)
                    p.drawLine(int(pt_top.x()) - cap_px, int(pt_top.y()),
                               int(pt_top.x()) + cap_px, int(pt_top.y()))
                    p.drawLine(int(pt_bot.x()) - cap_px, int(pt_bot.y()),
                               int(pt_bot.x()) + cap_px, int(pt_bot.y()))
            if item.error_xs is not None:
                err_pen = QPen(item.color.darker(150), 1)
                p.setPen(err_pen)
                cap_px = 4
                for i, (xi, yi) in enumerate(zip(dxs, dys)):
                    if i >= len(item.error_xs):
                        break
                    ex = item.error_xs[i]
                    pt_l = self._to_pt_log(xi - ex, yi, x0, dx, y0_eff, dy_eff, pr, log_x, log_y)
                    pt_r = self._to_pt_log(xi + ex, yi, x0, dx, y0_eff, dy_eff, pr, log_x, log_y)
                    p.drawLine(pt_l, pt_r)
                    p.drawLine(int(pt_l.x()), int(pt_l.y()) - cap_px,
                               int(pt_l.x()), int(pt_l.y()) + cap_px)
                    p.drawLine(int(pt_r.x()), int(pt_r.y()) - cap_px,
                               int(pt_r.x()), int(pt_r.y()) + cap_px)
            r = item.size / 2.0
            for i, (xi, yi) in enumerate(zip(dxs, dys)):
                pt = self._to_pt_log(xi, yi, x0, dx, y0_eff, dy_eff, pr, log_x, log_y)
                is_sel = (item.selected_idx is not None and i == item.selected_idx)
                if is_sel:
                    hi_col = QColor(item.color); hi_col.setAlpha(80)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QBrush(hi_col))
                    p.drawEllipse(pt, r * 2.5, r * 2.5)
                p.setPen(QPen(item.color.darker(150), 2 if is_sel else 1))
                p.setBrush(QBrush(item.color))
                p.drawEllipse(pt, r, r)
            if item.annotations is not None:
                ann_col = QColor(fg); ann_col.setAlpha(200)
                p.setPen(ann_col)
                p.setFont(c.font)
                for i, (xi, yi) in enumerate(zip(item.xs, item.ys)):
                    if i >= len(item.annotations):
                        break
                    ann = item.annotations[i]
                    if not ann:
                        continue
                    pt = self._to_pt_log(xi, yi, x0, dx, y0_eff, dy_eff, pr, log_x, log_y)
                    p.drawText(int(pt.x()) + int(item.size / 2) + 2, int(pt.y()) - 2, ann)
        if self._crosshair_enabled and self._mouse_pos is not None and not self._rb_active:
            self._paint_crosshair(p, pr, x0, dx, y0, dy, fg, bg, fm)
        if self._rb_active and self._rb_start and self._rb_end:
            rb = QRectF(self._rb_start, self._rb_end).normalized()
            rb_col = QColor(fg); rb_col.setAlpha(40)
            p.setBrush(QBrush(rb_col))
            rb_border = QColor(fg); rb_border.setAlpha(160)
            p.setPen(QPen(rb_border, 1, Qt.PenStyle.DashLine))
            p.drawRect(rb)
        p.setClipping(False)
        if self._show_latest:
            self._paint_latest_points(p, pr, x0, dx, y0, dy, fm)
        if self._show_analytics:
            self._paint_analytics(p, pr, fg, bg, fm)
        if c.show_legend:
            self._paint_legend(p, pr, fg, bg, fm)
        for ruler in c.rulers:
            if ruler.visible:
                self._paint_ruler(p, ruler, pr, x0, dx, y0, dy, fg, bg, fm)
        c._notify_viewport_changed()
        p.end()

    def _paint_crosshair(self, p, pr, x0, dx, y0, dy, fg, bg, fm):
        mp = self._mouse_pos
        c = self._chart
        nearest = self._find_nearest(mp, pr, x0, dx, y0, dy)
        if nearest is None:
            ch_col = QColor(fg); ch_col.setAlpha(80)
            p.setPen(QPen(ch_col, 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(int(mp.x()), pr.top(), int(mp.x()), pr.bottom())
            p.drawLine(pr.left(), int(mp.y()), pr.right(), int(mp.y()))
            xv, yv = self._screen_to_data(mp.x(), mp.y(), x0, dx, y0, dy, pr, c.log_x, c.log_y)
            self._paint_crosshair_axis_labels(p, pr, xv, yv, mp, fm, fg, bg)
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
                tg_col = QColor(fg); tg_col.setAlpha(140)
                p.setPen(QPen(tg_col, 1, Qt.PenStyle.DotLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawLine(tp0, tp1)
        self._paint_tooltip(p, pr, xi, yi, snap, fg, bg, fm, item)
        self._paint_crosshair_axis_labels(p, pr, xi, yi, snap, fm, fg, bg)