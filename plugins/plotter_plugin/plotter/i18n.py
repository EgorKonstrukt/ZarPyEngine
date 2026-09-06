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
from typing import Callable, Dict, Optional

_STRINGS: Dict[str, str] = {
    "chart_widget.btn_fit":                     "Fit",
    "chart_widget.btn_fit_tip":                 "Auto-fit view to data (double-click on chart)",
    "chart_widget.btn_csv":                     "CSV",
    "chart_widget.btn_csv_tip":                 "Export data to CSV",
    "chart_widget.btn_img":                     "Image",
    "chart_widget.btn_img_tip":                 "Export chart as PNG image",
    "chart_widget.btn_analytics":               "Stats",
    "chart_widget.btn_analytics_tip":           "Show / hide analytics panel",
    "chart_widget.btn_fit_mode":                "Fit",
    "chart_widget.btn_fit_mode_tip":            "Select approximation mode",
    "chart_widget.btn_autofit_toggle":          "Auto-fit",
    "chart_widget.btn_autofit_toggle_tip":      "Auto-fit view when data changes",
    "chart_widget.btn_latest":                  "Latest",
    "chart_widget.btn_latest_tip":              "Show latest point values on axis rulers",
    "chart_widget.btn_crosshair":               "Crosshair",
    "chart_widget.btn_crosshair_tip":           "Toggle crosshair cursor mode",
    "chart_widget.btn_origin_axes":             "Origin",
    "chart_widget.btn_origin_axes_tip":         "Show axes through origin (calculator mode)",
    "chart_widget.csv_title":                   "Export to CSV",
    "chart_widget.csv_filter":                  "CSV files (*.csv);;All files (*)",
    "chart_widget.img_title":                   "Export image",
    "chart_widget.img_filter":                  "PNG images (*.png);;All files (*)",
    "chart_widget.tooltip_x":                   "X",
    "chart_widget.tooltip_y":                   "Y",
    "chart_widget.tooltip_slope":               "slope",
    "chart_widget.tooltip_series":              "Series",
    "chart_widget.analytics_line":              "Line {n}",
    "chart_widget.analytics_scatter":           "Scatter {n}",
    "chart_widget.analytics_fit":               "Fit {n}",
    "chart_widget.analytics_n":                 "n",
    "chart_widget.analytics_xmin":              "x min",
    "chart_widget.analytics_xmax":              "x max",
    "chart_widget.analytics_ymin":              "y min",
    "chart_widget.analytics_ymax":              "y max",
    "chart_widget.analytics_mean":              "mean y",
    "chart_widget.analytics_median":            "median y",
    "chart_widget.analytics_std":               "std y",
    "chart_widget.analytics_integral":          "integral",
    "chart_widget.analytics_range":             "range y",
    "chart_widget.analytics_skew":              "skew y",
    "chart_widget.legend_label":                "Series {n}",
    "chart_widget.ctx_legend":                  "Legend",
    "chart_widget.ctx_approximation":           "Approximation",
    "chart_widget.ctx_grid":                    "Grid",
    "chart_widget.ctx_grid_x":                  "X spacing",
    "chart_widget.ctx_grid_y":                  "Y spacing",
    "chart_widget.ctx_sparse":                  "Sparse",
    "chart_widget.ctx_normal":                  "Normal",
    "chart_widget.ctx_dense":                   "Dense",
    "chart_widget.ctx_export_csv":              "Export CSV",
    "chart_widget.ctx_export_img":              "Export Image",
    "chart_widget.ctx_reset_view":              "Reset view",
    "chart_widget.ctx_log_scale":               "Log scale",
    "chart_widget.ctx_zoom_lock":               "Zoom axis lock",
    "chart_widget.ctx_zoom_both":               "Both axes",
    "chart_widget.ctx_zoom_x":                  "X only",
    "chart_widget.ctx_zoom_y":                  "Y only",
    "chart_widget.ctx_clear_range":             "Clear range selection",
    "chart_widget.ctx_display":                 "Display",
    "chart_widget.ctx_origin_axes":             "Origin axes",
    "chart_widget.ctx_crosshair":               "Crosshair",
    "chart_widget.ctx_show_dots":               "Show data points",
    "chart_widget.ctx_fill_under":              "Fill under curves",
    "chart_widget.ctx_antialias":               "Antialiasing",
    "chart_widget.ctx_series":                  "Series",
    "chart_widget.ctx_series_visible":          "Visible",
    "chart_widget.ctx_series_color":            "Color…",
    "chart_widget.ctx_series_fill":             "Fill under",
    "chart_widget.ctx_rulers":                  "Rulers",
    "chart_widget.ctx_add_ruler":               "Add ruler",
    "chart_widget.ctx_clear_rulers":            "Clear rulers",
    "chart_widget.ctx_snap_to_data":            "Snap to data",
    "chart_widget.ctx_tangent":                 "Show tangent",
    "chart_widget.ctx_copy_coords":             "Copy coordinates",
    "chart_widget.ruler_label":                 "d={d}  dx={dx}  dy={dy}  ∠={angle}°",
    "chart_widget.origin_x_label":              "x",
    "chart_widget.origin_y_label":              "y",
}

_custom_tr: Optional[Callable[[str], str]] = None


def set_tr(fn: Callable[[str], str]):
    global _custom_tr
    _custom_tr = fn


def tr(key: str, **kwargs) -> str:
    if _custom_tr is not None:
        try:
            text = _custom_tr(key)
            return text.format(**kwargs) if kwargs else text
        except Exception:
            pass
    text = _STRINGS.get(key, key.split(".")[-1])
    return text.format(**kwargs) if kwargs else text


def update_strings(mapping: Dict[str, str]):
    _STRINGS.update(mapping)