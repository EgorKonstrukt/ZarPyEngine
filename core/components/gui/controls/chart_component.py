# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import ChartPlotter
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class ChartComponent(GuiWidgetComponentBase):
    _widget_class = ChartPlotter
    _widget_obj_name = "GuiChart"
    _gizmo_icon_color = (74, 122, 181)
    _gizmo_icon_label = "CH"
    _default_text = ""
    _default_w = 320.0
    _default_h = 200.0

    _chart_title: str = ""
    _x_label: str = "X"
    _y_label: str = "Y"
    _log_x: bool = False
    _log_y: bool = False
    _show_legend: bool = True
    _show_toolbar: bool = False
    _show_sidebar: bool = False
    _autofit: bool = True
    _crosshair: bool = True
    _latest_point: bool = False
    _origin_axes: bool = False
    _grid_px_x: int = 80
    _grid_px_y: int = 60

    @property
    def chart(self) -> Optional[object]:
        w = self._widget_ref
        if w is None:
            return None
        try:
            return w.chart()
        except Exception:
            return None

    def plot(self, *args, **kwargs):
        w = self.chart
        return w.plot(*args, **kwargs) if w is not None else None

    def addScatter(self, *args, **kwargs):
        w = self.chart
        return w.addScatter(*args, **kwargs) if w is not None else None

    def addFit(self, *args, **kwargs):
        w = self.chart
        return w.addFit(*args, **kwargs) if w is not None else None

    def addLine(self, *args, **kwargs):
        w = self.chart
        return w.addLine(*args, **kwargs) if w is not None else None

    def addFunction(self, *args, **kwargs):
        w = self.chart
        return w.addFunction(*args, **kwargs) if w is not None else None

    def addRuler(self, *args, **kwargs):
        w = self.chart
        return w.addRuler(*args, **kwargs) if w is not None else None

    def removeItem(self, item):
        w = self.chart
        if w is not None:
            w.removeItem(item)

    def clearAll(self):
        w = self.chart
        if w is not None:
            w.clearAll()

    def autofit(self):
        w = self.chart
        if w is not None:
            w.autofit()

    def setViewport(self, *args, **kwargs):
        w = self.chart
        if w is not None:
            w.setViewport(*args, **kwargs)

    def viewport(self):
        w = self.chart
        return w.viewport() if w is not None else None

    def onViewportChanged(self, callback):
        w = self.chart
        if w is not None:
            w.onViewportChanged(callback)

    def removeViewportChangedCallback(self, callback):
        w = self.chart
        if w is not None:
            w.removeViewportChangedCallback(callback)

    def exportCsv(self):
        w = self.chart
        if w is not None:
            w.exportCsv()

    def exportImage(self):
        w = self.chart
        if w is not None:
            w.exportImage()

    def grabImage(self):
        w = self.chart
        return w.grabImage() if w is not None else None

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_chart_title", "Title", FieldType.STRING),
            InspectorField("_x_label", "X Label", FieldType.STRING),
            InspectorField("_y_label", "Y Label", FieldType.STRING),
            InspectorField("_log_x", "Log X", FieldType.BOOL),
            InspectorField("_log_y", "Log Y", FieldType.BOOL),
            InspectorField("_show_legend", "Legend", FieldType.BOOL),
            InspectorField("_show_toolbar", "Toolbar", FieldType.BOOL),
            InspectorField("_show_sidebar", "Sidebar", FieldType.BOOL),
            InspectorField("_autofit", "Autofit", FieldType.BOOL),
            InspectorField("_crosshair", "Crosshair", FieldType.BOOL),
            InspectorField("_latest_point", "Latest Pt", FieldType.BOOL),
            InspectorField("_origin_axes", "Origin Axes", FieldType.BOOL),
            InspectorField("_grid_px_x", "Grid X", FieldType.INT, min_val=20, max_val=400),
            InspectorField("_grid_px_y", "Grid Y", FieldType.INT, min_val=20, max_val=400),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        try:
            chart = w.chart()
        except Exception:
            return
        if chart is None:
            return
        chart.setTitle(self._chart_title)
        chart.setLabel("bottom", self._x_label)
        chart.setLabel("left", self._y_label)
        chart.setLegendVisible(self._show_legend)
        chart.setLogScale(self._log_x, self._log_y)
        chart.setAutofitEnabled(self._autofit)
        chart.setCrosshairVisible(self._crosshair)
        chart.setLatestPointVisible(self._latest_point)
        chart.setOriginAxesVisible(self._origin_axes)
        chart.setGridDensity(int(self._grid_px_x), int(self._grid_px_y))
        chart.setToolbarVisible(bool(self._show_toolbar))
        chart.setSidebarVisible(bool(self._show_sidebar))

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if not w:
            return
        try:
            chart = w.chart()
        except Exception:
            return
        if chart is None:
            return
        self._chart_title = getattr(chart, "label_title", "") or ""
        self._x_label = getattr(chart, "label_bottom", "") or ""
        self._y_label = getattr(chart, "label_left", "") or ""
        self._log_x = bool(getattr(chart, "log_x", False))
        self._log_y = bool(getattr(chart, "log_y", False))
        self._show_legend = bool(getattr(chart, "show_legend", True))

    def serialize(self) -> dict:
        d = super().serialize()
        d["_chart_title"] = self._chart_title
        d["_x_label"] = self._x_label
        d["_y_label"] = self._y_label
        d["_log_x"] = self._log_x
        d["_log_y"] = self._log_y
        d["_show_legend"] = self._show_legend
        d["_show_toolbar"] = self._show_toolbar
        d["_show_sidebar"] = self._show_sidebar
        d["_autofit"] = self._autofit
        d["_crosshair"] = self._crosshair
        d["_latest_point"] = self._latest_point
        d["_origin_axes"] = self._origin_axes
        d["_grid_px_x"] = self._grid_px_x
        d["_grid_px_y"] = self._grid_px_y
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ChartComponent:
        inst: ChartComponent = super().deserialize(data)
        inst._chart_title = data.get("_chart_title", "")
        inst._x_label = data.get("_x_label", "X")
        inst._y_label = data.get("_y_label", "Y")
        inst._log_x = data.get("_log_x", False)
        inst._log_y = data.get("_log_y", False)
        inst._show_legend = data.get("_show_legend", True)
        inst._show_toolbar = data.get("_show_toolbar", False)
        inst._show_sidebar = data.get("_show_sidebar", False)
        inst._autofit = data.get("_autofit", True)
        inst._crosshair = data.get("_crosshair", True)
        inst._latest_point = data.get("_latest_point", False)
        inst._origin_axes = data.get("_origin_axes", False)
        inst._grid_px_x = data.get("_grid_px_x", 80)
        inst._grid_px_y = data.get("_grid_px_y", 60)
        return inst