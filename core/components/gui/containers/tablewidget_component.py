# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import TableWidget
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class TableWidgetComponent(GuiWidgetComponentBase):
    _widget_class = TableWidget
    _widget_obj_name = "GuiTableWidget"
    _default_text = "Table"
    _default_w = 300.0
    _default_h = 200.0

    widget_text: str = "Table"
    _rows: int = 3
    _cols: int = 3
    _current_row: int = -1
    _current_col: int = -1
    _show_grid: bool = True
    _headers_visible: bool = True

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_rows", "Rows", FieldType.INT, min_val=1, max_val=100),
            InspectorField("_cols", "Cols", FieldType.INT, min_val=1, max_val=100),
            InspectorField("_current_row", "Current Row", FieldType.INT, min_val=-1, max_val=100000),
            InspectorField("_current_col", "Current Col", FieldType.INT, min_val=-1, max_val=100000),
            InspectorField("_show_grid", "Show Grid", FieldType.BOOL),
            InspectorField("_headers_visible", "Headers Visible", FieldType.BOOL),
        ] + cls._common_inspector_fields()

    def _create_widget(self, canvas):
        super()._create_widget(canvas)
        w = self._widget_ref
        if not w:
            return
        w.setRowCount(self._rows)
        w.setColumnCount(self._cols)
        w.setShowGrid(self._show_grid)

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        if w.rowCount() != self._rows:
            w.setRowCount(self._rows)
        if w.columnCount() != self._cols:
            w.setColumnCount(self._cols)
        w.setShowGrid(self._show_grid)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._rows = w.rowCount()
            self._cols = w.columnCount()
            self._current_row = w.currentRow()
            self._current_col = w.currentColumn()
            self._show_grid = w.showGrid()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_rows"] = self._rows
        d["_cols"] = self._cols
        d["_current_row"] = self._current_row
        d["_current_col"] = self._current_col
        d["_show_grid"] = self._show_grid
        d["_headers_visible"] = self._headers_visible
        return d

    @classmethod
    def deserialize(cls, data: dict) -> TableWidgetComponent:
        inst: TableWidgetComponent = super().deserialize(data)
        inst._rows = data.get("_rows", 3)
        inst._cols = data.get("_cols", 3)
        inst._current_row = data.get("_current_row", -1)
        inst._current_col = data.get("_current_col", -1)
        inst._show_grid = data.get("_show_grid", True)
        inst._headers_visible = data.get("_headers_visible", True)
        return inst
