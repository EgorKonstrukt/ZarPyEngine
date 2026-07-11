# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional
from PyQt6.QtWidgets import QWidget, QTabWidget
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import TabWidget
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class TabWidgetComponent(GuiWidgetComponentBase):
    _widget_class = TabWidget
    _widget_obj_name = "GuiTabWidget"
    _default_text = "Tab"
    _default_w = 300.0
    _default_h = 200.0

    widget_text: str = "Tab"
    _current_index: int = 0
    _tabs: list[str] = None
    _tab_position: int = 0
    _document_mode: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._tabs is None:
            self._tabs = ["Tab 1", "Tab 2"]

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_current_index", "Current Tab", FieldType.INT, min_val=0, max_val=100),
            InspectorField("_tabs", "Tab Names", FieldType.LIST),
            InspectorField("_tab_position", "Tab Position", FieldType.INT, min_val=0, max_val=3),
            InspectorField("_document_mode", "Document Mode", FieldType.BOOL),
        ] + cls._common_inspector_fields()

    def _create_widget(self, canvas):
        super()._create_widget(canvas)
        self._rebuild_tabs()

    def _rebuild_tabs(self):
        w = self._widget_ref
        if not w:
            return
        while w.count():
            w.removeTab(0)
        if self._tabs:
            for tab_name in self._tabs:
                w.addTab(QWidget(w), tab_name)
        if 0 <= self._current_index < w.count():
            w.setCurrentIndex(self._current_index)

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        pos_map = {0: QTabWidget.TabPosition.North, 1: QTabWidget.TabPosition.South,
                   2: QTabWidget.TabPosition.West, 3: QTabWidget.TabPosition.East}
        w.setTabPosition(pos_map.get(self._tab_position, QTabWidget.TabPosition.North))
        w.setDocumentMode(self._document_mode)
        if 0 <= self._current_index < w.count():
            w.setCurrentIndex(self._current_index)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._current_index = w.currentIndex()
            self._document_mode = w.documentMode()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_current_index"] = self._current_index
        d["_tabs"] = self._tabs
        d["_tab_position"] = self._tab_position
        d["_document_mode"] = self._document_mode
        return d

    @classmethod
    def deserialize(cls, data: dict) -> TabWidgetComponent:
        inst: TabWidgetComponent = super().deserialize(data)
        inst._current_index = data.get("_current_index", 0)
        inst._tabs = data.get("_tabs", ["Tab 1", "Tab 2"])
        inst._tab_position = data.get("_tab_position", 0)
        inst._document_mode = data.get("_document_mode", False)
        return inst
