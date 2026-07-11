# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import ListWidget
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class ListWidgetComponent(GuiWidgetComponentBase):
    _widget_class = ListWidget
    _widget_obj_name = "GuiListWidget"
    _default_text = "List"
    _default_w = 200.0
    _default_h = 200.0

    widget_text: str = "List"
    _items: list[str] = None
    _current_index: int = -1
    _selection_mode: int = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._items is None:
            self._items = ["Item 1", "Item 2", "Item 3"]

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_items", "Items", FieldType.LIST),
            InspectorField("_current_index", "Current Index", FieldType.INT, min_val=-1, max_val=100000),
            InspectorField("_selection_mode", "Selection Mode", FieldType.INT, min_val=0, max_val=3),
        ] + cls._common_inspector_fields()

    def _create_widget(self, canvas):
        super()._create_widget(canvas)
        w = self._widget_ref
        if w and self._items:
            w.addItems(self._items)
            if 0 <= self._current_index < len(self._items):
                w.setCurrentRow(self._current_index)

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        if 0 <= self._current_index < w.count():
            w.setCurrentRow(self._current_index)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._items = [w.item(i).text() for i in range(w.count()) if w.item(i)]
            self._current_index = w.currentRow()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_items"] = self._items
        d["_current_index"] = self._current_index
        d["_selection_mode"] = self._selection_mode
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ListWidgetComponent:
        inst: ListWidgetComponent = super().deserialize(data)
        inst._items = data.get("_items", ["Item 1", "Item 2", "Item 3"])
        inst._current_index = data.get("_current_index", -1)
        inst._selection_mode = data.get("_selection_mode", 1)
        return inst
