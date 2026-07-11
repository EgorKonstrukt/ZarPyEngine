# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import Dropdown
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class DropdownComponent(GuiWidgetComponentBase):
    _widget_class = Dropdown
    _widget_obj_name = "GuiDropdown"
    _default_text = "Dropdown"
    _default_w = 200.0
    _default_h = 32.0

    _items: list[str] = None
    _current_index: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._items is None:
            self._items = ["Option 1", "Option 2", "Option 3"]

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_items", "Items", FieldType.LIST),
            InspectorField("_current_index", "Current", FieldType.INT, min_val=0, max_val=1000),
        ] + cls._common_inspector_fields()

    def _create_widget(self, canvas):
        if self._widget_ref:
            return
        super()._create_widget(canvas)
        if self._widget_ref:
            for item in (self._items or []):
                self._widget_ref.addItem(item)
            if self._current_index < self._widget_ref.count():
                self._widget_ref.setCurrentIndex(self._current_index)

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if w:
            if w.count() == 0 and self._items:
                for item in self._items:
                    w.addItem(item)
            if self._current_index < w.count():
                w.setCurrentIndex(self._current_index)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._current_index = w.currentIndex()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_items"] = self._items
        d["_current_index"] = self._current_index
        return d

    @classmethod
    def deserialize(cls, data: dict) -> DropdownComponent:
        inst: DropdownComponent = super().deserialize(data)
        inst._items = data.get("_items", ["Option 1", "Option 2", "Option 3"])
        inst._current_index = data.get("_current_index", 0)
        return inst
