# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import TreeWidget
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class TreeWidgetComponent(GuiWidgetComponentBase):
    _widget_class = TreeWidget
    _widget_obj_name = "GuiTreeWidget"
    _default_text = "Tree"
    _default_w = 200.0
    _default_h = 200.0

    widget_text: str = "Tree"
    _column_count: int = 1
    _header_text: str = "Tree"
    _root_is_decorated: bool = True
    _animated: bool = False

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_column_count", "Columns", FieldType.INT, min_val=1, max_val=10),
            InspectorField("_header_text", "Header", FieldType.STRING),
            InspectorField("_root_is_decorated", "Decorated", FieldType.BOOL),
            InspectorField("_animated", "Animated", FieldType.BOOL),
        ] + cls._common_inspector_fields()

    def _create_widget(self, canvas):
        super()._create_widget(canvas)
        w = self._widget_ref
        if not w:
            return
        w.setColumnCount(self._column_count)
        w.setHeaderLabel(self._header_text)

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        was_animated = w.isAnimated()
        was_decorated = w.rootIsDecorated()
        if was_animated != self._animated:
            w.setAnimated(self._animated)
        if was_decorated != self._root_is_decorated:
            w.setRootIsDecorated(self._root_is_decorated)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._column_count = w.columnCount()
            hdr = w.headerItem()
            if hdr:
                self._header_text = hdr.text(0)
            self._root_is_decorated = w.rootIsDecorated()
            self._animated = w.isAnimated()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_column_count"] = self._column_count
        d["_header_text"] = self._header_text
        d["_root_is_decorated"] = self._root_is_decorated
        d["_animated"] = self._animated
        return d

    @classmethod
    def deserialize(cls, data: dict) -> TreeWidgetComponent:
        inst: TreeWidgetComponent = super().deserialize(data)
        inst._column_count = data.get("_column_count", 1)
        inst._header_text = data.get("_header_text", "Tree")
        inst._root_is_decorated = data.get("_root_is_decorated", True)
        inst._animated = data.get("_animated", False)
        return inst
