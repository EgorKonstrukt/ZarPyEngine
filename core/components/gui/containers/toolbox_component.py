# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import ToolBox
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class ToolBoxComponent(GuiWidgetComponentBase):
    _widget_class = ToolBox
    _widget_obj_name = "GuiToolBox"
    _gizmo_icon_color = (140, 120, 80)
    _gizmo_icon_label = "TB"
    _default_text = "ToolBox"
    _default_w = 200.0
    _default_h = 200.0

    _current_index: int = 0

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_current_index", "Current Index", FieldType.INT, min_val=0, max_val=1000),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        if 0 <= self._current_index < w.count() and w.currentIndex() != self._current_index:
            w.setCurrentIndex(self._current_index)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._current_index = w.currentIndex()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_current_index"] = self._current_index
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ToolBoxComponent:
        inst: ToolBoxComponent = super().deserialize(data)
        inst._current_index = data.get("_current_index", 0)
        return inst
