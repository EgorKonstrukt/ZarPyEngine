# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import ScrollBar
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class ScrollBarComponent(GuiWidgetComponentBase):
    _widget_class = ScrollBar
    _widget_obj_name = "GuiScrollBar"
    _gizmo_icon_color = (160, 120, 80)
    _gizmo_icon_label = "SB"
    _default_text = "Scroll"
    _default_w = 120.0
    _default_h = 20.0

    _orientation: str = "horizontal"
    _value: int = 0
    _min: int = 0
    _max: int = 100

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_orientation", "Orientation", FieldType.STRING),
            InspectorField("_value", "Value", FieldType.INT, min_val=-100000, max_val=100000),
            InspectorField("_min", "Min", FieldType.INT, min_val=-100000, max_val=100000),
            InspectorField("_max", "Max", FieldType.INT, min_val=-100000, max_val=100000),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        from PyQt6.QtCore import Qt
        orient_map = {"horizontal": Qt.Orientation.Horizontal,
                      "vertical": Qt.Orientation.Vertical}
        new_orient = orient_map.get(self._orientation, Qt.Orientation.Horizontal)
        if w.orientation() != new_orient:
            w.setOrientation(new_orient)
        if w.minimum() != self._min or w.maximum() != self._max:
            w.setRange(self._min, self._max)
        if w.value() != self._value:
            w.setValue(self._value)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            from PyQt6.QtCore import Qt
            orient_rev = {Qt.Orientation.Horizontal: "horizontal",
                          Qt.Orientation.Vertical: "vertical"}
            self._orientation = orient_rev.get(w.orientation(), "horizontal")
            self._value = w.value()
            self._min = w.minimum()
            self._max = w.maximum()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_orientation"] = self._orientation
        d["_value"] = self._value
        d["_min"] = self._min
        d["_max"] = self._max
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ScrollBarComponent:
        inst: ScrollBarComponent = super().deserialize(data)
        inst._orientation = data.get("_orientation", "horizontal")
        inst._value = data.get("_value", 0)
        inst._min = data.get("_min", 0)
        inst._max = data.get("_max", 100)
        return inst
