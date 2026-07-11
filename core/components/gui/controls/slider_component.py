# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import Slider
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class SliderComponent(GuiWidgetComponentBase):
    _widget_class = Slider
    _widget_obj_name = "GuiSlider"
    _default_text = "Slider"
    _default_w = 200.0
    _default_h = 30.0

    _min_val: int = 0
    _max_val: int = 100
    _value: int = 0

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_min_val", "Min", FieldType.INT, min_val=-100000, max_val=100000),
            InspectorField("_max_val", "Max", FieldType.INT, min_val=-100000, max_val=100000),
            InspectorField("_value", "Value", FieldType.INT, min_val=-100000, max_val=100000),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if w:
            if w.minimum() != self._min_val or w.maximum() != self._max_val:
                w.setRange(self._min_val, self._max_val)
            if not w.isSliderDown() and w.value() != self._value:
                w.setValue(self._value)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._min_val = w.minimum()
            self._max_val = w.maximum()
            self._value = w.value()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_min_val"] = self._min_val
        d["_max_val"] = self._max_val
        d["_value"] = self._value
        return d

    @classmethod
    def deserialize(cls, data: dict) -> SliderComponent:
        inst: SliderComponent = super().deserialize(data)
        inst._min_val = data.get("_min_val", 0)
        inst._max_val = data.get("_max_val", 100)
        inst._value = data.get("_value", 0)
        return inst
