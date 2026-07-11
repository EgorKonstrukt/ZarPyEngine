# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import ProgressBar
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class ProgressBarComponent(GuiWidgetComponentBase):
    _widget_class = ProgressBar
    _widget_obj_name = "GuiProgressBar"
    _default_text = "Progress"
    _default_w = 200.0
    _default_h = 24.0

    _min_val: int = 0
    _max_val: int = 100
    _value: int = 50

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_min_val", "Min", FieldType.INT, min_val=0, max_val=100000),
            InspectorField("_max_val", "Max", FieldType.INT, min_val=0, max_val=100000),
            InspectorField("_value", "Value", FieldType.INT, min_val=0, max_val=100000),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if w:
            w.setRange(self._min_val, self._max_val)
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
    def deserialize(cls, data: dict) -> ProgressBarComponent:
        inst: ProgressBarComponent = super().deserialize(data)
        inst._min_val = data.get("_min_val", 0)
        inst._max_val = data.get("_max_val", 100)
        inst._value = data.get("_value", 50)
        return inst
