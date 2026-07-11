# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import SpinBox
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class SpinBoxComponent(GuiWidgetComponentBase):
    _widget_class = SpinBox
    _widget_obj_name = "GuiSpinBox"
    _default_text = ""
    _default_w = 100.0
    _default_h = 32.0

    _min_val: int = 0
    _max_val: int = 100
    _value: int = 0
    _step: int = 1
    _prefix: str = ""
    _suffix: str = ""
    _wrapping: bool = False

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_min_val", "Min", FieldType.INT, min_val=-100000, max_val=100000),
            InspectorField("_max_val", "Max", FieldType.INT, min_val=-100000, max_val=100000),
            InspectorField("_value", "Value", FieldType.INT, min_val=-100000, max_val=100000),
            InspectorField("_step", "Step", FieldType.INT, min_val=1, max_val=100000),
            InspectorField("_prefix", "Prefix", FieldType.STRING),
            InspectorField("_suffix", "Suffix", FieldType.STRING),
            InspectorField("_wrapping", "Wrapping", FieldType.BOOL),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        w.setRange(self._min_val, self._max_val)
        if w.value() != self._value:
            w.setValue(self._value)
        w.setSingleStep(self._step)
        w.setPrefix(self._prefix)
        w.setSuffix(self._suffix)
        w.setWrapping(self._wrapping)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._min_val = w.minimum()
            self._max_val = w.maximum()
            self._value = w.value()
            self._step = w.singleStep()
            self._prefix = w.prefix()
            self._suffix = w.suffix()
            self._wrapping = w.wrapping()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_min_val"] = self._min_val
        d["_max_val"] = self._max_val
        d["_value"] = self._value
        d["_step"] = self._step
        d["_prefix"] = self._prefix
        d["_suffix"] = self._suffix
        d["_wrapping"] = self._wrapping
        return d

    @classmethod
    def deserialize(cls, data: dict) -> SpinBoxComponent:
        inst: SpinBoxComponent = super().deserialize(data)
        inst._min_val = data.get("_min_val", 0)
        inst._max_val = data.get("_max_val", 100)
        inst._value = data.get("_value", 0)
        inst._step = data.get("_step", 1)
        inst._prefix = data.get("_prefix", "")
        inst._suffix = data.get("_suffix", "")
        inst._wrapping = data.get("_wrapping", False)
        return inst
