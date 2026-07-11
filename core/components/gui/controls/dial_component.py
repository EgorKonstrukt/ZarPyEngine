# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import Dial
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class DialComponent(GuiWidgetComponentBase):
    _widget_class = Dial
    _widget_obj_name = "GuiDial"
    _default_text = ""
    _default_w = 80.0
    _default_h = 80.0

    _min_val: int = 0
    _max_val: int = 100
    _value: int = 0
    _step: int = 1
    _page_step: int = 10
    _wrapping: bool = False
    _notches_visible: bool = True
    _notch_target: float = 10.0

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_min_val", "Min", FieldType.INT, min_val=-100000, max_val=100000),
            InspectorField("_max_val", "Max", FieldType.INT, min_val=-100000, max_val=100000),
            InspectorField("_value", "Value", FieldType.INT, min_val=-100000, max_val=100000),
            InspectorField("_step", "Step", FieldType.INT, min_val=1, max_val=100000),
            InspectorField("_page_step", "Page Step", FieldType.INT, min_val=1, max_val=100000),
            InspectorField("_wrapping", "Wrapping", FieldType.BOOL),
            InspectorField("_notches_visible", "Notches", FieldType.BOOL),
            InspectorField("_notch_target", "Notch Target", FieldType.FLOAT, min_val=0.1, max_val=1000),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        if w.minimum() != self._min_val or w.maximum() != self._max_val:
            w.setRange(self._min_val, self._max_val)
        if not w.isSliderDown() and w.value() != self._value:
            w.setValue(self._value)
        if w.singleStep() != self._step:
            w.setSingleStep(self._step)
        if w.pageStep() != self._page_step:
            w.setPageStep(self._page_step)
        if w.wrapping() != self._wrapping:
            w.setWrapping(self._wrapping)
        if w.notchesVisible() != self._notches_visible:
            w.setNotchesVisible(self._notches_visible)
        if w.notchTarget() != self._notch_target:
            w.setNotchTarget(self._notch_target)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._min_val = w.minimum()
            self._max_val = w.maximum()
            self._value = w.value()
            self._step = w.singleStep()
            self._page_step = w.pageStep()
            self._wrapping = w.wrapping()
            self._notches_visible = w.notchesVisible()
            self._notch_target = w.notchTarget()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_min_val"] = self._min_val
        d["_max_val"] = self._max_val
        d["_value"] = self._value
        d["_step"] = self._step
        d["_page_step"] = self._page_step
        d["_wrapping"] = self._wrapping
        d["_notches_visible"] = self._notches_visible
        d["_notch_target"] = self._notch_target
        return d

    @classmethod
    def deserialize(cls, data: dict) -> DialComponent:
        inst: DialComponent = super().deserialize(data)
        inst._min_val = data.get("_min_val", 0)
        inst._max_val = data.get("_max_val", 100)
        inst._value = data.get("_value", 0)
        inst._step = data.get("_step", 1)
        inst._page_step = data.get("_page_step", 10)
        inst._wrapping = data.get("_wrapping", False)
        inst._notches_visible = data.get("_notches_visible", True)
        inst._notch_target = data.get("_notch_target", 10.0)
        return inst
