# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import LCDNumber
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class LCDComponent(GuiWidgetComponentBase):
    _widget_class = LCDNumber
    _widget_obj_name = "GuiLCDNumber"
    _gizmo_icon_color = (60, 180, 60)
    _gizmo_icon_label = "LC"
    _default_text = ""
    _default_w = 120.0
    _default_h = 40.0

    _digit_count: int = 5
    _value: float = 0.0
    _mode: str = "dec"

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_digit_count", "Digit Count", FieldType.INT, min_val=1, max_val=20),
            InspectorField("_value", "Value", FieldType.FLOAT, min_val=-999999, max_val=999999),
            InspectorField("_mode", "Mode", FieldType.STRING),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        if w.digitCount() != self._digit_count:
            w.setDigitCount(self._digit_count)
        if w.value() != self._value:
            w.display(self._value)
        mode_map = {"hex": w.setHexMode, "dec": w.setDecMode,
                    "oct": w.setOctMode, "bin": w.setBinMode}
        mode_map.get(self._mode, w.setDecMode)()

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._digit_count = w.digitCount()
            self._value = w.value()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_digit_count"] = self._digit_count
        d["_value"] = self._value
        d["_mode"] = self._mode
        return d

    @classmethod
    def deserialize(cls, data: dict) -> LCDComponent:
        inst: LCDComponent = super().deserialize(data)
        inst._digit_count = data.get("_digit_count", 5)
        inst._value = data.get("_value", 0.0)
        inst._mode = data.get("_mode", "dec")
        return inst
