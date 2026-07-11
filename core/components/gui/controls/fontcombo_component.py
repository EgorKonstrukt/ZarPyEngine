# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import FontCombo
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class FontComboComponent(GuiWidgetComponentBase):
    _widget_class = FontCombo
    _widget_obj_name = "GuiFontCombo"
    _gizmo_icon_color = (80, 120, 180)
    _gizmo_icon_label = "FC"
    _default_text = "Font"
    _default_w = 160.0
    _default_h = 28.0

    _font_filters: int = 0
    _sample_text: str = ""

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_font_filters", "Font Filters", FieldType.INT, min_val=0, max_val=256),
            InspectorField("_sample_text", "Sample Text", FieldType.STRING),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        from PyQt6.QtWidgets import QFontComboBox
        ff = QFontComboBox.FontFilter(self._font_filters)
        if w.fontFilters() != ff:
            w.setFontFilters(ff)
        if self._sample_text and w.currentText() != self._sample_text:
            w.setCurrentText(self._sample_text)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._font_filters = w.fontFilters().value
            self._sample_text = w.currentText()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_font_filters"] = self._font_filters
        d["_sample_text"] = self._sample_text
        return d

    @classmethod
    def deserialize(cls, data: dict) -> FontComboComponent:
        inst: FontComboComponent = super().deserialize(data)
        inst._font_filters = data.get("_font_filters", 0)
        inst._sample_text = data.get("_sample_text", "")
        return inst
