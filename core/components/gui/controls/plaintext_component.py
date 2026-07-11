# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import PlainText
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class PlainTextComponent(GuiWidgetComponentBase):
    _widget_class = PlainText
    _widget_obj_name = "GuiPlainText"
    _gizmo_icon_color = (100, 100, 160)
    _gizmo_icon_label = "PT"
    _default_text = ""
    _default_w = 200.0
    _default_h = 150.0

    widget_text: str = ""
    _read_only: bool = False
    _placeholder_text: str = ""
    _tab_stop_width: int = 40
    _line_wrap: bool = True

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_read_only", "Read Only", FieldType.BOOL),
            InspectorField("_placeholder_text", "Placeholder", FieldType.STRING),
            InspectorField("_tab_stop_width", "Tab Width", FieldType.INT, min_val=8, max_val=200),
            InspectorField("_line_wrap", "Line Wrap", FieldType.BOOL),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        txt = self.widget_text
        self.widget_text = ""
        super().sync_to_widget()
        self.widget_text = txt
        w = self._widget_ref
        if not w:
            return
        if w.isReadOnly() != self._read_only:
            w.setReadOnly(self._read_only)
        if w.placeholderText() != self._placeholder_text:
            w.setPlaceholderText(self._placeholder_text)
        if w.tabStopDistance() != self._tab_stop_width:
            w.setTabStopDistance(self._tab_stop_width)
        from PyQt6.QtWidgets import QPlainTextEdit
        target_mode = (QPlainTextEdit.LineWrapMode.WidgetWidth if self._line_wrap
                       else QPlainTextEdit.LineWrapMode.NoWrap)
        if w.lineWrapMode() != target_mode:
            w.setLineWrapMode(target_mode)
        if w.toPlainText() != txt:
            w.setPlainText(txt)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self.widget_text = w.toPlainText()
            self._read_only = w.isReadOnly()
            self._placeholder_text = w.placeholderText()
            self._tab_stop_width = w.tabStopDistance()
            from PyQt6.QtWidgets import QPlainTextEdit
            self._line_wrap = w.lineWrapMode() == QPlainTextEdit.LineWrapMode.WidgetWidth

    def serialize(self) -> dict:
        d = super().serialize()
        d["_read_only"] = self._read_only
        d["_placeholder_text"] = self._placeholder_text
        d["_tab_stop_width"] = self._tab_stop_width
        d["_line_wrap"] = self._line_wrap
        return d

    @classmethod
    def deserialize(cls, data: dict) -> PlainTextComponent:
        inst: PlainTextComponent = super().deserialize(data)
        inst._read_only = data.get("_read_only", False)
        inst._placeholder_text = data.get("_placeholder_text", "")
        inst._tab_stop_width = data.get("_tab_stop_width", 40)
        inst._line_wrap = data.get("_line_wrap", True)
        return inst
