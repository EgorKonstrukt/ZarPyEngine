# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import TextEdit
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class TextEditComponent(GuiWidgetComponentBase):
    _widget_class = TextEdit
    _widget_obj_name = "GuiTextEdit"
    _default_text = ""
    _default_w = 200.0
    _default_h = 150.0

    widget_text: str = ""
    _read_only: bool = False
    _placeholder_text: str = ""
    _wrap_mode: int = 0
    _tab_stop_width: int = 40
    _accept_rich_text: bool = True

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_read_only", "Read Only", FieldType.BOOL),
            InspectorField("_placeholder_text", "Placeholder", FieldType.STRING),
            InspectorField("_wrap_mode", "Wrap Mode", FieldType.INT, min_val=0, max_val=3),
            InspectorField("_tab_stop_width", "Tab Width", FieldType.INT, min_val=8, max_val=200),
            InspectorField("_accept_rich_text", "Rich Text", FieldType.BOOL),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        if w.isReadOnly() != self._read_only:
            w.setReadOnly(self._read_only)
        if w.placeholderText() != self._placeholder_text:
            w.setPlaceholderText(self._placeholder_text)
        if w.tabStopDistance() != self._tab_stop_width:
            w.setTabStopDistance(self._tab_stop_width)
        from PyQt6.QtWidgets import QTextEdit
        wrap_modes = {0: QTextEdit.LineWrapMode.NoWrap, 1: QTextEdit.LineWrapMode.WidgetWidth}
        target_wrap = wrap_modes.get(self._wrap_mode, QTextEdit.LineWrapMode.NoWrap)
        if w.lineWrapMode() != target_wrap:
            w.setLineWrapMode(target_wrap)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._read_only = w.isReadOnly()
            self._placeholder_text = w.placeholderText()
            self._tab_stop_width = w.tabStopDistance()
            self._accept_rich_text = w.acceptRichText()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_read_only"] = self._read_only
        d["_placeholder_text"] = self._placeholder_text
        d["_wrap_mode"] = self._wrap_mode
        d["_tab_stop_width"] = self._tab_stop_width
        d["_accept_rich_text"] = self._accept_rich_text
        return d

    @classmethod
    def deserialize(cls, data: dict) -> TextEditComponent:
        inst: TextEditComponent = super().deserialize(data)
        inst._read_only = data.get("_read_only", False)
        inst._placeholder_text = data.get("_placeholder_text", "")
        inst._wrap_mode = data.get("_wrap_mode", 0)
        inst._tab_stop_width = data.get("_tab_stop_width", 40)
        inst._accept_rich_text = data.get("_accept_rich_text", True)
        return inst
