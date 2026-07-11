# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import GroupBox
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class GroupBoxComponent(GuiWidgetComponentBase):
    _widget_class = GroupBox
    _widget_obj_name = "GuiGroupBox"
    _default_text = "Group"
    _default_w = 200.0
    _default_h = 200.0

    widget_text: str = "Group"
    _flat: bool = False
    _checkable: bool = False
    _checked: bool = True
    _alignment: int = 0

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_flat", "Flat", FieldType.BOOL),
            InspectorField("_checkable", "Checkable", FieldType.BOOL),
            InspectorField("_checked", "Checked", FieldType.BOOL),
            InspectorField("_alignment", "Title Align", FieldType.INT, min_val=0, max_val=2),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        w.setFlat(self._flat)
        w.setCheckable(self._checkable)
        w.setChecked(self._checked)
        align_map = {0: 0, 1: 1, 2: 2}
        from PyQt6.QtCore import Qt
        w.setAlignment(align_map.get(self._alignment, Qt.AlignmentFlag.AlignLeft))

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._flat = w.isFlat()
            self._checkable = w.isCheckable()
            self._checked = w.isChecked()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_flat"] = self._flat
        d["_checkable"] = self._checkable
        d["_checked"] = self._checked
        d["_alignment"] = self._alignment
        return d

    @classmethod
    def deserialize(cls, data: dict) -> GroupBoxComponent:
        inst: GroupBoxComponent = super().deserialize(data)
        inst._flat = data.get("_flat", False)
        inst._checkable = data.get("_checkable", False)
        inst._checked = data.get("_checked", True)
        inst._alignment = data.get("_alignment", 0)
        return inst
