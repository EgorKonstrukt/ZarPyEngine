# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import Toggle
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class ToggleComponent(GuiWidgetComponentBase):
    _widget_class = Toggle
    _widget_obj_name = "GuiToggle"
    _default_text = "Toggle"
    _default_w = 100.0
    _default_h = 30.0

    _checked: bool = False

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_checked", "Checked", FieldType.BOOL),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if w:
            w.setChecked(self._checked)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._checked = w.isChecked()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_checked"] = self._checked
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ToggleComponent:
        inst: ToggleComponent = super().deserialize(data)
        inst._checked = data.get("_checked", False)
        return inst
