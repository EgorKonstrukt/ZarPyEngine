# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import Label
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class LabelComponent(GuiWidgetComponentBase):
    _widget_class = Label
    _widget_obj_name = "GuiLabel"
    _default_text = "Label"
    _default_w = 100.0
    _default_h = 30.0

    widget_text: str = "Label"
    _halign: str = "left"

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_halign", "H Align", FieldType.STRING),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if w:
            from PyQt6.QtCore import Qt
            align_map = {"left": Qt.AlignmentFlag.AlignLeft,
                         "center": Qt.AlignmentFlag.AlignCenter,
                         "right": Qt.AlignmentFlag.AlignRight}
            w.setAlignment(align_map.get(self._halign, Qt.AlignmentFlag.AlignLeft))
