# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import ToolButton
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class ToolButtonComponent(GuiWidgetComponentBase):
    _widget_class = ToolButton
    _widget_obj_name = "GuiToolButton"
    _gizmo_icon_color = (140, 140, 80)
    _gizmo_icon_label = "Tb"
    _default_text = "Tool"
    _default_w = 30.0
    _default_h = 30.0

    _popup_mode: int = 0
    _arrow_type: int = 0
    _tooltip: str = ""

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_popup_mode", "Popup Mode", FieldType.INT, min_val=0, max_val=2),
            InspectorField("_arrow_type", "Arrow Type", FieldType.INT, min_val=0, max_val=4),
            InspectorField("_tooltip", "Tooltip", FieldType.STRING),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        from PyQt6.QtWidgets import QToolButton
        popup_map = {0: QToolButton.ToolButtonPopupMode.DelayedPopup,
                     1: QToolButton.ToolButtonPopupMode.MenuButtonPopup,
                     2: QToolButton.ToolButtonPopupMode.InstantPopup}
        new_mode = popup_map.get(self._popup_mode, QToolButton.ToolButtonPopupMode.DelayedPopup)
        if w.popupMode() != new_mode:
            w.setPopupMode(new_mode)
        from PyQt6.QtCore import Qt
        arrow_map = {0: Qt.ArrowType.NoArrow, 1: Qt.ArrowType.UpArrow,
                     2: Qt.ArrowType.DownArrow, 3: Qt.ArrowType.LeftArrow,
                     4: Qt.ArrowType.RightArrow}
        new_arrow = arrow_map.get(self._arrow_type, Qt.ArrowType.NoArrow)
        if w.arrowType() != new_arrow:
            w.setArrowType(new_arrow)
        if w.toolTip() != self._tooltip:
            w.setToolTip(self._tooltip)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            from PyQt6.QtWidgets import QToolButton
            popup_rev = {QToolButton.ToolButtonPopupMode.DelayedPopup: 0,
                         QToolButton.ToolButtonPopupMode.MenuButtonPopup: 1,
                         QToolButton.ToolButtonPopupMode.InstantPopup: 2}
            self._popup_mode = popup_rev.get(w.popupMode(), 0)
            from PyQt6.QtCore import Qt
            arrow_rev = {Qt.ArrowType.NoArrow: 0, Qt.ArrowType.UpArrow: 1,
                         Qt.ArrowType.DownArrow: 2, Qt.ArrowType.LeftArrow: 3,
                         Qt.ArrowType.RightArrow: 4}
            self._arrow_type = arrow_rev.get(w.arrowType(), 0)
            self._tooltip = w.toolTip()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_popup_mode"] = self._popup_mode
        d["_arrow_type"] = self._arrow_type
        d["_tooltip"] = self._tooltip
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ToolButtonComponent:
        inst: ToolButtonComponent = super().deserialize(data)
        inst._popup_mode = data.get("_popup_mode", 0)
        inst._arrow_type = data.get("_arrow_type", 0)
        inst._tooltip = data.get("_tooltip", "")
        return inst
