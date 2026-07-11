# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import Calendar
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class CalendarComponent(GuiWidgetComponentBase):
    _widget_class = Calendar
    _widget_obj_name = "GuiCalendar"
    _gizmo_icon_color = (80, 140, 120)
    _gizmo_icon_label = "CA"
    _default_text = "Calendar"
    _default_w = 250.0
    _default_h = 200.0

    _grid_visible: bool = True
    _navigation_bar_visible: bool = True

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_grid_visible", "Grid Visible", FieldType.BOOL),
            InspectorField("_navigation_bar_visible", "Nav Bar Visible", FieldType.BOOL),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        if w.isGridVisible() != self._grid_visible:
            w.setGridVisible(self._grid_visible)
        if w.isNavigationBarVisible() != self._navigation_bar_visible:
            w.setNavigationBarVisible(self._navigation_bar_visible)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self._grid_visible = w.isGridVisible()
            self._navigation_bar_visible = w.isNavigationBarVisible()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_grid_visible"] = self._grid_visible
        d["_navigation_bar_visible"] = self._navigation_bar_visible
        return d

    @classmethod
    def deserialize(cls, data: dict) -> CalendarComponent:
        inst: CalendarComponent = super().deserialize(data)
        inst._grid_visible = data.get("_grid_visible", True)
        inst._navigation_bar_visible = data.get("_navigation_bar_visible", True)
        return inst
