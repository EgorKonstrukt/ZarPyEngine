# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import Splitter
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class SplitterComponent(GuiWidgetComponentBase):
    _widget_class = Splitter
    _widget_obj_name = "GuiSplitter"
    _gizmo_icon_color = (140, 100, 60)
    _gizmo_icon_label = "SP"
    _default_text = "Splitter"
    _default_w = 200.0
    _default_h = 24.0

    _orientation: str = "horizontal"
    _children_collapsible: bool = False
    _handle_width: int = 3

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_orientation", "Orientation", FieldType.STRING),
            InspectorField("_children_collapsible", "Collapsible", FieldType.BOOL),
            InspectorField("_handle_width", "Handle Width", FieldType.INT, min_val=1, max_val=50),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w:
            return
        from PyQt6.QtCore import Qt
        orient_map = {"horizontal": Qt.Orientation.Horizontal,
                      "vertical": Qt.Orientation.Vertical}
        new_orient = orient_map.get(self._orientation, Qt.Orientation.Horizontal)
        if w.orientation() != new_orient:
            w.setOrientation(new_orient)
        if w.childrenCollapsible() != self._children_collapsible:
            w.setChildrenCollapsible(self._children_collapsible)
        if w.handleWidth() != self._handle_width:
            w.setHandleWidth(self._handle_width)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            from PyQt6.QtCore import Qt
            orient_rev = {Qt.Orientation.Horizontal: "horizontal",
                          Qt.Orientation.Vertical: "vertical"}
            self._orientation = orient_rev.get(w.orientation(), "horizontal")
            self._children_collapsible = w.childrenCollapsible()
            self._handle_width = w.handleWidth()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_orientation"] = self._orientation
        d["_children_collapsible"] = self._children_collapsible
        d["_handle_width"] = self._handle_width
        return d

    @classmethod
    def deserialize(cls, data: dict) -> SplitterComponent:
        inst: SplitterComponent = super().deserialize(data)
        inst._orientation = data.get("_orientation", "horizontal")
        inst._children_collapsible = data.get("_children_collapsible", False)
        inst._handle_width = data.get("_handle_width", 3)
        return inst
