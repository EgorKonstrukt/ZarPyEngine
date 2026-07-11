# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class GuiWidgetComponent(GuiWidgetComponentBase):
    """Legacy component that adapts to any widget type via widget_type field."""

    widget_type: str = "button"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("widget_type", "Type", FieldType.STRING),
        ] + cls._common_inspector_fields()

    def _create_widget_from_type(self, canvas):
        from core.components.gui import _ensure_component_map
        comp_class = _ensure_component_map().get(self.widget_type)
        if not comp_class:
            return
        t = self.transform
        comp = comp_class(
            widget_text=self.widget_text,
            widget_width=self.widget_width,
            widget_height=self.widget_height,
        )
        comp._anchor = self._anchor
        comp._bg_color = self._bg_color
        comp._text_color = self._text_color
        comp._border_color = self._border_color
        comp._border_width = self._border_width
        comp._border_radius = self._border_radius
        comp._font_size = self._font_size
        comp._font_bold = self._font_bold
        comp._opacity = self._opacity
        comp._visible = self._visible
        if t:
            comp.transform.local_position = (t.local_position.x, t.local_position.y, 0)
            comp.transform.local_scale = (t.local_scale.x, t.local_scale.y, t.local_scale.z)
        comp._create_widget(canvas)
        self._widget_ref = comp._widget_ref
