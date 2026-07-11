# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import HtmlView
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class HtmlComponent(GuiWidgetComponentBase):
    _widget_class = HtmlView
    _widget_obj_name = "GuiHtmlView"
    _gizmo_icon_color = (100, 160, 120)
    _gizmo_icon_label = "HT"
    _default_text = ""
    _default_w = 300.0
    _default_h = 200.0

    widget_text: str = ""
    _open_external_links: bool = True

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("widget_text", "HTML", FieldType.STRING),
            InspectorField("_open_external_links", "Open Ext Links", FieldType.BOOL),
        ] + cls._common_inspector_fields()

    def sync_to_widget(self):
        html = self.widget_text
        self.widget_text = ""
        super().sync_to_widget()
        self.widget_text = html
        w = self._widget_ref
        if not w:
            return
        w.setOpenExternalLinks(self._open_external_links)
        if w.toHtml() != html:
            w.setHtml(html)

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if w:
            self.widget_text = w.toHtml()
            self._open_external_links = w.openExternalLinks()

    def serialize(self) -> dict:
        d = super().serialize()
        d["_open_external_links"] = self._open_external_links
        return d

    @classmethod
    def deserialize(cls, data: dict) -> HtmlComponent:
        inst: HtmlComponent = super().deserialize(data)
        inst._open_external_links = data.get("_open_external_links", True)
        return inst
