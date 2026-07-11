# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import Image
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class ImageComponent(GuiWidgetComponentBase):
    _widget_class = Image
    _widget_obj_name = "GuiImage"
    _default_text = "Image"
    _default_w = 100.0
    _default_h = 100.0

    _source: str = ""

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_source", "Source", FieldType.RESOURCE_PATH),
        ] + cls._common_inspector_fields()

    def _create_widget(self, canvas):
        if self._widget_ref:
            return
        super()._create_widget(canvas)
        if self._widget_ref and self._source:
            self._widget_ref.source = self._source

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if w:
            w.source = self._source

    def serialize(self) -> dict:
        d = super().serialize()
        d["_source"] = self._source
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ImageComponent:
        inst: ImageComponent = super().deserialize(data)
        inst._source = data.get("_source", "")
        return inst
