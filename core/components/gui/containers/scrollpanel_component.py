# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import ScrollPanel
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class ScrollPanelComponent(GuiWidgetComponentBase):
    _widget_class = ScrollPanel
    _widget_obj_name = "GuiScrollPanel"
    _default_text = "ScrollPanel"
    _default_w = 200.0
    _default_h = 200.0

    @classmethod
    def _inspector_fields(cls):
        return cls._common_inspector_fields()
