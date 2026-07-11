# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import Panel


@ComponentRegistry.register
class PanelComponent(GuiWidgetComponentBase):
    _widget_class = Panel
    _widget_obj_name = "GuiPanel"
    _default_text = "Panel"
    _default_w = 200.0
    _default_h = 200.0

    @classmethod
    def _inspector_fields(cls):
        return cls._common_inspector_fields()
