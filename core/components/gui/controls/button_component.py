# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import Button


@ComponentRegistry.register
class ButtonComponent(GuiWidgetComponentBase):
    _widget_class = Button
    _widget_obj_name = "GuiButton"
    _default_text = "Button"
    _default_w = 120.0
    _default_h = 36.0

    widget_text: str = "Button"

    @classmethod
    def _inspector_fields(cls):
        return cls._common_inspector_fields()
