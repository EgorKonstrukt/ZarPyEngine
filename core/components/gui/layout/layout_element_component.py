# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from PyQt6.QtWidgets import QWidget, QSizePolicy
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import InspectorField, FieldType
if TYPE_CHECKING:
    from core.gui.canvas import GuiCanvas


@ComponentRegistry.register
class LayoutElementComponent(Component):
    _allow_multiple = False
    _show_gizmo_icon = True
    _gizmo_icon_color = (100, 180, 100)
    _gizmo_icon_label = "LE"

    _min_width: int = -1
    _min_height: int = -1
    _preferred_width: int = -1
    _preferred_height: int = -1
    _flexible_width: float = 0.0
    _flexible_height: float = 0.0
    _ignore_layout: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            if hasattr(self.__class__, k):
                setattr(self, k, v)

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_min_width", "Min Width", FieldType.INT, min_val=-1, max_val=10000),
            InspectorField("_min_height", "Min Height", FieldType.INT, min_val=-1, max_val=10000),
            InspectorField("_preferred_width", "Pref Width", FieldType.INT, min_val=-1, max_val=10000),
            InspectorField("_preferred_height", "Pref Height", FieldType.INT, min_val=-1, max_val=10000),
            InspectorField("_flexible_width", "Flex Width", FieldType.FLOAT, min_val=0, max_val=100),
            InspectorField("_flexible_height", "Flex Height", FieldType.FLOAT, min_val=0, max_val=100),
            InspectorField("_ignore_layout", "Ignore Layout", FieldType.BOOL),
        ]

    def _sync_layout_element(self, widget: Optional[QWidget]):
        if not widget:
            return
        sp = widget.sizePolicy()
        if self._min_width >= 0:
            widget.setMinimumWidth(self._min_width)
        if self._min_height >= 0:
            widget.setMinimumHeight(self._min_height)
        if self._preferred_width >= 0:
            sp.setHorizontalStretch(0)
            widget.setFixedWidth(self._preferred_width)
        if self._preferred_height >= 0:
            sp.setVerticalStretch(0)
            widget.setFixedHeight(self._preferred_height)
        h_policy = QSizePolicy.Policy.Ignored if self._ignore_layout else QSizePolicy.Policy.Preferred
        v_policy = QSizePolicy.Policy.Ignored if self._ignore_layout else QSizePolicy.Policy.Preferred
        if self._flexible_width > 0:
            h_policy = QSizePolicy.Policy.Expanding
            sp.setHorizontalStretch(int(self._flexible_width))
        if self._flexible_height > 0:
            v_policy = QSizePolicy.Policy.Expanding
            sp.setVerticalStretch(int(self._flexible_height))
        sp.setHorizontalPolicy(h_policy)
        sp.setVerticalPolicy(v_policy)
        widget.setSizePolicy(sp)
