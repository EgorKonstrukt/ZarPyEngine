# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout, QWidget, QSizePolicy
from PyQt6.QtCore import Qt
from core.ecs.ecs import Component, ComponentRegistry
if TYPE_CHECKING:
    from core.gui.canvas import GuiCanvas


def _find_entity_widget(entity):
    from core.components.gui import _ensure_component_map
    for c in _ensure_component_map().values():
        comp = entity.get_component_by_name(c.__name__)
        if comp and hasattr(comp, '_widget_ref') and comp._widget_ref:
            return comp._widget_ref
    return None


def _ours(child):
    return isinstance(child, QWidget) and hasattr(child, '_widget_id')


def _apply_layout_children(parent: QWidget, layout, ctrl_w: bool, ctrl_h: bool):
    for child in parent.children():
        if _ours(child) and layout.indexOf(child) == -1:
            layout.addWidget(child)
            child.setSizePolicy(
                QSizePolicy.Policy.Expanding if ctrl_w else QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Expanding if ctrl_h else QSizePolicy.Policy.Preferred,
            )


def _apply_grid_children(parent: QWidget, layout, cols: int):
    row, col = 0, 0
    for child in parent.children():
        if _ours(child):
            layout.addWidget(child, row, col)
            col += 1
            if col >= cols:
                col = 0
                row += 1


@ComponentRegistry.register
class HorizontalLayoutComponent(Component):
    _allow_multiple = False
    _show_gizmo_icon = True
    _gizmo_icon_color = (60, 140, 200)
    _gizmo_icon_label = "H"
    _spacing: int = 4
    _padding_left: int = 0
    _padding_top: int = 0
    _padding_right: int = 0
    _padding_bottom: int = 0
    _alignment: int = 0
    _child_control_width: bool = False
    _child_control_height: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            if hasattr(self.__class__, k):
                setattr(self, k, v)

    def _sync_layout(self, canvas: GuiCanvas):
        if not self.entity:
            return
        parent_w = _find_entity_widget(self.entity)
        if not parent_w:
            return
        existing = parent_w.layout()
        if isinstance(existing, QHBoxLayout):
            layout = existing
        else:
            if existing:
                _clear_layout(existing)
            layout = QHBoxLayout(parent_w)
            layout.setContentsMargins(int(self._padding_left), int(self._padding_top),
                                      int(self._padding_right), int(self._padding_bottom))
            layout.setSpacing(int(self._spacing))
            parent_w.setLayout(layout)
        align_map = {0: Qt.AlignmentFlag.AlignLeft, 1: Qt.AlignmentFlag.AlignHCenter,
                     2: Qt.AlignmentFlag.AlignRight}
        layout.setAlignment(align_map.get(self._alignment, Qt.AlignmentFlag.AlignLeft))
        _apply_layout_children(parent_w, layout, self._child_control_width, self._child_control_height)


@ComponentRegistry.register
class VerticalLayoutComponent(Component):
    _allow_multiple = False
    _show_gizmo_icon = True
    _gizmo_icon_color = (60, 140, 200)
    _gizmo_icon_label = "V"
    _spacing: int = 4
    _padding_left: int = 0
    _padding_top: int = 0
    _padding_right: int = 0
    _padding_bottom: int = 0
    _alignment: int = 0
    _child_control_width: bool = True
    _child_control_height: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            if hasattr(self.__class__, k):
                setattr(self, k, v)

    def _sync_layout(self, canvas: GuiCanvas):
        if not self.entity:
            return
        parent_w = _find_entity_widget(self.entity)
        if not parent_w:
            return
        existing = parent_w.layout()
        if isinstance(existing, QVBoxLayout):
            layout = existing
        else:
            if existing:
                _clear_layout(existing)
            layout = QVBoxLayout(parent_w)
            layout.setContentsMargins(int(self._padding_left), int(self._padding_top),
                                      int(self._padding_right), int(self._padding_bottom))
            layout.setSpacing(int(self._spacing))
            parent_w.setLayout(layout)
        align_map = {0: Qt.AlignmentFlag.AlignTop, 1: Qt.AlignmentFlag.AlignVCenter,
                     2: Qt.AlignmentFlag.AlignBottom}
        layout.setAlignment(align_map.get(self._alignment, Qt.AlignmentFlag.AlignTop))
        _apply_layout_children(parent_w, layout, self._child_control_width, self._child_control_height)


@ComponentRegistry.register
class GridLayoutComponent(Component):
    _allow_multiple = False
    _show_gizmo_icon = True
    _gizmo_icon_color = (60, 140, 200)
    _gizmo_icon_label = "G"
    _spacing_x: int = 4
    _spacing_y: int = 4
    _padding_left: int = 0
    _padding_top: int = 0
    _padding_right: int = 0
    _padding_bottom: int = 0
    _columns: int = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            if hasattr(self.__class__, k):
                setattr(self, k, v)

    def _sync_layout(self, canvas: GuiCanvas):
        if not self.entity:
            return
        parent_w = _find_entity_widget(self.entity)
        if not parent_w:
            return
        existing = parent_w.layout()
        if isinstance(existing, QGridLayout):
            layout = existing
        else:
            if existing:
                _clear_layout(existing)
            layout = QGridLayout(parent_w)
            layout.setContentsMargins(int(self._padding_left), int(self._padding_top),
                                      int(self._padding_right), int(self._padding_bottom))
            layout.setHorizontalSpacing(int(self._spacing_x))
            layout.setVerticalSpacing(int(self._spacing_y))
            parent_w.setLayout(layout)
        _apply_grid_children(parent_w, layout, int(self._columns))


def _clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().setParent(None)
        elif item.layout():
            _clear_layout(item.layout())
