# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional
from PyQt6.QtWidgets import QWidget, QMdiArea, QMdiSubWindow, QTabWidget
from PyQt6.QtCore import Qt
from core.ecs.ecs import ComponentRegistry
from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
from core.gui.widgets import MdiArea, ANCHOR_STRETCH_ALL
from core.components.inspector_meta import InspectorField, FieldType


@ComponentRegistry.register
class MdiAreaComponent(GuiWidgetComponentBase):
    _widget_class = MdiArea
    _widget_obj_name = "GuiMdiArea"
    _default_text = "MdiArea"
    _default_w = 400.0
    _default_h = 300.0

    _view_mode: int = 0
    _tab_shape: int = 0
    _scrollbar_policy: int = 1
    _activation_order: int = 0

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("_view_mode", "View Mode", FieldType.INT, min_val=0, max_val=1),
            InspectorField("_tab_shape", "Tab Shape", FieldType.INT, min_val=0, max_val=1),
            InspectorField("_scrollbar_policy", "Scrollbars", FieldType.INT, min_val=0, max_val=2),
            InspectorField("_activation_order", "Activation Order", FieldType.INT, min_val=0, max_val=2),
        ] + cls._common_inspector_fields()

    def _create_widget(self, canvas):
        if self._widget_ref:
            return
        cls = self._widget_class
        if not cls:
            return
        parent_w = self._get_parent_widget(canvas)
        sw, sh = int(canvas._screen_w), int(canvas._screen_h)
        widget = cls(0, 0, sw, sh, parent=parent_w)
        widget._widget_id = self.entity.id if self.entity else None
        widget._canvas_ref = canvas
        widget._anchor = ANCHOR_STRETCH_ALL
        canvas.add_widget(widget, parent_w)
        widget.lower()
        self._widget_ref = widget
        self._apply_style_to_widget()
        self.sync_to_widget()

    def sync_to_widget(self):
        super().sync_to_widget()
        w = self._widget_ref
        if not w or not isinstance(w, QMdiArea):
            return
        w._anchor = ANCHOR_STRETCH_ALL
        cr = getattr(w, '_canvas_ref', None)
        if cr and hasattr(cr, '_screen_w') and hasattr(cr, '_screen_h'):
            w.setMinimumSize(0, 0)
            w.setMaximumSize(16777215, 16777215)
            w.setGeometry(0, 0, int(cr._screen_w), int(cr._screen_h))
        w.setStyleSheet("background-color: transparent;")
        if hasattr(w, 'viewport') and w.viewport():
            w.viewport().setStyleSheet("background-color: transparent;")
        w.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        if hasattr(w, 'viewport') and w.viewport():
            w.viewport().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        if self._view_mode == 0:
            w.setViewMode(QMdiArea.ViewMode.SubWindowView)
        else:
            w.setViewMode(QMdiArea.ViewMode.TabbedView)
        shape_map = {0: QTabWidget.TabShape.Rounded, 1: QTabWidget.TabShape.Triangular}
        w.setTabShape(shape_map.get(self._tab_shape, QTabWidget.TabShape.Rounded))
        sb_map = {
            0: Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            1: Qt.ScrollBarPolicy.ScrollBarAsNeeded,
            2: Qt.ScrollBarPolicy.ScrollBarAlwaysOn,
        }
        w.setHorizontalScrollBarPolicy(sb_map.get(self._scrollbar_policy, Qt.ScrollBarPolicy.ScrollBarAsNeeded))
        w.setVerticalScrollBarPolicy(sb_map.get(self._scrollbar_policy, Qt.ScrollBarPolicy.ScrollBarAsNeeded))
        ao_map = {
            0: QMdiArea.WindowOrder.CreationOrder,
            1: QMdiArea.WindowOrder.StackingOrder,
            2: QMdiArea.WindowOrder.ActivationHistoryOrder,
        }
        w.setActivationOrder(ao_map.get(self._activation_order, QMdiArea.WindowOrder.CreationOrder))

    def update_from_widget(self):
        super().update_from_widget()
        w = self._widget_ref
        if not w or not isinstance(w, QMdiArea):
            return
        self._view_mode = 0 if w.viewMode() == QMdiArea.ViewMode.SubWindowView else 1
        self._tab_shape = 0 if w.tabShape() == QTabWidget.TabShape.Rounded else 1

    def add_sub_window(self, widget: QWidget, title: str = "Window") -> Optional[QMdiSubWindow]:
        w = self._widget_ref
        if not w or not isinstance(w, QMdiArea):
            return None
        sw = QMdiSubWindow(w)
        sw.setWidget(widget)
        sw.setWindowTitle(title)
        sw.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        w.addSubWindow(sw)
        sw.show()
        return sw

    def tile_windows(self):
        w = self._widget_ref
        if w and isinstance(w, QMdiArea):
            w.tileSubWindows()

    def cascade_windows(self):
        w = self._widget_ref
        if w and isinstance(w, QMdiArea):
            w.cascadeSubWindows()

    def close_all(self):
        w = self._widget_ref
        if w and isinstance(w, QMdiArea):
            w.closeAllSubWindows()

    @property
    def sub_window_list(self) -> list[QMdiSubWindow]:
        w = self._widget_ref
        if w and isinstance(w, QMdiArea):
            return w.subWindowList()
        return []

    @property
    def active_sub_window(self) -> Optional[QMdiSubWindow]:
        w = self._widget_ref
        if w and isinstance(w, QMdiArea):
            return w.activeSubWindow()
        return None

    def serialize(self) -> dict:
        d = super().serialize()
        d["_view_mode"] = self._view_mode
        d["_tab_shape"] = self._tab_shape
        d["_scrollbar_policy"] = self._scrollbar_policy
        d["_activation_order"] = self._activation_order
        return d

    @classmethod
    def deserialize(cls, data: dict) -> MdiAreaComponent:
        inst: MdiAreaComponent = super().deserialize(data)
        inst._view_mode = data.get("_view_mode", 0)
        inst._tab_shape = data.get("_tab_shape", 0)
        inst._scrollbar_policy = data.get("_scrollbar_policy", 1)
        inst._activation_order = data.get("_activation_order", 0)
        return inst
