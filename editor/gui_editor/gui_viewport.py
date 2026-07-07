# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import uuid
from typing import Optional
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QDockWidget,
                             QSplitter, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from core.gui import GuiCanvas, GuiApi
from core.ecs import ComponentRegistry
from core.gui.widgets import (
    GuiWidget, Panel, Button, Label, Slider, TextInput, Image,
    Toggle, ProgressBar, Dropdown, ScrollPanel, HtmlView,
    Splitter, StackedWidget, ToolBox, Calendar, LCDNumber,
    PlainText, ScrollBar, ToolButton, FontCombo, MdiArea,
)
from core.gui.system import GuiCanvasSystem
from core.components.gui import _ensure_component_map
from editor.gui_editor.gui_toolbar import GuiEditorToolbar
from editor.gui_editor.widget_palette import WidgetPalette
from core.editor_scale import scale, scale_xy

GUI_COMPONENT_MAP = _ensure_component_map()


WIDGET_TYPE_MAP = {
    "panel": Panel, "label": Label, "button": Button, "slider": Slider,
    "textinput": TextInput, "image": Image, "toggle": Toggle,
    "progressbar": ProgressBar, "dropdown": Dropdown, "scrollpanel": ScrollPanel,
    "html": HtmlView,
    "splitter": Splitter, "stackedwidget": StackedWidget, "toolbox": ToolBox,
    "calendar": Calendar, "lcdnumber": LCDNumber, "plaintext": PlainText,
    "scrollbar": ScrollBar, "toolbutton": ToolButton, "fontcombo": FontCombo,
    "mdiarea": MdiArea,
}

PALETTE_TO_COMPONENT = {
    "panel": "PanelComponent", "label": "LabelComponent", "button": "ButtonComponent",
    "slider": "SliderComponent", "textinput": "TextInputComponent", "image": "ImageComponent",
    "toggle": "ToggleComponent", "progressbar": "ProgressBarComponent",
    "dropdown": "DropdownComponent", "scrollpanel": "ScrollPanelComponent",
    "html": "HtmlComponent",
    "splitter": "SplitterComponent", "stackedwidget": "StackedComponent",
    "toolbox": "ToolBoxComponent", "calendar": "CalendarComponent",
    "lcdnumber": "LCDComponent", "plaintext": "PlainTextComponent",
    "scrollbar": "ScrollBarComponent", "toolbutton": "ToolButtonComponent",
    "fontcombo": "FontComboComponent",
    "horizontallayout": "HorizontalLayoutComponent",
    "verticallayout": "VerticalLayoutComponent",
    "gridlayout": "GridLayoutComponent",
    "layoutelement": "LayoutElementComponent",
    "mdiarea": "MdiAreaComponent",
}

_LAYOUT_ONLY = {"horizontallayout", "verticallayout", "gridlayout", "layoutelement"}


class GuiEditorViewport(QWidget):
    entity_selected = pyqtSignal(object)

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._gui_api: Optional[GuiApi] = None
        self._clipboard_data: Optional[dict] = None
        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(self._sync_scene)
        self._sync_timer.start(33)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._toolbar = GuiEditorToolbar()
        self._toolbar.setFixedHeight(scale(32))
        layout.addWidget(self._toolbar)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._palette = WidgetPalette()
        splitter.addWidget(self._palette)
        self._canvas = GuiCanvas()
        self._canvas.edit_mode = True
        self._canvas.setAcceptDrops(True)
        splitter.addWidget(self._canvas)
        splitter.setSizes([140, 600])
        layout.addWidget(splitter)
        self._setup_connections()
        self._gui_name = "editor_gui"
        self._create_default_api()
        GuiCanvasSystem.instance()

    def _setup_connections(self):
        self._toolbar.toggle_grid.connect(lambda v: setattr(self._canvas, "_show_grid", v) or self._canvas.update())
        self._toolbar.toggle_snap.connect(lambda v: setattr(self._canvas, "snap_to_grid", v))
        self._toolbar.toggle_auto_align.connect(lambda v: setattr(self._canvas, "auto_align", v))
        self._toolbar.grid_size_changed.connect(lambda v: setattr(self._canvas, "grid_size", v))
        self._toolbar.clear_requested.connect(self._on_clear)
        self._toolbar.save_requested.connect(self._on_save)
        self._toolbar.load_requested.connect(self._on_load)
        self._toolbar._mode_btn.toggled.connect(self._on_mode_toggled)
        self._toolbar.zoom_changed.connect(self._on_zoom)
        self._toolbar.screen_w_changed.connect(lambda v: setattr(self._canvas, 'screen_width', v))
        self._toolbar.screen_h_changed.connect(lambda v: setattr(self._canvas, 'screen_height', v))
        self._canvas.zoom_changed.connect(lambda z: self._toolbar.set_zoom_label(int(round(z * 100))))
        self._palette.widget_added.connect(self._on_palette_add)
        self._canvas.widget_selected.connect(self._on_canvas_selection)
        self._canvas.widget_changed.connect(self._on_widget_changed)
        self._canvas.copy_requested.connect(self._on_copy)
        self._canvas.paste_requested.connect(self._on_paste)
        self._canvas.selected_widget = None
        self._canvas.update()
        self._canvas.installEventFilter(self)

    def _on_zoom(self, delta: float):
        z = self._canvas.zoom + delta
        self._canvas.zoom = z

    def _create_default_api(self):
        self._gui_api = GuiApi(self._gui_name, canvas=self._canvas)
        if self._engine:
            self._engine.on("play_stop", self._on_play_stop)
        self._engine.on("scene_loaded", self._on_scene_loaded)

    def _on_play_stop(self, _):
        self._canvas.clear()
        if self._engine and self._engine.scene:
            from core.gui.system import _find_gui_comp
            for e in self._engine.scene.get_all_entities():
                comp = _find_gui_comp(e)
                if comp:
                    comp._widget_ref = None

    def _on_scene_loaded(self, _):
        self._canvas.clear()
        if self._engine and self._engine.scene:
            from core.gui.system import _find_gui_comp
            for e in self._engine.scene.get_all_entities():
                comp = _find_gui_comp(e)
                if comp:
                    comp._widget_ref = None

    def _sync_scene(self):
        if not self._engine or not hasattr(self._engine, 'scene') or not self._engine.scene:
            return
        GuiCanvasSystem.instance().sync_all(self._engine.scene, self._canvas)

    def _on_palette_add(self, widget_type: str):
        if not self._engine or not self._engine.scene:
            return
        scene = self._engine.scene
        comp_class = GUI_COMPONENT_MAP.get(widget_type)
        if widget_type in _LAYOUT_ONLY:
            from core.components.gui.horizontal_layout_component import (
                HorizontalLayoutComponent, VerticalLayoutComponent, GridLayoutComponent,
            )
            from core.components.gui.layout_element_component import LayoutElementComponent
            _LCOMP = {
                "horizontallayout": HorizontalLayoutComponent,
                "verticallayout": VerticalLayoutComponent,
                "gridlayout": GridLayoutComponent,
                "layoutelement": LayoutElementComponent,
            }
            lc = _LCOMP.get(widget_type)
            if not lc:
                return
            sel = self._canvas.selected_widget
            entity = None
            if sel and hasattr(sel, '_widget_id'):
                entity = scene.get_entity(sel._widget_id)
            if entity:
                comp = lc()
                entity.add_component(comp)
            return
        if not comp_class:
            return
        name = f"UI_{widget_type}_{uuid.uuid4().hex[:6]}"
        entity = scene.create_entity(name)
        sel = self._canvas.selected_widget
        if sel and hasattr(sel, '_widget_id') and sel._widget_id:
            pe = scene.get_entity(sel._widget_id)
            if pe and pe.parent:
                pe = pe.parent
            if pe:
                entity.set_parent(pe)
        from core.components.transform import Transform
        t = Transform()
        t.local_position = (50, 50, 0)
        t.local_scale = (1, 1, 1)
        entity.add_component(t)
        cw = getattr(comp_class, '_default_w', 120)
        ch = getattr(comp_class, '_default_h', 36)
        comp = comp_class(widget_width=cw, widget_height=ch)
        entity.add_component(comp)
        comp._create_widget(self._canvas)
        target = comp._sub_window_ref or comp._widget_ref
        self._canvas.selected_widget = target

    def _on_widget_changed(self, widget):
        if not self._engine or not self._engine.scene:
            return
        if widget and hasattr(widget, '_widget_id') and widget._widget_id:
            entity = self._engine.scene.get_entity(widget._widget_id)
            if entity:
                for comp_cls in GUI_COMPONENT_MAP.values():
                    comp = entity.get_component_by_name(comp_cls.__name__)
                    if comp and hasattr(comp, 'update_from_widget'):
                        comp.update_from_widget()
                        return

    def _on_canvas_selection(self, widget: Optional[GuiWidget]):
        if widget and hasattr(widget, '_widget_id') and widget._widget_id:
            entity = self._engine.scene.get_entity(widget._widget_id) if self._engine.scene else None
            if entity:
                self.entity_selected.emit(entity)
                return
        self.entity_selected.emit(None)

    def _on_copy(self):
        if not self._engine or not self._engine.scene:
            return
        w = self._canvas.selected_widget
        if not w or not hasattr(w, '_widget_id') or not w._widget_id:
            return
        entity = self._engine.scene.get_entity(w._widget_id)
        if not entity:
            return
        self._clipboard_data = entity.serialize()

    def _on_paste(self):
        if not self._clipboard_data or not self._engine or not self._engine.scene:
            return
        scene = self._engine.scene
        data = self._clipboard_data
        name = data.get("name", "Entity")
        import copy
        data = copy.deepcopy(data)
        entity = scene.create_entity(name + "_copy")
        from core.components.transform import Transform
        data["id"] = entity.id
        data["parent"] = None
        for cd in data.get("components", []):
            ctype = cd.get("type")
            comp_cls = ComponentRegistry.get(ctype)
            if comp_cls:
                comp = comp_cls.deserialize(cd)
                entity.add_component(comp)
        t = entity.transform
        if t:
            t.local_position = (t.local_position.x + 25, t.local_position.y + 25, t.local_position.z)
        from core.gui.system import _find_gui_comp
        comp = _find_gui_comp(entity)
        if comp and hasattr(comp, '_create_widget'):
            comp._create_widget(self._canvas)
            target = comp._sub_window_ref or comp._widget_ref
            self._canvas.selected_widget = target

    def _on_clear(self):
        if not self._engine or not self._engine.scene:
            return
        reply = QMessageBox.question(self, "Clear Canvas", "Remove all GUI widgets?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            to_remove = []
            comp_names = [c.__name__ for c in GUI_COMPONENT_MAP.values()]
            for e in self._engine.scene.get_all_entities():
                for cn in comp_names:
                    if e.get_component_by_name(cn):
                        to_remove.append(e)
                        break
            for e in to_remove:
                self._engine.scene.remove_entity(e.id)
            self._canvas.clear()
            self._canvas.scene_modified.emit()

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save GUI Layout", "assets/",
                                               "GUI Layout (*.guilayout)")
        if path:
            if not path.endswith(".guilayout"):
                path += ".guilayout"
            self._canvas.save_to_file(path)

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load GUI Layout", "assets/",
                                               "GUI Layout (*.guilayout)")
        if path:
            self._canvas.load_from_file(path)

    def _on_mode_toggled(self, edit_mode: bool):
        self._canvas.edit_mode = edit_mode

    @property
    def canvas(self) -> GuiCanvas:
        return self._canvas

    @property
    def gui_api(self) -> Optional[GuiApi]:
        return self._gui_api

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if obj is self._canvas and event.type() == QEvent.Type.MouseButtonRelease:
            sel = self._canvas.selected_widget
            if sel and hasattr(sel, '_widget_id') and sel._widget_id:
                entity = self._engine.scene.get_entity(sel._widget_id) if self._engine.scene else None
                if entity:
                    self.entity_selected.emit(entity)
        return super().eventFilter(obj, event)

    def load_config(self, config):
        pass

    def save_config(self, config):
        pass
