# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from core.gui.canvas import GuiCanvas
from core.gui.api import GuiApi

if TYPE_CHECKING:
    from core.ecs.ecs import Entity


_GUI_COMP_NAMES: list[str] = []
_GUI_COMP_TYPES: frozenset = frozenset()
_LAYOUT_COMP_NAMES: list[str] = []
_LAYOUT_COMP_TYPES: frozenset = frozenset()
_LAYOUT_ELEMENT_NAME: str = ""
_LAYOUT_ELEMENT_TYPE = None


def _get_gui_comp_names() -> list[str]:
    global _GUI_COMP_NAMES
    if not _GUI_COMP_NAMES:
        from core.components.gui import _ensure_component_map
        _GUI_COMP_NAMES = [c.__name__ for c in _ensure_component_map().values()]
    return _GUI_COMP_NAMES


def _get_gui_comp_types() -> frozenset:
    global _GUI_COMP_TYPES
    if not _GUI_COMP_TYPES:
        from core.components.gui import _ensure_component_map
        _GUI_COMP_TYPES = frozenset(_ensure_component_map().values())
    return _GUI_COMP_TYPES


def _get_layout_comp_names() -> list[str]:
    global _LAYOUT_COMP_NAMES
    if not _LAYOUT_COMP_NAMES:
        from core.components.gui import LAYOUT_COMP_NAMES as _LCN
        _LAYOUT_COMP_NAMES = list(_LCN)
    return _LAYOUT_COMP_NAMES


def _get_layout_comp_types() -> frozenset:
    global _LAYOUT_COMP_TYPES
    if not _LAYOUT_COMP_TYPES:
        from core.components import gui as _gui_mod
        _LAYOUT_COMP_TYPES = frozenset(getattr(_gui_mod, n) for n in _get_layout_comp_names())
    return _LAYOUT_COMP_TYPES


def _get_layout_element_name() -> str:
    global _LAYOUT_ELEMENT_NAME
    if not _LAYOUT_ELEMENT_NAME:
        from core.components.gui import LAYOUT_ELEMENT_NAME
        _LAYOUT_ELEMENT_NAME = LAYOUT_ELEMENT_NAME
    return _LAYOUT_ELEMENT_NAME


def _get_layout_element_type():
    global _LAYOUT_ELEMENT_TYPE
    if _LAYOUT_ELEMENT_TYPE is None:
        from core.components import gui as _gui_mod
        _LAYOUT_ELEMENT_TYPE = getattr(_gui_mod, _get_layout_element_name())
    return _LAYOUT_ELEMENT_TYPE


def _find_gui_comp(entity: Entity):
    gui_types = _get_gui_comp_types()
    type_map = entity._type_map
    for cls in type_map:
        if cls in gui_types:
            return type_map[cls][0]
    return None


class GuiCanvasSystem:
    _instance: Optional[GuiCanvasSystem] = None

    def __init__(self):
        self._dirty_entities: set[str] = set()
        GuiCanvasSystem._instance = self

    @staticmethod
    def instance() -> GuiCanvasSystem:
        if GuiCanvasSystem._instance is None:
            GuiCanvasSystem._instance = GuiCanvasSystem()
        return GuiCanvasSystem._instance

    def mark_dirty(self, entity_id: str):
        self._dirty_entities.add(entity_id)

    def sync_entity(self, entity: Entity, canvas: GuiCanvas):
        comp = _find_gui_comp(entity)
        if not comp:
            return
        if not comp._widget_ref:
            comp._create_widget(canvas)
        else:
            comp.sync_to_widget()
        self._dirty_entities.discard(entity.id)
        self._sync_layout_for(entity, canvas)

    def rebuild_from_scene(self, scene, canvas: GuiCanvas):
        if scene:
            for entity in scene.get_all_entities():
                comp = _find_gui_comp(entity)
                if comp:
                    comp._widget_ref = None
                    comp._sub_window_ref = None
        canvas.clear()
        if not scene:
            return
        for entity in _sorted_by_depth(scene):
            comp = _find_gui_comp(entity)
            if comp and entity.active:
                comp._create_widget(canvas)
            self._sync_layout_for(entity, canvas)

    def sync_all(self, scene, canvas: GuiCanvas):
        if not scene:
            return
        gui_types = _get_gui_comp_types()
        has_gui = False
        for entity in scene.get_all_entities():
            type_map = entity._type_map
            for cls in type_map:
                if cls in gui_types:
                    has_gui = True
                    break
            if has_gui:
                break
        if not has_gui:
            return
        for entity in _sorted_by_depth(scene):
            comp = _find_gui_comp(entity)
            if not comp or not entity.active:
                continue
            if not comp._widget_ref:
                comp._create_widget(canvas)
            else:
                comp.sync_to_widget()
            self._sync_layout_for(entity, canvas)

    def _sync_layout_for(self, entity, canvas):
        layout_types = _get_layout_comp_types()
        type_map = entity._type_map
        has_layout = False
        for cls in type_map:
            if cls in layout_types:
                has_layout = True
                break
        if not has_layout:
            return
        for lname in _get_layout_comp_names():
            lc = entity.get_component_by_name(lname)
            if lc and hasattr(lc, '_sync_layout'):
                lc._sync_layout(canvas)
        le = entity.get_component_by_name(_get_layout_element_name())
        if le and hasattr(le, '_sync_layout_element'):
            from core.components.gui import _ensure_component_map
            for c in _ensure_component_map().values():
                comp = entity.get_component_by_name(c.__name__)
                if comp and hasattr(comp, '_widget_ref'):
                    le._sync_layout_element(comp._widget_ref)
                    break


def _sorted_by_depth(scene):
    def _depth(e):
        d = 0
        seen = set()
        while e.parent and e.parent.id not in seen:
            seen.add(e.parent.id)
            d += 1
            if d > 1000:
                return 0
            e = e.parent
        return d
    return sorted(scene.get_all_entities(), key=_depth)
