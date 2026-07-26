# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os

from PyQt6.QtWidgets import QDockWidget
from PyQt6.QtCore import Qt

try:
    import qtawesome as qta
except ImportError:
    qta = None

DOCK_ICONS = {
    "ViewportDock": "fa5s.eye",
    "HierarchyDock": "fa5s.sitemap",
    "InspectorDock": "fa5s.search",
    "ConsoleDock": "fa5s.terminal",
    "ProfilerDock": "fa5s.chart-bar",
    "PluginManagerDock": "fa5s.puzzle-piece",
    "ProjectDock": "fa5s.folder-open",
    "PlayDock": "fa5s.play",
    "TerminalDock": "fa5s.window-maximize",
    "UndoHistoryDock": "fa5s.undo",
    "CollaborationDock": "fa5s.users",
    "MeshEditorDock": "fa5s.draw-polygon",
    "TerrainEditorDock": "fa5s.mountain",
    "AnimationDock": "fa5s.film",
    "AnimatorDock": "fa5s.running",
    "ShadersDock": "fa5s.code",
    "ScriptEditorDock": "fa5s.file-code",
    "TracemallocDebugDock": "fa5s.bug",
    "TimeTravelDock": "fa5s.clock",
    "VersionControlDock": "fa5s.code-branch",
    "GuiEditorDock": "fa5s.object-group",
}

from editor.scene_viewport import SceneViewport
from editor.panels.hierarchy_panel import HierarchyPanel
from editor.inspector import InspectorPanel
from editor.panels.console_panel import ConsolePanel
from editor.panels.profiler_panel import ProfilerPanel
from editor.panels.plugin_manager_panel import PluginManagerPanel
from editor.panels.project_panel import ProjectPanel

from editor.panels.play_window import PlayDockPanel
from editor.panels.terminal_panel import TerminalPanel
from editor.panels.undo_history_panel import UndoHistoryPanel
from editor.panels.collaboration_panel import CollaborationPanel
from editor.panels.mesh_editor_panel import MeshEditorPanel
from editor.panels.terrain_panel import TerrainPanel
from editor.panels.animation_panel import AnimationPanel
from editor.panels.animator_panel import AnimatorPanel
from editor.panels.scripts_panel import ScriptsPanel
from editor.panels.script_editor_panel import ScriptEditorPanel
from editor.panels.tracemalloc_panel import TracemallocPanel
from editor.panels.time_travel_panel import TimeTravelPanel
from editor.panels.vcs_panel import VcsPanel
from editor.gui_editor.gui_viewport import GuiEditorViewport

_AREA_MAP = {
    "left": Qt.DockWidgetArea.LeftDockWidgetArea,
    "right": Qt.DockWidgetArea.RightDockWidgetArea,
    "top": Qt.DockWidgetArea.TopDockWidgetArea,
    "bottom": Qt.DockWidgetArea.BottomDockWidgetArea,
}


def _apply_dock_icons(mw):
    if qta is None:
        return
    for dock in mw._docks:
        obj = dock.objectName()
        icon_name = DOCK_ICONS.get(obj)
        if icon_name:
            dock.setWindowIcon(qta.icon(icon_name, color="#d4d4d4"))


def _make_dock(mw, title: str, widget, obj_name: str = "") -> QDockWidget:
    dock = QDockWidget(title, mw)
    if obj_name:
        dock.setObjectName(obj_name)
    dock.setWidget(widget)
    dock.setMinimumSize(40, 40)
    dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    dock.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    return dock


def setup_docks(mw):
    mw._docks: list[QDockWidget] = []
    register_default_docks(mw)
    register_plugin_docks(mw)
    add_all_docks(mw)
    state = mw._settings.value("windowState")
    if state is not None and mw.restoreState(state):
        mw._layout_restored = True
    else:
        mw._layout_restored = False
        build_dock_layout(mw)


def register_dock(mw, dock: QDockWidget, area: Qt.DockWidgetArea) -> QDockWidget:
    dock.setMinimumSize(40, 40)
    mw._docks.append(dock)
    return dock


def register_default_docks(mw):
    from core.config.config import get_global_config
    mw._viewport = SceneViewport(mw._engine, mw)
    mw._engine.viewport = mw._viewport
    mw._viewport.load_config(get_global_config())
    mw._viewport.camera.load_config(get_global_config())
    mw._viewport.gizmo.load_config(get_global_config())
    mw._viewport_dock = QDockWidget("Viewport", mw)
    mw._viewport_dock.setObjectName("ViewportDock")
    mw._viewport_dock.setWidget(mw._viewport)
    mw._viewport_dock.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    mw._viewport_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._viewport_dock.setContentsMargins(0, 0, 0, 0)
    mw._viewport_dock.setStyleSheet(
        "QDockWidget { border: none; } QDockWidget::title { padding: 2px 6px; font-size: 11px; }")
    mw._viewport_dock.setMinimumSize(40, 40)
    mw._viewport_dock.topLevelChanged.connect(mw._viewport.on_dock_top_level_changed)
    register_dock(mw, mw._viewport_dock, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._hierarchy = HierarchyPanel(mw._engine, mw)
    mw._hierarchy.load_config(get_global_config())
    mw._hierarchy.setObjectName("HierarchyDock")
    mw._hierarchy.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._hierarchy.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._hierarchy, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._inspector = InspectorPanel(mw._engine, mw)
    mw._inspector.load_config(get_global_config())
    mw._inspector.setObjectName("InspectorDock")
    mw._inspector.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._inspector.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._inspector, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._console = ConsolePanel(mw)
    mw._console.load_config(get_global_config())
    mw._console.setObjectName("ConsoleDock")
    mw._console.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._console.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._console, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._profiler = ProfilerPanel(mw._engine, mw)
    mw._profiler.load_config(get_global_config())
    mw._profiler.setObjectName("ProfilerDock")
    mw._profiler.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._profiler.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._profiler, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._plugin_mgr = PluginManagerPanel(mw._engine, mw)
    mw._plugin_mgr.setObjectName("PluginManagerDock")
    mw._plugin_mgr.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._plugin_mgr.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._plugin_mgr, Qt.DockWidgetArea.LeftDockWidgetArea)
    _assets_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))
    mw._project = ProjectPanel(mw._engine, _assets_root, mw)
    mw._project.load_config(get_global_config())
    mw._project.setObjectName("ProjectDock")
    mw._project.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._project.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._project, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._play_dock = PlayDockPanel(mw._engine, mw)
    register_dock(mw, mw._play_dock, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._terminal = TerminalPanel(mw)
    mw._terminal.setObjectName("TerminalDock")
    mw._terminal.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    register_dock(mw, mw._terminal, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._gui_editor_widget = GuiEditorViewport(mw._engine, mw)
    mw._gui_editor = QDockWidget("GUI Editor", mw)
    mw._gui_editor.setObjectName("GuiEditorDock")
    mw._gui_editor.setWidget(mw._gui_editor_widget)
    mw._gui_editor.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._gui_editor.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._gui_editor, Qt.DockWidgetArea.LeftDockWidgetArea)
    from editor.main_window.handlers import on_entity_selected
    mw._gui_editor_widget.entity_selected.connect(lambda e: on_entity_selected(mw, e))
    mw._undo_history = UndoHistoryPanel(mw)
    mw._undo_history.setObjectName("UndoHistoryDock")
    mw._undo_history.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._undo_history.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._undo_history, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._collab_panel = CollaborationPanel(mw._engine, mw)
    mw._collab_panel.setObjectName("CollaborationDock")
    mw._collab_panel.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    if mw._engine.collab_manager:
        mw._collab_panel.set_collaboration_manager(mw._engine.collab_manager)
    from editor.viewport.collaboration import setup_collab_undo_redo_hooks
    setup_collab_undo_redo_hooks(mw._engine)
    register_dock(mw, mw._collab_panel, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._mesh_editor = MeshEditorPanel(mw._engine, mw)
    mw._mesh_editor.setObjectName("MeshEditorDock")
    register_dock(mw, mw._mesh_editor, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._terrain_editor = TerrainPanel(mw._engine, mw)
    mw._terrain_editor.setObjectName("TerrainEditorDock")
    if mw._engine.collab_manager:
        mw._terrain_editor.set_collaboration_manager(mw._engine.collab_manager)
    register_dock(mw, mw._terrain_editor, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._animation = AnimationPanel(mw._engine, mw)
    mw._animation.load_config(get_global_config())
    mw._animation.setObjectName("AnimationDock")
    mw._animation.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._animation.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._animation, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._animator = AnimatorPanel(mw._engine, mw)
    mw._animator.load_config(get_global_config())
    mw._animator.setObjectName("AnimatorDock")
    mw._animator.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._animator.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._animator, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._animator.state_selected_signal.connect(mw._inspector.show_animator_state)
    mw._animator.transition_selected_signal.connect(mw._inspector.show_animator_transition)
    mw._animator.selection_cleared.connect(mw._inspector.clear_animator_mode)
    mw._scripts = ScriptsPanel(mw._engine, mw)
    mw._scripts.setObjectName("ShadersDock")
    mw._scripts.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._scripts.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._scripts, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._script_editor = ScriptEditorPanel(mw._engine, mw)
    mw._script_editor.setObjectName("ScriptEditorDock")
    mw._script_editor.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._script_editor.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._script_editor, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._tracemalloc = TracemallocPanel(mw)
    mw._tracemalloc.setObjectName("TracemallocDebugDock")
    mw._tracemalloc.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._tracemalloc.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._tracemalloc, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._time_travel = TimeTravelPanel(mw._engine, mw)
    mw._time_travel.setObjectName("TimeTravelDock")
    mw._time_travel.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._time_travel.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._time_travel, Qt.DockWidgetArea.LeftDockWidgetArea)
    mw._vcs = VcsPanel(mw)
    mw._vcs.setObjectName("VersionControlDock")
    mw._vcs.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    mw._vcs.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable |
        QDockWidget.DockWidgetFeature.DockWidgetClosable)
    register_dock(mw, mw._vcs, Qt.DockWidgetArea.LeftDockWidgetArea)
    _apply_dock_icons(mw)


def register_plugin_docks(mw):
    registry = mw._engine.plugin_ui_registry
    for info in registry["docks"]:
        try:
            widget = info["widget_factory"]()
            if widget is None:
                continue
            title = info["title"]
            area_name = info.get("area", "left")
            area = _AREA_MAP.get(area_name, Qt.DockWidgetArea.LeftDockWidgetArea)
            plugin_name = info.get("plugin", "plugin")
            obj_name = f"PluginDock_{plugin_name}_{title.replace(' ', '_')}"
            dock = _make_dock(mw, title, widget, obj_name)
            register_dock(mw, dock, area)
        except Exception as e:
            from core.foundation.logger import Logger
            Logger.error(f"Failed to create plugin dock '{info.get('title', '?')}': {e}")


def add_all_docks(mw):
    area = Qt.DockWidgetArea.LeftDockWidgetArea
    mw.addDockWidget(area, mw._hierarchy)
    mw.addDockWidget(area, mw._viewport_dock)
    mw.addDockWidget(area, mw._inspector)
    mw.addDockWidget(area, mw._play_dock)
    mw.addDockWidget(area, mw._gui_editor)
    mw.addDockWidget(area, mw._console)
    mw.addDockWidget(area, mw._profiler)
    mw.addDockWidget(area, mw._project)
    mw.addDockWidget(area, mw._terminal)
    mw.addDockWidget(area, mw._undo_history)
    mw.addDockWidget(area, mw._plugin_mgr)
    mw.addDockWidget(area, mw._collab_panel)
    mw.addDockWidget(area, mw._mesh_editor)
    mw.addDockWidget(area, mw._terrain_editor)
    mw.addDockWidget(area, mw._animation)
    mw.addDockWidget(area, mw._animator)
    mw.addDockWidget(area, mw._scripts)
    mw.addDockWidget(area, mw._script_editor)
    mw.addDockWidget(area, mw._tracemalloc)
    mw.addDockWidget(area, mw._time_travel)
    mw.addDockWidget(area, mw._vcs)
    for dock in mw._docks:
        if dock not in (                         mw._hierarchy, mw._viewport_dock, mw._inspector,
                        mw._play_dock, mw._gui_editor,
                        mw._console, mw._profiler, mw._project,
                        mw._terminal, mw._undo_history, mw._plugin_mgr,
                        mw._collab_panel, mw._mesh_editor, mw._terrain_editor, mw._animation,
                        mw._animator, mw._scripts, mw._script_editor, mw._tracemalloc,
                        mw._time_travel, mw._vcs):
            mw.addDockWidget(area, dock)


def build_dock_layout(mw):
    mw.splitDockWidget(mw._hierarchy, mw._viewport_dock, Qt.Orientation.Horizontal)
    mw.splitDockWidget(mw._viewport_dock, mw._inspector, Qt.Orientation.Horizontal)
    mw.splitDockWidget(mw._hierarchy, mw._collab_panel, Qt.Orientation.Vertical)
    mw.splitDockWidget(mw._viewport_dock, mw._project, Qt.Orientation.Vertical)
    mw.splitDockWidget(mw._inspector, mw._console, Qt.Orientation.Vertical)
    mw.tabifyDockWidget(mw._viewport_dock, mw._play_dock)
    mw.tabifyDockWidget(mw._viewport_dock, mw._gui_editor)
    mw.tabifyDockWidget(mw._viewport_dock, mw._animation)
    mw.tabifyDockWidget(mw._viewport_dock, mw._animator)
    mw.tabifyDockWidget(mw._viewport_dock, mw._scripts)
    mw.tabifyDockWidget(mw._scripts, mw._script_editor)
    mw.tabifyDockWidget(mw._project, mw._undo_history)
    mw.tabifyDockWidget(mw._project, mw._profiler)
    mw.tabifyDockWidget(mw._project, mw._plugin_mgr)
    mw.tabifyDockWidget(mw._project, mw._mesh_editor)
    mw.tabifyDockWidget(mw._project, mw._terrain_editor)
    mw.tabifyDockWidget(mw._project, mw._tracemalloc)
    mw.tabifyDockWidget(mw._project, mw._time_travel)
    mw.tabifyDockWidget(mw._project, mw._vcs)
    mw.tabifyDockWidget(mw._console, mw._terminal)
    mw._viewport_dock.raise_()
    mw._hierarchy.raise_()
    mw._inspector.raise_()
    mw._project.raise_()
    mw._console.raise_()
