# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os
from collections import defaultdict

from PyQt6.QtGui import QAction, QKeySequence

from editor.main_window.handlers import (
    new_scene, open_scene, save_scene, save_scene_as,
    toggle_play_stop,
    undo, redo,
    open_global_settings, open_project_settings,
    show_build_dialog, show_about,
    on_entity_selected,
)
from editor.main_window.project import open_project_manager, open_project_browse
from editor.main_window.state import reset_layout


def setup_menu(mw):
    mb = mw.menuBar()
    file_menu = mb.addMenu("File")
    new_act = QAction("New Scene", mw)
    new_act.setShortcut(QKeySequence("Ctrl+N"))
    new_act.triggered.connect(lambda: new_scene(mw))
    file_menu.addAction(new_act)
    open_act = QAction("Open Scene...", mw)
    open_act.setShortcut(QKeySequence("Ctrl+O"))
    open_act.triggered.connect(lambda: open_scene(mw))
    file_menu.addAction(open_act)
    save_act = QAction("Save Scene", mw)
    save_act.setShortcut(QKeySequence("Ctrl+S"))
    save_act.triggered.connect(lambda: save_scene(mw))
    file_menu.addAction(save_act)
    save_as_act = QAction("Save Scene As...", mw)
    save_as_act.setShortcut(QKeySequence("Ctrl+Shift+S"))
    save_as_act.triggered.connect(lambda: save_scene_as(mw))
    file_menu.addAction(save_as_act)
    file_menu.addSeparator()
    project_mgr_act = QAction("Project Manager...", mw)
    project_mgr_act.triggered.connect(lambda: open_project_manager(mw))
    file_menu.addAction(project_mgr_act)
    open_proj_act = QAction("Open Project...", mw)
    open_proj_act.triggered.connect(lambda: open_project_browse(mw))
    file_menu.addAction(open_proj_act)
    file_menu.addSeparator()
    exit_act = QAction("Exit", mw)
    exit_act.setShortcut(QKeySequence("Alt+F4"))
    exit_act.triggered.connect(mw.close)
    file_menu.addAction(exit_act)

    edit_menu = mb.addMenu("Edit")
    mw._undo_act = QAction("Undo", mw)
    mw._undo_act.triggered.connect(lambda: undo(mw))
    mw._undo_act.setEnabled(False)
    edit_menu.addAction(mw._undo_act)
    mw._redo_act = QAction("Redo", mw)
    mw._redo_act.triggered.connect(lambda: redo(mw))
    mw._redo_act.setEnabled(False)
    edit_menu.addAction(mw._redo_act)
    edit_menu.addSeparator()
    gs_act = QAction("Global Settings...", mw)
    gs_act.triggered.connect(lambda: open_global_settings(mw))
    edit_menu.addAction(gs_act)
    ps_act = QAction("Project Settings...", mw)
    ps_act.triggered.connect(lambda: open_project_settings(mw))
    edit_menu.addAction(ps_act)

    go_menu = mb.addMenu("GameObject")
    create_empty = QAction("Create Empty", mw)
    create_empty.setShortcut(QKeySequence("Ctrl+Shift+N"))
    create_empty.triggered.connect(mw._hierarchy._create_entity)
    go_menu.addAction(create_empty)
    primitives_menu = go_menu.addMenu("3D Object")
    for name in ["cube", "sphere", "plane"]:
        act = QAction(name.capitalize(), mw)
        act.triggered.connect(lambda checked=False, n=name: mw._hierarchy._create_primitive(n))
        primitives_menu.addAction(act)
    probuilder_menu = go_menu.addMenu("ProBuilder Shape")
    from core.components.mesh_editor.primitives import get_primitive_names
    for name in get_primitive_names():
        act = QAction(name, mw)
        act.triggered.connect(lambda checked=False, n=name: mw._hierarchy._create_probuilder_primitive(n))
        probuilder_menu.addAction(act)
    lights_menu = go_menu.addMenu("Light")
    for ltype in ["sun", "directional", "point", "spot"]:
        act = QAction("Sun" if ltype == "sun" else ltype.replace("_", " ").title(), mw)
        act.triggered.connect(lambda checked=False, lt=ltype: mw._hierarchy._create_light(lt))
        lights_menu.addAction(act)
    effects_menu = go_menu.addMenu("Effects")
    for label, comp_cls in [("Sky", "Sky"), ("Clouds", "Cloud")]:
        act = QAction(label, mw)
        act.triggered.connect(lambda checked=False, n=label, cc=comp_cls: mw._hierarchy._create_from_component(n, cc, None))
        effects_menu.addAction(act)
    cam_act = QAction("Camera", mw)
    cam_act.triggered.connect(mw._hierarchy._create_camera)
    go_menu.addAction(cam_act)

    game_menu = mb.addMenu("Game")
    play_stop_act = QAction("Play/Stop", mw)
    play_stop_act.setShortcut(QKeySequence("Shift+F10"))
    play_stop_act.triggered.connect(lambda: toggle_play_stop(mw))
    game_menu.addAction(play_stop_act)

    view_menu = mb.addMenu("View")
    for dock in mw._docks:
        view_menu.addAction(dock.toggleViewAction())
    view_menu.addSeparator()
    reset_layout_act = QAction("Reset Layout", mw)
    reset_layout_act.triggered.connect(lambda: reset_layout(mw))
    view_menu.addAction(reset_layout_act)

    tools_menu = mb.addMenu("Tools")
    pm_act = QAction("Plugin Manager", mw)
    pm_act.setShortcut(QKeySequence("Ctrl+Shift+P"))
    pm_act.triggered.connect(mw._plugin_mgr.show)
    tools_menu.addAction(pm_act)
    tools_menu.addSeparator()
    vcs_act = QAction("Version Control", mw)
    vcs_act.setShortcut(QKeySequence("Ctrl+Shift+V"))
    vcs_act.triggered.connect(lambda: mw._vcs.show() or mw._vcs.raise_())
    tools_menu.addAction(vcs_act)
    tools_menu.addSeparator()
    mesh_editor_act = QAction("Mesh Editor", mw)
    mesh_editor_act.setShortcut(QKeySequence("Ctrl+Shift+M"))
    mesh_editor_act.triggered.connect(lambda: _show_mesh_editor(mw))
    tools_menu.addAction(mesh_editor_act)
    tools_menu.addSeparator()
    gui_act = QAction("GUI Editor", mw)
    gui_act.setShortcut(QKeySequence("Ctrl+Shift+G"))
    gui_act.triggered.connect(lambda: mw._gui_editor.show() or mw._gui_editor.raise_())
    tools_menu.addAction(gui_act)
    tools_menu.addSeparator()
    bs_act = QAction("Build Settings...", mw)
    bs_act.setShortcut(QKeySequence("Ctrl+Shift+U"))
    bs_act.triggered.connect(lambda: _show_build_settings(mw))
    tools_menu.addAction(bs_act)
    build_act = QAction("Build Project...", mw)
    build_act.setShortcut(QKeySequence("Ctrl+Shift+B"))
    build_act.triggered.connect(lambda: show_build_dialog(mw))
    tools_menu.addAction(build_act)

    add_plugin_menu_items(mw, mb)

    help_menu = mb.addMenu("Help")
    about_act = QAction("About Zarin Engine", mw)
    about_act.triggered.connect(lambda: show_about(mw))
    help_menu.addAction(about_act)


def _show_mesh_editor(mw):
    mw._mesh_editor.show()
    mw._mesh_editor.raise_()
    sel = getattr(mw._viewport, '_selected_entities', None)
    if sel and len(sel) > 0:
        mw._mesh_editor.set_entity(sel[0])


def _show_build_settings(mw):
    from editor.build_settings_dialog import BuildSettingsDialog
    project_root = mw._engine._project_path or os.getcwd()
    dlg = BuildSettingsDialog(project_root, mw)
    dlg.exec()


def add_plugin_menu_items(mw, mb):
    registry = mw._engine.plugin_ui_registry
    items = registry["menu_items"]
    if not items:
        return
    by_plugin: dict[str, list[dict]] = {}
    for item in items:
        by_plugin.setdefault(item.get("plugin", "Plugins"), []).append(item)
    for plugin_name in sorted(by_plugin.keys()):
        plugin_items = by_plugin[plugin_name]
        parent_menu = mb.addMenu(plugin_name)
        for item in plugin_items:
            try:
                act = QAction(item["text"], mw)
                shortcut = item.get("shortcut")
                if shortcut:
                    try:
                        act.setShortcut(QKeySequence(shortcut))
                    except Exception:
                        pass
                act.triggered.connect(item["callback"])
                parent_menu.addAction(act)
            except Exception as e:
                from core.foundation.logger import Logger
                Logger.error(f"Failed to add menu item '{item.get('text', '?')}': {e}")
