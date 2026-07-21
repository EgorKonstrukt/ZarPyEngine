# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os
from collections import defaultdict

from PyQt6.QtGui import QAction, QKeySequence, QFont
from PyQt6.QtWidgets import QMenu, QMessageBox, QInputDialog, QDialog, QVBoxLayout, QPlainTextEdit

from editor.panels.vcs_panel import _Git, _DiffView, _find_git

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
    mesh_editor_act = QAction("Mesh Editor", mw)
    mesh_editor_act.setShortcut(QKeySequence("Ctrl+Shift+M"))
    mesh_editor_act.triggered.connect(lambda: _show_mesh_editor(mw))
    tools_menu.addAction(mesh_editor_act)
    terrain_editor_act = QAction("Terrain Editor", mw)
    terrain_editor_act.setShortcut(QKeySequence("Ctrl+Shift+T"))
    terrain_editor_act.triggered.connect(lambda: _show_terrain_editor(mw))
    tools_menu.addAction(terrain_editor_act)
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

    vcs_menu = mb.addMenu("VCS")
    _setup_vcs_menu(mw, vcs_menu)

    add_plugin_menu_items(mw, mb)

    help_menu = mb.addMenu("Help")
    bug_act = QAction("Report Bug...", mw)
    bug_act.triggered.connect(lambda: _show_bug_report(mw))
    help_menu.addAction(bug_act)
    help_menu.addSeparator()
    about_act = QAction("About Zarin Engine", mw)
    about_act.triggered.connect(lambda: show_about(mw))
    help_menu.addAction(about_act)


def _setup_vcs_menu(mw, vcs_menu):
    _act_commit = QAction("Commit...", mw)
    _act_commit.setShortcut(QKeySequence("Ctrl+Shift+C"))
    _act_commit.triggered.connect(lambda: _vcs_commit(mw))
    vcs_menu.addAction(_act_commit)

    _act_diff = QAction("Diff...", mw)
    _act_diff.setShortcut(QKeySequence("Ctrl+Shift+D"))
    _act_diff.triggered.connect(lambda: _vcs_diff(mw))
    vcs_menu.addAction(_act_diff)

    _act_history = QAction("Show History", mw)
    _act_history.setShortcut(QKeySequence("Ctrl+Shift+H"))
    _act_history.triggered.connect(lambda: _vcs_log(mw))
    vcs_menu.addAction(_act_history)

    _act_vcs_panel = QAction("Version Control Panel", mw)
    _act_vcs_panel.setShortcut(QKeySequence("Ctrl+Shift+V"))
    _act_vcs_panel.triggered.connect(lambda: mw._vcs.show() or mw._vcs.raise_())
    vcs_menu.addAction(_act_vcs_panel)

    vcs_menu.addSeparator()
    push_menu = vcs_menu.addMenu("Push")
    _act_push = QAction("Push to origin", mw)
    _act_push.triggered.connect(lambda: _vcs_push(mw, "origin"))
    push_menu.addAction(_act_push)
    _act_push_all = QAction("Push to all remotes", mw)
    _act_push_all.triggered.connect(lambda: _vcs_push_all(mw))
    push_menu.addAction(_act_push_all)

    pull_menu = vcs_menu.addMenu("Pull")
    _act_pull = QAction("Pull from origin", mw)
    _act_pull.triggered.connect(lambda: _vcs_pull(mw, "origin"))
    pull_menu.addAction(_act_pull)

    _act_fetch = QAction("Fetch", mw)
    _act_fetch.triggered.connect(lambda: _vcs_fetch(mw))
    vcs_menu.addAction(_act_fetch)

    vcs_menu.addSeparator()
    branch_menu = vcs_menu.addMenu("Branch")
    _act_new_branch = QAction("New Branch...", mw)
    _act_new_branch.triggered.connect(lambda: _vcs_new_branch(mw))
    branch_menu.addAction(_act_new_branch)
    _act_switch = QAction("Switch Branch...", mw)
    _act_switch.triggered.connect(lambda: _vcs_switch_branch(mw))
    branch_menu.addAction(_act_switch)
    branch_menu.addSeparator()
    _act_show_branches = QAction("Show All Branches", mw)
    _act_show_branches.triggered.connect(lambda: _vcs_show_branches(mw))
    branch_menu.addAction(_act_show_branches)

    stash_menu = vcs_menu.addMenu("Stash")
    _act_stash_push = QAction("Stash Changes...", mw)
    _act_stash_push.triggered.connect(lambda: _vcs_stash_push(mw))
    stash_menu.addAction(_act_stash_push)
    _act_stash_pop = QAction("Pop Stash...", mw)
    _act_stash_pop.triggered.connect(lambda: _vcs_stash_pop(mw))
    stash_menu.addAction(_act_stash_pop)

    vcs_menu.addSeparator()
    _act_create_repo = QAction("Create Repository...", mw)
    _act_create_repo.triggered.connect(lambda: _vcs_create_repo(mw))
    vcs_menu.addAction(_act_create_repo)


def _vcs_git(mw) -> _Git | None:
    if hasattr(mw, "_vcs") and hasattr(mw._vcs, "_git"):
        return mw._vcs._git
    g = _Git()
    eng = getattr(mw, "_engine", None)
    if eng:
        project_path = getattr(eng, "_project_path", "")
        if project_path:
            g.detect(project_path)
    return g


def _vcs_ensure_repo(mw) -> bool:
    git = _vcs_git(mw)
    if git and git.available and git.repo_root:
        return True
    QMessageBox.information(mw, "No Repository", "No git repository detected.")
    return False


def _vcs_commit(mw):
    if not _vcs_ensure_repo(mw):
        return
    msg, ok = QInputDialog.getText(mw, "Commit", "Commit message:")
    if not ok or not msg.strip():
        return
    git = _vcs_git(mw)
    rc, out, err = git.run_sync("commit", "-m", msg.strip())
    if rc == 0:
        QMessageBox.information(mw, "Committed", f"Committed:\n{msg.strip()}")
        if hasattr(mw, "_vcs"):
            mw._vcs._refresh_all()
    else:
        QMessageBox.critical(mw, "Commit Failed", f"Error:\n{err or out}")


def _vcs_diff(mw):
    if not _vcs_ensure_repo(mw):
        return
    git = _vcs_git(mw)
    rc, out, _ = git.run_sync("diff", "--no-color")
    if not out.strip():
        rc2, out2, _ = git.run_sync("diff", "--cached", "--no-color")
        out = out2
    if not out.strip():
        out = "No changes against HEAD"
    dlg = QDialog(mw)
    dlg.setWindowTitle("Working Tree Diff")
    dlg.setMinimumSize(700, 500)
    layout = QVBoxLayout(dlg)
    dv = _DiffView()
    dv.show_diff(out)
    layout.addWidget(dv)
    dlg.exec()


def _vcs_log(mw):
    if not _vcs_ensure_repo(mw):
        return
    git = _vcs_git(mw)
    rc, out, _ = git.run_sync("log", "--oneline", "--graph", "--decorate", "--all", "-50")
    if rc != 0 or not out.strip():
        out = "No commits yet."
    dlg = QDialog(mw)
    dlg.setWindowTitle("Git Log")
    dlg.setMinimumSize(700, 500)
    layout = QVBoxLayout(dlg)
    te = QPlainTextEdit()
    te.setReadOnly(True)
    te.setFont(QFont("Courier New", 10))
    te.setPlainText(out)
    layout.addWidget(te)
    dlg.exec()


def _vcs_push(mw, remote: str):
    if not _vcs_ensure_repo(mw):
        return
    git = _vcs_git(mw)
    proc = git.push(remote)
    proc.finished.connect(lambda rc, out, err: _vcs_async_done(mw, rc, out, err, f"Push to {remote}"))
    if hasattr(mw, "_vcs"):
        mw._vcs._progress.show()
        mw._vcs._progress.setRange(0, 0)


def _vcs_push_all(mw):
    if not _vcs_ensure_repo(mw):
        return
    git = _vcs_git(mw)
    for remote in git.remotes():
        proc = git.push(remote["name"])
        proc.finished.connect(lambda rc, out, err, r=remote["name"]: _vcs_async_done(mw, rc, out, err, f"Push to {r}"))


def _vcs_pull(mw, remote: str):
    if not _vcs_ensure_repo(mw):
        return
    git = _vcs_git(mw)
    proc = git.pull(remote)
    proc.finished.connect(lambda rc, out, err: _vcs_async_done(mw, rc, out, err, f"Pull from {remote}"))


def _vcs_fetch(mw):
    if not _vcs_ensure_repo(mw):
        return
    git = _vcs_git(mw)
    proc = git.fetch()
    proc.finished.connect(lambda rc, out, err: _vcs_async_done(mw, rc, out, err, "Fetch"))


def _vcs_async_done(mw, rc, out, err, label):
    if hasattr(mw, "_vcs"):
        mw._vcs._progress.hide()
    if rc == 0:
        if hasattr(mw, "_vcs"):
            mw._vcs._commit_panel.set_status(f"{label} completed")
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(500, mw._vcs._refresh_all)
        QMessageBox.information(mw, label, f"{label} completed successfully.")
    else:
        QMessageBox.critical(mw, f"{label} Failed", f"{err or out}")


def _vcs_new_branch(mw):
    if not _vcs_ensure_repo(mw):
        return
    name, ok = QInputDialog.getText(mw, "New Branch", "Branch name:")
    if not ok or not name.strip():
        return
    git = _vcs_git(mw)
    rc, out, err = git.create_branch(name.strip())
    if rc == 0:
        QMessageBox.information(mw, "Branch Created", f"Branch '{name.strip()}' created.")
        if hasattr(mw, "_vcs"):
            mw._vcs._refresh_branches()
    else:
        QMessageBox.critical(mw, "Branch Failed", f"Error:\n{err or out}")


def _vcs_switch_branch(mw):
    if not _vcs_ensure_repo(mw):
        return
    git = _vcs_git(mw)
    rc, out, _ = git.run_sync("branch", "-a")
    if rc != 0:
        return
    branches = [b.strip() for b in out.strip().split("\n") if b.strip()]
    lines = []
    for b in branches:
        name = b[2:] if b.startswith("* ") else b
        lines.append(name)
    current = git.current_branch()
    dlg = QInputDialog(mw)
    dlg.setWindowTitle("Switch Branch")
    dlg.setLabelText(f"Current: {current}\nSelect branch:")
    dlg.setComboBoxItems(lines)
    dlg.setOption(QInputDialog.InputDialogOption.UseListViewForComboBoxItems)
    if dlg.exec() != QInputDialog.DialogCode.Accepted:
        return
    target = dlg.textValue()
    if not target or target == current:
        return
    reply = QMessageBox.question(mw, "Switch Branch", f"Switch to '{target}'?",
                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    if reply != QMessageBox.StandardButton.Yes:
        return
    rc, out, err = git.switch_branch(target)
    if rc == 0:
        QMessageBox.information(mw, "Switched", f"Switched to '{target}'")
        if hasattr(mw, "_vcs"):
            mw._vcs._refresh_all()
    else:
        QMessageBox.critical(mw, "Switch Failed", f"Error:\n{err or out}")


def _vcs_show_branches(mw):
    if not _vcs_ensure_repo(mw):
        return
    git = _vcs_git(mw)
    rc, out, _ = git.run_sync("branch", "-a")
    if rc != 0:
        out = "No branches."
    dlg = QDialog(mw)
    dlg.setWindowTitle("Branches")
    dlg.setMinimumSize(500, 400)
    layout = QVBoxLayout(dlg)
    te = QPlainTextEdit()
    te.setReadOnly(True)
    te.setFont(QFont("Courier New", 10))
    te.setPlainText(out)
    layout.addWidget(te)
    dlg.exec()


def _vcs_stash_push(mw):
    if not _vcs_ensure_repo(mw):
        return
    msg, ok = QInputDialog.getText(mw, "Stash Changes", "Stash message (optional):")
    if not ok:
        return
    git = _vcs_git(mw)
    rc, out, err = git.stash_push(msg)
    if rc == 0:
        QMessageBox.information(mw, "Stashed", "Changes stashed.")
        if hasattr(mw, "_vcs"):
            mw._vcs._refresh_all()
    else:
        QMessageBox.critical(mw, "Stash Failed", f"Error:\n{err or out}")


def _vcs_stash_pop(mw):
    if not _vcs_ensure_repo(mw):
        return
    git = _vcs_git(mw)
    stashes = git.stash_list()
    if not stashes:
        QMessageBox.information(mw, "No Stashes", "No stashes found.")
        return
    ref = stashes[0]["ref"]
    rc, out, err = git.stash_pop(ref)
    if rc == 0:
        QMessageBox.information(mw, "Popped", f"Popped {ref}")
        if hasattr(mw, "_vcs"):
            mw._vcs._refresh_all()
    else:
        QMessageBox.critical(mw, "Pop Failed", f"Error:\n{err or out}")


def _vcs_create_repo(mw):
    if hasattr(mw, "_vcs"):
        mw._vcs._create_repo()


def _show_mesh_editor(mw):
    mw._mesh_editor.show()
    mw._mesh_editor.raise_()
    sel = getattr(mw._viewport, '_selected_entities', None)
    if sel and len(sel) > 0:
        mw._mesh_editor.set_entity(sel[0])


def _show_terrain_editor(mw):
    mw._terrain_editor.show()
    mw._terrain_editor.raise_()
    sel = getattr(mw._viewport, '_selected_entities', None)
    if sel and len(sel) > 0:
        mw._terrain_editor.set_entity(sel[0])


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


def _show_bug_report(mw):
    from editor.bug_report import show_bug_report_dialog
    show_bug_report_dialog(mw)
