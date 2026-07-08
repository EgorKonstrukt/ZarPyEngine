# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import json
import os

from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import QTimer

from editor.project_manager import ProjectManagerDialog, _add_recent_project
from editor.splash import SplashScreen


def switch_project(mw, project_path: str):
    SplashScreen.show_message(f"Opening project: {os.path.basename(project_path)}...")
    QTimer.singleShot(0, lambda: _do_switch_project(mw, project_path))


def _do_switch_project(mw, project_path: str):
    project_path = os.path.abspath(project_path)
    assets_dir = os.path.join(project_path, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(os.path.join(project_path, "scenes"), exist_ok=True)
    settings_path = os.path.join(project_path, "ProjectSettings.json")
    name = os.path.basename(project_path)
    if os.path.exists(settings_path):
        try:
            with open(settings_path) as f:
                data = json.load(f)
            name = data.get("project", {}).get("name", name)
        except Exception:
            pass
    _add_recent_project(name, project_path)
    mw._engine._project_path = project_path
    mw._project.set_project_root(assets_dir)
    from core.audio_system import AudioSystem
    audio = AudioSystem.instance()
    if audio:
        audio.apply_project_audio_config()

    # Reload BuildSettings for this project
    from core.build_settings import BuildSettings
    bs = BuildSettings.instance() or BuildSettings()
    bs.load(os.path.join(project_path, "BuildSettings.json"))

    if os.path.isdir(os.path.join(project_path, "scenes")):
        scenes_dir = os.path.join(project_path, "scenes")
    else:
        scenes_dir = project_path
    scene_name = os.path.join(scenes_dir, f"{name}.zpes")
    if os.path.exists(scene_name):
        SplashScreen.show_message(f"Loading scene: {name}.zpes...")
        mw._engine.load_scene(scene_name)
        mw._hierarchy.refresh()
    else:
        SplashScreen.show_message(f"Creating new scene: {name}...")
        mw._engine.new_scene(name)
        mw._hierarchy.refresh()
        from editor.main_window.handlers import on_entity_selected
        on_entity_selected(mw, None)
    mw._engine._project_settings_path = settings_path
    mw._render_toolbar.load_state()
    SplashScreen.hide_splash()


def open_project_manager(mw):
    dlg = ProjectManagerDialog(mw)
    dlg.project_selected.connect(lambda path: switch_project(mw, path))
    dlg.exec()


def open_project_browse(mw):
    path = QFileDialog.getExistingDirectory(mw, "Open Project")
    if path:
        switch_project(mw, path)
