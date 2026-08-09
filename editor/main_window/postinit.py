# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QTimer

from core.foundation.logger import Logger
from editor.project_manager import _get_recent_projects
from editor.splash import SplashScreen


def _register_scene_tab(mw):
    if mw._engine.scene:
        path = mw._engine.scene.path or ""
        name = os.path.splitext(os.path.basename(path))[0] if path else (mw._engine.scene.name or "Untitled")
        mw._scene_tab_manager.add_tab(name, path=path, scene=mw._engine.scene)


def post_init(mw):
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QFont
        from core.config.config import get_global_config
        cfg = get_global_config()
        scale = cfg.get("editor.ui_scale", 75)
        base_size = cfg.get("editor.font_size", 12)
        f = QFont()
        f.setPointSizeF(base_size * scale / 100.0)
        QApplication.setFont(f)
        recent = _get_recent_projects(1)
        if recent:
            project_path = recent[0].get("path", "")
            if os.path.isdir(project_path):
                from editor.main_window.project import switch_project
                switch_project(mw, project_path)
                QTimer.singleShot(0, lambda: initial_dock_sizes(mw))
                QTimer.singleShot(100, lambda: load_renderer_config(mw))
                QTimer.singleShot(200, lambda: _register_scene_tab(mw))
                return
        SplashScreen.show_message("Creating sample scene...")
        scene = mw._engine.new_scene("SampleScene")
        from core.components import Transform, MeshFilter, MeshRenderer, Light, LightType, Camera
        from core.components.rendering.environment.sky import Sky
        from core.components.rendering.environment.clouds import Cloud
        from core.maths.math3d import Vec3
        dir_light = scene.create_entity("Directional Light")
        t = Transform()
        t.local_euler_angles = Vec3(-45, 45, 0)
        dir_light.add_component(t)
        l = Light()
        l.light_type = LightType.DIRECTIONAL
        l.intensity = 1.0
        dir_light.add_component(l)
        cube = scene.create_entity("Cube")
        ct = Transform()
        cube.add_component(ct)
        mf = MeshFilter()
        mf.mesh_name = "cube"
        cube.add_component(mf)
        cube.add_component(MeshRenderer())
        cam = scene.create_entity("Main Camera")
        cam_t = Transform()
        cam_t.local_position = Vec3(0, 2, 8)
        cam.add_component(cam_t)
        cam.add_component(Camera())
        sky_ent = scene.create_entity("Sky")
        sky_ent.add_component(Sky())
        mw._scene_tab_manager.add_tab("SampleScene", scene=scene)
        mw._hierarchy.refresh()
        QTimer.singleShot(0, lambda: initial_dock_sizes(mw))
        from core.config.config import get_global_config
        if mw._viewport.renderer:
            mw._viewport.renderer.load_config(get_global_config())
        else:
            QTimer.singleShot(100, lambda: load_renderer_config(mw))
        Logger.info("Zarin Engine Editor started. Welcome!")
    except Exception as e:
        Logger.error(f"_post_init error: {e}")


def initial_dock_sizes(mw):
    if mw._layout_restored:
        return
    mw.resizeDocks(
        [mw._hierarchy, mw._viewport_dock, mw._inspector],
        [200, 1420, 300], Qt.Orientation.Horizontal)
    mw.resizeDocks(
        [mw._hierarchy, mw._collab_panel],
        [620, 370], Qt.Orientation.Vertical)
    mw.resizeDocks(
        [mw._viewport_dock, mw._project],
        [620, 370], Qt.Orientation.Vertical)
    mw.resizeDocks(
        [mw._inspector, mw._console],
        [620, 370], Qt.Orientation.Vertical)


def load_renderer_config(mw):
    from core.config.config import get_global_config
    if mw._viewport.renderer:
        mw._viewport.renderer.load_config(get_global_config())
