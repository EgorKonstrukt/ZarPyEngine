# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

import sys
import os
import traceback
import json
import multiprocessing
from datetime import datetime

from PyQt6.QtGui import QSurfaceFormat

try:
    if __compiled__:
        sys.frozen = True
except NameError:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SCENE = "SampleScene.zpes"

_LOG_FILE = None


def _log(msg: str):
    global _LOG_FILE
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line)
    if _LOG_FILE is None:
        try:
            _LOG_FILE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_diag.txt"), "w", encoding="utf-8")
        except Exception:
            pass
    if _LOG_FILE:
        _LOG_FILE.write(line + "\n")
        _LOG_FILE.flush()


def excepthook(exc_type, exc_value, exc_traceback):
    from core.logger import Logger
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    Logger.error(f"Unhandled exception: {exc_value}\n{tb_str}")
    _log(f"Unhandled exception: {exc_value}")


sys.excepthook = excepthook


def _resolve_startup_scene(project_root: str) -> str:
    _log(f"project_root: {project_root}")

    # 1. CLI argument
    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        _log(f"CLI arg: {arg_path}")
        if os.path.exists(arg_path):
            return arg_path

    # 2. BuildSettings.json вЂ” first scene in list
    build_settings_path = os.path.join(project_root, "BuildSettings.json")
    _log(f"Checking BuildSettings at: {build_settings_path}  exists={os.path.exists(build_settings_path)}")
    if os.path.exists(build_settings_path):
        try:
            with open(build_settings_path, "r") as f:
                bs = json.load(f)
            scenes = bs.get("scenes", [])
            _log(f"BuildSettings scenes: {scenes}")
            if scenes:
                startup = scenes[0]
                _log(f"Raw startup scene: {repr(startup)}")
                for prefix in ("scenes/", "scenes\\"):
                    if startup.startswith(prefix):
                        startup = startup[len(prefix):]
                        break
                if not os.path.isabs(startup):
                    startup = os.path.join(project_root, "scenes", startup)
                _log(f"Resolved scene path: {startup}  exists={os.path.exists(startup)}")
                if os.path.exists(startup):
                    return startup
        except Exception as e:
            _log(f"Error reading BuildSettings: {e}")

    # 3. ProjectSettings.json вЂ” legacy default_scene
    settings_path = os.path.join(project_root, "ProjectSettings.json")
    _log(f"Checking ProjectSettings at: {settings_path}  exists={os.path.exists(settings_path)}")
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r") as f:
                ps = json.load(f)
            sp = ps.get("project", {}).get("default_scene", "")
            _log(f"ProjectSettings default_scene: {sp}")
            if sp:
                full = os.path.join(project_root, sp) if not os.path.isabs(sp) else sp
                _log(f"Resolved default_scene: {full}  exists={os.path.exists(full)}")
                if os.path.exists(full):
                    return full
        except Exception as e:
            _log(f"Error reading ProjectSettings: {e}")

    # 4. Fallback
    fallback = os.path.join(project_root, "scenes", DEFAULT_SCENE)
    _log(f"Using fallback: {fallback}  exists={os.path.exists(fallback)}")
    if os.path.exists(fallback):
        return fallback
    return DEFAULT_SCENE


def main():
    multiprocessing.freeze_support()
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    QApplication.setAttribute(Qt.ApplicationAttribute(79))
    app = QApplication(sys.argv)
    app.setApplicationName("Zarin Player")
    app.setOrganizationName("Zarin")
    app.setStyle("Fusion")

    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setVersion(4, 6)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    from core.config import get_global_config
    _cfg = get_global_config()
    fmt.setSwapInterval(0 if not _cfg.get("rendering.vsync", True) else 1)
    QSurfaceFormat.setDefaultFormat(fmt)

    from core.engine import Engine
    from core.game_viewport import GameViewport
    from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
    from PyQt6.QtCore import QTimer

    project_root = os.path.dirname(os.path.abspath(__file__))
    _log(f"__file__ = {__file__}")
    _log(f"sys.executable = {sys.executable}")
    _log(f"project_root = {project_root}")
    _log(f"CWD = {os.getcwd()}")
    _log(f"args = {sys.argv}")

    engine = Engine()
    engine._project_path = project_root
    engine.initialize()
    from core.audio_system import AudioSystem
    audio = AudioSystem.instance()
    if audio:
        audio.apply_project_audio_config()
    # Load plugins listed in BuildSettings.json
    build_settings_path = os.path.join(project_root, "BuildSettings.json")
    build_plugins = []
    if os.path.exists(build_settings_path):
        try:
            with open(build_settings_path) as f:
                bs = json.load(f)
            build_plugins = bs.get("build_plugins", [])
        except Exception as e:
            _log(f"Error reading BuildSettings plugins: {e}")
    _log(f"Build plugins: {build_plugins}")
    if build_plugins:
        for name in build_plugins:
            module_name = "plugins." + name if not name.startswith("plugins.") else name
            engine.plugin_manager.load_module(module_name)
    else:
        import importlib, pkgutil
        try:
            pkg = importlib.import_module("plugins")
            for importer, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
                if modname.startswith("_"):
                    continue
                engine.plugin_manager.load_module("plugins." + modname)
                if ispkg:
                    subpkg = importlib.import_module("plugins." + modname)
                    for _, subname, _ in pkgutil.iter_modules(subpkg.__path__):
                        if not subname.startswith("_"):
                            engine.plugin_manager.load_module(f"plugins.{modname}.{subname}")
        except Exception:
            pass
    _log(f"Registered plugins: {list(engine.plugin_manager._plugins.keys())}")
    physics = engine.plugin_manager.get("PhysicsPlugin")
    if physics:
        _log(f"PhysicsPlugin: solver={type(physics._solver).__name__ if physics._solver else None}, physics_scene={physics._physics_scene is not None}, mode={physics._simulation_mode}, enabled={physics._enabled}")
    else:
        _log("PhysicsPlugin: NOT FOUND")

    scene_path = _resolve_startup_scene(project_root)
    _log(f"Final scene_path: {scene_path}")

    window = QMainWindow()
    window.setWindowTitle("Zarin Player")
    container = QWidget()
    window.setCentralWidget(container)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    viewport = GameViewport(engine, window)
    engine.viewport = viewport
    layout.addWidget(viewport)

    screen = app.primaryScreen()
    if screen:
        window.resize(screen.size() * 0.8)
    else:
        window.resize(1280, 720)
    window.show()

    _log(f"Loading scene: {scene_path}")
    if os.path.exists(scene_path):
        scene = engine.load_scene(scene_path)
        _log(f"engine.load_scene returned: {scene}")
        if scene:
            _log(f"Scene loaded OK: {scene_path}")
            def _on_start_play():
                _log(f"Timer fired: starting play, scene={engine.scene is not None}")
                engine.start_play()
                _log(f"start_play done")
                physics = engine.plugin_manager.get("PhysicsPlugin")
                if physics:
                    _log(f"After start_play: physics_scene={physics._physics_scene is not None}, bodies_loaded={len(physics._physics_scene._entity_to_body) if physics._physics_scene else 0}")
            QTimer.singleShot(100, _on_start_play)
        else:
            _log(f"Scene FAILED to load (returned None)")
    else:
        _log(f"Startup scene NOT FOUND: {scene_path}")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
