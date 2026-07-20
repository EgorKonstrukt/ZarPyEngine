# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

import sys
import os
import json
import traceback
import multiprocessing
import subprocess

try:
    if __compiled__:
        sys.frozen = True
except NameError:
    pass

def _ensure_extensions():
    _dir = os.path.dirname(os.path.abspath(__file__))
    _marker = os.path.join(_dir, ".build_done")
    _pyx = os.path.join(_dir, "core", "_convex_hull.pyx")
    if os.path.exists(_marker):
        return
    if not os.path.exists(_pyx):
        return
    import glob as _glob
    if _glob.glob(os.path.join(_dir, "core", "_convex_hull*.pyd")):
        with open(_marker, "w") as f:
            f.write("ok")
        return
    print("[Zarin Engine] Building native extensions...", file=sys.stderr)
    try:
        subprocess.check_call(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=_dir,
        )
        with open(_marker, "w") as f:
            f.write("ok")
        print("[Zarin Engine] Extensions built successfully.", file=sys.stderr)
    except Exception as e:
        print(f"[Zarin Engine] Extension build failed: {e}", file=sys.stderr)

_ensure_extensions()

from core.foundation.logger import Logger

if (
    "wayland" in os.environ.get("XDG_SESSION_TYPE", "").lower()
    and "QT_QPA_PLATFORM" not in os.environ
):
    print(
        "[Zarin Engine] Wayland detected. Falling back to XCB for OpenGL compatibility.",
        file=sys.stderr,
    )
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ["QT_XCB_GL_INTEGRATION"] = "glx"
elif os.environ.get("QT_QPA_PLATFORM", "") == "wayland":
    print(
        "[Zarin Engine] Note: running on Wayland with Qt6+OpenGL may cause window visibility issues.",
        file=sys.stderr,
    )

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)
def excepthook(exc_type, exc_value, exc_traceback):
    from core.foundation.logger import Logger
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    Logger.error(f"Unhandled exception: {exc_value}\n{tb_str}")
    print(f"[Zarin Engine] Unhandled exception:\n{tb_str}", file=sys.stderr)
sys.excepthook = excepthook

from editor.bug_report import install_hooks as _install_bug_hooks
_install_bug_hooks()

def run_headless_mcp():
    os.environ["ZARIN_MCP_MODE"] = "stdio"
    import os as _os
    _orig_stdout = sys.stdout
    sys.stdout.flush()
    _orig_fd1 = _os.dup(1)
    _os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        from core.engine.engine import Engine
        engine = Engine()
        engine._mcp_mode = "stdio"
        engine.initialize()
        for fname in sorted(os.listdir("plugins")):
            fpath = os.path.join("plugins", fname)
            if fname.endswith(".py") and not fname.startswith("_"):
                engine.plugin_manager.load_from_file(fpath)
        engine.plugin_manager.load_directory("plugins/user")
        engine.new_scene("Scene")
    finally:
        sys.stdout.flush()
        _os.dup2(_orig_fd1, 1)
        _os.close(_orig_fd1)
        sys.stdout = _orig_stdout
    engine.plugin_manager.load_package("plugins/zarin_mcp")

def main():
    multiprocessing.freeze_support()
    import argparse
    parser = argparse.ArgumentParser(description="Zarin Engine")
    parser.add_argument("--mcp", action="store_true", help="Run in headless MCP stdio server mode")
    parser.add_argument("file", nargs="?", default=None, help="Scene file to open")
    args, _ = parser.parse_known_args()
    if args.mcp:
        run_headless_mcp()
        return
    if args.file and args.file.endswith(".zpes"):
        from editor.ipc_server import send_file_to_running_instance
        if send_file_to_running_instance(args.file):
            print(f"[Zarin Engine] Sent '{args.file}' to running editor instance.")
            return
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QSurfaceFormat, QIcon
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setVersion(4, 6)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    from core.config.config import get_global_config
    _cfg = get_global_config()
    fmt.setSwapInterval(0 if not _cfg.get("rendering.vsync", True) else 1)
    QSurfaceFormat.setDefaultFormat(fmt)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setApplicationName("Zarin")
    app.setOrganizationName("Zarin")
    app.setStyle("Fusion")
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zarin_icon.svg")
    app.setWindowIcon(QIcon(icon_path))
    from editor.splash import SplashScreen
    splash = SplashScreen()
    splash.set_total_steps(8)
    splash.show()
    app.processEvents()
    from core.engine.engine import Engine
    splash.advance("Initializing engine core...")
    engine = Engine()
    splash.advance("Running engine subsystems...")
    engine.initialize()
    splash.advance("Loading plugins...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    build_settings_path = os.path.join(project_root, "BuildSettings.json")
    build_plugins = []
    if os.path.exists(build_settings_path):
        try:
            with open(build_settings_path) as f:
                bs = json.load(f)
            build_plugins = bs.get("build_plugins", [])
        except Exception:
            pass
    if build_plugins:
        for name in build_plugins:
            module_name = "plugins." + name if not name.startswith("plugins.") else name
            engine.plugin_manager.load_module(module_name)
    else:
        engine.plugin_manager.load_directory("plugins")
    splash.advance("Loading user plugins...")
    if build_plugins:
        pass  # user plugins included in build_plugins list
    else:
        engine.plugin_manager.load_directory("plugins/user")
    splash.advance("Setting up collaboration...")
    from core.network.collaboration import CollaborationManager
    engine.collab_manager = CollaborationManager(engine)
    splash.advance("Building editor window...")
    from editor.main_window import EditorMainWindow
    splash.advance("Initializing editor panels...")
    window = EditorMainWindow(engine)
    engine.viewport = window._viewport
    if not window._restored_geometry_once:
        screen = app.primaryScreen()
        if screen:
            sg = screen.availableGeometry()
            window.resize(1920, 1080)
            window.move((sg.width() - 1920) // 2, (sg.height() - 1080) // 2)
    splash.advance("Ready!")
    app.processEvents()
    from editor.ipc_server import IpcServer
    def _open_file_from_ipc(path: str):
        from editor.main_window.handlers import open_scene_by_path
        def _do():
            open_scene_by_path(window, path)
        QTimer.singleShot(0, _do)
    ipc_server = IpcServer(_open_file_from_ipc)
    ipc_server.try_bind()
    window.destroyed.connect(ipc_server.stop)

    if args.file and os.path.exists(args.file):
        if args.file.endswith(".zpes"):
            from editor.main_window.handlers import open_scene_by_path
            open_scene_by_path(window, args.file)
        elif args.file.endswith(".zterr"):
            if hasattr(window, '_terrain_editor') and window._terrain_editor is not None:
                window._terrain_editor.load_graph(args.file)
                window._terrain_editor.show()
                window._terrain_editor.raise_()

    window.showNormal()
    SplashScreen.hide_splash()
    window.raise_()
    window.activateWindow()
    QTimer.singleShot(200, window.raise_)
    QTimer.singleShot(500, window.activateWindow)
    sys.exit(app.exec())
if __name__ == "__main__":
    main()
