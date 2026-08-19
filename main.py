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

def _apply_python_optimization():
    if getattr(sys, "frozen", False):
        return
    if os.environ.get("_ZARIN_OPT_REEXEC") == "1":
        return
    _eng = {}
    try:
        from pathlib import Path
        _settings = Path.home() / ".zarin" / "settings.json"
        if _settings.exists():
            with open(_settings) as _f:
                _eng = json.load(_f).get("engine", {})
    except Exception:
        pass
    jit_wanted = bool(_eng.get("python_jit", False))
    opt_wanted = int(_eng.get("python_optimize", 0))
    unbuf_wanted = bool(_eng.get("python_unbuffered", False))
    nobc_wanted = bool(_eng.get("python_no_bytecode", False))
    jit_current = bool(getattr(sys, "_jit_enabled", False))
    opt_current = int(getattr(sys.flags, "optimize", 0))
    changed = False
    new_env = os.environ.copy()
    new_env["_ZARIN_OPT_REEXEC"] = "1"
    if jit_wanted and not jit_current:
        new_env["PYTHON_JIT"] = "1"
        changed = True
    elif not jit_wanted and jit_current:
        new_env["PYTHON_JIT"] = "0"
        changed = True
    if unbuf_wanted and "PYTHONUNBUFFERED" not in os.environ:
        new_env["PYTHONUNBUFFERED"] = "1"
        changed = True
    elif not unbuf_wanted and "PYTHONUNBUFFERED" in os.environ:
        new_env.pop("PYTHONUNBUFFERED", None)
        changed = True
    if nobc_wanted and "PYTHONDONTWRITEBYTECODE" not in os.environ:
        new_env["PYTHONDONTWRITEBYTECODE"] = "1"
        changed = True
    elif not nobc_wanted and "PYTHONDONTWRITEBYTECODE" in os.environ:
        new_env.pop("PYTHONDONTWRITEBYTECODE", None)
        changed = True
    if opt_wanted != opt_current:
        changed = True
    if not changed:
        return
    new_args = [sys.executable]
    if opt_wanted >= 2:
        new_args.append("-OO")
    elif opt_wanted >= 1:
        new_args.append("-O")
    new_args.extend(sys.argv)
    print(f"[Zarin Engine] Relaunching with optimizations (jit={jit_wanted}, opt={opt_wanted}, unbuffered={unbuf_wanted}, no_bytecode={nobc_wanted})...", file=sys.stderr)
    _proc = subprocess.Popen(new_args, env=new_env)
    _proc.wait()
    sys.exit(_proc.returncode)

_apply_python_optimization()

def _check_extensions_missing():
    if getattr(sys, "frozen", False):
        return False
    _dir = os.path.dirname(os.path.abspath(__file__))
    _pyd_dir = os.path.join(_dir, "core")
    _pyx_dir = os.path.join(_dir, "core", "pyx")
    _mod_names = [
        "_convex_hull", "_bvh_build", "_raytracing_data", "_math_vec",
        "_culling", "_transform_batch", "_render_utils", "_physics_utils",
        "_core_batch", "_types", "_ecs_batch", "_render_batch",
        "_octree_batch", "_constraint_batch", "_curve_batch",
        "_constraint_update", "_physics_sync", "_mesh_import",
        "_skinning", "_audio_dsp_cy", "_raycast", "_shadow_batch",
        "math_helpers",
    ]
    import importlib.machinery
    _suffixes = importlib.machinery.EXTENSION_SUFFIXES
    def _ext_path(_m):
        for _entry in os.listdir(_pyd_dir):
            for _s in _suffixes:
                if _entry == _m + _s:
                    return os.path.join(_pyd_dir, _entry)
        return None
    for _m in _mod_names:
        _p = _ext_path(_m)
        if _p is None:
            return True
        _src = os.path.join(_pyx_dir, _m + ".pyx")
        if os.path.exists(_src) and os.path.getmtime(_src) > os.path.getmtime(_p):
            return True
    return False

def _build_extensions_with_splash(splash):
    import re
    from PyQt6.QtWidgets import QApplication
    _dir = os.path.dirname(os.path.abspath(__file__))
    print("[Zarin Engine] Native extensions missing or outdated. Building...", file=sys.stderr)
    splash.set_progress(0, "Building native extensions...")
    QApplication.processEvents()
    try:
        proc = subprocess.Popen(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        total_exts = 23
        built = 0
        for line in proc.stdout:
            stripped = line.rstrip()
            if not stripped:
                continue
            sys.stdout.write(stripped + "\n")
            sys.stdout.flush()
            low = stripped.lower()
            if "compiling" in low and ".pyx" in low:
                m = re.search(r"(\S+\.pyx)", stripped)
                name = os.path.basename(m.group(1)) if m else "module"
                built += 1
                pct = int(built * 80 / total_exts)
                splash.set_progress(pct, f"Compiling {name} ({built}/{total_exts})...")
                QApplication.processEvents()
            elif "building" in low and "extension" in low:
                m = re.search(r"building '([^']+)'", stripped)
                name = m.group(1).split(".")[-1] if m else ""
                if name:
                    splash.set_progress(min(95, int(built * 80 / total_exts)),
                                        f"Building {name}...")
                    QApplication.processEvents()
        proc.wait()
        if proc.returncode == 0:
            splash.set_progress(100, "Extensions built successfully.")
            QApplication.processEvents()
            print("[Zarin Engine] Extensions built successfully.", file=sys.stderr)
        else:
            splash.set_progress(100, "Extension build failed.")
            QApplication.processEvents()
            print(f"[Zarin Engine] Extension build failed (exit code {proc.returncode}).", file=sys.stderr)
    except Exception as e:
        print(f"[Zarin Engine] Extension build failed: {e}", file=sys.stderr)

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

def main():
    multiprocessing.freeze_support()
    import argparse
    parser = argparse.ArgumentParser(description="Zarin Engine")
    parser.add_argument("file", nargs="?", default=None, help="Scene file to open")
    args, _ = parser.parse_known_args()
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
    if _check_extensions_missing():
        _build_extensions_with_splash(splash)
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
    if os.path.isdir("plugins"):
        engine.plugin_manager.load_directory("plugins")
    elif build_plugins:
        for name in build_plugins:
            module_name = "plugins." + name if not name.startswith("plugins.") else name
            engine.plugin_manager.load_module(module_name)
    splash.advance("Loading user plugins...")
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
