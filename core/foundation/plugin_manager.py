# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import importlib
import importlib.util
import ctypes
import json
import platform
import sys
import os
import shutil
import zipfile
import tempfile
from typing import Any, Callable, Optional, TYPE_CHECKING
from core.foundation.logger import Logger
from core.ecs.pool import plugin as _get_plugin_pool
if TYPE_CHECKING:
    from core.engine.engine import Engine


def _zarin_user_dir(name: str = "") -> str:
    base = os.path.join(os.path.expanduser("~"), ".zarin")
    if name:
        base = os.path.join(base, name)
    os.makedirs(base, exist_ok=True)
    return base


def current_architecture() -> str:
    sys_name = sys.platform
    machine = platform.machine().lower()
    if sys_name == "win32":
        sys_name = "win"
    elif sys_name == "darwin":
        sys_name = "macos"
    if machine in ("amd64", "x86_64", "x64"):
        machine = "x86_64"
    elif machine in ("aarch64", "arm64"):
        machine = "arm64"
    elif machine in ("i386", "i686", "x86"):
        machine = "x86"
    return f"{sys_name}_{machine}"


def _platform_matches(manifest_arch: str) -> bool:
    if not manifest_arch or manifest_arch == "any":
        return True
    current = current_architecture()
    if manifest_arch == current:
        return True
    return False


def _plugin_cache_root() -> str:
    return _zarin_user_dir("plugins")


def _resolve_zplugin(name: str, version: str) -> str:
    safe = name.replace("/", "_").replace("\\", "_").strip()
    return os.path.join(_plugin_cache_root(), f"{safe}-{version}")


def _project_root():
    cur = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.basename(cur) == "core":
            return os.path.dirname(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PluginBase:
    NAME: str = "UnnamedPlugin"
    VERSION: str = "0.0.1"
    DESCRIPTION: str = ""
    SYSTEM: bool = False

    def __init__(self):
        self._engine: Optional[Engine] = None
        self._enabled: bool = True
        self._config: dict = {}
        self._config_path: Optional[str] = None
        self._docks: list[dict] = []
        self._toolbar_actions: list[dict] = []
        self._menu_items: list[dict] = []
        self._file_openers: list[dict] = []
        self._components: list[type] = []
        self._patches = None
        self._bundled_library_dir: Optional[str] = None

    def initialize(self, engine: Engine):
        self._engine = engine
        self._load_config()

    def shutdown(self):
        self._save_config()
        self.unpatch_all()

    @property
    def patches(self):
        if self._patches is None:
            from core.foundation.patcher import PatchTracker
            self._patches = PatchTracker()
        return self._patches

    def unpatch_all(self):
        if self._patches is not None:
            try:
                self._patches.restore_all()
            except Exception as e:
                Logger.warning(f"[{self.NAME}] Failed to restore patches: {e}")
            self._patches = None

    def patch_component(self, component_name: str, method_patches: Optional[dict[str, Callable]] = None,
                        extra_inspector_fields: Optional[list] = None,
                        filter_extensions: Optional[dict[str, str]] = None,
                        wraps: Optional[dict[str, dict]] = None):
        return self.patches.patch_component(
            component_name,
            method_patches=method_patches,
            extra_inspector_fields=extra_inspector_fields,
            filter_extensions=filter_extensions,
            wraps=wraps,
        )

    def add_inspector_field(self, component_name: str, field):
        return self.patches.add_inspector_field(component_name, field)

    def extend_file_filter(self, component_name: str, field_name: str, extra_filter: str):
        return self.patches.extend_file_filter(component_name, field_name, extra_filter)

    def patch_function(self, module, func_name: str, wrapper: Callable):
        self.patches.patch_function(module, func_name, wrapper)

    @property
    def bundled_library_dir(self) -> Optional[str]:
        return self._bundled_library_dir

    @property
    def resource_dir(self) -> Optional[str]:
        path = getattr(self, "_native_plugin_path", None)
        if path:
            base = os.path.splitext(path)[0]
            res = base + "_resources"
            if os.path.isdir(res):
                return res
        return None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, v: bool):
        self._enabled = v

    def step(self, dt: float):
        pass

    def pre_step(self, dt: float):
        pass

    def on_viewport_ready(self, viewport):
        pass

    def on_scene_loaded(self, scene):
        pass

    def on_scene_unloaded(self, scene):
        pass

    def on_project_opened(self):
        pass

    def on_play_start(self):
        pass

    def on_play_stop(self):
        pass

    @property
    def engine(self) -> Optional[Engine]:
        return self._engine

    # ---- Config API ----

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any):
        self._config[key] = value
        self._save_config()

    def _config_dir(self) -> str:
        base = os.path.join(_project_root(), "config", "plugins")
        os.makedirs(base, exist_ok=True)
        return base

    def _load_config(self):
        name = self.NAME.replace("/", "_").replace("\\", "_")
        self._config_path = os.path.join(self._config_dir(), f"{name}.json")
        try:
            if os.path.isfile(self._config_path):
                with open(self._config_path, "r") as f:
                    self._config = json.load(f)
        except Exception as e:
            Logger.warning(f"[{self.NAME}] Failed to load config: {e}")
            self._config = {}

    def _save_config(self):
        if not self._config_path:
            return
        try:
            with open(self._config_path, "w") as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            Logger.warning(f"[{self.NAME}] Failed to save config: {e}")

    # ---- Dock Registration ----

    def register_dock(self, title: str, widget_factory: Callable[[], Any],
                      area: str = "left", tab_group: Optional[str] = None):
        self._docks.append({
            "title": title,
            "widget_factory": widget_factory,
            "area": area,
            "tab_group": tab_group,
        })

    # ---- Toolbar Registration ----

    def add_toolbar_button(self, text: str, callback: Callable,
                           icon: Optional[str] = None, tooltip: str = ""):
        self._toolbar_actions.append({
            "text": text,
            "callback": callback,
            "icon": icon,
            "tooltip": tooltip or text,
        })

    # ---- Menu Registration ----

    def add_menu_item(self, menu_name: str, text: str, callback: Callable,
                      shortcut: Optional[str] = None):
        self._menu_items.append({
            "menu": menu_name,
            "text": text,
            "callback": callback,
            "shortcut": shortcut,
        })

    # ---- File Opener Registration ----

    def register_file_opener(self, extensions, handler: Callable[[str], Any],
                             label: str = ""):
        if isinstance(extensions, str):
            extensions = [extensions]
        normed = []
        for ext in extensions:
            e = str(ext).strip().lower()
            if not e:
                continue
            if not e.startswith("."):
                e = "." + e
            if e not in normed:
                normed.append(e)
        if not normed or not callable(handler):
            Logger.warning(f"[{self.NAME}] register_file_opener needs extensions and a handler")
            return
        self._file_openers.append({
            "extensions": normed,
            "handler": handler,
            "label": label or self.NAME,
        })

    # ---- Component Registration ----

    def register_component(self, comp_cls: type):
        if not isinstance(comp_cls, type):
            Logger.warning(f"[{self.NAME}] register_component expects a class, got {type(comp_cls).__name__}")
            return
        from core.ecs.ecs import Component, ComponentRegistry
        if not issubclass(comp_cls, Component):
            Logger.warning(f"[{self.NAME}] register_component: {comp_cls.__name__} must inherit from Component")
            return
        ComponentRegistry.register(comp_cls)
        self._components.append(comp_cls)


def open_file_with_plugin(engine: Any, path: str) -> bool:
    try:
        reg = getattr(engine, "plugin_ui_registry", None) or {}
    except Exception:
        return False
    ext = os.path.splitext(path or "")[1].lower()
    if not ext:
        return False
    for opener in reg.get("file_openers", []):
        try:
            registered = [str(e).lower() for e in opener.get("extensions", [])]
        except Exception:
            continue
        if ext not in registered:
            continue
        handler = opener.get("handler")
        if not callable(handler):
            continue
        try:
            handler(path)
            return True
        except Exception as e:
            Logger.error(f"[Plugin] File opener '{opener.get('label', '?')}' failed for '{path}': {e}", e)
            return False
    return False


class PluginManager:
    def __init__(self):
        self._plugins: dict[str, PluginBase] = {}
        self._load_order: list[str] = []
        self._engine: Optional[Engine] = None

    def set_engine(self, engine: Engine):
        self._engine = engine

    def register(self, plugin: PluginBase):
        name = plugin.NAME
        if name in self._plugins:
            Logger.warning(f"Plugin '{name}' already registered, skipping.")
            return
        try:
            plugin.initialize(self._engine)
            self._plugins[name] = plugin
            self._load_order.append(name)
            Logger.info(f"Plugin '{name}' v{plugin.VERSION} loaded.")
            self._notify_ui_registrations(plugin)
        except Exception as e:
            Logger.error(f"Failed to init plugin '{name}': {e}", e)

    def _notify_ui_registrations(self, plugin: PluginBase):
        if self._engine is None:
            return
        reg = getattr(self._engine, "plugin_ui_registry", None)
        if reg is None:
            return
        plugin_name = plugin.NAME
        for dock in plugin._docks:
            reg["docks"].append({**dock, "plugin": plugin_name})
        for action in plugin._toolbar_actions:
            reg["toolbar_actions"].append({**action, "plugin": plugin_name})
        for item in plugin._menu_items:
            reg["menu_items"].append({**item, "plugin": plugin_name})
        for opener in plugin._file_openers:
            reg.setdefault("file_openers", []).append({**opener, "plugin": plugin_name})

    def _register_instances(self, mod, path: str, bundled_libs: Optional[str] = None,
                            manifest: Optional[dict] = None, payload_dir: Optional[str] = None) -> bool:
        registered = False
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and issubclass(obj, PluginBase) and obj is not PluginBase:
                inst = obj()
                inst._native_plugin_path = path
                inst._bundled_library_dir = bundled_libs
                if manifest:
                    inst._manifest = manifest
                if payload_dir:
                    inst._payload_dir = payload_dir
                self.register(inst)
                registered = True
        return registered

    def _load_python_plugin(self, mod_name: str, path: str, bundled_libs: Optional[str] = None,
                            manifest: Optional[dict] = None, native_compiled: bool = False):
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._register_instances(mod, path, bundled_libs, manifest)
        comp_dir = os.path.splitext(path)[0] + "_components"
        self._auto_load_components(comp_dir, None)

    def _auto_load_components(self, comp_dir: str, plugin_module_name: Optional[str] = None):
        if not os.path.isdir(comp_dir):
            return
        key = (plugin_module_name or "zplugin").replace(".", "_")
        for fname in sorted(os.listdir(comp_dir)):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            base = fname[:-3]
            mod = None
            if plugin_module_name:
                try:
                    mod = importlib.import_module(plugin_module_name + ".components." + base)
                except Exception as e:
                    Logger.warning(f"[Plugin] Could not import component '{base}' from {plugin_module_name}: {e}")
                    mod = None
            if mod is None:
                mod_name = "_plugin_comp_" + key + "_" + base
                fpath = os.path.join(comp_dir, fname)
                try:
                    spec = importlib.util.spec_from_file_location(mod_name, fpath)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                except Exception as e:
                    Logger.error(f"[Plugin] Failed to auto-load component module '{fname}': {e}", e)
                    continue
            registered = []
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and obj.__module__ == mod.__name__:
                    from core.ecs.ecs import Component, ComponentRegistry
                    if issubclass(obj, Component):
                        ComponentRegistry.register(obj)
                        registered.append(obj.__name__)
            if registered:
                Logger.info(f"[Plugin] Auto-registered components from {fname}: {', '.join(registered)}")

    def load_from_file(self, path: str):
        try:
            if path.endswith(".zplugin"):
                self.load_zplugin(path)
            elif path.endswith(".py"):
                self._load_python_plugin("_zplugin", path)
            elif path.endswith(".pyd"):
                self._load_python_plugin(os.path.splitext(os.path.basename(path))[0], path)
            elif path.endswith(".dll") or path.endswith(".so"):
                lib = ctypes.CDLL(path)
                get_plugin = lib.get_plugin
                get_plugin.restype = ctypes.py_object
                plugin = get_plugin()
                if isinstance(plugin, PluginBase):
                    plugin._native_plugin_path = path
                    self.register(plugin)
                else:
                    Logger.error(f"DLL '{path}' get_plugin() must return PluginBase instance.")
        except Exception as e:
            Logger.error(f"Failed to load plugin from '{path}': {e}", e)

    def load_directory(self, dirpath: str):
        if not os.path.isdir(dirpath):
            return
        for fname in sorted(os.listdir(dirpath)):
            fpath = os.path.join(dirpath, fname)
            if fname.endswith(".zplugin") and not fname.startswith("_"):
                self.load_zplugin(fpath)
            elif fname.endswith(".py") and not fname.startswith("_"):
                self.load_from_file(fpath)
            elif os.path.isdir(fpath):
                init_path = os.path.join(fpath, "__init__.py")
                if os.path.isfile(init_path):
                    self.load_package(fpath)

    def _activate_bundled_libs(self, libs_dir: str, deps: list):
        if not os.path.isdir(libs_dir):
            return
        lib_paths = [libs_dir]
        for entry in sorted(os.listdir(libs_dir)):
            child = os.path.join(libs_dir, entry)
            if os.path.isdir(child) and os.path.isfile(os.path.join(child, "__init__.py")):
                lib_paths.append(os.path.dirname(child))
            elif (os.path.isfile(child) and (entry.endswith(".pyd") or entry.endswith(".so") or entry.endswith(".dll"))):
                lib_paths.append(os.path.dirname(child))
        for lp in lib_paths:
            if os.path.isdir(lp) and lp not in sys.path:
                sys.path.insert(0, lp)
                Logger.info(f"[zplugin] Bundled library path activated: {lp}")
        if deps:
            missing = []
            for dep in deps:
                name = dep.split(">=")[0].split("==")[0].split("<")[0].strip()
                try:
                    importlib.import_module(name)
                except Exception:
                    missing.append(dep)
            if missing:
                Logger.warning(f"[zplugin] Bundled library missing dependencies: {missing}")

    def _compile_if_needed(self, payload_dir: str, manifest: dict):
        cython_srcs = manifest.get("cython_modules", [])
        if not cython_srcs:
            return True
        outdir = os.path.join(payload_dir, "_native")
        os.makedirs(outdir, exist_ok=True)
        built = []
        for rel in cython_srcs:
            src = os.path.join(payload_dir, rel)
            if not os.path.isfile(src):
                continue
            base = os.path.splitext(os.path.basename(rel))[0]
            target = os.path.join(outdir, base + ".pyd")
            if os.path.isfile(target) and _platform_matches(manifest.get("architecture", "")):
                built.append(target)
                continue
            if self._compile_cython_module(src, outdir):
                built.append(target)
        if built:
            if outdir not in sys.path:
                sys.path.insert(0, outdir)
        return len(built) == len(cython_srcs)

    def _compile_cython_module(self, src: str, outdir: str) -> bool:
        try:
            import Cython
            from setuptools import Extension, setup
        except ImportError as e:
            Logger.error(f"[zplugin] Cython required for cython_modules but not installed: {e}")
            return False
        try:
            from setuptools.command.build_ext import build_ext
            from Cython.Build import cythonize
        except ImportError as e:
            Logger.error(f"[zplugin] Cython build dependencies missing: {e}")
            return False
        import subprocess
        src_dir = os.path.dirname(src)
        pyx = os.path.basename(src)
        setup_py = os.path.join(outdir, "_cython_build_setup.py")
        script = (
            "from setuptools import setup, Extension\n"
            "from Cython.Build import cythonize\n"
            f"setup(ext_modules=cythonize([Extension('zpl_native', [r'{pyx}'], "
            f"extra_compile_args=['/O2'] if __import__('sys').platform == 'win32' else ['-O3'])], "
            f"build_dir=r'{outdir}'), options={{'build_ext': {{'build_lib': r'{outdir}'}}}})\n"
        )
        with open(setup_py, "w", encoding="utf-8") as f:
            f.write(script)
        try:
            result = subprocess.run(
                [sys.executable, setup_py, "build_ext", "--inplace"],
                capture_output=True, text=True, cwd=src_dir,
            )
            if result.returncode == 0:
                for f in os.listdir(outdir):
                    if f.endswith(".pyd"):
                        Logger.info(f"[zplugin] Cython module built: {os.path.join(outdir, f)}")
                        return True
            Logger.error(f"[zplugin] Cython compile failed: {result.stderr[-800:]}")
        except Exception as e:
            Logger.error(f"[zplugin] Cython compile error: {e}")
        return False

    def _extract_zplugin(self, path: str) -> dict:
        manifest = None
        try:
            with zipfile.ZipFile(path) as zf:
                if "manifest.json" not in zf.namelist():
                    raise ValueError("zplugin archive missing manifest.json")
                manifest_name = next((n for n in zf.namelist() if n.endswith("manifest.json")), None)
                with zf.open(manifest_name) as f:
                    manifest = json.loads(f.read().decode("utf-8"))
        except Exception as e:
            Logger.error(f"[zplugin] Invalid package '{path}': {e}", e)
            return {}
        if not isinstance(manifest, dict):
            Logger.error(f"[zplugin] Invalid manifest in '{path}'")
            return {}
        name = str(manifest.get("name", "unnamed"))
        version = str(manifest.get("version", "0.0.1"))
        dest = _resolve_zplugin(name, version)
        if os.path.exists(dest):
            shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(dest, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix="zpl_", dir=_zarin_user_dir("tmp"))
        safe_names = set()
        if not _platform_matches(manifest.get("architecture", "any")):
            has_native = False
            with zipfile.ZipFile(path) as zf:
                for n in zf.namelist():
                    if n.endswith(".pyd") or n.endswith(".so") or n.endswith(".dll"):
                        has_native = True
                        break
            if has_native and not manifest.get("cython_modules"):
                Logger.error(
                    f"[zplugin] '{name}' targets {manifest.get('architecture')}, "
                    f"current is {current_architecture()} and no source fallback is available."
                )
                shutil.rmtree(tmp, ignore_errors=True)
                return {}
        with zipfile.ZipFile(path) as zf:
            for n in zf.namelist():
                target = os.path.join(tmp, n)
                if os.path.isabs(n) or ".." in os.path.normpath(n):
                    shutil.rmtree(tmp, ignore_errors=True)
                    raise ValueError(f"[zplugin] Unsafe path in archive: {n}")
                if n.endswith("/"):
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(n) as srcf, open(target, "wb") as dstf:
                        shutil.copyfileobj(srcf, dstf)
                    safe_names.add(n)
        libs_dir = os.path.join(tmp, "libs")
        self._activate_bundled_libs(libs_dir, manifest.get("dependencies", []))
        self._compile_if_needed(tmp, manifest)
        entry = manifest.get("entry", "plugin.py")
        full_entry = None
        for n in safe_names:
            if n.endswith("/" + entry) or n == entry:
                full_entry = os.path.join(tmp, n)
                break
        if full_entry is None:
            if os.path.isfile(os.path.join(tmp, entry)):
                full_entry = os.path.join(tmp, entry)
        if full_entry is None:
            Logger.error(f"[zplugin] Entry point '{entry}' not found in '{path}'")
            shutil.rmtree(tmp, ignore_errors=True)
            return {}
        merged = {k: v for k, v in manifest.items()}
        merged["payload_dir"] = tmp
        merged["entry_file"] = full_entry
        try:
            manifest_cache = {"__zpl_meta": True, "manifest": merged, "payload": tmp}
            with open(os.path.join(tmp, ".zpl_meta.json"), "w", encoding="utf-8") as f:
                json.dump(manifest_cache, f, indent=2)
        except Exception:
            pass
        return merged

    def load_zplugin(self, path: str):
        try:
            manifest = self._extract_zplugin(path)
            if not manifest or "entry_file" not in manifest:
                return
            entry_file = manifest["entry_file"]
            payload_dir = manifest.get("payload_dir", os.path.dirname(entry_file))
            libs_dir = os.path.join(payload_dir, "libs")
            self._activate_bundled_libs(libs_dir, manifest.get("dependencies", []))
            arch = manifest.get("architecture", "any")
            arch_ok = _platform_matches(arch)
            loaded = False
            pkg = manifest.get("module")
            if pkg and entry_file.endswith(".py"):
                if payload_dir not in sys.path:
                    sys.path.insert(0, payload_dir)
                try:
                    mod = importlib.import_module(pkg)
                    loaded = self._register_instances(
                        mod, os.path.join(payload_dir, pkg), libs_dir, manifest, payload_dir)
                except Exception as e:
                    Logger.error(f"[zplugin] Package import failed for '{pkg}': {e}", e)
            if not loaded and entry_file.endswith(".pyd") and not arch_ok:
                base = os.path.splitext(entry_file)[0]
                if os.path.isfile(base + ".py"):
                    entry_file = base + ".py"
            if not loaded:
                if entry_file.endswith(".pyd") or entry_file.endswith(".so") or entry_file.endswith(".dll"):
                    self._load_python_plugin(os.path.splitext(os.path.basename(entry_file))[0],
                                             entry_file, libs_dir, manifest)
                    loaded = True
                elif entry_file.endswith(".py"):
                    self._load_python_plugin("_zplugin", entry_file, libs_dir, manifest)
                    loaded = True
            if not loaded:
                Logger.error(f"[zplugin] '{manifest.get('name')}' not loadable on {current_architecture()}")
            name = manifest.get("name", "unknown")
            Logger.info(f"[zplugin] Loaded package '{name}' v{manifest.get('version', '')} from {os.path.basename(path)}")
        except Exception as e:
            Logger.error(f"Failed to load zplugin '{path}': {e}", e)

    def load_package(self, dirpath: str):
        try:
            basename = os.path.basename(dirpath)
            if basename.startswith("_"):
                return
            canon = "plugins." + basename
            mod = None
            try:
                mod = importlib.import_module(canon)
            except Exception:
                init_path = os.path.join(dirpath, "__init__.py")
                pkg_name = "_plugin_" + basename
                spec = importlib.util.spec_from_file_location(pkg_name, init_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and issubclass(obj, PluginBase) and obj is not PluginBase:
                    inst = obj()
                    inst._native_plugin_path = dirpath
                    self.register(inst)
            comp_dir = os.path.join(dirpath, "components")
            self._auto_load_components(comp_dir, canon)
        except Exception as e:
            Logger.error(f"Failed to load plugin package '{dirpath}': {e}")

    def load_module(self, module_name: str):
        """Load a plugin from a compiled (Nuitka) module by dotted name."""
        try:
            mod = importlib.import_module(module_name)
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and issubclass(obj, PluginBase) and obj is not PluginBase:
                    inst = obj()
                    inst._native_plugin_path = module_name
                    self.register(inst)
        except Exception as e:
            Logger.error(f"Failed to load plugin module '{module_name}': {e}", e)

    def get(self, name: str) -> Optional[PluginBase]:
        return self._plugins.get(name)

    def get_all(self) -> list[PluginBase]:
        return [self._plugins[n] for n in self._load_order if n in self._plugins]

    def get_system_plugins(self) -> list[PluginBase]:
        return [p for p in self.get_all() if p.SYSTEM]

    def shutdown_all(self):
        for name in reversed(self._load_order):
            p = self._plugins.get(name)
            if p:
                try:
                    p.shutdown()
                except Exception as e:
                    Logger.error(f"Error shutting down plugin '{name}': {e}", e)

    def _notify_all(self, method_name: str, *args):
        plugins = self.get_all()
        if len(plugins) < 4:
            for p in plugins:
                try:
                    getattr(p, method_name)(*args)
                except Exception as e:
                    Logger.error(f"Plugin {method_name} error: {e}", e)
            return
        from concurrent.futures import as_completed
        futures = []
        for p in plugins:
            futures.append(_get_plugin_pool().submit(getattr(p, method_name), *args))
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                Logger.error(f"Plugin notify {method_name} error: {e}")

    def notify_scene_loaded(self, scene):
        self._notify_all("on_scene_loaded", scene)

    def notify_scene_unloaded(self, scene):
        self._notify_all("on_scene_unloaded", scene)

    def notify_project_opened(self):
        for p in self.get_all():
            try:
                p.on_project_opened()
            except Exception as e:
                Logger.error(f"Plugin on_project_opened error: {e}", e)

    def notify_play_start(self):
        self._notify_all("on_play_start")

    def notify_play_stop(self):
        self._notify_all("on_play_stop")
