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
import sys
import os
from typing import Any, Callable, Optional, TYPE_CHECKING
from core.foundation.logger import Logger
from core.ecs.pool import plugin as _get_plugin_pool
if TYPE_CHECKING:
    from core.engine.engine import Engine


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
        self._components: list[type] = []

    def initialize(self, engine: Engine):
        self._engine = engine
        self._load_config()

    def shutdown(self):
        self._save_config()

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

    # ---- Native Plugin Resource Path ----

    @property
    def resource_dir(self) -> Optional[str]:
        path = getattr(self, "_native_plugin_path", None)
        if path:
            base = os.path.splitext(path)[0]
            res = base + "_resources"
            if os.path.isdir(res):
                return res
        return None


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

    def _load_python_plugin(self, mod_name: str, path: str):
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and issubclass(obj, PluginBase) and obj is not PluginBase:
                inst = obj()
                inst._native_plugin_path = path
                self.register(inst)

    def load_from_file(self, path: str):
        try:
            if path.endswith(".py"):
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
            if fname.endswith(".py") and not fname.startswith("_"):
                self.load_from_file(fpath)
            elif os.path.isdir(fpath):
                init_path = os.path.join(fpath, "__init__.py")
                if os.path.isfile(init_path):
                    self.load_package(fpath)

    def load_package(self, dirpath: str):
        try:
            init_path = os.path.join(dirpath, "__init__.py")
            pkg_name = "_plugin_" + os.path.basename(dirpath)
            spec = importlib.util.spec_from_file_location(pkg_name, init_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and issubclass(obj, PluginBase) and obj is not PluginBase:
                    inst = obj()
                    inst._native_plugin_path = dirpath
                    self.register(inst)
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
