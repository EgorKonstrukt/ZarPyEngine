# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import importlib
import importlib.util
import ctypes
import hashlib
import json
import platform
import sys
import os
import shutil
import zipfile
import tempfile
from typing import Any, Callable, Optional, TYPE_CHECKING
from core.foundation.logger import Logger
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


PLUGIN_API_VERSION = 1


def _parse_version(text) -> tuple:
    out = []
    cur = ""
    for ch in str(text).strip():
        if ch.isdigit():
            cur += ch
        else:
            if cur:
                out.append(int(cur))
                cur = ""
            if ch.isalnum():
                out.append(ch.lower())
    if cur:
        out.append(int(cur))
    return tuple(out)


def _compare_versions(a, b) -> int:
    pa = _parse_version(a)
    pb = _parse_version(b)
    for x, y in zip(pa, pb):
        if x == y:
            continue
        if isinstance(x, int) and isinstance(y, int):
            return -1 if x < y else 1
        if isinstance(x, int):
            return -1
        if isinstance(y, int):
            return 1
        return -1 if str(x) < str(y) else 1
    if len(pa) == len(pb):
        return 0
    longer = pa if len(pa) > len(pb) else pb
    sign = 1 if len(pa) > len(pb) else -1
    for x in longer[min(len(pa), len(pb)):]:
        if x != 0 and x != "":
            return sign
    return 0


def _split_spec(spec: str) -> list:
    ops = []
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        for op in ("==", "!=", ">=", "<=", "~=", ">", "<", "="):
            if part.startswith(op):
                ops.append((op, part[len(op):].strip()))
                break
        else:
            ops.append(("==", part))
    return ops


def _compatible_release(version, want) -> bool:
    if _compare_versions(version, want) < 0:
        return False
    wp = _parse_version(want)
    prefix = wp[:-1] if len(wp) > 1 else wp
    vp = _parse_version(version)
    return tuple(vp[:len(prefix)]) == tuple(prefix)


def _spec_satisfied(spec, version) -> bool:
    if not spec:
        return True
    for op, want in _split_spec(spec):
        if op in ("==", "=") and want.endswith(".*"):
            prefix = _parse_version(want[:-2])
            vp = _parse_version(version)
            if tuple(vp[:len(prefix)]) != tuple(prefix):
                return False
            continue
        c = _compare_versions(version, want)
        if op == "==" or op == "=":
            ok = c == 0
        elif op == "!=":
            ok = c != 0
        elif op == ">=":
            ok = c >= 0
        elif op == "<=":
            ok = c <= 0
        elif op == ">":
            ok = c > 0
        elif op == "<":
            ok = c < 0
        elif op == "~=":
            ok = _compatible_release(version, want)
        else:
            return False
        if not ok:
            return False
    return True


def _pep508_name(spec: str) -> str:
    s = str(spec or "").split(";")[0].strip()
    name = ""
    for ch in s:
        if ch.isalnum() or ch in "-_.":
            name += ch
        else:
            break
    return name


def _split_requirement(spec: str):
    s = str(spec or "").split(";")[0].strip()
    name = _pep508_name(s)
    rest = s[len(name):].strip()
    if rest.startswith("["):
        end = rest.find("]")
        rest = rest[end + 1:].strip() if end != -1 else ""
    return name, rest


def _split_plugin_require(req):
    if isinstance(req, dict):
        return str(req.get("name", "")), str(req.get("version", ""))
    s = str(req or "")
    name = _pep508_name(s)
    return name, s[len(name):].strip()


def _normalize_dist_name(name: str) -> str:
    out = []
    for ch in str(name or "").lower():
        out.append("-" if ch in "-_." else ch)
    collapsed = []
    for ch in out:
        if ch == "-" and collapsed and collapsed[-1] == "-":
            continue
        collapsed.append(ch)
    return "".join(collapsed).strip("-")


def _installed_dist_version(name: str):
    try:
        from importlib import metadata as _md
        return _md.version(_normalize_dist_name(name))
    except Exception:
        return None


def _read_package_manifest(dirpath: str) -> dict:
    try:
        path = os.path.join(dirpath, "manifest.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    return loaded
    except Exception:
        pass
    return {}


def _read_archive_manifest(path: str) -> dict:
    try:
        with zipfile.ZipFile(path) as zf:
            mname = next((n for n in zf.namelist() if n.endswith("manifest.json")), None)
            if mname is None:
                return {}
            loaded = json.loads(zf.open(mname).read().decode("utf-8"))
            if isinstance(loaded, dict):
                return loaded
    except Exception:
        pass
    return {}


def _scan_provides(libs_dir: str) -> dict:
    provides = {}
    try:
        for entry in sorted(os.listdir(libs_dir)):
            if entry.startswith((".", "_")) or entry == "__pycache__":
                continue
            full = os.path.join(libs_dir, entry)
            if entry.endswith(".py") and os.path.isfile(full):
                provides[entry[:-3]] = "unknown"
            elif os.path.isdir(full) and os.path.isfile(os.path.join(full, "__init__.py")):
                provides[entry] = "unknown"
    except Exception:
        pass
    return provides


_DERIVED_MANIFEST_KEYS = {"signature", "payload_dir", "entry_file", "entry_rel"}


def _canonical_manifest_bytes(manifest: dict) -> bytes:
    slim = {k: v for k, v in manifest.items() if k not in _DERIVED_MANIFEST_KEYS}
    return json.dumps(slim, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _self_provided_modules(kind, path, manifest: dict) -> set:
    out = set()
    if isinstance(manifest, dict):
        provides = manifest.get("provides")
        if isinstance(provides, dict):
            for mod in provides:
                key = _normalize_dist_name(mod)
                if key:
                    out.add(key)
    if kind == "package" and path:
        for mod in _scan_provides(os.path.join(path, "libs")):
            out.add(_normalize_dist_name(mod))
    return out


def _zplugin_fingerprint(path: str) -> dict:
    try:
        st = os.stat(path)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return {"sha256": h.hexdigest(), "size": st.st_size, "mtime": st.st_mtime}
    except Exception:
        return {}


class PluginBase:
    NAME: str = "UnnamedPlugin"
    VERSION: str = "0.0.1"
    DESCRIPTION: str = ""
    SYSTEM: bool = False

    def __init__(self):
        self._engine: Optional[Engine] = None
        self._manager: Optional[PluginManager] = None
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

    def on_main_window_ready(self, main_window):
        pass

    @property
    def engine(self) -> Optional[Engine]:
        return self._engine

    def main_window(self):
        eng = self._engine
        if eng is None:
            return None
        get = getattr(eng, "get_main_window", None)
        if not callable(get):
            return None
        try:
            return get()
        except Exception:
            return None

    # ---- Config API ----

    def get_config(self, key: str, default: Any = None, section: Optional[str] = None) -> Any:
        if section:
            sec = self._config.get(section)
            if isinstance(sec, dict):
                return sec.get(key, default)
            return default
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any, section: Optional[str] = None):
        if section:
            sec = self._config.get(section)
            if not isinstance(sec, dict):
                sec = {}
                self._config[section] = sec
            sec[key] = value
        else:
            self._config[key] = value
        self._save_config()

    def save_config(self):
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

    def _sync_ui(self, kind: str, entry: dict):
        mgr = getattr(self, "_manager", None)
        eng = self._engine
        if mgr is None or eng is None:
            return
        try:
            mgr.sync_plugin_ui(self, kind, entry)
        except Exception as e:
            Logger.error(f"[{self.NAME}] UI sync failed: {e}", e)

    # ---- Dock Registration ----

    def register_dock(self, title: str, widget_factory: Callable[[], Any],
                      area: str = "left", tab_group: Optional[str] = None,
                      icon: Optional[str] = None):
        entry = {
            "title": title,
            "widget_factory": widget_factory,
            "area": area,
            "tab_group": tab_group,
            "icon": icon,
        }
        self._docks.append(entry)
        self._sync_ui("docks", entry)
        return entry

    # ---- Toolbar Registration ----

    def add_toolbar_button(self, text: str, callback: Callable,
                           icon: Optional[str] = None, tooltip: str = ""):
        entry = {
            "text": text,
            "callback": callback,
            "icon": icon,
            "tooltip": tooltip or text,
        }
        self._toolbar_actions.append(entry)
        self._sync_ui("toolbar_actions", entry)
        return entry

    # ---- Menu Registration ----

    def add_menu_item(self, menu_name: str, text: str, callback: Callable,
                      shortcut: Optional[str] = None, icon: Optional[str] = None):
        entry = {
            "menu": menu_name,
            "text": text,
            "callback": callback,
            "shortcut": shortcut,
            "icon": icon,
        }
        self._menu_items.append(entry)
        self._sync_ui("menu_items", entry)
        return entry

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
            return None
        entry = {
            "extensions": normed,
            "handler": handler,
            "label": label or self.NAME,
        }
        self._file_openers.append(entry)
        self._sync_ui("file_openers", entry)
        return entry

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
        self._status: dict[str, dict] = {}
        self._pending: list[dict] = []
        self._libraries: dict[str, dict] = {}
        self._library_packs: dict[str, dict] = {}
        self._lib_owners: list[tuple] = []

    def plugin_status(self, name: str) -> dict:
        return dict(self._status.get(name, {"state": "unknown", "reason": ""}))

    def all_plugin_statuses(self) -> dict:
        return {n: dict(v) for n, v in self._status.items()}

    def _set_status(self, name: str, state: str, reason: str = ""):
        self._status[str(name)] = {"state": state, "reason": str(reason or "")}

    def set_engine(self, engine: Engine):
        self._engine = engine

    def _origin_is_user(self, origin) -> bool:
        parts = str(origin or "").replace("\\", "/").replace(".", "/").split("/")
        return "user" in parts

    def register(self, plugin: PluginBase):
        name = plugin.NAME
        if name in self._plugins:
            old = self._plugins[name]
            new_path = str(getattr(plugin, "_native_plugin_path", "") or "")
            old_path = str(getattr(old, "_native_plugin_path", "") or "")
            if new_path and new_path != old_path and self._origin_is_user(new_path) and not self._origin_is_user(old_path):
                Logger.warning(f"Plugin '{name}' from user dir overrides '{old_path}'.")
                self.unregister(name)
            else:
                Logger.warning(f"Plugin '{name}' already registered from '{old_path}', skipping '{new_path}'.")
                return
        plugin._manager = self
        try:
            plugin.initialize(self._engine)
            self._plugins[name] = plugin
            self._load_order.append(name)
            Logger.info(f"Plugin '{name}' v{plugin.VERSION} loaded.")
            self._set_status(name, "loaded")
            self._notify_ui_registrations(plugin)
        except Exception as e:
            Logger.error(f"Failed to init plugin '{name}': {e}", e)
            self._set_status(name, "failed", str(e))

    def unregister(self, name: str) -> bool:
        plugin = self._plugins.get(name)
        if plugin is None:
            return False
        try:
            plugin.shutdown()
        except Exception as e:
            Logger.error(f"Error shutting down plugin '{name}': {e}", e)
        self._plugins.pop(name, None)
        if name in self._load_order:
            self._load_order.remove(name)
        for entries in (plugin._docks, plugin._toolbar_actions,
                        plugin._menu_items, plugin._file_openers):
            for entry in entries:
                if isinstance(entry, dict):
                    entry.pop("_synced", None)
        eng = self._engine
        reg = getattr(eng, "plugin_ui_registry", None) if eng is not None else None
        if reg is not None:
            for kind in ("docks", "toolbar_actions", "menu_items", "file_openers"):
                try:
                    reg[kind] = [e for e in reg.get(kind, []) if e.get("plugin") != name]
                except Exception as e:
                    Logger.warning(f"[Plugin] registry cleanup failed for '{kind}': {e}")
            self._fire_runtime({"unregistered": name}, reg)
        plugin._manager = None
        return True

    def reload_plugin_package(self, dirpath: str) -> bool:
        norm = os.path.normpath(dirpath)
        doomed = [n for n, p in self._plugins.items()
                  if os.path.normpath(str(getattr(p, "_native_plugin_path", "") or "")) == norm]
        for n in doomed:
            self.unregister(n)
        base = os.path.basename(norm)
        prefixes = ("plugins." + base, "_plugin_" + base)
        for modname in [m for m in list(sys.modules)
                        if m in prefixes or m.startswith(prefixes[0] + ".") or m.startswith(prefixes[1] + ".")]:
            try:
                del sys.modules[modname]
            except Exception:
                pass
        try:
            self.load_package(dirpath)
            return True
        except Exception as e:
            Logger.error(f"Failed to reload plugin package '{dirpath}': {e}", e)
            return False

    def sync_plugin_ui(self, plugin: PluginBase, kind: str, entry: dict):
        eng = self._engine
        reg = getattr(eng, "plugin_ui_registry", None) if eng is not None else None
        if reg is None:
            return
        if not isinstance(entry, dict) or entry.get("_synced"):
            return
        merged = {k: v for k, v in entry.items() if not k.startswith("_")}
        merged["plugin"] = plugin.NAME
        reg.setdefault(kind, []).append(merged)
        entry["_synced"] = True
        self._fire_runtime({"plugin": plugin.NAME, kind: [merged]}, reg)

    def _fire_runtime(self, payload: dict, reg=None):
        if reg is None:
            eng = self._engine
            reg = getattr(eng, "plugin_ui_registry", None) if eng is not None else None
        if not reg:
            return
        for cb in list(reg.get("runtime_listeners", [])):
            try:
                cb(payload)
            except Exception as e:
                Logger.error(f"[Plugin] UI runtime listener failed: {e}", e)

    def _plugin_security_config(self) -> dict:
        cfg = {"allow_unsigned": "warn", "registry_url": "",
               "check_updates_on_startup": False, "trusted_keys_dir": ""}
        try:
            from core.config.config import get_global_config
            g = get_global_config()
            for k in cfg:
                v = g.get("plugins." + k, None)
                if v is not None:
                    cfg[k] = v
        except Exception:
            pass
        return cfg

    def _trusted_keys(self) -> set:
        keys = set()
        custom = str(self._plugin_security_config().get("trusted_keys_dir", "") or "")
        dirs = []
        if custom:
            dirs.append(custom if os.path.isabs(custom) else os.path.join(_project_root(), custom))
        dirs.append(os.path.join(_project_root(), "config", "trusted_keys"))
        for d in dirs:
            try:
                if not os.path.isdir(d):
                    continue
                for fname in sorted(os.listdir(d)):
                    if not fname.endswith(".pub"):
                        continue
                    with open(os.path.join(d, fname), "r", encoding="utf-8") as f:
                        for token in f.read().split():
                            token = token.strip().lower()
                            if len(token) == 64:
                                try:
                                    bytes.fromhex(token)
                                    keys.add(token)
                                except Exception:
                                    pass
            except Exception:
                pass
        return keys

    def verify_manifest_signature(self, manifest: dict):
        sig = manifest.get("signature") if isinstance(manifest, dict) else None
        if not isinstance(sig, dict) or not sig.get("sig"):
            return (False, "unsigned")
        keys = self._trusted_keys()
        if not keys:
            return (False, "no trusted keys configured")
        msg = _canonical_manifest_bytes(manifest)
        by = str(sig.get("by", "") or "").lower()
        candidates = [by] if by in keys else sorted(keys)
        from core.foundation.ed25519 import verify_hex
        for pub in candidates:
            try:
                if verify_hex(str(sig.get("sig", "")), msg, pub):
                    return (True, pub)
            except Exception:
                continue
        return (False, "signature mismatch")

    def verify_zplugin_file(self, path: str):
        manifest = _read_archive_manifest(path)
        if not manifest:
            return (False, "missing or unreadable manifest.json", {})
        ok, reason = self.verify_manifest_signature(manifest)
        if ok:
            return (True, "signed", manifest)
        if reason != "unsigned":
            return (False, reason, manifest)
        policy = str(self._plugin_security_config().get("allow_unsigned", "warn"))
        if policy == "strict":
            return (False, "unsigned package blocked by policy", manifest)
        return (True, "unsigned", manifest)

    def _verify_payload_files(self, payload_dir: str, manifest: dict) -> list:
        errors = []
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, dict):
            return ["manifest has no file index"]
        for rel, want in files.items():
            p = os.path.join(payload_dir, str(rel).replace("/", os.sep))
            if not os.path.isfile(p):
                errors.append(f"missing file '{rel}'")
                continue
            try:
                if _hash_file(p) != str(want).lower():
                    errors.append(f"modified file '{rel}'")
            except Exception as e:
                errors.append(f"unreadable file '{rel}': {e}")
        for root, _dirs, fns in os.walk(payload_dir):
            for fn in fns:
                if fn.endswith((".pyd", ".so", ".dll")):
                    rel = os.path.relpath(os.path.join(root, fn), payload_dir).replace(os.sep, "/")
                    if rel.startswith("_native/"):
                        continue
                    if rel not in files:
                        errors.append(f"unindexed native module '{rel}'")
        return errors

    def _check_pip_requirement(self, spec: str, self_provided=frozenset()):
        name, constraint = _split_requirement(spec)
        if not name:
            return (False, f"bad requirement '{spec}'")
        if _normalize_dist_name(name) in self_provided:
            return (True, "")
        ver = _installed_dist_version(name)
        if ver is not None and _spec_satisfied(constraint, ver):
            return (True, "")
        lib = self._libraries.get(_normalize_dist_name(name))
        if lib is not None:
            lv = str(lib.get("version") or "")
            if not constraint or (lv and lv != "unknown" and _spec_satisfied(constraint, lv)):
                return (True, "")
        have = ver if ver is not None else (str(lib.get("version")) if lib else "")
        return (False, f"{spec} ({('have ' + have) if have else 'not installed'})")

    def _manifest_gate_errors(self, manifest: dict, self_provided=frozenset()) -> dict:
        blocked, missing = [], []
        if not isinstance(manifest, dict) or not manifest:
            return {"blocked": blocked, "missing": missing}
        api = manifest.get("engine_api")
        if api and not _spec_satisfied(str(api), str(PLUGIN_API_VERSION)):
            blocked.append(f"engine API {api} not satisfied (current {PLUGIN_API_VERSION})")
        pyreq = manifest.get("python_requires")
        if pyreq and not _spec_satisfied(str(pyreq), platform.python_version()):
            blocked.append(f"python {pyreq} not satisfied (current {platform.python_version()})")
        for dep in manifest.get("dependencies", []) or []:
            ok, reason = self._check_pip_requirement(dep, self_provided)
            if not ok:
                missing.append(f"missing dependency: {reason}")
        for req in manifest.get("requires", []) or []:
            name, verspec = _split_plugin_require(req)
            p = self._plugins.get(name)
            if p is None or not _spec_satisfied(verspec, str(p.VERSION)):
                missing.append(f"requires plugin '{name}{(' ' + verspec) if verspec else ''}'")
        return {"blocked": blocked, "missing": missing}

    def _gate_or_defer(self, kind: str, path: str, manifest: dict) -> bool:
        name = str(manifest.get("name", "") or os.path.basename(path))
        gates = self._manifest_gate_errors(manifest, _self_provided_modules(kind, path, manifest))
        if gates["blocked"]:
            self._set_status(name, "blocked", "; ".join(gates["blocked"]))
            Logger.error(f"[Plugin] '{name}' blocked: {'; '.join(gates['blocked'])}")
            return False
        if gates["missing"]:
            self._pending.append({"kind": kind, "path": path, "manifest": dict(manifest)})
            self._set_status(name, "unresolved", "; ".join(gates["missing"]))
            Logger.warning(f"[Plugin] '{name}' deferred: {'; '.join(gates['missing'])}")
            return False
        return True

    def recheck_pending(self):
        self._drain_pending()

    def _drain_pending(self):
        items, self._pending = self._pending, []
        for item in items:
            manifest = item.get("manifest", {})
            name = str(manifest.get("name", "") or os.path.basename(item.get("path", "")))
            gates = self._manifest_gate_errors(manifest, _self_provided_modules(item.get("kind"), item.get("path"), manifest))
            if gates["blocked"]:
                self._set_status(name, "blocked", "; ".join(gates["blocked"]))
                continue
            if gates["missing"]:
                self._pending.append(item)
                self._set_status(name, "unresolved", "; ".join(gates["missing"]))
                continue
            kind, path = item.get("kind"), item.get("path")
            if kind == "zplugin":
                self.load_zplugin(path)
            elif kind == "package":
                self.load_package(path)
            elif kind == "pyfile":
                self.load_from_file(path)

    def _record_lib_owner(self, mod, owner, path):
        key = (_normalize_dist_name(mod), str(owner), str(path))
        if key not in self._lib_owners:
            self._lib_owners.append(key)

    def _register_library_pack(self, pack_name, version, provides, libs_dir):
        provides = dict(provides) if isinstance(provides, dict) else {}
        if not provides and libs_dir:
            provides = _scan_provides(libs_dir)
        modules = []
        for mod, ver in provides.items():
            key = _normalize_dist_name(mod)
            if not key:
                continue
            self._libraries[key] = {"owner": str(pack_name), "version": str(ver or "unknown"),
                                    "path": str(libs_dir or "")}
            modules.append(key)
        if libs_dir:
            for mod in provides:
                self._record_lib_owner(mod, pack_name, libs_dir)
        self._library_packs[str(pack_name)] = {"version": str(version or "unknown"), "modules": modules}
        self._set_status(str(pack_name), "loaded")

    def _register_library_package(self, dirpath: str, manifest: dict):
        name = str(manifest.get("name", "") or os.path.basename(dirpath))
        version = str(manifest.get("version", "unknown"))
        libs_dir = os.path.join(dirpath, "libs")
        if os.path.isdir(libs_dir):
            self._activate_bundled_libs(libs_dir, manifest.get("dependencies", []), name)
        self._register_library_pack(name, version, manifest.get("provides", {}),
                                    libs_dir if os.path.isdir(libs_dir) else "")
        Logger.info(f"[Plugin] Library pack '{name}' v{version} ready.")

    def _register_library_archive(self, manifest: dict, payload_dir: str):
        name = str(manifest.get("name", "unnamed"))
        version = str(manifest.get("version", "unknown"))
        libs_dir = os.path.join(payload_dir, "libs")
        self._register_library_pack(name, version, manifest.get("provides", {}),
                                    libs_dir if os.path.isdir(libs_dir) else "")
        Logger.info(f"[Plugin] Library pack '{name}' v{version} ready.")

    def _installed_plugin_version(self, name: str):
        p = self._plugins.get(name)
        if p is not None:
            return str(p.VERSION)
        pack = self._library_packs.get(name)
        if pack is not None:
            return str(pack.get("version", "unknown"))
        return None

    def library_conflicts(self) -> list:
        by_mod = {}
        for mod, owner, path in self._lib_owners:
            by_mod.setdefault(mod, []).append({"owner": owner, "path": path})
        out = []
        for mod in sorted(by_mod):
            owners = by_mod[mod]
            if len({o["path"] for o in owners}) > 1:
                out.append({"module": mod, "owners": owners})
        return out

    def doctor(self) -> dict:
        by_state = {}
        for n, s in self._status.items():
            by_state.setdefault(s.get("state", "unknown"), {})[n] = s.get("reason", "")
        return {
            "conflicts": self.library_conflicts(),
            "unresolved": by_state.get("unresolved", {}),
            "blocked": by_state.get("blocked", {}),
            "failed": by_state.get("failed", {}),
            "libraries": {k: dict(v) for k, v in self._library_packs.items()},
        }

    def fetch_plugin_index(self, url=None, timeout=15):
        url = url or str(self._plugin_security_config().get("registry_url", "") or "")
        if not url:
            return {}
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=timeout) as r:
                loaded = json.loads(r.read().decode("utf-8"))
                return loaded if isinstance(loaded, dict) else {}
        except Exception as e:
            Logger.error(f"[Plugin] registry fetch failed: {e}")
            return {}

    def check_plugin_updates(self, index=None, url=None):
        if index is None:
            index = self.fetch_plugin_index(url)
        entries = index.get("plugins", []) if isinstance(index, dict) else []
        out = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            name = str(e.get("name", "") or "")
            avail = str(e.get("version", "") or "")
            if not name or not avail:
                continue
            cur = self._installed_plugin_version(name)
            if cur is None or cur == "unknown":
                continue
            try:
                if _compare_versions(avail, cur) > 0:
                    out.append({"name": name, "current": cur, "available": avail,
                                "url": str(e.get("download_url", "") or ""),
                                "sha256": str(e.get("sha256", "") or ""),
                                "summary": str(e.get("summary", "") or "")})
            except Exception:
                continue
        return out

    def download_plugin_file(self, url, dest_dir, expected_sha256=None, timeout=60):
        try:
            import urllib.request
            os.makedirs(dest_dir, exist_ok=True)
            base = os.path.basename(str(url).split("?")[0]) or "download.zplugin"
            fd, tmp = tempfile.mkstemp(dir=dest_dir, suffix=".part")
            os.close(fd)
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
                if expected_sha256 and _hash_file(tmp) != str(expected_sha256).lower():
                    raise ValueError("sha256 mismatch")
                final = os.path.join(dest_dir, base)
                if os.path.abspath(final) == os.path.abspath(tmp):
                    return final
                if os.path.exists(final):
                    os.remove(final)
                os.replace(tmp, final)
                return final
            except Exception:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                raise
        except Exception as e:
            Logger.error(f"[Plugin] download failed: {e}")
            return ""

    def _find_installed_archive(self, name: str, plugins_dir="plugins"):
        try:
            for fname in sorted(os.listdir(plugins_dir)):
                if not fname.endswith(".zplugin"):
                    continue
                manifest = _read_archive_manifest(os.path.join(plugins_dir, fname))
                if manifest.get("name") == name:
                    return os.path.join(plugins_dir, fname)
        except Exception:
            pass
        return ""

    def _pending_queue_path(self, plugins_dir="plugins"):
        d = os.path.join(plugins_dir, ".pending")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "updates.json")

    def _read_pending_queue(self, plugins_dir="plugins"):
        try:
            with open(self._pending_queue_path(plugins_dir), "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    return [q for q in loaded if isinstance(q, dict)]
        except Exception:
            pass
        return []

    def _write_pending_queue(self, queue, plugins_dir="plugins"):
        try:
            with open(self._pending_queue_path(plugins_dir), "w", encoding="utf-8") as f:
                json.dump(queue, f, indent=2)
            return True
        except Exception as e:
            Logger.error(f"[Plugin] could not write update queue: {e}")
            return False

    def install_plugin_file(self, src, plugins_dir="plugins"):
        try:
            if str(src).endswith(".zplugin"):
                ok, reason, _m = self.verify_zplugin_file(src)
                if not ok:
                    return (False, reason)
            os.makedirs(plugins_dir, exist_ok=True)
            dest = os.path.join(plugins_dir, os.path.basename(src))
            shutil.copy2(src, dest)
            return (True, dest)
        except Exception as e:
            return (False, str(e))

    def stage_plugin_update(self, name, url, sha256=None, plugins_dir="plugins"):
        staged = self.download_plugin_file(url, os.path.join(plugins_dir, ".pending"), sha256)
        if not staged:
            return False
        queue = [q for q in self._read_pending_queue(plugins_dir) if q.get("name") != name]
        queue.append({"name": name, "staged": staged,
                      "current": self._find_installed_archive(name, plugins_dir)})
        return self._write_pending_queue(queue, plugins_dir)

    def apply_pending_updates(self, plugins_dir="plugins"):
        queue = self._read_pending_queue(plugins_dir)
        if not queue:
            return []
        backup_dir = os.path.join(plugins_dir, ".backup")
        os.makedirs(backup_dir, exist_ok=True)
        applied, remaining = [], []
        for q in queue:
            try:
                name, staged = q.get("name", ""), q.get("staged", "")
                if not name or not staged or not os.path.isfile(staged):
                    raise ValueError("staged file missing")
                target = os.path.join(plugins_dir, os.path.basename(staged))
                current = q.get("current") or ""
                backup = os.path.join(backup_dir, os.path.basename(target) + ".bak")
                if os.path.exists(backup):
                    os.remove(backup)
                sidecar = backup + ".json"
                if os.path.exists(sidecar):
                    os.remove(sidecar)
                if current and os.path.isfile(current) and os.path.abspath(current) != os.path.abspath(target):
                    shutil.copy2(current, backup)
                    with open(sidecar, "w", encoding="utf-8") as f:
                        json.dump({"original": os.path.basename(current)}, f)
                    os.remove(current)
                elif os.path.isfile(target):
                    os.replace(target, backup)
                os.replace(staged, target)
                applied.append(name)
            except Exception as e:
                Logger.error(f"[Plugin] update apply failed for '{q.get('name', '?')}': {e}")
                remaining.append(q)
        self._write_pending_queue(remaining, plugins_dir)
        return applied

    def rollback_plugin(self, name, plugins_dir="plugins"):
        try:
            current = self._find_installed_archive(name, plugins_dir)
            if not current:
                return False
            backup = os.path.join(plugins_dir, ".backup", os.path.basename(current) + ".bak")
            if not os.path.isfile(backup):
                return False
            sidecar = backup + ".json"
            original = ""
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    if isinstance(saved, dict):
                        original = str(saved.get("original", "") or "")
            except Exception:
                pass
            os.remove(current)
            if original and original != os.path.basename(current):
                os.replace(backup, os.path.join(plugins_dir, original))
            else:
                os.replace(backup, current)
            try:
                if os.path.isfile(sidecar):
                    os.remove(sidecar)
            except Exception:
                pass
            return True
        except Exception as e:
            Logger.error(f"[Plugin] rollback failed for '{name}': {e}")
            return False

    def _notify_ui_registrations(self, plugin: PluginBase):
        if self._engine is None:
            return
        pairs = (
            ("docks", plugin._docks),
            ("toolbar_actions", plugin._toolbar_actions),
            ("menu_items", plugin._menu_items),
            ("file_openers", plugin._file_openers),
        )
        for kind, entries in pairs:
            for entry in list(entries):
                self.sync_plugin_ui(plugin, kind, entry)

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

    def _topo_plan(self, entries):
        by_name = {}
        for kind, path, manifest in entries:
            name = str(manifest.get("name", "") or path)
            by_name.setdefault(name, []).append((kind, path, manifest))
        names = sorted(by_name)
        deps = {}
        for name in names:
            want = set()
            for kind, path, manifest in by_name[name]:
                for req in manifest.get("requires", []) or []:
                    rn, _rs = _split_plugin_require(req)
                    if rn and rn in by_name and rn != name:
                        want.add(rn)
            deps[name] = want
        ordered, done = [], set()
        while len(done) < len(names):
            progressed = False
            for name in names:
                if name in done:
                    continue
                if deps[name] <= done:
                    ordered.append(name)
                    done.add(name)
                    progressed = True
            if not progressed:
                for name in names:
                    if name not in done:
                        ordered.append(name)
                        done.add(name)
                break
        out = []
        for name in ordered:
            out.extend(by_name[name])
        return out

    def _load_planned_entry(self, kind: str, path: str, manifest: dict):
        if manifest and not self._gate_or_defer(kind, path, manifest):
            return
        if kind == "zplugin":
            self.load_zplugin(path)
        else:
            self.load_package(path)

    def load_directory(self, dirpath: str):
        if not os.path.isdir(dirpath):
            return
        zplugins, packages, pyfiles = [], [], []
        for fname in sorted(os.listdir(dirpath)):
            fpath = os.path.join(dirpath, fname)
            if fname.endswith(".zplugin") and not fname.startswith("_"):
                zplugins.append(fpath)
            elif fname.endswith(".py") and not fname.startswith("_"):
                pyfiles.append(fpath)
            elif os.path.isdir(fpath):
                init_path = os.path.join(fpath, "__init__.py")
                if os.path.isfile(init_path):
                    packages.append(fpath)
        planned = []
        for path in zplugins:
            planned.append(("zplugin", path, _read_archive_manifest(path)))
        for d in packages:
            planned.append(("package", d, _read_package_manifest(d)))
        libs = [(k, p, m) for k, p, m in planned if m.get("type") == "library"]
        rest = [(k, p, m) for k, p, m in planned if m.get("type") != "library"]
        for kind, path, manifest in libs:
            self._load_planned_entry(kind, path, manifest)
        for kind, path, manifest in self._topo_plan(rest):
            self._load_planned_entry(kind, path, manifest)
        for path in pyfiles:
            self.load_from_file(path)
        self._drain_pending()

    def _activate_bundled_libs(self, libs_dir: str, deps: list, owner: str = ""):
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
        if owner:
            for mod in _scan_provides(libs_dir):
                self._record_lib_owner(mod, owner, libs_dir)
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
        ok, reason, manifest = self.verify_zplugin_file(path)
        if not ok:
            Logger.error(f"[zplugin] Rejected '{path}': {reason}")
            self._set_status(str(manifest.get("name", os.path.basename(path))), "blocked", reason)
            return {}
        if reason == "unsigned":
            Logger.warning(f"[zplugin] Loading unsigned package '{path}'")
        name = str(manifest.get("name", "unnamed"))
        version = str(manifest.get("version", "0.0.1"))
        dest = _resolve_zplugin(name, version)
        fingerprint = _zplugin_fingerprint(path)
        cached = self._read_zplugin_cache(dest, fingerprint)
        if cached is not None:
            return cached
        tmp = tempfile.mkdtemp(prefix="zpl_", dir=_zarin_user_dir("tmp"))
        try:
            return self._extract_zplugin_payload(path, manifest, name, tmp, dest, fingerprint)
        except Exception as e:
            Logger.error(f"[zplugin] Extraction failed for '{path}': {e}", e)
            self._set_status(name, "blocked", f"extraction failed: {e}")
            shutil.rmtree(tmp, ignore_errors=True)
            return {}

    def _extract_zplugin_payload(self, path: str, manifest: dict, name: str, tmp: str, dest: str,
                                 fingerprint: dict) -> dict:
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
        entry = manifest.get("entry", "") or ("plugin.py" if manifest.get("type") != "library" else "")
        entry_rel = None
        if entry:
            for n in safe_names:
                if n.endswith("/" + entry) or n == entry:
                    entry_rel = n
                    break
            if entry_rel is None and os.path.isfile(os.path.join(tmp, entry)):
                entry_rel = entry
        if entry_rel is None and manifest.get("type") != "library":
            Logger.error(f"[zplugin] Entry point '{entry}' not found in '{path}'")
            shutil.rmtree(tmp, ignore_errors=True)
            return {}
        if os.path.exists(dest):
            shutil.rmtree(dest, ignore_errors=True)
        shutil.move(tmp, dest)
        self._activate_bundled_libs(os.path.join(dest, "libs"), manifest.get("dependencies", []), name)
        self._compile_if_needed(dest, manifest)
        if isinstance(manifest.get("files"), dict):
            problems = self._verify_payload_files(dest, manifest)
            if problems:
                Logger.error(f"[zplugin] Payload verification failed for '{path}': {'; '.join(problems)}")
                self._set_status(name, "blocked", "; ".join(problems))
                shutil.rmtree(dest, ignore_errors=True)
                return {}
        merged = {k: v for k, v in manifest.items()}
        merged["payload_dir"] = dest
        merged["entry_file"] = os.path.join(dest, entry_rel) if entry_rel else ""
        merged["entry_rel"] = entry_rel or ""
        try:
            manifest_cache = {
                "__zpl_meta": True,
                "manifest": merged,
                "payload": dest,
                "fingerprint": fingerprint,
            }
            with open(os.path.join(dest, ".zpl_meta.json"), "w", encoding="utf-8") as f:
                json.dump(manifest_cache, f, indent=2)
        except Exception as e:
            Logger.warning(f"[zplugin] Could not write cache meta: {e}")
        return merged

    def _read_zplugin_cache(self, dest: str, fingerprint: dict) -> Optional[dict]:
        if not fingerprint or not os.path.isdir(dest):
            return None
        try:
            with open(os.path.join(dest, ".zpl_meta.json"), "r", encoding="utf-8") as f:
                cached = json.load(f)
        except Exception:
            return None
        if not isinstance(cached, dict) or cached.get("__zpl_meta") is not True:
            return None
        if cached.get("fingerprint") != fingerprint:
            return None
        stored = cached.get("manifest")
        if not isinstance(stored, dict):
            return None
        ok, _reason = self.verify_manifest_signature(stored)
        if stored.get("signature") and not ok:
            return None
        if isinstance(stored.get("files"), dict) and self._verify_payload_files(dest, stored):
            return None
        entry_rel = stored.get("entry_rel", "")
        full_entry = os.path.join(dest, entry_rel) if entry_rel else ""
        if entry_rel and not os.path.isfile(full_entry):
            return None
        self._activate_bundled_libs(os.path.join(dest, "libs"), stored.get("dependencies", []),
                                    str(stored.get("name", "")))
        self._compile_if_needed(dest, stored)
        merged = {k: v for k, v in stored.items()}
        merged["payload_dir"] = dest
        merged["entry_file"] = full_entry
        return merged

    def load_zplugin(self, path: str):
        try:
            pre = _read_archive_manifest(path)
            if pre and not self._gate_or_defer("zplugin", path, pre):
                return
            manifest = self._extract_zplugin(path)
            if not manifest or ("entry_file" not in manifest and manifest.get("type") != "library"):
                return
            entry_file = manifest.get("entry_file", "")
            payload_dir = manifest.get("payload_dir", "") or (os.path.dirname(entry_file) if entry_file else "")
            libs_dir = os.path.join(payload_dir, "libs")
            self._activate_bundled_libs(libs_dir, manifest.get("dependencies", []),
                                        str(manifest.get("name", "")))
            arch = manifest.get("architecture", "any")
            arch_ok = _platform_matches(arch)
            loaded = False
            pkg = manifest.get("module")
            if entry_file and pkg and entry_file.endswith(".py"):
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
            if entry_file and not loaded:
                if entry_file.endswith(".pyd") or entry_file.endswith(".so") or entry_file.endswith(".dll"):
                    self._load_python_plugin(os.path.splitext(os.path.basename(entry_file))[0],
                                             entry_file, libs_dir, manifest)
                    loaded = True
                elif entry_file.endswith(".py"):
                    self._load_python_plugin("_zplugin", entry_file, libs_dir, manifest)
                    loaded = True
            if not loaded and manifest.get("type") != "library":
                Logger.error(f"[zplugin] '{manifest.get('name')}' not loadable on {current_architecture()}")
            if manifest.get("type") == "library":
                self._register_library_archive(manifest, manifest.get("payload_dir", ""))
            name = manifest.get("name", "unknown")
            Logger.info(f"[zplugin] Loaded package '{name}' v{manifest.get('version', '')} from {os.path.basename(path)}")
        except Exception as e:
            Logger.error(f"Failed to load zplugin '{path}': {e}", e)

    def load_package(self, dirpath: str):
        try:
            basename = os.path.basename(dirpath)
            if basename.startswith("_"):
                return
            manifest = _read_package_manifest(dirpath)
            if manifest and not self._gate_or_defer("package", dirpath, manifest):
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
            if manifest.get("type") == "library":
                self._register_library_package(dirpath, manifest)
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
        for p in self.get_all():
            try:
                getattr(p, method_name)(*args)
            except Exception as e:
                Logger.error(f"Plugin {method_name} error: {e}", e)

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

    def notify_main_window_ready(self, window):
        self._notify_all("on_main_window_ready", window)

    def notify_play_start(self):
        self._notify_all("on_play_start")

    def notify_play_stop(self):
        self._notify_all("on_play_stop")
