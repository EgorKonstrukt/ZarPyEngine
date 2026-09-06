# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import importlib
import sys
from typing import Any, Callable, Optional

from core.foundation.logger import Logger


class _InspectorFieldDriver:
    def __init__(self):
        self._append: dict[type, list] = {}
        self._filter_ext: dict[type, dict[str, str]] = {}
        self._wrapped: set[type] = set()

    def add_field(self, comp_cls: type, field):
        self._append.setdefault(comp_cls, []).append(field)
        self._ensure_wrapper(comp_cls)

    def extend_filter(self, comp_cls: type, field_name: str, extra_filter: str):
        self._filter_ext.setdefault(comp_cls, {})[field_name] = extra_filter
        self._ensure_wrapper(comp_cls)

    def _ensure_wrapper(self, comp_cls: type):
        if comp_cls in self._wrapped:
            return
        original = getattr(comp_cls, "_inspector_fields", None)
        if original is None:
            return
        self._wrapped.add(comp_cls)

        def wrapped(cls=None):
            fields = []
            if original is not None:
                base = original()
                if base is not None:
                    fields = list(base)
            for field in self._append.get(comp_cls, []):
                fields.append(field)
            fmap = self._filter_ext.get(comp_cls, {})
            if fmap:
                for field in fields:
                    fname = getattr(field, "name", None)
                    if fname in fmap and hasattr(field, "file_filter"):
                        cur = field.file_filter or ""
                        if fmap[fname] not in cur:
                            field.file_filter = (cur + ";;" + fmap[fname]).strip(";")
            return fields

        setattr(comp_cls, "_inspector_fields", classmethod(wrapped))

    def restore(self, comp_cls: type):
        if comp_cls in self._wrapped:
            self._wrapped.discard(comp_cls)
            self._append.pop(comp_cls, None)
            self._filter_ext.pop(comp_cls, None)


_component_patcher = _InspectorFieldDriver()


class ComponentPatcher:
    def __init__(self):
        self._method_backups: dict[type, dict[str, Any]] = {}
        self._inspector_mods: set[type] = set()

    def patch_method(self, comp_cls: type, method_name: str, replacement: Callable):
        if not hasattr(comp_cls, method_name):
            Logger.warning(f"[Patcher] Component {comp_cls.__name__} has no '{method_name}'")
            return
        backups = self._method_backups.setdefault(comp_cls, {})
        if method_name not in backups:
            backups[method_name] = getattr(comp_cls, method_name)
        setattr(comp_cls, method_name, replacement)

    def wrap_method(self, comp_cls: type, method_name: str,
                    before: Optional[Callable] = None, after: Optional[Callable] = None):
        original = getattr(comp_cls, method_name, None)
        if original is None:
            return
        backups = self._method_backups.setdefault(comp_cls, {})
        if method_name not in backups:
            backups[method_name] = original

        def wrapper(*args, **kwargs):
            result_slot = []
            proceed = True
            if before is not None:
                try:
                    r = before(*args, **kwargs)
                    if r is not None and r is not True:
                        result_slot.append(r)
                        proceed = False
                except Exception as e:
                    Logger.error(f"[Patcher] before hook error on {comp_cls.__name__}.{method_name}: {e}", e)
            if proceed:
                result = original(*args, **kwargs)
                if after is not None:
                    try:
                        after(*args, **kwargs)
                    except Exception as e:
                        Logger.error(f"[Patcher] after hook error on {comp_cls.__name__}.{method_name}: {e}", e)
                return result
            return result_slot[0] if result_slot else None

        setattr(comp_cls, method_name, wrapper)

    def add_inspector_field(self, comp_cls: type, field):
        _component_patcher.add_field(comp_cls, field)
        self._inspector_mods.add(comp_cls)

    def extend_file_filter(self, comp_cls: type, field_name: str, extra_filter: str):
        _component_patcher.extend_filter(comp_cls, field_name, extra_filter)
        self._inspector_mods.add(comp_cls)

    def restore(self):
        for comp_cls, backups in self._method_backups.items():
            for name, original in backups.items():
                try:
                    setattr(comp_cls, name, original)
                except Exception as e:
                    Logger.error(f"[Patcher] restore error on {comp_cls.__name__}.{name}: {e}")
        self._method_backups.clear()
        for comp_cls in self._inspector_mods:
            try:
                _component_patcher.restore(comp_cls)
            except Exception:
                pass
        self._inspector_mods.clear()


def patch_component(component_name: str, method_patches: Optional[dict[str, Callable]] = None,
                    extra_inspector_fields: Optional[list] = None,
                    filter_extensions: Optional[dict[str, str]] = None,
                    wraps: Optional[dict[str, dict]] = None) -> ComponentPatcher:
    from core.ecs.ecs import ComponentRegistry
    comp_cls = ComponentRegistry.get(component_name)
    if comp_cls is None:
        Logger.warning(f"[Patcher] Unknown component '{component_name}'")
        return ComponentPatcher()
    patcher = ComponentPatcher()
    if method_patches:
        for name, replacement in method_patches.items():
            patcher.patch_method(comp_cls, name, replacement)
    if wraps:
        for name, hooks in wraps.items():
            patcher.wrap_method(comp_cls, name, hooks.get("before"), hooks.get("after"))
    if extra_inspector_fields:
        for field in extra_inspector_fields:
            patcher.add_inspector_field(comp_cls, field)
    if filter_extensions:
        for fname, extra in filter_extensions.items():
            patcher.extend_file_filter(comp_cls, fname, extra)
    return patcher


class FunctionPatcher:
    def __init__(self):
        self._module_backups: dict[object, dict[str, Any]] = {}

    def patch(self, module: Any, func_name: str, wrapper: Callable):
        if isinstance(module, str):
            try:
                module = importlib.import_module(module)
            except Exception as e:
                Logger.error(f"[FunctionPatcher] Cannot import module '{module}': {e}", e)
                return
        if module is None or not hasattr(module, func_name):
            Logger.warning(f"[FunctionPatcher] Module has no '{func_name}'")
            return
        backups = self._module_backups.setdefault(module, {})
        if func_name not in backups:
            backups[func_name] = getattr(module, func_name)
        setattr(module, func_name, wrapper)

    def restore(self):
        for module, backups in self._module_backups.items():
            for name, original in backups.items():
                try:
                    setattr(module, name, original)
                except Exception as e:
                    Logger.error(f"[FunctionPatcher] restore error on {getattr(module, '__name__', '?')}.{name}: {e}")
        self._module_backups.clear()


class PatchTracker:
    def __init__(self):
        self._component_patchers: list[ComponentPatcher] = []
        self._function_patchers: list[FunctionPatcher] = []
        self._unregister_hooks: list[Callable] = []

    def patch_component(self, *args, **kwargs) -> ComponentPatcher:
        patcher = patch_component(*args, **kwargs)
        self._component_patchers.append(patcher)
        return patcher

    def add_inspector_field(self, component_name: str, field):
        from core.ecs.ecs import ComponentRegistry
        comp_cls = ComponentRegistry.get(component_name)
        if comp_cls is None:
            Logger.warning(f"[PatchTracker] Unknown component '{component_name}'")
            return None
        patcher = ComponentPatcher()
        patcher.add_inspector_field(comp_cls, field)
        self._component_patchers.append(patcher)
        return patcher

    def extend_file_filter(self, component_name: str, field_name: str, extra_filter: str):
        from core.ecs.ecs import ComponentRegistry
        comp_cls = ComponentRegistry.get(component_name)
        if comp_cls is None:
            Logger.warning(f"[PatchTracker] Unknown component '{component_name}'")
            return None
        patcher = ComponentPatcher()
        patcher.extend_file_filter(comp_cls, field_name, extra_filter)
        self._component_patchers.append(patcher)
        return patcher

    def patch_function(self, module: Any, func_name: str, wrapper: Callable):
        patcher = FunctionPatcher()
        patcher.patch(module, func_name, wrapper)
        self._function_patchers.append(patcher)

    def adopt(self, patcher: ComponentPatcher):
        self._component_patchers.append(patcher)

    def register_component(self, comp_cls: type):
        def unregister():
            from core.ecs.ecs import ComponentRegistry
            name = getattr(comp_cls, "__name__", None)
            if name:
                ComponentRegistry._registry.pop(name, None)
                ComponentRegistry._categories.pop(name, None)
        self._unregister_hooks.append(unregister)

    def restore_all(self):
        for patcher in reversed(self._component_patchers):
            try:
                patcher.restore()
            except Exception:
                pass
        for patcher in reversed(self._function_patchers):
            try:
                patcher.restore()
            except Exception:
                pass
        for hook in reversed(self._unregister_hooks):
            try:
                hook()
            except Exception:
                pass
        self._component_patchers.clear()
        self._function_patchers.clear()
        self._unregister_hooks.clear()