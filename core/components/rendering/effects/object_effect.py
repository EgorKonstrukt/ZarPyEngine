# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
import numpy as np
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField


@ComponentRegistry.register
class ObjectEffect(Component):
    _icon = "ObjectEffect.png"
    _gizmo_icon_color = (255, 140, 30)
    _gizmo_icon_label = "FX"
    _allow_multiple = True

    _registry: list[ObjectEffect] = []

    fx_uniform_defaults: dict = {}

    def __init__(self):
        super().__init__()
        self._time_offset: float = 0.0

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return []

    def on_awake(self):
        if self not in self._registry:
            self._registry.append(self)

    def on_destroy(self):
        if self in self._registry:
            self._registry.remove(self)

    def on_disable(self):
        if self in self._registry:
            self._registry.remove(self)

    def on_enable(self):
        if self not in self._registry:
            self._registry.append(self)

    @classmethod
    def cleanup_registry(cls):
        for fx in list(cls._registry):
            try:
                fx.on_destroy()
            except Exception:
                pass
        cls._registry.clear()

    @classmethod
    def fx_geometry_shader(cls) -> "str | None":
        return None

    @classmethod
    def fx_fragment_uniforms(cls) -> str:
        return ""

    @classmethod
    def fx_fragment_snippet(cls) -> str:
        return ""

    def _set(self, prog, name: str, value):
        try:
            if name in prog:
                prog[name].value = value
        except Exception:
            pass

    def _set_vec(self, prog, name: str, arr):
        try:
            if name in prog:
                prog[name].write(np.array(arr, dtype=np.float32).tobytes())
        except Exception:
            pass

    def _set_vec_bytes(self, prog, name: str, arr: np.ndarray):
        try:
            if name in prog:
                prog[name].write(arr.tobytes())
        except Exception:
            pass

    @classmethod
    def reset_defaults(cls, prog):
        for name, val in cls.fx_uniform_defaults.items():
            try:
                if name in prog:
                    if isinstance(val, (list, tuple, np.ndarray)):
                        prog[name].write(np.array(val, dtype=np.float32).tobytes())
                    else:
                        prog[name].value = val
            except Exception:
                pass

    @classmethod
    def reset_all_defaults(cls, prog):
        for sub in cls.__subclasses__():
            try:
                sub.reset_defaults(prog)
            except Exception:
                pass

    def bind(self, prog, time_s: float):
        self._set(prog, "u_time", time_s - self._time_offset)
        self._apply(prog)

    def _apply(self, prog):
        pass
