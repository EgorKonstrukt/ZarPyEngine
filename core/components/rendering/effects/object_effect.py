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

    def bind(self, prog, time_s: float):
        self._set(prog, "u_time", time_s - self._time_offset)
        self._apply(prog)

    def _apply(self, prog):
        pass
