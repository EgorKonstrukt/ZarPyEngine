# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun
#
# DirectionalShadow owns the settings for directional/sun shadow maps (CSM).
# If no active DirectionalShadow component is present, directional shadows are
# disabled for the scene.

from __future__ import annotations

from typing import Optional

from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField

_DEFAULT_SPLITS_FOR = {2: [0.5], 3: [0.2, 0.6], 4: [0.05, 0.13, 0.3]}


@ComponentRegistry.register
class DirectionalShadow(Component):
    _registry: list[DirectionalShadow] = []
    _allow_multiple = False
    _gizmo_icon_label = "DSH"
    _gizmo_icon_color = (150, 130, 70)

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("enabled", "Enabled", FieldType.BOOL),
            InspectorField("_shadow_resolution", "Resolution", FieldType.INT_SLIDER,
                           min_val=256, max_val=4096, step=256, decimals=0),
            InspectorField("_shadow_distance", "Shadow Distance", FieldType.SLIDER,
                           min_val=1.0, max_val=500.0, step=1.0, decimals=0),
            InspectorField("_cascade_count", "Cascades", FieldType.INT_SLIDER,
                           min_val=1, max_val=4, step=1, decimals=0),
            InspectorField("_cascade_splits", "Cascade Boundaries", FieldType.SPLIT_GRADIENT,
                           min_val=50.0, max_val=500.0, step=1.0, decimals=0),
        ]

    def __init__(self):
        super().__init__()
        self._shadow_resolution: int = 4096
        self._shadow_distance: float = 50.0
        self._cascade_splits: list = list(_DEFAULT_SPLITS_FOR[4])
        self._set_cascade_count(4)

    @property
    def _cascade_count(self) -> int:
        return self.__cc

    @_cascade_count.setter
    def _cascade_count(self, value: int):
        self._set_cascade_count(value)

    def _set_cascade_count(self, value: int) -> int:
        count = max(1, min(int(value), 4))
        self.__cc = count
        n = count - 1
        current = list(self._cascade_splits or [])
        defaults = _DEFAULT_SPLITS_FOR.get(count, [])
        current = [min(1.0, max(0.001, float(x))) for x in current]
        out = []
        for i in range(n):
            if i < len(current):
                out.append(current[i])
            elif i < len(defaults):
                out.append(defaults[i])
            else:
                out.append(1.0 if n > 0 else 0.0)
        self._cascade_splits = sorted(out)
        return count

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "_shadow_resolution": self._shadow_resolution,
            "_shadow_distance": self._shadow_distance,
            "_cascade_count": self._cascade_count,
            "_cascade_splits": list(self._cascade_splits),
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> DirectionalShadow:
        inst = super().deserialize(data)
        inst._shadow_resolution = int(data.get("_shadow_resolution", 4096))
        inst._shadow_distance = float(data.get("_shadow_distance", 50.0))
        inst._cascade_count = int(data.get("_cascade_count", 4))
        inst._cascade_splits = list(data.get("_cascade_splits", list(_DEFAULT_SPLITS_FOR[inst._cascade_count])))
        return inst

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
    def find_active(cls) -> Optional[DirectionalShadow]:
        return next((s for s in cls._registry if s.enabled), None)
