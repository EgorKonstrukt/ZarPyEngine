# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun
#
# SpotShadow owns the settings for spot-light shadow maps.
# If no active SpotShadow component is present, spot shadows are disabled for
# the scene.

from __future__ import annotations

from typing import Optional

from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField


@ComponentRegistry.register
class SpotShadow(Component):
    _registry: list[SpotShadow] = []
    _allow_multiple = False
    _gizmo_icon_label = "SSH"
    _gizmo_icon_color = (120, 150, 90)

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("enabled", "Enabled", FieldType.BOOL),
            InspectorField("_shadow_resolution", "Resolution", FieldType.INT_SLIDER,
                           min_val=128, max_val=2048, step=128, decimals=0),
        ]

    def __init__(self):
        super().__init__()
        self._shadow_resolution: int = 1024

    def serialize(self) -> dict:
        d = super().serialize()
        d["_shadow_resolution"] = self._shadow_resolution
        return d

    @classmethod
    def deserialize(cls, data: dict) -> SpotShadow:
        inst = super().deserialize(data)
        inst._shadow_resolution = int(data.get("_shadow_resolution", 1024))
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
    def find_active(cls) -> Optional[SpotShadow]:
        return next((s for s in cls._registry if s.enabled), None)
