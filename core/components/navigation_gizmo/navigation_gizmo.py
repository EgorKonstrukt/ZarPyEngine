# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from enum import Enum

from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField


class CoordinateSystem(Enum):
    XYZ = 0
    YZX = 1
    ZXY = 2
    XZY = 3
    YXZ = 4
    ZYX = 5


@ComponentRegistry.register
class NavigationGizmo(Component):
    _editor_hidden: bool = True
    _show_gizmo_icon: bool = False
    _gizmo_pass: str = ""

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("rect_size", "Rect Size", FieldType.FLOAT, min_val=40.0, max_val=400.0, step=1.0),
            InspectorField("corner_offset_x", "Corner Offset X", FieldType.FLOAT, step=1.0),
            InspectorField("corner_offset_y", "Corner Offset Y", FieldType.FLOAT, step=1.0),
            InspectorField("pivot_distance", "Pivot Distance", FieldType.FLOAT, min_val=0.0, step=0.1),
            InspectorField("drag_enabled", "Drag To Orbit", FieldType.BOOL),
            InspectorField("click_enabled", "Click To Snap", FieldType.BOOL),
            InspectorField("drag_sensitivity", "Drag Sensitivity", FieldType.FLOAT, min_val=0.001, max_val=0.1, step=0.001, decimals=3),
            InspectorField("coordinate_system", "Coordinate System", FieldType.ENUM, enum_class=CoordinateSystem),
        ]

    def __init__(self):
        super().__init__()
        self.rect_size: float = 100.0
        self.corner_offset_x: float = 12.0
        self.corner_offset_y: float = 12.0
        self.pivot_distance: float = 5.0
        self.drag_enabled: bool = True
        self.click_enabled: bool = True
        self.drag_sensitivity: float = 0.01
        self.coordinate_system: CoordinateSystem = CoordinateSystem.XYZ
