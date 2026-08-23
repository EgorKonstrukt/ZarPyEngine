# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Vec3, Quat

@ComponentRegistry.register
class ARSession(Component):
    _icon = "ARSession.png"
    _gizmo_icon_color = (120, 220, 200)
    _gizmo_icon_label = "AR"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("floor_y", "Floor Y", FieldType.FLOAT, min_val=-10.0, max_val=10.0, step=0.05, decimals=2),
            InspectorField("enable_automatic_passthrough", "Automatic Passthrough", FieldType.BOOL),
            InspectorField("match_viewport_orientation", "Match Viewport Orientation", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.floor_y: float = 0.0
        self.enable_automatic_passthrough: bool = True
        self.match_viewport_orientation: bool = True

    def on_update(self, dt: float):
        return

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"floor_y": self.floor_y, "enable_automatic_passthrough": self.enable_automatic_passthrough,
                  "match_viewport_orientation": self.match_viewport_orientation})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ARSession:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.floor_y = float(data.get("floor_y", 0.0))
        c.enable_automatic_passthrough = bool(data.get("enable_automatic_passthrough", True))
        c.match_viewport_orientation = bool(data.get("match_viewport_orientation", True))
        return c


@ComponentRegistry.register
class ARCameraBackground(Component):
    _icon = "ARCameraBackground.png"
    _gizmo_icon_color = (180, 220, 255)
    _gizmo_icon_label = "BG"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("use_passthrough", "Use Passthrough", FieldType.BOOL),
            InspectorField("occlusion", "Occlusion", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.use_passthrough: bool = True
        self.occlusion: bool = True

    def on_update(self, dt: float):
        return

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"use_passthrough": self.use_passthrough, "occlusion": self.occlusion})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ARCameraBackground:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.use_passthrough = bool(data.get("use_passthrough", True))
        c.occlusion = bool(data.get("occlusion", True))
        return c


@ComponentRegistry.register
class ARPlaneManager(Component):
    _icon = "ARPlaneManager.png"
    _gizmo_icon_color = (200, 200, 160)
    _gizmo_icon_label = "PL"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("detected_planes", "Detected Planes", FieldType.ENUM, enum_options=["Nothing", "Horizontal", "Vertical", "Everything"]),
            InspectorField("spawn_plane_prefab", "Spawn Plane Prefab", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.detected_planes: str = "Everything"
        self.spawn_plane_prefab: bool = True

    def on_update(self, dt: float):
        return

    def get_planes(self):
        try:
            from plugins.vr_plugin import vr_core
            return vr_core.get_ar_planes()
        except Exception:
            return []

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"detected_planes": self.detected_planes, "spawn_plane_prefab": self.spawn_plane_prefab})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ARPlaneManager:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.detected_planes = data.get("detected_planes", "Everything")
        c.spawn_plane_prefab = bool(data.get("spawn_plane_prefab", True))
        return c


@ComponentRegistry.register
class ARRaycastManager(Component):
    _icon = "ARRaycastManager.png"
    _gizmo_icon_color = (200, 180, 255)
    _gizmo_icon_label = "RC"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("max_raycast_distance", "Max Distance", FieldType.FLOAT, min_val=1.0, max_val=100.0, step=1.0, decimals=1),
        ]

    def __init__(self):
        super().__init__()
        self.max_raycast_distance: float = 30.0

    def on_update(self, dt: float):
        return

    def Raycast(self, origin, direction):
        try:
            from plugins.vr_plugin import vr_core
            return vr_core.ar_raycast(origin, direction, self.max_raycast_distance)
        except Exception:
            return (None, float('inf'))

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"max_raycast_distance": self.max_raycast_distance})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ARRaycastManager:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.max_raycast_distance = float(data.get("max_raycast_distance", 30.0))
        return c


@ComponentRegistry.register
class ARAnchorManager(Component):
    _icon = "ARAnchorManager.png"
    _gizmo_icon_color = (255, 200, 200)
    _gizmo_icon_label = "AN"
    _show_gizmo_icon = True

    def __init__(self):
        super().__init__()
        self._anchors: list = []

    def on_update(self, dt: float):
        return

    def AddAnchor(self, position, orientation=(0.0, 0.0, 0.0, 1.0), trackable_id=None):
        try:
            from plugins.vr_plugin import vr_core
            aid = vr_core.add_ar_anchor(position, orientation, trackable_id)
            if aid is not None:
                self._anchors.append(aid)
            return aid
        except Exception:
            return None

    def RemoveAnchor(self, anchor_id):
        try:
            from plugins.vr_plugin import vr_core
            vr_core.remove_ar_anchor(anchor_id)
            self._anchors = [a for a in self._anchors if a != anchor_id]
        except Exception:
            pass

    def GetAllAnchors(self):
        try:
            from plugins.vr_plugin import vr_core
            return vr_core.get_ar_anchors()
        except Exception:
            return []

    def serialize(self) -> dict:
        d = super().serialize()
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ARAnchorManager:
        c = cls()
        c.enabled = data.get("enabled", True)
        return c


@ComponentRegistry.register
class ARPointCloudManager(Component):
    _icon = "ARPointCloudManager.png"
    _gizmo_icon_color = (200, 230, 160)
    _gizmo_icon_label = "PC"
    _show_gizmo_icon = True

    def __init__(self):
        super().__init__()
        self._confidence_threshold: float = 0.5

    def on_update(self, dt: float):
        return

    def GetPointCloud(self):
        try:
            from plugins.vr_plugin import vr_core
            return vr_core.get_ar_point_cloud()
        except Exception:
            return []

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"confidence_threshold": self._confidence_threshold})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ARPointCloudManager:
        c = cls()
        c.enabled = data.get("enabled", True)
        c._confidence_threshold = float(data.get("confidence_threshold", 0.5))
        return c
