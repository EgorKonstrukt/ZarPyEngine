# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Vec3, Quat

@ComponentRegistry.register
class XRSmoothMoveProvider(Component):
    _icon = "XRSmoothMoveProvider.png"
    _gizmo_icon_color = (120, 200, 255)
    _gizmo_icon_label = "M"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("enable_strafe", "Enable Strafe", FieldType.BOOL),
            InspectorField("enable_fly", "Enable Fly", FieldType.BOOL),
            InspectorField("thumbstick_deadzone", "Deadzone", FieldType.FLOAT, min_val=0.0, max_val=0.9, step=0.05, decimals=2),
        ]

    def __init__(self):
        super().__init__()
        self.enable_strafe: bool = True
        self.enable_fly: bool = True
        self.thumbstick_deadzone: float = 0.2

    def on_update(self, dt: float):
        return

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"enable_strafe": self.enable_strafe, "enable_fly": self.enable_fly, "thumbstick_deadzone": self.thumbstick_deadzone})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> XRSmoothMoveProvider:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.enable_strafe = bool(data.get("enable_strafe", True))
        c.enable_fly = bool(data.get("enable_fly", True))
        c.thumbstick_deadzone = float(data.get("thumbstick_deadzone", 0.2))
        return c


@ComponentRegistry.register
class XRSnapTurnProvider(Component):
    _icon = "XRSnapTurnProvider.png"
    _gizmo_icon_color = (255, 200, 120)
    _gizmo_icon_label = "S"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("snap_degrees", "Snap Degrees", FieldType.FLOAT, min_val=5.0, max_val=90.0, step=5.0, decimals=1),
            InspectorField("activation_button", "Activation", FieldType.ENUM, enum_options=["Left Thumbstick", "Right Thumbstick"]),
            InspectorField("enable_teleport", "Teleport On Trigger", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.snap_degrees: float = 30.0
        self.activation_button: str = "Right Thumbstick"
        self.enable_teleport: bool = True
        self._prev_snap: bool = False

    def on_update(self, dt: float):
        return

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"snap_degrees": self.snap_degrees, "activation_button": self.activation_button, "enable_teleport": self.enable_teleport})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> XRSnapTurnProvider:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.snap_degrees = float(data.get("snap_degrees", 30.0))
        c.activation_button = data.get("activation_button", "Right Thumbstick")
        c.enable_teleport = bool(data.get("enable_teleport", True))
        return c


@ComponentRegistry.register
class XRTeleportationProvider(Component):
    _icon = "XRTeleportationProvider.png"
    _gizmo_icon_color = (160, 255, 180)
    _gizmo_icon_label = "T"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("activation_button", "Activation", FieldType.ENUM, enum_options=["Left Trigger", "Right Trigger"]),
            InspectorField("teleport_height", "Teleport Height", FieldType.FLOAT, min_val=0.0, max_val=3.0, step=0.05, decimals=2),
            InspectorField("snap_to_floor", "Snap To Floor", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.activation_button: str = "Right Trigger"
        self.teleport_height: float = 0.0
        self.snap_to_floor: bool = True
        self._prev_teleport: bool = False

    def on_update(self, dt: float):
        return

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"activation_button": self.activation_button, "teleport_height": self.teleport_height, "snap_to_floor": self.snap_to_floor})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> XRTeleportationProvider:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.activation_button = data.get("activation_button", "Right Trigger")
        c.teleport_height = float(data.get("teleport_height", 0.0))
        c.snap_to_floor = bool(data.get("snap_to_floor", True))
        return c


def apply_locomotion_providers(scene, dt: float):
    rig_ent = None
    rig_comp = None
    snap = None
    teleport = None
    for e in scene.get_all_entities():
        if e.get_component(XRRig) is not None:
            rig_ent = e
            rig_comp = e.get_component(XRRig)
        if e.get_component(XRSnapTurnProvider) is not None:
            snap = e.get_component(XRSnapTurnProvider)
        if e.get_component(XRTeleportationProvider) is not None:
            teleport = e.get_component(XRTeleportationProvider)
    if rig_ent is None or rig_comp is None:
        return
    try:
        from plugins.vr_plugin import vr_core
    except Exception:
        return
    if not (vr_core.is_active() and vr_core.session_running()):
        return
    tr = rig_ent.transform
    if tr is None:
        return
    if snap is not None and snap.enabled:
        _apply_snap_turn(vr_core, rig_comp, rig_ent, tr, snap, dt)
    if teleport is not None and teleport.enabled:
        _apply_teleport(vr_core, rig_comp, rig_ent, tr, teleport, dt)


def _apply_snap_turn(vr_core, rig_comp, rig_ent, tr, snap, dt):
    idx = 0 if snap.activation_button.startswith("Left") else 1
    ax = vr_core._vr_state.controllers[idx].thumbstick[0]
    if abs(ax) > 0.7 and not snap._prev_snap:
        sign = 1.0 if ax > 0 else -1.0
        delta = math.radians(snap.snap_degrees) * sign
        rig_comp.rig_yaw -= delta
        try:
            from core.engine.engine import Engine
            eng = Engine.instance()
            if eng and eng.viewport and eng.viewport._cam:
                cam = eng.viewport._cam
                cam._yaw -= math.degrees(delta)
                if cam._yaw > 180.0:
                    cam._yaw -= 360.0
                if cam._yaw < -180.0:
                    cam._yaw += 360.0
        except Exception:
            pass
        try:
            tr.local_euler_angles = Vec3(0, math.degrees(rig_comp.rig_yaw), 0)
        except Exception:
            tr.local_rotation = Quat.from_euler(0, rig_comp.rig_yaw, 0)
        vr_core._vr_state.rig_yaw = rig_comp.rig_yaw
    snap._prev_snap = abs(ax) > 0.7


def _apply_teleport(vr_core, rig_comp, rig_ent, tr, teleport, dt):
    idx = 0 if teleport.activation_button.startswith("Left") else 1
    pressed = vr_core._vr_state.controllers[idx].trigger > 0.6
    if pressed and not teleport._prev_teleport:
        ray = vr_core.get_controller_ray(idx)
        if ray is not None:
            origin, fwd = ray
            tgt = _raycast_floor(vr_core, origin, fwd, teleport.snap_to_floor)
            if tgt is not None:
                wp = tr.position
                new_pos = Vec3(tgt[0], (teleport.teleport_height if teleport.snap_to_floor else tgt[1]), tgt[2])
                tr.position = new_pos
                rig_comp.velocity = Vec3.zero()
    teleport._prev_teleport = pressed


def _raycast_floor(vr_core, origin, fwd, use_ar):
    best, dist = vr_core.ar_raycast(origin, fwd, 100.0)
    if best is not None:
        return best[0]
    ent, hit, d = vr_core.raycast_scene(origin, fwd, 100.0)
    if ent is not None and hit is not None:
        return hit
    return None
