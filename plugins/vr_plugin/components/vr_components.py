# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import numpy as np
from core.ecs.ecs import Component, ComponentRegistry, Entity, Scene
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Vec3, Quat

@ComponentRegistry.register
class VR(Component):
    _update_always = True
    _icon = "VR.png"
    _gizmo_icon_color = (80, 180, 255)
    _gizmo_icon_label = "VR"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("move_speed", "Move Speed", FieldType.FLOAT, min_val=0.1, max_val=20.0, step=0.1, decimals=2),
            InspectorField("turn_speed", "Turn Speed", FieldType.FLOAT, min_val=1.0, max_val=180.0, step=1.0, decimals=1),
            InspectorField("vert_speed", "Vertical Speed", FieldType.FLOAT, min_val=0.1, max_val=20.0, step=0.1, decimals=2),
            InspectorField("acceleration", "Acceleration", FieldType.FLOAT, min_val=0.1, max_val=50.0, step=0.5, decimals=2),
            InspectorField("damping", "Damping", FieldType.FLOAT, min_val=0.1, max_val=20.0, step=0.1, decimals=2),
            InspectorField("ipd", "IPD", FieldType.FLOAT, min_val=0.01, max_val=0.2, step=0.001, decimals=3),
        ]

    def __init__(self):
        super().__init__()
        self.move_speed: float = 5.0
        self.turn_speed: float = 80.0
        self.vert_speed: float = 2.5
        self.acceleration: float = 12.0
        self.damping: float = 8.0
        self.ipd: float = 0.063
        self.rig_yaw: float = 0.0
        self.velocity: Vec3 = Vec3.zero()
        self._is_active: bool = False

    def on_update(self, dt: float):
        try:
            from plugins.vr_plugin import vr_core
            if not vr_core.is_active():
                return
            from plugins.vr_plugin import vr_core
            if not vr_core.is_active() or not vr_core.session_running():
                vel = self.velocity
                if vel.length() > 0.001:
                    damp = math.exp(-self.damping * dt)
                    vel = vel * damp
                    self.velocity = vel
                    tr = self.transform
                    if tr:
                        tr.position = tr.position + vel * dt
                return
            left = vr_core._vr_state.controllers[0]
            right = vr_core._vr_state.controllers[1]
            lx, ly = left.thumbstick
            rx, ry = right.thumbstick
            dz = vr_core.VR_DEADZONE
            def dead(v):
                if abs(v) < dz:
                    return 0.0
                s = (abs(v)-dz)/(1.0-dz)
                return math.copysign(s, v)
            lx = dead(lx)
            ly = dead(ly)
            rx = dead(rx)
            ry = dead(ry)
            hmd_q = vr_core._controller_world_quat(vr_core._vr_state._hmd_quat)
            hmd_fwd = vr_core._quat_to_fwd(*hmd_q)
            fwd_h = (hmd_fwd[0], 0.0, hmd_fwd[2])
            flen = math.sqrt(fwd_h[0]*fwd_h[0] + fwd_h[2]*fwd_h[2])
            if flen > 1e-6:
                fwd_h = (fwd_h[0]/flen, 0.0, fwd_h[2]/flen)
            else:
                fwd_h = (0.0, 0.0, -1.0)
            right_v = (-fwd_h[2], 0.0, fwd_h[0])
            speed = float(self.move_speed)
            try:
                from core.engine.engine import Engine
                eng = Engine.instance()
                if eng and eng.viewport and hasattr(eng.viewport, '_cam'):
                    cam = eng.viewport._cam
                    speed = float(getattr(cam, '_move_speed', speed))
                    self.move_speed = speed
                    self.acceleration = float(getattr(cam, '_acceleration', self.acceleration))
                    self.damping = float(getattr(cam, '_damping', self.damping))
            except Exception:
                pass
            desired_x = 0.0
            desired_z = 0.0
            desired_y = 0.0
            if abs(lx) > 1e-6 or abs(ly) > 1e-6:
                desired_x = (right_v[0]*lx + fwd_h[0]*ly) * speed
                desired_z = (right_v[2]*lx + fwd_h[2]*ly) * speed
            if abs(ry) > 1e-6:
                desired_y = ry * speed
            tr = self.transform
            if not tr:
                return
            vel = self.velocity
            accel = self.acceleration
            lerp = min(1.0, dt * accel)
            desired = Vec3(desired_x, desired_y, desired_z)
            vel = vel + (desired - vel) * lerp
            if desired.length() < 1e-6:
                damp_factor = math.exp(-self.damping * dt)
                vel = vel * damp_factor
                if vel.length() < 0.001:
                    vel = Vec3.zero()
            self.velocity = vel
            tr.position = tr.position + vel * dt
            try:
                ipd_s = vr_core.get_ipd() / 0.063
                tr.local_scale = Vec3(ipd_s, ipd_s, ipd_s)
            except Exception:
                pass
            if abs(rx) > 1e-6:
                delta = rx * self.turn_speed * dt
                self.rig_yaw -= math.radians(delta)
                if self.rig_yaw > math.pi:
                    self.rig_yaw -= 2*math.pi
                if self.rig_yaw < -math.pi:
                    self.rig_yaw += 2*math.pi
                try:
                    eul = tr.local_euler_angles
                    tr.local_euler_angles = Vec3(eul.x, math.degrees(self.rig_yaw), eul.z)
                except Exception:
                    tr.local_rotation = Quat.from_euler(0.0, self.rig_yaw, 0.0)
                try:
                    from plugins.vr_plugin import vr_core as _vc2
                    _vc2._vr_state.rig_yaw = self.rig_yaw
                    if hasattr(_vc2, '_vr_state'):
                        pass
                    eng = None
                    try:
                        from core.engine.engine import Engine
                        eng = Engine.instance()
                        if eng and eng.viewport:
                            eng.viewport._cam._yaw -= delta
                            if eng.viewport._cam._yaw > 180.0:
                                eng.viewport._cam._yaw -= 360.0
                            if eng.viewport._cam._yaw < -180.0:
                                eng.viewport._cam._yaw += 360.0
                    except Exception:
                        pass
                except Exception:
                    pass
            else:
                try:
                    from plugins.vr_plugin import vr_core as _vc3
                    _vc3._vr_state.rig_yaw = self.rig_yaw
                except Exception:
                    pass
        except Exception:
            pass

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "move_speed": self.move_speed,
            "turn_speed": self.turn_speed,
            "vert_speed": self.vert_speed,
            "acceleration": self.acceleration,
            "damping": self.damping,
            "ipd": self.ipd,
            "rig_yaw": self.rig_yaw,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> VR:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.move_speed = float(data.get("move_speed", 5.0))
        c.turn_speed = float(data.get("turn_speed", 80.0))
        c.vert_speed = float(data.get("vert_speed", 2.5))
        c.acceleration = float(data.get("acceleration", 12.0))
        c.damping = float(data.get("damping", 8.0))
        c.ipd = float(data.get("ipd", 0.063))
        c.rig_yaw = float(data.get("rig_yaw", 0.0))
        return c

@ComponentRegistry.register
class VRHead(Component):
    _update_always = True
    _icon = "VRHead.png"
    _gizmo_icon_color = (255, 200, 80)
    _gizmo_icon_label = "H"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("show_mesh", "Show Mesh", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.show_mesh: bool = True

    def on_update(self, dt: float):
        try:
            from plugins.vr_plugin import vr_core
            tr = self.transform
            if not tr:
                return
            is_active = vr_core.is_active() and vr_core.session_running()
            if not is_active:
                try:
                    parent = self.entity.parent
                    if parent:
                        tr.local_position = Vec3(0.0, 1.6, 0.0)
                    else:
                        tr.position = Vec3(0.0, 1.6, 0.0)
                except Exception:
                    pass
                return
            pos = vr_core.get_hmd_world_pos()
            quat = vr_core.get_hmd_world_quat()
            tr.position = Vec3(pos[0], pos[1], pos[2])
            tr.local_rotation = Quat(quat[0], quat[1], quat[2], quat[3])
            try:
                vr_core._vr_state.ipd_override = float(getattr(self.entity.get_component(VR), 'ipd', vr_core.get_ipd())) if self.entity.get_component(VR) else vr_core.get_ipd()
            except Exception:
                pass
        except Exception:
            pass

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"show_mesh": self.show_mesh})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> VRHead:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.show_mesh = bool(data.get("show_mesh", True))
        return c

@ComponentRegistry.register
class VRController(Component):
    _update_always = True
    _icon = "VRController.png"
    _gizmo_icon_color = (80, 200, 255)
    _gizmo_icon_label = "C"
    _show_gizmo_icon = True
    _allow_multiple = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("hand", "Hand", FieldType.ENUM, enum_options=["Left", "Right"]),
            InspectorField("show_ray", "Show Ray", FieldType.BOOL),
            InspectorField("ray_length", "Ray Length", FieldType.FLOAT, min_val=0.5, max_val=100.0, step=0.5, decimals=1),
            InspectorField("trigger_threshold", "Trigger Threshold", FieldType.FLOAT, min_val=0.1, max_val=1.0, step=0.05, decimals=2),
        ]

    def __init__(self, hand: str = "Left"):
        super().__init__()
        self.hand: str = hand
        self.show_ray: bool = True
        self.ray_length: float = 10.0
        self.trigger_threshold: float = 0.6
        self._hovered: object = None

    def on_update(self, dt: float):
        try:
            from plugins.vr_plugin import vr_core
            tr = self.transform
            if not tr:
                return
            is_active = vr_core.is_active() and vr_core.session_running()
            if not is_active:
                off = Vec3(-0.25, 1.2, -0.2) if self.hand == "Left" else Vec3(0.25, 1.2, -0.2)
                try:
                    parent = self.entity.parent
                    if parent:
                        tr.local_position = off
                    else:
                        tr.position = off
                except Exception:
                    pass
                return
            idx = 0 if self.hand == "Left" else 1
            pos = vr_core.get_controller_world_pos(idx)
            quat = vr_core.get_controller_world_quat(idx)
            if pos is not None:
                tr.position = Vec3(pos[0], pos[1], pos[2])
            if quat is not None:
                tr.local_rotation = Quat(quat[0], quat[1], quat[2], quat[3])
        except Exception:
            pass

    def gizmo_lines(self):
        try:
            from plugins.vr_plugin import vr_core
            if not self.show_ray or not vr_core.is_active():
                return []
            idx = 0 if self.hand == "Left" else 1
            ray = vr_core.get_controller_ray(idx)
            if ray is None:
                return []
            origin, fwd = ray
            ent, hit, dist = vr_core.raycast_scene(origin, fwd)
            has_hit = ent is not None and hit is not None
            length = dist if has_hit and dist < self.ray_length else self.ray_length
            end = (origin[0]+fwd[0]*length, origin[1]+fwd[1]*length, origin[2]+fwd[2]*length)
            s = Vec3(origin[0], origin[1], origin[2])
            e = Vec3(end[0], end[1], end[2])
            col = [1.0, 0.95, 0.2, 1.0] if has_hit else ([0.2, 0.85, 1.0, 1.0] if self.hand=="Left" else [1.0, 0.35, 0.35, 1.0])
            return [(s, e, col)]
        except Exception:
            return []

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"hand": self.hand, "show_ray": self.show_ray, "ray_length": self.ray_length, "trigger_threshold": self.trigger_threshold})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> VRController:
        c = cls(hand=data.get("hand", "Left"))
        c.enabled = data.get("enabled", True)
        c.show_ray = bool(data.get("show_ray", True))
        c.ray_length = float(data.get("ray_length", 10.0))
        c.trigger_threshold = float(data.get("trigger_threshold", 0.6))
        return c

@ComponentRegistry.register
class VRLeftController(VRController):
    def __init__(self):
        super().__init__(hand="Left")

@ComponentRegistry.register
class VRRightController(VRController):
    def __init__(self):
        super().__init__(hand="Right")

@ComponentRegistry.register
class VRRay(Component):
    _icon = "VRRay.png"
    _gizmo_icon_color = (255, 255, 80)
    _gizmo_icon_label = "R"
    _show_gizmo_icon = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("length", "Length", FieldType.FLOAT, min_val=0.5, max_val=100.0, step=0.5, decimals=1),
            InspectorField("show_on_hover", "Show On Hover", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.length: float = 10.0
        self.show_on_hover: bool = True

    def gizmo_lines(self):
        try:
            from plugins.vr_plugin import vr_core
            ent = self.entity
            if not ent:
                return []
            ctrl = ent.get_component(VRController)
            if not ctrl:
                ctrl = ent.get_component(VRLeftController)
            if not ctrl:
                ctrl = ent.get_component(VRRightController)
            if not ctrl:
                return []
            return ctrl.gizmo_lines()
        except Exception:
            return []

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"length": self.length, "show_on_hover": self.show_on_hover})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> VRRay:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.length = float(data.get("length", 10.0))
        c.show_on_hover = bool(data.get("show_on_hover", True))
        return c

@ComponentRegistry.register
class VRSelection(Component):
    _icon = "VRSelection.png"
    _gizmo_icon_color = (255, 80, 80)
    _gizmo_icon_label = "S"
    _show_gizmo_icon = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("hover_color", "Hover Color", FieldType.COLOR),
            InspectorField("select_on_trigger", "Select On Trigger", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.hover_color: list[float] = [1.0, 0.95, 0.2, 1.0]
        self.select_on_trigger: bool = True
        self._hovered_entity = None

    def on_update(self, dt: float):
        try:
            from plugins.vr_plugin import vr_core
            if not vr_core.is_active():
                return
            from plugins.vr_plugin import vr_core
            if not vr_core.is_active() or not self.select_on_trigger:
                return
            ent = self.entity
            if not ent:
                return
            ctrl = None
            idx = -1
            c = ent.get_component(VRController)
            if c:
                idx = 0 if c.hand == "Left" else 1
                ctrl = c
            else:
                cl = ent.get_component(VRLeftController)
                cr = ent.get_component(VRRightController)
                if cl:
                    idx = 0
                    ctrl = cl
                elif cr:
                    idx = 1
                    ctrl = cr
            if ctrl is None or idx < 0:
                return
            from plugins.vr_plugin.vr_core import _vr_state
            controller_state = _vr_state.controllers[idx]
            pressed = controller_state.trigger > ctrl.trigger_threshold
            prev = _vr_state._prev_trigger[idx]
            if pressed and not prev:
                ray = vr_core.get_controller_ray(idx)
                if ray is None:
                    return
                origin, fwd = ray
                hit_ent, hit, dist = vr_core.raycast_scene(origin, fwd)
                try:
                    from core.engine.engine import Engine
                    eng = Engine.instance()
                    vp = eng.viewport if eng else None
                    if vp:
                        if hit_ent is not None:
                            vp.set_selected_entity(hit_ent)
                            if hasattr(vp, 'entity_selected'):
                                vp.entity_selected.emit(hit_ent)
                        else:
                            vp.set_selected_entity(None)
                            if hasattr(vp, 'entity_selected'):
                                vp.entity_selected.emit(None)
                            try:
                                vp._selected_entities = []
                                vp._selected_set_version = getattr(vp, '_selected_set_version', 0) + 1
                            except Exception:
                                pass
                        vp.update()
                except Exception:
                    pass
            _vr_state._prev_trigger[idx] = pressed
        except Exception:
            pass

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"hover_color": self.hover_color, "select_on_trigger": self.select_on_trigger})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> VRSelection:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.hover_color = list(data.get("hover_color", [1.0, 0.95, 0.2, 1.0]))
        c.select_on_trigger = bool(data.get("select_on_trigger", True))
        return c
