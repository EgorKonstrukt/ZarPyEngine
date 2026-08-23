# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Vec3, Quat

_HEAD_DEFAULT = Vec3(0.0, 1.6, 0.0)
_LEFT_DEFAULT = Vec3(-0.25, 1.2, -0.2)
_RIGHT_DEFAULT = Vec3(0.25, 1.2, -0.2)

@ComponentRegistry.register
class XRRig(Component):
    _icon = "XRRig.png"
    _gizmo_icon_color = (80, 180, 255)
    _gizmo_icon_label = "XR"
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
            InspectorField("tracking_origin_mode", "Tracking Origin Mode", FieldType.ENUM, enum_options=["Device", "Floor"]),
            InspectorField("enable_locomotion", "Enable Locomotion", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.move_speed: float = 5.0
        self.turn_speed: float = 80.0
        self.vert_speed: float = 2.5
        self.acceleration: float = 12.0
        self.damping: float = 8.0
        self.ipd: float = 0.063
        self.tracking_origin_mode: str = "Floor"
        self.enable_locomotion: bool = True
        self.rig_yaw: float = 0.0
        self.velocity: Vec3 = Vec3.zero()

    def on_update(self, dt: float):
        return

    def camera(self):
        try:
            for child in self.entity.children:
                if child.get_component(XRTrackedPoseDriver) and child.get_component(XRTrackedPoseDriver).pose_source in ("CenterEye", "Head"):
                    return child
        except Exception:
            pass
        return None

    def get_controller(self, hand: str):
        try:
            for child in self.entity.children:
                c = child.get_component(XRController)
                if c is not None and c.controller_hand == hand:
                    return child
        except Exception:
            pass
        return None

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "move_speed": self.move_speed, "turn_speed": self.turn_speed,
            "vert_speed": self.vert_speed, "acceleration": self.acceleration,
            "damping": self.damping, "ipd": self.ipd,
            "tracking_origin_mode": self.tracking_origin_mode,
            "enable_locomotion": self.enable_locomotion,
            "rig_yaw": self.rig_yaw,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> XRRig:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.move_speed = float(data.get("move_speed", 5.0))
        c.turn_speed = float(data.get("turn_speed", 80.0))
        c.vert_speed = float(data.get("vert_speed", 2.5))
        c.acceleration = float(data.get("acceleration", 12.0))
        c.damping = float(data.get("damping", 8.0))
        c.ipd = float(data.get("ipd", 0.063))
        c.tracking_origin_mode = data.get("tracking_origin_mode", "Floor")
        c.enable_locomotion = bool(data.get("enable_locomotion", True))
        c.rig_yaw = float(data.get("rig_yaw", 0.0))
        return c


@ComponentRegistry.register
class XRTrackedPoseDriver(Component):
    _icon = "XRTrackedPoseDriver.png"
    _gizmo_icon_color = (255, 200, 80)
    _gizmo_icon_label = "T"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("pose_source", "Pose Source", FieldType.ENUM,
                           enum_options=["Head", "CenterEye", "LeftEye", "RightEye", "LeftHand", "RightHand"]),
        ]

    def __init__(self):
        super().__init__()
        self.pose_source: str = "Head"

    def on_update(self, dt: float):
        return

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"pose_source": self.pose_source})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> XRTrackedPoseDriver:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.pose_source = data.get("pose_source", "Head")
        return c


@ComponentRegistry.register
class XRController(Component):
    _icon = "XRController.png"
    _gizmo_icon_color = (80, 200, 255)
    _gizmo_icon_label = "C"
    _show_gizmo_icon = True
    _allow_multiple = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("controller_hand", "Hand", FieldType.ENUM, enum_options=["Left", "Right"]),
            InspectorField("model_prefab", "Model Prefab", FieldType.STRING),
            InspectorField("enable_input", "Enable Input", FieldType.BOOL),
            InspectorField("rotation_offset", "Grip Rotation Offset", FieldType.VEC3),
            InspectorField("haptic_amplitude", "Haptic Amplitude", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.05, decimals=2),
        ]

    def __init__(self, controller_hand: str = "Left"):
        super().__init__()
        self.controller_hand: str = controller_hand
        self.model_prefab: str = ""
        self.enable_input: bool = True
        self.rotation_offset: Vec3 = Vec3.zero()
        self.haptic_amplitude: float = 1.0
        self.model: str = "OculusTouch"
        self._model_entity = None
        self._current_model = None
        self._trigger: float = 0.0
        self._grip: float = 0.0
        self._thumbstick: tuple = (0.0, 0.0)
        self._select: bool = False
        self._activate: bool = False

    def on_attach(self, entity):
        super().on_attach(entity)
        try:
            if self._model_entity is None and self.model != "None":
                self.set_model(self.model)
        except Exception:
            pass

    def on_update(self, dt: float):
        return

    def set_model(self, model: str):
        self.model = model
        self._current_model = model
        if self.entity is None:
            return
        try:
            from core.ecs.ecs import Entity
            if self._model_entity is not None and self._model_entity in self.entity.children:
                self._model_entity.set_parent(None, preserve_world=False)
                self._model_entity = None
            if model == "None":
                return
            paths = {
                "OculusQuest2": "core/3d_models/OculusQ2/OculusQ2.fbx",
                "OculusTouch": "core/3d_models/OculusTouch/OculusTouch.fbx",
                "GenericHMD": "core/3d_models/generic_hmd/generic_hmd.obj",
                "ViveController": "core/3d_models/vive_controller/vive_controller.obj",
                "ValveIndex": "core/3d_models/valve_index/valve_index.obj",
            }
            path = paths.get(model)
            if path is None:
                return
            me = Entity(name="XRControllerModel")
            mf = MeshFilter()
            mf.mesh_path = path
            mf.mesh_name = "mesh"
            me.add_component(mf)
            from core.components.mesh_filter import MeshRenderer
            mr = MeshRenderer()
            mr.enabled = True
            me.add_component(mr)
            try:
                import os
                mat_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "materials", "vr_unlit.mat")
                from core.components.material import Material
                mr.material = Material.load(mat_path)
            except Exception:
                pass
            scale = 1.0
            if model == "OculusQuest2":
                scale = 0.00035
            elif model == "GenericHMD":
                scale = 1.0
            me.transform.local_scale = Vec3(scale, scale, scale)
            me.set_parent(self.entity, preserve_world=False)
            self._model_entity = me
        except Exception:
            pass

    def get_controller_index(self) -> int:
        return 0 if self.controller_hand == "Left" else 1

    def SendHapticImpulse(self, amplitude: float = 1.0, duration: float = 0.1, frequency: float = 1.0):
        try:
            from plugins.vr_plugin import vr_core
            vr_core.trigger_haptic(self.get_controller_index(), frequency=frequency,
                                   amplitude=amplitude, duration_s=duration)
        except Exception:
            pass

    def SendHapticState(self, amplitude: float = 1.0, frequency: float = 1.0):
        try:
            from plugins.vr_plugin import vr_core
            vr_core.trigger_haptic(self.get_controller_index(), frequency=frequency,
                                   amplitude=amplitude, duration_s=0.05)
        except Exception:
            pass

    def CancelHaptic(self):
        try:
            from plugins.vr_plugin import vr_core
            vr_core.stop_haptic(self.get_controller_index())
        except Exception:
            pass

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "controller_hand": self.controller_hand,
            "model_prefab": self.model_prefab,
            "enable_input": self.enable_input,
            "rotation_offset": [self.rotation_offset.x, self.rotation_offset.y, self.rotation_offset.z],
            "haptic_amplitude": self.haptic_amplitude,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> XRController:
        c = cls(controller_hand=data.get("controller_hand", "Left"))
        c.enabled = data.get("enabled", True)
        c.model_prefab = data.get("model_prefab", "")
        c.enable_input = bool(data.get("enable_input", True))
        ro = data.get("rotation_offset", [0, 0, 0])
        c.rotation_offset = Vec3(float(ro[0]), float(ro[1]), float(ro[2]))
        c.haptic_amplitude = float(data.get("haptic_amplitude", 1.0))
        return c


@ComponentRegistry.register
class XRHand(Component):
    _icon = "XRHand.png"
    _gizmo_icon_color = (120, 220, 160)
    _gizmo_icon_label = "H"
    _show_gizmo_icon = True
    _allow_multiple = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("hand", "Hand", FieldType.ENUM, enum_options=["Left", "Right"]),
            InspectorField("hand_tracking", "Use Hand Tracking", FieldType.BOOL),
        ]

    def __init__(self, hand: str = "Left"):
        super().__init__()
        self.hand: str = hand
        self.hand_tracking: bool = True
        self.joints: list = []

    def on_update(self, dt: float):
        return

    def get_controller_index(self) -> int:
        return 0 if self.hand == "Left" else 1

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"hand": self.hand, "hand_tracking": self.hand_tracking})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> XRHand:
        c = cls(hand=data.get("hand", "Left"))
        c.enabled = data.get("enabled", True)
        c.hand_tracking = bool(data.get("hand_tracking", True))
        return c


@ComponentRegistry.register
class XRCull(Component):
    _icon = "XRCull.png"
    _gizmo_icon_color = (255, 80, 80)
    _gizmo_icon_label = "X"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("hide_from_eye", "Hide From Eye View", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.hide_from_eye: bool = True

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"hide_from_eye": self.hide_from_eye})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> XRCull:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.hide_from_eye = bool(data.get("hide_from_eye", True))
        return c
