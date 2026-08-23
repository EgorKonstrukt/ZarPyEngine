# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from core.ecs.ecs import Component, ComponentRegistry, Entity, Scene
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Vec3, Quat

@ComponentRegistry.register
class XRInteractionManager(Component):
    _icon = "XRInteractionManager.png"
    _gizmo_icon_color = (200, 160, 255)
    _gizmo_icon_label = "IM"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("enable_hover", "Enable Hover", FieldType.BOOL),
            InspectorField("max_raycast_distance", "Max Raycast Distance", FieldType.FLOAT, min_val=1.0, max_val=200.0, step=1.0, decimals=1),
        ]

    def __init__(self):
        super().__init__()
        self.enable_hover: bool = True
        self.max_raycast_distance: float = 50.0

    def on_update(self, dt: float):
        return

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"enable_hover": self.enable_hover, "max_raycast_distance": self.max_raycast_distance})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> XRInteractionManager:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.enable_hover = bool(data.get("enable_hover", True))
        c.max_raycast_distance = float(data.get("max_raycast_distance", 50.0))
        return c


@ComponentRegistry.register
class XRBaseInteractor(Component):
    _icon = "XRInteractor.png"
    _gizmo_icon_color = (255, 220, 120)
    _gizmo_icon_label = "I"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("hand", "Hand", FieldType.ENUM, enum_options=["Left", "Right"]),
            InspectorField("interaction_layers", "Interaction Layers", FieldType.STRING),
            InspectorField("ray_length", "Ray Length", FieldType.FLOAT, min_val=0.5, max_val=100.0, step=0.5, decimals=1),
            InspectorField("select_on", "Select On", FieldType.ENUM, enum_options=["Trigger", "Grip"]),
            InspectorField("enable_haptic_on_hover", "Haptic On Hover", FieldType.BOOL),
        ]

    def __init__(self, hand: str = "Right"):
        super().__init__()
        self.hand: str = hand
        self.interaction_layers: str = "Default"
        self.ray_length: float = 10.0
        self.select_on: str = "Trigger"
        self.enable_haptic_on_hover: bool = True
        self._hover: object = None
        self._selected: object = None

    def on_update(self, dt: float):
        return

    def controller_index(self) -> int:
        return 0 if self.hand == "Left" else 1

    def get_ray(self):
        try:
            from plugins.vr_plugin import vr_core
            return vr_core.get_controller_ray(self.controller_index())
        except Exception:
            return None

    def get_pose(self):
        try:
            from plugins.vr_plugin import vr_core
            idx = self.controller_index()
            p = vr_core.get_controller_world_pos(idx)
            q = vr_core.get_controller_world_quat(idx)
            if p is None:
                return None
            return (p, q)
        except Exception:
            return None

    def gizmo_lines(self):
        try:
            if not self.enabled:
                return []
            ray = self.get_ray()
            if ray is None:
                return []
            origin, fwd = ray
            length = self.ray_length
            end = (origin[0]+fwd[0]*length, origin[1]+fwd[1]*length, origin[2]+fwd[2]*length)
            from core.maths.math3d import Vec3
            col = [0.2, 0.85, 1.0, 1.0] if self.hand == "Left" else [1.0, 0.35, 0.35, 1.0]
            return [(Vec3(*origin), Vec3(*end), col)]
        except Exception:
            return []

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "hand": self.hand, "interaction_layers": self.interaction_layers,
            "ray_length": self.ray_length, "select_on": self.select_on,
            "enable_haptic_on_hover": self.enable_haptic_on_hover,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> XRBaseInteractor:
        c = cls(hand=data.get("hand", "Right"))
        c.enabled = data.get("enabled", True)
        c.interaction_layers = data.get("interaction_layers", "Default")
        c.ray_length = float(data.get("ray_length", 10.0))
        c.select_on = data.get("select_on", "Trigger")
        c.enable_haptic_on_hover = bool(data.get("enable_haptic_on_hover", True))
        return c


@ComponentRegistry.register
class XRRayInteractor(XRBaseInteractor):
    def __init__(self, hand: str = "Right"):
        super().__init__(hand=hand)

    @classmethod
    def deserialize(cls, data: dict) -> XRRayInteractor:
        c = cls(hand=data.get("hand", "Right"))
        c.enabled = data.get("enabled", True)
        c.interaction_layers = data.get("interaction_layers", "Default")
        c.ray_length = float(data.get("ray_length", 10.0))
        c.select_on = data.get("select_on", "Trigger")
        c.enable_haptic_on_hover = bool(data.get("enable_haptic_on_hover", True))
        return c


@ComponentRegistry.register
class XRDirectInteractor(XRBaseInteractor):
    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        fields = XRBaseInteractor._inspector_fields()
        fields.append(InspectorField("reach_radius", "Reach Radius", FieldType.FLOAT, min_val=0.05, max_val=2.0, step=0.05, decimals=2))
        return fields

    def __init__(self, hand: str = "Right"):
        super().__init__(hand=hand)
        self.reach_radius: float = 0.15

    @classmethod
    def deserialize(cls, data: dict) -> XRDirectInteractor:
        c = cls(hand=data.get("hand", "Right"))
        c.enabled = data.get("enabled", True)
        c.interaction_layers = data.get("interaction_layers", "Default")
        c.ray_length = float(data.get("ray_length", 10.0))
        c.select_on = data.get("select_on", "Trigger")
        c.enable_haptic_on_hover = bool(data.get("enable_haptic_on_hover", True))
        c.reach_radius = float(data.get("reach_radius", 0.15))
        return c


@ComponentRegistry.register
class XRPokeInteractor(XRBaseInteractor):
    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        fields = XRBaseInteractor._inspector_fields()
        fields.append(InspectorField("poke_radius", "Poke Radius", FieldType.FLOAT, min_val=0.01, max_val=0.5, step=0.01, decimals=2))
        return fields

    def __init__(self, hand: str = "Right"):
        super().__init__(hand=hand)
        self.poke_radius: float = 0.05

    @classmethod
    def deserialize(cls, data: dict) -> XRPokeInteractor:
        c = cls(hand=data.get("hand", "Right"))
        c.enabled = data.get("enabled", True)
        c.interaction_layers = data.get("interaction_layers", "Default")
        c.ray_length = float(data.get("ray_length", 10.0))
        c.select_on = data.get("select_on", "Trigger")
        c.enable_haptic_on_hover = bool(data.get("enable_haptic_on_hover", True))
        c.poke_radius = float(data.get("poke_radius", 0.05))
        return c


@ComponentRegistry.register
class XRBaseInteractable(Component):
    _icon = "XRInteractable.png"
    _gizmo_icon_color = (160, 255, 180)
    _gizmo_icon_label = "X"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("interaction_layers", "Interaction Layers", FieldType.STRING),
            InspectorField("select_on_hover", "Select Mode", FieldType.ENUM, enum_options=["Toggle", "Hold"]),
        ]

    def __init__(self):
        super().__init__()
        self.interaction_layers: str = "Default"
        self.select_on_hover: str = "Hold"
        self._hovered_by = None
        self._selected_by = None

    def on_update(self, dt: float):
        return

    def on_hover_entered(self, interactor):
        pass

    def on_hover_exited(self, interactor):
        pass

    def on_select_entered(self, interactor):
        pass

    def on_select_exited(self, interactor):
        pass

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"interaction_layers": self.interaction_layers, "select_on_hover": self.select_on_hover})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> XRBaseInteractable:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.interaction_layers = data.get("interaction_layers", "Default")
        c.select_on_hover = data.get("select_on_hover", "Hold")
        return c


@ComponentRegistry.register
class XRGrabInteractable(XRBaseInteractable):
    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        fields = XRBaseInteractable._inspector_fields()
        fields.append(InspectorField("attach_to_point", "Attach To Controller", FieldType.BOOL))
        fields.append(InspectorField("throw_smoothing", "Throw Smoothing", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.05, decimals=2))
        return fields

    def __init__(self):
        super().__init__()
        self.attach_to_point: bool = True
        self.throw_smoothing: float = 0.5
        self._original_parent = None
        self._original_local_pos = None
        self._original_local_rot = None
        self._vel_samples = []

    def on_select_entered(self, interactor):
        try:
            ent = self.entity
            tr = ent.transform
            self._original_parent = ent.parent
            self._original_local_pos = tr.local_position
            self._original_local_rot = tr.local_rotation
            if self.attach_to_point and interactor.entity is not None:
                ent.set_parent(interactor.entity, preserve_world=True)
                ent._is_static = False
            self._vel_samples = []
        except Exception:
            pass

    def on_select_exited(self, interactor):
        try:
            ent = self.entity
            if self._original_parent is not None:
                ent.set_parent(self._original_parent, preserve_world=True)
            elif ent.parent is not None:
                ent.set_parent(None, preserve_world=True)
            if self._original_local_pos is not None:
                ent.transform.local_position = self._original_local_pos
            if self._original_local_rot is not None:
                ent.transform.local_rotation = self._original_local_rot
        except Exception:
            pass

    @classmethod
    def deserialize(cls, data: dict) -> XRGrabInteractable:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.interaction_layers = data.get("interaction_layers", "Default")
        c.select_on_hover = data.get("select_on_hover", "Hold")
        c.attach_to_point = bool(data.get("attach_to_point", True))
        c.throw_smoothing = float(data.get("throw_smoothing", 0.5))
        return c


def _layer_match(a: str, b: str) -> bool:
    if not a or a == "Default" or not b or b == "Default":
        return True
    return a == b


def update_interaction(scene, dt: float, manager=None):
    if scene is None:
        return
    interactors = []
    interactables = []
    for e in scene.get_all_entities():
        if not e.active:
            continue
        inter = e.get_component(XRBaseInteractor)
        if inter is not None and inter.enabled:
            interactors.append(inter)
        inter2 = e.get_component(XRGrabInteractable)
        if inter2 is None:
            inter2 = e.get_component(XRBaseInteractable)
        if inter2 is not None and inter2.enabled:
            interactables.append(inter2)
    try:
        from plugins.vr_plugin import vr_core
    except Exception:
        return
    max_dist = manager.max_raycast_distance if manager is not None else 50.0
    for inter in interactors:
        idx = inter.controller_index()
        if inter.select_on == "Trigger":
            pressed = vr_core._vr_state.controllers[idx].trigger
            threshold = 0.6
        else:
            pressed = vr_core._vr_state.controllers[idx].grip
            threshold = 0.5
        is_pressed = pressed > threshold
        prev = getattr(inter, '_was_pressed', False)
        just_pressed = is_pressed and not prev
        just_released = (not is_pressed) and prev
        inter._was_pressed = is_pressed
        target = _resolve_target(inter, interactables, vr_core, max_dist)
        if target is not None and target is not inter._hover:
            if inter._hover is not None:
                inter._hover.on_hover_exited(inter)
                inter._hover._hovered_by = None
            inter._hover = target
            target._hovered_by = inter
            target.on_hover_entered(inter)
            if inter.enable_haptic_on_hover:
                inter.entity.get_component(XRHaptics) if inter.entity else None
                _haptic(inter, 0.3, 0.04)
        elif target is None and inter._hover is not None:
            inter._hover.on_hover_exited(inter)
            inter._hover._hovered_by = None
            inter._hover = None
        if just_pressed and target is not None:
            inter._selected = target
            target._selected_by = inter
            target.on_select_entered(inter)
            _haptic(inter, 1.0, 0.06)
        if just_released and inter._selected is not None:
            inter._selected.on_select_exited(inter)
            inter._selected._selected_by = None
            inter._selected = None


def _haptic(inter, amp, dur):
    try:
        from plugins.vr_plugin.components.xr_haptics import XRHaptics
        hap = inter.entity.get_component(XRHaptics) if inter.entity else None
        if hap is not None:
            hap.SendHapticImpulse(amplitude=amp, duration=dur)
        else:
            from plugins.vr_plugin import vr_core
            vr_core.trigger_haptic(inter.controller_index(), amplitude=amp, duration_s=dur)
    except Exception:
        pass


def _resolve_target(inter, interactables, vr_core, max_dist):
    candidates = [it for it in interactables if _layer_match(inter.interaction_layers, it.interaction_layers)]
    if isinstance(inter, XRRayInteractor):
        ray = inter.get_ray()
        if ray is None:
            return None
        origin, fwd = ray
        ent, hit, dist = vr_core.raycast_scene(origin, fwd, max_dist)
        if ent is not None:
            for it in candidates:
                if it.entity is ent:
                    return it
        return None
    pose = inter.get_pose()
    if pose is None:
        return None
    pos = pose[0]
    radius = getattr(inter, 'reach_radius', 0.15) if isinstance(inter, XRDirectInteractor) else getattr(inter, 'poke_radius', 0.05)
    best = None
    best_d = float('inf')
    for it in candidates:
        tr = it.entity.transform if it.entity else None
        if tr is None:
            continue
        p = tr.position
        d = math.sqrt((p.x-pos[0])**2 + (p.y-pos[1])**2 + (p.z-pos[2])**2)
        if d < radius and d < best_d:
            best_d = d
            best = it
    return best
