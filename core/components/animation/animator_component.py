# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry
from core.components.animation.animator_controller import (
    AnimatorController,
    AnimatorState,
    AnimatorTransition,
    AnimatorParameter,
    AnimatorParameterType,
    AnimatorConditionMode,
)


@ComponentRegistry.register
class Animator(Component):
    _updates: bool = True

    def __init__(self):
        super().__init__()
        self.controller: str = ""
        self.avatar: str = ""
        self.apply_root_motion: bool = False
        self.update_mode: str = "normal"
        self._controller_cache: Optional[AnimatorController] = None
        self._parameters: dict[str, float | int | bool] = {}
        self._triggers: set[str] = set()
        self._current_state: Optional[str] = None
        self._current_time: float = 0.0
        self._transitioning: bool = False
        self._transition_from: Optional[str] = None
        self._transition_to: Optional[str] = None
        self._transition_time: float = 0.0
        self._transition_duration: float = 0.0
        self._previous_state_time: float = 0.0

    @classmethod
    def _inspector_fields(cls):
        from core.components.inspector_meta import InspectorField, FieldType
        return [
            InspectorField("controller", "Controller", FieldType.ASSET, resource_type="animcontroller"),
            InspectorField("avatar", "Avatar", FieldType.ASSET, resource_type="animcontroller"),
            InspectorField("apply_root_motion", "Apply Root Motion", FieldType.BOOL),
        ]

    def _get_controller(self) -> Optional[AnimatorController]:
        if not self.controller:
            return None
        if self._controller_cache is None:
            self._controller_cache = AnimatorController.load(self.controller)
        return self._controller_cache

    def set_float(self, name: str, value: float):
        self._parameters[name] = value

    def set_int(self, name: str, value: int):
        self._parameters[name] = value

    def set_bool(self, name: str, value: bool):
        self._parameters[name] = value

    def set_trigger(self, name: str):
        self._triggers.add(name)

    def reset_trigger(self, name: str):
        self._triggers.discard(name)

    def get_float(self, name: str) -> float:
        v = self._parameters.get(name)
        if isinstance(v, (int, float)):
            return float(v)
        p = self._get_param_def(name)
        return p.default_float if p else 0.0

    def get_int(self, name: str) -> int:
        v = self._parameters.get(name)
        if isinstance(v, int):
            return v
        p = self._get_param_def(name)
        return p.default_int if p else 0

    def get_bool(self, name: str) -> bool:
        v = self._parameters.get(name)
        if isinstance(v, bool):
            return v
        p = self._get_param_def(name)
        return p.default_bool if p else False

    def _get_param_def(self, name: str) -> Optional[AnimatorParameter]:
        ctrl = self._get_controller()
        if not ctrl:
            return None
        return ctrl.get_parameter(name)

    def _init_parameters(self):
        ctrl = self._get_controller()
        if not ctrl:
            return
        for p in ctrl.parameters:
            if p.name not in self._parameters:
                if p.param_type == AnimatorParameterType.FLOAT:
                    self._parameters[p.name] = p.default_float
                elif p.param_type == AnimatorParameterType.INT:
                    self._parameters[p.name] = p.default_int
                elif p.param_type == AnimatorParameterType.BOOL:
                    self._parameters[p.name] = p.default_bool

    def on_start(self):
        self._init_parameters()
        ctrl = self._get_controller()
        if not ctrl or not ctrl.layers:
            return
        layer = ctrl.layers[0]
        if layer.default_state:
            self._current_state = layer.default_state
        elif layer.states:
            self._current_state = layer.states[0].name

    def on_update(self, dt: float):
        ctrl = self._get_controller()
        if not ctrl or not self._current_state:
            return
        layer = ctrl.layers[0] if ctrl.layers else None
        if not layer:
            return

        state = ctrl.get_state(0, self._current_state)
        if not state:
            return

        if self._transitioning:
            self._transition_time += dt
            progress = self._transition_time / max(self._transition_duration, 0.001)
            if progress >= 1.0:
                self._transitioning = False
                self._current_state = self._transition_to
                self._current_time = 0.0
                self._previous_state_time = 0.0
                state = ctrl.get_state(0, self._current_state)
                if not state:
                    return
            else:
                self._previous_state_time += dt * self._get_state_speed(self._transition_from)
                self._current_time += dt * self._get_state_speed(self._transition_to)
                self._apply_transition_blend(self._transition_from, self._transition_to, progress)
                return

        for trans in state.transitions:
            if trans.mute:
                continue
            if self._evaluate_transition(trans, state):
                self._start_transition(state, trans)
                return

        self._current_time += dt * state.speed
        clip_len = self._get_clip_length(state)
        if clip_len > 0:
            if self._current_time >= clip_len:
                if self._is_loop(state):
                    self._current_time %= clip_len
                else:
                    self._current_time = clip_len

        self._apply_state(state)

    def _get_state_speed(self, state_name: str) -> float:
        ctrl = self._get_controller()
        if not ctrl:
            return 1.0
        s = ctrl.get_state(0, state_name)
        return s.speed if s else 1.0

    def _evaluate_transition(self, trans: AnimatorTransition, state: AnimatorState) -> bool:
        if trans.has_exit_time:
            clip_len = self._get_clip_length(state)
            if clip_len > 0 and (self._current_time / clip_len) < trans.exit_time:
                return False
        for cond in trans.conditions:
            if not self._evaluate_condition(cond):
                return False
        return True

    def _evaluate_condition(self, cond) -> bool:
        value = self._parameters.get(cond.parameter, 0.0)
        is_trigger = cond.parameter in self._triggers
        if cond.mode == AnimatorConditionMode.IF:
            if cond.parameter in self._triggers:
                result = is_trigger
                self._triggers.discard(cond.parameter)
                return result
            return bool(value)
        elif cond.mode == AnimatorConditionMode.IF_NOT:
            return not bool(value)
        elif cond.mode == AnimatorConditionMode.GREATER:
            return float(value) > cond.threshold
        elif cond.mode == AnimatorConditionMode.LESS:
            return float(value) < cond.threshold
        elif cond.mode == AnimatorConditionMode.EQUALS:
            return float(value) == cond.threshold
        elif cond.mode == AnimatorConditionMode.NOT_EQUAL:
            return float(value) != cond.threshold
        return False

    def _start_transition(self, from_state: AnimatorState, trans: AnimatorTransition):
        self._transitioning = True
        self._transition_from = from_state.name
        self._transition_to = trans.destination_state
        self._transition_time = 0.0
        self._transition_duration = trans.transition_duration
        self._previous_state_time = self._current_time

    def _apply_transition_blend(self, from_name: str, to_name: str, t: float):
        ctrl = self._get_controller()
        if not ctrl:
            return
        from_state = ctrl.get_state(0, from_name)
        to_state = ctrl.get_state(0, to_name)
        if not from_state or not to_state:
            return
        from_vals = self._evaluate_state_at_time(from_state, self._previous_state_time)
        to_vals = self._evaluate_state_at_time(to_state, self._current_time)
        ent = self._entity
        if ent is None:
            return
        eased = 1.0 - math.pow(1.0 - t, 3) if t < 0.5 else math.pow(t * 2.0 - 1.0, 3) * 0.5 + 0.5
        all_keys = set(from_vals.keys()) | set(to_vals.keys())
        for path in all_keys:
            fv = from_vals.get(path, 0.0)
            tv = to_vals.get(path, 0.0)
            blended = fv + (tv - fv) * eased
            self._apply_value(ent, path, blended)

    def _apply_state(self, state: AnimatorState):
        ent = self._entity
        if ent is None:
            return
        vals = self._evaluate_state_at_time(state, self._current_time)
        for path, value in vals.items():
            self._apply_value(ent, path, value)

    def _evaluate_state_at_time(self, state: AnimatorState, time: float) -> dict[str, float]:
        if not state.clip:
            return {}
        clip = state.clip
        return clip.evaluate(time)

    def _get_clip_length(self, state: AnimatorState) -> float:
        if state.clip:
            return state.clip.length
        return 0.0

    def _is_loop(self, state: AnimatorState) -> bool:
        if state.clip:
            return state.clip.loop
        return True

    def _apply_value(self, entity, path: str, value: float):
        parts = path.split("/")
        if len(parts) < 2:
            return
        comp_name = parts[0]
        prop_name = parts[1]
        for c in entity.get_all_components():
            if type(c).__name__ == comp_name:
                if hasattr(c, prop_name):
                    setattr(c, prop_name, value)

    def serialize(self) -> dict:
        data = super().serialize()
        data["controller"] = self.controller
        data["avatar"] = self.avatar
        data["apply_root_motion"] = self.apply_root_motion
        data["update_mode"] = self.update_mode
        param_data = {}
        for k, v in self._parameters.items():
            if isinstance(v, bool):
                param_data[k] = {"type": "bool", "value": v}
            elif isinstance(v, int):
                param_data[k] = {"type": "int", "value": v}
            else:
                param_data[k] = {"type": "float", "value": v}
        data["_parameters"] = param_data
        data["_triggers"] = list(self._triggers)
        data["_current_state"] = self._current_state
        data["_current_time"] = self._current_time
        return data

    @classmethod
    def deserialize(cls, data: dict) -> Animator:
        inst = super().deserialize(data)
        inst.controller = data.get("controller", "")
        inst.avatar = data.get("avatar", "")
        inst.apply_root_motion = data.get("apply_root_motion", False)
        inst.update_mode = data.get("update_mode", "normal")
        inst._parameters = {}
        for k, vd in data.get("_parameters", {}).items():
            if vd["type"] == "bool":
                inst._parameters[k] = vd["value"]
            elif vd["type"] == "int":
                inst._parameters[k] = int(vd["value"])
            else:
                inst._parameters[k] = float(vd["value"])
        inst._triggers = set(data.get("_triggers", []))
        inst._current_state = data.get("_current_state")
        inst._current_time = data.get("_current_time", 0.0)
        return inst
