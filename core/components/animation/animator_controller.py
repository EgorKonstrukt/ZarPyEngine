# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
from typing import Optional
from enum import Enum
from dataclasses import dataclass, field
from core.components.animation.animation_clip import AnimationClip


class AnimatorParameterType(Enum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    TRIGGER = "trigger"


class AnimatorConditionMode(Enum):
    IF = "if"
    IF_NOT = "if_not"
    GREATER = "greater"
    LESS = "less"
    EQUALS = "equals"
    NOT_EQUAL = "not_equal"


@dataclass
class AnimatorCondition:
    parameter: str = ""
    mode: AnimatorConditionMode = AnimatorConditionMode.IF
    threshold: float = 0.0

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "mode": self.mode.value,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnimatorCondition:
        return cls(
            parameter=d.get("parameter", ""),
            mode=AnimatorConditionMode(d.get("mode", "if")),
            threshold=d.get("threshold", 0.0),
        )


@dataclass
class AnimatorTransition:
    destination_state: str = ""
    conditions: list[AnimatorCondition] = field(default_factory=list)
    has_exit_time: bool = False
    exit_time: float = 0.0
    transition_duration: float = 0.25
    has_fixed_duration: bool = True
    offset: float = 0.0
    mute: bool = False
    solo: bool = False

    def to_dict(self) -> dict:
        return {
            "destination_state": self.destination_state,
            "conditions": [c.to_dict() for c in self.conditions],
            "has_exit_time": self.has_exit_time,
            "exit_time": self.exit_time,
            "transition_duration": self.transition_duration,
            "has_fixed_duration": self.has_fixed_duration,
            "offset": self.offset,
            "mute": self.mute,
            "solo": self.solo,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnimatorTransition:
        return cls(
            destination_state=d.get("destination_state", ""),
            conditions=[AnimatorCondition.from_dict(c) for c in d.get("conditions", [])],
            has_exit_time=d.get("has_exit_time", False),
            exit_time=d.get("exit_time", 0.0),
            transition_duration=d.get("transition_duration", 0.25),
            has_fixed_duration=d.get("has_fixed_duration", True),
            offset=d.get("offset", 0.0),
            mute=d.get("mute", False),
            solo=d.get("solo", False),
        )


@dataclass
class AnimatorState:
    name: str = "New State"
    clip: Optional[AnimationClip] = None
    clip_path: str = ""
    speed: float = 1.0
    motion_time: float = 0.0
    tag: str = ""
    transitions: list[AnimatorTransition] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "clip_path": self.clip_path or (self.clip._path if self.clip else ""),
            "speed": self.speed,
            "motion_time": self.motion_time,
            "tag": self.tag,
            "transitions": [t.to_dict() for t in self.transitions],
            "x": self.x,
            "y": self.y,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnimatorState:
        return cls(
            name=d.get("name", "New State"),
            clip_path=d.get("clip_path", ""),
            speed=d.get("speed", 1.0),
            motion_time=d.get("motion_time", 0.0),
            tag=d.get("tag", ""),
            transitions=[AnimatorTransition.from_dict(t) for t in d.get("transitions", [])],
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
        )


@dataclass
class AnimatorParameter:
    name: str = "NewParam"
    param_type: AnimatorParameterType = AnimatorParameterType.FLOAT
    default_float: float = 0.0
    default_int: int = 0
    default_bool: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.param_type.value,
            "default_float": self.default_float,
            "default_int": self.default_int,
            "default_bool": self.default_bool,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnimatorParameter:
        return cls(
            name=d.get("name", "NewParam"),
            param_type=AnimatorParameterType(d.get("type", "float")),
            default_float=d.get("default_float", 0.0),
            default_int=d.get("default_int", 0),
            default_bool=d.get("default_bool", False),
        )


@dataclass
class AnimatorControllerLayer:
    name: str = "Base Layer"
    states: list[AnimatorState] = field(default_factory=list)
    default_state: str = ""
    weight: float = 1.0
    blending_mode: str = "override"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "states": [s.to_dict() for s in self.states],
            "default_state": self.default_state,
            "weight": self.weight,
            "blending_mode": self.blending_mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnimatorControllerLayer:
        return cls(
            name=d.get("name", "Base Layer"),
            states=[AnimatorState.from_dict(s) for s in d.get("states", [])],
            default_state=d.get("default_state", ""),
            weight=d.get("weight", 1.0),
            blending_mode=d.get("blending_mode", "override"),
        )


class AnimatorController:
    def __init__(self, name: str = "New Controller"):
        self.name: str = name
        self.parameters: list[AnimatorParameter] = []
        self.layers: list[AnimatorControllerLayer] = [AnimatorControllerLayer()]
        self._path: Optional[str] = None

    def add_parameter(self, param: AnimatorParameter):
        self.parameters.append(param)

    def remove_parameter(self, name: str):
        self.parameters = [p for p in self.parameters if p.name != name]

    def get_parameter(self, name: str) -> Optional[AnimatorParameter]:
        for p in self.parameters:
            if p.name == name:
                return p
        return None

    def add_state(self, layer_index: int = 0, state: Optional[AnimatorState] = None) -> AnimatorState:
        if layer_index >= len(self.layers):
            layer_index = 0
        if state is None:
            state = AnimatorState()
        self.layers[layer_index].states.append(state)
        return state

    def remove_state(self, layer_index: int, state_name: str):
        if layer_index >= len(self.layers):
            return
        layer = self.layers[layer_index]
        layer.states = [s for s in layer.states if s.name != state_name]
        if layer.default_state == state_name:
            layer.default_state = layer.states[0].name if layer.states else ""

    def get_state(self, layer_index: int, state_name: str) -> Optional[AnimatorState]:
        if layer_index >= len(self.layers):
            return None
        for s in self.layers[layer_index].states:
            if s.name == state_name:
                return s
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "parameters": [p.to_dict() for p in self.parameters],
            "layers": [l.to_dict() for l in self.layers],
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnimatorController:
        ctrl = cls(name=d.get("name", "New Controller"))
        ctrl.parameters = [AnimatorParameter.from_dict(p) for p in d.get("parameters", [])]
        ctrl.layers = [AnimatorControllerLayer.from_dict(l) for l in d.get("layers", [])]
        return ctrl

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        self._path = path

    def save_unity(self, path: str):
        from core.components.animation.unity.exporters import controller_to_unity_yaml
        import os
        import hashlib

        with open(path, "w") as f:
            f.write(controller_to_unity_yaml(self))
        meta = path + ".meta"
        if not os.path.exists(meta):
            with open(meta, "w") as f:
                f.write(
                    "fileFormatVersion: 2\n"
                    "guid: %s\n"
                    "NativeFormatImporter:\n"
                    "  externalObjects: {}\n"
                    "  mainObjectFileID: 9100000\n"
                    "  userData: \n"
                    "  assetBundleName: \n"
                    "  assetBundleVariant: \n" % hashlib.md5(path.encode("utf-8")).hexdigest()
                )
        self._path = path

    @classmethod
    def load(cls, path: str) -> AnimatorController:
        with open(path, encoding="utf-8", errors="replace") as fh:
            header = fh.read(4096)
        fh = open(path, encoding="utf-8", errors="replace")
        try:
            if header.lstrip().startswith("%YAML") or header.lstrip().startswith(
                "%TAG"
            ) or header.lstrip().startswith("--- !u!"):
                from core.components.animation.unity.controller_importer import (
                    import_controller,
                )

                ctrl = import_controller(path)
            else:
                ctrl = cls.from_dict(json.load(fh))
        finally:
            fh.close()
        ctrl._path = path
        return ctrl
