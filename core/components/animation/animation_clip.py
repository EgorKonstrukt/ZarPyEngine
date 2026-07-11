# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Optional
from core.foundation.curve import Curve


@dataclass
class AnimationEvent:
    time: float = 0.0
    function_name: str = ""
    string_parameter: str = ""
    float_parameter: float = 0.0
    int_parameter: int = 0

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "function_name": self.function_name,
            "string_parameter": self.string_parameter,
            "float_parameter": self.float_parameter,
            "int_parameter": self.int_parameter,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnimationEvent:
        return cls(
            time=d.get("time", 0.0),
            function_name=d.get("function_name", ""),
            string_parameter=d.get("string_parameter", ""),
            float_parameter=d.get("float_parameter", 0.0),
            int_parameter=d.get("int_parameter", 0),
        )


class AnimationClip:
    def __init__(self, name: str = "New Clip", length: float = 1.0):
        self.name: str = name
        self.length: float = length
        self.loop: bool = True
        self.curves: dict[str, Curve] = {}
        self.events: list[AnimationEvent] = []
        self._path: Optional[str] = None

    def add_curve(self, property_path: str) -> Curve:
        if property_path not in self.curves:
            self.curves[property_path] = Curve()
        return self.curves[property_path]

    def remove_curve(self, property_path: str):
        self.curves.pop(property_path, None)

    def add_event(self, event: AnimationEvent):
        self.events.append(event)
        self.events.sort(key=lambda e: e.time)

    def remove_event(self, event: AnimationEvent):
        if event in self.events:
            self.events.remove(event)

    def evaluate(self, time: float) -> dict[str, float]:
        result: dict[str, float] = {}
        for path, curve in self.curves.items():
            result[path] = curve.evaluate(time)
        return result

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "length": self.length,
            "loop": self.loop,
            "curves": {path: curve.to_dict() for path, curve in self.curves.items()},
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnimationClip:
        clip = cls(name=d.get("name", "New Clip"), length=d.get("length", 1.0))
        clip.loop = d.get("loop", True)
        for path, cd in d.get("curves", {}).items():
            clip.curves[path] = Curve.from_dict(cd)
        for ed in d.get("events", []):
            clip.events.append(AnimationEvent.from_dict(ed))
        return clip

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        self._path = path

    @classmethod
    def load(cls, path: str) -> AnimationClip:
        with open(path) as f:
            d = json.load(f)
        clip = cls.from_dict(d)
        clip._path = path
        return clip
