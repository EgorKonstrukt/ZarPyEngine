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
from core.foundation.curve import Curve, QuaternionCurve


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


@dataclass
class RawCurve:
    attribute: str = ""
    bone_path: str = ""
    class_id: int = 0
    script: str = ""
    curve: Curve = field(default_factory=Curve)


class AnimationClip:
    def __init__(self, name: str = "New Clip", length: float = 1.0):
        self.name: str = name
        self.length: float = length
        self.loop: bool = True
        self.curves: dict[str, Curve] = {}
        self.rotation_curves: dict[str, QuaternionCurve] = {}
        self.bone_curves: dict[tuple[str, str], Curve] = {}
        self.bone_rotation_curves: dict[tuple[str, str], QuaternionCurve] = {}
        self.raw_curves: dict[str, RawCurve] = {}
        self.curve_targets: dict[str, str] = {}
        self.annotations: dict[str, dict] = {}
        self.sample_rate: int = 60
        self.events: list[AnimationEvent] = []
        self._path: Optional[str] = None

    def add_curve(self, property_path: str, bone_path: str = "") -> Curve:
        if property_path not in self.curves:
            self.curves[property_path] = Curve()
        self.bone_curves[(bone_path, property_path)] = self.curves[property_path]
        return self.curves[property_path]

    def add_rotation_curve(self, property_path: str, bone_path: str = "") -> QuaternionCurve:
        if property_path not in self.rotation_curves:
            self.rotation_curves[property_path] = QuaternionCurve()
        self.bone_rotation_curves[(bone_path, property_path)] = self.rotation_curves[property_path]
        return self.rotation_curves[property_path]

    def remove_curve(self, property_path: str):
        self.curves.pop(property_path, None)
        self.rotation_curves.pop(property_path, None)
        for key in [k for k in self.bone_curves if k[1] == property_path]:
            self.bone_curves.pop(key, None)
        for key in [k for k in self.bone_rotation_curves if k[1] == property_path]:
            self.bone_rotation_curves.pop(key, None)

    def curve_paths(self) -> list[str]:
        return list(self.curves.keys()) + list(self.rotation_curves.keys())

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

    def evaluate_rotation(self, time: float) -> dict[str, tuple]:
        result: dict[str, tuple] = {}
        for path, curve in self.rotation_curves.items():
            result[path] = curve.evaluate(time)
        return result

    def evaluate_full(self, time: float) -> tuple:
        return self.evaluate(time), self.evaluate_rotation(time)

    def evaluate_all(self, time: float) -> list:
        return [
            (bone_path, prop, curve.evaluate(time))
            for (bone_path, prop), curve in self.bone_curves.items()
        ]

    def evaluate_rotations_all(self, time: float) -> list:
        return [
            (bone_path, prop, curve.evaluate(time))
            for (bone_path, prop), curve in self.bone_rotation_curves.items()
        ]

    def curve_targets_for(self, property_path: str) -> dict:
        annotation = self.annotations.get(property_path, {})
        return {
            "bone_path": annotation.get("bone_path", ""),
            "class_id": annotation.get("class_id", 0),
            "raw": property_path in self.raw_curves,
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "length": self.length,
            "loop": self.loop,
            "bone_curves": [
                {"bone_path": bone, "property": prop, "curve": curve.to_dict()}
                for (bone, prop), curve in self.bone_curves.items()
            ],
            "bone_rotation_curves": [
                {"bone_path": bone, "property": prop, "keys": curve.keys_dict()}
                for (bone, prop), curve in self.bone_rotation_curves.items()
            ],
            "curves": {path: curve.to_dict() for path, curve in self.curves.items()},
            "rotation_curves": {
                path: {"keys": curve.keys_dict()}
                for path, curve in self.rotation_curves.items()
            },
            "raw_curves": {
                path: {
                    "attribute": rc.attribute,
                    "bone_path": rc.bone_path,
                    "class_id": rc.class_id,
                    "script": rc.script,
                    "curve": rc.curve.to_dict(),
                }
                for path, rc in self.raw_curves.items()
            },
            "curve_targets": dict(self.curve_targets),
            "annotations": dict(self.annotations),
            "sample_rate": self.sample_rate,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnimationClip:
        clip = cls(name=d.get("name", "New Clip"), length=d.get("length", 1.0))
        clip.loop = d.get("loop", True)
        for path, cd in d.get("curves", {}).items():
            clip.curves[path] = Curve.from_dict(cd)
        for path, rd in d.get("rotation_curves", {}).items():
            clip.rotation_curves[path] = QuaternionCurve.from_keys_dict(
                rd.get("keys", [])
            )
        for path, rd in d.get("raw_curves", {}).items():
            clip.raw_curves[path] = RawCurve(
                attribute=rd.get("attribute", ""),
                bone_path=rd.get("bone_path", ""),
                class_id=rd.get("class_id", 0),
                script=rd.get("script", ""),
                curve=Curve.from_dict(rd.get("curve", {})),
            )
        clip.curve_targets.update(d.get("curve_targets", {}))
        clip.annotations.update(d.get("annotations", {}))
        clip.sample_rate = d.get("sample_rate", 60)
        for ed in d.get("events", []):
            clip.events.append(AnimationEvent.from_dict(ed))
        for bd in d.get("bone_curves", []):
            clip.bone_curves[(bd.get("bone_path", ""), bd.get("property", ""))] = \
                Curve.from_dict(bd.get("curve", {}))
        for bd in d.get("bone_rotation_curves", []):
            clip.bone_rotation_curves[(bd.get("bone_path", ""), bd.get("property", ""))] = \
                QuaternionCurve.from_keys_dict(bd.get("keys", []))
        if not d.get("bone_curves"):
            for path, curve in clip.curves.items():
                clip.bone_curves[("", path)] = curve
        if not d.get("bone_rotation_curves"):
            for path, curve in clip.rotation_curves.items():
                clip.bone_rotation_curves[("", path)] = curve
        return clip

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        self._path = path

    def save_unity(self, path: str):
        from core.components.animation.unity.exporters import clip_to_unity_yaml

        with open(path, "w") as f:
            f.write(clip_to_unity_yaml(self))
        meta = path + ".meta"
        if not os.path.exists(meta):
            with open(meta, "w") as f:
                f.write(
                    "fileFormatVersion: 2\n"
                    "guid: %s\n"
                    "NativeFormatImporter:\n"
                    "  externalObjects: {}\n"
                    "  mainObjectFileID: 7400000\n"
                    "  userData: \n"
                    "  assetBundleName: \n"
                    "  assetBundleVariant: \n" % _guid_for_text(path)
                )
        self._path = path

    @classmethod
    def load(cls, path: str) -> AnimationClip:
        with open(path, encoding="utf-8", errors="replace") as f:
            header = f.read(4096)
        f = open(path, encoding="utf-8", errors="replace")
        try:
            if header.lstrip().startswith("%YAML") or header.lstrip().startswith(
                "%TAG"
            ) or header.lstrip().startswith("--- !u!"):
                from core.components.animation.unity.anim_importer import import_anim

                clip = import_anim(f.read())
            else:
                clip = cls.from_dict(json.load(f))
        finally:
            f.close()
        clip._path = path
        return clip


def _guid_for_text(text: str) -> str:
    import hashlib

    return hashlib.md5(text.encode("utf-8")).hexdigest()
