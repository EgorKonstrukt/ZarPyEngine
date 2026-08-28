# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry
from core.components.animation.animation_clip import AnimationClip


def _as_quat(value):
    from core.maths.math3d import Quat
    if isinstance(value, Quat):
        return value
    try:
        t = tuple(value)
        if len(t) == 4 and all(isinstance(c, (int, float)) for c in t):
            return Quat(t[0], t[1], t[2], t[3])
    except TypeError:
        pass
    return value


def _resolve_bone_path(root, bone_path: str):
    if not bone_path:
        return root
    parts = [p for p in bone_path.split("/") if p]
    if not parts:
        return root
    current = root
    if parts[0] and parts[0] == (getattr(root, "name", "") or ""):
        parts = parts[1:]
    for seg in parts:
        found = None
        for child in (current.children if current is not None else []):
            if child.name == seg:
                found = child
                break
        if found is None:
            return current or root
        current = found
    return current or root


@ComponentRegistry.register
class Animation(Component):
    _updates: bool = True

    def __init__(self):
        super().__init__()
        self.clip: str = ""
        self.play_on_start: bool = True
        self.speed: float = 1.0
        self._time: float = 0.0
        self._is_playing: bool = False
        self._clip_cache: Optional[AnimationClip] = None

    @classmethod
    def _inspector_fields(cls):
        from core.components.inspector_meta import InspectorField, FieldType
        return [
            InspectorField("clip", "Animation Clip", FieldType.ASSET, resource_type="animclip"),
            InspectorField("play_on_start", "Play on Start", FieldType.BOOL),
            InspectorField("speed", "Speed", FieldType.FLOAT),
        ]

    def _get_clip(self) -> Optional[AnimationClip]:
        if not self.clip:
            return None
        if self._clip_cache is None:
            self._clip_cache = AnimationClip.load(self.clip)
        return self._clip_cache

    def play(self):
        clip = self._get_clip()
        self._is_playing = True
        if self._time >= (clip.length if clip else 1.0):
            self._time = 0.0

    def stop(self):
        self._is_playing = False

    def pause(self):
        self._is_playing = False

    def on_start(self):
        clip = self._get_clip()
        if self.play_on_start and clip:
            self.play()

    def on_update(self, dt: float):
        clip = self._get_clip()
        if not self._is_playing or not clip:
            return
        self._time += dt * self.speed
        clip_len = clip.length
        if self._time >= clip_len:
            if clip.loop:
                self._time %= clip_len
            else:
                self._time = clip_len
                self._is_playing = False
                return
        self._apply_curves()

    def _apply_curves(self):
        clip = self._get_clip()
        if not clip:
            return
        ent = self._entity
        if ent is None:
            return
        for bone_path, prop, value in clip.evaluate_all(self._time):
            target = _resolve_bone_path(ent, bone_path)
            self._apply_value(target, prop, value)
        for bone_path, prop, quat in clip.evaluate_rotations_all(self._time):
            target = _resolve_bone_path(ent, bone_path)
            self._apply_rotation(target, prop, quat)

    def _apply_value(self, entity, path: str, value: float):
        try:
            from core.components.properties import write_prop
            write_prop(entity, path, value)
        except Exception:
            pass

    def _apply_rotation(self, entity, path: str, quat):
        if entity is None:
            return
        try:
            from core.components.properties import write_prop
            write_prop(entity, path, _as_quat(quat))
        except Exception:
            pass

    def serialize(self) -> dict:
        data = super().serialize()
        data["clip"] = self.clip
        data["play_on_start"] = self.play_on_start
        data["speed"] = self.speed
        data["_time"] = self._time
        data["_is_playing"] = self._is_playing
        return data

    @classmethod
    def deserialize(cls, data: dict) -> Animation:
        inst = super().deserialize(data)
        inst.clip = data.get("clip", "")
        inst.play_on_start = data.get("play_on_start", True)
        inst.speed = data.get("speed", 1.0)
        inst._time = data.get("_time", 0.0)
        inst._is_playing = data.get("_is_playing", False)
        return inst
