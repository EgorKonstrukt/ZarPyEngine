# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import Component, ComponentRegistry
from core.audio.audio_system import AudioSystem
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Vec3


@ComponentRegistry.register
class AudioListener(Component):
    _icon = "AudioListener.png"
    _gizmo_icon_color = (80, 220, 80)
    _gizmo_icon_label = "L"
    _show_gizmo_icon: bool = False

    @classmethod
    def _inspector_fields(cls) -> list:
        return [
            InspectorField("doppler_factor", "Doppler Factor", FieldType.FLOAT, min_val=0.0, max_val=10.0, step=0.01, decimals=2),
            InspectorField("speed_of_sound", "Speed Of Sound", FieldType.FLOAT, min_val=0.1, max_val=10000.0, step=0.1, decimals=1),
        ]

    def __init__(self):
        super().__init__()
        self.doppler_factor: float = 1.0
        self.speed_of_sound: float = 343.3
        self._prev_pos: Vec3 | None = None

    def on_start(self):
        audio_sys = AudioSystem.instance()
        if audio_sys:
            audio_sys.set_doppler_factor(self.doppler_factor)
            audio_sys.set_speed_of_sound(self.speed_of_sound)

    def on_update(self, dt: float):
        audio_sys = AudioSystem.instance()
        if not audio_sys:
            return
        audio_sys.set_doppler_factor(self.doppler_factor)
        audio_sys.set_speed_of_sound(self.speed_of_sound)
        tr = self.transform
        pos = tr.position
        forward = tr.forward
        up = tr.up
        audio_sys.set_listener_position((pos.x, pos.y, pos.z))
        audio_sys.set_listener_orientation((forward.x, forward.y, forward.z), (up.x, up.y, up.z))
        if self._prev_pos is not None and dt > 0:
            vel = (
                (pos.x - self._prev_pos.x) / dt,
                (pos.y - self._prev_pos.y) / dt,
                (pos.z - self._prev_pos.z) / dt,
            )
        else:
            vel = (0.0, 0.0, 0.0)
        self._prev_pos = Vec3(pos.x, pos.y, pos.z)
        audio_sys.set_listener_velocity(vel)

    def serialize(self) -> dict:
        d = super().serialize()
        d["doppler_factor"] = self.doppler_factor
        d["speed_of_sound"] = self.speed_of_sound
        return d

    @classmethod
    def deserialize(cls, data: dict) -> AudioListener:
        al = cls()
        al.enabled = data.get("enabled", True)
        al.doppler_factor = data.get("doppler_factor", 1.0)
        al.speed_of_sound = data.get("speed_of_sound", 343.3)
        return al
