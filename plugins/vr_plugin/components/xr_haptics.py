# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField

@ComponentRegistry.register
class XRHaptics(Component):
    _icon = "XRHaptics.png"
    _gizmo_icon_color = (255, 160, 60)
    _gizmo_icon_label = "V"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("channel", "Channel", FieldType.ENUM, enum_options=["Left", "Right"]),
            InspectorField("default_amplitude", "Default Amplitude", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.05, decimals=2),
            InspectorField("default_duration", "Default Duration", FieldType.FLOAT, min_val=0.01, max_val=2.0, step=0.01, decimals=2),
            InspectorField("default_frequency", "Default Frequency", FieldType.FLOAT, min_val=0.0, max_val=2.0, step=0.05, decimals=2),
        ]

    def __init__(self):
        super().__init__()
        self.channel: str = "Right"
        self.default_amplitude: float = 1.0
        self.default_duration: float = 0.1
        self.default_frequency: float = 1.0

    def on_update(self, dt: float):
        return

    def _idx(self) -> int:
        return 0 if self.channel == "Left" else 1

    def SendHapticImpulse(self, amplitude: float = None, duration: float = None, frequency: float = None):
        try:
            from plugins.vr_plugin import vr_core
            amp = self.default_amplitude if amplitude is None else amplitude
            dur = self.default_duration if duration is None else duration
            freq = self.default_frequency if frequency is None else frequency
            vr_core.trigger_haptic(self._idx(), frequency=freq, amplitude=amp, duration_s=dur)
        except Exception:
            pass

    def SendHapticState(self, amplitude: float = None, frequency: float = None):
        self.SendHapticImpulse(amplitude=amplitude, duration=0.05, frequency=frequency)

    def CancelHaptic(self):
        try:
            from plugins.vr_plugin import vr_core
            vr_core.stop_haptic(self._idx())
        except Exception:
            pass

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "channel": self.channel,
            "default_amplitude": self.default_amplitude,
            "default_duration": self.default_duration,
            "default_frequency": self.default_frequency,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> XRHaptics:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.channel = data.get("channel", "Right")
        c.default_amplitude = float(data.get("default_amplitude", 1.0))
        c.default_duration = float(data.get("default_duration", 0.1))
        c.default_frequency = float(data.get("default_frequency", 1.0))
        return c
