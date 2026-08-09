# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from core.ecs.ecs import Component, ComponentRegistry
from core.maths.math3d import Vec3
from core.components.inspector_meta import FieldType, InspectorField
from core.audio.audio_system import AudioSystem, AudioSourceManager
from core.audio.audio_efx import (
    efx_available, create_effect, delete_effect, set_effect_type,
    set_effect_param_i, set_effect_param_f,
    create_aux_slot, delete_aux_slot, set_aux_slot_effect, set_aux_slot_gain,
    AL_EFFECT_REVERB,
    AL_REVERB_DENSITY, AL_REVERB_DIFFUSION,
    AL_REVERB_GAIN, AL_REVERB_GAINHF,
    AL_REVERB_DECAY_TIME, AL_REVERB_DECAY_HFRATIO,
    AL_REVERB_REFLECTIONS_GAIN, AL_REVERB_REFLECTIONS_DELAY,
    AL_REVERB_LATE_REVERB_GAIN, AL_REVERB_LATE_REVERB_DELAY,
    AL_REVERB_AIR_ABSORPTION_GAINHF, AL_REVERB_ROOM_ROLLOFF_FACTOR,
    AL_REVERB_DECAY_HFLIMIT,
)
from core.foundation.logger import Logger


@ComponentRegistry.register
class ReverbZone(Component):
    _icon = "ReverbZone.png"
    _gizmo_icon_color = (80, 140, 220)
    _gizmo_icon_label = "RZ"
    _show_gizmo_icon: bool = True
    _gizmo_pass = "audio"

    _effect_id: int = 0
    _slot_id: int = 0

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("min_distance", "Min Distance", FieldType.FLOAT, min_val=0.0, max_val=10000.0, step=0.5, decimals=2),
            InspectorField("max_distance", "Max Distance", FieldType.FLOAT, min_val=0.0, max_val=10000.0, step=1.0, decimals=2),
            InspectorField("wet_mix", "Wet Mix", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01),
            InspectorField("density", "Density", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01),
            InspectorField("diffusion", "Diffusion", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01),
            InspectorField("reverb_gain", "Reverb Gain", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01),
            InspectorField("gain_hf", "Gain HF", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01),
            InspectorField("decay_time", "Decay Time", FieldType.FLOAT, min_val=0.1, max_val=20.0, step=0.1, decimals=2),
            InspectorField("decay_hf_ratio", "Decay HF Ratio", FieldType.FLOAT, min_val=0.1, max_val=2.0, step=0.01),
            InspectorField("reflections_gain", "Reflections Gain", FieldType.FLOAT, min_val=0.0, max_val=3.16, step=0.01),
            InspectorField("reflections_delay", "Reflections Delay", FieldType.FLOAT, min_val=0.0, max_val=0.3, step=0.001, decimals=3),
            InspectorField("late_reverb_gain", "Late Reverb Gain", FieldType.FLOAT, min_val=0.0, max_val=10.0, step=0.01),
            InspectorField("late_reverb_delay", "Late Reverb Delay", FieldType.FLOAT, min_val=0.0, max_val=0.1, step=0.001, decimals=3),
        ]

    def __init__(self):
        super().__init__()
        self.min_distance: float = 1.0
        self.max_distance: float = 50.0
        self.wet_mix: float = 0.5
        self.density: float = 1.0
        self.diffusion: float = 1.0
        self.reverb_gain: float = 0.32
        self.gain_hf: float = 0.89
        self.decay_time: float = 1.49
        self.decay_hf_ratio: float = 0.83
        self.reflections_gain: float = 0.05
        self.reflections_delay: float = 0.007
        self.late_reverb_gain: float = 0.001
        self.late_reverb_delay: float = 0.011

    def on_enable(self):
        if not efx_available():
            Logger.warning("ReverbZone: EFX not available on this system")
            return
        try:
            self._effect_id = create_effect()
            set_effect_type(self._effect_id, AL_EFFECT_REVERB)
            self._apply_effect_params()
            self._slot_id = create_aux_slot()
            set_aux_slot_effect(self._slot_id, self._effect_id)
            set_aux_slot_gain(self._slot_id, self.wet_mix)
        except Exception as e:
            Logger.error(f"ReverbZone: failed to create effect/slot: {e}")
            self._cleanup()

    def on_disable(self):
        self._disconnect_all_sources()
        self._cleanup()

    def on_destroy(self):
        self._disconnect_all_sources()
        self._cleanup()

    def _cleanup(self):
        if self._slot_id:
            try:
                set_aux_slot_effect(self._slot_id, 0)
                delete_aux_slot(self._slot_id)
            except Exception:
                pass
            self._slot_id = 0
        if self._effect_id:
            try:
                delete_effect(self._effect_id)
            except Exception:
                pass
            self._effect_id = 0

    def _apply_effect_params(self):
        if not self._effect_id:
            return
        eid = self._effect_id
        set_effect_param_f(eid, AL_REVERB_DENSITY, self.density)
        set_effect_param_f(eid, AL_REVERB_DIFFUSION, self.diffusion)
        set_effect_param_f(eid, AL_REVERB_GAIN, self.reverb_gain)
        set_effect_param_f(eid, AL_REVERB_GAINHF, self.gain_hf)
        set_effect_param_f(eid, AL_REVERB_DECAY_TIME, self.decay_time)
        set_effect_param_f(eid, AL_REVERB_DECAY_HFRATIO, self.decay_hf_ratio)
        set_effect_param_f(eid, AL_REVERB_REFLECTIONS_GAIN, self.reflections_gain)
        set_effect_param_f(eid, AL_REVERB_REFLECTIONS_DELAY, self.reflections_delay)
        set_effect_param_f(eid, AL_REVERB_LATE_REVERB_GAIN, self.late_reverb_gain)
        set_effect_param_f(eid, AL_REVERB_LATE_REVERB_DELAY, self.late_reverb_delay)
        set_effect_param_i(eid, AL_REVERB_DECAY_HFLIMIT, 1)

    def _disconnect_all_sources(self):
        if not self._slot_id:
            return
        mgr = AudioSourceManager.instance()
        if not mgr:
            return
        for src_id in mgr.get_active_source_ids():
            current_slot = mgr.get_source_aux_slot(src_id)
            if current_slot == self._slot_id:
                mgr.set_source_aux_send(src_id, 0)

    def on_update(self, dt: float):
        if not self._slot_id or not self.enabled:
            return
        mgr = AudioSourceManager.instance()
        if not mgr:
            return
        tr = self.transform
        if not tr:
            return
        zone_pos = tr.position
        listener_pos = self._get_listener_pos()

        for src_id in mgr.get_active_source_ids():
            src_pos = mgr.get_source_position(src_id)
            if src_pos is None:
                continue
            dx = src_pos[0] - zone_pos.x
            dy = src_pos[1] - zone_pos.y
            dz = src_pos[2] - zone_pos.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist <= self.max_distance:
                mgr.set_source_aux_send(src_id, self._slot_id)
            else:
                current = mgr.get_source_aux_slot(src_id)
                if current == self._slot_id:
                    mgr.set_source_aux_send(src_id, 0)

    @staticmethod
    def _get_listener_pos() -> tuple[float, float, float]:
        audio_sys = AudioSystem.instance()
        if audio_sys:
            return audio_sys._listener_pos
        return (0.0, 0.0, 0.0)

    def gizmo_lines(self) -> list[tuple[Vec3, Vec3, list[float]]]:
        tr = self.transform
        if not tr:
            return []
        pos = tr.position
        min_r = self.min_distance
        max_r = self.max_distance
        lines: list[tuple[Vec3, Vec3, list[float]]] = []
        segments = 24
        inner_color = [0.3, 0.5, 1.0, 0.6]
        outer_color = [0.2, 0.3, 0.8, 0.3]
        if min_r > 0.01:
            for axis_idx in range(3):
                pts = []
                for i in range(segments + 1):
                    theta = 2.0 * math.pi * i / segments
                    if axis_idx == 0:
                        pt = Vec3(0, math.cos(theta) * min_r, math.sin(theta) * min_r)
                    elif axis_idx == 1:
                        pt = Vec3(math.cos(theta) * min_r, 0, math.sin(theta) * min_r)
                    else:
                        pt = Vec3(math.cos(theta) * min_r, math.sin(theta) * min_r, 0)
                    pts.append(pos + pt)
                for i in range(segments):
                    lines.append((pts[i], pts[i + 1], inner_color))
        if max_r > 0.01 and abs(max_r - min_r) > 0.01:
            for axis_idx in range(3):
                pts = []
                for i in range(segments + 1):
                    theta = 2.0 * math.pi * i / segments
                    if axis_idx == 0:
                        pt = Vec3(0, math.cos(theta) * max_r, math.sin(theta) * max_r)
                    elif axis_idx == 1:
                        pt = Vec3(math.cos(theta) * max_r, 0, math.sin(theta) * max_r)
                    else:
                        pt = Vec3(math.cos(theta) * max_r, math.sin(theta) * max_r, 0)
                    pts.append(pos + pt)
                for i in range(segments):
                    lines.append((pts[i], pts[i + 1], outer_color))
        return lines

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "wet_mix": self.wet_mix,
            "density": self.density,
            "diffusion": self.diffusion,
            "reverb_gain": self.reverb_gain,
            "gain_hf": self.gain_hf,
            "decay_time": self.decay_time,
            "decay_hf_ratio": self.decay_hf_ratio,
            "reflections_gain": self.reflections_gain,
            "reflections_delay": self.reflections_delay,
            "late_reverb_gain": self.late_reverb_gain,
            "late_reverb_delay": self.late_reverb_delay,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ReverbZone:
        z = cls()
        z.enabled = data.get("enabled", True)
        z.min_distance = data.get("min_distance", 1.0)
        z.max_distance = data.get("max_distance", 50.0)
        z.wet_mix = data.get("wet_mix", 0.5)
        z.density = data.get("density", 1.0)
        z.diffusion = data.get("diffusion", 1.0)
        z.reverb_gain = data.get("reverb_gain", 0.32)
        z.gain_hf = data.get("gain_hf", 0.89)
        z.decay_time = data.get("decay_time", 1.49)
        z.decay_hf_ratio = data.get("decay_hf_ratio", 0.83)
        z.reflections_gain = data.get("reflections_gain", 0.05)
        z.reflections_delay = data.get("reflections_delay", 0.007)
        z.late_reverb_gain = data.get("late_reverb_gain", 0.001)
        z.late_reverb_delay = data.get("late_reverb_delay", 0.011)
        return z
