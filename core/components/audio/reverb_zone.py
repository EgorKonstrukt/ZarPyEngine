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
    efx_available, ensure_efx, reverb_enabled,
    create_effect, delete_effect, is_effect,
    apply_reverb_mb, get_reverb_preset, reverb_preset_names, normalize_preset_name,
    gain_to_mb,
    create_aux_slot, delete_aux_slot, is_aux_slot,
    set_aux_slot_effect, set_aux_slot_gain,
)
from core.foundation.logger import Logger


@ComponentRegistry.register
class ReverbZone(Component):
    _icon = "ReverbZone.png"
    _gizmo_icon_color = (80, 140, 220)
    _gizmo_icon_label = "RZ"
    _show_gizmo_icon = True
    _gizmo_pass = "audio"
    _instances = set()
    _last_routing_frame = -1
    _preset_params = frozenset({
        "room", "room_hf", "room_lf", "decay_time", "decay_hf_ratio",
        "reflections", "reflections_delay", "reverb", "reverb_delay",
        "diffusion", "density", "hf_reference", "lf_reference", "room_rolloff_factor",
    })

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        try:
            options = list(reverb_preset_names())
        except Exception:
            options = ["Off", "Generic", "User"]
        if "User" not in options:
            options = options + ["User"]
        return [
            InspectorField("min_distance", "Min Distance", FieldType.FLOAT, min_val=0.0, max_val=10000.0, step=0.5, decimals=2),
            InspectorField("max_distance", "Max Distance", FieldType.FLOAT, min_val=0.0, max_val=10000.0, step=1.0, decimals=2),
            InspectorField("preset", "Reverb Preset", FieldType.ENUM, enum_options=options),
            InspectorField("room", "Room", FieldType.FLOAT, min_val=-10000.0, max_val=0.0, step=10.0, decimals=0),
            InspectorField("room_hf", "Room HF", FieldType.FLOAT, min_val=-10000.0, max_val=0.0, step=10.0, decimals=0),
            InspectorField("room_lf", "Room LF", FieldType.FLOAT, min_val=-10000.0, max_val=0.0, step=10.0, decimals=0),
            InspectorField("decay_time", "Decay Time", FieldType.FLOAT, min_val=0.1, max_val=20.0, step=0.1, decimals=2),
            InspectorField("decay_hf_ratio", "Decay HFRatio", FieldType.FLOAT, min_val=0.1, max_val=2.0, step=0.01, decimals=2),
            InspectorField("reflections", "Reflections", FieldType.FLOAT, min_val=-10000.0, max_val=1000.0, step=10.0, decimals=0),
            InspectorField("reflections_delay", "Reflections Delay", FieldType.FLOAT, min_val=0.0, max_val=0.3, step=0.001, decimals=3),
            InspectorField("reverb", "Reverb", FieldType.FLOAT, min_val=-10000.0, max_val=2000.0, step=10.0, decimals=0),
            InspectorField("reverb_delay", "Reverb Delay", FieldType.FLOAT, min_val=0.0, max_val=0.1, step=0.001, decimals=3),
            InspectorField("hf_reference", "HFReference", FieldType.FLOAT, min_val=1000.0, max_val=20000.0, step=10.0, decimals=0),
            InspectorField("lf_reference", "LFReference", FieldType.FLOAT, min_val=20.0, max_val=1000.0, step=5.0, decimals=0),
            InspectorField("room_rolloff_factor", "Room Rolloff Factor", FieldType.FLOAT, min_val=0.0, max_val=10.0, step=0.01, decimals=2),
            InspectorField("diffusion", "Diffusion", FieldType.FLOAT, min_val=0.0, max_val=100.0, step=0.5, decimals=1),
            InspectorField("density", "Density", FieldType.FLOAT, min_val=0.0, max_val=100.0, step=0.5, decimals=1),
        ]

    def __init__(self):
        object.__setattr__(self, "_suspend_preset_sync", True)
        super().__init__()
        object.__setattr__(self, "preset", "Generic")
        object.__setattr__(self, "min_distance", 10.0)
        object.__setattr__(self, "max_distance", 15.0)
        object.__setattr__(self, "room", -1000.0)
        object.__setattr__(self, "room_hf", -100.0)
        object.__setattr__(self, "room_lf", 0.0)
        object.__setattr__(self, "decay_time", 1.49)
        object.__setattr__(self, "decay_hf_ratio", 0.83)
        object.__setattr__(self, "reflections", -2602.0)
        object.__setattr__(self, "reflections_delay", 0.007)
        object.__setattr__(self, "reverb", 200.0)
        object.__setattr__(self, "reverb_delay", 0.011)
        object.__setattr__(self, "diffusion", 100.0)
        object.__setattr__(self, "density", 100.0)
        object.__setattr__(self, "hf_reference", 5000.0)
        object.__setattr__(self, "lf_reference", 250.0)
        object.__setattr__(self, "room_rolloff_factor", 0.0)
        object.__setattr__(self, "wet_mix", 1.0)
        object.__setattr__(self, "_effect_id", 0)
        object.__setattr__(self, "_slot_id", 0)
        object.__setattr__(self, "_params_sig", None)
        object.__setattr__(self, "_applied_preset", "Generic")
        object.__setattr__(self, "_preset_params_sig", None)
        object.__setattr__(self, "_suspend_preset_sync", False)
        try:
            self._preset_params_sig = self._params_signature()
            self._params_sig = None
        except Exception:
            pass

    def __setattr__(self, name: str, value):
        suspend = False
        try:
            suspend = bool(object.__getattribute__(self, "_suspend_preset_sync"))
        except Exception:
            suspend = True
        if suspend or name.startswith("_") or name in ("min_distance", "max_distance", "wet_mix", "enabled"):
            object.__setattr__(self, name, value)
            return
        if name == "preset":
            try:
                normalized = normalize_preset_name(value)
            except Exception:
                normalized = "User"
            object.__setattr__(self, "preset", normalized)
            if suspend:
                return
            if normalized == "User" or normalized == "Off":
                try:
                    object.__setattr__(self, "_applied_preset", normalized)
                except Exception:
                    pass
                try:
                    object.__setattr__(self, "_params_sig", None)
                except Exception:
                    pass
                try:
                    self._refresh_live_effect()
                except Exception:
                    pass
                return
            try:
                data = get_reverb_preset(normalized)
            except Exception:
                data = None
            if data is None:
                try:
                    object.__setattr__(self, "preset", "User")
                    object.__setattr__(self, "_applied_preset", "User")
                except Exception:
                    pass
                return
            try:
                object.__setattr__(self, "_suspend_preset_sync", True)
                for key, attr in (
                    ("room", "room"), ("room_hf", "room_hf"), ("room_lf", "room_lf"),
                    ("decay_time", "decay_time"), ("decay_hf_ratio", "decay_hf_ratio"),
                    ("reflections", "reflections"), ("reflections_delay", "reflections_delay"),
                    ("reverb", "reverb"), ("reverb_delay", "reverb_delay"),
                    ("diffusion", "diffusion"), ("density", "density"),
                    ("hf_reference", "hf_reference"), ("lf_reference", "lf_reference"),
                    ("room_rolloff", "room_rolloff_factor"),
                ):
                    try:
                        object.__setattr__(self, attr, float(data[key]))
                    except Exception:
                        pass
            finally:
                try:
                    object.__setattr__(self, "_suspend_preset_sync", False)
                except Exception:
                    pass
            try:
                object.__setattr__(self, "_applied_preset", normalized)
            except Exception:
                pass
            try:
                object.__setattr__(self, "_preset_params_sig", self._params_signature())
            except Exception:
                pass
            try:
                object.__setattr__(self, "_params_sig", None)
            except Exception:
                pass
            try:
                self._refresh_live_effect()
            except Exception:
                pass
            return
        if name in ReverbZone._preset_params:
            object.__setattr__(self, name, value)
            if suspend:
                return
            try:
                current = str(object.__getattribute__(self, "preset"))
            except Exception:
                current = "User"
            if current != "User" and current != "Off":
                try:
                    object.__setattr__(self, "preset", "User")
                except Exception:
                    pass
                try:
                    object.__setattr__(self, "_applied_preset", "User")
                except Exception:
                    pass
            try:
                object.__setattr__(self, "_params_sig", None)
            except Exception:
                pass
            try:
                self._refresh_live_effect()
            except Exception:
                pass
            return
        object.__setattr__(self, name, value)

    def _refresh_live_effect(self):
        try:
            effect = int(getattr(self, "_effect_id", 0) or 0)
            if not effect:
                return
            self._apply_effect_params(force=True)
        except Exception:
            pass

    def _is_entity_active(self) -> bool:
        try:
            ent = self._entity
            if ent is None:
                return True
            return bool(ent._active)
        except Exception:
            return True

    def _is_active(self) -> bool:
        try:
            return bool(self.enabled) and bool(self._is_entity_active())
        except Exception:
            return False

    def _reverb_allowed(self) -> bool:
        try:
            return bool(reverb_enabled())
        except Exception:
            return True

    def _is_off(self) -> bool:
        try:
            return str(getattr(self, "preset", "")) == "Off"
        except Exception:
            return False

    def _params_signature(self):
        try:
            return (
                round(float(self.room), 2),
                round(float(self.room_hf), 2),
                round(float(self.room_lf), 2),
                round(float(self.decay_time), 4),
                round(float(self.decay_hf_ratio), 4),
                round(float(self.reflections), 2),
                round(float(self.reflections_delay), 4),
                round(float(self.reverb), 2),
                round(float(self.reverb_delay), 4),
                round(float(self.diffusion), 2),
                round(float(self.density), 2),
                round(float(self.hf_reference), 1),
                round(float(self.lf_reference), 1),
                round(float(self.room_rolloff_factor), 3),
            )
        except Exception:
            return None

    def _ensure_registered(self):
        try:
            ReverbZone._instances.add(self)
        except Exception:
            pass

    def _unregister(self):
        try:
            ReverbZone._instances.discard(self)
        except Exception:
            pass

    def _sync_preset_tracking(self):
        try:
            current = str(getattr(self, "preset", "User"))
        except Exception:
            return
        if current == "User" or current == "Off":
            try:
                self._applied_preset = current
            except Exception:
                pass
            return
        try:
            data = get_reverb_preset(current)
        except Exception:
            data = None
        if data is None:
            try:
                object.__setattr__(self, "_suspend_preset_sync", True)
                object.__setattr__(self, "preset", "User")
            finally:
                try:
                    object.__setattr__(self, "_suspend_preset_sync", False)
                except Exception:
                    pass
            try:
                self._applied_preset = "User"
            except Exception:
                pass
            return
        try:
            sig = self._params_signature()
        except Exception:
            sig = None
        if self._applied_preset != current:
            self._applied_preset = current
            self._preset_params_sig = sig
            self._params_sig = None

    def apply_preset_now(self, name: str) -> bool:
        try:
            self.preset = str(name)
        except Exception:
            return False
        if self._effect_id:
            self._apply_effect_params(force=True)
        return True

    def _apply_effect_params(self, force: bool = False) -> bool:
        if not self._effect_id:
            return False
        if self._is_off():
            return False
        sig = self._params_signature()
        if not force and sig is not None and sig == self._params_sig:
            return True
        try:
            ok = bool(apply_reverb_mb(
                int(self._effect_id),
                float(self.room), float(self.room_hf), float(self.room_lf),
                float(self.decay_time), float(self.decay_hf_ratio),
                float(self.reflections), float(self.reflections_delay),
                float(self.reverb), float(self.reverb_delay),
                float(self.diffusion), float(self.density),
                float(self.hf_reference), float(self.lf_reference),
                float(self.room_rolloff_factor),
            ))
            if ok:
                try:
                    set_aux_slot_gain(int(self._slot_id), 1.0)
                except Exception:
                    pass
                self._params_sig = sig
                self._preset_params_sig = sig
            return bool(ok)
        except Exception as e:
            try:
                Logger.error(str("ReverbZone: failed to apply params: ") + str(e))
            except Exception:
                pass
            return False

    def _disconnect_owned_sources(self):
        try:
            slot = int(getattr(self, "_slot_id", 0) or 0)
        except Exception:
            return
        if not slot:
            return
        try:
            mgr = AudioSourceManager.instance()
        except Exception:
            mgr = None
        if not mgr:
            return
        try:
            for src_id in list(mgr.get_active_source_ids()):
                try:
                    if int(mgr.get_source_aux_slot(int(src_id))) == int(slot):
                        try:
                            mgr.set_source_aux_send(int(src_id), 0, 0, 0)
                        except Exception:
                            pass
                        try:
                            if hasattr(mgr, "_source_aux_gain"):
                                mgr._source_aux_gain[int(src_id)] = 0.0
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    def _cleanup(self):
        try:
            self._disconnect_owned_sources()
        except Exception:
            pass
        try:
            slot = int(getattr(self, "_slot_id", 0) or 0)
        except Exception:
            slot = 0
        try:
            effect = int(getattr(self, "_effect_id", 0) or 0)
        except Exception:
            effect = 0
        if slot:
            try:
                try:
                    set_aux_slot_effect(int(slot), 0)
                except Exception:
                    pass
                delete_aux_slot(int(slot))
            except Exception:
                pass
            try:
                self._slot_id = 0
            except Exception:
                pass
        if effect:
            try:
                delete_effect(int(effect))
            except Exception:
                pass
            try:
                self._effect_id = 0
            except Exception:
                pass
        self._params_sig = None

    def _ensure_effect(self) -> bool:
        try:
            if not self._is_active():
                return False
            if not self._reverb_allowed():
                return False
            if self._is_off():
                try:
                    self._cleanup()
                except Exception:
                    pass
                return False
            if not bool(ensure_efx()):
                return False
        except Exception:
            return False
        try:
            effect = int(getattr(self, "_effect_id", 0) or 0)
            slot = int(getattr(self, "_slot_id", 0) or 0)
        except Exception:
            effect = 0
            slot = 0
        valid_effect = False
        valid_slot = False
        try:
            valid_effect = bool(effect) and bool(is_effect(int(effect)))
        except Exception:
            valid_effect = False
        try:
            valid_slot = bool(slot) and bool(is_aux_slot(int(slot)))
        except Exception:
            valid_slot = False
        if valid_effect and valid_slot:
            try:
                self._apply_effect_params(force=False)
            except Exception:
                pass
            return True
        try:
            self._cleanup()
        except Exception:
            pass
        try:
            new_effect = int(create_effect())
            ok = bool(apply_reverb_mb(
                int(new_effect),
                float(self.room), float(self.room_hf), float(self.room_lf),
                float(self.decay_time), float(self.decay_hf_ratio),
                float(self.reflections), float(self.reflections_delay),
                float(self.reverb), float(self.reverb_delay),
                float(self.diffusion), float(self.density),
                float(self.hf_reference), float(self.lf_reference),
                float(self.room_rolloff_factor),
            ))
            if not ok:
                try:
                    delete_effect(int(new_effect))
                except Exception:
                    pass
                return False
            new_slot = int(create_aux_slot())
            set_aux_slot_effect(int(new_slot), int(new_effect))
            set_aux_slot_gain(int(new_slot), 1.0)
            self._effect_id = int(new_effect)
            self._slot_id = int(new_slot)
            self._params_sig = self._params_signature()
            self._preset_params_sig = self._params_sig
            return True
        except Exception as e:
            try:
                Logger.error(str("ReverbZone: failed to create effect/slot: ") + str(e))
            except Exception:
                pass
            try:
                self._cleanup()
            except Exception:
                pass
            return False

    def _zone_position(self):
        try:
            tr = self.transform
            if tr is None:
                return (0.0, 0.0, 0.0)
            p = tr.position
            return (float(p.x), float(p.y), float(p.z))
        except Exception:
            return (0.0, 0.0, 0.0)

    def compute_weight(self, src_pos) -> float:
        try:
            if self._is_off():
                return 0.0
            zone_pos = self._zone_position()
            dx = float(src_pos[0]) - float(zone_pos[0])
            dy = float(src_pos[1]) - float(zone_pos[1])
            dz = float(src_pos[2]) - float(zone_pos[2])
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        except Exception:
            return 0.0
        try:
            min_d = float(self.min_distance)
            max_d = float(self.max_distance)
        except Exception:
            return 0.0
        if max_d <= 0.0:
            return 0.0
        if min_d < 0.0:
            min_d = 0.0
        if max_d < min_d:
            max_d = min_d
        if dist <= min_d:
            return 1.0
        if dist >= max_d:
            return 0.0
        span = max_d - min_d
        if span <= 0.0001:
            return 1.0
        t = (dist - min_d) / span
        if t < 0.0:
            t = 0.0
        if t > 1.0:
            t = 1.0
        return 1.0 - t

    def _is_routable(self) -> bool:
        try:
            if not self._is_active():
                return False
            if not self._reverb_allowed():
                return False
            if self._is_off():
                return False
            if not bool(efx_available()):
                return False
            slot = int(getattr(self, "_slot_id", 0) or 0)
            if not slot:
                return False
            return bool(is_aux_slot(int(slot)))
        except Exception:
            return False

    @staticmethod
    def _current_frame() -> int:
        try:
            from core.engine.engine import Engine
            eng = Engine.instance()
            if eng is not None and hasattr(eng, "_frame_count"):
                return int(eng._frame_count)
        except Exception:
            pass
        return -1

    @staticmethod
    def update_routing(force: bool = False):
        try:
            frame = ReverbZone._current_frame()
            if not force and frame >= 0 and int(ReverbZone._last_routing_frame) == int(frame):
                return
            if frame >= 0:
                ReverbZone._last_routing_frame = int(frame)
        except Exception:
            pass
        try:
            mgr = AudioSourceManager.instance()
        except Exception:
            mgr = None
        if mgr is None:
            return
        try:
            allowed = bool(reverb_enabled())
        except Exception:
            allowed = True
        try:
            has_efx = bool(efx_available())
        except Exception:
            has_efx = False
        if not allowed or not has_efx:
            try:
                for src_id in list(mgr.get_active_source_ids()):
                    try:
                        cur = int(mgr.get_source_aux_slot(int(src_id)))
                    except Exception:
                        continue
                    if cur == 0:
                        continue
                    owned = False
                    try:
                        for z in list(ReverbZone._instances):
                            try:
                                if int(getattr(z, "_slot_id", 0) or 0) == int(cur) and int(cur) != 0:
                                    owned = True
                                    break
                            except Exception:
                                pass
                    except Exception:
                        pass
                    if owned:
                        try:
                            mgr.set_source_aux_send(int(src_id), 0, 0, 0)
                        except Exception:
                            pass
            except Exception:
                pass
            return
        active = []
        try:
            for z in list(ReverbZone._instances):
                try:
                    if z._is_routable():
                        active.append(z)
                except Exception:
                    pass
        except Exception:
            pass
        if not active:
            try:
                for src_id in list(mgr.get_active_source_ids()):
                    try:
                        cur = int(mgr.get_source_aux_slot(int(src_id)))
                    except Exception:
                        continue
                    if cur == 0:
                        continue
                    owned = False
                    try:
                        for z in list(ReverbZone._instances):
                            try:
                                if int(getattr(z, "_slot_id", 0) or 0) == int(cur) and int(cur) != 0:
                                    owned = True
                                    break
                            except Exception:
                                pass
                    except Exception:
                        pass
                    if owned:
                        try:
                            mgr.set_source_aux_send(int(src_id), 0, 0, 0)
                        except Exception:
                            pass
                        try:
                            if hasattr(mgr, "_source_aux_gain"):
                                mgr._source_aux_gain[int(src_id)] = 0.0
                        except Exception:
                            pass
            except Exception:
                pass
            return
        try:
            sources = list(mgr.get_active_source_ids())
        except Exception:
            return
        for src_id in sources:
            try:
                pos = mgr.get_source_position(int(src_id))
            except Exception:
                continue
            if pos is None:
                continue
            best = None
            best_w = 0.0
            for z in active:
                try:
                    w = float(z.compute_weight(pos))
                except Exception:
                    w = 0.0
                if w > best_w:
                    best_w = w
                    best = z
            if best is None or best_w <= 0.001:
                try:
                    cur = int(mgr.get_source_aux_slot(int(src_id)))
                except Exception:
                    continue
                if cur == 0:
                    continue
                owned = False
                try:
                    for z in list(ReverbZone._instances):
                        try:
                            if int(getattr(z, "_slot_id", 0) or 0) == int(cur) and int(cur) != 0:
                                owned = True
                                break
                        except Exception:
                            pass
                except Exception:
                    pass
                if owned:
                    try:
                        mgr.set_source_aux_send(int(src_id), 0, 0, 0)
                    except Exception:
                        pass
                    try:
                        if hasattr(mgr, "_source_aux_gain"):
                            mgr._source_aux_gain[int(src_id)] = 0.0
                    except Exception:
                        pass
            else:
                try:
                    mgr.set_source_aux_send_weighted(int(src_id), int(best._slot_id), float(best_w), 0)
                except Exception:
                    pass

    def on_awake(self):
        self._ensure_registered()
        try:
            self._sync_preset_tracking()
        except Exception:
            pass

    def on_start(self):
        self._ensure_registered()
        try:
            self._sync_preset_tracking()
        except Exception:
            pass
        try:
            self._ensure_effect()
        except Exception:
            pass

    def on_enable(self):
        self._ensure_registered()
        try:
            self._sync_preset_tracking()
        except Exception:
            pass
        try:
            self._ensure_effect()
        except Exception:
            pass

    def on_disable(self):
        try:
            self._disconnect_owned_sources()
        except Exception:
            pass
        try:
            self._cleanup()
        except Exception:
            pass

    def on_destroy(self):
        try:
            self._disconnect_owned_sources()
        except Exception:
            pass
        try:
            self._cleanup()
        except Exception:
            pass
        try:
            self._unregister()
        except Exception:
            pass

    def on_update(self, dt: float):
        self._ensure_registered()
        try:
            self._sync_preset_tracking()
        except Exception:
            pass
        if not self._is_active():
            if getattr(self, "_slot_id", 0) or getattr(self, "_effect_id", 0):
                try:
                    self._disconnect_owned_sources()
                except Exception:
                    pass
                try:
                    self._cleanup()
                except Exception:
                    pass
            return
        try:
            if not self._reverb_allowed():
                return
        except Exception:
            pass
        if self._is_off():
            if getattr(self, "_slot_id", 0) or getattr(self, "_effect_id", 0):
                try:
                    self._disconnect_owned_sources()
                except Exception:
                    pass
                try:
                    self._cleanup()
                except Exception:
                    pass
            return
        try:
            if not bool(ensure_efx()):
                return
        except Exception:
            return
        try:
            self._ensure_effect()
        except Exception:
            return
        try:
            self._apply_effect_params(force=False)
        except Exception:
            pass
        try:
            ReverbZone.update_routing(force=False)
        except Exception:
            pass

    @staticmethod
    def _get_listener_pos():
        try:
            audio_sys = AudioSystem.instance()
            if audio_sys:
                return audio_sys._listener_pos
        except Exception:
            pass
        return (0.0, 0.0, 0.0)

    def gizmo_lines(self):
        try:
            tr = self.transform
        except Exception:
            return []
        if not tr:
            return []
        try:
            pos = tr.position
            min_r = float(self.min_distance)
            max_r = float(self.max_distance)
        except Exception:
            return []
        lines = []
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
            "preset": str(getattr(self, "preset", "Generic")),
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "room": self.room,
            "room_hf": self.room_hf,
            "room_lf": self.room_lf,
            "decay_time": self.decay_time,
            "decay_hf_ratio": self.decay_hf_ratio,
            "reflections": self.reflections,
            "reflections_delay": self.reflections_delay,
            "reverb": self.reverb,
            "reverb_delay": self.reverb_delay,
            "diffusion": self.diffusion,
            "density": self.density,
            "hf_reference": self.hf_reference,
            "lf_reference": self.lf_reference,
            "room_rolloff_factor": self.room_rolloff_factor,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict):
        z = cls()
        try:
            z.enabled = data.get("enabled", True)
        except Exception:
            pass
        try:
            object.__setattr__(z, "_suspend_preset_sync", True)
            z.min_distance = float(data.get("min_distance", 10.0))
            z.max_distance = float(data.get("max_distance", 15.0))
            if "room" in data:
                z.room = float(data.get("room", -1000.0))
                z.room_hf = float(data.get("room_hf", -100.0))
                z.room_lf = float(data.get("room_lf", 0.0))
                z.decay_time = float(data.get("decay_time", 1.49))
                z.decay_hf_ratio = float(data.get("decay_hf_ratio", 0.83))
                z.reflections = float(data.get("reflections", -2602.0))
                z.reflections_delay = float(data.get("reflections_delay", 0.007))
                z.reverb = float(data.get("reverb", 200.0))
                z.reverb_delay = float(data.get("reverb_delay", 0.011))
                z.diffusion = float(data.get("diffusion", 100.0))
                z.density = float(data.get("density", 100.0))
                z.hf_reference = float(data.get("hf_reference", 5000.0))
                z.lf_reference = float(data.get("lf_reference", 250.0))
                z.room_rolloff_factor = float(data.get("room_rolloff_factor", 0.0))
                raw_preset = str(data.get("preset", "User"))
                try:
                    z.preset = normalize_preset_name(raw_preset)
                except Exception:
                    z.preset = "User"
                if z.preset != "User" and z.preset != "Off":
                    try:
                        table = get_reverb_preset(z.preset)
                        if table is not None:
                            matches = (
                                abs(float(z.room) - float(table["room"])) < 0.5
                                and abs(float(z.room_hf) - float(table["room_hf"])) < 0.5
                                and abs(float(z.decay_time) - float(table["decay_time"])) < 0.001
                            )
                            if not matches:
                                z.preset = "User"
                    except Exception:
                        pass
            else:
                legacy_preset = str(data.get("preset", "Generic"))
                try:
                    table = get_reverb_preset(legacy_preset)
                except Exception:
                    table = None
                if table is not None and legacy_preset != "User":
                    z.room = float(table["room"])
                    z.room_hf = float(table["room_hf"])
                    z.room_lf = float(table["room_lf"])
                    z.decay_time = float(table["decay_time"])
                    z.decay_hf_ratio = float(table["decay_hf_ratio"])
                    z.reflections = float(table["reflections"])
                    z.reflections_delay = float(table["reflections_delay"])
                    z.reverb = float(table["reverb"])
                    z.reverb_delay = float(table["reverb_delay"])
                    z.diffusion = float(table["diffusion"])
                    z.density = float(table["density"])
                    z.hf_reference = float(table["hf_reference"])
                    z.lf_reference = float(table["lf_reference"])
                    z.room_rolloff_factor = float(table.get("room_rolloff", 0.0))
                    try:
                        z.preset = normalize_preset_name(legacy_preset)
                    except Exception:
                        z.preset = "Generic"
                else:
                    try:
                        lin_density = float(data.get("density", 1.0))
                        lin_diffusion = float(data.get("diffusion", 1.0))
                        lin_gain = float(data.get("reverb_gain", 0.32))
                        lin_gain_hf = float(data.get("gain_hf", 0.89))
                        lin_refl = float(data.get("reflections_gain", 0.05))
                        lin_late = float(data.get("late_reverb_gain", 1.26))
                    except Exception:
                        lin_density = 1.0
                        lin_diffusion = 1.0
                        lin_gain = 0.32
                        lin_gain_hf = 0.89
                        lin_refl = 0.05
                        lin_late = 1.26
                    z.density = max(0.0, min(100.0, lin_density * 100.0))
                    z.diffusion = max(0.0, min(100.0, lin_diffusion * 100.0))
                    try:
                        z.room = float(gain_to_mb(lin_gain))
                    except Exception:
                        z.room = -1000.0
                    try:
                        hend = float(gain_to_mb(lin_gain_hf)) - float(gain_to_mb(lin_gain))
                        z.room_hf = max(-10000.0, min(0.0, float(gain_to_mb(lin_gain_hf))))
                    except Exception:
                        z.room_hf = -100.0
                    z.room_lf = 0.0
                    z.decay_time = float(data.get("decay_time", 1.49))
                    z.decay_hf_ratio = float(data.get("decay_hf_ratio", 0.83))
                    try:
                        z.reflections = float(gain_to_mb(lin_refl))
                    except Exception:
                        z.reflections = -2602.0
                    z.reflections_delay = float(data.get("reflections_delay", 0.007))
                    try:
                        z.reverb = float(gain_to_mb(lin_late))
                    except Exception:
                        z.reverb = 200.0
                    z.reverb_delay = float(data.get("late_reverb_delay", 0.011))
                    z.hf_reference = 5000.0
                    z.lf_reference = 250.0
                    z.room_rolloff_factor = float(data.get("room_rolloff_factor", data.get("room_rolloff", 0.0)))
                    z.preset = "User"
                z.min_distance = float(data.get("min_distance", 10.0))
                z.max_distance = float(data.get("max_distance", 15.0))
            try:
                z._applied_preset = str(getattr(z, "preset", "User"))
            except Exception:
                pass
            try:
                z._preset_params_sig = z._params_signature()
            except Exception:
                pass
            z._params_sig = None
        finally:
            try:
                object.__setattr__(z, "_suspend_preset_sync", False)
            except Exception:
                pass
        return z
