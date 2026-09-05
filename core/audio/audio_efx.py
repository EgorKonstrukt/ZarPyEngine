# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import ctypes

try:
    import openal
except Exception:
    openal = None

__all__ = [
    "EFXError", "efx_available", "ensure_efx", "invalidate_efx_cache",
    "create_effect", "delete_effect", "is_effect", "set_effect_type",
    "set_effect_param_i", "set_effect_param_f",
    "create_filter", "delete_filter", "is_filter", "set_filter_type",
    "set_filter_param_i", "set_filter_param_f",
    "create_aux_slot", "delete_aux_slot", "is_aux_slot",
    "set_aux_slot_effect", "set_aux_slot_gain",
    "apply_reverb_params", "apply_eax_preset", "get_eax_preset", "eax_preset_names",
    "apply_reverb_mb", "get_reverb_preset", "reverb_preset_names", "normalize_preset_name",
    "mb_to_gain", "gain_to_mb",
    "REVERB_PRESETS_MB", "EAX_PRESETS",
    "get_max_aux_sends", "get_aux_send_enum", "get_direct_filter_enum",
    "set_source_aux_send", "set_source_direct_filter",
    "reverb_enabled", "occlusion_enabled", "spatialization_enabled",
]


class EFXError(RuntimeError):
    pass


_efx_initialized = False
_efx_available = False
_efx_no_device = False
_max_aux_sends = 0
_enum_cache: dict = {}

_alGenEffects = None
_alDeleteEffects = None
_alIsEffect = None
_alEffecti = None
_alEffectf = None
_alGenFilters = None
_alDeleteFilters = None
_alIsFilter = None
_alFilteri = None
_alFilterf = None
_alGenAuxiliaryEffectSlots = None
_alDeleteAuxiliaryEffectSlots = None
_alIsAuxiliaryEffectSlot = None
_alAuxiliaryEffectSloti = None
_alAuxiliaryEffectSlotf = None

AL_EFFECT_NULL = 0
AL_EFFECT_REVERB = 1
AL_EFFECT_CHORUS = 2
AL_EFFECT_DISTORTION = 3
AL_EFFECT_ECHO = 4
AL_EFFECT_FLANGER = 5
AL_EFFECT_FREQUENCY_SHIFTER = 6
AL_EFFECT_VOCAL_MORPHER = 7
AL_EFFECT_PITCH_SHIFTER = 8
AL_EFFECT_RING_MODULATOR = 9
AL_EFFECT_AUTOWAH = 10
AL_EFFECT_COMPRESSOR = 11
AL_EFFECT_EQUALIZER = 12
AL_EFFECT_EAXREVERB = 32768
AL_EFFECT_TYPE = 32769
AL_EFFECTSLOT_EFFECT = 1
AL_EFFECTSLOT_GAIN = 2
AL_EFFECTSLOT_AUXILIARY_SEND_AUTO = 3
AL_REVERB_DENSITY = 1
AL_REVERB_DIFFUSION = 2
AL_REVERB_GAIN = 3
AL_REVERB_GAINHF = 4
AL_REVERB_DECAY_TIME = 5
AL_REVERB_DECAY_HFRATIO = 6
AL_REVERB_REFLECTIONS_GAIN = 7
AL_REVERB_REFLECTIONS_DELAY = 8
AL_REVERB_LATE_REVERB_GAIN = 9
AL_REVERB_LATE_REVERB_DELAY = 10
AL_REVERB_AIR_ABSORPTION_GAINHF = 11
AL_REVERB_ROOM_ROLLOFF_FACTOR = 12
AL_REVERB_DECAY_HFLIMIT = 13
AL_EAXREVERB_DENSITY = 1
AL_EAXREVERB_DIFFUSION = 2
AL_EAXREVERB_GAIN = 3
AL_EAXREVERB_GAINHF = 4
AL_EAXREVERB_GAINLF = 5
AL_EAXREVERB_DECAY_TIME = 6
AL_EAXREVERB_DECAY_HFRATIO = 7
AL_EAXREVERB_DECAY_LFRATIO = 8
AL_EAXREVERB_REFLECTIONS_GAIN = 9
AL_EAXREVERB_REFLECTIONS_DELAY = 10
AL_EAXREVERB_REFLECTIONS_PAN = 11
AL_EAXREVERB_LATE_REVERB_GAIN = 12
AL_EAXREVERB_LATE_REVERB_DELAY = 13
AL_EAXREVERB_LATE_REVERB_PAN = 14
AL_EAXREVERB_ECHO_TIME = 15
AL_EAXREVERB_ECHO_DEPTH = 16
AL_EAXREVERB_MODULATION_TIME = 17
AL_EAXREVERB_MODULATION_DEPTH = 18
AL_EAXREVERB_AIR_ABSORPTION_GAINHF = 19
AL_EAXREVERB_HFREFERENCE = 20
AL_EAXREVERB_LFREFERENCE = 21
AL_EAXREVERB_ROOM_ROLLOFF_FACTOR = 22
AL_EAXREVERB_DECAY_HFLIMIT = 23
AL_FILTER_NULL = 0
AL_FILTER_LOWPASS = 1
AL_FILTER_HIGHPASS = 2
AL_FILTER_BANDPASS = 3
AL_FILTER_TYPE = 32769
AL_LOWPASS_GAIN = 1
AL_LOWPASS_GAINHF = 2
AL_HIGHPASS_GAIN = 1
AL_HIGHPASS_GAINLF = 2
AL_BANDPASS_GAIN = 1
AL_BANDPASS_GAINLF = 2
AL_BANDPASS_GAINHF = 3
AL_DIRECT_FILTER = 131077
AL_AUXILIARY_SEND_FILTER = 131078
AL_DIRECT_FILTER_GAINHF_AUTO = 131082
AL_AUXILIARY_SEND_FILTER_GAIN_AUTO = 131083
AL_MAX_AUXILIARY_SENDS = 2

EAX_PRESETS: dict = {
    "Generic": {"density": 1.0, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.89, "decay_time": 1.49, "decay_hf_ratio": 0.83, "reflections_gain": 0.05, "reflections_delay": 0.007, "late_gain": 1.26, "late_delay": 0.011, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 1},
    "PaddedCell": {"density": 0.1715, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.1, "decay_time": 0.17, "decay_hf_ratio": 0.1, "reflections_gain": 0.25, "reflections_delay": 0.001, "late_gain": 1.26, "late_delay": 0.002, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 1},
    "Room": {"density": 0.4287, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.592, "decay_time": 0.4, "decay_hf_ratio": 0.83, "reflections_gain": 0.121, "reflections_delay": 0.002, "late_gain": 1.26, "late_delay": 0.003, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 1},
    "Bathroom": {"density": 0.1715, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.1, "decay_time": 1.49, "decay_hf_ratio": 0.54, "reflections_gain": 0.653, "reflections_delay": 0.007, "late_gain": 3.27, "late_delay": 0.011, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "LivingRoom": {"density": 0.9767, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.1, "decay_time": 0.5, "decay_hf_ratio": 0.1, "reflections_gain": 0.205, "reflections_delay": 0.003, "late_gain": 0.28, "late_delay": 0.004, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "StoneRoom": {"density": 1.0, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.89, "decay_time": 2.31, "decay_hf_ratio": 0.64, "reflections_gain": 0.441, "reflections_delay": 0.012, "late_gain": 1.1, "late_delay": 0.017, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "Auditorium": {"density": 1.0, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.89, "decay_time": 4.32, "decay_hf_ratio": 0.59, "reflections_gain": 0.464, "reflections_delay": 0.02, "late_gain": 1.1, "late_delay": 0.03, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 1},
    "ConcertHall": {"density": 1.0, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.89, "decay_time": 3.92, "decay_hf_ratio": 0.7, "reflections_gain": 0.242, "reflections_delay": 0.02, "late_gain": 0.994, "late_delay": 0.029, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 1},
    "Cave": {"density": 1.0, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.89, "decay_time": 2.91, "decay_hf_ratio": 1.3, "reflections_gain": 0.5, "reflections_delay": 0.015, "late_gain": 0.706, "late_delay": 0.022, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "Arena": {"density": 1.0, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.89, "decay_time": 7.24, "decay_hf_ratio": 0.33, "reflections_gain": 0.261, "reflections_delay": 0.02, "late_gain": 1.018, "late_delay": 0.03, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 1},
    "Hangar": {"density": 1.0, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.89, "decay_time": 10.05, "decay_hf_ratio": 0.23, "reflections_gain": 0.5, "reflections_delay": 0.02, "late_gain": 1.256, "late_delay": 0.03, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "CarpetedHallway": {"density": 0.4287, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.1, "decay_time": 0.3, "decay_hf_ratio": 0.1, "reflections_gain": 0.121, "reflections_delay": 0.002, "late_gain": 0.153, "late_delay": 0.03, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "Hallway": {"density": 0.3643, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.89, "decay_time": 1.49, "decay_hf_ratio": 0.59, "reflections_gain": 0.245, "reflections_delay": 0.007, "late_gain": 1.661, "late_delay": 0.011, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 1},
    "StoneCorridor": {"density": 1.0, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.89, "decay_time": 2.7, "decay_hf_ratio": 0.79, "reflections_gain": 0.639, "reflections_delay": 0.012, "late_gain": 1.04, "late_delay": 0.017, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "Alley": {"density": 1.0, "diffusion": 0.3, "gain": 0.32, "gain_hf": 0.89, "decay_time": 1.49, "decay_hf_ratio": 0.86, "reflections_gain": 0.25, "reflections_delay": 0.007, "late_gain": 0.995, "late_delay": 0.011, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "Forest": {"density": 1.0, "diffusion": 0.3, "gain": 0.32, "gain_hf": 0.89, "decay_time": 1.49, "decay_hf_ratio": 0.54, "reflections_gain": 0.052, "reflections_delay": 0.162, "late_gain": 0.768, "late_delay": 0.088, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "City": {"density": 1.0, "diffusion": 0.5, "gain": 0.32, "gain_hf": 0.89, "decay_time": 1.49, "decay_hf_ratio": 0.67, "reflections_gain": 0.073, "reflections_delay": 0.007, "late_gain": 0.142, "late_delay": 0.011, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "Mountains": {"density": 1.0, "diffusion": 0.27, "gain": 0.32, "gain_hf": 0.89, "decay_time": 1.49, "decay_hf_ratio": 0.21, "reflections_gain": 0.107, "reflections_delay": 0.3, "late_gain": 0.3, "late_delay": 0.1, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "Quarry": {"density": 1.0, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.89, "decay_time": 1.49, "decay_hf_ratio": 0.83, "reflections_gain": 0.0, "reflections_delay": 0.061, "late_gain": 1.778, "late_delay": 0.025, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 1},
    "Plain": {"density": 1.0, "diffusion": 0.21, "gain": 0.32, "gain_hf": 0.89, "decay_time": 1.49, "decay_hf_ratio": 0.5, "reflections_gain": 0.058, "reflections_delay": 0.179, "late_gain": 0.108, "late_delay": 0.1, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "ParkingLot": {"density": 1.0, "diffusion": 1.0, "gain": 0.32, "gain_hf": 1.0, "decay_time": 1.65, "decay_hf_ratio": 1.5, "reflections_gain": 0.208, "reflections_delay": 0.008, "late_gain": 0.265, "late_delay": 0.012, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "SewerPipe": {"density": 0.3071, "diffusion": 0.8, "gain": 0.32, "gain_hf": 0.89, "decay_time": 2.81, "decay_hf_ratio": 0.14, "reflections_gain": 0.625, "reflections_delay": 0.012, "late_gain": 1.0, "late_delay": 0.017, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "Underwater": {"density": 1.0, "diffusion": 1.0, "gain": 0.32, "gain_hf": 0.1, "decay_time": 1.49, "decay_hf_ratio": 0.1, "reflections_gain": 0.596, "reflections_delay": 0.007, "late_gain": 7.079, "late_delay": 0.011, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "Drugged": {"density": 0.4287, "diffusion": 0.5, "gain": 0.32, "gain_hf": 1.0, "decay_time": 8.39, "decay_hf_ratio": 1.39, "reflections_gain": 0.875, "reflections_delay": 0.002, "late_gain": 3.108, "late_delay": 0.03, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "Dizzy": {"density": 0.3643, "diffusion": 0.6, "gain": 0.32, "gain_hf": 0.59, "decay_time": 17.23, "decay_hf_ratio": 0.56, "reflections_gain": 0.139, "reflections_delay": 0.02, "late_gain": 0.493, "late_delay": 0.03, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
    "Psychotic": {"density": 0.0625, "diffusion": 0.5, "gain": 0.32, "gain_hf": 1.0, "decay_time": 7.56, "decay_hf_ratio": 0.91, "reflections_gain": 0.486, "reflections_delay": 0.02, "late_gain": 2.0, "late_delay": 0.03, "air_gain_hf": 0.994, "room_rolloff": 0.0, "decay_hf_limit": 0},
}

REVERB_PRESETS_MB: dict = {
    "Off": {"room": -10000, "room_hf": -10000, "room_lf": 0, "decay_time": 1.0, "decay_hf_ratio": 1.0, "reflections": -10000, "reflections_delay": 0.02, "reverb": -10000, "reverb_delay": 0.04, "diffusion": 0.0, "density": 0.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Generic": {"room": -1000, "room_hf": -100, "room_lf": 0, "decay_time": 1.49, "decay_hf_ratio": 0.83, "reflections": -2602, "reflections_delay": 0.007, "reverb": 200, "reverb_delay": 0.011, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Padded Cell": {"room": -1400, "room_hf": -6000, "room_lf": 0, "decay_time": 0.17, "decay_hf_ratio": 0.10, "reflections": -1204, "reflections_delay": 0.001, "reverb": 207, "reverb_delay": 0.002, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Room": {"room": -1000, "room_hf": -454, "room_lf": 0, "decay_time": 0.40, "decay_hf_ratio": 0.83, "reflections": -1646, "reflections_delay": 0.002, "reverb": 53, "reverb_delay": 0.003, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Bathroom": {"room": -1000, "room_hf": -1200, "room_lf": 0, "decay_time": 1.49, "decay_hf_ratio": 0.54, "reflections": -370, "reflections_delay": 0.007, "reverb": 1030, "reverb_delay": 0.011, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Living Room": {"room": -1000, "room_hf": -6000, "room_lf": 0, "decay_time": 0.50, "decay_hf_ratio": 0.10, "reflections": -1376, "reflections_delay": 0.003, "reverb": -1104, "reverb_delay": 0.004, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Stone Room": {"room": -1000, "room_hf": -300, "room_lf": 0, "decay_time": 2.31, "decay_hf_ratio": 0.64, "reflections": -711, "reflections_delay": 0.012, "reverb": 83, "reverb_delay": 0.017, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Auditorium": {"room": -1000, "room_hf": -476, "room_lf": 0, "decay_time": 4.32, "decay_hf_ratio": 0.59, "reflections": -789, "reflections_delay": 0.020, "reverb": -289, "reverb_delay": 0.030, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Concert Hall": {"room": -1000, "room_hf": -500, "room_lf": 0, "decay_time": 3.92, "decay_hf_ratio": 0.70, "reflections": -1230, "reflections_delay": 0.020, "reverb": -2, "reverb_delay": 0.029, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Cave": {"room": -1000, "room_hf": 0, "room_lf": 0, "decay_time": 2.91, "decay_hf_ratio": 1.30, "reflections": -602, "reflections_delay": 0.015, "reverb": -302, "reverb_delay": 0.022, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Arena": {"room": -1000, "room_hf": -698, "room_lf": 0, "decay_time": 7.24, "decay_hf_ratio": 0.33, "reflections": -1166, "reflections_delay": 0.020, "reverb": 16, "reverb_delay": 0.030, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Hangar": {"room": -1000, "room_hf": -1000, "room_lf": 0, "decay_time": 10.05, "decay_hf_ratio": 0.23, "reflections": -602, "reflections_delay": 0.020, "reverb": 198, "reverb_delay": 0.030, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Carpeted Hallway": {"room": -1000, "room_hf": -4000, "room_lf": 0, "decay_time": 0.30, "decay_hf_ratio": 0.10, "reflections": -1831, "reflections_delay": 0.002, "reverb": -1630, "reverb_delay": 0.030, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Hallway": {"room": -1000, "room_hf": -300, "room_lf": 0, "decay_time": 1.49, "decay_hf_ratio": 0.59, "reflections": -1219, "reflections_delay": 0.007, "reverb": 441, "reverb_delay": 0.011, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Stone Corridor": {"room": -1000, "room_hf": -237, "room_lf": 0, "decay_time": 2.70, "decay_hf_ratio": 0.79, "reflections": -383, "reflections_delay": 0.012, "reverb": 33, "reverb_delay": 0.017, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Alley": {"room": -1000, "room_hf": -270, "room_lf": 0, "decay_time": 1.49, "decay_hf_ratio": 0.86, "reflections": -1204, "reflections_delay": 0.007, "reverb": -4, "reverb_delay": 0.011, "diffusion": 70.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Forest": {"room": -1000, "room_hf": -3300, "room_lf": 0, "decay_time": 1.49, "decay_hf_ratio": 0.54, "reflections": -2560, "reflections_delay": 0.162, "reverb": -229, "reverb_delay": 0.088, "diffusion": 70.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "City": {"room": -1000, "room_hf": -800, "room_lf": 0, "decay_time": 1.49, "decay_hf_ratio": 0.67, "reflections": -2273, "reflections_delay": 0.007, "reverb": -1691, "reverb_delay": 0.011, "diffusion": 50.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Mountains": {"room": -1000, "room_hf": -2500, "room_lf": 0, "decay_time": 1.49, "decay_hf_ratio": 0.21, "reflections": -2780, "reflections_delay": 0.300, "reverb": -1434, "reverb_delay": 0.100, "diffusion": 27.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Quarry": {"room": -1000, "room_hf": -1000, "room_lf": 0, "decay_time": 1.49, "decay_hf_ratio": 0.83, "reflections": -10000, "reflections_delay": 0.061, "reverb": 500, "reverb_delay": 0.025, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Plain": {"room": -1000, "room_hf": -2000, "room_lf": 0, "decay_time": 1.49, "decay_hf_ratio": 0.50, "reflections": -2466, "reflections_delay": 0.179, "reverb": -1926, "reverb_delay": 0.100, "diffusion": 21.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Parking Lot": {"room": -1000, "room_hf": 0, "room_lf": 0, "decay_time": 1.65, "decay_hf_ratio": 1.50, "reflections": -1363, "reflections_delay": 0.008, "reverb": -1153, "reverb_delay": 0.012, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Sewer Pipe": {"room": -1000, "room_hf": -1000, "room_lf": 0, "decay_time": 2.81, "decay_hf_ratio": 0.14, "reflections": -429, "reflections_delay": 0.014, "reverb": 648, "reverb_delay": 0.021, "diffusion": 80.0, "density": 60.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Underwater": {"room": -1000, "room_hf": -4000, "room_lf": 0, "decay_time": 1.49, "decay_hf_ratio": 0.10, "reflections": -449, "reflections_delay": 0.007, "reverb": 1700, "reverb_delay": 0.011, "diffusion": 100.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Drugged": {"room": -1000, "room_hf": 0, "room_lf": 0, "decay_time": 8.39, "decay_hf_ratio": 1.39, "reflections": -115, "reflections_delay": 0.002, "reverb": 985, "reverb_delay": 0.030, "diffusion": 50.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Dizzy": {"room": -1000, "room_hf": -613, "room_lf": 0, "decay_time": 17.23, "decay_hf_ratio": 0.56, "reflections": -1818, "reflections_delay": 0.020, "reverb": -613, "reverb_delay": 0.030, "diffusion": 60.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
    "Psychotic": {"room": -1000, "room_hf": 0, "room_lf": 0, "decay_time": 7.56, "decay_hf_ratio": 0.91, "reflections": -626, "reflections_delay": 0.020, "reverb": 774, "reverb_delay": 0.030, "diffusion": 50.0, "density": 100.0, "hf_reference": 5000.0, "lf_reference": 250.0, "room_rolloff": 0.0},
}

REVERB_PRESET_ORDER: list = [
    "Off", "Generic", "Padded Cell", "Room", "Bathroom", "Living Room",
    "Stone Room", "Auditorium", "Concert Hall", "Cave", "Arena", "Hangar",
    "Carpeted Hallway", "Hallway", "Stone Corridor", "Alley", "Forest",
    "City", "Mountains", "Quarry", "Plain", "Parking Lot", "Sewer Pipe",
    "Underwater", "Drugged", "Dizzy", "Psychotic", "User",
]


def invalidate_efx_cache():
    global _efx_initialized, _efx_available, _efx_no_device, _max_aux_sends
    global _alGenEffects, _alDeleteEffects, _alIsEffect, _alEffecti, _alEffectf
    global _alGenFilters, _alDeleteFilters, _alIsFilter, _alFilteri, _alFilterf
    global _alGenAuxiliaryEffectSlots, _alDeleteAuxiliaryEffectSlots
    global _alIsAuxiliaryEffectSlot, _alAuxiliaryEffectSloti, _alAuxiliaryEffectSlotf
    _efx_initialized = False
    _efx_available = False
    _efx_no_device = False
    _max_aux_sends = 0
    _enum_cache.clear()
    _alGenEffects = None
    _alDeleteEffects = None
    _alIsEffect = None
    _alEffecti = None
    _alEffectf = None
    _alGenFilters = None
    _alDeleteFilters = None
    _alIsFilter = None
    _alFilteri = None
    _alFilterf = None
    _alGenAuxiliaryEffectSlots = None
    _alDeleteAuxiliaryEffectSlots = None
    _alIsAuxiliaryEffectSlot = None
    _alAuxiliaryEffectSloti = None
    _alAuxiliaryEffectSlotf = None


def _resolve_enum(almod, name: bytes, fallback: int) -> int:
    try:
        cached = _enum_cache.get(name)
        if cached is not None:
            return int(cached)
        try:
            value = int(almod.alGetEnumValue(name))
        except Exception:
            value = 0
        if value == 0:
            value = fallback
        _enum_cache[name] = value
        return value
    except Exception:
        return fallback


def _load_proc(almod, alcmod, dev, name: bytes, functype):
    try:
        ptr = almod.alGetProcAddress(name)
    except Exception:
        ptr = 0
    if not ptr and alcmod is not None and dev is not None:
        try:
            ptr = alcmod.alcGetProcAddress(dev, name)
        except Exception:
            ptr = 0
    if not ptr:
        return None
    try:
        return functype(ptr)
    except Exception:
        return None


def ensure_efx() -> bool:
    global _efx_initialized, _efx_available, _efx_no_device, _max_aux_sends
    global _alGenEffects, _alDeleteEffects, _alIsEffect, _alEffecti, _alEffectf
    global _alGenFilters, _alDeleteFilters, _alIsFilter, _alFilteri, _alFilterf
    global _alGenAuxiliaryEffectSlots, _alDeleteAuxiliaryEffectSlots
    global _alIsAuxiliaryEffectSlot, _alAuxiliaryEffectSloti, _alAuxiliaryEffectSlotf
    if _efx_initialized:
        if _efx_available:
            return True
        if _efx_no_device:
            try:
                import openal as _oal_check
                _dev_check = _oal_check.oalGetDevice()
                _init_check = _oal_check.oalGetInit()
            except Exception:
                return False
            if not _init_check or _dev_check is None:
                return False
            _efx_initialized = False
            _efx_no_device = False
        else:
            return False
    if openal is None:
        _efx_initialized = True
        _efx_available = False
        _efx_no_device = False
        return False
    try:
        import openal.al as almod
        from openal import alc as alcmod
        import openal as oalmod
        try:
            dev = oalmod.oalGetDevice()
        except Exception:
            dev = None
        if dev is None:
            try:
                initialized = bool(oalmod.oalGetInit())
            except Exception:
                initialized = False
            if not initialized:
                _efx_initialized = True
                _efx_available = False
                _efx_no_device = True
                return False
            _efx_initialized = True
            _efx_available = False
            _efx_no_device = True
            return False
        try:
            present = bool(alcmod.alcIsExtensionPresent(dev, b"ALC_EXT_EFX"))
        except Exception:
            present = False
        if not present:
            _efx_initialized = True
            _efx_available = False
            _efx_no_device = False
            return False
        funcs = {
            b"alGenEffects": ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
            b"alDeleteEffects": ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
            b"alIsEffect": ctypes.CFUNCTYPE(ctypes.c_ubyte, ctypes.c_uint),
            b"alEffecti": ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_int, ctypes.c_int),
            b"alEffectf": ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_int, ctypes.c_float),
            b"alGenFilters": ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
            b"alDeleteFilters": ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
            b"alIsFilter": ctypes.CFUNCTYPE(ctypes.c_ubyte, ctypes.c_uint),
            b"alFilteri": ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_int, ctypes.c_int),
            b"alFilterf": ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_int, ctypes.c_float),
            b"alGenAuxiliaryEffectSlots": ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
            b"alDeleteAuxiliaryEffectSlots": ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
            b"alIsAuxiliaryEffectSlot": ctypes.CFUNCTYPE(ctypes.c_ubyte, ctypes.c_uint),
            b"alAuxiliaryEffectSloti": ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_int, ctypes.c_int),
            b"alAuxiliaryEffectSlotf": ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_int, ctypes.c_float),
        }
        loaded: dict = {}
        for fname, ftype in funcs.items():
            fn = _load_proc(almod, alcmod, dev, fname, ftype)
            if fn is None:
                _efx_initialized = True
                _efx_available = False
                _efx_no_device = False
                return False
            loaded[fname] = fn
        _alGenEffects = loaded[b"alGenEffects"]
        _alDeleteEffects = loaded[b"alDeleteEffects"]
        _alIsEffect = loaded[b"alIsEffect"]
        _alEffecti = loaded[b"alEffecti"]
        _alEffectf = loaded[b"alEffectf"]
        _alGenFilters = loaded[b"alGenFilters"]
        _alDeleteFilters = loaded[b"alDeleteFilters"]
        _alIsFilter = loaded[b"alIsFilter"]
        _alFilteri = loaded[b"alFilteri"]
        _alFilterf = loaded[b"alFilterf"]
        _alGenAuxiliaryEffectSlots = loaded[b"alGenAuxiliaryEffectSlots"]
        _alDeleteAuxiliaryEffectSlots = loaded[b"alDeleteAuxiliaryEffectSlots"]
        _alIsAuxiliaryEffectSlot = loaded[b"alIsAuxiliaryEffectSlot"]
        _alAuxiliaryEffectSloti = loaded[b"alAuxiliaryEffectSloti"]
        _alAuxiliaryEffectSlotf = loaded[b"alAuxiliaryEffectSlotf"]
        _resolve_enum(almod, b"AL_EFFECT_TYPE", AL_EFFECT_TYPE)
        _resolve_enum(almod, b"AL_EFFECT_REVERB", AL_EFFECT_REVERB)
        _resolve_enum(almod, b"AL_FILTER_TYPE", AL_FILTER_TYPE)
        _resolve_enum(almod, b"AL_FILTER_LOWPASS", AL_FILTER_LOWPASS)
        _resolve_enum(almod, b"AL_AUXILIARY_SEND_FILTER", AL_AUXILIARY_SEND_FILTER)
        _resolve_enum(almod, b"AL_DIRECT_FILTER", AL_DIRECT_FILTER)
        try:
            _max_aux_sends = 1
            try:
                import openal.alc as _alcq
                _val = _alcq.alcGetEnumValue(dev, b"ALC_MAX_AUXILIARY_SENDS")
                if _val:
                    _max_aux_sends = max(1, int(_val))
            except Exception:
                pass
        except Exception:
            _max_aux_sends = 1
        _efx_initialized = True
        _efx_available = True
        _efx_no_device = False
        return True
    except Exception:
        _efx_initialized = True
        _efx_available = False
        _efx_no_device = False
        return False


def efx_available() -> bool:
    if _efx_available:
        return True
    if _efx_initialized and not _efx_no_device:
        return False
    return ensure_efx()


def _effect_type_enum() -> int:
    cached = _enum_cache.get(b"AL_EFFECT_TYPE")
    if cached is not None:
        return int(cached)
    return AL_EFFECT_TYPE


def _filter_type_enum() -> int:
    cached = _enum_cache.get(b"AL_FILTER_TYPE")
    if cached is not None:
        return int(cached)
    return AL_FILTER_TYPE


def _reverb_type_enum() -> int:
    cached = _enum_cache.get(b"AL_EFFECT_REVERB")
    if cached is not None:
        return int(cached)
    return AL_EFFECT_REVERB


def _lowpass_enum() -> int:
    cached = _enum_cache.get(b"AL_FILTER_LOWPASS")
    if cached is not None:
        return int(cached)
    return AL_FILTER_LOWPASS


def create_effect() -> int:
    if not efx_available():
        raise EFXError("EFX not available")
    effect_id = ctypes.c_uint()
    _alGenEffects(1, ctypes.pointer(effect_id))
    if effect_id.value == 0:
        raise EFXError("Failed to create effect")
    return effect_id.value


def delete_effect(effect_id: int):
    if effect_id == 0:
        return
    if not _efx_available or _alDeleteEffects is None:
        return
    try:
        val = ctypes.c_uint(int(effect_id))
        _alDeleteEffects(1, ctypes.pointer(val))
    except Exception:
        pass


def is_effect(effect_id: int) -> bool:
    if effect_id == 0 or not _efx_available or _alIsEffect is None:
        return False
    try:
        return bool(_alIsEffect(int(effect_id)))
    except Exception:
        return False


def set_effect_type(effect_id: int, etype: int):
    if not _efx_available or _alEffecti is None:
        return
    try:
        _alEffecti(int(effect_id), int(_effect_type_enum()), int(etype))
    except Exception:
        pass


def set_effect_param_i(effect_id: int, param: int, value: int):
    if not _efx_available or _alEffecti is None:
        return
    try:
        _alEffecti(int(effect_id), int(param), int(value))
    except Exception:
        pass


def set_effect_param_f(effect_id: int, param: int, value: float):
    if not _efx_available or _alEffectf is None:
        return
    try:
        _alEffectf(int(effect_id), int(param), float(value))
    except Exception:
        pass


def create_filter() -> int:
    if not efx_available():
        raise EFXError("EFX not available")
    filter_id = ctypes.c_uint()
    _alGenFilters(1, ctypes.pointer(filter_id))
    if filter_id.value == 0:
        raise EFXError("Failed to create filter")
    return filter_id.value


def delete_filter(filter_id: int):
    if filter_id == 0:
        return
    if not _efx_available or _alDeleteFilters is None:
        return
    try:
        val = ctypes.c_uint(int(filter_id))
        _alDeleteFilters(1, ctypes.pointer(val))
    except Exception:
        pass


def is_filter(filter_id: int) -> bool:
    if filter_id == 0 or not _efx_available or _alIsFilter is None:
        return False
    try:
        return bool(_alIsFilter(int(filter_id)))
    except Exception:
        return False


def set_filter_type(filter_id: int, ftype: int):
    if not _efx_available or _alFilteri is None:
        return
    try:
        _alFilteri(int(filter_id), int(_filter_type_enum()), int(ftype))
    except Exception:
        pass


def set_filter_param_i(filter_id: int, param: int, value: int):
    if not _efx_available or _alFilteri is None:
        return
    try:
        _alFilteri(int(filter_id), int(param), int(value))
    except Exception:
        pass


def set_filter_param_f(filter_id: int, param: int, value: float):
    if not _efx_available or _alFilterf is None:
        return
    try:
        _alFilterf(int(filter_id), int(param), float(value))
    except Exception:
        pass


def create_aux_slot() -> int:
    if not efx_available():
        raise EFXError("EFX not available")
    slot_id = ctypes.c_uint()
    _alGenAuxiliaryEffectSlots(1, ctypes.pointer(slot_id))
    if slot_id.value == 0:
        raise EFXError("Failed to create auxiliary effect slot")
    return slot_id.value


def delete_aux_slot(slot_id: int):
    if slot_id == 0:
        return
    if not _efx_available or _alDeleteAuxiliaryEffectSlots is None:
        return
    try:
        val = ctypes.c_uint(int(slot_id))
        _alDeleteAuxiliaryEffectSlots(1, ctypes.pointer(val))
    except Exception:
        pass


def is_aux_slot(slot_id: int) -> bool:
    if slot_id == 0 or not _efx_available or _alIsAuxiliaryEffectSlot is None:
        return False
    try:
        return bool(_alIsAuxiliaryEffectSlot(int(slot_id)))
    except Exception:
        return False


def set_aux_slot_effect(slot_id: int, effect_id: int):
    if not _efx_available or _alAuxiliaryEffectSloti is None:
        return
    try:
        _alAuxiliaryEffectSloti(int(slot_id), int(AL_EFFECTSLOT_EFFECT), int(effect_id))
    except Exception:
        pass


def set_aux_slot_gain(slot_id: int, gain: float):
    if not _efx_available or _alAuxiliaryEffectSlotf is None:
        return
    try:
        v = max(0.0, min(1.0, float(gain)))
        _alAuxiliaryEffectSlotf(int(slot_id), int(AL_EFFECTSLOT_GAIN), v)
    except Exception:
        pass


def apply_reverb_params(effect_id: int, density: float, diffusion: float, gain: float, gain_hf: float, decay_time: float, decay_hf_ratio: float, reflections_gain: float, reflections_delay: float, late_gain: float, late_delay: float, air_gain_hf: float, room_rolloff: float, decay_hf_limit: int) -> bool:
    if not _efx_available or int(effect_id) == 0:
        return False
    try:
        set_effect_type(int(effect_id), int(_reverb_type_enum()))
        set_effect_param_f(int(effect_id), AL_REVERB_DENSITY, max(0.0, min(1.0, float(density))))
        set_effect_param_f(int(effect_id), AL_REVERB_DIFFUSION, max(0.0, min(1.0, float(diffusion))))
        set_effect_param_f(int(effect_id), AL_REVERB_GAIN, max(0.0, min(1.0, float(gain))))
        set_effect_param_f(int(effect_id), AL_REVERB_GAINHF, max(0.0, min(1.0, float(gain_hf))))
        set_effect_param_f(int(effect_id), AL_REVERB_DECAY_TIME, max(0.1, min(20.0, float(decay_time))))
        set_effect_param_f(int(effect_id), AL_REVERB_DECAY_HFRATIO, max(0.1, min(2.0, float(decay_hf_ratio))))
        set_effect_param_f(int(effect_id), AL_REVERB_REFLECTIONS_GAIN, max(0.0, min(3.16, float(reflections_gain))))
        set_effect_param_f(int(effect_id), AL_REVERB_REFLECTIONS_DELAY, max(0.0, min(0.3, float(reflections_delay))))
        set_effect_param_f(int(effect_id), AL_REVERB_LATE_REVERB_GAIN, max(0.0, min(10.0, float(late_gain))))
        set_effect_param_f(int(effect_id), AL_REVERB_LATE_REVERB_DELAY, max(0.0, min(0.1, float(late_delay))))
        set_effect_param_f(int(effect_id), AL_REVERB_AIR_ABSORPTION_GAINHF, max(0.892, min(1.0, float(air_gain_hf))))
        set_effect_param_f(int(effect_id), AL_REVERB_ROOM_ROLLOFF_FACTOR, max(0.0, min(10.0, float(room_rolloff))))
        set_effect_param_i(int(effect_id), AL_REVERB_DECAY_HFLIMIT, 1 if int(decay_hf_limit) else 0)
        return True
    except Exception:
        return False


def get_eax_preset(name: str):
    if not name:
        return None
    return EAX_PRESETS.get(str(name))


def eax_preset_names() -> list:
    return sorted(EAX_PRESETS.keys())


def apply_eax_preset(effect_id: int, preset_name: str) -> bool:
    preset = EAX_PRESETS.get(str(preset_name))
    if preset is None:
        preset = get_reverb_preset_as_eax(str(preset_name))
    if preset is None or int(effect_id) == 0:
        return False
    return bool(apply_reverb_params(int(effect_id), preset["density"], preset["diffusion"], preset["gain"], preset["gain_hf"], preset["decay_time"], preset["decay_hf_ratio"], preset["reflections_gain"], preset["reflections_delay"], preset["late_gain"], preset["late_delay"], preset["air_gain_hf"], preset["room_rolloff"], preset["decay_hf_limit"]))


def mb_to_gain(mb: float) -> float:
    try:
        v = float(mb)
    except Exception:
        return 0.0
    if v <= -10000.0:
        return 0.0
    if v >= 2000.0:
        return 10.0
    return float(10.0 ** (v / 2000.0))


def gain_to_mb(gain: float) -> float:
    try:
        g = float(gain)
    except Exception:
        return -10000.0
    if g <= 0.00001:
        return -10000.0
    if g >= 10.0:
        return 2000.0
    import math as _math
    return float(2000.0 * _math.log10(g))


def normalize_preset_name(name: str) -> str:
    try:
        raw = str(name or "").strip()
    except Exception:
        return "User"
    if not raw:
        return "User"
    low = raw.lower().replace("_", " ").replace("-", " ")
    compact = "".join(low.split())
    mapping = {
        "off": "Off",
        "generic": "Generic",
        "paddedcell": "Padded Cell",
        "padded": "Padded Cell",
        "room": "Room",
        "bathroom": "Bathroom",
        "livingroom": "Living Room",
        "stoneroom": "Stone Room",
        "auditorium": "Auditorium",
        "concerthall": "Concert Hall",
        "cave": "Cave",
        "arena": "Arena",
        "hangar": "Hangar",
        "carpetedhallway": "Carpeted Hallway",
        "hallway": "Hallway",
        "stonecorridor": "Stone Corridor",
        "alley": "Alley",
        "forest": "Forest",
        "city": "City",
        "mountains": "Mountains",
        "quarry": "Quarry",
        "plain": "Plain",
        "parkinglot": "Parking Lot",
        "sewerpipe": "Sewer Pipe",
        "underwater": "Underwater",
        "drugged": "Drugged",
        "dizzy": "Dizzy",
        "psychotic": "Psychotic",
        "user": "User",
        "custom": "User",
    }
    if compact in mapping:
        return mapping[compact]
    for ordered in REVERB_PRESET_ORDER:
        if ordered.lower().replace(" ", "") == compact:
            return ordered
    return "User"


def get_reverb_preset(name: str):
    if not name:
        return None
    normalized = normalize_preset_name(name)
    preset = REVERB_PRESETS_MB.get(normalized)
    if preset is not None:
        return dict(preset)
    return None


def reverb_preset_names() -> list:
    return list(REVERB_PRESET_ORDER)


def get_reverb_preset_as_eax(name: str):
    preset = get_reverb_preset(name)
    if preset is None:
        return None
    try:
        return {
            "density": max(0.0, min(1.0, float(preset["density"]) / 100.0)),
            "diffusion": max(0.0, min(1.0, float(preset["diffusion"]) / 100.0)),
            "gain": float(mb_to_gain(preset["room"])),
            "gain_hf": float(mb_to_gain(preset["room_hf"])),
            "decay_time": float(preset["decay_time"]),
            "decay_hf_ratio": float(preset["decay_hf_ratio"]),
            "reflections_gain": float(mb_to_gain(preset["reflections"])),
            "reflections_delay": float(preset["reflections_delay"]),
            "late_gain": float(mb_to_gain(preset["reverb"])),
            "late_delay": float(preset["reverb_delay"]),
            "air_gain_hf": 0.994,
            "room_rolloff": float(preset.get("room_rolloff", 0.0)),
            "decay_hf_limit": 1,
        }
    except Exception:
        return None


def apply_reverb_mb(effect_id: int, room: float, room_hf: float, room_lf: float, decay_time: float, decay_hf_ratio: float, reflections: float, reflections_delay: float, reverb: float, reverb_delay: float, diffusion: float, density: float, hf_reference: float, lf_reference: float, room_rolloff: float) -> bool:
    if not _efx_available or int(effect_id) == 0:
        return False
    try:
        eax_type = 32768
        try:
            cached = _enum_cache.get(b"AL_EFFECT_EAXREVERB")
            if cached is not None:
                eax_type = int(cached)
            else:
                import openal.al as _almod
                eax_type = int(_resolve_enum(_almod, b"AL_EFFECT_EAXREVERB", 32768))
        except Exception:
            eax_type = 32768
        set_effect_type(int(effect_id), int(eax_type))
        gain = float(mb_to_gain(room))
        gain_hf = float(mb_to_gain(room_hf))
        gain_lf = float(mb_to_gain(room_lf))
        refl_gain = float(mb_to_gain(reflections))
        late_gain = float(mb_to_gain(reverb))
        dens = max(0.0, min(1.0, float(density) / 100.0))
        diff = max(0.0, min(1.0, float(diffusion) / 100.0))
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_DENSITY), float(dens))
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_DIFFUSION), float(diff))
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_GAIN), max(0.0, min(1.0, float(gain))))
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_GAINHF), max(0.0, min(1.0, float(gain_hf))))
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_GAINLF), max(0.0, min(1.0, float(gain_lf))))
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_DECAY_TIME), max(0.1, min(20.0, float(decay_time))))
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_DECAY_HFRATIO), max(0.1, min(2.0, float(decay_hf_ratio))))
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_DECAY_LFRATIO), 1.0)
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_REFLECTIONS_GAIN), max(0.0, min(3.16, float(refl_gain))))
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_REFLECTIONS_DELAY), max(0.0, min(0.3, float(reflections_delay))))
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_LATE_REVERB_GAIN), max(0.0, min(10.0, float(late_gain))))
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_LATE_REVERB_DELAY), max(0.0, min(0.1, float(reverb_delay))))
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_ECHO_TIME), 0.25)
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_ECHO_DEPTH), 0.0)
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_MODULATION_TIME), 0.25)
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_MODULATION_DEPTH), 0.0)
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_AIR_ABSORPTION_GAINHF), 0.994)
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_HFREFERENCE), max(1000.0, min(20000.0, float(hf_reference))))
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_LFREFERENCE), max(20.0, min(1000.0, float(lf_reference))))
        set_effect_param_f(int(effect_id), int(AL_EAXREVERB_ROOM_ROLLOFF_FACTOR), max(0.0, min(10.0, float(room_rolloff))))
        set_effect_param_i(int(effect_id), int(AL_EAXREVERB_DECAY_HFLIMIT), 1)
        return True
    except Exception:
        pass
    try:
        return bool(apply_reverb_params(int(effect_id), max(0.0, min(1.0, float(density) / 100.0)), max(0.0, min(1.0, float(diffusion) / 100.0)), float(mb_to_gain(room)), float(mb_to_gain(room_hf)), float(decay_time), float(decay_hf_ratio), float(mb_to_gain(reflections)), float(reflections_delay), float(mb_to_gain(reverb)), float(reverb_delay), 0.994, float(room_rolloff), 1))
    except Exception:
        return False


def get_max_aux_sends() -> int:
    if not _efx_available:
        return 1
    try:
        return max(1, int(_max_aux_sends) if int(_max_aux_sends) > 0 else 1)
    except Exception:
        return 1


def get_aux_send_enum() -> int:
    cached = _enum_cache.get(b"AL_AUXILIARY_SEND_FILTER")
    if cached is not None:
        try:
            return int(cached)
        except Exception:
            pass
    if _efx_available:
        try:
            import openal.al as almod
            return int(_resolve_enum(almod, b"AL_AUXILIARY_SEND_FILTER", AL_AUXILIARY_SEND_FILTER))
        except Exception:
            pass
    return int(AL_AUXILIARY_SEND_FILTER)


def get_direct_filter_enum() -> int:
    cached = _enum_cache.get(b"AL_DIRECT_FILTER")
    if cached is not None:
        try:
            return int(cached)
        except Exception:
            pass
    if _efx_available:
        try:
            import openal.al as almod
            return int(_resolve_enum(almod, b"AL_DIRECT_FILTER", AL_DIRECT_FILTER))
        except Exception:
            pass
    return int(AL_DIRECT_FILTER)


def set_source_aux_send(source_id: int, slot_id: int, filter_id: int = 0, send_index: int = 0) -> bool:
    if int(source_id) == 0:
        return False
    if not _efx_available:
        return False
    try:
        import openal.al as almod
        send_enum = int(get_aux_send_enum())
        sends = int(get_max_aux_sends())
        idx = max(0, min(int(send_index), max(0, sends - 1)))
        try:
            almod.alSource3i(int(source_id), int(send_enum), int(slot_id), int(idx), int(filter_id))
        except Exception:
            return False
        try:
            err = int(almod.alGetError())
            return err == 0
        except Exception:
            return True
    except Exception:
        return False


def set_source_direct_filter(source_id: int, filter_id: int = 0) -> bool:
    if int(source_id) == 0:
        return False
    if not _efx_available:
        return False
    try:
        import openal.al as almod
        direct_enum = int(get_direct_filter_enum())
        try:
            almod.alSourcei(int(source_id), int(direct_enum), int(filter_id))
        except Exception:
            return False
        try:
            err = int(almod.alGetError())
            return err == 0
        except Exception:
            return True
    except Exception:
        return False


def _read_audio_flag(key: str, default: bool) -> bool:
    try:
        from core.engine.engine import Engine
        eng = Engine.instance()
        if eng is not None and getattr(eng, "_project_path", None):
            try:
                from core.config.config import get_project_config
                pcfg = get_project_config(eng._project_path)
                value = pcfg.get(key, None)
                if value is not None:
                    return bool(value)
            except Exception:
                pass
    except Exception:
        pass
    try:
        from core.config.config import get_global_config
        cfg = get_global_config()
        value = cfg.get(key, default)
        return bool(value) if value is not None else bool(default)
    except Exception:
        return bool(default)


def reverb_enabled() -> bool:
    return bool(_read_audio_flag("audio.enable_reverb", True))


def occlusion_enabled() -> bool:
    return bool(_read_audio_flag("audio.enable_occlusion", True))


def spatialization_enabled() -> bool:
    return bool(_read_audio_flag("audio.enable_spatialization", True))
