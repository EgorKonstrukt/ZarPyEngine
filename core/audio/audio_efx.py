# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import ctypes
from typing import Optional

try:
    import openal
except Exception:
    openal = None

__all__ = [
    "EFXError", "efx_available", "ensure_efx",
    "create_effect", "delete_effect", "set_effect_type",
    "set_effect_param_i", "set_effect_param_f",
    "create_aux_slot", "delete_aux_slot",
    "set_aux_slot_effect", "set_aux_slot_gain",
]

class EFXError(RuntimeError):
    pass

_efx_initialized = False
_efx_available = False

_alGenEffects = None
_alDeleteEffects = None
_alEffecti = None
_alEffectf = None
_alGenAuxiliaryEffectSlots = None
_alDeleteAuxiliaryEffectSlots = None
_alAuxiliaryEffectSloti = None
_alAuxiliaryEffectSlotf = None

AL_EFFECT_NULL = 0
AL_EFFECT_REVERB = 1
AL_EFFECT_TYPE = 32769
AL_EFFECTSLOT_EFFECT = 1
AL_EFFECTSLOT_GAIN = 2
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


def ensure_efx() -> bool:
    global _efx_initialized, _efx_available, _alGenEffects, _alDeleteEffects
    global _alEffecti, _alEffectf
    global _alGenAuxiliaryEffectSlots, _alDeleteAuxiliaryEffectSlots
    global _alAuxiliaryEffectSloti, _alAuxiliaryEffectSlotf

    if _efx_initialized:
        return _efx_available

    _efx_initialized = True
    try:
        import openal.al as al
        from openal import alc

        dev = openal.oalGetDevice()
        if dev is None:
            return False
        if not alc.alcIsExtensionPresent(dev, b"ALC_EXT_EFX"):
            return False

        funcs = {
            b"alGenEffects": ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
            b"alDeleteEffects": ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
            b"alEffecti": ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_int, ctypes.c_int),
            b"alEffectf": ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_int, ctypes.c_float),
            b"alGenAuxiliaryEffectSlots": ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
            b"alDeleteAuxiliaryEffectSlots": ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
            b"alAuxiliaryEffectSloti": ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_int, ctypes.c_int),
            b"alAuxiliaryEffectSlotf": ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_int, ctypes.c_float),
        }

        for name, restype in funcs.items():
            ptr = al.alGetProcAddress(name)
            if not ptr:
                return False
            globals()["_al" + name[2:].decode()] = ctypes.cast(ptr, restype)

        _efx_available = True
        return True
    except Exception:
        return False


def efx_available() -> bool:
    if not _efx_initialized:
        ensure_efx()
    return _efx_available


def create_effect() -> int:
    if not efx_available():
        raise EFXError("EFX not available")
    effect_id = ctypes.c_uint()
    _alGenEffects(1, ctypes.pointer(effect_id))
    if effect_id.value == 0:
        raise EFXError("Failed to create effect")
    return effect_id.value


def delete_effect(effect_id: int):
    if not efx_available() or effect_id == 0:
        return
    val = ctypes.c_uint(effect_id)
    _alDeleteEffects(1, ctypes.pointer(val))


def set_effect_type(effect_id: int, etype: int):
    if not efx_available():
        return
    _alEffecti(effect_id, AL_EFFECT_TYPE, etype)


def set_effect_param_i(effect_id: int, param: int, value: int):
    if not efx_available():
        return
    _alEffecti(effect_id, param, value)


def set_effect_param_f(effect_id: int, param: int, value: float):
    if not efx_available():
        return
    _alEffectf(effect_id, param, value)


def create_aux_slot() -> int:
    if not efx_available():
        raise EFXError("EFX not available")
    slot_id = ctypes.c_uint()
    _alGenAuxiliaryEffectSlots(1, ctypes.pointer(slot_id))
    if slot_id.value == 0:
        raise EFXError("Failed to create auxiliary effect slot")
    return slot_id.value


def delete_aux_slot(slot_id: int):
    if not efx_available() or slot_id == 0:
        return
    val = ctypes.c_uint(slot_id)
    _alDeleteAuxiliaryEffectSlots(1, ctypes.pointer(val))


def set_aux_slot_effect(slot_id: int, effect_id: int):
    if not efx_available():
        return
    _alAuxiliaryEffectSloti(slot_id, AL_EFFECTSLOT_EFFECT, effect_id)


def set_aux_slot_gain(slot_id: int, gain: float):
    if not efx_available():
        return
    _alAuxiliaryEffectSlotf(slot_id, AL_EFFECTSLOT_GAIN, gain)
