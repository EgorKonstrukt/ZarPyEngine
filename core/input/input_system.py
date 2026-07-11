# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Callable
import time
from collections import defaultdict

class KeyCode:
    SPACE = 32
    APOSTROPHE = 39
    COMMA = 44
    MINUS = 45
    PERIOD = 46
    SLASH = 47
    KEY_0 = 48
    KEY_1 = 49
    KEY_2 = 50
    KEY_3 = 51
    KEY_4 = 52
    KEY_5 = 53
    KEY_6 = 54
    KEY_7 = 55
    KEY_8 = 56
    KEY_9 = 57
    SEMICOLON = 59
    EQUAL = 61
    A = 65
    B = 66
    C = 67
    D = 68
    E = 69
    F = 70
    G = 71
    H = 72
    I = 73
    J = 74
    K = 75
    L = 76
    M = 77
    N = 78
    O = 79
    P = 80
    Q = 81
    R = 82
    S = 83
    T = 84
    U = 85
    V = 86
    W = 87
    X = 88
    Y = 89
    Z = 90
    LEFT_BRACKET = 91
    BACKSLASH = 92
    RIGHT_BRACKET = 93
    GRAVE_ACCENT = 96
    WORLD_1 = 161
    WORLD_2 = 162
    ESCAPE = 256
    ENTER = 257
    TAB = 258
    BACKSPACE = 259
    INSERT = 260
    DELETE = 261
    RIGHT = 262
    LEFT = 263
    DOWN = 264
    UP = 265
    PAGE_UP = 266
    PAGE_DOWN = 267
    HOME = 268
    END = 269
    CAPS_LOCK = 280
    SCROLL_LOCK = 281
    NUM_LOCK = 282
    PRINT_SCREEN = 283
    PAUSE = 284
    F1 = 290
    F2 = 291
    F3 = 292
    F4 = 293
    F5 = 294
    F6 = 295
    F7 = 296
    F8 = 297
    F9 = 298
    F10 = 299
    F11 = 300
    F12 = 301
    F13 = 302
    F14 = 303
    F15 = 304
    F16 = 305
    F17 = 306
    F18 = 307
    F19 = 308
    F20 = 309
    F21 = 310
    F22 = 311
    F23 = 312
    F24 = 313
    F25 = 314
    KP_0 = 320
    KP_1 = 321
    KP_2 = 322
    KP_3 = 323
    KP_4 = 324
    KP_5 = 325
    KP_6 = 326
    KP_7 = 327
    KP_8 = 328
    KP_9 = 329
    KP_DECIMAL = 330
    KP_DIVIDE = 331
    KP_MULTIPLY = 332
    KP_SUBTRACT = 333
    KP_ADD = 334
    KP_ENTER = 335
    KP_EQUAL = 336
    LEFT_SHIFT = 340
    LEFT_CONTROL = 341
    LEFT_ALT = 342
    LEFT_SUPER = 343
    RIGHT_SHIFT = 344
    RIGHT_CONTROL = 345
    RIGHT_ALT = 346
    RIGHT_SUPER = 347
    MENU = 348
    MOUSE_LEFT = 1000
    MOUSE_RIGHT = 1001
    MOUSE_MIDDLE = 1002
    MOUSE_BACK = 1003
    MOUSE_FORWARD = 1004

    _NAME_MAP: dict[str, int] = None

    @classmethod
    def from_name(cls, name: str) -> int:
        if cls._NAME_MAP is None:
            cls._NAME_MAP = {}
            for attr_name in dir(cls):
                if attr_name.isupper():
                    val = getattr(cls, attr_name)
                    if isinstance(val, int):
                        cls._NAME_MAP[attr_name] = val
                        cls._NAME_MAP[attr_name.lower()] = val
        return cls._NAME_MAP.get(name, cls._NAME_MAP.get(name.upper(), 0))

class InputAxis:
    def __init__(self, positive: list[int] = None, negative: list[int] = None,
                 alt_positive: list[int] = None, alt_negative: list[int] = None,
                 gravity: float = 3.0, dead: float = 0.001, sensitivity: float = 1.0,
                 snap: bool = False, invert: bool = False):
        self.positive = positive or []
        self.negative = negative or []
        self.alt_positive = alt_positive or []
        self.alt_negative = alt_negative or []
        self.gravity = gravity
        self.dead = dead
        self.sensitivity = sensitivity
        self.snap = snap
        self.invert = invert
        self._value: float = 0.0
        self._raw_value: float = 0.0

class InputButton:
    def __init__(self, keys: list[int] = None, alt_keys: list[int] = None):
        self.keys = keys or []
        self.alt_keys = alt_keys or []

class InputState:
    def __init__(self):
        self._held: dict[int, bool] = {}
        self._mouse_held: dict[int, bool] = {}
        self._acc_down: set[int] = set()
        self._acc_up: set[int] = set()
        self._frame_down: set[int] = set()
        self._frame_up: set[int] = set()
        self._mouse_pos: tuple[float, float] = (0.0, 0.0)
        self._mouse_delta: tuple[float, float] = (0.0, 0.0)
        self._scroll_delta: tuple[float, float] = (0.0, 0.0)
        self._any_key_down: bool = False
        self._any_key: bool = False
        self._axes: dict[str, InputAxis] = {}
        self._buttons: dict[str, InputButton] = {}
        self._event_callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._input_enabled: bool = True
        self._cursor_locked: bool = False
        self._cursor_visible: bool = True
        self._frame_count: int = 0
        self._time: float = 0.0

    def define_axis(self, name: str, axis: InputAxis):
        self._axes[name] = axis

    def define_button(self, name: str, button: InputButton):
        self._buttons[name] = button

    def on_event(self, event_name: str, callback: Callable):
        self._event_callbacks[event_name].append(callback)

    def off_event(self, event_name: str, callback: Callable):
        if event_name in self._event_callbacks:
            try:
                self._event_callbacks[event_name].remove(callback)
            except ValueError:
                pass

    def _fire_event(self, event_name: str, data=None):
        for cb in self._event_callbacks.get(event_name, []):
            try:
                cb(data)
            except Exception:
                pass

    def begin_frame(self):
        self._frame_count += 1
        self._time = time.perf_counter()
        self._frame_down = set(self._acc_down)
        self._frame_up = set(self._acc_up)
        self._acc_down.clear()
        self._acc_up.clear()
        self._scroll_delta = (0.0, 0.0)
        self._any_key_down = False
        self._fire_event("frame_begin", None)

    def end_frame(self):
        self._fire_event("frame_end", None)

    @staticmethod
    def _is_mouse_key(key: int) -> bool:
        return key >= 1000

    def press_key(self, key: int):
        if self._is_mouse_key(key):
            if not self._mouse_held.get(key, False):
                self._acc_down.add(key)
            self._mouse_held[key] = True
        else:
            if not self._held.get(key, False):
                self._acc_down.add(key)
            self._held[key] = True
        self._any_key = True
        self._any_key_down = True
        self._fire_event("key_down", key)

    def release_key(self, key: int):
        if self._is_mouse_key(key):
            if self._mouse_held.get(key, False):
                self._acc_up.add(key)
            self._mouse_held[key] = False
        else:
            if self._held.get(key, False):
                self._acc_up.add(key)
            self._held[key] = False
        self._fire_event("key_up", key)

    def set_mouse_pos(self, x: float, y: float, delta_x: float = 0.0, delta_y: float = 0.0):
        self._mouse_pos = (x, y)
        self._mouse_delta = (self._mouse_delta[0] + delta_x, self._mouse_delta[1] + delta_y)

    def set_scroll(self, dx: float, dy: float):
        self._scroll_delta = (self._scroll_delta[0] + dx, self._scroll_delta[1] + dy)

    def reset_all(self):
        self._held.clear()
        self._mouse_held.clear()
        self._acc_down.clear()
        self._acc_up.clear()
        self._frame_down.clear()
        self._frame_up.clear()
        self._mouse_pos = (0.0, 0.0)
        self._mouse_delta = (0.0, 0.0)
        self._scroll_delta = (0.0, 0.0)

class classproperty:
    def __init__(self, fget):
        self.fget = fget
    def __get__(self, instance, owner):
        return self.fget(owner)

class classproperty_setter:
    def __init__(self, fget=None, fset=None):
        self.fget = fget
        self.fset = fset
    def __get__(self, instance, owner):
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(owner)
    def __set__(self, instance, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(type(instance) if instance else None, value)
    def setter(self, fset):
        return classproperty_setter(self.fget, fset)

class Input:
    _state: InputState = InputState()

    _INPUT_EVENTS = {
        "any_key_down", "any_key_up", "key_down", "key_up",
        "mouse_down", "mouse_up", "mouse_move", "scroll",
        "frame_begin", "frame_end",
    }

    @classmethod
    def _state_ref(cls) -> InputState:
        return cls._state

    @classmethod
    def begin_frame(cls):
        cls._state.begin_frame()

    @classmethod
    def end_frame(cls):
        cls._state.end_frame()

    @classmethod
    def GetKey(cls, key: int) -> bool:
        if not cls._state._input_enabled:
            return False
        return cls._state._held.get(key, False)

    @classmethod
    def GetKeyDown(cls, key: int) -> bool:
        if not cls._state._input_enabled:
            return False
        return key in cls._state._frame_down

    @classmethod
    def GetKeyUp(cls, key: int) -> bool:
        if not cls._state._input_enabled:
            return False
        return key in cls._state._frame_up

    @classmethod
    def GetMouseButton(cls, button: int) -> bool:
        if not cls._state._input_enabled:
            return False
        mk = KeyCode.MOUSE_LEFT + button
        return cls._state._mouse_held.get(mk, False)

    @classmethod
    def GetMouseButtonDown(cls, button: int) -> bool:
        if not cls._state._input_enabled:
            return False
        mk = KeyCode.MOUSE_LEFT + button
        return mk in cls._state._frame_down

    @classmethod
    def GetMouseButtonUp(cls, button: int) -> bool:
        if not cls._state._input_enabled:
            return False
        mk = KeyCode.MOUSE_LEFT + button
        return mk in cls._state._frame_up

    @classmethod
    def GetButton(cls, name: str) -> bool:
        btn = cls._state._buttons.get(name)
        if not btn:
            return False
        for k in btn.keys:
            if cls.GetKey(k):
                return True
        for k in btn.alt_keys:
            if cls.GetKey(k):
                return True
        return False

    @classmethod
    def GetButtonDown(cls, name: str) -> bool:
        btn = cls._state._buttons.get(name)
        if not btn:
            return False
        for k in btn.keys:
            if cls.GetKeyDown(k):
                return True
        for k in btn.alt_keys:
            if cls.GetKeyDown(k):
                return True
        return False

    @classmethod
    def GetButtonUp(cls, name: str) -> bool:
        btn = cls._state._buttons.get(name)
        if not btn:
            return False
        for k in btn.keys:
            if cls.GetKeyUp(k):
                return True
        for k in btn.alt_keys:
            if cls.GetKeyUp(k):
                return True
        return False

    @classmethod
    def GetAxis(cls, name: str) -> float:
        axis = cls._state._axes.get(name)
        if not axis or not cls._state._input_enabled:
            return 0.0
        positive = any(cls.GetKey(k) for k in axis.positive) or any(cls.GetKey(k) for k in axis.alt_positive)
        negative = any(cls.GetKey(k) for k in axis.negative) or any(cls.GetKey(k) for k in axis.alt_negative)
        raw = (1.0 if positive else 0.0) - (1.0 if negative else 0.0)
        axis._raw_value = raw
        if abs(raw) < axis.dead or (axis.snap and positive and negative):
            raw = 0.0
        dt = 0.016
        if raw != 0:
            axis._value += raw * axis.sensitivity * dt
            axis._value = max(-1.0, min(1.0, axis._value))
        else:
            if axis._value > 0:
                axis._value = max(0.0, axis._value - axis.gravity * dt)
            elif axis._value < 0:
                axis._value = min(0.0, axis._value + axis.gravity * dt)
        return -axis._value if axis.invert else axis._value

    @classmethod
    def GetAxisRaw(cls, name: str) -> float:
        axis = cls._state._axes.get(name)
        if not axis or not cls._state._input_enabled:
            return 0.0
        positive = any(cls.GetKey(k) for k in axis.positive) or any(cls.GetKey(k) for k in axis.alt_positive)
        negative = any(cls.GetKey(k) for k in axis.negative) or any(cls.GetKey(k) for k in axis.alt_negative)
        raw = (1.0 if positive else 0.0) - (1.0 if negative else 0.0)
        if abs(raw) < axis.dead or (axis.snap and positive and negative):
            raw = 0.0
        return -raw if axis.invert else raw

    @classmethod
    def DefineAxis(cls, name: str, positive: list[int], negative: list[int],
                   alt_positive: list[int] = None, alt_negative: list[int] = None,
                   gravity: float = 3.0, dead: float = 0.001, sensitivity: float = 1.0,
                   snap: bool = False, invert: bool = False):
        cls._state.define_axis(name, InputAxis(positive, negative, alt_positive, alt_negative,
                                                gravity, dead, sensitivity, snap, invert))

    @classmethod
    def DefineButton(cls, name: str, keys: list[int], alt_keys: list[int] = None):
        cls._state.define_button(name, InputButton(keys, alt_keys))

    @classproperty
    def mousePosition(cls) -> tuple[float, float]:
        return cls._state._mouse_pos

    @classproperty
    def mouseDelta(cls) -> tuple[float, float]:
        return cls._state._mouse_delta

    @classproperty
    def mouseScrollDelta(cls) -> tuple[float, float]:
        return cls._state._scroll_delta

    @classproperty
    def anyKey(cls) -> bool:
        return cls._state._any_key

    @classproperty
    def anyKeyDown(cls) -> bool:
        return cls._state._any_key_down

    @classproperty
    def inputEnabled(cls) -> bool:
        return cls._state._input_enabled

    @classmethod
    def set_input_enabled(cls, value: bool):
        cls._state._input_enabled = value

    @classproperty
    def cursorLocked(cls) -> bool:
        return cls._state._cursor_locked

    @classmethod
    def set_cursor_locked(cls, value: bool):
        cls._state._cursor_locked = value

    @classproperty
    def cursorVisible(cls) -> bool:
        return cls._state._cursor_visible

    @classmethod
    def set_cursor_visible(cls, value: bool):
        cls._state._cursor_visible = value

    @classmethod
    def OnEvent(cls, event_name: str, callback: Callable):
        if event_name in cls._INPUT_EVENTS:
            cls._state.on_event(event_name, callback)

    @classmethod
    def OffEvent(cls, event_name: str, callback: Callable):
        cls._state.off_event(event_name, callback)

    @classmethod
    def Reset(cls):
        cls._state.reset_all()
