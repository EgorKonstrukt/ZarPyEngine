# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Callable
import os
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
        self._mouse_axes: set[str] = set()
        self._buttons: dict[str, InputButton] = {}
        self._event_callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._input_enabled: bool = True
        self._cursor_locked: bool = False
        self._cursor_visible: bool = True
        self._frame_count: int = 0
        self._time: float = 0.0
        self._last_time: float = time.perf_counter()
        self._dt: float = 0.0
        self._mouse_sensitivity: float = 1.0
        self._invert_mouse_x: bool = False
        self._invert_mouse_y: bool = False
        self._control_scheme: str = "fps"

    def define_axis(self, name: str, axis: InputAxis):
        self._axes[name] = axis

    def define_button(self, name: str, button: InputButton):
        self._buttons[name] = button

    def define_mouse_axis(self, name: str):
        self._mouse_axes.add(name.lower())

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
        now = time.perf_counter()
        self._dt = max(0.0, min(0.1, now - self._last_time))
        self._last_time = now
        self._frame_count += 1
        self._time = now
        self._frame_down = set(self._acc_down)
        self._frame_up = set(self._acc_up)
        self._acc_down.clear()
        self._acc_up.clear()
        self._any_key_down = False
        self._any_key = bool(self._held or self._mouse_held or self._frame_down)
        self._update_axes(self._dt)
        self._fire_event("frame_begin", None)

    def end_frame(self):
        self._scroll_delta = (0.0, 0.0)
        self._mouse_delta = (0.0, 0.0)
        self._fire_event("frame_end", None)

    def _update_axes(self, dt: float):
        for axis in self._axes.values():
            positive = any(self._held.get(k, False) for k in axis.positive) or \
                       any(self._held.get(k, False) for k in axis.alt_positive)
            negative = any(self._held.get(k, False) for k in axis.negative) or \
                       any(self._held.get(k, False) for k in axis.alt_negative)
            raw = (1.0 if positive else 0.0) - (1.0 if negative else 0.0)
            axis._raw_value = raw
            if abs(raw) < axis.dead or (axis.snap and positive and negative):
                raw = 0.0
            if raw != 0:
                axis._value += raw * axis.sensitivity * dt
                axis._value = max(-1.0, min(1.0, axis._value))
            else:
                if axis._value > 0:
                    axis._value = max(0.0, axis._value - axis.gravity * dt)
                elif axis._value < 0:
                    axis._value = min(0.0, axis._value + axis.gravity * dt)

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
        if not cls._state._input_enabled:
            return 0.0
        if name.lower() in cls._state._mouse_axes:
            return cls._get_mouse_axis(name)
        axis = cls._state._axes.get(name)
        if not axis:
            return 0.0
        return -axis._value if axis.invert else axis._value

    @classmethod
    def _get_mouse_axis(cls, name: str) -> float:
        dx, dy = cls._state._mouse_delta
        sens = cls._state._mouse_sensitivity
        if name.lower() == "mouse x":
            v = dx * sens
            return -v if cls._state._invert_mouse_x else v
        v = dy * sens
        return -v if cls._state._invert_mouse_y else v

    @classmethod
    def GetAxisRaw(cls, name: str) -> float:
        if not cls._state._input_enabled:
            return 0.0
        if name.lower() in cls._state._mouse_axes:
            return cls._get_mouse_axis(name)
        axis = cls._state._axes.get(name)
        if not axis:
            return 0.0
        return -axis._raw_value if axis.invert else axis._raw_value

    @classmethod
    def DefineAxis(cls, name: str, positive: list[int], negative: list[int],
                   alt_positive: list[int] = None, alt_negative: list[int] = None,
                   gravity: float = 3.0, dead: float = 0.001, sensitivity: float = 1.0,
                   snap: bool = False, invert: bool = False):
        cls._state.define_axis(name, InputAxis(positive, negative, alt_positive, alt_negative,
                                                gravity, dead, sensitivity, snap, invert))

    @classmethod
    def DefineMouseAxis(cls, name: str):
        cls._state.define_mouse_axis(name)

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

    @classproperty
    def deltaTime(cls) -> float:
        return cls._state._dt

    @classproperty
    def mouseSensitivity(cls) -> float:
        return cls._state._mouse_sensitivity

    @classmethod
    def set_mouse_sensitivity(cls, value: float):
        cls._state._mouse_sensitivity = max(0.0, float(value))

    @classproperty
    def invertMouseY(cls) -> bool:
        return cls._state._invert_mouse_y

    @classmethod
    def set_invert_mouse_y(cls, value: bool):
        cls._state._invert_mouse_y = bool(value)

    @classproperty
    def invertMouseX(cls) -> bool:
        return cls._state._invert_mouse_x

    @classmethod
    def set_invert_mouse_x(cls, value: bool):
        cls._state._invert_mouse_x = bool(value)

    @classproperty
    def controlScheme(cls) -> str:
        return cls._state._control_scheme

    @classmethod
    def set_control_scheme(cls, value: str):
        cls._state._control_scheme = "tps" if str(value).lower() == "tps" else "fps"

    @classmethod
    def LoadProjectBindings(cls, project_path: str = None):
        from core.config.config import get_project_config
        cfg = get_project_config(project_path) if project_path else get_project_config(os.getcwd())
        inp = cfg.to_dict().get("input", {})

        scheme = str(inp.get("control_scheme", "fps")).lower()
        cls.set_control_scheme("tps" if scheme == "tps" else "fps")
        cls.set_mouse_sensitivity(inp.get("mouse_sensitivity", 1.0))
        cls.set_invert_mouse_y(bool(inp.get("invert_mouse_y", False)))
        cls.set_invert_mouse_x(bool(inp.get("invert_mouse_x", False)))

        gravity = float(inp.get("axis_gravity", 3.0))
        sensitivity = float(inp.get("axis_sensitivity", 1.0))
        dead = float(inp.get("axis_dead", 0.001))

        cls._state._axes.clear()
        cls._state._buttons.clear()
        cls._state._mouse_axes.clear()

        axis_map = {
            "horizontal": ("Horizontal", None),
            "vertical": ("Vertical", None),
            "fire_axis": ("Fire", None),
            "jump_axis": ("Jump", None),
        }
        for key, (axis_name, _) in axis_map.items():
            binding = inp.get(key)
            if binding:
                pos, neg = cls._split_axis_binding(binding)
                if pos or neg:
                    cls.DefineAxis(axis_name, pos, neg, gravity=gravity, dead=dead, sensitivity=sensitivity)

        for btn_key, btn_name in [("jump", "Jump"), ("fire", "Fire"),
                                  ("crouch", "Crouch"), ("sprint", "Sprint"),
                                  ("interact", "Interact"), ("reload", "Reload")]:
            binding = inp.get(btn_key)
            if not binding:
                continue
            keys, alt = cls._split_button_binding(binding)
            alt_binding = inp.get("alt_" + btn_key)
            if alt_binding:
                _, extra_alt = cls._split_button_binding(alt_binding)
                alt = alt + extra_alt
            if keys or alt:
                cls.DefineButton(btn_name, keys, alt)

        for mkey in ("mouse_axis_x", "mouse_axis_y"):
            mname = inp.get(mkey)
            if not mname:
                mname = "Mouse X" if mkey == "mouse_axis_x" else "Mouse Y"
            cls.DefineMouseAxis(mname)

    @staticmethod
    def _resolve_code(part: str):
        part = part.strip().lower()
        if part.startswith("mouse"):
            rest = part[5:].strip()
            if rest.isdigit():
                return KeyCode.MOUSE_LEFT + int(rest)
            named = {"left": KeyCode.MOUSE_LEFT, "right": KeyCode.MOUSE_RIGHT,
                     "middle": KeyCode.MOUSE_MIDDLE, "back": KeyCode.MOUSE_BACK,
                     "forward": KeyCode.MOUSE_FORWARD}
            return named.get(rest, 0)
        return KeyCode.from_name(part)

    @staticmethod
    def _split_axis_binding(binding) -> tuple:
        if isinstance(binding, (list, tuple)):
            parts = list(binding)
        else:
            parts = str(binding).split(",")
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return [], []
        explicit = any(p.startswith("-") for p in parts)
        positive, negative = [], []
        if explicit:
            for part in parts:
                if part.startswith("-"):
                    code = Input._resolve_code(part[1:])
                    if code:
                        negative.append(code)
                else:
                    code = Input._resolve_code(part)
                    if code:
                        positive.append(code)
        elif len(parts) == 2:
            neg = Input._resolve_code(parts[0])
            pos = Input._resolve_code(parts[1])
            if neg:
                negative.append(neg)
            if pos:
                positive.append(pos)
        else:
            for part in parts:
                code = Input._resolve_code(part)
                if code:
                    positive.append(code)
        return positive, negative

    @staticmethod
    def _split_button_binding(binding) -> tuple:
        if isinstance(binding, (list, tuple)):
            parts = list(binding)
        else:
            parts = str(binding).split(",")
        keys = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            code = Input._resolve_code(part)
            if code:
                keys.append(code)
        return keys, []

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
