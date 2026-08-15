# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import threading
import platform
from typing import Optional
from core.input.input_system import Input, KeyCode

try:
    from pynput import mouse, keyboard
    _pynput_available = True
    if platform.system() == "Linux" and "DISPLAY" not in __import__("os").environ:
        _pynput_available = False
except Exception:
    _pynput_available = False
    mouse = None
    keyboard = None

_PYNPUT_TO_KEY = {}
_QT_TO_KEY = {}

def _build_pynput_maps():
    if not _pynput_available or keyboard is None:
        return
    for c in 'abcdefghijklmnopqrstuvwxyz':
        _PYNPUT_TO_KEY[keyboard.KeyCode.from_char(c)] = getattr(KeyCode, c.upper())
        _PYNPUT_TO_KEY[keyboard.KeyCode.from_char(c.upper())] = getattr(KeyCode, c.upper())
    for d, code in [('0', KeyCode.KEY_0), ('1', KeyCode.KEY_1),
                    ('2', KeyCode.KEY_2), ('3', KeyCode.KEY_3),
                    ('4', KeyCode.KEY_4), ('5', KeyCode.KEY_5),
                    ('6', KeyCode.KEY_6), ('7', KeyCode.KEY_7),
                    ('8', KeyCode.KEY_8), ('9', KeyCode.KEY_9)]:
        _PYNPUT_TO_KEY[keyboard.KeyCode.from_char(d)] = code
    punctuation = [(' ', KeyCode.SPACE), ('-', KeyCode.MINUS), ('=', KeyCode.EQUAL),
                   ('[', KeyCode.LEFT_BRACKET), (']', KeyCode.RIGHT_BRACKET),
                   ('\\', KeyCode.BACKSLASH), (';', KeyCode.SEMICOLON),
                   ("'", KeyCode.APOSTROPHE), (',', KeyCode.COMMA),
                   ('.', KeyCode.PERIOD), ('/', KeyCode.SLASH), ('`', KeyCode.GRAVE_ACCENT)]
    for ch, code in punctuation:
        try:
            _PYNPUT_TO_KEY[keyboard.KeyCode.from_char(ch)] = code
        except Exception:
            pass
    special = {
        keyboard.Key.shift: KeyCode.LEFT_SHIFT, keyboard.Key.shift_r: KeyCode.RIGHT_SHIFT,
        keyboard.Key.ctrl: KeyCode.LEFT_CONTROL, keyboard.Key.ctrl_r: KeyCode.RIGHT_CONTROL,
        keyboard.Key.alt: KeyCode.LEFT_ALT, keyboard.Key.alt_r: KeyCode.RIGHT_ALT,
        keyboard.Key.alt_gr: KeyCode.RIGHT_ALT,
        keyboard.Key.cmd: KeyCode.LEFT_SUPER, keyboard.Key.cmd_r: KeyCode.RIGHT_SUPER,
        keyboard.Key.space: KeyCode.SPACE, keyboard.Key.enter: KeyCode.ENTER,
        keyboard.Key.backspace: KeyCode.BACKSPACE, keyboard.Key.tab: KeyCode.TAB,
        keyboard.Key.esc: KeyCode.ESCAPE,
        keyboard.Key.insert: KeyCode.INSERT, keyboard.Key.delete: KeyCode.DELETE,
        keyboard.Key.page_up: KeyCode.PAGE_UP, keyboard.Key.page_down: KeyCode.PAGE_DOWN,
        keyboard.Key.home: KeyCode.HOME, keyboard.Key.end: KeyCode.END,
        keyboard.Key.up: KeyCode.UP, keyboard.Key.down: KeyCode.DOWN,
        keyboard.Key.left: KeyCode.LEFT, keyboard.Key.right: KeyCode.RIGHT,
        keyboard.Key.caps_lock: KeyCode.CAPS_LOCK,
        keyboard.Key.num_lock: KeyCode.NUM_LOCK, keyboard.Key.scroll_lock: KeyCode.SCROLL_LOCK,
        keyboard.Key.print_screen: KeyCode.PRINT_SCREEN, keyboard.Key.pause: KeyCode.PAUSE,
        keyboard.Key.f1: KeyCode.F1, keyboard.Key.f2: KeyCode.F2,
        keyboard.Key.f3: KeyCode.F3, keyboard.Key.f4: KeyCode.F4,
        keyboard.Key.f5: KeyCode.F5, keyboard.Key.f6: KeyCode.F6,
        keyboard.Key.f7: KeyCode.F7, keyboard.Key.f8: KeyCode.F8,
        keyboard.Key.f9: KeyCode.F9, keyboard.Key.f10: KeyCode.F10,
        keyboard.Key.f11: KeyCode.F11, keyboard.Key.f12: KeyCode.F12,
    }
    _PYNPUT_TO_KEY.update(special)

def _build_qt_map():
    if _QT_TO_KEY:
        return
    from PyQt6.QtCore import Qt as QtC
    _QT_TO_KEY.update({
        QtC.Key.Key_Space: KeyCode.SPACE, QtC.Key.Key_Apostrophe: KeyCode.APOSTROPHE,
        QtC.Key.Key_Comma: KeyCode.COMMA, QtC.Key.Key_Minus: KeyCode.MINUS,
        QtC.Key.Key_Period: KeyCode.PERIOD, QtC.Key.Key_Slash: KeyCode.SLASH,
        QtC.Key.Key_0: KeyCode.KEY_0, QtC.Key.Key_1: KeyCode.KEY_1,
        QtC.Key.Key_2: KeyCode.KEY_2, QtC.Key.Key_3: KeyCode.KEY_3,
        QtC.Key.Key_4: KeyCode.KEY_4, QtC.Key.Key_5: KeyCode.KEY_5,
        QtC.Key.Key_6: KeyCode.KEY_6, QtC.Key.Key_7: KeyCode.KEY_7,
        QtC.Key.Key_8: KeyCode.KEY_8, QtC.Key.Key_9: KeyCode.KEY_9,
        QtC.Key.Key_Semicolon: KeyCode.SEMICOLON, QtC.Key.Key_Equal: KeyCode.EQUAL,
        QtC.Key.Key_A: KeyCode.A, QtC.Key.Key_B: KeyCode.B,
        QtC.Key.Key_C: KeyCode.C, QtC.Key.Key_D: KeyCode.D,
        QtC.Key.Key_E: KeyCode.E, QtC.Key.Key_F: KeyCode.F,
        QtC.Key.Key_G: KeyCode.G, QtC.Key.Key_H: KeyCode.H,
        QtC.Key.Key_I: KeyCode.I, QtC.Key.Key_J: KeyCode.J,
        QtC.Key.Key_K: KeyCode.K, QtC.Key.Key_L: KeyCode.L,
        QtC.Key.Key_M: KeyCode.M, QtC.Key.Key_N: KeyCode.N,
        QtC.Key.Key_O: KeyCode.O, QtC.Key.Key_P: KeyCode.P,
        QtC.Key.Key_Q: KeyCode.Q, QtC.Key.Key_R: KeyCode.R,
        QtC.Key.Key_S: KeyCode.S, QtC.Key.Key_T: KeyCode.T,
        QtC.Key.Key_U: KeyCode.U, QtC.Key.Key_V: KeyCode.V,
        QtC.Key.Key_W: KeyCode.W, QtC.Key.Key_X: KeyCode.X,
        QtC.Key.Key_Y: KeyCode.Y, QtC.Key.Key_Z: KeyCode.Z,
        QtC.Key.Key_BracketLeft: KeyCode.LEFT_BRACKET,
        QtC.Key.Key_Backslash: KeyCode.BACKSLASH,
        QtC.Key.Key_BracketRight: KeyCode.RIGHT_BRACKET,
        QtC.Key.Key_QuoteLeft: KeyCode.GRAVE_ACCENT,
        QtC.Key.Key_Escape: KeyCode.ESCAPE,
        QtC.Key.Key_Enter: KeyCode.ENTER, QtC.Key.Key_Return: KeyCode.ENTER,
        QtC.Key.Key_Tab: KeyCode.TAB, QtC.Key.Key_Backspace: KeyCode.BACKSPACE,
        QtC.Key.Key_Insert: KeyCode.INSERT, QtC.Key.Key_Delete: KeyCode.DELETE,
        QtC.Key.Key_Right: KeyCode.RIGHT, QtC.Key.Key_Left: KeyCode.LEFT,
        QtC.Key.Key_Down: KeyCode.DOWN, QtC.Key.Key_Up: KeyCode.UP,
        QtC.Key.Key_PageUp: KeyCode.PAGE_UP, QtC.Key.Key_PageDown: KeyCode.PAGE_DOWN,
        QtC.Key.Key_Home: KeyCode.HOME, QtC.Key.Key_End: KeyCode.END,
        QtC.Key.Key_CapsLock: KeyCode.CAPS_LOCK,
        QtC.Key.Key_ScrollLock: KeyCode.SCROLL_LOCK,
        QtC.Key.Key_NumLock: KeyCode.NUM_LOCK,
        QtC.Key.Key_Print: KeyCode.PRINT_SCREEN, QtC.Key.Key_Pause: KeyCode.PAUSE,
        QtC.Key.Key_F1: KeyCode.F1, QtC.Key.Key_F2: KeyCode.F2,
        QtC.Key.Key_F3: KeyCode.F3, QtC.Key.Key_F4: KeyCode.F4,
        QtC.Key.Key_F5: KeyCode.F5, QtC.Key.Key_F6: KeyCode.F6,
        QtC.Key.Key_F7: KeyCode.F7, QtC.Key.Key_F8: KeyCode.F8,
        QtC.Key.Key_F9: KeyCode.F9, QtC.Key.Key_F10: KeyCode.F10,
        QtC.Key.Key_F11: KeyCode.F11, QtC.Key.Key_F12: KeyCode.F12,
        QtC.Key.Key_Shift: KeyCode.LEFT_SHIFT,
        QtC.Key.Key_Control: KeyCode.LEFT_CONTROL,
        QtC.Key.Key_Alt: KeyCode.LEFT_ALT,
        QtC.Key.Key_Meta: KeyCode.LEFT_SUPER,
    })

if _pynput_available:
    _build_pynput_maps()

class InputManager:
    _instance: Optional[InputManager] = None

    @classmethod
    def instance(cls) -> InputManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: list[tuple[int, bool]] = []
        self._pending_mouse: list[tuple[float, float, bool]] = []
        self._pending_mouse_delta: list[tuple[float, float]] = []
        self._pending_scroll: list[tuple[float, float]] = []
        self._kb_listener = None
        self._mouse_listener = None
        self._running = False
        self._mouse_x: float = 0.0
        self._mouse_y: float = 0.0
        self._surface_w: int = 1
        self._surface_h: int = 1

    def start(self):
        if self._running:
            return
        self._running = True
        if not _pynput_available:
            return
        try:
            self._mouse_listener = mouse.Listener(
                on_move=self._on_mouse_move,
                on_click=self._on_mouse_click,
                on_scroll=self._on_mouse_scroll
            )
            self._mouse_listener.start()
        except Exception:
            self._mouse_listener = None
        try:
            self._kb_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self._kb_listener.start()
        except Exception:
            self._kb_listener = None

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._mouse_listener:
            try: self._mouse_listener.stop()
            except Exception: pass
        if self._kb_listener:
            try: self._kb_listener.stop()
            except Exception: pass

    def _on_mouse_move(self, x, y):
        with self._lock:
            self._pending_mouse.append((float(x), float(y), True))

    def _on_mouse_click(self, x, y, button, pressed):
        btn = self._pynput_mouse_to_key(button)
        if btn is None:
            return
        with self._lock:
            self._pending.append((btn, pressed))

    def _on_mouse_scroll(self, x, y, dx, dy):
        with self._lock:
            self._pending_scroll.append((float(dx), float(dy)))

    def _on_key_press(self, key):
        vk = self._pynput_to_vk(key)
        if vk is not None:
            self._pending.append((vk, True))

    def _on_key_release(self, key):
        vk = self._pynput_to_vk(key)
        if vk is not None:
            self._pending.append((vk, False))

    @staticmethod
    def _pynput_mouse_to_key(button) -> Optional[int]:
        if mouse is None:
            return None
        m = {
            mouse.Button.left: KeyCode.MOUSE_LEFT,
            mouse.Button.right: KeyCode.MOUSE_RIGHT,
            mouse.Button.middle: KeyCode.MOUSE_MIDDLE,
            getattr(mouse.Button, 'back', None): KeyCode.MOUSE_BACK,
            getattr(mouse.Button, 'forward', None): KeyCode.MOUSE_FORWARD,
        }
        return m.get(button)

    @staticmethod
    def _pynput_to_vk(key):
        if keyboard is None:
            return None
        if isinstance(key, keyboard.KeyCode):
            if key in _PYNPUT_TO_KEY:
                return _PYNPUT_TO_KEY[key]
            if key.char and key.char.isprintable():
                return ord(key.char)
            return None
        if isinstance(key, keyboard.Key):
            return _PYNPUT_TO_KEY.get(key)
        return None

    @staticmethod
    def qt_key_to_vk(qt_key: int) -> Optional[int]:
        _build_qt_map()
        return _QT_TO_KEY.get(qt_key)

    def feed_key(self, vk: int, pressed: bool):
        if pressed:
            Input._state_ref().press_key(vk)
        else:
            Input._state_ref().release_key(vk)

    def feed_mouse_button(self, btn: int, pressed: bool):
        mk = KeyCode.MOUSE_LEFT + btn
        self.feed_key(mk, pressed)

    def feed_scroll(self, dx: float, dy: float):
        Input._state_ref().set_scroll(dx, dy)

    def set_surface_size(self, w: int, h: int):
        self._surface_w = max(1, w)
        self._surface_h = max(1, h)

    def surface_size(self) -> tuple[int, int]:
        return (self._surface_w, self._surface_h)

    def new_frame(self):
        Input.end_frame()
        st = Input._state_ref()
        st._mouse_delta = (0.0, 0.0)
        st._scroll_delta = (0.0, 0.0)

        with self._lock:
            pending = self._pending
            self._pending = []
            mouse_updates = self._pending_mouse
            self._pending_mouse = []
            mouse_delta_updates = self._pending_mouse_delta
            self._pending_mouse_delta = []
            scroll_updates = self._pending_scroll
            self._pending_scroll = []

        for vk, pressed in pending:
            if pressed:
                Input._state_ref().press_key(vk)
            else:
                Input._state_ref().release_key(vk)

        if mouse_updates:
            new_mx, new_my, _ = mouse_updates[-1]
            dx = new_mx - self._mouse_x
            dy = new_my - self._mouse_y
            self._mouse_x, self._mouse_y = new_mx, new_my
            Input._state_ref().set_mouse_pos(self._mouse_x, self._mouse_y, dx, dy)
        else:
            Input._state_ref().set_mouse_pos(self._mouse_x, self._mouse_y)

        if mouse_delta_updates:
            total_dx = sum(d[0] for d in mouse_delta_updates)
            total_dy = sum(d[1] for d in mouse_delta_updates)
            st = Input._state_ref()
            st._mouse_delta = (st._mouse_delta[0] + total_dx, st._mouse_delta[1] + total_dy)

        for sx, sy in scroll_updates:
            Input._state_ref().set_scroll(sx, sy)

        Input.begin_frame()

    @property
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def is_key_pressed(vk: int) -> bool:
        return Input.GetKey(vk)

    @staticmethod
    def key_just_pressed(vk: int) -> bool:
        return Input.GetKeyDown(vk)

    @staticmethod
    def key_just_released(vk: int) -> bool:
        return Input.GetKeyUp(vk)

    @staticmethod
    def is_mouse_pressed(btn: int) -> bool:
        return Input.GetMouseButton(btn)

    @staticmethod
    def mouse_just_pressed(btn: int) -> bool:
        return Input.GetMouseButtonDown(btn)

    @staticmethod
    def mouse_just_released(btn: int) -> bool:
        return Input.GetMouseButtonUp(btn)

    @staticmethod
    def get_mouse_pos() -> tuple[int, int]:
        pos = Input.mousePosition
        return (int(pos[0]), int(pos[1]))

    def consume_scroll(self) -> float:
        return Input.mouseScrollDelta[1]

from core.input.constants import (KEY_W, KEY_A, KEY_S, KEY_D, KEY_Q, KEY_E, KEY_R, KEY_F,
                              KEY_SHIFT, KEY_DELETE, KEY_CTRL, KEY_ALT, KEY_SPACE,
                              MOUSE_LEFT, MOUSE_RIGHT, MOUSE_MIDDLE)
