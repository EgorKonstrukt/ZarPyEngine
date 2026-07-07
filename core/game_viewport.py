# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import moderngl
from typing import Optional, TYPE_CHECKING
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QSurfaceFormat, QKeyEvent, QMouseEvent, QCursor, QGuiApplication
from core.input_system import Input
from core.input.input_manager import InputManager
from core.logger import Logger

if TYPE_CHECKING:
    from core.engine import Engine


class GameViewport(QOpenGLWidget):
    """Standalone game viewport for player builds. No editor dependencies."""

    def __init__(self, engine: "Engine", parent=None):
        super().__init__(parent)
        self._engine = engine
        self._ctx: Optional[moderngl.Context] = None
        self._renderer = None
        self._screen_fbo = None
        self._mouse_captured: bool = False
        self._input_manager = InputManager.instance()
        from core.config import get_global_config
        cfg = get_global_config()
        self._vsync_enabled = cfg.get("rendering.vsync", True)
        self._target_fps = cfg.get("rendering.target_fps", 60)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        engine.on("play_stop", self._on_play_stop)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setSwapInterval(1 if self._vsync_enabled else 0)
        self.setFormat(fmt)
        self._apply_config()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, False)

    def _on_play_stop(self, _=None):
        if self._mouse_captured:
            self._release_mouse()

    def _bind_screen_fbo(self):
        fbo_id = self.defaultFramebufferObject()
        self._screen_fbo = self._ctx.detect_framebuffer(fbo_id)
        self._screen_fbo.use()

    def showEvent(self, event):
        super().showEvent(event)
        self.update()

    def _apply_config(self):
        if not self._vsync_enabled:
            self._timer.setTimerType(Qt.TimerType.PreciseTimer)
            self._timer.setInterval(1)
        else:
            fps = self._target_fps
            if fps <= 0 or fps > 240:
                fps = 240
            self._timer.setInterval(max(1, int(1000.0 / fps)))
        self._timer.start()

    def initializeGL(self):
        try:
            self._ctx = moderngl.create_context(standalone=False)
            self._bind_screen_fbo()
            if not self._vsync_enabled:
                try:
                    import ctypes
                    opengl32 = ctypes.windll.opengl32
                    addr = opengl32.wglGetProcAddress(b"wglSwapIntervalEXT")
                    if addr:
                        func = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(addr)
                        func(0)
                except Exception:
                    pass
            from core.renderer import Renderer
            self._renderer = Renderer(self._ctx)
            self._renderer.initialize()
        except Exception as e:
            Logger.error(f"GameViewport GL init error: {e}", e)

    def resizeGL(self, w: int, h: int):
        dpr = self.devicePixelRatio()
        pw, ph = int(w * dpr), int(h * dpr)
        if self._ctx:
            self._ctx.viewport = (0, 0, pw, ph)
            self._bind_screen_fbo()

    def paintGL(self):
        if not self._ctx or not self._renderer:
            return
        try:
            self._bind_screen_fbo()
            scene = self._engine.scene
            if not scene:
                self._ctx.clear(0.1, 0.1, 0.1, 1.0)
                return
            from core.components import Camera
            cam_entity = None
            for e in scene.get_entities_with_component(Camera):
                if e.active:
                    cam_entity = e
                    break
            if not cam_entity:
                self._ctx.clear(0.1, 0.1, 0.1, 1.0)
                return
            cam = cam_entity.get_component(Camera)
            tr = cam_entity.transform
            if not cam or not tr:
                self._ctx.clear(0.1, 0.1, 0.1, 1.0)
                return
            cc = cam.clear_color
            self._ctx.clear(*cc[:3], 1.0)
            dpr = self.devicePixelRatio()
            pw, ph = int(self.width() * dpr), int(self.height() * dpr)
            aspect = pw / max(1, ph)
            view = cam.get_view_matrix()
            proj = cam.get_projection_matrix(aspect)
            self._renderer.show_grid = False
            self._renderer.render_scene(scene, view, proj, tr.position, pw, ph, self._screen_fbo)
        except Exception as e:
            Logger.error(f"GameViewport render error: {e}", e)

    def _tick(self):
        if self._engine.play_mode and self.isVisible():
            prof = self._engine._profiler
            if prof:
                prof.capture_frame()
            self._input_manager.new_frame()
            with self._engine._scene_lock:
                self._engine.tick()
            self.update()

    def _capture_mouse(self):
        if self._mouse_captured:
            return
        self._mouse_captured = True
        self.grabMouse()
        Input.set_cursor_locked(True)
        QGuiApplication.setOverrideCursor(Qt.CursorShape.BlankCursor)
        self._center_cursor()
        from core.input.input_manager import InputManager
        im = InputManager.instance()
        center = self.mapToGlobal(QPoint(self.width() // 2, self.height() // 2))
        im._mouse_x = center.x()
        im._mouse_y = center.y()

    def _release_mouse(self):
        if not self._mouse_captured:
            return
        self._mouse_captured = False
        Input.set_cursor_locked(False)
        QGuiApplication.restoreOverrideCursor()
        self.releaseMouse()

    def _center_cursor(self):
        center = self.mapToGlobal(QPoint(self.width() // 2, self.height() // 2))
        QCursor.setPos(center)

    def keyPressEvent(self, event: QKeyEvent):
        if self._engine.play_mode:
            kc = self._input_manager.qt_key_to_vk(event.key())
            if kc is not None:
                with self._input_manager._lock:
                    self._input_manager._pending.append((kc, True))
            if event.key() == Qt.Key.Key_Escape and self._mouse_captured:
                self._release_mouse()
            event.accept()
            return
        event.ignore()

    def keyReleaseEvent(self, event: QKeyEvent):
        if self._engine.play_mode:
            kc = self._input_manager.qt_key_to_vk(event.key())
            if kc is not None:
                with self._input_manager._lock:
                    self._input_manager._pending.append((kc, False))
            event.accept()
            return
        event.ignore()

    def _mouse_button_index(self, qt_btn) -> int:
        from PyQt6.QtCore import Qt
        if qt_btn == Qt.MouseButton.LeftButton:
            return 0
        if qt_btn == Qt.MouseButton.RightButton:
            return 1
        if qt_btn == Qt.MouseButton.MiddleButton:
            return 2
        return 0

    def mousePressEvent(self, event: QMouseEvent):
        if self._engine.play_mode:
            btn = self._mouse_button_index(event.button())
            self._input_manager.feed_mouse_button(btn, True)
            if not self._mouse_captured:
                self._capture_mouse()
            event.accept()
            return
        event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._mouse_captured:
            center = QPoint(self.width() // 2, self.height() // 2)
            local_pos = event.position()
            dx = local_pos.x() - center.x()
            dy = local_pos.y() - center.y()
            with self._input_manager._lock:
                self._input_manager._pending_mouse_delta.append((dx, dy))
            self._center_cursor()
            event.accept()
            return
        event.ignore()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._mouse_captured:
            btn = self._mouse_button_index(event.button())
            self._input_manager.feed_mouse_button(btn, False)
            self.grabMouse()
            event.accept()
            return
        event.ignore()

    def wheelEvent(self, event):
        if self._engine.play_mode:
            delta = event.angleDelta()
            self._input_manager.feed_scroll(delta.x(), delta.y())
            event.accept()
            return
        event.ignore()

    def leaveEvent(self, event):
        if self._mouse_captured:
            self._release_mouse()

    def focusOutEvent(self, event):
        if self._mouse_captured:
            self._release_mouse()
        super().focusOutEvent(event)
