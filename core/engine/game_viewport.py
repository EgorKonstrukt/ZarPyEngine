# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import moderngl
import time
from typing import Optional, TYPE_CHECKING
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QSurfaceFormat, QKeyEvent, QMouseEvent, QCursor, QGuiApplication, QPainter, QFont
from core.input.input_system import Input
from core.input.input_manager import InputManager
from core.foundation.logger import Logger
from core.renderer.render_stats import (
    _SPIKE_LOG,
    build_stats_rows,
    collect_render_stats,
    compute_frame_metrics,
    draw_stats_panel,
    log_spike,
)

if TYPE_CHECKING:
    from core.engine.engine import Engine


class GameViewport(QOpenGLWidget):
    """Standalone game viewport for player builds. No editor dependencies."""

    def __init__(self, engine: "Engine", parent=None):
        super().__init__(parent)
        self._engine = engine
        self._ctx: Optional[moderngl.Context] = None
        self._renderer = None
        self._screen_fbo = None
        self._mouse_captured: bool = False
        self._cursor_blank: bool = False
        self._input_manager = InputManager.instance()
        self._last_mouse_x: int = 0
        self._last_mouse_y: int = 0
        from core.config.config import get_global_config
        cfg = get_global_config()
        self._vsync_enabled = cfg.get("rendering.vsync", True)
        self._target_fps = cfg.get("rendering.target_fps", 60)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        engine.on("play_stop", self._on_play_stop)
        self._stats_enabled: bool = False
        self._fps: float = 0.0
        self._fps_accum: float = 0.0
        self._fps_frames: int = 0
        self._last_paint_time: float = 0.0
        self._paint_dt: float = 0.016
        self._last_render_ms: float = 0.0
        self._last_paint_full_ms: float = 0.0
        self._frame_times_ms: list[float] = []
        self._physical_w: int = 0
        self._physical_h: int = 0
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
        if self._mouse_captured or self._cursor_blank:
            Input.set_cursor_visible(True)
            Input.set_cursor_locked(False)
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
            tgt = int(self._target_fps) if self._target_fps else 0
            if tgt <= 0 or tgt == 60:
                self._timer.setInterval(0)
            else:
                tgt = max(1, min(360, tgt))
                self._timer.setInterval(max(1, int(1000.0 / tgt)))
        else:
            self._timer.setTimerType(Qt.TimerType.CoarseTimer)
            tgt = int(self._target_fps) if self._target_fps else 60
            tgt = max(1, min(360, tgt))
            self._timer.setInterval(max(1, int(1000.0 / tgt)))
        if self.isVisible() or not self._vsync_enabled:
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
        self._physical_w, self._physical_h = pw, ph
        if self._ctx:
            self._ctx.viewport = (0, 0, pw, ph)
            self._bind_screen_fbo()

    def paintGL(self):
        if not self._ctx or not self._renderer:
            return
        now = time.perf_counter()
        if self._last_paint_time > 0:
            self._paint_dt = now - self._last_paint_time
            if self._paint_dt > 0.05:
                self._paint_dt = 0.05
            elif self._paint_dt < 0.0:
                self._paint_dt = 0.0
            self._fps_accum += self._paint_dt
            self._fps_frames += 1
            if self._fps_accum >= 0.5:
                self._fps = self._fps_frames / self._fps_accum
                self._fps_accum = 0.0
                self._fps_frames = 0
        self._last_paint_time = now
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
            rw, rh = cam.compute_render_size(pw, ph)
            aspect = rw / max(1, rh)
            view = cam.get_view_matrix()
            proj = cam.get_projection_matrix(aspect)
            self._renderer.show_grid = False
            _t0 = time.perf_counter()
            self._renderer.render_scene(scene, view, proj, tr.position, rw, rh, self._screen_fbo, display_w=pw, display_h=ph)
            self._last_render_ms = (time.perf_counter() - _t0) * 1000.0
            if self._stats_enabled:
                self._draw_stats_overlay()
        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            Logger.error(f"GameViewport render error: {e}", e)
        self._last_paint_full_ms = (time.perf_counter() - now) * 1000.0

    def _tick(self):
        if self._engine.play_mode and self.isVisible():
            prof = self._engine._profiler
            if prof:
                prof.capture_frame()
            self._input_manager.new_frame()
            if self._engine._game_worker is None:
                with self._engine._scene_lock:
                    self._engine.tick()
            self._tick_editor_cameras()
            self._sync_cursor()
            self.update()

    def _tick_editor_cameras(self):
        try:
            from core.components.rendering.cameras.editor_camera import EditorCamera
            scene = self._engine.scene
            if not scene:
                return
            dt = self._paint_dt if self._paint_dt > 0 else 0.016
            for e in scene.get_entities_with_component(EditorCamera):
                if e.active:
                    comp = e.get_component(EditorCamera)
                    if comp.enabled:
                        comp.on_update(dt)
        except Exception as e:
            Logger.warning(f"_tick_editor_cameras error: {e}")

    def _sync_cursor(self):
        locked = Input.cursorLocked
        visible = Input.cursorVisible
        if self._mouse_captured:
            if not self._cursor_blank:
                QGuiApplication.setOverrideCursor(Qt.CursorShape.BlankCursor)
                self._cursor_blank = True
            return
        want_blank = not visible
        want_grab = locked
        if want_grab and not self._mouse_captured:
            self._mouse_captured = True
            self.grabMouse()
            self._center_cursor()
        elif not want_grab and self._mouse_captured:
            self._mouse_captured = False
            self.releaseMouse()
        if want_blank and not self._cursor_blank:
            QGuiApplication.setOverrideCursor(Qt.CursorShape.BlankCursor)
            self._cursor_blank = True
        elif not want_blank and self._cursor_blank:
            QGuiApplication.restoreOverrideCursor()
            self._cursor_blank = False

    def _release_mouse(self):
        if self._mouse_captured:
            self._mouse_captured = False
            self.releaseMouse()
        if self._cursor_blank:
            QGuiApplication.restoreOverrideCursor()
            self._cursor_blank = False

    def _center_cursor(self):
        center = self.mapToGlobal(QPoint(self.width() // 2, self.height() // 2))
        QCursor.setPos(center)

    def _get_physical_dims(self):
        if self._physical_w > 0 and self._physical_h > 0:
            return self._physical_w, self._physical_h
        dpr = self.devicePixelRatio()
        return int(self.width() * dpr), int(self.height() * dpr)

    def _draw_stats_overlay(self):
        painter = QPainter(self)
        try:
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            font = QFont("Consolas", 9)
            font.setStyleStrategy(QFont.StyleStrategy.ForceOutline)
            painter.setFont(font)

            paint_dt = self._paint_dt if self._paint_dt > 0 else 0.016
            if paint_dt > 0:
                self._frame_times_ms.append(paint_dt * 1000.0)
                if len(self._frame_times_ms) > 300:
                    self._frame_times_ms.pop(0)

            if self._frame_times_ms and self._frame_times_ms[-1] > 33.0:
                prof = getattr(self._engine, '_profiler', None)
                if prof is not None and getattr(prof, 'enabled', False):
                    log_spike(self._frame_times_ms[-1], prof)

            m = compute_frame_metrics(self._frame_times_ms)
            live_fps = self._fps if self._fps > 0 else 0.0
            if live_fps > 0:
                m['fps'] = live_fps
                m['avg_fps'] = live_fps
            st = collect_render_stats(self._engine, self._renderer)
            fw, fh = self._get_physical_dims()
            timings = {
                'cpu_ms': paint_dt * 1000.0,
                'render_ms': self._last_render_ms or 0.0,
                'gizmo_ms': 0.0,
                'overlay_ms': 0.0,
                'paint_ms': self._last_paint_full_ms or 0.0,
                'res': f"{fw}x{fh}",
            }
            rows = build_stats_rows(m, st, timings)
            draw_stats_panel(painter, rows, self._frame_times_ms, _SPIKE_LOG)
            painter.restore()
        except Exception as e:
            Logger.error(f"GameViewport stats overlay error: {e}", e)
        finally:
            painter.end()

    def keyPressEvent(self, event: QKeyEvent):
        if self._engine.play_mode:
            kc = self._input_manager.qt_key_to_vk(event.key())
            if kc is not None:
                with self._input_manager._lock:
                    self._input_manager._pending.append((kc, True))
            if event.key() == Qt.Key.Key_Escape and self._mouse_captured:
                self._release_mouse()
            if event.key() == Qt.Key.Key_F3 and (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                self._stats_enabled = not self._stats_enabled
                event.accept()
                return
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
            if event.button() == Qt.MouseButton.RightButton and not self._mouse_captured:
                self._mouse_captured = True
                self._cursor_blank = True
                self.grabMouse()
                QGuiApplication.setOverrideCursor(Qt.CursorShape.BlankCursor)
                self._center_cursor()
            pos = event.position()
            self._last_mouse_x = int(pos.x())
            self._last_mouse_y = int(pos.y())
            self._forward_to_editor_cam("on_mouse_press", btn, 0, 0, bool(event.modifiers() & Qt.KeyboardModifier.AltModifier))
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
            self._forward_to_editor_cam_delta(dx, dy)
            self._center_cursor()
            event.accept()
            return
        event.ignore()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._engine.play_mode:
            btn = self._mouse_button_index(event.button())
            self._input_manager.feed_mouse_button(btn, False)
            if event.button() == Qt.MouseButton.RightButton and self._mouse_captured:
                Input.set_cursor_visible(True)
                self._mouse_captured = False
                self.releaseMouse()
                QGuiApplication.restoreOverrideCursor()
                self._cursor_blank = False
            self._forward_to_editor_cam("on_mouse_release", btn)
            event.accept()
            return
        event.ignore()

    def wheelEvent(self, event):
        if self._engine.play_mode:
            delta = event.angleDelta()
            self._input_manager.feed_scroll(delta.x(), delta.y())
            self._forward_to_editor_cam("on_scroll", delta.y())
            event.accept()
            return
        event.ignore()

    def _forward_to_editor_cam(self, method: str, *args, **kwargs):
        try:
            from core.components.rendering.cameras.editor_camera import EditorCamera
            scene = self._engine.scene
            if not scene:
                return
            for e in scene.get_entities_with_component(EditorCamera):
                if e.active:
                    comp = e.get_component(EditorCamera)
                    if comp.enabled:
                        getattr(comp, method)(*args, **kwargs)
        except Exception as e:
            Logger.warning(f"_forward_to_editor_cam error: {e}")

    def _forward_to_editor_cam_delta(self, dx: float, dy: float):
        try:
            from core.components.rendering.cameras.editor_camera import EditorCamera
            scene = self._engine.scene
            if not scene:
                return
            for e in scene.get_entities_with_component(EditorCamera):
                if e.active:
                    comp = e.get_component(EditorCamera)
                    if comp.enabled:
                        comp.on_mouse_delta(dx, dy)
        except Exception as e:
            Logger.warning(f"_forward_to_editor_cam_delta error: {e}")

    def leaveEvent(self, event):
        if self._mouse_captured:
            Input.set_cursor_visible(True)
            self._release_mouse()

    def focusOutEvent(self, event):
        if self._mouse_captured:
            Input.set_cursor_visible(True)
            self._release_mouse()
        super().focusOutEvent(event)
