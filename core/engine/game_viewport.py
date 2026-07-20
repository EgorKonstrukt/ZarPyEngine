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
from PyQt6.QtCore import Qt, QTimer, QPoint, QRect
from PyQt6.QtGui import QSurfaceFormat, QKeyEvent, QMouseEvent, QCursor, QGuiApplication, QPainter, QColor, QFont, QFontMetrics, QPen, QBrush
from core.input.input_system import Input
from core.input.input_manager import InputManager
from core.input.constants import KEY_W, KEY_A, KEY_S, KEY_D, KEY_Q, KEY_E
from core.math.math3d import Vec3, Quat
from core.foundation.logger import Logger

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
        self._fallback_yaw: float = 0.0
        self._fallback_pitch: float = 0.0
        self._fallback_vel_x: float = 0.0
        self._fallback_vel_y: float = 0.0
        self._fallback_vel_z: float = 0.0
        self._fallback_right_mouse: bool = False
        self._fallback_move_speed: float = 5.0
        self._fallback_rotate_speed: float = 0.3
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
            self._renderer.render_scene(scene, view, proj, tr.position, rw, rh, self._screen_fbo, display_w=pw, display_h=ph)
            if self._stats_enabled:
                self._draw_stats_overlay()
        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            Logger.error(f"GameViewport render error: {e}", e)

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
            else:
                self._tick_fallback_camera()
            self._sync_cursor()
            self.update()

    def _find_camera_entity(self):
        scene = self._engine.scene
        if not scene:
            return None
        try:
            from core.components import Camera
            for e in scene.get_entities_with_component(Camera):
                if e.active:
                    return e
        except Exception:
            pass
        return None

    def _has_editor_camera(self) -> bool:
        scene = self._engine.scene
        if not scene:
            return False
        try:
            from core.components.rendering.cameras.editor_camera import EditorCamera
            for e in scene.get_entities_with_component(EditorCamera):
                if e.active:
                    comp = e.get_component(EditorCamera)
                    if comp.enabled:
                        return True
        except Exception:
            pass
        return False

    def _tick_fallback_camera(self):
        if self._has_editor_camera():
            return
        dt = self._paint_dt if self._paint_dt > 0 else 0.016
        self._fallback_cam_update(dt)

    def _tick_editor_cameras(self):
        try:
            from core.components.rendering.cameras.editor_camera import EditorCamera
            scene = self._engine.scene
            if not scene:
                return
            dt = self._paint_dt if self._paint_dt > 0 else 0.016
            found = False
            for e in scene.get_entities_with_component(EditorCamera):
                if e.active:
                    comp = e.get_component(EditorCamera)
                    if comp.enabled:
                        comp.on_update(dt)
                        found = True
            if not found:
                self._fallback_cam_update(dt)
        except Exception as e:
            Logger.warning(f"_tick_editor_cameras error: {e}")

    def _fallback_cam_update(self, dt: float):
        cam_e = self._find_camera_entity()
        if not cam_e:
            return
        t = cam_e.transform
        if not t:
            return
        im = self._input_manager
        if dt > 0.05:
            dt = 0.05
        if self._fallback_right_mouse:
            import math
            fwd = Vec3(
                -math.cos(math.radians(self._fallback_pitch)) * math.sin(math.radians(self._fallback_yaw)),
                -math.sin(math.radians(self._fallback_pitch)),
                -math.cos(math.radians(self._fallback_pitch)) * math.cos(math.radians(self._fallback_yaw))
            ).normalized()
            right = fwd.cross(Vec3.up()).normalized()
            speed = self._fallback_move_speed
            accel = Vec3.zero()
            if im.is_key_pressed(KEY_W):
                accel = accel + fwd * speed
            if im.is_key_pressed(KEY_S):
                accel = accel - fwd * speed
            if im.is_key_pressed(KEY_A):
                accel = accel - right * speed
            if im.is_key_pressed(KEY_D):
                accel = accel + right * speed
            if im.is_key_pressed(KEY_E):
                accel = accel + Vec3.up() * speed
            if im.is_key_pressed(KEY_Q):
                accel = accel - Vec3.up() * speed
            facc = dt * 12.0
            self._fallback_vel_x += (accel.x - self._fallback_vel_x) * min(facc, 1.0)
            self._fallback_vel_y += (accel.y - self._fallback_vel_y) * min(facc, 1.0)
            self._fallback_vel_z += (accel.z - self._fallback_vel_z) * min(facc, 1.0)
            t.position = Vec3(
                t.position.x + self._fallback_vel_x * dt,
                t.position.y + self._fallback_vel_y * dt,
                t.position.z + self._fallback_vel_z * dt
            )
        else:
            self._fallback_vel_x *= 0.85
            self._fallback_vel_y *= 0.85
            self._fallback_vel_z *= 0.85
            if abs(self._fallback_vel_x) > 0.001 or abs(self._fallback_vel_y) > 0.001 or abs(self._fallback_vel_z) > 0.001:
                t.position = Vec3(
                    t.position.x + self._fallback_vel_x * dt,
                    t.position.y + self._fallback_vel_y * dt,
                    t.position.z + self._fallback_vel_z * dt
                )

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

            fps = self._fps if self._fps > 0 else 0.0
            paint_dt = self._paint_dt if self._paint_dt > 0 else 0.016

            if paint_dt > 0:
                self._frame_times_ms.append(paint_dt * 1000.0)
                if len(self._frame_times_ms) > 300:
                    self._frame_times_ms.pop(0)

            sorted_ft = sorted(self._frame_times_ms)
            n = len(sorted_ft)
            p1_count = max(1, int(n * 0.01))
            p01_count = max(1, int(n * 0.001))
            p1_low = sum(sorted_ft[-p1_count:]) / p1_count if sorted_ft else 0.0
            p01_low = sum(sorted_ft[-p01_count:]) / p01_count if sorted_ft else 0.0
            cpu_ms = paint_dt * 1000.0

            eng = self._engine
            tps = eng.tps if hasattr(eng, 'tps') else 0.0
            time_scale = eng.time_scale if hasattr(eng, 'time_scale') else 1.0

            r = self._renderer
            triangles = getattr(r, '_triangles_drawn', 0) or 0
            vertices = getattr(r, '_vertices_drawn', 0) or 0
            draw_calls = getattr(r, '_draw_calls', 0) or 0
            culled_visible = getattr(r, '_culled_visible', 0) or 0
            culled_total = getattr(r, '_culled_total', 0) or 0
            culled_str = f"{culled_visible}/{culled_total}"
            particles = getattr(r, '_particle_count', 0) or 0
            batches = 0
            instanced = 0
            batcher = getattr(r, '_batcher', None)
            if batcher is not None:
                batches = batcher.batches
                instanced = batcher.instanced
            gizmo_draws = 0
            gizmo_lines = 0
            giz = getattr(r, '_gizmo', None)
            if giz is not None:
                gizmo_draws = getattr(giz, '_stat_draws', 0) or 0
                gizmo_lines = getattr(giz, '_stat_lines', 0) or 0

            fw, fh = self._get_physical_dims()

            def _fmt_count(cnt):
                if cnt >= 1_000_000:
                    return f"{cnt/1_000_000:.1f}M"
                if cnt >= 1_000:
                    return f"{cnt/1_000:.1f}k"
                return str(cnt)

            stats_lines = [
                f"FPS: {fps:.1f}  |  1%: {1000.0/max(p1_low,0.1):.1f}  |  0.1%: {1000.0/max(p01_low,0.1):.1f}  |  CPU: {cpu_ms:.1f}ms  |  Res: {fw}x{fh}",
                f"TPS: {tps:.0f}  |  TS: {time_scale:.2f}",
                f"Culled: {culled_str}  |  Draw Calls: {draw_calls}  |  Tris: {_fmt_count(triangles)}  |  Verts: {_fmt_count(vertices)}",
                f"Particles: {_fmt_count(particles)}  |  Batches: {batches}  |  Instanced: {instanced}  |  Gizmo Draws: {gizmo_draws}  |  GLines: {_fmt_count(gizmo_lines)}",
            ]

            text_color = QColor(255, 255, 255)
            label_color = QColor(160, 160, 160)
            bg_color = QColor(0, 0, 0, 160)
            border_color = QColor(80, 80, 80, 200)
            padding = 6
            line_height = 15
            total_h = len(stats_lines) * line_height + padding * 2

            fm = QFontMetrics(font)
            max_w = max(fm.horizontalAdvance(line) for line in stats_lines) + padding * 2
            max_w = max(max_w, 460)

            x = 8
            y = 35
            rect = QRect(x, y, int(max_w), total_h)
            painter.fillRect(rect, bg_color)
            painter.setPen(QPen(border_color, 1))
            painter.drawRect(rect)

            for i, line in enumerate(stats_lines):
                cx = x + padding
                cy = y + padding + i * line_height
                for idx, seg in enumerate(line.split("  |  ")):
                    seg = seg.strip()
                    if not seg:
                        continue
                    if ":" in seg:
                        lab, val = seg.split(":", 1)
                        lab = lab.strip() + ": "
                        val = val.strip()
                        painter.setPen(label_color)
                        painter.drawText(QRect(cx, cy, fm.horizontalAdvance(lab), line_height),
                                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, lab)
                        cx += fm.horizontalAdvance(lab)
                        painter.setPen(text_color)
                        painter.drawText(QRect(cx, cy, fm.horizontalAdvance(val), line_height),
                                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, val)
                        cx += fm.horizontalAdvance(val)
                    else:
                        painter.setPen(text_color)
                        painter.drawText(QRect(cx, cy, fm.horizontalAdvance(seg), line_height),
                                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, seg)
                        cx += fm.horizontalAdvance(seg)
                    if idx < len(line.split("  |  ")) - 1:
                        painter.setPen(QColor(100, 100, 100))
                        sep = " | "
                        painter.drawText(QRect(cx, cy, fm.horizontalAdvance(sep), line_height),
                                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, sep)
                        cx += fm.horizontalAdvance(sep)

            chart_y = y + total_h + 6
            chart_h = 30
            chart_rect = QRect(x, chart_y, int(max_w), chart_h)
            painter.fillRect(chart_rect, bg_color)
            painter.setPen(QPen(border_color, 1))
            painter.drawRect(chart_rect)
            ft_list = self._frame_times_ms
            n_bars = min(len(ft_list), chart_rect.width() - 4)
            if n_bars > 1:
                bar_w = (chart_rect.width() - 4) / n_bars
                max_ft = max(max(ft_list[-n_bars:]) * 1.1, 16.0)
                for bi in range(n_bars):
                    ft_val = ft_list[-n_bars + bi]
                    bh = max(1, int((ft_val / max_ft) * (chart_h - 4)))
                    bar_x = chart_rect.x() + 2 + int(bar_w * bi)
                    bar_y = chart_rect.bottom() - 2 - bh
                    if ft_val > 33.0:
                        painter.setBrush(QBrush(QColor(255, 80, 80, 180)))
                    elif ft_val > 16.0:
                        painter.setBrush(QBrush(QColor(255, 200, 80, 160)))
                    else:
                        painter.setBrush(QBrush(QColor(80, 200, 80, 140)))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRect(QRect(int(bar_x), bar_y, max(1, int(bar_w)), bh))
                painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
                ref_y = chart_rect.bottom() - 2 - int((16.0 / max_ft) * (chart_h - 4))
                if ref_y > chart_rect.y() + 2:
                    painter.drawLine(chart_rect.x() + 2, ref_y, chart_rect.right() - 2, ref_y)
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
            found = False
            for e in scene.get_entities_with_component(EditorCamera):
                if e.active:
                    comp = e.get_component(EditorCamera)
                    if comp.enabled:
                        getattr(comp, method)(*args, **kwargs)
                        found = True
            if not found:
                self._fallback_cam_method(method, *args, **kwargs)
        except Exception as e:
            Logger.warning(f"_forward_to_editor_cam error: {e}")

    def _fallback_cam_method(self, method: str, *args, **kwargs):
        if method == "on_mouse_press":
            btn = args[0] if args else 0
            if btn == 1:
                self._fallback_right_mouse = True
        elif method == "on_mouse_release":
            btn = args[0] if args else 0
            if btn == 1:
                self._fallback_right_mouse = False
        elif method == "on_scroll":
            delta = args[0] if args else 0
            cam_e = self._find_camera_entity()
            if cam_e and cam_e.transform:
                import math
                fwd = Vec3(
                    -math.cos(math.radians(self._fallback_pitch)) * math.sin(math.radians(self._fallback_yaw)),
                    -math.sin(math.radians(self._fallback_pitch)),
                    -math.cos(math.radians(self._fallback_pitch)) * math.cos(math.radians(self._fallback_yaw))
                ).normalized()
                t = cam_e.transform
                t.position = t.position + fwd * delta * 0.5

    def _forward_to_editor_cam_delta(self, dx: float, dy: float):
        try:
            from core.components.rendering.cameras.editor_camera import EditorCamera
            scene = self._engine.scene
            if not scene:
                return
            found = False
            for e in scene.get_entities_with_component(EditorCamera):
                if e.active:
                    comp = e.get_component(EditorCamera)
                    if comp.enabled:
                        comp.on_mouse_delta(dx, dy)
                        found = True
            if not found:
                self._fallback_cam_delta(dx, dy)
        except Exception as e:
            Logger.warning(f"_forward_to_editor_cam_delta error: {e}")

    def _fallback_cam_delta(self, dx: float, dy: float):
        if self._fallback_right_mouse:
            cam_e = self._find_camera_entity()
            if not cam_e or not cam_e.transform:
                return
            t = cam_e.transform
            self._fallback_yaw -= dx * self._fallback_rotate_speed
            self._fallback_pitch = max(-89.0, min(89.0, self._fallback_pitch + dy * self._fallback_rotate_speed))
            t.local_rotation = Quat.from_euler(-self._fallback_pitch, self._fallback_yaw, 0.0)

    def leaveEvent(self, event):
        if self._mouse_captured:
            Input.set_cursor_visible(True)
            self._release_mouse()

    def focusOutEvent(self, event):
        if self._mouse_captured:
            Input.set_cursor_visible(True)
            self._release_mouse()
        super().focusOutEvent(event)
