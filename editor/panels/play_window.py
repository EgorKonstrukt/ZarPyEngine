# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
import moderngl
from typing import Optional, TYPE_CHECKING
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFrame)
from PyQt6.QtCore import Qt, QTimer, QEvent, QPoint
from PyQt6.QtGui import QSurfaceFormat, QKeyEvent, QMouseEvent, QWheelEvent, QCursor, QGuiApplication
from core.input.input_system import Input
from core.foundation.logger import Logger
if TYPE_CHECKING:
    from core.engine.engine import Engine
    from core.gui.canvas import GuiCanvas

class PlayViewport(QOpenGLWidget):
    def __init__(self, engine: Engine, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._ctx: Optional[moderngl.Context] = None
        self._renderer = None
        self._screen_fbo = None
        self._overlay_canvas: Optional[GuiCanvas] = None
        self._overlay_container: Optional[QWidget] = None
        self._mouse_captured: bool = False
        self._cursor_blank: bool = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
        engine.on("play_stop", self._on_play_stop)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        self.setFormat(fmt)

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

    def initializeGL(self):
        try:
            self._ctx = moderngl.create_context(standalone=False)
            self._bind_screen_fbo()
            from core.renderer.renderer import Renderer
            self._renderer = Renderer(self._ctx)
            self._renderer.initialize()
        except Exception as e:
            Logger.error(f"PlayViewport GL init error: {e}", e)

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
            from core.components import Camera, Transform
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
            if self._overlay_canvas and self._overlay_canvas.edit_mode:
                from PyQt6.QtGui import QPainter
                qp = QPainter(self)
                qp.setRenderHint(QPainter.RenderHint.Antialiasing)
                self._overlay_canvas._render_overlay(qp)
                qp.end()
        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            Logger.error(f"PlayViewport render error: {e}", e)

    def _tick(self):
        if self._engine.play_mode and self.isVisible():
            self._sync_cursor()
            self.update()
            canvas = self._overlay_canvas
            if canvas is not None:
                try:
                    from core.gui.system import GuiCanvasSystem
                    GuiCanvasSystem.instance().sync_all(self._engine.scene, canvas)
                except Exception:
                    pass

    def _sync_cursor(self):
        locked = Input.cursorLocked
        visible = Input.cursorVisible
        want_blank = locked or (not visible)
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

    def keyPressEvent(self, event: QKeyEvent):
        if self._engine.play_mode:
            from core.input.input_manager import InputManager
            im = InputManager.instance()
            with im._lock:
                im._pending.append((event.nativeVirtualKey(), True))
            if event.key() == Qt.Key.Key_Escape and self._mouse_captured:
                self._release_mouse()
            event.accept()
            return
        event.ignore()

    def keyReleaseEvent(self, event: QKeyEvent):
        if self._engine.play_mode:
            from core.input.input_manager import InputManager
            im = InputManager.instance()
            with im._lock:
                im._pending.append((event.nativeVirtualKey(), False))
            event.accept()
            return
        event.ignore()

    def mousePressEvent(self, event: QMouseEvent):
        if self._engine.play_mode:
            event.accept()
            return
        event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._mouse_captured:
            center = QPoint(self.width() // 2, self.height() // 2)
            local_pos = event.position()
            dx = local_pos.x() - center.x()
            dy = local_pos.y() - center.y()
            from core.input.input_manager import InputManager
            im = InputManager.instance()
            with im._lock:
                im._pending_mouse_delta.append((dx, dy))
            self._center_cursor()
            event.accept()
            return
        event.ignore()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._mouse_captured:
            event.accept()
            return
        event.ignore()

    def leaveEvent(self, event):
        if self._mouse_captured:
            self._release_mouse()

    def show_overlay(self, canvas: GuiCanvas):
        self.hide_overlay()
        self._overlay_canvas = canvas
        vw, vh = self.width(), self.height()
        container = QWidget(self)
        container.setObjectName("PlayOverlayContainer")
        container.setGeometry(0, 0, vw, vh)
        container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        container.setStyleSheet("QWidget#PlayOverlayContainer { background: transparent; }")
        container.show()
        root = canvas._root
        root.setParent(container)
        root.setGeometry(0, 0, vw, vh)
        root.setVisible(True)
        root.installEventFilter(self)
        self._overlay_root = root
        self._overlay_container = container

    def hide_overlay(self):
        canvas = self._overlay_canvas
        if canvas and self._overlay_container:
            root = canvas._root
            root.removeEventFilter(self)
            root.setParent(canvas)
            canvas._update_root_geometry()
            root.setVisible(True)
            self._overlay_container.deleteLater()
            self._overlay_container = None
        self._overlay_root = None
        self._overlay_canvas = None

    def eventFilter(self, watched, event):
        root = getattr(self, "_overlay_root", None)
        if watched is root and root is not None:
            et = event.type()
            if et in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
                      QEvent.Type.MouseMove, QEvent.Type.Wheel):
                pos = event.position()
                if root.childAt(int(pos.x()), int(pos.y())) is None:
                    QGuiApplication.sendEvent(self, event)
                    return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._overlay_container and self._overlay_canvas:
            vw, vh = self.width(), self.height()
            self._overlay_container.setGeometry(0, 0, vw, vh)
            canvas = self._overlay_canvas
            canvas._update_root_geometry()

    def focusOutEvent(self, event):
        if self._mouse_captured:
            self._release_mouse()
        super().focusOutEvent(event)


class PlayDockPanel(QDockWidget):
    def __init__(self, engine: Engine, parent=None):
        super().__init__("Game", parent)
        self._engine = engine
        self._paused = False
        self._setup_ui()
        self.setObjectName("PlayDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable)

    def _setup_ui(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)
        step_btn = QPushButton("Step")
        step_btn.clicked.connect(self._step)
        toolbar.addWidget(step_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)
        self._viewport = PlayViewport(self._engine, self)
        layout.addWidget(self._viewport, 1)
        self.setWidget(w)

    def _toggle_pause(self):
        self._paused = not self._paused
        mw = self.parent()
        ts = mw._ts_sb.value() if hasattr(mw, '_ts_sb') else 1.0
        self._engine.time_scale = 0.0 if self._paused else ts

    def _step(self):
        if self._paused:
            self._engine.time_scale = self._engine.fixed_dt
            self._engine.tick()
            self._engine.time_scale = 0.0
