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
                             QPushButton, QLabel, QFrame)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QSurfaceFormat, QKeyEvent, QMouseEvent, QWheelEvent
from core.math3d import Vec3
from core.logger import Logger
if TYPE_CHECKING:
    from core.engine import Engine
    from core.ecs import Scene
from core.editor_scale import scale, scale_xy

class PrefabViewport(QOpenGLWidget):
    def __init__(self, engine: Engine, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._ctx: Optional[moderngl.Context] = None
        self._renderer = None
        self._screen_fbo = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        self.setFormat(fmt)

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
            Logger.error(f"PrefabViewport GL init error: {e}", e)

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
            from core.components import Camera, Transform
            scene = None
            pnl = self.parent()
            while pnl and not hasattr(pnl, '_edit_scene'):
                pnl = pnl.parent()
            if pnl and pnl._edit_scene:
                scene = pnl._edit_scene
            if not scene:
                self._ctx.clear(0.1, 0.1, 0.1, 1.0)
                return
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
            self._renderer.show_grid = True
            self._renderer.render_scene(scene, view, proj, tr.position, pw, ph, self._screen_fbo)
        except Exception as e:
            Logger.error(f"PrefabViewport render error: {e}", e)

    def _tick(self):
        if self.isVisible():
            self.update()

    def keyPressEvent(self, event: QKeyEvent):
        event.ignore()

    def keyReleaseEvent(self, event: QKeyEvent):
        event.ignore()

    def mousePressEvent(self, event: QMouseEvent):
        pass

    def mouseMoveEvent(self, event: QMouseEvent):
        pass

    def mouseReleaseEvent(self, event: QMouseEvent):
        pass

    def wheelEvent(self, event: QWheelEvent):
        pass


class PrefabEditorPanel(QDockWidget):
    def __init__(self, engine: Engine, parent=None):
        super().__init__("Prefab Editor", parent)
        self._engine = engine
        self._prefab_path: Optional[str] = None
        self._edit_scene: Optional[Scene] = None
        self._saved_scene: Optional[Scene] = None
        self._saved_hierarchy = None
        self._setup_ui()
        self.setObjectName("PrefabEditorDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.visibilityChanged.connect(self._on_visibility_changed)

    def _setup_ui(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 4, 4, 4)
        self._prefab_name_label = QLabel("No prefab open")
        self._prefab_name_label.setStyleSheet("color: #88ccff; font-weight: bold; font-size: 11px;")
        toolbar.addWidget(self._prefab_name_label)
        toolbar.addStretch()
        self._save_btn = QPushButton("Save")
        self._save_btn.setFixedHeight(scale(24))
        self._save_btn.clicked.connect(self._on_save)
        self._save_btn.setEnabled(False)
        toolbar.addWidget(self._save_btn)
        self._return_btn = QPushButton("Return to Scene")
        self._return_btn.setFixedHeight(scale(24))
        self._return_btn.clicked.connect(self._on_return)
        self._return_btn.setEnabled(False)
        toolbar.addWidget(self._return_btn)
        layout.addLayout(toolbar)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)
        self._viewport = PrefabViewport(self._engine, self)
        layout.addWidget(self._viewport, 1)
        self.setWidget(w)

    def load_config(self, config) -> None:
        pass

    def open_prefab(self, path: str):
        from core.prefab import Prefab
        from core.ecs import Scene, ComponentRegistry
        pref = Prefab.load(path)
        if not pref:
            Logger.warning(f"Cannot load prefab: {path}")
            return
        self._saved_scene = self._engine.scene
        self._prefab_path = path
        self._edit_scene = Scene(f"Prefab: {pref.name}")
        from core.components import Transform, Camera
        cam_entity = self._edit_scene.create_entity("Prefab Camera")
        cam_entity.add_component(Transform())
        from core.math3d import Vec3
        cam_entity.transform.local_position = Vec3(0, 2, 5)
        cam_entity.add_component(Camera())
        self._engine._plugin_manager.notify_scene_loaded(self._edit_scene)
        self._engine._emit_event("scene_loaded", self._edit_scene)
        roots = pref.instantiate(self._edit_scene, ComponentRegistry)
        prefab_file = path.replace("\\", "/").split("/")[-1]
        self._prefab_name_label.setText(f"Prefab: {pref.name} ({prefab_file})")
        self._save_btn.setEnabled(True)
        self._return_btn.setEnabled(True)

    def _on_save(self):
        if not self._prefab_path or not self._edit_scene:
            return
        from core.prefab import Prefab, PrefabLibrary
        pref = Prefab.load(self._prefab_path)
        if not pref:
            return
        candidate_roots = self._edit_scene.get_root_entities()
        prefab_roots = [e for e in candidate_roots
                        if e.is_prefab_instance and e._prefab_guid == pref.guid]
        if prefab_roots:
            pref.capture(prefab_roots)
        else:
            pref.roots_data = []
        pref.save(self._prefab_path)
        PrefabLibrary.invalidate(self._prefab_path)

    def _on_return(self):
        if self._saved_scene:
            self._engine._plugin_manager.notify_scene_loaded(self._saved_scene)
            self._engine._emit_event("scene_loaded", self._saved_scene)
        self._edit_scene = None
        self._saved_scene = None
        self._prefab_path = None
        self._prefab_name_label.setText("No prefab open")
        self._save_btn.setEnabled(False)
        self._return_btn.setEnabled(False)
        self.hide()
        from PyQt6.QtWidgets import QApplication
        for w in QApplication.topLevelWidgets():
            if hasattr(w, '_hierarchy'):
                w._hierarchy.refresh()

    def _on_visibility_changed(self, visible: bool):
        if not visible and self._edit_scene and self._saved_scene:
            self._on_return()
