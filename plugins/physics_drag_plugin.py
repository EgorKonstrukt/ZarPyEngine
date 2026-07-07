# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from PyQt6.QtCore import QObject, QEvent, Qt
from PyQt6.QtGui import QMouseEvent
from core.plugin_manager import PluginBase
from core.logger import Logger
from core.math3d import Vec3
if TYPE_CHECKING:
    from editor.scene_viewport import SceneViewport
    from core.physics.physics_scene import PhysicsScene


class _DragFilter(QObject):
    def __init__(self, viewport: SceneViewport, plugin: PhysicsDragPlugin):
        super().__init__(viewport)
        self._viewport = viewport
        self._plugin = plugin
        self._constraint_id: Optional[int] = None
        self._anchor: Optional[Vec3] = None
        self._body_pos: Optional[Vec3] = None

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.MouseButtonPress:
            return self._on_press(event)
        if event.type() == QEvent.Type.MouseMove:
            return self._on_move(event)
        if event.type() == QEvent.Type.MouseButtonRelease:
            return self._on_release(event)
        return super().eventFilter(obj, event)

    def _on_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        mods = event.modifiers()
        if not (mods & Qt.KeyboardModifier.ShiftModifier) or not (mods & Qt.KeyboardModifier.ControlModifier):
            return False
        ps = self._plugin._physics_scene
        if ps is None:
            return False
        engine = self._plugin.engine
        if engine is None or not engine.play_mode:
            return False
        lx, ly = int(event.position().x()), int(event.position().y())
        origin, direction = self._viewport.screen_to_ray(lx, ly)
        result = ps.ray_cast(origin, direction, 1000.0)
        if result is None:
            return False
        entity_id = result.get("entity_id")
        body_id = result.get("body_id")
        if entity_id is None or body_id is None:
            return False
        entity = engine.scene.get_entity(entity_id)
        if entity is None:
            return False
        from core.components import Rigidbody
        rb = entity.get_component(Rigidbody)
        if rb is None or rb.is_kinematic or rb.mass <= 0:
            return False
        hit_pos = Vec3(result["position"][0], result["position"][1], result["position"][2])
        cid = ps.create_drag_constraint(body_id, hit_pos)
        if cid is not None:
            self._constraint_id = cid
            self._anchor = hit_pos
            self._body_pos = self._get_body_world_pos(entity)
        return True

    def _on_move(self, event: QMouseEvent) -> bool:
        if self._constraint_id is None or self._anchor is None:
            return False
        ps = self._plugin._physics_scene
        if ps is None:
            return False
        lx, ly = int(event.position().x()), int(event.position().y())
        target = self._viewport.screen_to_plane(lx, ly, self._anchor)
        ps.update_drag_constraint(self._constraint_id, target)
        self._anchor = target
        self._viewport.add_debug_line(
            self._body_pos, self._anchor,
            [1.0, 0.8, 0.2, 1.0],
        )
        return True

    def _on_release(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton or self._constraint_id is None:
            return False
        ps = self._plugin._physics_scene
        if ps is not None:
            ps.remove_drag_constraint(self._constraint_id)
        self._constraint_id = None
        self._anchor = None
        self._body_pos = None
        return True

    def _get_body_world_pos(self, entity) -> Vec3:
        tr = entity.transform
        if tr is not None:
            return tr.position
        return Vec3.zero()


class PhysicsDragPlugin(PluginBase):
    NAME = "PhysicsDragPlugin"
    VERSION = "0.1.0"
    DESCRIPTION = "Editor tool: Ctrl+Shift+Click to drag physics objects."
    SYSTEM = False

    def __init__(self):
        super().__init__()
        self._physics_scene: Optional[PhysicsScene] = None
        self._filter: Optional[_DragFilter] = None

    def initialize(self, engine):
        super().initialize(engine)
        self._refresh_physics_scene()
        Logger.info("[PhysicsDragPlugin] initialized.")

    def on_viewport_ready(self, viewport):
        if self._filter is not None:
            return
        self._filter = _DragFilter(viewport, self)
        viewport.installEventFilter(self._filter)
        Logger.info("[PhysicsDragPlugin] viewport filter installed.")

    def on_scene_loaded(self, scene):
        self._refresh_physics_scene()

    def on_play_start(self):
        self._refresh_physics_scene()

    def _refresh_physics_scene(self):
        engine = self.engine
        if engine is None:
            return
        pp = engine.plugin_manager.get("PhysicsPlugin")
        if pp is not None:
            self._physics_scene = getattr(pp, 'physics_scene', None)

    def shutdown(self):
        if self._filter is not None:
            vp = self._filter._viewport
            vp.removeEventFilter(self._filter)
            self._filter = None
        self._physics_scene = None
