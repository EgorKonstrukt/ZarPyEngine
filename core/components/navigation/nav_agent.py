# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry
from core.maths.math3d import Vec3
from core.components.inspector_meta import FieldType, InspectorField
from core.shaders.nav_pathfinding import NavWorld


@ComponentRegistry.register
class NavAgent(Component):
    _icon = "NavAgent.png"
    _gizmo_icon_color = (60, 180, 255)
    _gizmo_icon_label = "N"
    _gizmo_pass = "nav"
    _inspector_buttons = [
        ("_btn_set_target", "Set Target"),
        ("_btn_stop", "Stop"),
    ]

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("agent_radius", "Agent Radius", FieldType.FLOAT, min_val=0.01, max_val=100.0, step=0.1),
            InspectorField("agent_height", "Agent Height", FieldType.FLOAT, min_val=0.01, max_val=100.0, step=0.1),
            InspectorField("agent_padding", "Wall Padding", FieldType.FLOAT, min_val=0.0, max_val=100.0, step=0.1),
            InspectorField("max_climb", "Max Climb", FieldType.FLOAT, min_val=0.0, max_val=100.0, step=0.1),
            InspectorField("max_slope", "Max Slope", FieldType.FLOAT, min_val=0.0, max_val=89.0, step=1.0),
            InspectorField("max_speed", "Max Speed", FieldType.FLOAT, min_val=0.0, max_val=1000.0, step=0.1),
            InspectorField("max_angular_speed", "Angular Speed", FieldType.FLOAT, min_val=0.0, max_val=1000.0, step=1.0),
            InspectorField("flying", "Flying Mode", FieldType.BOOL),
            InspectorField("auto_move", "Auto Move", FieldType.BOOL),
            InspectorField("show_path_cells", "Show Path Cells", FieldType.BOOL),
            InspectorField("target_x", "Target X", FieldType.FLOAT, min_val=-10000.0, max_val=10000.0, step=0.1),
            InspectorField("target_y", "Target Y", FieldType.FLOAT, min_val=-10000.0, max_val=10000.0, step=0.1),
            InspectorField("target_z", "Target Z", FieldType.FLOAT, min_val=-10000.0, max_val=10000.0, step=0.1),
            InspectorField("target_entity_id", "Target Entity", FieldType.GAMEOBJECT),
            InspectorField("repath_interval", "Repath Interval", FieldType.FLOAT, min_val=0.0, max_val=60.0, step=0.1),
            InspectorField("arrival_distance", "Arrival Distance", FieldType.FLOAT, min_val=0.01, max_val=100.0, step=0.1),
            InspectorField("nav_resolution", "Nav Resolution", FieldType.INT, min_val=8, max_val=1024, step=1),
            InspectorField("nav_world_size", "Nav World Size", FieldType.FLOAT, min_val=10.0, max_val=10000.0, step=10.0),
        ]

    def __init__(self):
        super().__init__()
        self.agent_radius: float = 0.5
        self.agent_height: float = 2.0
        self.agent_padding: float = 0.0
        self.max_climb: float = 0.5
        self.max_slope: float = 45.0
        self.max_speed: float = 5.0
        self.max_angular_speed: float = 360.0
        self.flying: bool = False
        self.auto_move: bool = False
        self.show_path_cells: bool = False
        self.target_x: float = 0.0
        self.target_y: float = 0.0
        self.target_z: float = 0.0
        self.target_entity_id: str = ""
        self.repath_interval: float = 1.0
        self.arrival_distance: float = 0.5
        self.nav_resolution: int = 512
        self.nav_world_size: float = 500.0

        self._path: list[Vec3] = []
        self._path_index: int = 0
        self._repath_timer: float = 0.0
        self._nav_world: NavWorld = None
        self._last_nav_grid_version: int = -1
        self._pending_req_id: Optional[str] = None
        self._blocked_frames: int = 0
        self._fail_streak: int = 0
        self._last_path_goal = None
        self._path_grid_gid = None

    def on_start(self):
        self._repath_timer = 0.0
        if self.auto_move:
            self._request_path()

    def on_update(self, dt: float):
        if not self.auto_move:
            return
        if not self._ensure_nav_world():
            return

        self._poll_path_result()

        self._repath_timer += dt
        eff_interval = self.repath_interval * min(4, 2 ** min(self._fail_streak, 2))
        if self._repath_timer >= eff_interval:
            self._repath_timer = 0.0
            self._request_path()

        if not self._path or self._path_index >= len(self._path):
            return

        tr = self.transform
        if not tr:
            return

        target = self._path[self._path_index]
        current_pos = tr.local_position
        try:
            seg_free = self._nav_world.has_los(current_pos, target, self.flying)
        except Exception:
            seg_free = True
        if not seg_free:
            self._blocked_frames += 1
            adv = self._path_index
            try:
                for k in range(self._path_index + 1, len(self._path)):
                    if self._nav_world.has_los(current_pos, self._path[k], self.flying):
                        adv = k
            except Exception:
                pass
            if adv > self._path_index:
                self._path_index = adv
                self._blocked_frames = 0
                target = self._path[self._path_index]
            else:
                if self._blocked_frames >= 3:
                    self._blocked_frames = 0
                    self._pending_req_id = None
                    self._repath_timer = self.repath_interval
                return
        else:
            self._blocked_frames = 0
        to_target = target - current_pos
        dist = to_target.length()

        if dist <= self.arrival_distance:
            self._path_index += 1
            if self._path_index >= len(self._path):
                self._path = []
                self._path_index = 0
            return

        direction = to_target * (1.0 / dist) if dist > 0.001 else Vec3(0, 0, 1)
        move = direction * min(self.max_speed * dt, dist)
        tr.local_position = current_pos + move

        if self.max_angular_speed > 0:
            current_angles = tr.local_euler_angles
            target_yaw = math.degrees(math.atan2(direction.x, direction.z))
            diff = target_yaw - current_angles.y
            while diff > 180.0: diff -= 360.0
            while diff < -180.0: diff += 360.0
            max_step = self.max_angular_speed * dt
            if abs(diff) <= max_step:
                new_yaw = target_yaw
            else:
                new_yaw = current_angles.y + max_step * (1.0 if diff > 0 else -1.0)
            tr.local_euler_angles = Vec3(current_angles.x, new_yaw, current_angles.z)

    def _poll_path_result(self):
        if self._pending_req_id is None:
            return
        result = self._nav_world.poll_result(self._pending_req_id)
        if result is None:
            return
        self._pending_req_id = None
        self._blocked_frames = 0
        if len(result) == 0:
            self._fail_streak = min(self._fail_streak + 1, 6)
            self._path = result
            self._path_index = 0
            return
        self._fail_streak = 0
        try:
            keep_old = False
            if len(self._path) >= 2 and self._path_index < len(self._path):
                goal_now = self._get_target_position()
                if goal_now is not None and self._last_path_goal is not None:
                    dx = goal_now.x - self._last_path_goal[0]
                    dy = goal_now.y - self._last_path_goal[1]
                    dz = goal_now.z - self._last_path_goal[2]
                    same_goal = dx * dx + dy * dy + dz * dz < 0.04
                else:
                    same_goal = goal_now is None and self._last_path_goal is None
                gid_now = id(self._nav_world._grid._grid)
                if same_goal and self._path_grid_gid == gid_now:
                    tr0 = self.transform
                    if tr0 is not None:
                        pos0 = tr0.local_position
                        keep_old = True
                        prev = pos0
                        for wp in self._path[self._path_index:]:
                            if not self._nav_world.has_los(prev, wp, self.flying):
                                keep_old = False
                                break
                            prev = wp
            if keep_old:
                return
        except Exception:
            pass
        self._path = result
        try:
            goal_now = self._get_target_position()
            if goal_now is not None:
                self._last_path_goal = (goal_now.x, goal_now.y, goal_now.z)
            else:
                self._last_path_goal = None
            self._path_grid_gid = id(self._nav_world._grid._grid)
        except Exception:
            pass
        if len(result) >= 2:
            tr = self.transform
            if tr:
                try:
                    pos = tr.local_position
                    far = 1
                    for k in range(1, len(result)):
                        if self._nav_world.has_los(pos, result[k], self.flying):
                            far = k
                        else:
                            break
                    self._path_index = far
                except Exception:
                    self._path_index = 1
            else:
                self._path_index = 1
        else:
            self._path_index = 0

    def _get_target_position(self) -> Optional[Vec3]:
        if self.target_entity_id:
            scene = self.entity._scene if self.entity else None
            if scene:
                target_ent = scene.get_entity(self.target_entity_id)
                if target_ent:
                    target_tr = target_ent.transform
                    if target_tr:
                        return target_tr.position
        return Vec3(self.target_x, self.target_y, self.target_z)

    def _request_path(self):
        if self._pending_req_id is not None:
            return
        if not self._ensure_nav_world():
            return
        tr = self.transform
        if not tr:
            return
        start = tr.local_position
        end = self._get_target_position()
        if end is None:
            return
        self._nav_world.rebuild_grid()
        padding = self.agent_padding if self.agent_padding > 0 else None
        if self.flying:
            self._nav_world.dilate_for_agent(padding if padding is not None else self.agent_radius)
        req_id = self._nav_world.find_path_gpu_deferred(
            start, end, self.agent_radius, self.agent_height, self.flying,
            self.max_climb, self.max_slope, padding
        )
        self._pending_req_id = req_id

    def _ensure_nav_world(self):
        if self._nav_world:
            return True
        scene = self.entity._scene if self.entity else None
        if not scene:
            return False
        nw = NavWorld(self.nav_resolution, self.nav_world_size)
        nw.set_scene(scene)
        padding = self.agent_padding if self.agent_padding > 0 else self.agent_radius
        if self.flying:
            nw.dilate_for_agent(padding)
        self._nav_world = nw
        return True

    def set_target(self, pos: Vec3):
        self.target_entity_id = ""
        self.target_x = pos.x
        self.target_y = pos.y
        self.target_z = pos.z
        self.auto_move = True
        self._pending_req_id = None
        self._blocked_frames = 0
        self._fail_streak = 0
        self._repath_timer = self.repath_interval
        if self._ensure_nav_world():
            start = self.transform.local_position if self.transform else pos
            self._nav_world.rebuild_grid()
            padding = self.agent_padding if self.agent_padding > 0 else None
            if self.flying:
                self._nav_world.dilate_for_agent(padding if padding is not None else self.agent_radius)
            req_id = self._nav_world.find_path_gpu_deferred(
                start, pos, self.agent_radius, self.agent_height, self.flying,
                self.max_climb, self.max_slope, padding
            )
            self._pending_req_id = req_id

    def set_target_entity(self, entity) -> bool:
        if entity is None:
            return False
        tr = entity.transform
        if tr is None:
            return False
        self.target_entity_id = entity.id
        self.target_x = tr.position.x
        self.target_y = tr.position.y
        self.target_z = tr.position.z
        self.auto_move = True
        self._repath_timer = self.repath_interval
        return True

    def stop(self):
        self.auto_move = False
        self._path = []
        self._path_index = 0
        self._pending_req_id = None
        self._blocked_frames = 0
        self._fail_streak = 0

    def _btn_set_target(self):
        tr = self.transform
        if tr:
            pos = tr.local_position + Vec3(5, 0, 0)
            self.set_target(pos)

    def _btn_stop(self):
        self.stop()

    def has_path(self) -> bool:
        return len(self._path) > 0 and self._path_index < len(self._path)

    @property
    def current_path(self) -> list[Vec3]:
        return list(self._path)

    def gizmo_lines(self) -> list[tuple[Vec3, Vec3, list[float]]]:
        lines = []
        path_color = [0.2, 0.8, 1.0, 1.0]
        target_color = [1.0, 0.8, 0.2, 1.0]
        agent_color = [0.2, 1.0, 0.6, 0.5]

        if self.show_path_cells and self._nav_world:
            rects = self._nav_world.get_path_rects()
            if rects:
                for rct in rects:
                    aabb = rct.world_aabb(self._nav_world._grid)
                    depth_color = [0.1, 0.8, 0.3, 0.3]
                    corners = [
                        Vec3(aabb.min.x, aabb.min.y, aabb.min.z),
                        Vec3(aabb.max.x, aabb.min.y, aabb.min.z),
                        Vec3(aabb.max.x, aabb.max.y, aabb.min.z),
                        Vec3(aabb.min.x, aabb.max.y, aabb.min.z),
                        Vec3(aabb.min.x, aabb.min.y, aabb.max.z),
                        Vec3(aabb.max.x, aabb.min.y, aabb.max.z),
                        Vec3(aabb.max.x, aabb.max.y, aabb.max.z),
                        Vec3(aabb.min.x, aabb.max.y, aabb.max.z),
                    ]
                    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                    for a, b in edges:
                        lines.append((corners[a], corners[b], depth_color))
            else:
                aabbs = self._nav_world.get_path_aabbs()
                for aabb, depth in aabbs:
                    alpha = max(0.15, 1.0 - depth * 0.12)
                    depth_color = [0.2 + depth * 0.1, 0.9 - depth * 0.08, 0.4 + depth * 0.05, alpha]
                    corners = [
                        Vec3(aabb.min.x, aabb.min.y, aabb.min.z),
                        Vec3(aabb.max.x, aabb.min.y, aabb.min.z),
                        Vec3(aabb.max.x, aabb.max.y, aabb.min.z),
                        Vec3(aabb.min.x, aabb.max.y, aabb.min.z),
                        Vec3(aabb.min.x, aabb.min.y, aabb.max.z),
                        Vec3(aabb.max.x, aabb.min.y, aabb.max.z),
                        Vec3(aabb.max.x, aabb.max.y, aabb.max.z),
                        Vec3(aabb.min.x, aabb.max.y, aabb.max.z),
                    ]
                    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                    for a, b in edges:
                        lines.append((corners[a], corners[b], depth_color))

        tr = self.transform
        if tr:
            pos = tr.local_position
            r = self.agent_radius
            h = self.agent_height
            segments = 16
            for i in range(segments):
                theta1 = 2.0 * math.pi * i / segments
                theta2 = 2.0 * math.pi * (i + 1) / segments
                x1 = pos.x + r * math.cos(theta1)
                z1 = pos.z + r * math.sin(theta1)
                x2 = pos.x + r * math.cos(theta2)
                z2 = pos.z + r * math.sin(theta2)
                y_top = pos.y + h
                y_bot = pos.y
                lines.append((Vec3(x1, y_bot, z1), Vec3(x2, y_bot, z2), agent_color))
                lines.append((Vec3(x1, y_top, z1), Vec3(x2, y_top, z2), agent_color))
                lines.append((Vec3(x1, y_bot, z1), Vec3(x1, y_top, z1), agent_color))

        target_pos = Vec3(self.target_x, self.target_y, self.target_z)
        s = 0.5
        lines.append((target_pos + Vec3(-s, 0, -s), target_pos + Vec3(s, 0, s), target_color))
        lines.append((target_pos + Vec3(s, 0, -s), target_pos + Vec3(-s, 0, s), target_color))
        lines.append((target_pos + Vec3(0, -s, 0), target_pos + Vec3(0, s, 0), target_color))

        for i in range(len(self._path) - 1):
            a = self._path[i]
            b = self._path[i + 1]
            alpha = 1.0 - i / max(len(self._path), 1)
            lines.append((a, b, [path_color[0], path_color[1], path_color[2], alpha]))

        if self._path and self._path_index < len(self._path):
            wp = self._path[self._path_index]
            wp_color = [0.0, 1.0, 0.0, 1.0]
            ws = 0.3
            lines.append((wp + Vec3(-ws, 0, -ws), wp + Vec3(ws, 0, ws), wp_color))
            lines.append((wp + Vec3(ws, 0, -ws), wp + Vec3(-ws, 0, ws), wp_color))

        return lines

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "agent_radius": self.agent_radius,
            "agent_height": self.agent_height,
            "agent_padding": self.agent_padding,
            "max_climb": self.max_climb,
            "max_slope": self.max_slope,
            "max_speed": self.max_speed,
            "max_angular_speed": self.max_angular_speed,
            "flying": self.flying,
            "auto_move": self.auto_move,
            "show_path_cells": self.show_path_cells,
            "target_entity_id": self.target_entity_id,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "target_z": self.target_z,
            "repath_interval": self.repath_interval,
            "arrival_distance": self.arrival_distance,
            "nav_resolution": self.nav_resolution,
            "nav_world_size": self.nav_world_size,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> NavAgent:
        na = cls()
        na.enabled = data.get("enabled", True)
        na.agent_radius = data.get("agent_radius", 0.5)
        na.agent_height = data.get("agent_height", 2.0)
        na.agent_padding = data.get("agent_padding", 0.0)
        na.max_climb = data.get("max_climb", 0.5)
        na.max_slope = data.get("max_slope", 45.0)
        na.max_speed = data.get("max_speed", 5.0)
        na.max_angular_speed = data.get("max_angular_speed", 360.0)
        na.flying = data.get("flying", False)
        na.auto_move = data.get("auto_move", False)
        na.show_path_cells = data.get("show_path_cells", False)
        na.target_entity_id = data.get("target_entity_id", "")
        na.target_x = data.get("target_x", 0.0)
        na.target_y = data.get("target_y", 0.0)
        na.target_z = data.get("target_z", 0.0)
        na.repath_interval = data.get("repath_interval", 1.0)
        na.arrival_distance = data.get("arrival_distance", 0.5)
        na.nav_resolution = data.get("nav_resolution", 512)
        na.nav_world_size = data.get("nav_world_size", 500.0)
        return na
