# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
from core.math.math3d import Vec3
from core.components.mesh_editor import ProBuilderMesh
from editor.viewport.projection import screen_to_ray


HANDLE_PX = 10


def _build_model_mat(translate: np.ndarray, scale: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = scale
    m[1, 1] = scale
    m[2, 2] = scale
    m[3, :3] = translate[:3]
    return m.ravel(order='F')


class PbScaleGizmo:
    def __init__(self, viewport):
        self._vp = viewport
        self._dragging = False
        self._drag_corner = -1
        self._drag_entity = None
        self._drag_pb = None
        self._old_world_corners: np.ndarray | None = None
        self._world_offset: np.ndarray | None = None
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, val: bool):
        self._active = val
        if not val:
            self._dragging = False

    def _get_selected_pb(self):
        ents = self._vp._selected_entities
        if not ents:
            return None
        ent = ents[0]
        pb = ent.get_component_by_name("ProBuilderMesh")
        if pb and pb.enabled and pb.positions.size > 0:
            return ent, pb
        return None

    def _get_world_corners_and_offset(self, pb, ent):
        corners = pb.get_aabb_corners()
        if corners.size == 0:
            return None, None
        tr = ent.transform
        if not tr:
            return None, None
        wm = tr.world_matrix._d
        world_pos = wm[:3, 3].copy()
        ones = np.ones((8, 1), dtype=np.float32)
        homo = np.hstack([corners, ones])
        world = (wm @ homo.T).T
        world = world[:, :3] / world[:, 3:4]
        return world, world_pos

    def render(self):
        result = self._get_selected_pb()
        if not result:
            return
        ent, pb = result
        world_corners, _ = self._get_world_corners_and_offset(pb, ent)
        if world_corners is None:
            return
        if self._dragging and self._old_world_corners is not None:
            world_corners = self._old_world_corners
        renderer = self._vp._renderer
        fw, fh = self._vp._get_physical_dims()
        cam = self._vp._cam
        vp_mat = cam.get_view_matrix() * cam.get_projection_matrix(fw / max(1, fh))
        dist = self._get_handle_screen_scale(world_corners, vp_mat, fw, fh)
        num = world_corners.shape[0]
        instance_data = np.zeros((num, 16), dtype=np.float32)
        for i in range(num):
            instance_data[i] = _build_model_mat(world_corners[i], dist)
        renderer.render_instanced_gizmo('quad', instance_data, vp_mat, num)

    def _get_handle_screen_scale(self, corners, vp_mat, fw, fh):
        center = corners.mean(axis=0)
        clip = np.append(center, 1.0) @ vp_mat._d
        if abs(clip[3]) < 1e-8:
            return 0.1
        handle_world = HANDLE_PX / max(fw, fh) * 2.0 * abs(clip[3])
        return max(handle_world, 0.01)

    def hit_test(self, lx: int, ly: int) -> int:
        result = self._get_selected_pb()
        if not result:
            return -1
        ent, pb = result
        world_corners, _ = self._get_world_corners_and_offset(pb, ent)
        if world_corners is None:
            return -1
        fw, fh = self._vp._get_physical_dims()
        cam = self._vp._cam
        vp_mat = cam.get_view_matrix() * cam.get_projection_matrix(fw / max(1, fh))
        best = -1
        best_dist = HANDLE_PX + 1
        for i in range(8):
            clip = np.append(world_corners[i], 1.0) @ vp_mat._d
            if abs(clip[3]) < 1e-8:
                continue
            ndc = clip[:3] / clip[3]
            if ndc[2] < -1 or ndc[2] > 1:
                continue
            sx = (ndc[0] + 1.0) * 0.5 * fw
            sy = (1.0 - ndc[1]) * 0.5 * fh
            dist = np.hypot(sx - lx, sy - ly)
            if dist < best_dist:
                best_dist = dist
                best = i
        return best

    def on_mouse_press(self, lx: int, ly: int) -> bool:
        if not self._active:
            return False
        idx = self.hit_test(lx, ly)
        if idx < 0:
            return False
        result = self._get_selected_pb()
        if not result:
            return False
        self._dragging = True
        self._drag_corner = idx
        self._drag_entity, self._drag_pb = result
        wc, offset = self._get_world_corners_and_offset(self._drag_pb, self._drag_entity)
        self._old_world_corners = wc
        self._world_offset = offset
        return True

    def on_mouse_move(self, lx: int, ly: int):
        if not self._dragging or self._drag_corner < 0:
            return
        origin, dir_ = screen_to_ray(self._vp, lx, ly)
        opposite = self._old_world_corners[7 - self._drag_corner]
        t = np.dot(opposite - np.array([origin.x, origin.y, origin.z]), np.array([dir_.x, dir_.y, dir_.z]))
        if t < 0:
            t = 0
        new_world = np.array([origin.x, origin.y, origin.z]) + np.array([dir_.x, dir_.y, dir_.z]) * t
        self._drag_pb.scale_from_corner(self._drag_corner, self._old_world_corners, new_world, self._world_offset)
        self._drag_pb._gpu_dirty = True

    def on_mouse_release(self):
        self._dragging = False
        self._drag_corner = -1
        self._drag_entity = None
        self._drag_pb = None
        self._old_world_corners = None
        self._world_offset = None
