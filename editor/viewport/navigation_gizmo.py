# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter

from core.math.math3d import (
    Vec3,
    Vec4,
    mat4_mul_flat,
    mat4_mul_vec_flat,
    mat4_invert_flat,
    look_at_flat,
    rotate_around_axis_flat,
    axis_vector_flat,
    coordinate_system_axes,
    is_right_handed_cs,
    point_in_circle,
)
from core.components.navigation_gizmo.navigation_gizmo import CoordinateSystem


class _DrawList:
    def __init__(self, painter: QPainter, origin_x: float, origin_y: float):
        self._p = painter
        self._ox = origin_x
        self._oy = origin_y

    def _tx(self, x):
        return self._ox + x

    def _ty(self, y):
        return self._oy + y

    def add_line(self, ax, ay, bx, by, color, thickness):
        self._p.setPen(QColor.fromRgba(color))
        self._p.setBrush(Qt.BrushStyle.NoBrush)
        self._p.drawLine(int(self._tx(ax)), int(self._ty(ay)), int(self._tx(bx)), int(self._ty(by)))

    def add_circle_filled(self, cx, cy, r, color):
        self._p.setPen(Qt.PenStyle.NoPen)
        self._p.setBrush(QColor.fromRgba(color))
        self._p.drawEllipse(int(self._tx(cx) - r), int(self._ty(cy) - r), int(r * 2), int(r * 2))

    def add_circle(self, cx, cy, r, color, thickness):
        self._p.setPen(QColor.fromRgba(color))
        self._p.setBrush(Qt.BrushStyle.NoBrush)
        self._p.drawEllipse(int(self._tx(cx) - r), int(self._ty(cy) - r), int(r * 2), int(r * 2))

    def add_text(self, x, y, color, text):
        self._p.setPen(QColor.fromRgba(color))
        self._p.drawText(int(x), int(y), text)


def pack_rgba(r, g, b, a=255):
    return (int(a) << 24) | (int(b) << 16) | (int(g) << 8) | int(r)


def _darken(rgb, factor=0.6, a=255):
    r, g, b = rgb
    return pack_rgba(int(r * factor), int(g * factor), int(b * factor), a)


_X_COLOR = (244, 40, 40)
_Y_COLOR = (40, 244, 40)
_Z_COLOR = (90, 90, 210)
_HOVER_COLOR = (90, 90, 90)

_AXIS_COLORS = (_X_COLOR, _Y_COLOR, _Z_COLOR)
_AXIS_LABELS = ("X", "Y", "Z")


class _Config:
    lineThicknessScale = 0.017
    axisLengthScale = 0.33
    positiveRadiusScale = 0.075
    negativeRadiusScale = 0.05
    hoverCircleRadiusScale = 0.88
    dragThreshold = 3.0
    dragSensitivity = 0.01
    drag = True
    click = True
    xCircleFrontColor = pack_rgba(*_X_COLOR)
    xCircleBackColor = _darken(_X_COLOR)
    yCircleFrontColor = pack_rgba(*_Y_COLOR)
    yCircleBackColor = _darken(_Y_COLOR)
    zCircleFrontColor = pack_rgba(*_Z_COLOR)
    zCircleBackColor = _darken(_Z_COLOR)
    hoverCircleColor = pack_rgba(*_HOVER_COLOR, 90)


class _Interaction:
    def __init__(self):
        self.dragging = False
        self.dragged = False
        self.clickMousePos = (0.0, 0.0)


class _GizmoState:
    def __init__(self):
        self.config = _Config()
        self.interaction = _Interaction()
        self.mX = 0.0
        self.mY = 0.0
        self.mSize = 100.0
        self.draw_list = None


_state = _GizmoState()

config = _state.config


def set_rect(x: float, y: float, size: float):
    _state.mX = x
    _state.mY = y
    _state.mSize = size


def set_draw_list(draw_list):
    _state.draw_list = draw_list


def _orbit_view(view_matrix, mouse_delta, pivot_distance, cs):
    if pivot_distance <= 0.0 or (mouse_delta[0] == 0.0 and mouse_delta[1] == 0.0):
        return False
    model_mat = mat4_invert_flat(view_matrix)
    eye = Vec3(model_mat[12], model_mat[13], model_mat[14])
    forward = Vec3(model_mat[8], model_mat[9], model_mat[10])
    pivot_pos = eye - forward * pivot_distance
    offset = eye - pivot_pos
    camera_up = Vec3(model_mat[4], model_mat[5], model_mat[6])
    camera_right = Vec3(model_mat[0], model_mat[1], model_mat[2])

    orbit_up = Vec3(*axis_vector_flat(coordinate_system_axes(cs)[1]))
    ox, oy, oz = rotate_around_axis_flat(offset.x, offset.y, offset.z, orbit_up.x, orbit_up.y, orbit_up.z, -mouse_delta[0])
    ux, uy, uz = rotate_around_axis_flat(camera_up.x, camera_up.y, camera_up.z, orbit_up.x, orbit_up.y, orbit_up.z, -mouse_delta[0])
    rx, ry, rz = rotate_around_axis_flat(camera_right.x, camera_right.y, camera_right.z, orbit_up.x, orbit_up.y, orbit_up.z, -mouse_delta[0])
    offset = Vec3(ox, oy, oz)
    camera_up = Vec3(ux, uy, uz)
    camera_right = Vec3(rx, ry, rz)

    ox, oy, oz = rotate_around_axis_flat(offset.x, offset.y, offset.z, camera_right.x, camera_right.y, camera_right.z, -mouse_delta[1])
    ux, uy, uz = rotate_around_axis_flat(camera_up.x, camera_up.y, camera_up.z, camera_right.x, camera_right.y, camera_right.z, -mouse_delta[1])
    offset = Vec3(ox, oy, oz)
    camera_up = Vec3(ux, uy, uz)

    new_eye = pivot_pos + offset
    for i, v in enumerate(new_eye):
        view_matrix[12 + i] = v
    view_matrix[4] = camera_up.x; view_matrix[5] = camera_up.y; view_matrix[6] = camera_up.z
    view_matrix[0] = camera_right.x; view_matrix[1] = camera_right.y; view_matrix[2] = camera_right.z
    f = (pivot_pos - new_eye).normalized()
    z_sign = -1.0 if is_right_handed_cs(cs) else 1.0
    view_matrix[8] = z_sign * f.x; view_matrix[9] = z_sign * f.y; view_matrix[10] = z_sign * f.z
    return True


def _draw_positive_line(dl, center_x, center_y, axis_x, axis_y, color, radius, thickness, text, selected):
    line_end_x = center_x + axis_x
    line_end_y = center_y + axis_y
    dl.add_line(center_x, center_y, line_end_x, line_end_y, color, thickness)
    dl.add_circle_filled(line_end_x, line_end_y, radius, color)
    font = QFont("Segoe UI", max(8, int(radius)), QFont.Weight.Bold)
    dl._p.setFont(font)
    label_size = QFontMetrics(font).size(0, text)
    text_x = math.floor(line_end_x - 0.5 * label_size.width())
    text_y = math.floor(line_end_y - 0.5 * label_size.height()) + label_size.height()
    if selected:
        dl.add_circle(line_end_x, line_end_y, radius, pack_rgba(255, 255, 255, 255), 1.1)
        dl.add_text(text_x, text_y, pack_rgba(255, 255, 255, 255), text)
    else:
        dl.add_text(text_x, text_y, pack_rgba(0, 0, 0, 255), text)


def _draw_negative_line(dl, center_x, center_y, axis_x, axis_y, color, radius, selected):
    line_end_x = center_x - axis_x
    line_end_y = center_y - axis_y
    dl.add_circle_filled(line_end_x, line_end_y, radius, color)
    if selected:
        dl.add_circle(line_end_x, line_end_y, radius, pack_rgba(255, 255, 255, 255), 1.1)


def draw_gizmo(view_matrix, projection_matrix, pivot_distance=0.0, cs=CoordinateSystem.XYZ,
               mouse_pos=(0.0, 0.0), mouse_down=False, mouse_clicked=False,
               mouse_released=False, mouse_delta=(0.0, 0.0)):
    cfg = _state.config
    dl = _state.draw_list
    size = _state.mSize
    h_size = size * 0.5
    center_x = _state.mX + h_size
    center_y = _state.mY + h_size

    view_proj = mat4_mul_flat(view_matrix, projection_matrix)
    aspect_ratio = projection_matrix[5] / projection_matrix[0]
    view_proj[0] *= aspect_ratio
    view_proj[8] *= aspect_ratio

    axis_length = size * cfg.axisLengthScale
    axes = [
        Vec4(*mat4_mul_vec_flat(view_proj, axis_length, 0.0, 0.0, 0.0)),
        Vec4(*mat4_mul_vec_flat(view_proj, 0.0, axis_length, 0.0, 0.0)),
        Vec4(*mat4_mul_vec_flat(view_proj, 0.0, 0.0, axis_length, 0.0)),
    ]

    interactive = pivot_distance > 0.0
    mx, my = mouse_pos

    hover_circle_radius = h_size * cfg.hoverCircleRadiusScale
    hovered = interactive and point_in_circle(center_x, center_y, hover_circle_radius, mx, my)
    dragging = _state.interaction.dragging
    if cfg.hoverCircleColor != 0 and (hovered or dragging):
        dl.add_circle_filled(center_x, center_y, hover_circle_radius, cfg.hoverCircleColor)

    positive_radius = size * cfg.positiveRadiusScale
    negative_radius = size * cfg.negativeRadiusScale
    positive_closer = [0.0 >= a.w for a in axes]

    pairs = [(i, axes[i].w) for i in range(3)] + [(i + 3, -axes[i].w) for i in range(3)]
    pairs.sort(key=lambda p: p[1], reverse=True)

    selection = -1
    for it in reversed(pairs):
        idx = it[0]
        axis = idx % 3
        sign = 1.0 if idx < 3 else -1.0
        radius = positive_radius if idx < 3 else negative_radius
        ax = axes[axis].x * sign
        ay = -axes[axis].y * sign
        if point_in_circle(center_x + ax, center_y + ay, radius, mx, my):
            selection = idx
            break

    line_thickness = size * cfg.lineThicknessScale
    front_color = (cfg.xCircleFrontColor, cfg.yCircleFrontColor, cfg.zCircleFrontColor)
    back_color = (cfg.xCircleBackColor, cfg.yCircleBackColor, cfg.zCircleBackColor)
    for fst, _ in pairs:
        axis = fst % 3
        sign = 1.0 if fst < 3 else -1.0
        color = (front_color[axis] if positive_closer[axis] else back_color[axis]) if fst < 3 else \
                (back_color[axis] if positive_closer[axis] else front_color[axis])
        ax = axes[axis].x * sign
        ay = -axes[axis].y * sign
        selected = not dragging and selection == fst
        if fst < 3:
            _draw_positive_line(dl, center_x, center_y, ax, ay, color,
                                positive_radius, line_thickness, _AXIS_LABELS[axis], selected)
        else:
            _draw_negative_line(dl, center_x, center_y, ax, ay, color, negative_radius, selected)

    result = None

    if cfg.drag:
        if _state.interaction.dragging and mouse_down:
            dx = mx - _state.interaction.clickMousePos[0]
            dy = my - _state.interaction.clickMousePos[1]
            drag_threshold_sq = cfg.dragThreshold * cfg.dragThreshold
            if dx * dx + dy * dy > drag_threshold_sq:
                _state.interaction.dragged = True
                if _orbit_view(view_matrix, (mouse_delta[0] * cfg.dragSensitivity, mouse_delta[1] * cfg.dragSensitivity), pivot_distance, cs):
                    result = ("orbit", None)
        if mouse_clicked:
            _state.interaction.dragging = hovered
            _state.interaction.clickMousePos = (mx, my)
        if mouse_released:
            was_dragged = _state.interaction.dragged
            _state.interaction.dragging = False
            _state.interaction.dragged = False
            if was_dragged:
                result = ("orbit_end", None)

    if cfg.click and selection != -1 and mouse_released:
        model_mat = mat4_invert_flat(view_matrix)
        eye = Vec3(model_mat[12], model_mat[13], model_mat[14])
        forward = Vec3(model_mat[8], model_mat[9], model_mat[10])
        pivot_pos = eye - forward * pivot_distance
        selected_axis = selection % 3
        selected_sign = 1.0 if selection < 3 else -1.0
        coordinate_axes = coordinate_system_axes(cs)
        up_axis = coordinate_axes[1]
        reference_axis = coordinate_axes[2]
        reference_sign = -selected_sign if is_right_handed_cs(cs) else selected_sign
        eye_pos = pivot_pos + Vec3(*axis_vector_flat(selected_axis, selected_sign * pivot_distance))
        up = Vec3(*axis_vector_flat(reference_axis, reference_sign)) if selected_axis == up_axis else Vec3(*axis_vector_flat(up_axis))
        new_view = look_at_flat(
            (eye_pos.x, eye_pos.y, eye_pos.z),
            (pivot_pos.x, pivot_pos.y, pivot_pos.z),
            (up.x, up.y, up.z),
            int(cs),
        )
        for i in range(16):
            view_matrix[i] = new_view[i]
        result = ("click", selection)

    return result, selection, (center_x, center_y), (hovered or dragging)


def reset_interaction():
    _state.interaction.dragging = False
    _state.interaction.dragged = False


def set_hover_circle_color(color):
    _state.config.hoverCircleColor = color


def draw_navigation_gizmo_overlay(vp, qp):
    indicator = getattr(vp, "_navigation_gizmo", None)
    if indicator is None or not getattr(vp, "_navigation_gizmo_enabled", True):
        return
    if not vp._cam:
        return

    w = vp.width()
    h = vp.height()
    if w <= 0 or h <= 0:
        return

    cam = vp._cam
    size = indicator.rect_size

    toolbar_h = getattr(vp, "_toolbar", None)
    top_offset = (toolbar_h.height() if toolbar_h is not None else 30) + 8

    x = w - size - indicator.corner_offset_x
    y = top_offset + indicator.corner_offset_y

    fw, fh = vp._get_physical_dims()
    aspect = fw / max(1, fh)
    view = cam.get_view_matrix()
    proj = cam.get_projection_matrix(aspect)
    view_col = view.to_f32().tolist()
    proj_col = proj.to_f32().tolist()

    cfg = config
    cfg.lineThicknessScale = 0.017
    cfg.axisLengthScale = 0.33
    cfg.positiveRadiusScale = 0.075
    cfg.negativeRadiusScale = 0.05
    cfg.hoverCircleRadiusScale = 0.88
    cfg.dragThreshold = 3.0
    cfg.dragSensitivity = indicator.drag_sensitivity
    cfg.drag = indicator.drag_enabled
    cfg.click = indicator.click_enabled
    if getattr(cam, "_is_orthographic", False):
        fov_rad = math.radians(getattr(cam, "fov", 60.0))
        persp_proj5 = 1.0 / math.tan(fov_rad * 0.5)
        ortho_proj5 = proj_col[5]
        if ortho_proj5 > 1e-6:
            factor = persp_proj5 / ortho_proj5
            factor = max(0.01, min(factor, 1000.0))
            cfg.axisLengthScale *= factor

    set_rect(x, y, size)
    set_draw_list(_DrawList(qp, 0.0, 0.0))

    cs = indicator.coordinate_system.value if hasattr(indicator.coordinate_system, "value") else int(indicator.coordinate_system)

    mx, my = getattr(vp, "_ng_mouse_pos", (0.0, 0.0))
    down = getattr(vp, "_ng_mouse_down", False)
    clicked = getattr(vp, "_ng_clicked", False)
    released = getattr(vp, "_ng_released", False)
    delta = getattr(vp, "_ng_mouse_delta", (0.0, 0.0))

    result, selection, center, hov = draw_gizmo(
        view_col, proj_col,
        pivot_distance=indicator.pivot_distance,
        cs=cs,
        mouse_pos=(mx, my),
        mouse_down=down,
        mouse_clicked=clicked,
        mouse_released=released,
        mouse_delta=delta,
    )

    vp._ng_clicked = False
    vp._ng_released = False
    vp._ng_mouse_delta = (0.0, 0.0)
    vp._navigation_gizmo_hover = selection
    vp._navigation_gizmo_center = center
    vp._navigation_gizmo_interacting = hov

    if result is not None and result[0] == "click":
        snap_camera_to_axis(vp, result[1] % 3)


_WORLD_AXES = (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))
_AXIS_LABELS_3D = ("X", "Y", "Z")


def _get_gizmo_world_pos(vp):
    fw, fh = vp._get_physical_dims()
    if fw <= 0 or fh <= 0:
        return None, 0.0
    cam = vp._cam
    gizmo_dist = max((cam._position - cam._orbit_target).length(), 2.0)
    aspect = fw / max(1, fh)
    effective_fov = min(cam.fov, 90.0) if not cam.is_orthographic else 60.0
    tan_hfov = math.tan(math.radians(effective_fov) * 0.5)
    vx = 0.85 * aspect * gizmo_dist * tan_hfov
    vy = 0.8 * gizmo_dist * tan_hfov
    gizmo_pos = cam.position + cam.forward * gizmo_dist + cam._right() * vx + cam._up() * vy
    world_len = max(40.0 * gizmo_dist * tan_hfov / (fh * 0.5), 0.15)
    return gizmo_pos, world_len


def draw_axis_gizmo_api(vp, vp_mat):
    result = _get_gizmo_world_pos(vp)
    if result is None:
        return
    gizmo_pos, world_len = result
    if world_len < 0.01:
        return

    fw, fh = vp._get_physical_dims()

    neg_len = world_len * 0.5
    base_colors = ((1.0, 0.2, 0.2, 1.0), (0.2, 1.0, 0.2, 1.0), (0.2, 0.4, 1.0, 1.0))
    hover_col = (1.0, 1.0, 0.0, 1.0)

    tips = []
    neg_tips = []
    num_lines = 6
    starts = np.empty((num_lines, 3), dtype=np.float32)
    ends = np.empty((num_lines, 3), dtype=np.float32)
    cols = np.empty((num_lines, 4), dtype=np.float32)
    for i, (direction, color) in enumerate(zip(_WORLD_AXES, base_colors)):
        if i == vp._axis_gizmo_hover:
            col = hover_col
        else:
            col = color
        tip = gizmo_pos + direction * world_len
        nt = gizmo_pos - direction * neg_len
        tips.append(tip)
        neg_tips.append(nt)

        idx = i * 2
        starts[idx] = (gizmo_pos.x, gizmo_pos.y, gizmo_pos.z)
        ends[idx] = (tip.x, tip.y, tip.z)
        cols[idx] = col
        starts[idx + 1] = (gizmo_pos.x, gizmo_pos.y, gizmo_pos.z)
        ends[idx + 1] = (nt.x, nt.y, nt.z)
        cols[idx + 1] = (col[0] * 0.3, col[1] * 0.3, col[2] * 0.3, col[3])

    vp._renderer.render_gizmo_arrays(starts, ends, cols, vp_mat, fw, fh, thickness_multiplier=1.5)

    vp._axis_gizmo_tips_world = tips
    vp._axis_gizmo_neg_tips_world = neg_tips
    vp._axis_gizmo_center_world = gizmo_pos
    vp._axis_gizmo_world_len = world_len


def snap_camera_to_axis(vp, axis_idx):
    if axis_idx < 0 or axis_idx > 2:
        return
    cam = vp._cam
    target = Vec3.zero()
    if vp._selected_entities:
        t = vp._selected_entities[0].transform
        if t:
            target = t.position
    dist = max((cam._position - target).length(), 5.0)
    pos = target + _WORLD_AXES[axis_idx] * dist
    cam._position = pos
    cam.focus_on(target, dist)
