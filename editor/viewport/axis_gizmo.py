# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QPainterPath, QFontMetrics

from core.math3d import Vec3


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


def _world_to_screen(vp, pos):
    w, h = vp.width(), vp.height()
    if w <= 0 or h <= 0:
        return None
    try:
        mat = vp._cam.get_view_matrix() * vp._cam.get_projection_matrix(w / max(1, h))
        m = mat._d
    except Exception:
        return None
    c = np.array([pos.x, pos.y, pos.z, 1.0]) @ m
    if abs(c[3]) < 1e-8:
        return None
    n = c[:3] / c[3]
    sx = (n[0] + 1.0) * 0.5 * w
    sy = (1.0 - n[1]) * 0.5 * h
    return (sx, sy)


def draw_axis_gizmo_api(vp, vp_mat):
    if not vp._axis_gizmo_enabled or not vp._cam:
        return
    result = _get_gizmo_world_pos(vp)
    if result is None:
        return
    gizmo_pos, world_len = result
    if world_len < 0.01:
        return

    fw, fh = vp._get_physical_dims()

    neg_len = world_len * 0.5
    world_axes = [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)]
    base_colors = [(1.0, 0.2, 0.2, 1.0), (0.2, 1.0, 0.2, 1.0), (0.2, 0.4, 1.0, 1.0)]
    hover_col = (1.0, 1.0, 0.0, 1.0)

    tips = []
    neg_tips = []
    num_lines = 6
    starts = np.empty((num_lines, 3), dtype=np.float32)
    ends = np.empty((num_lines, 3), dtype=np.float32)
    cols = np.empty((num_lines, 4), dtype=np.float32)
    for i, (direction, color) in enumerate(zip(world_axes, base_colors)):
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


def draw_axis_gizmo_overlay(vp, painter):
    tips = getattr(vp, '_axis_gizmo_tips_world', None)
    neg_tips = getattr(vp, '_axis_gizmo_neg_tips_world', None)
    center_w = getattr(vp, '_axis_gizmo_center_world', None)
    if not tips or not neg_tips or not center_w or not vp._cam:
        return

    pos_list = [center_w] + tips + neg_tips
    screen_pts = [_world_to_screen(vp, p) for p in pos_list]
    if any(pt is None for pt in screen_pts):
        return

    cx, cy = screen_pts[0]
    tip_screens = screen_pts[1:4]
    neg_screens = screen_pts[4:7]

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    bg_rad = 42
    painter.setPen(QPen(QColor(55, 55, 62, 200), 1.0))
    painter.setBrush(QBrush(QColor(18, 18, 22, 175)))
    painter.drawEllipse(QPointF(cx, cy), bg_rad, bg_rad)

    colors_rgb = [(255, 70, 70), (70, 225, 80), (65, 135, 255)]
    hover_rgb = (255, 255, 110)

    for i in range(3):
        tx, ty = tip_screens[i]
        nx, ny = neg_screens[i]

        if i == vp._axis_gizmo_hover:
            r, g, b = hover_rgb
        else:
            r, g, b = colors_rgb[i]

        dx = tx - cx
        dy = ty - cy
        ln = math.sqrt(dx * dx + dy * dy)
        if ln < 1:
            continue
        dx /= ln
        dy /= ln

        arrow_len = 13
        arrow_half = 7
        hb_x = tx - dx * arrow_len
        hb_y = ty - dy * arrow_len
        px = -dy * arrow_half
        py = dx * arrow_half

        path = QPainterPath()
        path.moveTo(tx, ty)
        path.lineTo(hb_x + px, hb_y + py)
        path.lineTo(hb_x - px, hb_y - py)
        path.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(r, g, b, 220)))
        painter.drawPath(path)

        dot_r = 3
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(int(r * 0.4), int(g * 0.4), int(b * 0.4), 180)))
        painter.drawEllipse(QPointF(nx, ny), dot_r, dot_r)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(180, 180, 195, 200)))
    painter.drawEllipse(QPointF(cx, cy), 2.5, 2.5)

    font = QFont("Segoe UI", 9, QFont.Weight.Bold)
    font.setStyleStrategy(QFont.StyleStrategy.ForceOutline)
    painter.setFont(font)
    fm = QFontMetrics(font)

    for i in range(3):
        r, g, b = colors_rgb[i]
        tx, ty = tip_screens[i]
        dx = tx - cx
        dy = ty - cy
        ln = math.sqrt(dx * dx + dy * dy)
        if ln < 1:
            continue
        dx /= ln
        dy /= ln
        painter.setPen(QColor(min(255, int(r * 1.15)), min(255, int(g * 1.15)), min(255, int(b * 1.15)), 230))
        lx = tx + dx * 12
        ly = ty + dy * 12
        text = ("X", "Y", "Z")[i]
        br = fm.boundingRect(text)
        painter.drawText(QPointF(lx - br.width() / 2, ly + br.height() / 3), text)

    painter.restore()


def hit_test_axis_gizmo(vp, mx, my):
    tips = getattr(vp, '_axis_gizmo_tips_world', None)
    center = getattr(vp, '_axis_gizmo_center_world', None)
    if not tips or not center or not vp._cam:
        return -1
    screen_pts = [_world_to_screen(vp, p) for p in [center] + tips]
    if any(pt is None for pt in screen_pts):
        return -1
    cx, cy = screen_pts[0]
    dx = mx - cx
    dy = my - cy
    if math.sqrt(dx * dx + dy * dy) > 80:
        return -1
    best = -1
    best_d = 25.0
    for i, (sx, sy) in enumerate(screen_pts[1:]):
        d = math.sqrt((mx - sx) ** 2 + (my - sy) ** 2)
        if d < best_d:
            best_d = d
            best = i
    return best


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
    world_axes = [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)]
    pos = target + world_axes[axis_idx] * dist
    cam._position = pos
    cam.focus_on(target, dist)
