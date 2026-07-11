# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QFont as QF, QImage, QPainter as QP, QBrush as QB, QColor as QC


def get_or_create_icon_texture(vp, comp_type_name: str, icon_color: tuple, icon_label: str, icon_path: Optional[str]) -> Optional[Any]:
    if not icon_path:
        auto_path = os.path.join(os.path.dirname(__file__), '..', 'gizmo_icons', f'{comp_type_name}.png')
        if os.path.exists(auto_path):
            icon_path = auto_path
    if icon_path:
        tex = vp._renderer.create_icon_texture_from_png(icon_path)
        if tex:
            return tex
    key = f"__comp_icon_{comp_type_name}"
    tex = vp._renderer._icon_textures.get(key)
    if tex:
        return tex
    r, g, b = icon_color
    size = 32
    qimg = QImage(size, size, QImage.Format.Format_RGBA8888)
    qimg.fill(Qt.GlobalColor.transparent)
    p = QP(qimg)
    p.setRenderHint(QP.RenderHint.Antialiasing)
    bg = QC(r, g, b)
    p.setBrush(QB(bg))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(0, 0, size, size, 4, 4)
    if icon_label:
        p.setPen(QC(255, 255, 255))
        f3 = QF("Segoe UI", 14, QF.Weight.Bold)
        f3.setStyleStrategy(QF.StyleStrategy.ForceOutline)
        p.setFont(f3)
        p.drawText(QRect(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, icon_label[0].upper())
    p.end()
    rgba = qimg.bits().asstring(size * size * 4)
    return vp._renderer.create_icon_texture_from_data(rgba, size, size, key)


def render_component_icons_gl(vp):
    scene = vp._engine.scene
    if not scene or not vp._gizmo_icons_visible:
        return
    from core.components.transform import Transform
    from core.config.config import get_global_config
    cfg = get_global_config()
    if not cfg.get("gizmo.show_icons", True):
        return
    w, h = vp.width(), vp.height()
    if w <= 0 or h <= 0 or not vp._renderer:
        return
    dpr = vp.devicePixelRatio()
    pw, ph = w * dpr, h * dpr
    vp_mat = vp._cam.get_view_matrix() * vp._cam.get_projection_matrix(w / max(1, h))
    cam_pos = vp._cam.position
    icon_scale = cfg.get("gizmo.icon_scale", 2.0)
    base_size = 16 * icon_scale
    min_size = cfg.get("gizmo.icon_min_size", 8.0)
    max_size = cfg.get("gizmo.icon_max_size", 256.0)
    ref_distance = cfg.get("gizmo.icon_ref_distance", 4.5)
    near_fade_start = cfg.get("gizmo.icon_near_fade_start", 0.25)
    near_fade_end = cfg.get("gizmo.icon_near_fade_end", 2.5)
    for entity in scene.get_all_entities():
        if not entity.active:
            continue
        if len(entity._components) <= 1:
            continue
        t = entity._type_map.get(Transform)
        if not t:
            continue
        t = t[0]
        dist = (t.position - cam_pos).length()
        screen_scale = ref_distance / max(dist, 0.001)
        icon_size = max(min_size, min(max_size, base_size * screen_scale))
        alpha = 1.0
        if dist < near_fade_end:
            alpha = max(0.0, (dist - near_fade_start) / (near_fade_end - near_fade_start))
        from editor.viewport.projection import project_world_pos
        sp = project_world_pos(vp, t.position, vp_mat, pw, ph)
        if not sp:
            continue
        y_off = 0
        for comp in entity.get_all_components():
            if isinstance(comp, Transform):
                continue
            icon = comp.gizmo_icon
            if not icon:
                continue
            r, g, b, label = icon
            icon_path = getattr(comp, '_gizmo_icon_path', None)
            tex = get_or_create_icon_texture(vp, type(comp).__name__, (r, g, b), label, icon_path)
            if tex:
                sx, sy = sp[0], sp[1] + y_off
                sz = icon_size * dpr
                vp._renderer._render_icon(tex, sx, sy, sz, alpha, pw, ph)
            y_off += icon_size * dpr + 2 * dpr
