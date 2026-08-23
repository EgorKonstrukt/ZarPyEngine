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
    key = f"__comp_icon_{comp_type_name}"
    tex = vp._renderer._icon_textures.get(key)
    if tex:
        return tex
    if not icon_path:
        auto_path = os.path.join(os.path.dirname(__file__), '..', 'gizmo_icons', f'{comp_type_name}.png')
        if os.path.exists(auto_path):
            icon_path = auto_path
    if icon_path:
        tex = vp._renderer.create_icon_texture_from_png(icon_path)
        if tex:
            vp._renderer._icon_textures[key] = tex
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
    tex = vp._renderer.create_icon_texture_from_data(rgba, size, size, key)
    vp._renderer._icon_textures[key] = tex
    return tex


def _icon_entities(scene):
    from core.components.transform import Transform
    rv = getattr(scene, '_render_version', 0)
    cached = getattr(scene, '_icon_entities_cache', None)
    if cached is not None and cached[0] == rv:
        return cached[1]
    result = []
    for e in scene.get_all_entities():
        if not e.active:
            continue
        tlist = e._type_map.get(Transform)
        if not tlist:
            continue
        icons = []
        for clist in e._type_map.values():
            comp = clist[0]
            if isinstance(comp, Transform):
                continue
            icon = getattr(comp, 'gizmo_icon', None)
            if not icon:
                continue
            icons.append((comp, icon, getattr(comp, '_gizmo_icon_path', None)))
        if icons:
            result.append((tlist[0], icons))
    scene._icon_entities_cache = (rv, result)
    return result


def render_component_icons_gl(vp, vp_mat=None, pw=None, ph=None):
    scene = vp._engine.scene
    if not scene or not vp._gizmo_icons_visible:
        return
    from core.config.config import get_global_config
    cfg = get_global_config()
    if not cfg.get("gizmo.show_icons", True):
        return
    dpr = vp.devicePixelRatio()
    if vp_mat is None:
        w, h = vp.width(), vp.height()
        if w <= 0 or h <= 0 or not vp._renderer:
            return
        vp_mat = vp._cam.get_view_matrix() * vp._cam.get_projection_matrix(w / max(1, h))
    if pw is None or ph is None:
        w, h = vp.width(), vp.height()
        if w <= 0 or h <= 0 or not vp._renderer:
            return
        pw, ph = w * dpr, h * dpr
    cam_pos = vp._cam.position
    icon_scale = cfg.get("gizmo.icon_scale", 2.0)
    base_size = 16 * icon_scale
    min_size = cfg.get("gizmo.icon_min_size", 8.0)
    max_size = cfg.get("gizmo.icon_max_size", 256.0)
    ref_distance = cfg.get("gizmo.icon_ref_distance", 4.5)
    near_fade_start = cfg.get("gizmo.icon_near_fade_start", 0.25)
    near_fade_end = cfg.get("gizmo.icon_near_fade_end", 2.5)
    groups: dict = {}
    for t, icons in _icon_entities(scene):
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
        sz = icon_size * dpr
        for comp, icon, icon_path in icons:
            r, g, b, label = icon
            key = type(comp).__name__
            grp = groups.get(key)
            if grp is None:
                tex = get_or_create_icon_texture(vp, key, (r, g, b), label, icon_path)
                if tex is None:
                    y_off += sz + 2 * dpr
                    continue
                grp = (tex, [])
                groups[key] = grp
            grp[1].append((sp[0], sp[1] + y_off, sz, alpha))
            y_off += sz + 2 * dpr
    batches = [(tex, quads) for (tex, quads) in groups.values() if quads]
    if batches:
        vp._renderer._render_icons_batched(batches, pw, ph)
