# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import time

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPen

from core.math3d import Vec3


def render_remote_collaborator_gizmos(vp, vp_mat, cam_pos, fw, fh):
    collab = vp._engine.collab_manager
    if not collab or not collab.connected:
        return
    scene = vp._engine.scene
    if not scene:
        return
    aspect = fw / max(1, fh)
    view_mat = vp._cam.get_view_matrix()
    proj_mat = vp._cam.get_projection_matrix(aspect)
    for peer_id, peer in collab.peers.items():
        color = peer.color + [1.0]
        cpos_vec = Vec3(*peer.camera_pos)
        cfwd_vec = Vec3(*peer.camera_fwd)
        cup_vec = Vec3(*peer.camera_up)
        if cpos_vec.length() > 0.01:
            size = 0.15
            lines = []
            lines.append((cpos_vec - cfwd_vec * size, cpos_vec + cfwd_vec * size, color))
            r = cfwd_vec.cross(cup_vec).normalized()
            u = r.cross(cfwd_vec).normalized()
            lines.append((cpos_vec - r * size, cpos_vec + r * size, color))
            lines.append((cpos_vec - u * size, cpos_vec + u * size, color))
            vp._renderer.render_gizmo_lines(lines, vp_mat, cam_pos, fw, fh, thickness_multiplier=2.0)
        if peer.selected_entity_ids:
            for eid in peer.selected_entity_ids:
                e = scene.get_entity(eid)
                if not e:
                    continue
                t = e.transform
                if not t:
                    continue
                pos = t.position
                gizmo_size = 0.15
                axis_defs = [
                    (Vec3(1, 0, 0), [1.0, 0.2, 0.2] + color[3:]),
                    (Vec3(0, 1, 0), [0.2, 1.0, 0.2] + color[3:]),
                    (Vec3(0, 0, 1), [0.2, 0.4, 1.0] + color[3:]),
                ]
                highlight_color = [1.0, 1.0, 0.0] + color[3:]
                lines = []
                for i, (d, c) in enumerate(axis_defs):
                    line_color = highlight_color if (i + 1) == peer.gizmo_hover_axis else c
                    tip = pos + d * gizmo_size
                    lines.append((pos, tip, line_color))
                    hs = gizmo_size * 0.2
                    ref_up = Vec3.up() if abs(d.y) < 0.9 else Vec3.right()
                    p1 = d.cross(ref_up).normalized()
                    p2 = d.cross(p1).normalized()
                    base = tip - d * hs
                    lines.append((tip, base + p1 * hs, line_color))
                    lines.append((tip, base - p1 * hs, line_color))
                    lines.append((tip, base + p2 * hs, line_color))
                    lines.append((tip, base - p2 * hs, line_color))
                if lines:
                    vp._renderer.render_gizmo_lines(lines, vp_mat, cam_pos, fw, fh, thickness_multiplier=2.0)
                outline_color = [peer.color[0], peer.color[1], peer.color[2], 1.0]
                vp._renderer.render_entity_outline(e, t.world_matrix, view_mat, proj_mat, outline_color)
        stale_deltas = []
        for eid, dd in peer.transform_deltas.items():
            if time.time() - dd["time"] > 0.5:
                stale_deltas.append(eid)
                continue
            e = scene.get_entity(eid)
            if not e:
                stale_deltas.append(eid)
                continue
            t = e.transform
            if not t:
                continue
            pos = t.position
            dv = Vec3(*dd["pos"])
            if dv.length() < 0.001:
                continue
            from_pos = pos - dv
            d = dv.normalized()
            gs = 0.12
            ref_up = Vec3.up() if abs(d.y) < 0.9 else Vec3.right()
            p1 = d.cross(ref_up).normalized()
            p2 = d.cross(p1).normalized()
            base = pos - d * gs
            delta_lines = [
                (from_pos, pos, color),
                (pos, base + p1 * gs, color),
                (pos, base - p1 * gs, color),
                (pos, base + p2 * gs, color),
                (pos, base - p2 * gs, color),
            ]
            vp._renderer.render_gizmo_lines(delta_lines, vp_mat, cam_pos, fw, fh, thickness_multiplier=2.0)
        for k in stale_deltas:
            del peer.transform_deltas[k]


def draw_remote_cursors(vp, painter):
    collab = vp._engine.collab_manager
    if not collab or not collab.connected:
        return
    for peer_id, peer in collab.peers.items():
        sx, sy = peer.cursor_screen
        if sx <= 0 and sy <= 0:
            continue
        c = peer.color
        r, g, b = int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
        qc = QColor(r, g, b, 220)
        painter.setPen(QPen(qc, 2))
        painter.setBrush(QBrush(QColor(r, g, b, 80)))
        painter.drawEllipse(int(sx) - 6, int(sy) - 6, 12, 12)
        painter.setPen(qc)
        f = QFont("Segoe UI", 8)
        painter.setFont(f)
        painter.drawText(int(sx) + 10, int(sy) - 5, peer.name)


def send_collab_cursor(vp, sx, sy):
    collab = vp._engine.collab_manager
    if not collab or not collab.connected:
        return
    now = time.time()
    interval = vp._collab_cursor_interval
    if hasattr(collab, 'settings'):
        interval = collab.settings.cursor_interval
    if now - vp._collab_throttle_cursor < interval:
        return
    vp._collab_throttle_cursor = now
    from editor.viewport.projection import screen_to_world
    world = screen_to_world(vp, sx, sy)
    collab.send_cursor(sx, sy, [world.x, world.y, world.z])


def send_collab_camera(vp):
    collab = vp._engine.collab_manager
    if not collab or not collab.connected:
        return
    now = time.time()
    interval = vp._collab_camera_interval
    if hasattr(collab, 'settings'):
        interval = collab.settings.camera_interval
    if now - vp._collab_throttle_camera < interval:
        return
    vp._collab_throttle_camera = now
    cam = vp._cam
    collab.send_camera(
        [cam.position.x, cam.position.y, cam.position.z],
        [cam.forward.x, cam.forward.y, cam.forward.z],
        [cam._up().x, cam._up().y, cam._up().z],
    )


def send_collab_selection(vp):
    collab = vp._engine.collab_manager
    if not collab or not collab.connected:
        return
    ids = [e.id for e in vp._selected_entities]
    collab.send_selection(ids)


def is_collab_locked(vp) -> bool:
    collab = vp._engine.collab_manager
    return collab and collab.connected and collab.play_mode_active


def send_collab_gizmo_state(vp):
    collab = vp._engine.collab_manager
    if not collab or not collab.connected:
        return
    now = time.time()
    interval = vp._collab_gizmo_interval
    if hasattr(collab, 'settings'):
        interval = collab.settings.gizmo_interval
    if now - vp._collab_throttle_gizmo < interval:
        return
    state = (vp._gizmo.mode.value, vp._gizmo._hover_axis.value, vp._gizmo._dragging)
    if state == vp._collab_last_gizmo_state:
        return
    vp._collab_last_gizmo_state = state
    vp._collab_throttle_gizmo = now
    collab.send_gizmo_state(*state)


def send_collab_entity_create(vp, entity_data: dict):
    collab = vp._engine.collab_manager
    if collab and collab.connected:
        collab.send_entity_create(entity_data)


def send_collab_entity_delete(vp, entity_id: str):
    collab = vp._engine.collab_manager
    if collab and collab.connected:
        collab.send_entity_delete(entity_id)


def send_collab_transforms(vp):
    collab = vp._engine.collab_manager
    if not collab or not collab.connected:
        return
    now = time.time()
    interval = vp._collab_transform_interval
    if hasattr(collab, 'settings'):
        interval = collab.settings.transform_interval
    if now - vp._collab_throttle_transform < interval:
        return
    vp._collab_throttle_transform = now
    for ent in vp._selected_entities:
        t = ent.transform
        if not t:
            continue
        collab.send_transform(
            ent.id,
            t.local_position.to_list(),
            t.local_rotation.to_list(),
            t.local_scale.to_list(),
        )


def _collab_on_undo(cmd):
    from core.commands import CreateEntityCommand, DeleteEntityCommand, SetComponentCommand, \
        AddComponentCommand, RemoveComponentCommand, CompoundCommand, PasteEntitiesCommand, \
        InstantiatePrefabCommand
    from core.engine import Engine
    eng = Engine.instance()
    collab = eng.collab_manager if hasattr(eng, 'collab_manager') else None
    if not collab or not collab.connected:
        return

    if isinstance(cmd, CompoundCommand):
        for sub in cmd._commands:
            _collab_on_undo(sub)
        return

    if isinstance(cmd, CreateEntityCommand) and cmd._entity_id:
        collab.send_entity_delete(cmd._entity_id)
    elif isinstance(cmd, DeleteEntityCommand) and cmd._entity_data:
        collab.send_entity_create(cmd._entity_data)
    elif isinstance(cmd, SetComponentCommand):
        eid = getattr(cmd, '_entity_id', None)
        if eid:
            collab.send_component_update(eid, cmd._component_type.__name__, cmd._prop, cmd._old)
    elif isinstance(cmd, AddComponentCommand):
        eid = cmd._entity.id if hasattr(cmd._entity, 'id') else None
        if eid and cmd._added_key:
            collab.send_component_remove(eid, cmd._added_key)
    elif isinstance(cmd, RemoveComponentCommand):
        eid = cmd._entity.id if hasattr(cmd._entity, 'id') else None
        if eid and cmd._component_data:
            collab.send_component_add(eid, cmd._component_cls.__name__, cmd._component_data)
    elif isinstance(cmd, (PasteEntitiesCommand, InstantiatePrefabCommand)):
        for eid in cmd._spawned_ids:
            collab.send_entity_delete(eid)


def _collab_on_redo(cmd):
    from core.commands import CreateEntityCommand, DeleteEntityCommand, SetComponentCommand, \
        AddComponentCommand, RemoveComponentCommand, CompoundCommand, PasteEntitiesCommand, \
        InstantiatePrefabCommand
    from core.engine import Engine
    eng = Engine.instance()
    collab = eng.collab_manager if hasattr(eng, 'collab_manager') else None
    if not collab or not collab.connected:
        return

    if isinstance(cmd, CompoundCommand):
        for sub in cmd._commands:
            _collab_on_redo(sub)
        return

    if isinstance(cmd, CreateEntityCommand):
        e = cmd._scene.get_entity(cmd._entity_id) if cmd._entity_id else None
        if e:
            collab.send_entity_create(e.serialize())
    elif isinstance(cmd, DeleteEntityCommand) and cmd._entity_id:
        collab.send_entity_delete(cmd._entity_id)
    elif isinstance(cmd, SetComponentCommand):
        eid = getattr(cmd, '_entity_id', None)
        if eid:
            collab.send_component_update(eid, cmd._component_type.__name__, cmd._prop, cmd._new)
    elif isinstance(cmd, AddComponentCommand):
        eid = cmd._entity.id if hasattr(cmd._entity, 'id') else None
        if eid and cmd._added_key:
            comp = cmd._entity._components.get(cmd._added_key)
            if comp:
                comp_data = {k: getattr(comp, k) for k in vars(comp) if not k.startswith("_")}
                collab.send_component_add(eid, cmd._component_cls.__name__, comp_data)
    elif isinstance(cmd, RemoveComponentCommand):
        eid = cmd._entity.id if hasattr(cmd._entity, 'id') else None
        if eid and cmd._component_key:
            collab.send_component_remove(eid, cmd._component_key)
    elif isinstance(cmd, PasteEntitiesCommand):
        for d in cmd._entity_datas:
            collab.send_entity_create(d)
    elif isinstance(cmd, InstantiatePrefabCommand):
        for eid in cmd._spawned_ids:
            e = cmd._scene.get_entity(eid)
            if e:
                collab.send_entity_create(e.serialize())


def setup_collab_undo_redo_hooks(engine):
    from core.commands import get_history
    history = get_history()
    history.set_on_undo(_collab_on_undo)
    history.set_on_redo(_collab_on_redo)
