# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from editor.main_window.handlers import (
    on_entity_selected,
    on_entities_selected,
    on_entity_selected_from_viewport,
    on_entities_selected_from_viewport,
    on_entity_double_clicked,
    on_entity_dropped,
    on_gizmo_mode_changed,
    on_gizmo_space_changed,
    on_grid_toggled,
    on_snap_toggled,
    on_snap_t_changed,
    on_snap_r_changed,
    on_snap_s_changed,
    on_render_mode_changed,
    on_skybox_toggled,
    on_effects_toggled,
    on_camera_projection_changed,
    on_camera_2d_toggled,
    on_camera_type_changed,
    on_camera_2d_changed,
    on_import_model,
    on_file_selected,
    on_project_file_double_clicked,
    on_open_prefab_editor,
    on_undo_history_navigated,
)


def connect_signals(mw):
    mw._hierarchy.entity_selected.connect(lambda e: on_entity_selected(mw, e))
    mw._hierarchy.entities_selected.connect(lambda es: on_entities_selected(mw, es))
    mw._hierarchy.entity_double_clicked.connect(lambda eid: on_entity_double_clicked(mw, eid))
    mw._viewport.entity_selected.connect(lambda e: on_entity_selected_from_viewport(mw, e))
    mw._viewport.entities_selected.connect(lambda es: on_entities_selected_from_viewport(mw, es))
    mw._viewport.entity_dropped.connect(lambda p, w, e: on_entity_dropped(mw, p, w, e))
    mw._viewport.scene_modified.connect(mw._hierarchy.refresh)
    # Left section: SceneToolbar (gizmo mode, space)
    mw._scene_toolbar.gizmo_mode_changed.connect(lambda m: on_gizmo_mode_changed(mw, m))
    mw._scene_toolbar.gizmo_space_changed.connect(lambda s: on_gizmo_space_changed(mw, s))

    # Right section: RenderToolbar (render, camera, snap, grid, skybox, fx)
    mw._render_toolbar.render_mode_changed.connect(lambda m: on_render_mode_changed(mw, m))
    mw._render_toolbar.skybox_toggled.connect(lambda e: on_skybox_toggled(mw, e))
    mw._render_toolbar.effects_toggled.connect(lambda e: on_effects_toggled(mw, e))
    mw._render_toolbar.camera_projection_changed.connect(lambda: on_camera_projection_changed(mw))
    mw._render_toolbar.mode_2d_toggled.connect(lambda: on_camera_2d_toggled(mw))
    mw._render_toolbar.grid_toggled.connect(lambda e: on_grid_toggled(mw, e))
    mw._render_toolbar.snap_toggled.connect(lambda e: on_snap_toggled(mw, e))
    mw._render_toolbar.snap_translate_changed.connect(lambda v: on_snap_t_changed(mw, v))
    mw._render_toolbar.snap_rotate_changed.connect(lambda v: on_snap_r_changed(mw, v))
    mw._render_toolbar.snap_scale_changed.connect(lambda v: on_snap_s_changed(mw, v))

    mw._viewport.camera._on_projection_changed = lambda: on_camera_type_changed(mw)
    mw._viewport.camera._on_2d_mode_changed = lambda: on_camera_2d_changed(mw)
    mw._project.import_model_requested.connect(lambda p: on_import_model(mw, p))
    mw._project.file_selected.connect(lambda p: on_file_selected(mw, p))
    mw._project.file_double_clicked.connect(lambda p: on_project_file_double_clicked(mw, p))
    mw._hierarchy.select_prefab_asset.connect(lambda p: on_file_selected(mw, p))
    mw._hierarchy.open_prefab_editor.connect(lambda p: on_open_prefab_editor(mw, p))
    mw._inspector.open_prefab_editor.connect(lambda p: on_open_prefab_editor(mw, p))
    mw._viewport.gizmo.snap_enabled = mw._render_toolbar._snap_cb.isChecked()
    mw._viewport.gizmo.snap_translate = mw._render_toolbar._snap_t_sb.value()
    mw._viewport.gizmo.snap_rotate = mw._render_toolbar._snap_r_sb.value()
    mw._viewport.gizmo.snap_scale = mw._render_toolbar._snap_s_sb.value()
    mw._undo_history.history_navigated.connect(lambda: on_undo_history_navigated(mw))
