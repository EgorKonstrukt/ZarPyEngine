# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import json
import os

import qtawesome as qta
from PyQt6.QtWidgets import QMessageBox, QFileDialog
from PyQt6.QtCore import Qt, QTimer

from core.math.math3d import Vec3
from core.foundation.logger import Logger
from editor.splash import SplashScreen


def on_entity_selected(mw, entity):
    from core.foundation.commands import get_history
    get_history().set_current_selection(entity)
    mw._hierarchy.set_selected_entity(entity)
    mw._inspector.set_entity(entity)
    mw._viewport.set_selected_entity(entity)
    if hasattr(mw, '_mesh_editor') and mw._mesh_editor:
        mw._mesh_editor.set_entity(entity)
    if hasattr(mw, '_terrain_editor') and mw._terrain_editor:
        mw._terrain_editor.set_entity(entity)
    if hasattr(mw, '_animation') and mw._animation:
        mw._animation.set_entity(entity)


def on_entities_selected(mw, entities):
    from core.foundation.commands import get_history
    get_history().set_current_selection(list(entities) if entities else None)
    mw._viewport.set_selected_entities(entities)
    if entities:
        mw._inspector.set_selected_entities(entities)
    if hasattr(mw, '_animation') and mw._animation:
        mw._animation.set_entity(entities[0] if entities else None)
    if hasattr(mw, '_terrain_editor') and mw._terrain_editor:
        mw._terrain_editor.set_entity(entities[0] if entities else None)


def on_entity_selected_from_viewport(mw, entity):
    from core.foundation.commands import get_history
    get_history().set_current_selection(entity)
    mw._inspector.set_entity(entity)
    mw._hierarchy.set_selected_entity(entity)
    if hasattr(mw, '_mesh_editor') and mw._mesh_editor:
        mw._mesh_editor.set_entity(entity)
    if hasattr(mw, '_terrain_editor') and mw._terrain_editor:
        mw._terrain_editor.set_entity(entity)
    if hasattr(mw, '_animation') and mw._animation:
        mw._animation.set_entity(entity)


def on_entities_selected_from_viewport(mw, entities):
    from core.foundation.commands import get_history
    get_history().set_current_selection(list(entities) if entities else None)
    if entities:
        mw._inspector.set_selected_entities(entities)
        mw._hierarchy.set_selected_entities(entities)
    else:
        mw._inspector.set_entity(None)
        mw._hierarchy.set_selected_entity(None)
    if hasattr(mw, '_animation') and mw._animation:
        mw._animation.set_entity(entities[0] if entities else None)


def on_entity_double_clicked(mw, eid: str):
    if not mw._engine.scene:
        return
    entity = mw._engine.scene.get_entity(eid)
    if not entity:
        return
    t = entity.transform
    if t:
        mw._viewport.camera.frame_bounds(t.position)


def reset_camera(mw):
    mw._viewport.camera._position = Vec3(0.0, 3.0, 10.0)
    mw._viewport.camera._yaw = 0.0
    mw._viewport.camera._pitch = -15.0
    mw._viewport.camera._focus_active = False


def on_project_file_double_clicked(mw, path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".zpep":
        instantiate_prefab(mw, path)
    elif ext == ".zterr":
        if hasattr(mw, '_terrain_editor') and mw._terrain_editor is not None:
            mw._terrain_editor.load_graph(path)
            mw._terrain_editor.show()
            mw._terrain_editor.raise_()


def on_open_prefab_editor(mw, path: str):
    from core.ecs.prefab import Prefab
    from core.ecs.ecs import Scene
    pref = Prefab.load(path)
    if not pref:
        Logger.warning(f"Cannot load prefab: {path}")
        return
    tab_name = f"Prefab: {pref.name}"
    edit_scene = Scene(tab_name)
    pref.instantiate(edit_scene, mw._engine._component_registry)
    edit_scene.mark_clean()
    origin = mw._scene_tab_manager.active_tab
    mw._scene_tab_manager.add_tab(tab_name, scene=edit_scene)
    info = mw._scene_tab_manager.get_tab_info(tab_name)
    if info:
        info.prefab_path = path
        info.origin_tab = origin
    mw._prefab_mode = True
    mw._prefab_path = path
    mw._viewport._prefab_btns.show()
    from core.config.editor_scale import scale
    mw._viewport._toolbar.setFixedHeight(scale(56))
    _start_prefab_autosave(mw, tab_name)


def _start_prefab_autosave(mw, prefab_tab_name: str = ""):
    from PyQt6.QtCore import QTimer
    mw._prefab_autosave_tab = prefab_tab_name
    if hasattr(mw, '_prefab_save_timer') and mw._prefab_save_timer:
        mw._prefab_save_timer.stop()
    mw._prefab_save_timer = QTimer()
    mw._prefab_save_timer.setSingleShot(True)
    mw._prefab_save_timer.timeout.connect(lambda: _do_auto_save(mw))
    if hasattr(mw, '_prefab_modified_slot'):
        try:
            mw._viewport.scene_modified.disconnect(mw._prefab_modified_slot)
        except TypeError:
            pass
    mw._prefab_modified_slot = lambda: _on_prefab_modified(mw)
    mw._viewport.scene_modified.connect(mw._prefab_modified_slot)


def _on_prefab_modified(mw):
    if not mw._prefab_mode:
        return
    if hasattr(mw, '_prefab_save_timer') and mw._prefab_save_timer:
        mw._prefab_save_timer.start(2000)


def _do_auto_save(mw):
    if not mw._prefab_mode:
        return
    tab_name = getattr(mw, '_prefab_autosave_tab', '')
    mgr = getattr(mw, '_scene_tab_manager', None)
    if tab_name and mgr:
        info = mgr.get_tab_info(tab_name)
        if info and info.scene:
            _save_prefab_direct(mw._prefab_path, info.scene)
            return
    on_save_prefab(mw)


def on_return_to_scene(mw):
    if hasattr(mw, '_prefab_modified_slot'):
        try:
            mw._viewport.scene_modified.disconnect(mw._prefab_modified_slot)
        except TypeError:
            pass
        mw._prefab_modified_slot = None
    if hasattr(mw, '_prefab_save_timer') and mw._prefab_save_timer:
        mw._prefab_save_timer.stop()
        mw._prefab_save_timer = None
    mgr = mw._scene_tab_manager
    prefab_tab = mgr.find_prefab_tab() if mgr else None
    if not prefab_tab:
        mw._prefab_mode = False
        mw._prefab_path = None
        mw._viewport._prefab_btns.hide()
        from core.config.editor_scale import scale
        mw._viewport._toolbar.setFixedHeight(scale(30))
        return
    info = mgr.get_tab_info(prefab_tab)
    origin = info.origin_tab if info else None
    prefab_path = info.prefab_path if info else None
    if prefab_path:
        from core.ecs.prefab import Prefab, PrefabLibrary
        pref = PrefabLibrary.load(prefab_path)
        prefab_guid = pref.guid if pref else None
        if info and info.scene:
            _save_prefab_direct(prefab_path, info.scene)
        if origin and origin in mgr._tabs:
            origin_info = mgr.get_tab_info(origin)
            if origin_info and origin_info.scene and prefab_guid:
                _refresh_prefab_instances(origin_info.scene, prefab_guid, mw._engine._component_registry)
    mgr.remove_tab(prefab_tab)
    mw._prefab_mode = False
    mw._prefab_path = None
    mw._prefab_autosave_tab = ""
    mw._viewport._prefab_btns.hide()
    from core.config.editor_scale import scale
    mw._viewport._toolbar.setFixedHeight(scale(30))


def _refresh_prefab_instances(scene, prefab_guid, registry):
    """Replace all instances of a prefab in the scene with fresh ones from the updated prefab file."""
    from core.ecs.prefab import Prefab, PrefabLibrary
    prefab_path = PrefabLibrary.path_for_guid(prefab_guid)
    if not prefab_path:
        return
    prefab = PrefabLibrary.load(prefab_path)
    if not prefab:
        return
    all_entities = scene.get_all_entities()
    instances = [e for e in all_entities if getattr(e, '_prefab_guid', None) == prefab_guid]
    if not instances:
        return
    roots = Prefab.get_prefab_roots(instances)
    saved = []
    for root in roots:
        t = root.transform
        saved.append({
            "parent_id": root._parent.id if root._parent else None,
            "position": list(t.local_position) if t else None,
            "rotation": list(t.local_euler_angles) if t else None,
            "scale": list(t.local_scale) if t else None,
        })
        scene.remove_entity(root.id)
    for data in saved:
        spawned = prefab.instantiate(scene, registry)
        for new_root in spawned:
            if data["parent_id"]:
                parent_ent = scene.get_entity(data["parent_id"])
                if parent_ent:
                    new_root.set_parent(parent_ent)
            t = new_root.transform
            if t and data["position"]:
                from core.math.math3d import Vec3
                t.local_position = Vec3(*data["position"])
            if t and data["rotation"]:
                from core.math.math3d import Vec3
                t.local_euler_angles = Vec3(*data["rotation"])
            if t and data["scale"]:
                from core.math.math3d import Vec3
                t.local_scale = Vec3(*data["scale"])


def _save_prefab_direct(path: str, scene):
    from core.ecs.prefab import Prefab, PrefabLibrary
    pref = Prefab.load(path)
    if not pref:
        return
    roots = scene.get_root_entities()
    prefab_roots = [e for e in roots if getattr(e, '_prefab_guid', None) == pref.guid]
    if not prefab_roots:
        return
    pref.capture(prefab_roots)
    pref.save(path)
    PrefabLibrary._prefabs[path] = pref
    PrefabLibrary._guids[pref.guid] = path


def on_save_prefab(mw):
    if not mw._prefab_path or not mw._engine.scene:
        return
    _save_prefab_direct(mw._prefab_path, mw._engine.scene)
    pref.capture(prefab_roots)
    pref.save(mw._prefab_path)
    PrefabLibrary._prefabs[mw._prefab_path] = pref
    PrefabLibrary._guids[pref.guid] = mw._prefab_path


def instantiate_prefab(mw, path: str, world_pos=None):
    if not mw._engine.scene:
        return
    from core.ecs.prefab import PrefabLibrary
    from core.foundation.commands import InstantiatePrefabCommand, get_history
    pref = PrefabLibrary.load(path)
    if not pref:
        return
    cmd = InstantiatePrefabCommand(mw._engine.scene, pref, mw._engine._component_registry)
    get_history().execute(cmd)
    spawned = [mw._engine.scene.get_entity(eid) for eid in cmd._spawned_ids]
    spawned = [e for e in spawned if e]
    if spawned and world_pos is not None:
        t = spawned[0].transform
        if t:
            t.local_position = world_pos
    if spawned:
        on_entity_selected(mw, spawned[0])
    mw._hierarchy.refresh()


def on_entity_dropped(mw, path_or_type: str, world_pos, entity_under_cursor=None):
    if not mw._engine.scene:
        return
    ext = os.path.splitext(path_or_type)[1].lower()

    if ext == ".zpep":
        instantiate_prefab(mw, path_or_type, world_pos)
        return

    if ext == ".zmat":
        if entity_under_cursor:
            _apply_material_to_entity(mw, path_or_type, entity_under_cursor)
        else:
            _drop_material_on_scene(mw, path_or_type, world_pos)
        mw._hierarchy.refresh()
        return

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tiff", ".webp"}
    audio_exts = {".wav", ".mp3", ".ogg"}
    model_exts = {".obj", ".fbx", ".stl", ".gltf", ".glb", ".usdz"}

    if ext in image_exts:
        _drop_image_asset(mw, path_or_type, world_pos, entity_under_cursor)
    elif ext in audio_exts:
        _drop_audio_asset(mw, path_or_type, world_pos)
    elif ext in model_exts:
        _drop_model_asset(mw, path_or_type, world_pos)
    else:
        _drop_generic_asset(mw, path_or_type, world_pos)

    mw._hierarchy.refresh()
    on_entity_selected(mw, mw._engine.scene.get_entity_by_name(os.path.basename(path_or_type) or "Dropped Object"))


def _apply_material_to_entity(mw, path: str, entity):
    if not entity:
        return
    from core.components.rendering.renderers.mesh_renderer import MeshRenderer
    from core.components.rendering.renderers.skinned_mesh_renderer import SkinnedMeshRenderer
    mr = entity.get_component(MeshRenderer)
    if mr is None:
        mr = entity.get_component(SkinnedMeshRenderer)
    if not mr:
        return
    from core.assets.material import MaterialLibrary
    mat = MaterialLibrary.load(path)
    if mat:
        mr.materials[0]["path"] = path
        Logger.info(f"Applied material {path} to {entity.name}")


def _drop_material_on_scene(mw, path: str, world_pos):
    from core.components import Transform, MeshFilter, MeshRenderer
    name = os.path.splitext(os.path.basename(path))[0]
    e = mw._engine.scene.create_entity(name)
    t = Transform()
    if world_pos:
        t.local_position = world_pos
    e.add_component(t)
    mf = MeshFilter()
    mf.mesh_name = "cube"
    e.add_component(mf)
    mr = MeshRenderer()
    mr.materials[0]["path"] = path
    e.add_component(mr)
    Logger.info(f"Created entity with material {path}")


def _drop_image_asset(mw, path: str, world_pos, entity_under_cursor):
    if entity_under_cursor:
        from core.components.rendering.renderers.sprite_renderer import SpriteRenderer
        sr = entity_under_cursor.get_component(SpriteRenderer)
        if sr:
            rel = _rel_path(mw, path)
            sr.texture_path = rel or path
            Logger.info(f"Applied texture {path} to {entity_under_cursor.name}")
            return
        from core.components.rendering.renderers.mesh_renderer import MeshRenderer
        mr = entity_under_cursor.get_component(MeshRenderer)
        if mr is None:
            from core.components.rendering.renderers.skinned_mesh_renderer import SkinnedMeshRenderer
            mr = entity_under_cursor.get_component(SkinnedMeshRenderer)
        if mr:
            _create_material_and_apply(mw, path, entity_under_cursor, mr)
            return

    from core.components.rendering.renderers.sprite_renderer import SpriteRenderer
    from core.components import Transform
    name = os.path.splitext(os.path.basename(path))[0]
    e = mw._engine.scene.create_entity(name)
    t = Transform()
    if world_pos:
        t.local_position = world_pos
    e.add_component(t)
    sr = SpriteRenderer()
    rel = _rel_path(mw, path)
    sr.texture_path = rel or path
    e.add_component(sr)
    Logger.info(f"Created sprite entity from {path}")


def _create_material_and_apply(mw, texture_path: str, entity, mr):
    from core.assets.material import Material
    from core.components import Transform, MeshFilter
    mat_name = os.path.splitext(os.path.basename(texture_path))[0] + "_mat"
    mat = Material(mat_name)
    mat.shader_path = "default"
    rel = _rel_path(mw, texture_path)
    tex_path = rel or texture_path
    mat.properties["_MainTex"] = tex_path
    mat.properties["diffuseMap"] = tex_path
    mats_dir = os.path.join(mw._engine.project_root, "materials")
    os.makedirs(mats_dir, exist_ok=True)
    mat_path = os.path.join(mats_dir, mat_name + ".zmat").replace("\\", "/")
    root = mw._engine.project_root
    try:
        rel_path = os.path.relpath(mat_path, root).replace("\\", "/")
    except ValueError:
        rel_path = mat_path
    mat.save(mat_path, mw._engine.project_root)
    mr.materials[0]["path"] = rel_path
    Logger.info(f"Created material {mat_path} and applied to {entity.name}")


def _drop_audio_asset(mw, path: str, world_pos):
    from core.components.audio.audio_source import AudioSource
    from core.components import Transform
    name = os.path.splitext(os.path.basename(path))[0]
    e = mw._engine.scene.create_entity(name)
    t = Transform()
    if world_pos:
        t.local_position = world_pos
    e.add_component(t)
    src = AudioSource()
    rel = _rel_path(mw, path)
    src.clip_path = rel or path
    src.play_on_awake = False
    e.add_component(src)
    Logger.info(f"Created audio source from {path}")


def _drop_model_asset(mw, path: str, world_pos):
    name = os.path.splitext(os.path.basename(path))[0]
    root = mw._engine.project_root
    try:
        rel = os.path.relpath(path, root)
        mesh_path = rel.replace("\\", "/") if not rel.startswith("..") else os.path.abspath(path)
    except ValueError:
        mesh_path = os.path.abspath(path)

    from core.components.rendering.skeleton.skinned_factory import create_skinned_mesh_entity
    ent = create_skinned_mesh_entity(mw._engine.scene, path, name, mesh_path, world_pos)
    if ent is None:
        from core.components import Transform, MeshFilter, MeshRenderer
        e = mw._engine.scene.create_entity(name)
        t = Transform()
        if world_pos:
            t.local_position = world_pos
        e.add_component(t)
        mf = MeshFilter()
        mf.mesh_name = name
        mf.mesh_path = mesh_path
        e.add_component(mf)
        e.add_component(MeshRenderer())
        ent = e
    Logger.info(f"Created model entity from {path}")


def _drop_generic_asset(mw, path: str, world_pos):
    from core.components import Transform
    name = os.path.splitext(os.path.basename(path))[0] or "Dropped Object"
    e = mw._engine.scene.create_entity(name)
    t = Transform()
    if world_pos:
        t.local_position = world_pos
    e.add_component(t)


def _rel_path(mw, path: str) -> str:
    try:
        root = mw._engine.project_root
        rel = os.path.relpath(path, root)
        return rel.replace("\\", "/") if not rel.startswith("..") else ""
    except ValueError:
        return ""


def on_scene_loaded(mw, scene):
    tab_name = (mw._scene_tab_manager.active_tab
                if hasattr(mw, '_scene_tab_manager') and mw._scene_tab_manager and mw._scene_tab_manager.active_tab
                else (scene.name if scene else "None"))
    mw._status_scene_lbl.setText(f"Scene: {tab_name}")
    mw.setWindowTitle(f"Zarin Engine Editor - {tab_name}")
    if hasattr(mw, '_viewport') and mw._viewport and hasattr(mw._viewport, 'renderer') and mw._viewport.renderer:
        mw._viewport.renderer.clear_scene_caches()


def on_gizmo_mode_changed(mw, mode):
    mw._viewport.gizmo.mode = mode


def on_gizmo_space_changed(mw, space):
    mw._viewport.gizmo.space = space


def on_grid_toggled(mw, enabled: bool):
    if mw._viewport.renderer:
        mw._viewport.renderer.show_grid = enabled
    mw._render_toolbar.save_state()


def on_snap_toggled(mw, enabled: bool):
    mw._viewport.gizmo.snap_enabled = enabled
    mw._render_toolbar.save_state()


def on_gizmo_vis_toggled(mw, checked: bool):
    mw._viewport._gizmo_visible = checked
    mw._viewport._gizmo_icons_visible = checked
    mw._viewport.update()


def on_snap_t_changed(mw, val: float):
    mw._viewport.gizmo.snap_translate = val
    mw._viewport.set_grid_step(val)
    mw._render_toolbar.save_state()


def on_snap_r_changed(mw, val: float):
    mw._viewport.gizmo.snap_rotate = val
    mw._render_toolbar.save_state()


def on_snap_s_changed(mw, val: float):
    mw._viewport.gizmo.snap_scale = val
    mw._render_toolbar.save_state()


def on_render_mode_changed(mw, mode):
    if mw._viewport.renderer:
        mw._viewport.renderer.render_mode = mode


def on_skybox_toggled(mw, enabled: bool):
    if mw._viewport.renderer:
        mw._viewport.renderer.skybox_enabled = enabled
    play = getattr(mw, '_play_dock', None)
    if play and hasattr(play, '_viewport') and play._viewport._renderer:
        play._viewport._renderer.skybox_enabled = enabled
    mw._render_toolbar.save_state()


def on_effects_toggled(mw, enabled: bool):
    if mw._viewport.renderer:
        mw._viewport.renderer.set_effects_enabled(enabled)
    play = getattr(mw, '_play_dock', None)
    if play and hasattr(play, '_viewport') and play._viewport._renderer:
        play._viewport._renderer.set_effects_enabled(enabled)
    mw._render_toolbar.save_state()


def on_camera_projection_changed(mw):
    mw._viewport.camera.toggle_projection()
    is_ortho = mw._viewport.camera.is_orthographic
    mw._render_toolbar._cam_persp_btn.setChecked(not is_ortho)


def on_camera_type_changed(mw):
    is_ortho = mw._viewport.camera.is_orthographic
    btn = mw._render_toolbar._cam_persp_btn
    if is_ortho:
        btn.setIcon(qta.icon("fa5s.camera-retro", color="#d4d4d4"))
        btn.setText(" Ortho")
    else:
        btn.setIcon(qta.icon("fa5s.camera", color="#d4d4d4"))
        btn.setText(" Perspective")


def on_camera_2d_toggled(mw):
    mw._viewport.camera.toggle_2d_mode()
    on_camera_2d_changed(mw)


def on_camera_2d_changed(mw):
    is_2d = mw._viewport.camera.is_2d_mode
    mw._render_toolbar._cam_2d_btn.setChecked(is_2d)
    on_camera_type_changed(mw)


def on_play_start(mw, _):
    from editor.main_window.toolbar import _set_play_btn_style
    _set_play_btn_style(mw._play_btn, "Stop")
    mw._pause_btn.setEnabled(True)
    mw._status_mode_lbl.setText("Play Mode")
    if hasattr(mw, '_gui_editor_widget') and mw._gui_editor_widget.canvas:
        canvas = mw._gui_editor_widget.canvas
        mw._viewport._overlay_canvas = canvas
        if hasattr(mw, '_play_dock') and hasattr(mw._play_dock, '_viewport'):
            mw._play_dock._viewport.show_overlay(canvas)
            if mw._viewport.renderer and mw._play_dock._viewport._renderer:
                mw._play_dock._viewport._renderer.skybox_enabled = mw._viewport.renderer.skybox_enabled
                mw._play_dock._viewport._renderer.set_effects_enabled(mw._viewport.renderer.effects_enabled)
        canvas.edit_mode = False


def on_play_stop(mw, _):
    from editor.main_window.toolbar import _set_play_btn_style
    _set_play_btn_style(mw._play_btn, "Play")
    mw._pause_btn.setEnabled(False)
    mw._status_mode_lbl.setText("Edit Mode")
    if hasattr(mw, '_viewport'):
        mw._viewport._overlay_canvas = None
    if hasattr(mw, '_play_dock') and hasattr(mw._play_dock, '_viewport'):
        mw._play_dock._viewport.hide_overlay()
    if hasattr(mw, '_gui_editor_widget'):
        mw._gui_editor_widget.canvas.edit_mode = True


def _tab_info(mw):
    if hasattr(mw, '_scene_tab_manager') and mw._scene_tab_manager:
        return mw._scene_tab_manager.get_tab_info(mw._scene_tab_manager.active_tab) if mw._scene_tab_manager.active_tab else None
    return None


def toggle_play_stop(mw):
    if mw._engine.play_mode:
        mw._engine.stop_play()
        mw._viewport_dock.raise_()
        info = _tab_info(mw)
        if info and info.scene_snapshot and mw._engine.scene:
            from core.components.rendering.postfx.graphics_effect import GraphicsEffect
            GraphicsEffect.cleanup_registry()
            from core.ecs.ecs import Scene as S
            from core.engine.engine import Engine as Eng
            restored = S.deserialize(info.scene_snapshot, Eng.instance()._component_registry)
            restored.path = mw._engine.scene.path
            mw._engine._scene = restored
            mw._engine._plugin_manager.notify_scene_loaded(restored)
            mw._engine._emit_event("scene_loaded", restored)
            info.scene = restored
            info.scene_snapshot = None
            info.play_mode = False
        if hasattr(mw, "_pre_play_selected_id") and mw._pre_play_selected_id:
            e = mw._engine.scene.get_entity(mw._pre_play_selected_id)
            if e:
                on_entity_selected(mw, e)
    else:
        sel = getattr(mw._hierarchy, "_selected_entity", None)
        mw._pre_play_selected_id = sel.id if sel else None
        if mw._engine.scene:
            info = _tab_info(mw)
            if info:
                info.scene_snapshot = mw._engine.scene.serialize()
                info.play_mode = True
        mw._engine.start_play()
        mw._play_dock.raise_()


def toggle_pause(mw):
    if mw._play_dock:
        mw._play_dock._toggle_pause()
    info = _tab_info(mw)
    if info and info.scene_snapshot and mw._engine.scene:
        from core.components.rendering.postfx.graphics_effect import GraphicsEffect
        GraphicsEffect.cleanup_registry()
        from core.ecs.ecs import Scene as S
        from core.engine.engine import Engine as Eng
        restored = S.deserialize(info.scene_snapshot, Eng.instance()._component_registry)
        restored.path = mw._engine.scene.path
        mw._engine._scene = restored
        mw._engine._plugin_manager.notify_scene_loaded(restored)
        mw._engine._emit_event("scene_loaded", restored)
        info.scene = restored
        info.scene_snapshot = None
        info.play_mode = False
    if hasattr(mw, "_pre_play_selected_id") and mw._pre_play_selected_id:
        e = mw._engine.scene.get_entity(mw._pre_play_selected_id)
        if e:
            on_entity_selected(mw, e)


def new_scene(mw):
    mw._scene_tab_manager.add_tab("NewScene")


def open_scene(mw):
    path, _ = QFileDialog.getOpenFileName(mw, "Open Scene", "scenes/", "Scenes (*.zpes)")
    if path:
        _do_open_scene(mw, path)


def _confirm_discard_dirty(mw) -> bool:
    if mw._engine.scene and mw._engine.scene.dirty:
        reply = QMessageBox.question(mw, "Unsaved Changes", "Save current scene?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        if reply == QMessageBox.StandardButton.Yes:
            mw._engine.save_scene()
    return True


def _do_open_scene(mw, path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tab_name = os.path.splitext(os.path.basename(path))[0]
        mw._engine.resolve_scene_paths(data)
        from core.ecs.ecs import Scene, ComponentRegistry
        scene = Scene.deserialize(data, ComponentRegistry)
        scene.path = path
        scene.mark_clean()
        mw._scene_tab_manager.add_tab(tab_name, path=path, scene=scene)
    except Exception as e:
        Logger.error(f"Error opening scene: {e}", e)


def _sync_tab_after_save(mw):
    if hasattr(mw, '_scene_tab_manager') and mw._scene_tab_manager:
        active = mw._scene_tab_manager.active_tab
        if active:
            info = mw._scene_tab_manager.get_tab_info(active)
            if info:
                info.path = mw._engine.scene.path if mw._engine.scene else info.path


def save_scene(mw):
    if not mw._engine.scene:
        return
    if not mw._engine.scene.path:
        save_scene_as(mw)
    else:
        mw._engine.save_scene()
        _sync_tab_after_save(mw)


def save_scene_as(mw):
    if not mw._engine.scene:
        return
    path, _ = QFileDialog.getSaveFileName(mw, "Save Scene", "scenes/", "Scenes (*.zpes)")
    if path:
        if not path.endswith(".zpes"):
            path += ".zpes"
        mw._engine.save_scene(path)
        if hasattr(mw, '_scene_tab_manager') and mw._scene_tab_manager:
            active = mw._scene_tab_manager.active_tab
            if active:
                info = mw._scene_tab_manager.get_tab_info(active)
                if info:
                    info.path = path


def sync_after_undo(mw):
    from core.foundation.commands import get_history
    h = get_history()
    sel = h.current_selection if isinstance(getattr(h, 'current_selection', None), list) else (h.last_affected_entity or h.current_selection)
    scene = mw._engine.scene
    if isinstance(sel, list):
        if scene:
            resolved = []
            for e in sel:
                if e:
                    live = scene.get_entity(e.id)
                    if live:
                        resolved.append(live)
            sel = resolved
        else:
            sel = []
    elif sel and scene:
        live = scene.get_entity(sel.id)
        if live:
            sel = live
        else:
            sel = None
    mw._hierarchy.refresh()
    mw._hierarchy.blockSignals(True)
    if isinstance(sel, list):
        mw._hierarchy.set_selected_entities(sel)
        mw._inspector.set_selected_entities(sel)
        mw._viewport.set_selected_entities(sel)
    elif sel:
        mw._hierarchy.set_selected_entity(sel)
        mw._inspector.setUpdatesEnabled(False)
        mw._inspector.set_entity(sel)
        mw._inspector.setUpdatesEnabled(True)
        mw._viewport.set_selected_entity(sel)
        if hasattr(mw, '_animation') and mw._animation:
            mw._animation.set_entity(sel)
    else:
        sel_ent = mw._hierarchy._selected_entity
        mw._inspector.set_entity(sel_ent if sel_ent else None)
        mw._viewport.set_selected_entity(sel_ent if sel_ent else None)
        if hasattr(mw, '_animation') and mw._animation:
            mw._animation.set_entity(sel_ent if sel_ent else None)
    mw._hierarchy.blockSignals(False)
    if scene:
        scene.mark_dirty()


def undo(mw):
    from core.foundation.commands import get_history
    get_history().undo()
    sync_after_undo(mw)


def redo(mw):
    from core.foundation.commands import get_history
    get_history().redo()
    sync_after_undo(mw)


def on_undo_history_navigated(mw):
    sync_after_undo(mw)


def on_file_selected(mw, path: str):
    mw._inspector.show_import_settings(path)


def on_import_model(mw, path: str):
    if not mw._engine.scene:
        return
    name = os.path.splitext(os.path.basename(path))[0]
    root = mw._engine.project_root
    try:
        rel = os.path.relpath(path, root)
        mesh_path = rel.replace("\\", "/") if not rel.startswith("..") else os.path.abspath(path)
    except ValueError:
        mesh_path = os.path.abspath(path)

    from core.components.rendering.skeleton.skinned_factory import create_skinned_mesh_entity
    ent = create_skinned_mesh_entity(mw._engine.scene, path, name, mesh_path)
    if ent is None:
        from core.components import Transform, MeshFilter, MeshRenderer
        ent = mw._engine.scene.create_entity(name)
        ent.add_component(Transform())
        mf = MeshFilter()
        mf.mesh_name = os.path.splitext(os.path.relpath(path, "."))[0].replace("\\", "/")
        mf.mesh_path = mesh_path
        ent.add_component(mf)
        ent.add_component(MeshRenderer())
    mw._hierarchy.refresh()
    on_entity_selected(mw, ent)


def open_scene_by_path(mw, path: str):
    if not os.path.exists(path):
        return
    _do_open_scene(mw, path)


def open_global_settings(mw):
    from core.config.config import get_global_config
    from editor.settings_dialog import SettingsDialog
    cfg = get_global_config()
    dlg = SettingsDialog("Global Settings", cfg, mw)
    dlg.config_changed.connect(lambda key, value: on_global_config_changed(mw, key, value))
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dlg.show()


def on_global_config_changed(mw, key: str, value):
    from core.config.config import get_global_config
    cfg = get_global_config()
    if key == "editor.ui_scale":
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QFont
        base_size = cfg.get("editor.font_size", 12)
        f = QFont()
        f.setPointSizeF(base_size * value / 100.0)
        QApplication.setFont(f)
        return
    mw._viewport.load_config(cfg)
    mw._viewport.camera.load_config(cfg)
    mw._viewport.gizmo.load_config(cfg)
    if mw._viewport.renderer:
        mw._viewport.renderer.load_config(cfg)
    mw._hierarchy.load_config(cfg)
    mw._inspector.load_config(cfg)
    mw._console.load_config(cfg)
    mw._profiler.load_config(cfg)
    mw._project.load_config(cfg)
    if hasattr(mw, '_terminal'):
        mw._terminal.load_config(cfg)


def open_project_settings(mw):
    from core.config.config import get_project_config
    from editor.settings_dialog import SettingsDialog
    path = getattr(mw._engine, "_project_path", ".")
    cfg = get_project_config(path)
    dlg = SettingsDialog("Project Settings", cfg, mw)
    dlg.config_changed.connect(lambda key, value: on_project_config_changed(mw, key, value))
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dlg.show()


def on_project_config_changed(mw, key: str, value):
    pass


def show_build_dialog(mw):
    from editor.build_dialog import show_build_dialog as _show_build_dialog
    _show_build_dialog(mw)


def show_about(mw):
    from editor.about_dialog import AboutDialog
    dlg = AboutDialog(mw)
    dlg.exec()
