# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import ctypes
import time
import types
import numpy as np
import moderngl

from core.foundation.plugin_manager import PluginBase
from plugins.vr_plugin.components import (
    XRRig, XRTrackedPoseDriver, XRController, XRHand, XRCull,
    XRHaptics, XRInteractionManager, XRRayInteractor, XRDirectInteractor, XRPokeInteractor,
    XRGrabInteractable, XRSmoothMoveProvider, XRSnapTurnProvider, XRTeleportationProvider,
    ARSession, ARCameraBackground, ARPlaneManager, ARRaycastManager, ARAnchorManager, ARPointCloudManager,
)
from core.maths.math3d import Mat4, Vec3

try:
    from core._vr_batch import build_eye_view_proj, build_controller_mvp
    _HAS_VR_BATCH = True
except Exception:
    _HAS_VR_BATCH = False

def spawn_xr_rig(engine):
    try:
        if engine is None or engine.scene is None:
            return None
        sc = engine.scene
        for e in sc.get_all_entities():
            if e.get_component(XRRig) is not None:
                return e
        from core.components.mesh_filter import MeshFilter, MeshRenderer
        from core.components.material import Material
        import os
        mat_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "materials", "vr_unlit.mat")

        rig = sc.create_entity("XRRig")
        rig.add_component(XRRig())

        def _add_model(parent, path, scale):
            me = sc.create_entity("XRModel")
            mf = MeshFilter()
            mf.mesh_path = path
            mf.mesh_name = "mesh"
            me.add_component(mf)
            mr = MeshRenderer()
            mr.enabled = True
            try:
                mr.material = Material.load(mat_path)
            except Exception:
                pass
            me.transform.local_scale = Vec3(scale, scale, scale)
            me.set_parent(parent, preserve_world=False)
            return me

        hmd = sc.create_entity("XRHMD")
        hmd.add_component(XRTrackedPoseDriver())
        hmd.get_component(XRTrackedPoseDriver).pose_type = "Center"
        hmd.add_component(XRCull())
        hmd.transform.local_position = Vec3(0.0, 1.6, 0.0)
        hmd.set_parent(rig, preserve_world=False)
        _add_model(hmd, "core/3d_models/OculusQ2/OculusQ2.fbx", 0.00035)

        def _hand(name, hand, model, pos):
            he = sc.create_entity(name)
            tpd = XRTrackedPoseDriver()
            tpd.pose_type = "LeftHand" if hand == "Left" else "RightHand"
            he.add_component(tpd)
            he.add_component(XRController(controller_hand=hand))
            he.get_component(XRController).model = model
            he.transform.local_position = Vec3(*pos)
            he.set_parent(rig, preserve_world=False)
            _add_model(he, "core/3d_models/OculusTouch/OculusTouch.fbx", 1.0)
            he.add_component(XRHand(hand=hand))

        _hand("XRControllerLeft", "Left", "OculusTouch", (-0.25, 1.2, -0.2))
        _hand("XRControllerRight", "Right", "OculusTouch", (0.25, 1.2, -0.2))

        rig.add_component(XRInteractionManager())
        rig.add_component(XRSmoothMoveProvider())
        rig.add_component(XRSnapTurnProvider())
        rig.add_component(XRTeleportationProvider())
        return rig
    except Exception:
        return None

class _EyeCamera:
    DEFAULT_FOV = 60.0
    def __init__(self, view_mat, proj_mat, pos, fwd, fov_deg):
        self._view = view_mat
        self._proj = proj_mat
        self._pos = pos
        self._fwd = fwd
        self._fov = fov_deg
        self._is_2d_mode = False
        self._is_orthographic = False
        self._ortho_zoom_distance = 1.0
    def get_view_matrix(self):
        return self._view
    def get_projection_matrix(self, aspect=1.0):
        return self._proj
    @property
    def forward(self):
        return self._fwd
    @property
    def position(self):
        return self._pos
    @property
    def fov(self):
        return self._fov
    @property
    def is_2d_mode(self):
        return False
    @property
    def is_orthographic(self):
        return False

class VRPlugin(PluginBase):
    NAME = "VRPlugin"
    VERSION = "1.0.0"
    DESCRIPTION = "VR display integration with OpenXR"
    SYSTEM = True
    def __init__(self):
        super().__init__()
        self._viewport = None
        self._original_paintGL = None
        self._vr_active = False
        self._dock_widget = None
        self._vr_view_btn = None
        self._eye_view = True
    def initialize(self, engine):
        super().initialize(engine)
        self.register_dock("VR Control", self._create_dock, area="bottom")
    def _create_dock(self):
        from plugins.vr_plugin.vr_dock import VRControlWidget
        self._dock_widget = VRControlWidget(self._engine, self)
        return self._dock_widget
    def ensure_vr_entities(self, scene=None):
        try:
            from core.engine.engine import Engine
            eng = Engine.instance() if hasattr(self, '_engine') else None
            sc = scene or (eng.scene if eng else None)
            if sc is None:
                return
            from plugins.vr_plugin import vr_core
            if not vr_core.is_active():
                return
            spawn_xr_rig(sc)
            sc._dirty = True
            sc._render_version += 1
        except Exception:
            pass

    def remove_vr_entities(self, scene=None):
        try:
            from core.engine.engine import Engine
            eng = Engine.instance() if hasattr(self, '_engine') else None
            sc = scene or (eng.scene if eng else None)
            if sc is None:
                return
            to_remove = []
            for e in sc.get_all_entities():
                if e.get_component(XRRig) is not None:
                    to_remove.append(e._id)
            for eid in to_remove:
                try:
                    sc.remove_entity(eid)
                except Exception:
                    pass
            sc._dirty = True
            sc._render_version += 1
        except Exception:
            pass

    def on_scene_loaded(self, scene):
        try:
            from plugins.vr_plugin import vr_core
            if vr_core.is_active():
                self.ensure_vr_entities(scene)
            else:
                self.remove_vr_entities(scene)
        except Exception:
            pass

    def on_viewport_ready(self, vp):
        try:
            if self._vr_active:
                self._install_toolbar_button(vp)
            try:
                from core.engine.engine import Engine
                eng = Engine.instance()
                if eng and eng.scene:
                    from plugins.vr_plugin import vr_core
                    if vr_core.is_active():
                        self.ensure_vr_entities(eng.scene)
                    else:
                        self.remove_vr_entities(eng.scene)
            except Exception:
                pass
        except Exception:
            pass

    def toggle_vr(self):
        if self._vr_active:
            self._disable_vr()
        else:
            self._enable_vr()

    def spawn_xr_rig(self, engine=None):
        from core.engine.engine import Engine
        eng = engine or Engine.instance()
        return spawn_xr_rig(eng)
    def _install_toolbar_button(self, vp):
        try:
            import qtawesome as qta
            from PyQt6.QtWidgets import QPushButton
            if self._vr_view_btn is not None:
                try:
                    self._vr_view_btn.setVisible(True)
                    return
                except Exception:
                    pass
            bar = getattr(vp, '_toolbar', None)
            if bar is None:
                return
            lay = bar.layout()
            if lay is None:
                return
            cam_row = lay.itemAt(0).layout() if lay.count() > 0 else None
            if cam_row is None:
                return
            btn = QPushButton(qta.icon("fa5s.eye", color="#d4d4d4"), " Eye")
            btn.setCheckable(False)
            btn.setToolTip("Toggle VR Eye / Desktop view")
            btn.setMinimumWidth(70)
            def _on_click():
                self._eye_view = not self._eye_view
                try:
                    from plugins.vr_plugin import vr_core
                    vr_core.set_eye_view(self._eye_view)
                except Exception:
                    pass
                if self._eye_view:
                    btn.setText(" Eye")
                    btn.setIcon(qta.icon("fa5s.eye", color="#4fc3f7"))
                else:
                    btn.setText(" Desktop")
                    btn.setIcon(qta.icon("fa5s.desktop", color="#aed581"))
                try:
                    vp.update()
                except Exception:
                    pass
            btn.clicked.connect(_on_click)
            if self._eye_view:
                btn.setIcon(qta.icon("fa5s.eye", color="#4fc3f7"))
            else:
                btn.setIcon(qta.icon("fa5s.desktop", color="#aed581"))
            cam_row.addWidget(btn)
            self._vr_view_btn = btn
            self._eye_view = True
            try:
                from plugins.vr_plugin import vr_core
                vr_core.set_eye_view(True)
            except Exception:
                pass
        except Exception:
            pass
    def _remove_toolbar_button(self):
        if self._vr_view_btn is not None:
            try:
                self._vr_view_btn.setVisible(False)
                self._vr_view_btn.deleteLater()
            except Exception:
                pass
            self._vr_view_btn = None
    def _enable_vr(self):
        from plugins.vr_plugin import vr_core
        vp = self._engine.viewport
        if not vp or not vp._ctx:
            return
        self._viewport = vp
        if not vr_core._XR_AVAILABLE:
            from core.foundation.logger import Logger
            Logger.warning("[VR] pyopenxr not available")
            return
        try:
            vp.makeCurrent()
        except Exception:
            pass
        if not vr_core.initialize(vp._ctx):
            from core.foundation.logger import Logger
            Logger.error("[VR] OpenXR initialize failed - HMD instance not created")
            return
        vr_core._vr_toggle._ctx = vp._ctx
        vr_core._vr_toggle.enabled = True
        self._vr_active = True
        self._eye_view = True
        try:
            vr_core.set_eye_view(True)
        except Exception:
            pass
        self._install_toolbar_button(vp)
        try:
            self.ensure_vr_entities()
        except Exception:
            pass
        self._original_paintGL = vp.paintGL
        plugin = self
        def _vr_paint(self_):
            if getattr(self_, '_in_render_tick', False):
                return
            self_._in_render_tick = True
            _p0 = time.perf_counter()
            _paint_gap = _p0 - getattr(self_, '_last_paint_enter', _p0)
            self_._last_paint_enter = _p0
            self_._paint_dt = _paint_gap
            eng = self_._engine
            prof = eng._profiler
            prof.capture_frame()
            prof.start("frame")
            now = _p0
            self_._fps_accum += now - self_._last_paint_time
            self_._last_paint_time = now
            self_._fps_frames += 1
            if self_._fps_accum >= 0.5:
                self_._fps = self_._fps_frames / self_._fps_accum
                self_._fps_accum = 0.0
                self_._fps_frames = 0
            if not self_._ctx or not self_._renderer:
                self_._in_render_tick = False
                return
            if not self_._in_update:
                dt = now - self_._last_frame_time
                self_._last_frame_time = now
                self_._last_dt = dt
                if self_._im:
                    self_._im.new_frame()
                if self_._focused and self_.isActiveWindow():
                    from editor.viewport.picking import pick_entity
                    from core.input.constants import (KEY_Q, KEY_W, KEY_E, KEY_R, KEY_F, KEY_DELETE)
                    from core.gizmo.gizmo import GizmoMode
                    if self_._im and self_._im.key_just_pressed(KEY_Q):
                        self_._gizmo.mode = GizmoMode.NONE
                    elif self_._im and self_._im.key_just_pressed(KEY_W):
                        self_._gizmo.mode = GizmoMode.TRANSLATE
                    elif self_._im and self_._im.key_just_pressed(KEY_E):
                        self_._gizmo.mode = GizmoMode.ROTATE
                    elif self_._im and self_._im.key_just_pressed(KEY_R):
                        self_._gizmo.mode = GizmoMode.SCALE
                    elif self_._im and self_._im.key_just_pressed(KEY_F):
                        if self_._selected_entities:
                            t = self_._selected_entities[0].transform
                            if t:
                                self_._cam.frame_bounds(t.position)
                    elif self_._im and self_._im.key_just_pressed(KEY_DELETE):
                        if self_._selected_entities and eng.scene:
                            from core.foundation.commands import DeleteEntityCommand, get_history
                            for ent in list(self_._selected_entities):
                                cmd = DeleteEntityCommand(eng.scene, ent.id)
                                get_history().execute(cmd)
                            self_._selected_entities.clear()
                            self_._set_gizmo_entity(None)
                            self_.entity_selected.emit(None)
                if not eng.play_mode:
                    self_._update_editor_particles(dt, self_._selected_entities)
                self_._cam.update(dt)
                try:
                    from plugins.vr_plugin import vr_core as _vc
                    _vc.update_vr_entities(self_, dt)
                except Exception:
                    pass
            self_._update_status_labels()
            try:
                self_._bind_screen_fbo()
                scene = eng.scene
                cam_cc = self_._clear_color + [1.0] if scene else self_._no_scene_color + [1.0]
                self_._screen_fbo.clear(*cam_cc[:3], 1.0)
                self_._renderer.clear_color = self_._clear_color
                if scene:
                    fw, fh = self_._get_physical_dims()
                    rendered = False
                    try:
                        rendered = plugin._render_vr_frame(self_, scene, fw, fh)
                    except Exception:
                        rendered = False
                    if rendered:
                        pass
                    else:
                        aspect = fw / max(1, fh)
                        view = self_._cam.get_view_matrix()
                        proj = self_._cam.get_projection_matrix(aspect)
                        cam_pos = self_._cam.position
                        self_._renderer.render_scene(scene, view, proj, cam_pos, fw, fh, self_._screen_fbo,
                                                     set(self_._selected_entities),
                                                     self_._cam.near, self_._cam.far, self_._cam.fov)
                        vp_mat = view * proj
                        dpr = self_.devicePixelRatio()
                        self_._renderer._line_width = max(1.0, float(dpr) * 1.0)
                        if self_._gizmo_visible:
                            with eng._scene_lock:
                                from editor.viewport.rendering import render_component_gizmos
                                render_component_gizmos(self_, vp_mat)
                        with eng._scene_lock:
                            from editor.viewport.rendering import render_selection_bounds
                            render_selection_bounds(self_, vp_mat, time.perf_counter(), self_._last_dt)
                        try:
                            with eng._scene_lock:
                                from editor.viewport.component_icons import render_component_icons_gl
                                render_component_icons_gl(self_)
                        except Exception:
                            pass
                        if self_._debug_lines:
                            self_._renderer.render_gizmo_lines(self_._debug_lines, vp_mat, cam_pos, fw, fh)
                            self_._debug_lines.clear()
                        if self_._show_bvh_debug:
                            self_._render_bvh_debug()
                        with eng._scene_lock:
                            from editor.viewport.navigation_gizmo import draw_axis_gizmo_api
                            draw_axis_gizmo_api(self_, vp_mat)
                            self_._render_api_gizmos()
                        if self_._pb_scale_gizmo and self_._pb_scale_gizmo.active:
                            self_._pb_scale_gizmo.render()
                        try:
                            with eng._scene_lock:
                                from editor.viewport.collaboration import render_remote_collaborator_gizmos
                                render_remote_collaborator_gizmos(self_, vp_mat, cam_pos, fw, fh)
                        except Exception:
                            pass
                        if self_._gizmo_visible:
                            gizmo_result = self_._gizmo.get_gizmo_arrays(self_._cam, fw, fh)
                            if gizmo_result is not None:
                                gs, ge, gcol = gizmo_result
                                self_._renderer.render_gizmo_arrays(gs, ge, gcol, vp_mat, fw, fh)
                            else:
                                gizmo_lines = self_._gizmo.get_gizmo_lines(self_._cam, fw, fh)
                                if gizmo_lines:
                                    self_._renderer.render_gizmo_lines(gizmo_lines, vp_mat, cam_pos, fw, fh)
                        try:
                            from plugins.vr_plugin import vr_core as _vc2
                            if _vc2.is_active():
                                _vc2.render_desktop_vr(self_, view, proj, fw, fh)
                                _vc2.handle_trigger_selection(self_)
                        except Exception:
                            pass
                        if not self_._no_qt_overlay:
                            if self_._overlay_widget.width() != self_.width() or self_._overlay_widget.height() != self_.height():
                                self_._overlay_widget.resize(self_.width(), self_.height())
                eng.set_profiler_data("paint_total_ms", (time.perf_counter() - _p0) * 1000.0)
            except Exception:
                import traceback
                traceback.print_exc()
            prof.stop("frame")
            if self_._vsync_enabled:
                self_.update()
            self_._in_render_tick = False
        vp.paintGL = types.MethodType(_vr_paint, vp)
        from core.foundation.logger import Logger
        Logger.info("[VR] Viewport hooked for VR rendering.")
    def _render_vr_frame(self, vp, scene, fw, fh) -> bool:
        from plugins.vr_plugin import vr_core
        if not self._vr_active:
            return False
        vr_core.poll_xr_events()
        if not vr_core.session_running():
            return False
        rnd = vr_core.get_renderer()
        if rnd is None:
            if vr_core._vr_toggle._ctx is None:
                vr_core._vr_toggle._ctx = vp._ctx
            try:
                rnd = vr_core.VRRenderer(vp._ctx)
                vr_core._vr_toggle.renderer = rnd
            except Exception:
                return False
        if not vr_core._vr_state._frame_begun:
            if not vr_core.had_frames():
                return False
            try:
                vp._bind_screen_fbo()
                vp._ctx.viewport = (0, 0, fw, fh)
                rnd.compose_to_screen(vp._screen_fbo, fw, fh)
                vp._bind_screen_fbo()
                vp._ctx.viewport = (0, 0, fw, fh)
                return True
            except Exception:
                return False
        cam = vp._cam
        _vr_shared = {}
        rig_pos = None
        rig_yaw_val = None
        try:
            sc = scene
            for e in sc.get_all_entities():
                vr = e.get_component(XRRig)
                if vr:
                    tr = e.transform
                    if tr:
                        wp = tr.position
                        rig_pos = (wp.x, wp.y, wp.z)
                        rig_yaw_val = vr.rig_yaw
                    break
        except Exception:
            pass
        class Params:
            cam_pos = rig_pos if rig_pos is not None else (cam.position.x, cam.position.y, cam.position.z)
            cam_yaw = rig_yaw_val if rig_yaw_val is not None else math.radians(cam.yaw)
            cam_pitch = math.radians(cam.pitch)
        def render_eye(fbo, w, h, eye):
            al, ar, au, ad = eye['fov_angles']
            if _HAS_VR_BATCH:
                view_arr, proj_arr = build_eye_view_proj(
                    float(eye['pos'][0]), float(eye['pos'][1]), float(eye['pos'][2]),
                    float(eye['right'][0]), float(eye['right'][1]), float(eye['right'][2]),
                    float(eye['up'][0]), float(eye['up'][1]), float(eye['up'][2]),
                    float(eye['fwd'][0]), float(eye['fwd'][1]), float(eye['fwd'][2]),
                    math.tan(float(al)), math.tan(float(ar)), math.tan(float(au)), math.tan(float(ad)),
                    float(cam.near), float(cam.far))
                proj_mat = Mat4(proj_arr)
                view_mat = Mat4(view_arr)
                eye_fwd = Vec3(eye['fwd'][0], eye['fwd'][1], eye['fwd'][2])
            else:
                tl, tr, tu, td = math.tan(al), math.tan(ar), math.tan(au), math.tan(ad)
                proj_mat = Mat4()
                proj_mat._d[0,0] = 2.0 / (tr - tl)
                proj_mat._d[1,1] = 2.0 / (tu - td)
                proj_mat._d[2,0] = (tr + tl) / (tr - tl)
                proj_mat._d[2,1] = (tu + td) / (tu - td)
                proj_mat._d[2,2] = -(cam.far + cam.near) / (cam.far - cam.near)
                proj_mat._d[2,3] = -1.0
                proj_mat._d[3,2] = -(2.0 * cam.far * cam.near) / (cam.far - cam.near)
                proj_mat._d[3,3] = 0.0
                eye_pos = Vec3(eye['pos'][0], eye['pos'][1], eye['pos'][2])
                eye_target = Vec3(eye_pos.x + eye['fwd'][0], eye_pos.y + eye['fwd'][1], eye_pos.z + eye['fwd'][2])
                eye_up = Vec3(eye['up'][0], eye['up'][1], eye['up'][2])
                eye_fwd = Vec3(eye['fwd'][0], eye['fwd'][1], eye['fwd'][2])
                view_mat = Mat4.look_at(eye_pos, eye_target, eye_up)
            hmd_entities = []
            try:
                from core.components.rendering.renderers.mesh_renderer import MeshRenderer
                for ent in scene.get_all_entities():
                    cull = ent.get_component(XRCull)
                    if cull is not None and cull.enabled and cull.hide_from_eye:
                        mrh = ent.get_component(MeshRenderer)
                        if mrh is not None and mrh.enabled:
                            hmd_entities.append((ent, mrh))
                            mrh.enabled = False
            except Exception:
                pass
            eye_pos = Vec3(eye['pos'][0], eye['pos'][1], eye['pos'][2])
            vp._renderer.render_scene(
                scene, view_mat, proj_mat, eye_pos,
                w, h, fbo,
                set(vp._selected_entities),
                cam.near, cam.far, cam.fov,
                shared_cache=_vr_shared,
            )
            try:
                for ent, mrh in hmd_entities:
                    mrh.enabled = True
            except Exception:
                pass
            fbo.use()
            vp._ctx.viewport = (0, 0, w, h)
            vp_mat = view_mat * proj_mat
            fov_deg = math.degrees(au - ad)
            eye_cam = _EyeCamera(view_mat, proj_mat, eye_pos, eye_fwd, fov_deg)
            dpr = vp.devicePixelRatio()
            vp._renderer._line_width = max(1.0, float(dpr) * 1.0)
            eng = vp._engine
            play_mode = eng.play_mode if eng else False
            with eng._scene_lock:
                if vp._gizmo_visible:
                    from editor.viewport.rendering import render_component_gizmos
                    render_component_gizmos(vp, vp_mat, w, h)
                from editor.viewport.rendering import render_selection_bounds
                render_selection_bounds(vp, vp_mat, time.perf_counter(), vp._last_dt, fw=w, fh=h, cam_pos=eye_pos)
                if not play_mode:
                    try:
                        from editor.viewport.component_icons import render_component_icons_gl
                        render_component_icons_gl(vp, vp_mat=vp_mat, pw=w, ph=h)
                    except Exception:
                        pass
                vp._render_api_gizmos(override_vp_mat=vp_mat, fw=w, fh=h)
                if not play_mode:
                    try:
                        from editor.viewport.collaboration import render_remote_collaborator_gizmos
                        render_remote_collaborator_gizmos(vp, vp_mat, eye_pos, w, h)
                    except Exception:
                        pass
            if vp._debug_lines:
                vp._renderer.render_gizmo_lines(vp._debug_lines, vp_mat, eye_pos, w, h)
                vp._debug_lines.clear()
            if vp._show_bvh_debug and not play_mode:
                vp._render_bvh_debug()
            if not play_mode:
                from editor.viewport.navigation_gizmo import draw_axis_gizmo_api
                draw_axis_gizmo_api(vp, vp_mat, fw=w, fh=h)
            if vp._pb_scale_gizmo and vp._pb_scale_gizmo.active and not play_mode:
                vp._pb_scale_gizmo.render()
            if vp._gizmo_visible:
                gizmo_result = vp._gizmo.get_gizmo_arrays(eye_cam, w, h)
                if gizmo_result is not None:
                    gs, ge, gcol = gizmo_result
                    vp._renderer.render_gizmo_arrays(gs, ge, gcol, vp_mat, w, h, thickness_multiplier=1.0)
                else:
                    gizmo_lines = vp._gizmo.get_gizmo_lines(eye_cam, w, h)
                    if gizmo_lines:
                        vp._renderer.render_gizmo_lines(gizmo_lines, vp_mat, eye_pos, w, h, thickness_multiplier=1.0)
            hits = vr_core.get_controller_hits()
            lines = []
            for idx2, (ent, hit, dist, ray) in enumerate(hits):
                if ray is None:
                    continue
                origin, fwd = ray
                has_hit = ent is not None and hit is not None
                end = hit if has_hit else (origin[0]+fwd[0]*8.0, origin[1]+fwd[1]*8.0, origin[2]+fwd[2]*8.0)
                from core.maths.math3d import Vec3 as _Vec3
                col = [1.0, 0.95, 0.2, 1.0] if has_hit else ([0.2, 0.85, 1.0, 1.0] if idx2==0 else [1.0, 0.35, 0.35, 1.0])
                lines.append((_Vec3(*origin), _Vec3(*end), col))
            if lines:
                try:
                    vp._renderer.render_gizmo_lines(lines, vp_mat, eye_pos, w, h, thickness_multiplier=2.0)
                except Exception:
                    pass
        try:
            vr_core.render_vr_frame(render_eye, vp._ctx, vp._screen_fbo, Params(), fw, fh)
        except Exception as e:
            from core.foundation.logger import Logger
            Logger.error(f'[VR] render_vr_frame error: {e}')
            import traceback
            traceback.print_exc()
            try:
                vr_core.end_xr_frame()
            except Exception:
                pass
            if not vr_core.had_frames():
                return False
            try:
                vp._bind_screen_fbo()
                vp._ctx.viewport = (0, 0, fw, fh)
                rnd.compose_to_screen(vp._screen_fbo, fw, fh)
                vp._bind_screen_fbo()
                vp._ctx.viewport = (0, 0, fw, fh)
                return True
            except Exception:
                return False
        try:
            vp._bind_screen_fbo()
        except Exception:
            try:
                vp._screen_fbo.use()
            except Exception:
                pass
        vp._ctx.viewport = (0, 0, fw, fh)
        vp._ctx.disable(moderngl.DEPTH_TEST)
        vp._ctx.disable(moderngl.CULL_FACE)
        return vr_core.is_eye_view()
    def _disable_vr(self):
        from plugins.vr_plugin import vr_core
        if self._original_paintGL and self._viewport:
            try:
                self._viewport.paintGL = self._original_paintGL
            except Exception:
                pass
            try:
                self._viewport.update()
            except Exception:
                pass
        self._remove_toolbar_button()
        try:
            self.remove_vr_entities()
        except Exception:
            pass
        r = vr_core.get_renderer()
        if r is not None:
            try:
                r.release()
            except Exception:
                pass
            vr_core._vr_toggle.renderer = None
        vr_core._vr_toggle.enabled = False
        vr_core._vr_toggle._ctx = None
        try:
            vr_core.set_eye_view(True)
        except Exception:
            pass
        vr_core.shutdown()
        self._vr_active = False
        self._eye_view = True
        self._original_paintGL = None
        self._viewport = None
        from core.foundation.logger import Logger
        Logger.info("[VR] VR disabled, original viewport restored.")
    def shutdown(self):
        if self._vr_active:
            self._disable_vr()
        super().shutdown()

def get_plugin():
    return VRPlugin()
