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
from core.math.math3d import Mat4, Vec3

class VRPlugin(PluginBase):
    NAME = "VRPlugin"
    VERSION = "1.0.0"
    DESCRIPTION = "VR display integration with OpenXR"
    SYSTEM = False

    def __init__(self):
        super().__init__()
        self._viewport = None
        self._original_paintGL = None
        self._vr_active = False
        self._dock_widget = None

    def initialize(self, engine):
        super().initialize(engine)
        self.register_dock("VR Control", self._create_dock, area="bottom")

    def _create_dock(self):
        from plugins.vr_plugin.vr_dock import VRControlWidget
        self._dock_widget = VRControlWidget(self._engine, self)
        return self._dock_widget

    def toggle_vr(self):
        if self._vr_active:
            self._disable_vr()
        else:
            self._enable_vr()

    def _enable_vr(self):
        from plugins.vr_plugin import vr_core
        vp = self._engine.viewport
        if not vp or not vp._ctx:
            return
        self._viewport = vp
        if not vr_core._XR_AVAILABLE:
            return
        if not vr_core.initialize(vp._ctx):
            return
        self._vr_active = True
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
            self_._update_status_labels()
            try:
                self_._bind_screen_fbo()
                scene = eng.scene
                cam_cc = self_._clear_color + [1.0] if scene else self_._no_scene_color + [1.0]
                self_._screen_fbo.clear(*cam_cc[:3], 1.0)
                self_._renderer.clear_color = self_._clear_color
                if scene:
                    fw, fh = self_._get_physical_dims()
                    if plugin._render_vr_frame(self_, scene, fw, fh):
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
            return False
        if not vr_core.sync_hmd_pose():
            return False
        cam = vp._cam
        eyes = vr_core.get_eye_transforms(
            (cam.position.x, cam.position.y, cam.position.z),
            math.radians(cam.yaw),
            math.radians(cam.pitch),
        )
        for eye in eyes:
            efbo = rnd.eye_fbo(eye['eye_idx'])
            efbo.fbo.use()
            efbo.fbo.clear(*vp._clear_color, 1.0)
            efbo.fbo.viewport = (0, 0, efbo.w, efbo.h)
            proj_data = vr_core._make_proj_matrix(eye['fov_angles'], near=cam.near, far=cam.far)
            view_data = vr_core._make_view_matrix(eye['pos'], eye['fwd'], eye['right'], eye['up'])
            proj_mat = Mat4(np.array(proj_data, dtype=np.float64).reshape((4, 4), order='F'))
            view_mat = Mat4(np.array(view_data, dtype=np.float64).reshape((4, 4), order='F'))
            eye_pos = Vec3(eye['pos'][0], eye['pos'][1], eye['pos'][2])
            vp._renderer.render_scene(
                scene, view_mat, proj_mat, eye_pos,
                efbo.w, efbo.h, efbo.fbo,
                set(vp._selected_entities),
                cam.near, cam.far, cam.fov,
            )
        for eye in eyes:
            rnd.render_controllers_for_eye(eye)
        if vr_core.session_running() and len(vr_core._vr_state._swapchains) == 2:
            try:
                gl = vr_core._load_gl_funcs()
                for i, sc in enumerate(vr_core._vr_state._swapchains):
                    dst_tex = sc.acquire()
                    src_fbo_id = rnd.eye_fbo(i).fbo.glo
                    w, h = sc.w, sc.h
                    blit_fbo = (ctypes.c_uint * 1)(0)
                    gl.GenFramebuffers(1, blit_fbo)
                    gl.BindFramebuffer(0x8CA9, blit_fbo[0])
                    gl.FramebufferTexture2D(0x8CA9, 0x8CE0, 0x0DE1, dst_tex, 0)
                    gl.BindFramebuffer(0x8CA8, src_fbo_id)
                    gl.BlitFramebuffer(0, 0, w, h, 0, 0, w, h, 0x4000, 0x2600)
                    gl.BindFramebuffer(0x8D40, 0)
                    gl.DeleteFramebuffers(1, blit_fbo)
                    sc.release_image()
            except Exception as _blit_err:
                from core.foundation.logger import Logger
                Logger.error(f'[VR] Swapchain blit error: {_blit_err}')
        vp._bind_screen_fbo()
        vp._ctx.viewport = (0, 0, fw, fh)
        rnd.compose_to_screen(vp._screen_fbo, fw, fh)
        vr_core.end_xr_frame()
        vp._bind_screen_fbo()
        vp._ctx.viewport = (0, 0, fw, fh)
        vp._ctx.enable(moderngl.DEPTH_TEST)
        return True

    def _disable_vr(self):
        from plugins.vr_plugin import vr_core
        if self._original_paintGL and self._viewport:
            self._viewport.paintGL = self._original_paintGL
            self._viewport.update()
        r = vr_core.get_renderer()
        if r is not None:
            try:
                r.release()
            except Exception:
                pass
        vr_core.shutdown()
        self._vr_active = False
        from core.foundation.logger import Logger
        Logger.info("[VR] VR disabled, original viewport restored.")

    def shutdown(self):
        if self._vr_active:
            self._disable_vr()
        super().shutdown()


def get_plugin():
    return VRPlugin()
