from __future__ import annotations
import math
import ctypes
import numpy as np
import moderngl
from pathlib import Path

_SHADER_DIR = Path(__file__).parent

try:
    import xr
    _XR_AVAILABLE = True
except ImportError:
    _XR_AVAILABLE = False

EYE_LEFT = 0
EYE_RIGHT = 1
IPD_DEFAULT = 0.063
FOV_VERT_DEFAULT = math.radians(90.0)
FOV_HORIZ_DEFAULT = math.radians(100.0)
EYE_TEX_W = 1440
EYE_TEX_H = 1600
GL_RGBA8 = 0x8058
GL_SRGB8_ALPHA8 = 0x8C43

GRIP_THRESHOLD = 0.5
IPD_SCALE = 0.5


def _quat_to_mat3(qx, qy, qz, qw):
    x2, y2, z2 = qx * 2, qy * 2, qz * 2
    xx, yy, zz = qx * x2, qy * y2, qz * z2
    xy, xz, yz = qx * y2, qx * z2, qy * z2
    wx, wy, wz = qw * x2, qw * y2, qw * z2
    return (1-(yy+zz), xy-wz, xz+wy, xy+wz, 1-(xx+zz), yz-wx, xz-wy, yz+wx, 1-(xx+yy))


def _fov_to_tangents(al, ar, au, ad):
    return (math.tan(al), math.tan(ar), math.tan(au), math.tan(ad))


def _mat3_mul(a, b):
    result = []
    for r in range(3):
        for c in range(3):
            val = sum(a[r*3+k] * b[k*3+c] for k in range(3))
            result.append(val)
    return tuple(result)


def _rot_y_mat3(a):
    c, s = math.cos(a), math.sin(a)
    return (c, 0, s, 0, 1, 0, -s, 0, c)


def _rot_x_mat3(a):
    c, s = math.cos(a), math.sin(a)
    return (1, 0, 0, 0, c, -s, 0, s, c)


def _normalize3(v):
    l = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if l < 1e-9:
        return (0.0, 0.0, 1.0)
    return (v[0]/l, v[1]/l, v[2]/l)


def _get_wgl_handles():
    try:
        opengl32 = ctypes.windll.opengl32
        opengl32.wglGetCurrentDC.restype = ctypes.c_void_p
        opengl32.wglGetCurrentContext.restype = ctypes.c_void_p
        hdc = opengl32.wglGetCurrentDC()
        hglrc = opengl32.wglGetCurrentContext()
        if not hdc or not hglrc:
            return None, None
        return hdc, hglrc
    except Exception as e:
        print(f'[VR] Failed to get WGL handles: {e}')
        return None, None


def _call_get_graphics_requirements(instance, system_id):
    pfn = ctypes.cast(
        xr.get_instance_proc_addr(instance, 'xrGetOpenGLGraphicsRequirementsKHR'),
        xr.PFN_xrGetOpenGLGraphicsRequirementsKHR
    )
    gl_reqs = xr.GraphicsRequirementsOpenGLKHR()
    result = pfn(instance, system_id, ctypes.byref(gl_reqs))
    xr.check_result(xr.Result(result))


def _make_binding(hdc, hglrc):
    binding = xr.GraphicsBindingOpenGLWin32KHR()
    fields = [f[0] for f in binding._fields_]
    dc_field = next((f for f in fields if f.lower().replace('_', '') == 'hdc'), None)
    gl_field = next((f for f in fields if f.lower().replace('_', '') == 'hglrc'), None)
    if dc_field is None or gl_field is None:
        raise RuntimeError(f'Unknown GraphicsBindingOpenGLWin32KHR fields: {fields}')
    setattr(binding, dc_field, hdc)
    setattr(binding, gl_field, hglrc)
    return binding


class VREyeSwapchain:
    def __init__(self, session, w: int, h: int):
        self.w, self.h = w, h
        formats = xr.enumerate_swapchain_formats(session)
        chosen = GL_RGBA8 if GL_RGBA8 in formats else (GL_SRGB8_ALPHA8 if GL_SRGB8_ALPHA8 in formats else formats[0])
        sc_info = xr.SwapchainCreateInfo(
            usage_flags=xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT | xr.SwapchainUsageFlags.SAMPLED_BIT,
            format=chosen,
            sample_count=1,
            width=w,
            height=h,
            face_count=1,
            array_size=1,
            mip_count=1,
        )
        self.swapchain = xr.create_swapchain(session, sc_info)
        self.images = xr.enumerate_swapchain_images(self.swapchain, xr.SwapchainImageOpenGLKHR)
        self.format = chosen

    def acquire(self):
        idx = xr.acquire_swapchain_image(self.swapchain, xr.SwapchainImageAcquireInfo())
        xr.wait_swapchain_image(self.swapchain, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))
        return int(self.images[idx].image)

    def release_image(self):
        xr.release_swapchain_image(self.swapchain, xr.SwapchainImageReleaseInfo())

    def destroy(self):
        xr.destroy_swapchain(self.swapchain)


class VREyeFramebuffer:
    def __init__(self, ctx: moderngl.Context, w: int, h: int):
        self.w, self.h = w, h
        self.tex = ctx.texture((w, h), 4)
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth = ctx.depth_renderbuffer((w, h))
        self.fbo = ctx.framebuffer(color_attachments=[self.tex], depth_attachment=self.depth)

    def release(self):
        self.fbo.release()
        self.tex.release()
        self.depth.release()


class ControllerState:
    def __init__(self):
        self.pos = (0.0, 0.0, 0.0)
        self.quat = (0.0, 0.0, 0.0, 1.0)
        self.grip = 0.0
        self.valid = False


class VRState:
    IPD = IPD_DEFAULT
    EYE_TEX_W = EYE_TEX_W
    EYE_TEX_H = EYE_TEX_H

    def __init__(self):
        self.active = False
        self.session = None
        self.instance = None
        self.system_id = None
        self.space = None
        self._hmd_quat = (0.0, 0.0, 0.0, 1.0)
        self._hmd_pos = (0.0, 0.0, 0.0)
        self._eye_fovs = [
            (-FOV_HORIZ_DEFAULT*0.5, FOV_HORIZ_DEFAULT*0.5, FOV_VERT_DEFAULT*0.5, -FOV_VERT_DEFAULT*0.5),
            (-FOV_HORIZ_DEFAULT*0.5, FOV_HORIZ_DEFAULT*0.5, FOV_VERT_DEFAULT*0.5, -FOV_VERT_DEFAULT*0.5),
        ]
        self._eye_offsets = [(-self.IPD*0.5, 0.0, 0.0), (self.IPD*0.5, 0.0, 0.0)]
        self._eye_poses = [None, None]
        self._eye_positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        self._eye_quats = [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)]
        self._session_running = False
        self._frame_state = None
        self._frame_begun = False
        self._frame_discard = False
        self._swapchains: list = []
        self._display_time = 0
        self._binding = None
        self._hmd_pos_offset = (0.0, 0.0, 0.0)
        self._hmd_pos_origin = None
        self.controllers = [ControllerState(), ControllerState()]
        self._action_set = None
        self._grip_actions = [None, None]
        self._pose_actions = [None, None]
        self._pose_spaces = [None, None]
        self._ipd_pinch_active = False
        self._ipd_pinch_dist0 = 0.0
        self._ipd_pinch_ipd0 = IPD_DEFAULT
        self.ipd_override = IPD_DEFAULT
        self._xr_origin = (0.0, 0.0, 0.0)
        self._xr_mid = (0.0, 0.0, 0.0)
        self._xr_scale = 1.0
        self._frames_rendered = 0
        self._ever_running = False


_gl = None


def _load_gl_funcs():
    global _gl
    if _gl is not None:
        return _gl
    lib = ctypes.windll.opengl32
    def _get(name, restype, *argtypes):
        ptr = lib.wglGetProcAddress(name.encode())
        if not ptr:
            raise RuntimeError(f'wglGetProcAddress({name}) returned NULL')
        return ctypes.WINFUNCTYPE(restype, *argtypes)(ptr)
    _gl = type('GL', (), {
        'GenFramebuffers':      _get('glGenFramebuffers',      None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
        'DeleteFramebuffers':   _get('glDeleteFramebuffers',   None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
        'BindFramebuffer':      _get('glBindFramebuffer',      None, ctypes.c_uint, ctypes.c_uint),
        'FramebufferTexture2D': _get('glFramebufferTexture2D', None, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_int),
        'BlitFramebuffer':      _get('glBlitFramebuffer',      None, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint),
        'CheckFramebufferStatus': _get('glCheckFramebufferStatus', ctypes.c_uint, ctypes.c_uint),
        'DrawBuffer':           ctypes.WINFUNCTYPE(None, ctypes.c_uint)(ctypes.cast(lib.glDrawBuffer, ctypes.c_void_p).value),
        'ReadBuffer':           ctypes.WINFUNCTYPE(None, ctypes.c_uint)(ctypes.cast(lib.glReadBuffer, ctypes.c_void_p).value),
    })()
    return _gl


GL_DRAW_FRAMEBUFFER = 0x8CA9
GL_READ_FRAMEBUFFER = 0x8CA8
GL_FRAMEBUFFER = 0x8D40
GL_COLOR_ATTACHMENT0 = 0x8CE0
GL_TEXTURE_2D = 0x0DE1
GL_COLOR_BUFFER_BIT = 0x4000
GL_NEAREST = 0x2600
GL_FRAMEBUFFER_COMPLETE = 0x8CD5

_BLIT_FBO_IDS = None


def _get_blit_fbo_id(gl, idx):
    global _BLIT_FBO_IDS
    if _BLIT_FBO_IDS is None:
        arr = (ctypes.c_uint * 2)(0, 0)
        gl.GenFramebuffers(2, arr)
        _BLIT_FBO_IDS = arr
    return int(_BLIT_FBO_IDS[idx])


def _free_blit_fbos():
    global _BLIT_FBO_IDS
    if _BLIT_FBO_IDS is not None:
        try:
            gl = _load_gl_funcs()
            gl.DeleteFramebuffers(2, _BLIT_FBO_IDS)
        except Exception:
            pass
        _BLIT_FBO_IDS = None


_vr_state = VRState()


def is_available() -> bool:
    return _XR_AVAILABLE


def is_active() -> bool:
    return _vr_state.active


def _setup_input(instance, session):
    try:
        action_set_info = xr.ActionSetCreateInfo(
            action_set_name='zarkifs_actions',
            localized_action_set_name='ZarKIFS Actions',
            priority=0,
        )
        action_set = xr.create_action_set(instance, action_set_info)
        _vr_state._action_set = action_set

        hand_paths = [
            xr.string_to_path(instance, '/user/hand/left'),
            xr.string_to_path(instance, '/user/hand/right'),
        ]
        _vr_state._hand_paths = hand_paths

        for i, side in enumerate(['left', 'right']):
            grip_info = xr.ActionCreateInfo(
                action_type=xr.ActionType.FLOAT_INPUT,
                action_name=f'grip_{side}',
                localized_action_name=f'Grip {side.capitalize()}',
                count_subaction_paths=1,
                subaction_paths=ctypes.cast(
                    (xr.Path * 1)(hand_paths[i]), ctypes.POINTER(xr.Path)
                ),
            )
            _vr_state._grip_actions[i] = xr.create_action(action_set, grip_info)

            pose_info = xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name=f'hand_pose_{side}',
                localized_action_name=f'Hand Pose {side.capitalize()}',
                count_subaction_paths=1,
                subaction_paths=ctypes.cast(
                    (xr.Path * 1)(hand_paths[i]), ctypes.POINTER(xr.Path)
                ),
            )
            _vr_state._pose_actions[i] = xr.create_action(action_set, pose_info)

        grip_path_l = xr.string_to_path(instance, '/user/hand/left/input/squeeze/value')
        grip_path_r = xr.string_to_path(instance, '/user/hand/right/input/squeeze/value')
        pose_path_l = xr.string_to_path(instance, '/user/hand/left/input/grip/pose')
        pose_path_r = xr.string_to_path(instance, '/user/hand/right/input/grip/pose')

        profile_path = xr.string_to_path(instance, '/interaction_profiles/khr/simple_controller')
        bindings_simple = [
            xr.ActionSuggestedBinding(action=_vr_state._pose_actions[0], binding=pose_path_l),
            xr.ActionSuggestedBinding(action=_vr_state._pose_actions[1], binding=pose_path_r),
        ]
        try:
            xr.suggest_interaction_profile_bindings(instance, xr.InteractionProfileSuggestedBinding(
                interaction_profile=profile_path,
                count_suggested_bindings=len(bindings_simple),
                suggested_bindings=bindings_simple,
            ))
        except Exception:
            pass

        for profile_str, grip_l, grip_r in [
            ('/interaction_profiles/oculus/touch_controller',
             '/user/hand/left/input/squeeze/value',
             '/user/hand/right/input/squeeze/value'),
            ('/interaction_profiles/valve/index_controller',
             '/user/hand/left/input/squeeze/value',
             '/user/hand/right/input/squeeze/value'),
            ('/interaction_profiles/htc/vive_controller',
             '/user/hand/left/input/squeeze/click',
             '/user/hand/right/input/squeeze/click'),
            ('/interaction_profiles/microsoft/motion_controller',
             '/user/hand/left/input/squeeze/click',
             '/user/hand/right/input/squeeze/click'),
        ]:
            try:
                prof_path = xr.string_to_path(instance, profile_str)
                gl = xr.string_to_path(instance, grip_l)
                gr = xr.string_to_path(instance, grip_r)
                pl = xr.string_to_path(instance, '/user/hand/left/input/grip/pose')
                pr = xr.string_to_path(instance, '/user/hand/right/input/grip/pose')
                bindings = [
                    xr.ActionSuggestedBinding(action=_vr_state._grip_actions[0], binding=gl),
                    xr.ActionSuggestedBinding(action=_vr_state._grip_actions[1], binding=gr),
                    xr.ActionSuggestedBinding(action=_vr_state._pose_actions[0], binding=pl),
                    xr.ActionSuggestedBinding(action=_vr_state._pose_actions[1], binding=pr),
                ]
                xr.suggest_interaction_profile_bindings(instance, xr.InteractionProfileSuggestedBinding(
                    interaction_profile=prof_path,
                    count_suggested_bindings=len(bindings),
                    suggested_bindings=bindings,
                ))
            except Exception:
                pass

        attach_info = xr.SessionActionSetsAttachInfo(
            count_action_sets=1,
            action_sets=ctypes.cast((xr.ActionSet * 1)(action_set), ctypes.POINTER(xr.ActionSet)),
        )
        xr.attach_session_action_sets(session, attach_info)

        for i, side in enumerate(['left', 'right']):
            hand_path = xr.string_to_path(instance, f'/user/hand/{side}')
            space_info = xr.ActionSpaceCreateInfo(
                action=_vr_state._pose_actions[i],
                subaction_path=hand_path,
                pose_in_action_space=xr.Posef(),
            )
            _vr_state._pose_spaces[i] = xr.create_action_space(session, space_info)

        print('[VR] Input actions set up.')
    except Exception as e:
        print(f'[VR] Input setup failed: {e}')


def initialize(ctx: moderngl.Context) -> bool:
    global _vr_state
    if not _XR_AVAILABLE:
        print('[VR] pyopenxr not installed.')
        return False
    try:
        hdc, hglrc = _get_wgl_handles()
        if hdc is None or hglrc is None:
            print('[VR] Failed to retrieve WGL context handles.')
            return False
        exts = xr.enumerate_instance_extension_properties()
        avail = {
            e.extension_name.decode('utf-8') if isinstance(e.extension_name, bytes) else e.extension_name
            for e in exts
        }
        if "XR_KHR_opengl_enable" not in avail:
            print('[VR] OpenGL backend unsupported by runtime.')
            return False
        app_info = xr.ApplicationInfo(
            application_name='ZarKIFS_VR',
            application_version=1,
            engine_name='ZarKIFS',
            engine_version=1,
            api_version=xr.Version(1, 0, 0)
        )
        create_info = xr.InstanceCreateInfo(
            application_info=app_info,
            enabled_extension_names=["XR_KHR_opengl_enable"]
        )
        _vr_state.instance = xr.create_instance(create_info)
        sys_info = xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY)
        _vr_state.system_id = xr.get_system(_vr_state.instance, sys_info)
        _call_get_graphics_requirements(_vr_state.instance, _vr_state.system_id)
        binding = _make_binding(hdc, hglrc)
        session_info = xr.SessionCreateInfo(
            system_id=_vr_state.system_id,
            create_flags=xr.SessionCreateFlags(0),
        )
        session_info.next = ctypes.cast(ctypes.pointer(binding), ctypes.c_void_p)
        _vr_state._binding = binding
        _vr_state.session = xr.create_session(_vr_state.instance, session_info)
        space_info = xr.ReferenceSpaceCreateInfo(
            reference_space_type=xr.ReferenceSpaceType.LOCAL,
            pose_in_reference_space=xr.Posef()
        )
        _vr_state.space = xr.create_reference_space(_vr_state.session, space_info)
        _setup_input(_vr_state.instance, _vr_state.session)
        _vr_state.active = True
        print('[VR] OpenXR session and space created successfully.')
        return True
    except Exception as e:
        print(f'[VR] OpenXR init failed: {e}')
        _vr_state.active = False
        return False


def _create_swapchains():
    if _vr_state._swapchains:
        return
    try:
        views = xr.enumerate_view_configuration_views(
            _vr_state.instance,
            _vr_state.system_id,
            xr.ViewConfigurationType.PRIMARY_STEREO
        )
        for i in range(2):
            w = views[i].recommended_image_rect_width if i < len(views) else EYE_TEX_W
            h = views[i].recommended_image_rect_height if i < len(views) else EYE_TEX_H
            VRState.EYE_TEX_W = w
            VRState.EYE_TEX_H = h
            sc = VREyeSwapchain(_vr_state.session, w, h)
            _vr_state._swapchains.append(sc)
        print(f'[VR] Swapchains created {VRState.EYE_TEX_W}x{VRState.EYE_TEX_H}')
    except Exception as e:
        print(f'[VR] Swapchain creation failed: {e}')


def shutdown():
    global _vr_state
    try:
        for space in _vr_state._pose_spaces:
            if space is not None:
                try:
                    xr.destroy_space(space)
                except Exception:
                    pass
        _vr_state._pose_spaces = [None, None]
        if _vr_state._action_set is not None:
            try:
                xr.destroy_action_set(_vr_state._action_set)
            except Exception:
                pass
            _vr_state._action_set = None
        for sc in _vr_state._swapchains:
            try:
                sc.destroy()
            except Exception:
                pass
        _vr_state._swapchains.clear()
        if _vr_state._session_running and _vr_state.session:
            try:
                xr.end_session(_vr_state.session)
            except Exception:
                pass
            _vr_state._session_running = False
        if _vr_state.session:
            xr.destroy_session(_vr_state.session)
        if _vr_state.instance:
            xr.destroy_instance(_vr_state.instance)
    except Exception:
        pass
    _vr_state.active = False
    _vr_state.session = None
    _vr_state.instance = None
    _vr_state.space = None
    _vr_state.system_id = None
    _vr_state._session_running = False
    _free_blit_fbos()
    _vr_state._xr_layer = None
    _vr_state._frame_begun = False
    _vr_state._frame_discard = False
    _vr_state._frame_state = None
    for p_i in range(2):
        _vr_state._eye_poses[p_i] = None
    print('[VR] OpenXR resources destroyed.')


def _sync_controller_input():
    if not (_vr_state.active and _vr_state._session_running and _vr_state._action_set is not None):
        return
    hand_paths = getattr(_vr_state, '_hand_paths', None)
    if hand_paths is None:
        hand_paths = [
            xr.string_to_path(_vr_state.instance, '/user/hand/left'),
            xr.string_to_path(_vr_state.instance, '/user/hand/right'),
        ]
    try:
        active_sets = (xr.ActiveActionSet * 1)(
            xr.ActiveActionSet(action_set=_vr_state._action_set, subaction_path=xr.NULL_PATH)
        )
        xr.sync_actions(
            _vr_state.session,
            xr.ActionsSyncInfo(
                count_active_action_sets=1,
                active_action_sets=ctypes.cast(active_sets, ctypes.POINTER(xr.ActiveActionSet)),
            )
        )
    except xr.exception.SessionNotFocused:
        pass
    except Exception as e:
        if not getattr(_vr_state, '_sync_actions_warn', False):
            _vr_state._sync_actions_warn = True
            print(f'[VR] sync_actions exception: {e}')
    for i in range(2):
        ctrl = _vr_state.controllers[i]
        ctrl.valid = False
        if _vr_state._grip_actions[i] is not None:
            try:
                state = xr.get_action_state_float(
                    _vr_state.session,
                    xr.ActionStateGetInfo(
                        action=_vr_state._grip_actions[i],
                        subaction_path=hand_paths[i],
                    ),
                )
                ctrl.grip = float(state.current_state) if state.is_active else 0.0
            except Exception:
                ctrl.grip = 0.0
        if _vr_state._pose_spaces[i] is not None:
            try:
                loc = xr.locate_space(
                    _vr_state._pose_spaces[i],
                    _vr_state.space,
                    _vr_state._display_time,
                )
                flags = loc.location_flags
                pos_valid = bool(flags & xr.SpaceLocationFlags.POSITION_VALID_BIT.value)
                ori_valid = bool(flags & xr.SpaceLocationFlags.ORIENTATION_VALID_BIT.value)
                if not (pos_valid and ori_valid) and not getattr(_vr_state, f'_locate_warn_{i}', False):
                    setattr(_vr_state, f'_locate_warn_{i}', True)
                    print(f'[VR] Controller {i} locate_space flags={flags} display_time={_vr_state._display_time} pos_valid={pos_valid} ori_valid={ori_valid}')
                if pos_valid and ori_valid:
                    setattr(_vr_state, f'_locate_warn_{i}', False)
                    p = loc.pose.position
                    o = loc.pose.orientation
                    ctrl.pos = (p.x, p.y, p.z)
                    ctrl.quat = (o.x, o.y, o.z, o.w)
                    ctrl.valid = True
            except Exception as e:
                if not getattr(_vr_state, f'_locate_exc_{i}', False):
                    setattr(_vr_state, f'_locate_exc_{i}', True)
                    print(f'[VR] Controller {i} locate_space exception: {e}')
    _update_ipd_pinch()


def _update_ipd_pinch():
    c0, c1 = _vr_state.controllers[0], _vr_state.controllers[1]
    both_gripped = c0.grip > GRIP_THRESHOLD and c1.grip > GRIP_THRESHOLD
    both_valid = c0.valid and c1.valid

    if both_gripped and both_valid:
        p0, p1 = c0.pos, c1.pos
        dist = math.sqrt(sum((p0[k]-p1[k])**2 for k in range(3)))
        if not _vr_state._ipd_pinch_active:
            _vr_state._ipd_pinch_active = True
            _vr_state._ipd_pinch_dist0 = dist
            _vr_state._ipd_pinch_ipd0 = _vr_state.ipd_override
        else:
            delta = dist - _vr_state._ipd_pinch_dist0
            new_ipd = _vr_state._ipd_pinch_ipd0 + delta * IPD_SCALE
            _vr_state.ipd_override = max(0.010, min(0.200, new_ipd))
    else:
        _vr_state._ipd_pinch_active = False


def get_eye_transforms(*args, **kwargs) -> list[dict]:
    if len(args) == 1 and hasattr(args[0], 'cam_pos'):
        params = args[0]
        cam_pos = params.cam_pos
        cam_yaw = getattr(params, 'cam_yaw', 0.0)
        cam_pitch = getattr(params, 'cam_pitch', 0.0)
    elif len(args) >= 3:
        cam_pos = args[0]
        cam_yaw = args[1]
        cam_pitch = args[2]
        if isinstance(cam_pos, (list, tuple)) and len(cam_pos) == 3:
            cam_pos = tuple(cam_pos)
        else:
            try:
                cam_pos = (cam_pos.x, cam_pos.y, cam_pos.z)
            except Exception:
                cam_pos = tuple(cam_pos)
    elif 'cam_pos' in kwargs:
        cam_pos = kwargs['cam_pos']
        cam_yaw = kwargs.get('cam_yaw', 0.0)
        cam_pitch = kwargs.get('cam_pitch', 0.0)
    else:
        cam_pos = (0.0, 0.0, 0.0)
        cam_yaw = 0.0
        cam_pitch = 0.0
    eyes = []
    p0 = _vr_state._eye_positions[0]
    p1 = _vr_state._eye_positions[1]
    xr_ipd = math.sqrt(sum((p1[k]-p0[k])**2 for k in range(3)))
    use_xr_positions = 0.01 < xr_ipd < 0.12
    if use_xr_positions:
        mid = tuple((p0[k] + p1[k]) * 0.5 for k in range(3))

    hx, hy, hz = _vr_state._hmd_pos_offset
    base_pos = (
        cam_pos[0] + hx,
        cam_pos[1] + hy,
        cam_pos[2] + hz,
    )
    _vr_state._xr_origin = base_pos
    if use_xr_positions:
        xr_ipd_safe = max(xr_ipd, 1e-4)
        _vr_state._xr_mid = mid
        _vr_state._xr_scale = _vr_state.ipd_override / xr_ipd_safe
    else:
        _vr_state._xr_mid = (0.0, 0.0, 0.0)
        _vr_state._xr_scale = 1.0

    for i in range(2):
        if _vr_state.active:
            qx, qy, qz, qw = _vr_state._eye_quats[i]
            rot = _quat_to_mat3(qx, qy, qz, qw)
            r = ( rot[0],  rot[3],  rot[6])
            u = ( rot[1],  rot[4],  rot[7])
            f = (-rot[2], -rot[5], -rot[8])
            if use_xr_positions:
                exi, eyi, ezi = _vr_state._eye_positions[i]
                dx = exi - mid[0]
                dy = eyi - mid[1]
                dz = ezi - mid[2]
                ipd_scale = _vr_state.ipd_override / max(xr_ipd, 1e-4)
                ep = (
                    base_pos[0] + dx * ipd_scale,
                    base_pos[1] + dy * ipd_scale,
                    base_pos[2] + dz * ipd_scale,
                )
            else:
                sign = -0.5 if i == 0 else 0.5
                half_ipd = _vr_state.ipd_override * sign
                ep = (
                    base_pos[0] + r[0]*half_ipd,
                    base_pos[1] + r[1]*half_ipd,
                    base_pos[2] + r[2]*half_ipd,
                )
        else:
            yaw, pitch = cam_yaw, cam_pitch
            bf = (math.cos(pitch)*math.sin(yaw), math.sin(pitch), -math.cos(pitch)*math.cos(yaw))
            brx, brz = -bf[2], bf[0]
            rlen = math.sqrt(brx**2 + brz**2) or 1.0
            br = (brx/rlen, 0.0, brz/rlen)
            bu = (br[1]*bf[2]-br[2]*bf[1], br[2]*bf[0]-br[0]*bf[2], br[0]*bf[1]-br[1]*bf[0])
            ulen = math.sqrt(sum(v*v for v in bu)) or 1.0
            bu = tuple(v/ulen for v in bu)
            sign = -0.5 if i == 0 else 0.5
            half_ipd = _vr_state.ipd_override * sign
            ep = (
                base_pos[0] + br[0]*half_ipd,
                base_pos[1] + br[1]*half_ipd,
                base_pos[2] + br[2]*half_ipd,
            )
            rot = _mat3_mul(_rot_y_mat3(yaw), _rot_x_mat3(pitch))
            r = (rot[0], rot[3], rot[6])
            u = (rot[1], rot[4], rot[7])
            f = (rot[2], rot[5], rot[8])
        eyes.append({
            'pos': ep,
            'fwd':   f,
            'right': r,
            'up':    u,
            'fov_angles': _vr_state._eye_fovs[i],
            'eye_idx': i,
        })
    return eyes


def _xr_to_world(xr_pos: tuple) -> tuple:
    ox, oy, oz = _vr_state._xr_origin
    mx, my, mz = _vr_state._xr_mid
    s = _vr_state._xr_scale
    return (
        ox + (xr_pos[0] - mx) * s,
        oy + (xr_pos[1] - my) * s,
        oz + (xr_pos[2] - mz) * s,
    )


def set_uniforms_for_eye(set_fn, eye: dict, rw: int, rh: int):
    al, ar, au, ad = eye['fov_angles']
    tl = math.tan(al)
    tr = math.tan(ar)
    tu = math.tan(au)
    td = math.tan(ad)
    cx = (tl + tr) * 0.5
    cy = (tu + td) * 0.5
    hw = (tr - tl) * 0.5
    hh = (tu - td) * 0.5
    r, u, f = eye['right'], eye['up'], eye['fwd']
    fn = f
    flen = math.sqrt(fn[0]*fn[0] + fn[1]*fn[1] + fn[2]*fn[2])
    if flen > 1e-9:
        fn = (fn[0]/flen, fn[1]/flen, fn[2]/flen)
    fa = (fn[0] + cx*r[0] + cy*u[0],
          fn[1] + cx*r[1] + cy*u[1],
          fn[2] + cx*r[2] + cy*u[2])
    aspect = rw / rh if rh else 1.0
    ra = (r[0]*hw/aspect, r[1]*hw/aspect, r[2]*hw/aspect)
    ua = (u[0]*hh, u[1]*hh, u[2]*hh)
    set_fn('u_cam_pos',   eye['pos'])
    set_fn('u_cam_fwd',   fa)
    set_fn('u_cam_right', ra)
    set_fn('u_cam_up',    ua)
    set_fn('u_resolution', (float(rw), float(rh)))
    set_fn('u_fov', 1.0)


_CTRL_VERT = open(_SHADER_DIR / "shaders/vr/vr_controller_vert.glsl", "r", encoding="utf-8").read()

_CTRL_FRAG = open(_SHADER_DIR / "shaders/vr/vr_controller_frag.glsl", "r", encoding="utf-8").read()


def _make_proj_matrix(fov_angles, near=0.01, far=100.0):
    al, ar, au, ad = fov_angles
    tl, tr, tu, td = math.tan(al), math.tan(ar), math.tan(au), math.tan(ad)
    r_m_l = tr - tl
    t_m_b = tu - td
    r_p_l = tr + tl
    t_p_b = tu + td
    m = [0.0]*16
    m[0]  =  2.0 / r_m_l
    m[5]  =  2.0 / t_m_b
    m[8]  =  r_p_l / r_m_l
    m[9]  =  t_p_b / t_m_b
    m[10] = -(far + near) / (far - near)
    m[11] = -1.0
    m[14] = -(2.0 * far * near) / (far - near)
    return m


def _make_view_matrix(eye_pos, fwd, right, up):
    rx, ry, rz = right
    ux, uy, uz = up
    fx, fy, fz = fwd
    ex, ey, ez = eye_pos
    tx = -(rx*ex + ry*ey + rz*ez)
    ty = -(ux*ex + uy*ey + uz*ez)
    tz =  (fx*ex + fy*ey + fz*ez)
    return [
        rx,  ry,  rz, 0.0,
        ux,  uy,  uz, 0.0,
       -fx, -fy, -fz, 0.0,
        tx,  ty,  tz, 1.0,
    ]


def _mat4_mul(a, b):
    result = [0.0]*16
    for row in range(4):
        for col in range(4):
            s = 0.0
            for k in range(4):
                s += a[row + k*4] * b[k + col*4]
            result[row + col*4] = s
    return result


def _make_model_matrix(pos, quat, scale=1.0):
    rot = _quat_to_mat3(*quat)
    s = scale
    return [
        rot[0]*s, rot[3]*s, rot[6]*s, 0.0,
        rot[1]*s, rot[4]*s, rot[7]*s, 0.0,
        rot[2]*s, rot[5]*s, rot[8]*s, 0.0,
        pos[0],   pos[1],   pos[2],   1.0,
    ]


def _build_controller_mesh():
    verts = []
    indices = []

    def _add_box(cx, cy, cz, sx, sy, sz, nx_faces=True):
        hx, hy, hz = sx*0.5, sy*0.5, sz*0.5
        corners = [
            (-hx+cx, -hy+cy, -hz+cz),
            ( hx+cx, -hy+cy, -hz+cz),
            ( hx+cx,  hy+cy, -hz+cz),
            (-hx+cx,  hy+cy, -hz+cz),
            (-hx+cx, -hy+cy,  hz+cz),
            ( hx+cx, -hy+cy,  hz+cz),
            ( hx+cx,  hy+cy,  hz+cz),
            (-hx+cx,  hy+cy,  hz+cz),
        ]
        faces = [
            (0,1,2,3, ( 0, 0,-1)),
            (5,4,7,6, ( 0, 0, 1)),
            (1,5,6,2, ( 1, 0, 0)),
            (4,0,3,7, (-1, 0, 0)),
            (3,2,6,7, ( 0, 1, 0)),
            (0,4,5,1, ( 0,-1, 0)),
        ]
        for fi, (a,b,c,d, n) in enumerate(faces):
            base = len(verts) // 6
            for vi in (a,b,c,d):
                verts.extend(corners[vi])
                verts.extend(n)
            indices.extend([base, base+1, base+2, base, base+2, base+3])

    def _add_capsule(cx, cy, cz, radius, length, segments=8):
        half = length * 0.5
        for seg in range(segments):
            a0 = seg * 2*math.pi / segments
            a1 = (seg+1) * 2*math.pi / segments
            x0, z0 = math.cos(a0)*radius, math.sin(a0)*radius
            x1, z1 = math.cos(a1)*radius, math.sin(a1)*radius
            base = len(verts) // 6
            n0 = _normalize3((x0, 0, z0))
            n1 = _normalize3((x1, 0, z1))
            for (px, py, pz, nx, ny, nz) in [
                (cx+x0, cy-half, cz+z0, n0[0], n0[1], n0[2]),
                (cx+x1, cy-half, cz+z1, n1[0], n1[1], n1[2]),
                (cx+x1, cy+half, cz+z1, n1[0], n1[1], n1[2]),
                (cx+x0, cy+half, cz+z0, n0[0], n0[1], n0[2]),
            ]:
                verts.extend([px, py, pz, nx, ny, nz])
            indices.extend([base, base+1, base+2, base, base+2, base+3])

        for cap_y, cap_sign in [(-half, -1.0), (half, 1.0)]:
            rings = 4
            for ring in range(rings):
                phi0 = ring * (math.pi*0.5) / rings
                phi1 = (ring+1) * (math.pi*0.5) / rings
                for seg in range(segments):
                    a0 = seg * 2*math.pi / segments
                    a1 = (seg+1) * 2*math.pi / segments
                    for (phi, aa, ab) in [(phi0, a0, a1), (phi1, a0, a1)]:
                        pass
                    r0, r1 = math.cos(phi0)*radius, math.cos(phi1)*radius
                    h0, h1 = math.sin(phi0)*radius*cap_sign, math.sin(phi1)*radius*cap_sign
                    x00, z00 = math.cos(a0)*r0, math.sin(a0)*r0
                    x01, z01 = math.cos(a1)*r0, math.sin(a1)*r0
                    x10, z10 = math.cos(a0)*r1, math.sin(a0)*r1
                    x11, z11 = math.cos(a1)*r1, math.sin(a1)*r1
                    pts = [
                        (cx+x00, cy+cap_y+h0, cz+z00),
                        (cx+x01, cy+cap_y+h0, cz+z01),
                        (cx+x11, cy+cap_y+h1, cz+z11),
                        (cx+x10, cy+cap_y+h1, cz+z10),
                    ]
                    base = len(verts) // 6
                    for px, py, pz in pts:
                        nx, ny, nz = _normalize3((px-cx, (py-cy-cap_y)*cap_sign, pz-cz))
                        verts.extend([px, py, pz, nx, ny, nz])
                    indices.extend([base, base+1, base+2, base, base+2, base+3])

    _add_capsule(0.0, 0.0, 0.0, 0.018, 0.12)
    _add_box(0.0, -0.04, 0.022, 0.030, 0.020, 0.012)
    _add_box(0.0,  0.035, 0.010, 0.022, 0.015, 0.010)
    _add_box(-0.014, -0.015, 0.016, 0.008, 0.010, 0.008)
    _add_box( 0.014, -0.015, 0.016, 0.008, 0.010, 0.008)

    return np.array(verts, dtype='f4'), np.array(indices, dtype='i4')


class ControllerRenderer:
    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._prog = ctx.program(vertex_shader=_CTRL_VERT, fragment_shader=_CTRL_FRAG)
        self._uloc = {n: self._prog[n] for n in self._prog}
        self._gl_clear = ctypes.windll.opengl32.glClear

        verts, indices = _build_controller_mesh()
        self._vbo = ctx.buffer(verts.tobytes())
        self._ibo = ctx.buffer(indices.tobytes())
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._vbo, '3f 3f', 'in_pos', 'in_normal')],
            self._ibo,
        )

    def _set(self, name, val):
        if name in self._uloc:
            self._uloc[name].value = val

    def render_controller(self, ctrl: ControllerState, proj, view, color):
        if not ctrl.valid:
            return

        model = _make_model_matrix(_xr_to_world(ctrl.pos), ctrl.quat, scale=1.0)

        mv = _mat4_mul(view, model)
        mvp = _mat4_mul(proj, mv)

        rot = _quat_to_mat3(*ctrl.quat)
        nm = (
            rot[0], rot[3], rot[6],
            rot[1], rot[4], rot[7],
            rot[2], rot[5], rot[8],
        )

        self._set('u_mvp', tuple(mvp))
        self._set('u_normal_mat', nm)
        self._set('u_color', color)
        self._set('u_grip', ctrl.grip)
        self._vao.render(moderngl.TRIANGLES)

    def render_for_eye(self, eye: dict):
        self._gl_clear(0x00000100)
        self._ctx.enable(moderngl.DEPTH_TEST)
        proj = _make_proj_matrix(_vr_state._eye_fovs[eye['eye_idx']], near=0.001, far=50.0)
        view = _make_view_matrix(eye['pos'], eye['fwd'], eye['right'], eye['up'])
        colors = [(0.3, 0.6, 1.0), (1.0, 0.4, 0.3)]
        any_valid = any(c.valid for c in _vr_state.controllers)
        if not any_valid and not getattr(self, '_ctrl_warn_printed', False):
            self._ctrl_warn_printed = True
            print(f'[VR] Controllers not valid: grips={[c.grip for c in _vr_state.controllers]} valid={[c.valid for c in _vr_state.controllers]} pose_spaces={[s is not None for s in _vr_state._pose_spaces]} action_set={_vr_state._action_set is not None}')
        for i, ctrl in enumerate(_vr_state.controllers):
            self.render_controller(ctrl, proj, view, colors[i])
        self._ctx.disable(moderngl.DEPTH_TEST)

    def release(self):
        self._vbo.release()
        self._ibo.release()
        self._vao.release()
        self._prog.release()


class VRRenderer:
    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._fbos = [
            VREyeFramebuffer(ctx, VRState.EYE_TEX_W, VRState.EYE_TEX_H),
            VREyeFramebuffer(ctx, VRState.EYE_TEX_W, VRState.EYE_TEX_H),
        ]
        self._compose_prog = ctx.program(vertex_shader=_COMPOSE_VERT, fragment_shader=_COMPOSE_FRAG)
        self._compose_uloc = {n: self._compose_prog[n] for n in self._compose_prog}
        verts = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        self._compose_vao = ctx.simple_vertex_array(self._compose_prog, ctx.buffer(verts), 'in_position')
        self._ctrl_renderer = ControllerRenderer(ctx)

    def eye_fbo(self, eye_idx: int) -> VREyeFramebuffer:
        return self._fbos[eye_idx]

    def render_controllers_for_eye(self, eye: dict):
        fb = self._fbos[eye['eye_idx']].fbo
        fb.use()
        self._ctx.viewport = (0, 0, self._fbos[eye['eye_idx']].w, self._fbos[eye['eye_idx']].h)
        self._ctrl_renderer.render_for_eye(eye)

    def compose_to_screen(self, screen_fbo, wnd_w: int, wnd_h: int):
        screen_fbo.use()
        self._ctx.viewport = (0, 0, wnd_w, wnd_h)
        try:
            self._ctx.scissor = None
        except Exception:
            pass
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.disable(moderngl.BLEND)
        self._ctx.clear(0.0, 0.0, 0.0)
        self._fbos[EYE_LEFT].tex.use(location=0)
        self._fbos[EYE_RIGHT].tex.use(location=1)
        u = self._compose_uloc
        if 'u_eye_left' in u:
            u['u_eye_left'].value = 0
        if 'u_eye_right' in u:
            u['u_eye_right'].value = 1
        if 'u_resolution' in u:
            u['u_resolution'].value = (float(wnd_w), float(wnd_h))
        self._compose_vao.render(moderngl.TRIANGLE_STRIP)

    def release(self):
        for fb in self._fbos:
            fb.release()
        self._ctrl_renderer.release()
        self._compose_prog.release()


_COMPOSE_VERT = open(_SHADER_DIR / "shaders/vr/vr_compose_vert.glsl", "r", encoding="utf-8").read()
_COMPOSE_FRAG = open(_SHADER_DIR / "shaders/vr/vr_compose_frag.glsl", "r", encoding="utf-8").read()


class VRToggleState:
    def __init__(self):
        self.enabled = False
        self.renderer: VRRenderer | None = None
        self._ctx: moderngl.Context | None = None

    def toggle(self, ctx: moderngl.Context) -> bool:
        if self.enabled:
            self._disable()
        else:
            self._enable(ctx)
        return self.enabled

    def _enable(self, ctx: moderngl.Context):
        self._ctx = ctx
        if initialize(ctx):
            self.enabled = True
            print('[VR] VR mode enabled.')
        else:
            self.enabled = False
            print('[VR] VR initialization failed.')

    def _disable(self):
        if self.renderer is not None:
            self.renderer.release()
            self.renderer = None
        if _XR_AVAILABLE:
            shutdown()
        self.enabled = False
        print('[VR] VR mode disabled.')


_vr_toggle = VRToggleState()


def toggle_vr(ctx: moderngl.Context) -> bool:
    return _vr_toggle.toggle(ctx)


def vr_enabled() -> bool:
    return _vr_toggle.enabled or _vr_state.active


def get_hmd_pos_offset() -> tuple[float, float, float]:
    return _vr_state._hmd_pos_offset


def reset_hmd_origin():
    _vr_state._hmd_pos_origin = None


def get_ipd() -> float:
    return _vr_state.ipd_override


def set_ipd(ipd: float):
    _vr_state.ipd_override = max(0.010, min(0.200, ipd))


def get_renderer() -> VRRenderer | None:
    return _vr_toggle.renderer


def render_vr_frame(fractal_window_render_eye, ctx: moderngl.Context, screen_fbo, params, wnd_w: int, wnd_h: int):
    rnd = _vr_toggle.renderer
    if rnd is None:
        print('[VR] render_vr_frame: renderer is None, skipping')
        end_xr_frame()
        return
    use_xr = _vr_state.active and _vr_state._session_running and len(_vr_state._swapchains) == 2
    eyes = get_eye_transforms(params)
    for eye in eyes:
        efbo = rnd.eye_fbo(eye['eye_idx'])
        efbo.fbo.use()
        ctx.viewport = (0, 0, efbo.w, efbo.h)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)
        efbo.fbo.clear(0.0, 0.0, 0.0, 1.0, 1.0)
        fractal_window_render_eye(efbo.fbo, efbo.w, efbo.h, eye)

    for eye in eyes:
        rnd.render_controllers_for_eye(eye)

    if use_xr:
        try:
            gl = _load_gl_funcs()
            status_checked = [False]
            for i, sc in enumerate(_vr_state._swapchains):
                dst_tex = sc.acquire()
                try:
                    src_fb = rnd.eye_fbo(i)
                    src_fbo_id = src_fb.fbo.glo
                    w, h = sc.w, sc.h
                    sw, sh = src_fb.w, src_fb.h
                    blit_id = _get_blit_fbo_id(gl, i)
                    gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, blit_id)
                    gl.FramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst_tex, 0)
                    gl.DrawBuffer(GL_COLOR_ATTACHMENT0)
                    if not status_checked[0]:
                        if gl.CheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                            print(f'[VR] swapchain fbo {i} incomplete')
                        status_checked[0] = True
                    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, src_fbo_id)
                    gl.ReadBuffer(GL_COLOR_ATTACHMENT0)
                    gl.BlitFramebuffer(0, 0, sw, sh, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST)
                except Exception as e:
                    print(f'[VR] blit eye {i} error: {e}')
                finally:
                    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, 0)
                    gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
                    try:
                        sc.release_image()
                    except Exception as e:
                        print(f'[VR] release_image {i} error: {e}')
        except Exception as e:
            print(f'[VR] Swapchain blit error: {e}')
    rnd.compose_to_screen(screen_fbo, wnd_w, wnd_h)
    if use_xr:
        try:
            end_xr_frame()
        except Exception:
            pass


def poll_xr_events():
    if not (_XR_AVAILABLE and _vr_state.active and _vr_state.instance):
        return
    try:
        while True:
            try:
                event = xr.poll_event(_vr_state.instance)
            except xr.EventUnavailable:
                break
            if event.type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                try:
                    if isinstance(event, xr.EventDataSessionStateChanged):
                        ev = event
                    else:
                        ev = ctypes.cast(
                            ctypes.addressof(event),
                            ctypes.POINTER(xr.EventDataSessionStateChanged)
                        ).contents
                except Exception:
                    ev = ctypes.cast(
                        ctypes.addressof(event),
                        ctypes.POINTER(xr.EventDataSessionStateChanged)
                    ).contents
                if ev.state == xr.SessionState.READY and not _vr_state._session_running:
                    try:
                        xr.begin_session(
                            _vr_state.session,
                            xr.SessionBeginInfo(primary_view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO)
                        )
                        _vr_state._session_running = True
                        _vr_state._ever_running = True
                        _create_swapchains()
                        ctx_for_renderer = _vr_toggle._ctx
                        if ctx_for_renderer is None:
                            try:
                                import moderngl
                                ctx_for_renderer = moderngl.create_context(standalone=False)
                            except Exception:
                                ctx_for_renderer = None
                        if ctx_for_renderer is not None:
                            if _vr_toggle.renderer is not None:
                                try:
                                    _vr_toggle.renderer.release()
                                except Exception:
                                    pass
                                _vr_toggle.renderer = None
                            try:
                                _vr_toggle.renderer = VRRenderer(ctx_for_renderer)
                                _vr_toggle._ctx = ctx_for_renderer
                            except Exception as e:
                                print(f'[VR] VRRenderer creation failed: {e}')
                        print('[VR] Session started, swapchains ready.')
                    except Exception as e:
                        print(f'[VR] begin_session failed: {e}')
                elif ev.state in (xr.SessionState.STOPPING, xr.SessionState.EXITING) and _vr_state._session_running:
                    try:
                        end_xr_frame()
                    except Exception:
                        pass
                    try:
                        xr.end_session(_vr_state.session)
                    except Exception:
                        pass
                    _vr_state._session_running = False
                    _vr_state._frame_begun = False
                    _vr_state._frame_discard = False
                    _vr_state._frame_state = None
                    for p_i in range(2):
                        _vr_state._eye_poses[p_i] = None
    except Exception as e:
        print(f'[VR] poll_xr_events error: {e}')


def sync_hmd_pose():
    if not (_XR_AVAILABLE and _vr_state.active and _vr_state._session_running):
        return False
    try:
        frame_state = xr.wait_frame(_vr_state.session, xr.FrameWaitInfo())
    except Exception as e:
        print(f'[VR] wait_frame error: {e}')
        _vr_state._frame_state = None
        _vr_state._frame_begun = False
        _vr_state._frame_discard = False
        return False
    _vr_state._frame_state = frame_state
    _vr_state._display_time = frame_state.predicted_display_time
    try:
        xr.begin_frame(_vr_state.session, xr.FrameBeginInfo())
        _vr_state._frame_discard = False
    except Exception as e:
        _vr_state._frame_discard = True
        print(f'[VR] begin_frame discarded: {e}')
    _vr_state._frame_begun = True
    try:
        _, views = xr.locate_views(_vr_state.session, xr.ViewLocateInfo(
            view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO,
            display_time=frame_state.predicted_display_time,
            space=_vr_state.space
        ))
        for i, view in enumerate(views[:2]):
            p, o, f = view.pose.position, view.pose.orientation, view.fov
            _vr_state._eye_positions[i] = (p.x, p.y, p.z)
            _vr_state._eye_quats[i] = (o.x, o.y, o.z, o.w)
            _vr_state._hmd_pos = _vr_state._eye_positions[0]
            _vr_state._hmd_quat = _vr_state._eye_quats[0]
            _vr_state._eye_fovs[i] = (f.angle_left, f.angle_right, f.angle_up, f.angle_down)
            _vr_state._eye_poses[i] = view.pose
        p0 = _vr_state._eye_positions[0]
        p1 = _vr_state._eye_positions[1]
        mid_pos = ((p0[0]+p1[0])*0.5, (p0[1]+p1[1])*0.5, (p0[2]+p1[2])*0.5)
        if _vr_state._hmd_pos_origin is None:
            _vr_state._hmd_pos_origin = mid_pos
        ox, oy, oz = _vr_state._hmd_pos_origin
        _vr_state._hmd_pos_offset = (mid_pos[0]-ox, mid_pos[1]-oy, mid_pos[2]-oz)
        _sync_controller_input()
        return True
    except Exception as e:
        print(f'[VR] locate_views error: {e}')
        return False


def end_xr_frame():
    if not (_XR_AVAILABLE and _vr_state.active and _vr_state._frame_begun):
        return
    fs = _vr_state._frame_state
    try:
        layers = []
        if not _vr_state._frame_discard and fs is not None and len(_vr_state._swapchains) == 2 and all(p is not None for p in _vr_state._eye_poses):
            layer = getattr(_vr_state, '_xr_layer', None)
            if layer is None:
                sub_imgs = []
                fovs = []
                for i in range(2):
                    sc = _vr_state._swapchains[i]
                    sub_imgs.append(xr.SwapchainSubImage(
                        swapchain=sc.swapchain,
                        image_rect=xr.Rect2Di(
                            offset=xr.Offset2Di(x=0, y=0),
                            extent=xr.Extent2Di(width=sc.w, height=sc.h)
                        ),
                        image_array_index=0
                    ))
                    fovs.append(xr.Fovf(angle_left=0.0, angle_right=0.0, angle_up=0.0, angle_down=0.0))
                proj_views = (xr.CompositionLayerProjectionView * 2)(
                    xr.CompositionLayerProjectionView(sub_image=sub_imgs[0], fov=fovs[0]),
                    xr.CompositionLayerProjectionView(sub_image=sub_imgs[1], fov=fovs[1]))
                proj_layer = xr.CompositionLayerProjection()
                proj_layer.layer_flags = 0
                proj_layer.space = _vr_state.space
                proj_layer.view_count = 2
                proj_layer._views = ctypes.cast(proj_views, ctypes.POINTER(xr.CompositionLayerProjectionView))
                layers_arr = (ctypes.POINTER(xr.CompositionLayerBaseHeader) * 1)(
                    ctypes.cast(ctypes.byref(proj_layer), ctypes.POINTER(xr.CompositionLayerBaseHeader))
                )
                _vr_state._xr_layer = (proj_views, fovs, proj_layer, layers_arr)
            proj_views, fovs, proj_layer, layers_arr = _vr_state._xr_layer
            for i in range(2):
                fovs[i] = xr.Fovf(
                    angle_left=_vr_state._eye_fovs[i][0],
                    angle_right=_vr_state._eye_fovs[i][1],
                    angle_up=_vr_state._eye_fovs[i][2],
                    angle_down=_vr_state._eye_fovs[i][3]
                )
                proj_views[i].pose = _vr_state._eye_poses[i]
                proj_views[i].fov = fovs[i]
            layers = layers_arr
        xr.end_frame(_vr_state.session, xr.FrameEndInfo(
            display_time=_vr_state._display_time,
            environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
            layer_count=len(layers),
            layers=layers if layers else None,
        ))
        _vr_state._frames_rendered += 1
    except Exception as e:
        print(f'[VR] end_xr_frame error: {e}')
    finally:
        _vr_state._frame_begun = False
        _vr_state._frame_discard = False
        _vr_state._frame_state = None


def had_frames() -> bool:
    return _vr_state._frames_rendered > 0


def session_running() -> bool:
    return _vr_state._session_running


def get_controllers():
    return _vr_state.controllers