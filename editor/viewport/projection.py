# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import numpy as np
from core.maths.math3d import Vec3


def screen_to_world(vp, screen_x: int, screen_y: int) -> Vec3:
    w = vp.width()
    h = vp.height()
    if w <= 0 or h <= 0:
        return Vec3.zero()
    cam = vp._cam
    proj_mat = cam.get_projection_matrix(w / max(h, 1))
    view_mat = cam.get_view_matrix()
    inv_proj = proj_mat.inverted()
    inv_view = view_mat.inverted()
    ndc_x = (2.0 * screen_x) / w - 1.0
    ndc_y = 1.0 - (2.0 * screen_y) / h
    clip_near = [ndc_x, ndc_y, -1.0, 1.0]
    clip_far = [ndc_x, ndc_y, 1.0, 1.0]

    def transform_vec4(mat, v):
        d = mat._d
        return (
            d[0, 0] * v[0] + d[1, 0] * v[1] + d[2, 0] * v[2] + d[3, 0] * v[3],
            d[0, 1] * v[0] + d[1, 1] * v[1] + d[2, 1] * v[2] + d[3, 1] * v[3],
            d[0, 2] * v[0] + d[1, 2] * v[1] + d[2, 2] * v[2] + d[3, 2] * v[3],
            d[0, 3] * v[0] + d[1, 3] * v[1] + d[2, 3] * v[2] + d[3, 3] * v[3],
        )

    ex, ey, ez, ew = transform_vec4(inv_proj, clip_near)
    if abs(ew) > 0.001:
        eye_near = Vec3(ex / ew, ey / ew, ez / ew)
    else:
        return Vec3.zero()
    fx, fy, fz, fw = transform_vec4(inv_proj, clip_far)
    if abs(fw) > 0.001:
        eye_far = Vec3(fx / fw, fy / fw, fz / fw)
    else:
        return Vec3.zero()
    wx, wy, wz, ww = transform_vec4(inv_view, [eye_near.x, eye_near.y, eye_near.z, 1.0])
    if abs(ww) > 0.001:
        world_near = Vec3(wx / ww, wy / ww, wz / ww)
    else:
        return Vec3.zero()
    wx, wy, wz, ww = transform_vec4(inv_view, [eye_far.x, eye_far.y, eye_far.z, 1.0])
    if abs(ww) > 0.001:
        world_far = Vec3(wx / ww, wy / ww, wz / ww)
    else:
        return Vec3.zero()
    ray_origin = world_near
    ray_dir = (world_far - ray_origin).normalized()
    if abs(ray_dir.y) < 0.0001:
        return Vec3.zero()
    t = -ray_origin.y / ray_dir.y
    if t < 0:
        return Vec3.zero()
    return ray_origin + ray_dir * t


def world_to_screen(vp, world_pos: Vec3):
    w, h = vp.width(), vp.height()
    aspect = w / max(1, h)
    view = vp._cam.get_view_matrix()
    proj = vp._cam.get_projection_matrix(aspect)
    vp_mat = view * proj
    clip = vp_mat._d @ np.array([world_pos.x, world_pos.y, world_pos.z, 1.0])
    if abs(clip[3]) < 1e-6:
        return None
    ndc = clip[:3] / clip[3]
    sx = (ndc[0] + 1.0) * 0.5 * w
    sy = (1.0 - ndc[1]) * 0.5 * h
    if ndc[2] < -1 or ndc[2] > 1:
        return None
    return (sx, sy)


def screen_to_ray(vp, sx: int, sy: int) -> tuple[Vec3, Vec3]:
    pw, ph = vp._get_physical_dims()
    aspect = pw / max(1, ph)
    dpr = vp.devicePixelRatio()
    fx = sx * dpr
    fy = sy * dpr
    ndc_x = (2.0 * fx / pw) - 1.0
    ndc_y = 1.0 - (2.0 * fy / ph)
    view = vp._cam.get_view_matrix()
    proj = vp._cam.get_projection_matrix(aspect)
    inv_vp = (view * proj).inverted()
    near_ndc = np.array([ndc_x, ndc_y, -1.0, 1.0])
    far_ndc = np.array([ndc_x, ndc_y, 1.0, 1.0])
    near_w = near_ndc @ inv_vp._d
    far_w = far_ndc @ inv_vp._d
    near_w /= near_w[3]
    far_w /= far_w[3]
    origin = Vec3(float(near_w[0]), float(near_w[1]), float(near_w[2]))
    direction = Vec3(float(far_w[0] - near_w[0]), float(far_w[1] - near_w[1]), float(far_w[2] - near_w[2])).normalized()
    return origin, direction


def screen_to_plane(vp, sx: int, sy: int, plane_point: Vec3) -> Vec3:
    origin, direction = screen_to_ray(vp, sx, sy)
    cam_fwd = vp._cam.forward
    denom = direction.dot(cam_fwd)
    if abs(denom) < 0.0001:
        return plane_point
    t = (plane_point - origin).dot(cam_fwd) / denom
    if t < 0:
        return plane_point
    return origin + direction * t


def project_world_pos(vp, world_pos, vp_mat, vw, vh):
    clip = np.array([world_pos.x, world_pos.y, world_pos.z, 1.0]) @ vp_mat._d
    if abs(clip[3]) < 1e-6:
        return None
    ndc = clip[:3] / clip[3]
    if ndc[2] < -1.0 or ndc[2] > 1.0:
        return None
    sx = (ndc[0] + 1.0) * 0.5 * vw
    sy = (1.0 - ndc[1]) * 0.5 * vh
    if sx < -100 or sx > vw + 100 or sy < -100 or sy > vh + 100:
        return None
    return (sx, sy)
