# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import random
import threading
import numpy as np
from enum import Enum
from typing import Optional, List, Tuple, Dict, Callable, Any
from dataclasses import dataclass, field
from core.math.math3d import Vec3, Mat4


class GizmoType(Enum):
    POINT = "point"
    LINE = "line"
    CIRCLE = "circle"
    SPHERE = "sphere"
    BOX = "box"
    ARC = "arc"
    CAPSULE = "capsule"
    GRID = "grid"
    CONE = "cone"
    AXIS = "axis"
    TEXT = "text"
    RING = "ring"
    ARROW = "arrow"
    CROSS = "cross"
    DASHED = "dashed"
    BEZIER = "bezier"
    TRIANGLE = "triangle"
    POLY = "poly"
    CYLINDER = "cylinder"
    ELLIPSE = "ellipse"
    RECT = "rect"
    RAY = "ray"
    BBOX = "bbox"
    FRUSTUM = "frustum"
    HELIX = "helix"
    PARABOLA = "parabola"
    SPLINE = "spline"
    ICOSPHERE = "icosphere"
    LABEL = "label"
    ICON = "icon"
    TORUS = "torus"
    PIPE = "pipe"
    STAR = "star"
    PIE = "pie"
    WEDGE = "wedge"
    SPIRAL = "spiral"
    CHORD = "chord"
    HEMISPHERE = "hemisphere"
    PYRAMID = "pyramid"


class LineStyle(Enum):
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    HIDDEN = "hidden"


class _GizmoStyleCtx:
    def __init__(self, mgr, kwargs: dict):
        self._mgr = mgr
        self._kwargs = kwargs
    def __enter__(self):
        self._mgr.push_style(**self._kwargs)
        return self
    def __exit__(self, *args):
        self._mgr.pop_style()


@dataclass
class GizmoData:
    gizmo_type: GizmoType
    position: Tuple[float, float, float]
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    size: float = 1.0
    duration: float = 0.0
    layer: int = 0
    world_space: bool = True
    thickness: float = 1.0
    filled: bool = False
    alpha: float = 1.0
    text: str = ""
    end_position: Optional[Tuple[float, float, float]] = None
    normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    angle_start: float = 0.0
    angle_end: float = 360.0
    segments: int = 32
    inner_radius: float = 0.0
    points: Optional[List[Tuple[float, float, float]]] = None
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    dash_length: float = 0.3
    gap_length: float = 0.15
    font_size: int = 14
    cull_distance: float = -1.0
    on_click: Optional[Callable] = None
    unique_id: Optional[str] = None
    tag: str = ""
    _screen_pos: Optional[Tuple[int, int]] = None
    height: float = 0.0
    radius_y: float = 0.0
    arrow_size: float = 0.2
    turns: float = 3.0
    fov: float = 60.0
    near_plane: float = 0.1
    far_plane: float = 100.0
    min_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_point: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    subdivisions: int = 1
    rotation: Optional[Tuple[float, float, float, float]] = None
    transform: Optional[List[float]] = None
    billboard: bool = True
    texture_path: str = ''
    line_style: 'LineStyle' = LineStyle.SOLID


def _np_color(c: Tuple[float, float, float, float], n: int) -> np.ndarray:
    arr = np.empty((n, 4), dtype=np.float32)
    arr[:] = c
    return arr


def _quat_to_mat3(rot: Tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = rot
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy)],
        [2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)],
    ], dtype=np.float32)


def _euler_to_mat3(rot: Tuple[float, float, float]) -> np.ndarray:
    cx, cy, cz = (math.cos(math.radians(a)) for a in rot)
    sx, sy, sz = (math.sin(math.radians(a)) for a in rot)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float32)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float32)
    return Rz @ Ry @ Rx

def _apply_rotation(pts: np.ndarray, rot: Optional[Tuple[float, float, float, float]]) -> np.ndarray:
    if rot is None:
        return pts
    if len(rot) == 3:
        R = _euler_to_mat3(rot)
    else:
        R = _quat_to_mat3(rot)
    return pts @ R


def _apply_line_style(starts: np.ndarray, ends: np.ndarray, colors: np.ndarray,
                      style: 'LineStyle', dash_len: float = 0.3, gap_len: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if style == LineStyle.SOLID or starts.shape[0] == 0:
        return starts, ends, colors
    if style == LineStyle.HIDDEN:
        return starts[:0], ends[:0], colors[:0]
    n = starts.shape[0]
    new_s, new_e, new_c = [], [], []
    if style == LineStyle.DASHED:
        step = dash_len + gap_len
        for i in range(n):
            sx, sy, sz = starts[i]; ex, ey, ez = ends[i]
            dx, dy, dz = ex-sx, ey-sy, ez-sz
            ln = math.sqrt(dx*dx + dy*dy + dz*dz)
            if ln < 1e-8: continue
            nd = int(ln / step) if ln / step >= 1 else 1
            for j in range(nd):
                t0 = j * step
                t1 = min(j * step + dash_len, ln)
                new_s.append([sx + dx/ln*t0, sy + dy/ln*t0, sz + dz/ln*t0])
                new_e.append([sx + dx/ln*t1, sy + dy/ln*t1, sz + dz/ln*t1])
                new_c.append(colors[i])
    elif style == LineStyle.DOTTED:
        dot_len = max(gap_len * 0.15, 0.02)
        step = dot_len + gap_len
        for i in range(n):
            sx, sy, sz = starts[i]; ex, ey, ez = ends[i]
            dx, dy, dz = ex-sx, ey-sy, ez-sz
            ln = math.sqrt(dx*dx + dy*dy + dz*dz)
            if ln < 1e-8: continue
            nd = int(ln / step) if ln / step >= 1 else 1
            for j in range(nd):
                t0 = j * step
                t1 = min(j * step + dot_len, ln)
                new_s.append([sx + dx/ln*t0, sy + dy/ln*t0, sz + dz/ln*t0])
                new_e.append([sx + dx/ln*t1, sy + dy/ln*t1, sz + dz/ln*t1])
                new_c.append(colors[i])
    if not new_s:
        return starts[:0], ends[:0], colors[:0]
    return (np.array(new_s, dtype=np.float32), np.array(new_e, dtype=np.float32), np.array(new_c, dtype=np.float32))


_GIZMO_LINE_BUILDERS: Dict[GizmoType, Callable[[GizmoData], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {}


def _register(t: GizmoType):
    def wrapper(f):
        _GIZMO_LINE_BUILDERS[t] = f
        return f
    return wrapper


@_register(GizmoType.POINT)
def _build_point(g: GizmoData):
    p = g.position; s = g.size * 0.5
    c = np.array([[p[0]], [p[0]], [p[1]], [p[1]], [p[2]], [p[2]]], dtype=np.float32)
    pts = np.array([
        [p[0]-s, p[1], p[2]], [p[0]+s, p[1], p[2]],
        [p[0], p[1]-s, p[2]], [p[0], p[1]+s, p[2]],
        [p[0], p[1], p[2]-s], [p[0], p[1], p[2]+s],
    ], dtype=np.float32)
    starts = pts[0::2]; ends = pts[1::2]
    return starts, ends, _np_color(g.color, 3)


@_register(GizmoType.LINE)
def _build_line(g: GizmoData):
    if g.end_position is None:
        return None
    s = np.array([[g.position[0], g.position[1], g.position[2]]], dtype=np.float32)
    e = np.array([[g.end_position[0], g.end_position[1], g.end_position[2]]], dtype=np.float32)
    return s, e, _np_color(g.color, 1)


@_register(GizmoType.CIRCLE)
def _build_circle(g: GizmoData):
    p = g.position; n = g.normal; r = g.size; segs = g.segments
    nv = Vec3(n[0], n[1], n[2])
    ref = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(ref).normalized()
    p2 = nv.cross(p1).normalized()
    theta = np.linspace(0, 2.0 * math.pi, segs + 1, dtype=np.float32)
    ct = np.cos(theta) * r; st = np.sin(theta) * r
    po = np.array([p[0], p[1], p[2]], dtype=np.float32)
    p1a = np.array([p1.x, p1.y, p1.z], dtype=np.float32)
    p2a = np.array([p2.x, p2.y, p2.z], dtype=np.float32)
    pts = po + p1a * ct[:, None] + p2a * st[:, None]
    return pts[:-1], pts[1:], _np_color(g.color, segs)


_SPHERE_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}

def _get_sphere_edges(segs: int):
    cached = _SPHERE_CACHE.get(segs)
    if cached is not None:
        return cached
    lats = np.linspace(0, math.pi, segs + 1, dtype=np.float32)
    lons = np.linspace(0, 2.0 * math.pi, segs + 1, dtype=np.float32)
    lat_sin, lat_cos = np.sin(lats), np.cos(lats)
    lon_sin, lon_cos = np.sin(lons), np.cos(lons)
    nv = segs + 1
    verts = np.empty(((segs + 1) * (segs + 1), 3), dtype=np.float32)
    idx = 0
    for li in range(segs + 1):
        for lj in range(segs + 1):
            verts[idx, 0] = lat_sin[li] * lon_cos[lj]
            verts[idx, 1] = lat_cos[li]
            verts[idx, 2] = lat_sin[li] * lon_sin[lj]
            idx += 1
    lines_buf = []
    for li in range(segs):
        for lj in range(segs):
            i0 = li * nv + lj; i1 = li * nv + lj + 1
            i2 = (li + 1) * nv + lj; i3 = (li + 1) * nv + lj + 1
            lines_buf.extend([i0, i1, i1, i2, i2, i3, i3, i0])
    lidx = np.array(lines_buf, dtype=np.int32)
    starts = verts[lidx[0::2]]
    ends = verts[lidx[1::2]]
    _SPHERE_CACHE[segs] = (starts, ends)
    return starts, ends


@_register(GizmoType.SPHERE)
def _build_sphere(g: GizmoData):
    p = g.position; r = g.size; segs = max(g.segments // 2, 4)
    starts, ends = _get_sphere_edges(segs)
    n = starts.shape[0]
    starts = starts * r + np.array([p[0], p[1], p[2]], dtype=np.float32)
    ends = ends * r + np.array([p[0], p[1], p[2]], dtype=np.float32)
    return starts, ends, _np_color(g.color, n)


@_register(GizmoType.BOX)
def _build_box(g: GizmoData):
    p = g.position; h = g.size if isinstance(g.size, (tuple, list)) else (g.size, g.size, g.size)
    hx, hy, hz = h[0]*0.5, h[1]*0.5, h[2]*0.5
    corners = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], dtype=np.float32)
    corners = _apply_rotation(corners, g.rotation)
    corners += np.array([p[0], p[1], p[2]], dtype=np.float32)
    edges = np.array([(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)])
    return corners[edges[:, 0]], corners[edges[:, 1]], _np_color(g.color, 12)


@_register(GizmoType.ARC)
def _build_arc(g: GizmoData):
    p = g.position; n = g.normal; r = g.size
    a0 = math.radians(g.angle_start); a1 = math.radians(g.angle_end)
    segs = g.segments
    nv = Vec3(n[0], n[1], n[2])
    ref = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(ref).normalized()
    p2 = nv.cross(p1).normalized()
    theta = np.linspace(a0, a1, segs + 1, dtype=np.float32)
    ct = np.cos(theta) * r; st = np.sin(theta) * r
    po = np.array([p[0], p[1], p[2]], dtype=np.float32)
    p1a = np.array([p1.x, p1.y, p1.z], dtype=np.float32)
    p2a = np.array([p2.x, p2.y, p2.z], dtype=np.float32)
    pts = po + p1a * ct[:, None] + p2a * st[:, None]
    return pts[:-1], pts[1:], _np_color(g.color, segs)


@_register(GizmoType.CAPSULE)
def _build_capsule(g: GizmoData):
    p = g.position; ep = g.end_position; r = g.size
    if ep is None:
        ep = (p[0], p[1] + 1.0, p[2])
    segs = max(g.segments // 2, 8)
    dx, dy, dz = ep[0]-p[0], ep[1]-p[1], ep[2]-p[2]
    h = math.sqrt(dx*dx + dy*dy + dz*dz)
    if h < 1e-8:
        return None
    nx, ny, nz = dx/h, dy/h, dz/h
    ref = Vec3(0,1,0)
    axis = Vec3(nx, ny, nz)
    if abs(axis.dot(ref)) > 0.99:
        ref = Vec3(1,0,0)
    p1 = axis.cross(ref).normalized()
    p2 = axis.cross(p1).normalized()
    theta = np.linspace(0, 2.0 * math.pi, segs + 1, dtype=np.float32)
    ct = np.cos(theta) * r; st = np.sin(theta) * r
    n_verts = 3 * segs + 2 * segs * (segs // 2)
    starts = np.empty((n_verts, 3), dtype=np.float32)
    ends = np.empty((n_verts, 3), dtype=np.float32)
    idx = 0
    for top in (False, True):
        base_y = 0.0 if not top else h
        for i in range(segs):
            a0 = theta[i]; a1 = theta[i+1]
            c0, s0 = math.cos(a0), math.sin(a0)
            c1, s1 = math.cos(a1), math.sin(a1)
            for side in range(2):
                a = a0 if side == 0 else a1
                ca, sa = math.cos(a), math.sin(a)
                px = p[0] + r * (p1.x * ca + p2.x * sa)
                py = (p[1] + base_y) + r * (p1.y * ca + p2.y * sa)
                pz = p[2] + r * (p1.z * ca + p2.z * sa)
                if side == 0:
                    starts[idx, 0], starts[idx, 1], starts[idx, 2] = px, py, pz
                else:
                    ends[idx, 0], ends[idx, 1], ends[idx, 2] = px, py, pz
                    idx += 1
    for i in range(segs):
        a = theta[i]
        ca, sa = math.cos(a), math.sin(a)
        bx = p[0] + r * (p1.x * ca + p2.x * sa)
        by = p[1] + r * (p1.y * ca + p2.y * sa)
        bz = p[2] + r * (p1.z * ca + p2.z * sa)
        tx = p[0] + r * (p1.x * ca + p2.x * sa)
        ty = p[1] + h + r * (p1.y * ca + p2.y * sa)
        tz = p[2] + r * (p1.z * ca + p2.z * sa)
        starts[idx] = [bx, by, bz]; ends[idx] = [tx, ty, tz]; idx += 1
    theta_h = np.linspace(0, math.pi, segs // 2 + 1, dtype=np.float32)
    for hemi_top in (False, True):
        base_y = 0.0 if not hemi_top else h
        sign = 1 if hemi_top else -1
        for i in range(len(theta_h) - 1):
            for j in range(segs):
                a0 = theta_h[i]; a1 = theta_h[i+1]
                b0 = 2.0 * math.pi * j / segs
                r0 = r * math.sin(a0); y0 = sign * r * (1 - math.cos(a0))
                r1 = r * math.sin(a1); y1 = sign * r * (1 - math.cos(a1))
                x0 = r0 * math.cos(b0); z0 = r0 * math.sin(b0)
                x1 = r1 * math.cos(b0); z1 = r1 * math.sin(b0)
                starts[idx] = [p[0]+x0, p[1]+base_y+y0, p[2]+z0]
                ends[idx] = [p[0]+x1, p[1]+base_y+y1, p[2]+z1]
                idx += 1
    return starts, ends, _np_color(g.color, n_verts)


@_register(GizmoType.GRID)
def _build_grid(g: GizmoData):
    p = g.position; n = g.normal; sz = g.size; divs = max(g.segments, 2)
    nv = Vec3(n[0], n[1], n[2])
    ref = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(ref).normalized()
    p2 = nv.cross(p1).normalized()
    half = sz * 0.5
    vals = np.linspace(-half, half, divs + 1, dtype=np.float32)
    total_lines = (divs + 1) * 2
    p1a = np.array([p1.x, p1.y, p1.z], dtype=np.float32)
    p2a = np.array([p2.x, p2.y, p2.z], dtype=np.float32)
    po = np.array([p[0], p[1], p[2]], dtype=np.float32)
    starts = np.empty((total_lines, 3), dtype=np.float32)
    ends = np.empty((total_lines, 3), dtype=np.float32)
    starts[:divs+1] = po + p1a * (-half) + p2a * vals[:, None]
    ends[:divs+1]   = po + p1a * half + p2a * vals[:, None]
    starts[divs+1:] = po + p1a * vals[:, None] + p2a * (-half)
    ends[divs+1:]   = po + p1a * vals[:, None] + p2a * half
    return starts, ends, _np_color(g.color, total_lines)


@_register(GizmoType.CONE)
def _build_cone(g: GizmoData):
    p = g.position; n = g.normal; r = g.size; h = g.height if g.height > 0 else g.size * 2
    segs = max(g.segments, 8)
    nv = Vec3(n[0], n[1], n[2])
    p1 = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(p1).normalized()
    p2 = nv.cross(p1).normalized()
    tip = np.array([p[0] + n[0]*h, p[1] + n[1]*h, p[2] + n[2]*h], dtype=np.float32)
    theta = np.linspace(0, 2.0*math.pi, segs+1, dtype=np.float32)
    ct = np.cos(theta)*r; st = np.sin(theta)*r
    base = np.empty((segs+1, 3), dtype=np.float32)
    base[:, 0] = p[0] + p1.x*ct + p2.x*st
    base[:, 1] = p[1] + p1.y*ct + p2.y*st
    base[:, 2] = p[2] + p1.z*ct + p2.z*st
    total = segs * 2
    starts = np.empty((total, 3), dtype=np.float32); ends = np.empty((total, 3), dtype=np.float32)
    starts[:segs] = base[:-1]; ends[:segs] = base[1:]
    starts[segs:2*segs] = base[:-1]; ends[segs:2*segs] = tip
    return starts, ends, _np_color(g.color, total)


@_register(GizmoType.AXIS)
def _build_axis(g: GizmoData):
    p = g.position; ln = g.size
    cols = [(1,0,0,1), (0,1,0,1), (0,0,1,1)]
    starts = np.array([[p[0],p[1],p[2]]]*3, dtype=np.float32)
    ends = np.array([
        [p[0]+ln, p[1], p[2]], [p[0], p[1]+ln, p[2]], [p[0], p[1], p[2]+ln],
    ], dtype=np.float32)
    colors = np.zeros((3, 4), dtype=np.float32)
    for i in range(3):
        colors[i] = cols[i]
    return starts, ends, colors


@_register(GizmoType.RING)
def _build_ring(g: GizmoData):
    p = g.position; outer = g.size; inner = g.inner_radius
    n = g.normal; segs = g.segments
    nv = Vec3(n[0], n[1], n[2])
    p1 = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(p1).normalized()
    p2 = nv.cross(p1).normalized()
    theta = np.linspace(0, 2.0*math.pi, segs+1, dtype=np.float32)
    ct = np.cos(theta); st = np.sin(theta)
    outer_pts = np.empty((segs+1, 3), dtype=np.float32)
    inner_pts = np.empty((segs+1, 3), dtype=np.float32)
    outer_pts[:, 0] = p[0] + p1.x*ct*outer + p2.x*st*outer
    outer_pts[:, 1] = p[1] + p1.y*ct*outer + p2.y*st*outer
    outer_pts[:, 2] = p[2] + p1.z*ct*outer + p2.z*st*outer
    inner_pts[:, 0] = p[0] + p1.x*ct*inner + p2.x*st*inner
    inner_pts[:, 1] = p[1] + p1.y*ct*inner + p2.y*st*inner
    inner_pts[:, 2] = p[2] + p1.z*ct*inner + p2.z*st*inner
    total = segs * 2
    starts = np.empty((total, 3), dtype=np.float32); ends = np.empty((total, 3), dtype=np.float32)
    starts[:segs] = outer_pts[:-1]; ends[:segs] = outer_pts[1:]
    starts[segs:] = inner_pts[:-1]; ends[segs:] = inner_pts[1:]
    return starts, ends, _np_color(g.color, total)


@_register(GizmoType.ARROW)
def _build_arrow(g: GizmoData):
    if g.end_position is None:
        return None
    s = np.array([[g.position[0], g.position[1], g.position[2]]], dtype=np.float32)
    e = np.array([[g.end_position[0], g.end_position[1], g.end_position[2]]], dtype=np.float32)
    dx = e[0,0] - s[0,0]; dy = e[0,1] - s[0,1]; dz = e[0,2] - s[0,2]
    ln = math.sqrt(dx*dx+dy*dy+dz*dz)
    if ln < 1e-8:
        return None
    head_sz = min(g.arrow_size if g.arrow_size > 0 else ln*0.2, ln*0.5)
    t = 1.0 - head_sz/ln
    hx = s[0,0] + dx*t; hy = s[0,1] + dy*t; hz = s[0,2] + dz*t
    nx, ny, nz = dx/ln, dy/ln, dz/ln
    ref = Vec3(0,1,0) if abs(ny)<0.9 else Vec3(1,0,0)
    ax = Vec3(nx, ny, nz)
    p1 = ax.cross(ref).normalized()
    p2 = ax.cross(p1).normalized()
    spread = head_sz * 0.5
    starts = np.empty((4, 3), dtype=np.float32)
    ends = np.empty((4, 3), dtype=np.float32)
    starts[0] = s[0]; ends[0] = e[0]
    for i in range(3):
        a = 2.0*math.pi*i/3
        ca, sa = math.cos(a)*spread, math.sin(a)*spread
        starts[i+1] = [hx, hy, hz]
        ends[i+1] = [e[0,0]+p1.x*ca+p2.x*sa, e[0,1]+p1.y*ca+p2.y*sa, e[0,2]+p1.z*ca+p2.z*sa]
    return starts, ends, _np_color(g.color, 4)


@_register(GizmoType.CROSS)
def _build_cross(g: GizmoData):
    p = g.position; s = g.size * 0.5
    starts = np.array([
        [p[0]-s, p[1], p[2]], [p[0], p[1]-s, p[2]], [p[0], p[1], p[2]-s],
    ], dtype=np.float32)
    ends = np.array([
        [p[0]+s, p[1], p[2]], [p[0], p[1]+s, p[2]], [p[0], p[1], p[2]+s],
    ], dtype=np.float32)
    return starts, ends, _np_color(g.color, 3)


@_register(GizmoType.DASHED)
def _build_dashed(g: GizmoData):
    if g.end_position is None:
        return None
    sx, sy, sz = g.position
    ex, ey, ez = g.end_position
    dx, dy, dz = ex-sx, ey-sy, ez-sz
    ln = math.sqrt(dx*dx+dy*dy+dz*dz)
    if ln < 1e-8:
        return None
    dash = g.dash_length; gap = g.gap_length
    step = dash + gap
    n = max(int(ln / step), 1)
    starts = np.empty((n, 3), dtype=np.float32); ends = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        t0 = i * step / ln; t1 = min((i * step + dash) / ln, 1.0)
        starts[i] = [sx+dx*t0, sy+dy*t0, sz+dz*t0]
        ends[i] = [sx+dx*t1, sy+dy*t1, sz+dz*t1]
    return starts, ends, _np_color(g.color, n)


@_register(GizmoType.BEZIER)
def _build_bezier(g: GizmoData):
    pts = g.points
    if not pts or len(pts) < 4:
        return None
    segs = g.segments
    p = np.array(pts, dtype=np.float32)
    t = np.linspace(0, 1, segs + 1, dtype=np.float32)[:, None]
    t2 = t*t; t3 = t2*t
    mt = 1-t; mt2 = mt*mt; mt3 = mt2*mt
    curve = mt3 * p[0] + 3*mt2*t * p[1] + 3*mt*t2 * p[2] + t3 * p[3]
    return curve[:-1], curve[1:], _np_color(g.color, segs)


@_register(GizmoType.TRIANGLE)
def _build_triangle(g: GizmoData):
    pts = g.points
    if not pts or len(pts) < 3:
        return None
    p = np.array(pts[:3], dtype=np.float32)
    starts = np.array([p[0], p[1], p[2]], dtype=np.float32)
    ends = np.array([p[1], p[2], p[0]], dtype=np.float32)
    return starts, ends, _np_color(g.color, 3)


@_register(GizmoType.POLY)
def _build_poly(g: GizmoData):
    pts = g.points
    if not pts or len(pts) < 3:
        return None
    p = np.array(pts, dtype=np.float32)
    n = len(pts)
    starts = p[:-1]; ends = p[1:]
    if not g.filled:
        starts = np.concatenate([starts, p[-1:]], axis=0)
        ends = np.concatenate([ends, p[:1]], axis=0)
    return starts, ends, _np_color(g.color, len(starts))


@_register(GizmoType.CYLINDER)
def _build_cylinder(g: GizmoData):
    p = g.position; n = g.normal; r = g.size; h = g.height if g.height > 0 else 1.0
    segs = max(g.segments, 8)
    nv = Vec3(n[0], n[1], n[2])
    p1 = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(p1).normalized()
    p2 = nv.cross(p1).normalized()
    theta = np.linspace(0, 2.0*math.pi, segs+1, dtype=np.float32)
    ct = np.cos(theta)*r; st = np.sin(theta)*r
    bot = np.empty((segs+1, 3), dtype=np.float32)
    top = np.empty((segs+1, 3), dtype=np.float32)
    bot[:, 0] = p[0] + p1.x*ct + p2.x*st
    bot[:, 1] = p[1] + p1.y*ct + p2.y*st
    bot[:, 2] = p[2] + p1.z*ct + p2.z*st
    top[:, 0] = p[0] + p1.x*ct + p2.x*st + n[0]*h
    top[:, 1] = p[1] + p1.y*ct + p2.y*st + n[1]*h
    top[:, 2] = p[2] + p1.z*ct + p2.z*st + n[2]*h
    total = segs * 3
    starts = np.empty((total, 3), dtype=np.float32); ends = np.empty((total, 3), dtype=np.float32)
    starts[:segs] = bot[:-1]; ends[:segs] = bot[1:]
    starts[segs:2*segs] = top[:-1]; ends[segs:2*segs] = top[1:]
    starts[2*segs:] = bot[:-1]; ends[2*segs:] = top[:-1]
    return starts, ends, _np_color(g.color, total)


@_register(GizmoType.ELLIPSE)
def _build_ellipse(g: GizmoData):
    p = g.position; n = g.normal; rx = g.size; ry = g.radius_y if g.radius_y > 0 else rx * 0.5
    segs = g.segments
    nv = Vec3(n[0], n[1], n[2])
    p1 = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(p1).normalized()
    p2 = nv.cross(p1).normalized()
    theta = np.linspace(0, 2.0*math.pi, segs+1, dtype=np.float32)
    ct = np.cos(theta); st = np.sin(theta)
    pts = np.empty((segs+1, 3), dtype=np.float32)
    pts[:, 0] = p[0] + p1.x*ct*rx + p2.x*st*ry
    pts[:, 1] = p[1] + p1.y*ct*rx + p2.y*st*ry
    pts[:, 2] = p[2] + p1.z*ct*rx + p2.z*st*ry
    return pts[:-1], pts[1:], _np_color(g.color, segs)


@_register(GizmoType.RECT)
def _build_rect(g: GizmoData):
    p = g.position; n = g.normal
    sz = g.size if isinstance(g.size, (tuple, list)) else (g.size, g.size)
    hw, hh = sz[0]*0.5 if isinstance(sz, (tuple,list)) else sz*0.5, (sz[1] if isinstance(sz,(tuple,list)) else sz)*0.5
    nv = Vec3(n[0], n[1], n[2])
    p1 = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(p1).normalized()
    p2 = nv.cross(p1).normalized()
    corners = np.array([[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]], dtype=np.float32)
    corners = _apply_rotation(corners, g.rotation)
    pts = np.empty((4, 3), dtype=np.float32)
    pts[:, 0] = p[0] + p1.x*corners[:,0] + p2.x*corners[:,1] + n[0]*corners[:,2]
    pts[:, 1] = p[1] + p1.y*corners[:,0] + p2.y*corners[:,1] + n[1]*corners[:,2]
    pts[:, 2] = p[2] + p1.z*corners[:,0] + p2.z*corners[:,1] + n[2]*corners[:,2]
    edges = np.array([(0,1),(1,2),(2,3),(3,0)])
    return pts[edges[:,0]], pts[edges[:,1]], _np_color(g.color, 4)


@_register(GizmoType.RAY)
def _build_ray(g: GizmoData):
    p = g.position; n = g.normal; ln = g.size
    ex = p[0] + n[0]*ln; ey = p[1] + n[1]*ln; ez = p[2] + n[2]*ln
    nv = Vec3(n[0], n[1], n[2])
    ref = Vec3(0,1,0) if abs(n[1])<0.9 else Vec3(1,0,0)
    p1 = nv.cross(ref).normalized()
    p2 = nv.cross(p1).normalized()
    head_sz = g.arrow_size if g.arrow_size > 0 else ln*0.15
    spread = head_sz * 0.5
    starts = np.empty((4, 3), dtype=np.float32); ends = np.empty((4, 3), dtype=np.float32)
    starts[0] = [p[0], p[1], p[2]]; ends[0] = [ex, ey, ez]
    for i in range(3):
        a = 2.0*math.pi*i/3
        ca, sa = math.cos(a)*spread, math.sin(a)*spread
        hx = ex - n[0]*head_sz; hy = ey - n[1]*head_sz; hz = ez - n[2]*head_sz
        starts[i+1] = [hx, hy, hz]
        ends[i+1] = [ex+p1.x*ca+p2.x*sa, ey+p1.y*ca+p2.y*sa, ez+p1.z*ca+p2.z*sa]
    return starts, ends, _np_color(g.color, 4)


@_register(GizmoType.BBOX)
def _build_bbox(g: GizmoData):
    mn = g.min_point; mx = g.max_point
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]], [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]], [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ], dtype=np.float32)
    edges = np.array([(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)])
    return corners[edges[:,0]], corners[edges[:,1]], _np_color(g.color, 12)


@_register(GizmoType.FRUSTUM)
def _build_frustum(g: GizmoData):
    p = g.position; n = g.normal; fov = g.fov; aspect = g.size if g.size > 0 else 1.5
    near = g.near_plane; far = g.far_plane
    nv = Vec3(n[0], n[1], n[2])
    ref = Vec3(0,1,0) if abs(n[1])<0.9 else Vec3(1,0,0)
    p1 = nv.cross(ref).normalized()
    p2 = nv.cross(p1).normalized()
    fov_rad = math.radians(fov)
    nh = math.tan(fov_rad * 0.5) * near; nw = nh * aspect
    fh = math.tan(fov_rad * 0.5) * far; fw = fh * aspect
    near_c = np.array([[-nw, -nh, near], [nw, -nh, near], [nw, nh, near], [-nw, nh, near]], dtype=np.float32)
    far_c = np.array([[-fw, -fh, far], [fw, -fh, far], [fw, fh, far], [-fw, fh, far]], dtype=np.float32)
    all_c = np.concatenate([near_c, far_c], axis=0)
    transformed = np.empty_like(all_c)
    transformed[:, 0] = p[0] + p1.x*all_c[:,0] + p2.x*all_c[:,1] + n[0]*all_c[:,2]
    transformed[:, 1] = p[1] + p1.y*all_c[:,0] + p2.y*all_c[:,1] + n[1]*all_c[:,2]
    transformed[:, 2] = p[2] + p1.z*all_c[:,0] + p2.z*all_c[:,1] + n[2]*all_c[:,2]
    edges = np.array([(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)])
    return transformed[edges[:,0]], transformed[edges[:,1]], _np_color(g.color, 12)


@_register(GizmoType.HELIX)
def _build_helix(g: GizmoData):
    p = g.position; n = g.normal; h = g.height if g.height > 0 else 2.0; r = g.size
    turns = max(g.turns, 1); segs = g.segments
    nv = Vec3(n[0], n[1], n[2])
    ref = Vec3(0,1,0) if abs(n[1])<0.9 else Vec3(1,0,0)
    p1 = nv.cross(ref).normalized()
    p2 = nv.cross(p1).normalized()
    total = segs * int(turns)
    theta = np.linspace(0, 2.0*math.pi*turns, total + 1, dtype=np.float32)
    ct = np.cos(theta)*r; st = np.sin(theta)*r
    vz = np.linspace(0, h, total + 1, dtype=np.float32)
    pts = np.empty((total + 1, 3), dtype=np.float32)
    pts[:, 0] = p[0] + p1.x*ct + p2.x*st + n[0]*vz
    pts[:, 1] = p[1] + p1.y*ct + p2.y*st + n[1]*vz
    pts[:, 2] = p[2] + p1.z*ct + p2.z*st + n[2]*vz
    return pts[:-1], pts[1:], _np_color(g.color, total)


@_register(GizmoType.PARABOLA)
def _build_parabola(g: GizmoData):
    s = g.position; ep = g.end_position
    if ep is None:
        return None
    h = g.height; segs = g.segments
    sx, sy, sz = s; ex, ey, ez = ep
    dx, dy, dz = ex-sx, ey-sy, ez-sz
    t = np.linspace(0, 1, segs+1, dtype=np.float32)
    x = sx + dx*t; y = sy + dy*t + h * 4 * t * (1-t); z = sz + dz*t
    pts = np.column_stack([x, y, z])
    return pts[:-1], pts[1:], _np_color(g.color, segs)


@_register(GizmoType.SPLINE)
def _build_spline(g: GizmoData):
    pts = g.points
    if not pts or len(pts) < 2:
        return None
    segs = g.segments
    p = np.array(pts, dtype=np.float32)
    n = len(p)
    total_segs = (n - 1) * segs
    t = np.linspace(0, 1, segs + 1, dtype=np.float32)[:, None]
    result = np.empty((total_segs + 1, 3), dtype=np.float32)
    idx = 0
    for i in range(n - 1):
        p0 = p[max(0, i-1)]
        p1 = p[i]; p2 = p[min(n-1, i+1)]; p3 = p[min(n-1, i+2)]
        t2 = t*t; t3 = t2*t
        mt = 1-t; mt2 = mt*mt; mt3 = mt2*mt
        catmull = 0.5 * (
            2*p1 + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t2 + (-p0+3*p1-3*p2+p3)*t3
        )
        if i < n - 2:
            result[idx:idx+segs+1] = catmull
        else:
            result[idx:] = catmull
            break
        idx += segs
    return result[:-1], result[1:], _np_color(g.color, total_segs)


_ICOSPHERE_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}

def _get_icosphere_edges(subdivisions: int):
    cached = _ICOSPHERE_CACHE.get(subdivisions)
    if cached is not None:
        return cached
    phi = (1.0 + math.sqrt(5.0)) * 0.5
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float32)
    faces = np.array([
        0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
        1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
        3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
        4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
    ], dtype=np.int32)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    for _ in range(subdivisions):
        edge_mid = {}
        new_faces = []
        n_verts = verts.shape[0]
        new_verts = []
        for i in range(0, len(faces), 3):
            a, b, c = int(faces[i]), int(faces[i+1]), int(faces[i+2])
            ab = (a, b) if a < b else (b, a)
            bc = (b, c) if b < c else (c, b)
            ca = (c, a) if c < a else (a, c)
            for pair in (ab, bc, ca):
                if pair not in edge_mid:
                    mid = (verts[pair[0]] + verts[pair[1]]) * 0.5
                    mid /= np.linalg.norm(mid)
                    edge_mid[pair] = n_verts + len(new_verts)
                    new_verts.append(mid)
            d, e, f = edge_mid[ab], edge_mid[bc], edge_mid[ca]
            new_faces.extend([a,d,f, d,b,e, f,e,c, d,e,f])
        if new_verts:
            verts = np.concatenate([verts, np.array(new_verts, dtype=np.float32)], axis=0)
        faces = np.array(new_faces, dtype=np.int32)
    edge_set = set()
    for i in range(0, len(faces), 3):
        a, b, c = int(faces[i]), int(faces[i+1]), int(faces[i+2])
        for pair in ((a,b),(b,c),(c,a)):
            edge_set.add((a,b) if a < b else (b,a))
    edge_list = np.array(list(edge_set), dtype=np.int32)
    result_starts = verts[edge_list[:, 0]]
    result_ends = verts[edge_list[:, 1]]
    _ICOSPHERE_CACHE[subdivisions] = (result_starts, result_ends)
    return result_starts, result_ends


@_register(GizmoType.ICOSPHERE)
def _build_icosphere(g: GizmoData):
    p = g.position; r = g.size
    starts, ends = _get_icosphere_edges(g.subdivisions)
    n = starts.shape[0]
    po = np.array([p[0], p[1], p[2]], dtype=np.float32)
    starts = starts * r + po
    ends = ends * r + po
    return starts, ends, _np_color(g.color, n)


def _get_label_render_data(g: GizmoData) -> Optional[dict]:
    if not g.text:
        return None
    return {
        'position': g.position,
        'text': g.text,
        'color': g.color,
        'font_size': g.font_size,
        'billboard': g.billboard,
        'world_space': g.world_space,
        'layer': g.layer,
        'duration': g.duration,
    }


@_register(GizmoType.LABEL)
def _build_label(g: GizmoData):
    return None


def _get_icon_render_data(g: GizmoData) -> Optional[dict]:
    if not g.text and not g.texture_path:
        return None
    return {
        'position': g.position,
        'text': g.text,
        'texture_path': g.texture_path,
        'color': g.color,
        'font_size': g.font_size,
        'billboard': g.billboard,
        'world_space': g.world_space,
        'layer': g.layer,
        'duration': g.duration,
    }


@_register(GizmoType.ICON)
def _build_icon(g: GizmoData):
    return None


@_register(GizmoType.TORUS)
def _build_torus(g: GizmoData):
    p = g.position; n = g.normal
    major = g.size; minor = g.inner_radius if g.inner_radius > 0 else major * 0.3
    segs_major = g.segments; segs_minor = max(g.segments // 2, 6)
    nv = Vec3(n[0], n[1], n[2])
    ref = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(ref).normalized()
    p2 = nv.cross(p1).normalized()
    theta = np.linspace(0, 2.0*math.pi, segs_major+1, dtype=np.float32)
    phi = np.linspace(0, 2.0*math.pi, segs_minor+1, dtype=np.float32)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    total = segs_major * segs_minor * 2
    starts = np.empty((total, 3), dtype=np.float32)
    ends = np.empty((total, 3), dtype=np.float32)
    idx = 0
    for i in range(segs_major):
        for j in range(segs_minor):
            cx = major * ct[i]; cy = 0; cz = major * st[i]
            for k in range(2):
                a = theta[i] if k == 0 else theta[i+1]
                ca, sa = math.cos(a), math.sin(a)
                rx = major + minor * cp[j]
                x = p1.x * rx * ca + p2.x * rx * sa + n[0] * (minor * sp[j])
                y = p1.y * rx * ca + p2.y * rx * sa + n[1] * (minor * sp[j])
                z = p1.z * rx * ca + p2.z * rx * sa + n[2] * (minor * sp[j])
                if k == 0:
                    starts[idx] = [p[0]+x, p[1]+y, p[2]+z]
                else:
                    ends[idx] = [p[0]+x, p[1]+y, p[2]+z]
                    idx += 1
    for i in range(segs_major):
        for j in range(segs_minor):
            a = theta[i]; ca, sa = math.cos(a), math.sin(a)
            rx = major + minor * cp[j]
            x0 = p1.x * rx * ca + p2.x * rx * sa + n[0] * minor * sp[j]
            y0 = p1.y * rx * ca + p2.y * rx * sa + n[1] * minor * sp[j]
            z0 = p1.z * rx * ca + p2.z * rx * sa + n[2] * minor * sp[j]
            rx2 = major + minor * cp[j+1]
            x1 = p1.x * rx2 * ca + p2.x * rx2 * sa + n[0] * minor * sp[j+1]
            y1 = p1.y * rx2 * ca + p2.y * rx2 * sa + n[1] * minor * sp[j+1]
            z1 = p1.z * rx2 * ca + p2.z * rx2 * sa + n[2] * minor * sp[j+1]
            starts[idx] = [p[0]+x0, p[1]+y0, p[2]+z0]
            ends[idx] = [p[0]+x1, p[1]+y1, p[2]+z1]
            idx += 1
    return starts, ends, _np_color(g.color, total)


@_register(GizmoType.PIPE)
def _build_pipe(g: GizmoData):
    p = g.position; n = g.normal; r = g.size
    h = g.height if g.height > 0 else 1.0
    inner = g.inner_radius
    segs = max(g.segments, 8)
    nv = Vec3(n[0], n[1], n[2])
    p1 = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(p1).normalized()
    p2 = nv.cross(p1).normalized()
    theta = np.linspace(0, 2.0*math.pi, segs+1, dtype=np.float32)
    ct, st = np.cos(theta), np.sin(theta)
    outer_bot = np.empty((segs+1, 3), dtype=np.float32)
    outer_top = np.empty((segs+1, 3), dtype=np.float32)
    inner_bot = np.empty((segs+1, 3), dtype=np.float32)
    inner_top = np.empty((segs+1, 3), dtype=np.float32)
    outer_bot[:,0] = p[0]+p1.x*ct*r+p2.x*st*r
    outer_bot[:,1] = p[1]+p1.y*ct*r+p2.y*st*r
    outer_bot[:,2] = p[2]+p1.z*ct*r+p2.z*st*r
    outer_top[:,0] = outer_bot[:,0]+n[0]*h; outer_top[:,1] = outer_bot[:,1]+n[1]*h; outer_top[:,2] = outer_bot[:,2]+n[2]*h
    if inner > 0:
        inner_bot[:,0] = p[0]+p1.x*ct*inner+p2.x*st*inner
        inner_bot[:,1] = p[1]+p1.y*ct*inner+p2.y*st*inner
        inner_bot[:,2] = p[2]+p1.z*ct*inner+p2.z*st*inner
        inner_top[:,0] = inner_bot[:,0]+n[0]*h; inner_top[:,1] = inner_bot[:,1]+n[1]*h; inner_top[:,2] = inner_bot[:,2]+n[2]*h
    total = segs * 4 + (segs * 2 if inner > 0 else 0)
    starts = np.empty((total, 3), dtype=np.float32); ends = np.empty((total, 3), dtype=np.float32)
    idx = 0
    for arr in (outer_bot, outer_top):
        starts[idx:idx+segs] = arr[:-1]; ends[idx:idx+segs] = arr[1:]; idx += segs
    starts[idx:idx+segs] = outer_bot[:-1]; ends[idx:idx+segs] = outer_top[:-1]; idx += segs
    if inner > 0:
        for arr in (inner_bot, inner_top):
            starts[idx:idx+segs] = arr[:-1]; ends[idx:idx+segs] = arr[1:]; idx += segs
        starts[idx:idx+segs] = inner_bot[:-1]; ends[idx:idx+segs] = inner_top[:-1]; idx += segs
    return starts, ends, _np_color(g.color, total)


@_register(GizmoType.STAR)
def _build_star(g: GizmoData):
    p = g.position; n = g.normal; outer = g.size; inner = g.inner_radius if g.inner_radius > 0 else outer * 0.4
    points = g.segments if g.segments >= 3 else 5
    nv = Vec3(n[0], n[1], n[2])
    p1 = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(p1).normalized()
    p2 = nv.cross(p1).normalized()
    total_pts = points * 2
    theta = np.linspace(0, 2.0*math.pi, total_pts+1, dtype=np.float32)[:-1]
    r_vals = np.empty(total_pts, dtype=np.float32)
    r_vals[0::2] = outer; r_vals[1::2] = inner
    ct = np.cos(theta) * r_vals; st = np.sin(theta) * r_vals
    pts = np.empty((total_pts, 3), dtype=np.float32)
    pts[:, 0] = p[0] + p1.x*ct + p2.x*st
    pts[:, 1] = p[1] + p1.y*ct + p2.y*st
    pts[:, 2] = p[2] + p1.z*ct + p2.z*st
    starts = pts; ends = np.roll(pts, -1, axis=0)
    return starts, ends, _np_color(g.color, total_pts)


@_register(GizmoType.PIE)
def _build_pie(g: GizmoData):
    p = g.position; n = g.normal; r = g.size
    a0 = math.radians(g.angle_start); a1 = math.radians(g.angle_end)
    segs = g.segments
    nv = Vec3(n[0], n[1], n[2])
    p1 = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(p1).normalized()
    p2 = nv.cross(p1).normalized()
    theta = np.linspace(a0, a1, segs+1, dtype=np.float32)
    ct = np.cos(theta)*r; st = np.sin(theta)*r
    arc = np.empty((segs+1, 3), dtype=np.float32)
    arc[:, 0] = p[0] + p1.x*ct + p2.x*st
    arc[:, 1] = p[1] + p1.y*ct + p2.y*st
    arc[:, 2] = p[2] + p1.z*ct + p2.z*st
    center = np.array([[p[0], p[1], p[2]]], dtype=np.float32)
    total = segs + 2
    starts = np.empty((total, 3), dtype=np.float32); ends = np.empty((total, 3), dtype=np.float32)
    starts[0] = center[0]; ends[0] = arc[0]
    starts[1:1+segs] = arc[:-1]; ends[1:1+segs] = arc[1:]
    starts[1+segs] = center[0]; ends[1+segs] = arc[-1]
    return starts, ends, _np_color(g.color, total)


@_register(GizmoType.WEDGE)
def _build_wedge(g: GizmoData):
    p = g.position; n = g.normal; r = g.size
    h = g.height if g.height > 0 else 1.0
    a0 = math.radians(g.angle_start); a1 = math.radians(g.angle_end)
    segs = g.segments
    nv = Vec3(n[0], n[1], n[2])
    p1 = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(p1).normalized()
    p2 = nv.cross(p1).normalized()
    theta = np.linspace(a0, a1, segs+1, dtype=np.float32)
    ct = np.cos(theta)*r; st = np.sin(theta)*r
    bot_arc = np.empty((segs+1, 3), dtype=np.float32)
    bot_arc[:, 0] = p[0] + p1.x*ct + p2.x*st
    bot_arc[:, 1] = p[1] + p1.y*ct + p2.y*st
    bot_arc[:, 2] = p[2] + p1.z*ct + p2.z*st
    top_arc = bot_arc + np.array([[n[0]*h, n[1]*h, n[2]*h]], dtype=np.float32)
    center_bot = np.array([[p[0], p[1], p[2]]], dtype=np.float32)
    center_top = center_bot + np.array([[n[0]*h, n[1]*h, n[2]*h]], dtype=np.float32)
    total = segs*2 + 6
    starts = np.empty((total, 3), dtype=np.float32); ends = np.empty((total, 3), dtype=np.float32)
    idx = 0
    starts[idx:idx+segs] = bot_arc[:-1]; ends[idx:idx+segs] = bot_arc[1:]; idx += segs
    starts[idx:idx+segs] = top_arc[:-1]; ends[idx:idx+segs] = top_arc[1:]; idx += segs
    starts[idx]=center_bot[0]; ends[idx]=bot_arc[0]; idx+=1
    starts[idx]=center_bot[0]; ends[idx]=bot_arc[-1]; idx+=1
    starts[idx]=center_top[0]; ends[idx]=top_arc[0]; idx+=1
    starts[idx]=center_top[0]; ends[idx]=top_arc[-1]; idx+=1
    starts[idx]=bot_arc[0]; ends[idx]=top_arc[0]; idx+=1
    starts[idx]=bot_arc[-1]; ends[idx]=top_arc[-1]; idx+=1
    return starts, ends, _np_color(g.color, total)


@_register(GizmoType.SPIRAL)
def _build_spiral(g: GizmoData):
    p = g.position; n = g.normal; max_r = g.size; h = g.height if g.height > 0 else 2.0
    turns = max(g.turns, 1); segs = g.segments
    nv = Vec3(n[0], n[1], n[2])
    ref = Vec3(0,1,0) if abs(n[1])<0.9 else Vec3(1,0,0)
    p1 = nv.cross(ref).normalized()
    p2 = nv.cross(p1).normalized()
    total = segs * int(turns)
    theta = np.linspace(0, 2.0*math.pi*turns, total+1, dtype=np.float32)
    radius = np.linspace(0, max_r, total+1, dtype=np.float32)
    ct = np.cos(theta)*radius; st = np.sin(theta)*radius
    vz = np.linspace(0, h, total+1, dtype=np.float32)
    pts = np.empty((total+1, 3), dtype=np.float32)
    pts[:, 0] = p[0] + p1.x*ct + p2.x*st + n[0]*vz
    pts[:, 1] = p[1] + p1.y*ct + p2.y*st + n[1]*vz
    pts[:, 2] = p[2] + p1.z*ct + p2.z*st + n[2]*vz
    return pts[:-1], pts[1:], _np_color(g.color, total)


@_register(GizmoType.CHORD)
def _build_chord(g: GizmoData):
    p = g.position; n = g.normal; r = g.size
    a0 = math.radians(g.angle_start); a1 = math.radians(g.angle_end)
    segs = g.segments
    nv = Vec3(n[0], n[1], n[2])
    p1 = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(p1).normalized()
    p2 = nv.cross(p1).normalized()
    theta = np.linspace(a0, a1, segs+1, dtype=np.float32)
    ct = np.cos(theta)*r; st = np.sin(theta)*r
    arc = np.empty((segs+1, 3), dtype=np.float32)
    arc[:, 0] = p[0] + p1.x*ct + p2.x*st
    arc[:, 1] = p[1] + p1.y*ct + p2.y*st
    arc[:, 2] = p[2] + p1.z*ct + p2.z*st
    starts = np.empty((segs+1, 3), dtype=np.float32); ends = np.empty((segs+1, 3), dtype=np.float32)
    starts[0] = arc[0]; ends[0] = arc[-1]
    starts[1:] = arc[:-1]; ends[1:] = arc[1:]
    return starts, ends, _np_color(g.color, segs+1)


@_register(GizmoType.HEMISPHERE)
def _build_hemisphere(g: GizmoData):
    p = g.position; n = g.normal; r = g.size; segs = max(g.segments // 2, 4)
    nv = Vec3(n[0], n[1], n[2])
    p1 = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(p1).normalized()
    p2 = nv.cross(p1).normalized()
    lats = np.linspace(0, math.pi*0.5, segs+1, dtype=np.float32)
    lons = np.linspace(0, 2.0*math.pi, segs+1, dtype=np.float32)
    verts = np.empty(((segs+1)*(segs+1), 3), dtype=np.float32)
    idx = 0
    for li in range(segs+1):
        la = lats[li]; cl = math.cos(la)*r; sl = math.sin(la)*r
        for lj in range(segs+1):
            lo = lons[lj]
            x = p1.x*cl*math.cos(lo) + p2.x*cl*math.sin(lo) + n[0]*sl
            y = p1.y*cl*math.cos(lo) + p2.y*cl*math.sin(lo) + n[1]*sl
            z = p1.z*cl*math.cos(lo) + p2.z*cl*math.sin(lo) + n[2]*sl
            verts[idx] = [p[0]+x, p[1]+y, p[2]+z]; idx += 1
    nv = segs+1
    lines_buf = []
    for li in range(segs):
        for lj in range(segs):
            i0=li*nv+lj; i1=li*nv+lj+1; i2=(li+1)*nv+lj; i3=(li+1)*nv+lj+1
            lines_buf.extend([i0,i1, i1,i2, i2,i3, i3,i0])
    lidx = np.array(lines_buf)
    return verts[lidx[0::2]], verts[lidx[1::2]], _np_color(g.color, len(lidx)//2)


@_register(GizmoType.PYRAMID)
def _build_pyramid(g: GizmoData):
    p = g.position; n = g.normal; r = g.size; h = g.height if g.height > 0 else r
    segs = max(g.segments, 4)
    nv = Vec3(n[0], n[1], n[2])
    ref = Vec3(1,0,0) if abs(n[1])<0.9 else Vec3(0,0,1)
    p1 = nv.cross(ref).normalized()
    p2 = nv.cross(p1).normalized()
    tip = np.array([p[0]+n[0]*h, p[1]+n[1]*h, p[2]+n[2]*h], dtype=np.float32)
    theta = np.linspace(0, 2.0*math.pi, segs+1, dtype=np.float32)[:-1]
    ct = np.cos(theta)*r; st = np.sin(theta)*r
    base = np.empty((segs, 3), dtype=np.float32)
    base[:, 0] = p[0] + p1.x*ct + p2.x*st
    base[:, 1] = p[1] + p1.y*ct + p2.y*st
    base[:, 2] = p[2] + p1.z*ct + p2.z*st
    total = segs * 2
    starts = np.empty((total, 3), dtype=np.float32); ends = np.empty((total, 3), dtype=np.float32)
    for i in range(segs):
        j = (i+1)%segs
        starts[i] = base[i]; ends[i] = base[j]
        starts[segs+i] = base[i]; ends[segs+i] = tip
    return starts, ends, _np_color(g.color, total)


class GizmosManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.draws: List[GizmoData] = []
        self.persistent_draws: List[GizmoData] = []
        self.unique_draws: Dict[str, GizmoData] = {}
        self.used_unique_keys: set = set()
        self.enabled: bool = True
        self.show_in_runtime: bool = True
        self._time: float = 0.0
        self._batches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._flat_starts = np.empty((0, 3), dtype=np.float32)
        self._flat_ends = np.empty((0, 3), dtype=np.float32)
        self._flat_colors = np.empty((0, 4), dtype=np.float32)
        self._flat_size = 0
        self._transform_stack: List[Optional[List[float]]] = []
        self._current_transform: Optional[List[float]] = None
        self._style_stack: List[dict] = []
        self._current_style: Optional[dict] = None
        self._revision: int = 0
        self._cached_revision: int = -1
        self._cache_starts: Optional[np.ndarray] = None
        self._cache_ends: Optional[np.ndarray] = None
        self._cache_colors: Optional[np.ndarray] = None

    def _resolve_style(self) -> dict:
        return dict(self._current_style) if self._current_style else {}

    def push_style(self, **kwargs):
        self._style_stack.append(self._current_style)
        base = dict(self._current_style) if self._current_style else {}
        base.update(kwargs)
        self._current_style = base

    def pop_style(self):
        if self._style_stack:
            self._current_style = self._style_stack.pop()

    def style(self, **kwargs):
        return _GizmoStyleCtx(self, kwargs)

    def update(self, dt: float):
        with self._lock:
            self._time += dt
            old_unique = len(self.unique_draws)
            for key in set(self.unique_draws.keys()) - self.used_unique_keys:
                del self.unique_draws[key]
            if len(self.unique_draws) != old_unique:
                self._revision += 1
            self.used_unique_keys.clear()

    def get_label_data(self) -> list[dict]:
        with self._lock:
            labels = []
            gizmos = list(self.persistent_draws)
            if self.unique_draws:
                gizmos.extend(self.unique_draws.values())
            for g in gizmos:
                if g.gizmo_type == GizmoType.LABEL and g.text:
                    data = _get_label_render_data(g)
                    if data:
                        labels.append(data)
            return labels

    def get_icon_data(self) -> list[dict]:
        with self._lock:
            icons = []
            gizmos = list(self.persistent_draws)
            if self.unique_draws:
                gizmos.extend(self.unique_draws.values())
            for g in gizmos:
                if g.gizmo_type == GizmoType.ICON:
                    data = _get_icon_render_data(g)
                    if data:
                        icons.append(data)
            return icons

    def _clear_cache(self):
        self._cache_starts = None
        self._cache_ends = None
        self._cache_colors = None
        self._cached_revision = -1

    def build_render_arrays(self):
        with self._lock:
            if self._revision == self._cached_revision:
                if self._cache_starts is not None:
                    return (self._cache_starts, self._cache_ends, self._cache_colors)
                return (None, None, None)
            gizmos = list(self.persistent_draws)
            if self.unique_draws:
                gizmos.extend(self.unique_draws.values())
            np_data_copies = None
            if self.enabled:
                np_data = self._get_render_data()
                if np_data is not None:
                    np_data_copies = (np.copy(np_data[0]), np.copy(np_data[1]), np.copy(np_data[2]))
            current_revision = self._revision
            self._batches.clear()
            self._flat_size = 0
        s_list = []
        e_list = []
        c_list = []
        if self.enabled:
            if np_data_copies is not None:
                s_list.append(np_data_copies[0])
                e_list.append(np_data_copies[1])
                c_list.append(np_data_copies[2])
            for g in gizmos:
                builder = _GIZMO_LINE_BUILDERS.get(g.gizmo_type)
                if builder is None:
                    continue
                try:
                    result = builder(g)
                except Exception:
                    continue
                if result is None:
                    continue
                s, e, c = result
                if g.line_style is not LineStyle.SOLID:
                    s, e, c = _apply_line_style(s, e, c, g.line_style, g.dash_length, g.gap_length)
                if s.shape[0] > 0:
                    s_list.append(s)
                    e_list.append(e)
                    c_list.append(c)
        if s_list:
            if len(s_list) == 1:
                self._cache_starts = s_list[0]
                self._cache_ends = e_list[0]
                self._cache_colors = c_list[0]
            else:
                self._cache_starts = np.concatenate(s_list)
                self._cache_ends = np.concatenate(e_list)
                self._cache_colors = np.concatenate(c_list)
        else:
            self._cache_starts = None
            self._cache_ends = None
            self._cache_colors = None
        self._cached_revision = current_revision
        return (self._cache_starts, self._cache_ends, self._cache_colors)

    def _clear_internal(self):
        self.persistent_draws.clear()
        self.draws.clear()
        self._batches.clear()
        self._flat_size = 0

    def clear(self):
        with self._lock:
            self._clear_internal()
            self._revision += 1
            self._clear_cache()

    def toggle(self, visible: bool = None):
        with self._lock:
            if visible is not None:
                self.enabled = visible
            else:
                self.enabled = not self.enabled

    def clear_persistent(self):
        with self._lock:
            self.persistent_draws.clear()
            self._revision += 1

    def clear_unique(self):
        with self._lock:
            self.unique_draws.clear()
            self._revision += 1

    def clear_tag(self, tag: str):
        with self._lock:
            old = len(self.persistent_draws)
            self.persistent_draws[:] = [g for g in self.persistent_draws if g.tag != tag]
            if len(self.persistent_draws) != old:
                self._revision += 1

    def draw_lines(self, starts: np.ndarray, ends: np.ndarray, colors: np.ndarray):
        if starts.shape[0] == 0:
            return
        with self._lock:
            old_sz = self._flat_size
            new_sz = old_sz + starts.shape[0]
            if new_sz > self._flat_starts.shape[0]:
                new_cap = int(new_sz * 1.5 + 4096)
                self._flat_starts = np.resize(self._flat_starts, (new_cap, 3))
                self._flat_ends = np.resize(self._flat_ends, (new_cap, 3))
                self._flat_colors = np.resize(self._flat_colors, (new_cap, 4))
            self._flat_starts[old_sz:new_sz] = starts
            self._flat_ends[old_sz:new_sz] = ends
            self._flat_colors[old_sz:new_sz] = colors
            self._flat_size = new_sz
            self._revision += 1

    def _get_render_data(self):
        if self._flat_size > 0:
            return (self._flat_starts[:self._flat_size],
                    self._flat_ends[:self._flat_size],
                    self._flat_colors[:self._flat_size])
        if self._batches:
            b = self._batches
            if len(b) == 1:
                return b[0]
            s_list, e_list, c_list = zip(*b)
            return (np.concatenate(s_list), np.concatenate(e_list), np.concatenate(c_list))
        return None

    def _add(self, g: GizmoData, **style_kw):
        s = self._resolve_style()
        s.update(style_kw)
        for k, v in s.items():
            if v is None:
                continue
            if k == 'color':
                g.color = self._resolve_color(v)
            elif hasattr(g, k):
                setattr(g, k, v)
        with self._lock:
            self.persistent_draws.append(g)
            self._revision += 1

    def _resolve_color(self, color) -> Tuple[float, float, float, float]:
        if isinstance(color, str):
            named = {
                'red': (1,0,0,1), 'green': (0,1,0,1), 'blue': (0,0,1,1),
                'white': (1,1,1,1), 'black': (0,0,0,1), 'yellow': (1,1,0,1),
                'cyan': (0,1,1,1), 'magenta': (1,0,1,1), 'gray': (0.5,0.5,0.5,1),
                'orange': (1,0.65,0,1), 'purple': (0.5,0,0.5,1),
                'pink': (1,0.41,0.71,1), 'brown': (0.65,0.16,0.16,1),
                'lime': (0,1,0,1), 'teal': (0,0.5,0.5,1), 'navy': (0,0,0.5,1),
                'maroon': (0.5,0,0,1), 'olive': (0.5,0.5,0,1), 'coral': (1,0.5,0.31,1),
                'gold': (1,0.84,0,1), 'silver': (0.75,0.75,0.75,1),
            }
            return named.get(color.lower(), (1,1,1,1))
        c = tuple(color)
        if len(c) == 3:
            return c + (1.0,)
        return c

    def _resolve_pos(self, pos) -> Tuple[float, float, float]:
        if isinstance(pos, Vec3):
            return (pos.x, pos.y, pos.z)
        return tuple(pos)

    def set_transform(self, position=None, rotation=None, scale=None):
        m = []
        if position:
            m.extend(self._resolve_pos(position))
        else:
            m.extend([0,0,0])
        if rotation:
            if isinstance(rotation, (tuple, list)) and len(rotation) == 4:
                m.extend(rotation)
            elif hasattr(rotation, 'x') and hasattr(rotation, 'y') and hasattr(rotation, 'z') and hasattr(rotation, 'w'):
                m.extend([rotation.x, rotation.y, rotation.z, rotation.w])
            else:
                m.extend([0,0,0,1])
        else:
            m.extend([0,0,0,1])
        if scale:
            m.extend(self._resolve_pos(scale))
        else:
            m.extend([1,1,1])
        self._current_transform = m

    def reset_transform(self):
        self._current_transform = None

    def push_transform(self, position=None, rotation=None, scale=None):
        self._transform_stack.append(self._current_transform)
        self.set_transform(position, rotation, scale)

    def pop_transform(self):
        if self._transform_stack:
            self._current_transform = self._transform_stack.pop()

    def apply_rotation(self, rot):
        if isinstance(rot, (tuple, list)):
            self._current_transform = self._current_transform or [0,0,0,0,0,1,1,1,1]
            self._current_transform[3:7] = rot
        elif hasattr(rot, 'x'):
            self._current_transform = self._current_transform or [0,0,0,0,0,1,1,1,1]
            self._current_transform[3:7] = [rot.x, rot.y, rot.z, rot.w]

    def _apply_transform(self, g: GizmoData) -> GizmoData:
        t = self._current_transform
        if t is None:
            return g
        if len(t) >= 7:
            g.rotation = tuple(t[3:7])
        return g

    def _draw_gizmo(self, gizmo_type, position=None, end_position=None, color='white',
                    size=1.0, thickness=1.0, duration=0.0, layer=0, world_space=True,
                    **kw):
        g = self._apply_transform(GizmoData(gizmo_type=gizmo_type,
            position=self._resolve_pos(position) if position is not None else (0,0,0),
            color=self._resolve_color(color), size=size, thickness=thickness,
            duration=duration, layer=layer, world_space=world_space, **kw))
        if end_position is not None:
            g.end_position = self._resolve_pos(end_position)
        self._add(g)

    def draw_point(self, position, color='white', size=3.0, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.POINT, position, color=color, size=size, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_line(self, start, end, color='white', thickness=1.0, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.LINE, start, end, color=color, thickness=thickness, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_circle(self, center, normal=(0,1,0), radius=1.0, color='white', filled=False, thickness=1.0, segments=32, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.CIRCLE, center, normal=self._resolve_pos(normal), size=radius, color=color, filled=filled, thickness=thickness, segments=segments, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_sphere(self, center, radius=1.0, color='white', filled=False, thickness=1.0, segments=32, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.SPHERE, center, size=radius, color=color, filled=filled, thickness=thickness, segments=segments, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_box(self, center, size=(1,1,1), color='white', filled=False, thickness=1.0, rotation=None, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.BOX, center, size=size, color=color, filled=filled, thickness=thickness, rotation=rotation, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_arc(self, center, normal=(0,1,0), radius=1.0, angle_start=0.0, angle_end=360.0, color='white', thickness=1.0, segments=32, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.ARC, center, normal=self._resolve_pos(normal), size=radius, color=color, angle_start=angle_start, angle_end=angle_end, segments=segments, thickness=thickness, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_capsule(self, start, end, radius=0.5, color='white', thickness=1.0, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.CAPSULE, start, end, size=radius, color=color, thickness=thickness, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_grid(self, center, normal=(0,1,0), size=10.0, divisions=10, color='white', thickness=1.0, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.GRID, center, normal=self._resolve_pos(normal), size=size, segments=divisions, color=color, thickness=thickness, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_cone(self, center, direction=(0,1,0), height=1.0, base_radius=0.5, color='white', filled=False, thickness=1.0, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.CONE, center, normal=self._resolve_pos(direction), size=base_radius, height=height, color=color, filled=filled, thickness=thickness, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_axis(self, origin, length=1.0, color='white', thickness=1.0, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.AXIS, origin, size=length, color=color, thickness=thickness, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_text(self, position, text, color='white', font_size=14, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.TEXT, position, text=text, color=color, font_size=font_size, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_ring(self, center, inner_radius=0.5, outer_radius=1.0, color='white', filled=False, thickness=1.0, segments=32, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.RING, center, size=outer_radius, inner_radius=inner_radius, color=color, filled=filled, thickness=thickness, segments=segments, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_arrow(self, start, end, color='white', thickness=2.0, arrow_size=0.0, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.ARROW, start, end, color=color, thickness=thickness, arrow_size=arrow_size, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_cross(self, center, size=1.0, color='white', thickness=1.0, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.CROSS, center, size=size, color=color, thickness=thickness, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_dashed_line(self, start, end, color='white', thickness=1.0, dash_length=0.3, gap_length=0.15, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.DASHED, start, end, color=color, thickness=thickness, dash_length=dash_length, gap_length=gap_length, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_bezier(self, points, color='white', thickness=1.0, segments=32, duration=0.0, layer=0, world_space=True, **kw):
        pts = [self._resolve_pos(p) for p in points]
        self._draw_gizmo(GizmoType.BEZIER, pts[0], points=pts, color=color, thickness=thickness, segments=segments, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_triangle(self, p0, p1, p2, color='white', filled=False, thickness=1.0, duration=0.0, layer=0, world_space=True, **kw):
        pts = [self._resolve_pos(p) for p in (p0, p1, p2)]
        self._draw_gizmo(GizmoType.TRIANGLE, pts[0], points=pts, color=color, filled=filled, thickness=thickness, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_poly(self, points, color='white', filled=False, thickness=1.0, duration=0.0, layer=0, world_space=True, **kw):
        pts = [self._resolve_pos(p) for p in points]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
        self._draw_gizmo(GizmoType.POLY, (cx, cy, cz), points=pts, color=color, filled=filled, thickness=thickness, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_cylinder(self, center, direction=(0,1,0), height=1.0, radius=0.5, color='white', filled=False, thickness=1.0, rotation=None, segments=32, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.CYLINDER, center, normal=self._resolve_pos(direction), size=radius, height=height, color=color, filled=filled, thickness=thickness, segments=segments, rotation=rotation, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_ellipse(self, center, normal=(0,1,0), radius_x=1.0, radius_y=0.5, color='white', thickness=1.0, segments=32, filled=False, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.ELLIPSE, center, normal=self._resolve_pos(normal), size=radius_x, radius_y=radius_y, color=color, thickness=thickness, segments=segments, filled=filled, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_rect(self, center, size=(1.0, 1.0), normal=(0, 0, 1), color='white', filled=False, thickness=1.0, rotation=None, duration=0.0, layer=0, world_space=True, cull_distance=-1.0, **kw):
        self._draw_gizmo(GizmoType.RECT, center, size=size, normal=self._resolve_pos(normal), color=color, filled=filled, thickness=thickness, rotation=rotation, duration=duration, layer=layer, world_space=world_space, cull_distance=cull_distance, **kw)

    def draw_ray(self, origin, direction, length=1.0, color='white', thickness=1.0, arrow_size=0.2, duration=0.0, layer=0, world_space=True, **kw):
        d = self._resolve_pos(direction)
        ln = math.sqrt(d[0]**2+d[1]**2+d[2]**2)
        if ln > 0:
            d = (d[0]/ln, d[1]/ln, d[2]/ln)
        self._draw_gizmo(GizmoType.RAY, origin, normal=d, size=length, color=color, thickness=thickness, arrow_size=arrow_size, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_bbox(self, min_point, max_point, color='white', thickness=1.0, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.BBOX, min_point=self._resolve_pos(min_point), max_point=self._resolve_pos(max_point), color=color, thickness=thickness, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_frustum(self, origin, direction, fov=60.0, aspect=1.5, near=0.1, far=10.0, color='white', thickness=1.0, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.FRUSTUM, origin, normal=self._resolve_pos(direction), size=aspect, fov=fov, near_plane=near, far_plane=far, color=color, thickness=thickness, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_helix(self, center, direction=(0,1,0), height=2.0, radius=0.5, turns=3.0, color='white', thickness=1.0, segments=64, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.HELIX, center, normal=self._resolve_pos(direction), size=radius, height=height, turns=turns, color=color, thickness=thickness, segments=segments, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_parabola(self, start, end, height=1.0, color='white', thickness=1.0, segments=32, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.PARABOLA, start, end, height=height, color=color, thickness=thickness, segments=segments, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_spline(self, points, color='white', thickness=1.0, segments=32, duration=0.0, layer=0, world_space=True, **kw):
        pts = [self._resolve_pos(p) for p in points]
        self._draw_gizmo(GizmoType.SPLINE, pts[0], points=pts, color=color, thickness=thickness, segments=segments, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_icosphere(self, center, radius=1.0, color='white', subdivisions=1, thickness=1.0, segments=32, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.ICOSPHERE, center, size=radius, color=color, subdivisions=subdivisions, thickness=thickness, segments=segments, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_label(self, position, text, color='white', font_size=14, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.LABEL, position, text=text, color=color, font_size=font_size, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_icon(self, position, text='', texture_path='', color='white', font_size=14, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.ICON, position, text=text, texture_path=texture_path, color=color, font_size=font_size, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_torus(self, center, normal=(0,1,0), major_radius=1.0, minor_radius=0.3, color='white', thickness=1.0, segments=32, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.TORUS, center, normal=self._resolve_pos(normal), size=major_radius, inner_radius=minor_radius, color=color, thickness=thickness, segments=segments, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_pipe(self, center, direction=(0,1,0), height=1.0, outer_radius=0.5, inner_radius=0.3, color='white', thickness=1.0, segments=32, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.PIPE, center, normal=self._resolve_pos(direction), size=outer_radius, inner_radius=inner_radius, height=height, color=color, thickness=thickness, segments=segments, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_star(self, center, normal=(0,1,0), outer_radius=1.0, inner_radius=0.4, points=5, color='white', thickness=1.0, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.STAR, center, normal=self._resolve_pos(normal), size=outer_radius, inner_radius=inner_radius, segments=points, color=color, thickness=thickness, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_pie(self, center, normal=(0,1,0), radius=1.0, angle_start=0.0, angle_end=90.0, color='white', thickness=1.0, segments=32, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.PIE, center, normal=self._resolve_pos(normal), size=radius, angle_start=angle_start, angle_end=angle_end, segments=segments, color=color, thickness=thickness, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_wedge(self, center, direction=(0,1,0), radius=1.0, height=1.0, angle_start=0.0, angle_end=90.0, color='white', thickness=1.0, segments=32, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.WEDGE, center, normal=self._resolve_pos(direction), size=radius, height=height, angle_start=angle_start, angle_end=angle_end, segments=segments, color=color, thickness=thickness, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_spiral(self, center, direction=(0,1,0), max_radius=1.0, height=2.0, turns=3.0, color='white', thickness=1.0, segments=64, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.SPIRAL, center, normal=self._resolve_pos(direction), size=max_radius, height=height, turns=turns, color=color, thickness=thickness, segments=segments, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_chord(self, center, normal=(0,1,0), radius=1.0, angle_start=0.0, angle_end=90.0, color='white', thickness=1.0, segments=32, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.CHORD, center, normal=self._resolve_pos(normal), size=radius, angle_start=angle_start, angle_end=angle_end, segments=segments, color=color, thickness=thickness, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_hemisphere(self, center, normal=(0,1,0), radius=1.0, color='white', thickness=1.0, segments=32, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.HEMISPHERE, center, normal=self._resolve_pos(normal), size=radius, color=color, thickness=thickness, segments=segments, duration=duration, layer=layer, world_space=world_space, **kw)

    def draw_pyramid(self, center, direction=(0,1,0), base_radius=0.5, height=1.0, sides=4, color='white', thickness=1.0, duration=0.0, layer=0, world_space=True, **kw):
        self._draw_gizmo(GizmoType.PYRAMID, center, normal=self._resolve_pos(direction), size=base_radius, height=height, segments=sides, color=color, thickness=thickness, duration=duration, layer=layer, world_space=world_space, **kw)

    @staticmethod
    def color_rgb(r: float, g: float, b: float, a: float = 1.0) -> Tuple[float, float, float, float]:
        return (r, g, b, a)

    @staticmethod
    def color_hex(hex_str: str) -> Tuple[float, float, float, float]:
        h = hex_str.lstrip('#')
        if len(h) == 6:
            r, g, b = int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255
            return (r, g, b, 1.0)
        elif len(h) == 8:
            r, g, b, a = int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255, int(h[6:8],16)/255
            return (r, g, b, a)
        return (1,1,1,1)

    @staticmethod
    def color_lerp(c1, c2, t: float) -> Tuple[float, float, float, float]:
        c1 = list(c1); c2 = list(c2)
        while len(c1) < 4: c1.append(1.0)
        while len(c2) < 4: c2.append(1.0)
        t = max(0, min(1, t))
        return tuple(a + (b-a)*t for a, b in zip(c1, c2))

    @staticmethod
    def color_random(alpha: float = 1.0) -> Tuple[float, float, float, float]:
        return (random.random(), random.random(), random.random(), alpha)

    @staticmethod
    def color_hsv(h: float, s: float, v: float, a: float = 1.0) -> Tuple[float, float, float, float]:
        h = h % 360
        c = v * s
        x = c * (1 - abs((h/60) % 2 - 1))
        m = v - c
        if h < 60: r,g,b=c,x,0
        elif h < 120: r,g,b=x,c,0
        elif h < 180: r,g,b=0,c,x
        elif h < 240: r,g,b=0,x,c
        elif h < 300: r,g,b=x,0,c
        else: r,g,b=c,0,x
        return (r+m, g+m, b+m, a)


_gizmos_instance: Optional[GizmosManager] = None

def get_gizmos():
    return _gizmos_instance
def set_gizmos(gm):
    global _gizmos_instance
    _gizmos_instance = gm


_PASSTHROUGH = {
    'clear', 'clear_persistent', 'clear_unique', 'clear_tag', 'update', 'toggle',
    'set_transform', 'reset_transform', 'push_transform', 'pop_transform',
    'push_style', 'pop_style',
}

def _delegate(method, *a, **kw):
    if _gizmos_instance:
        getattr(_gizmos_instance, method)(*a, **kw)

class _GizmosMeta(type):
    def __getattr__(cls, name):
        if name.startswith('draw_') or name in _PASSTHROUGH:
            return lambda *a, **kw: _delegate(name, *a, **kw)
        if name.startswith('color_') or name in ('style',):
            fn = getattr(GizmosManager, name, None)
            if fn:
                return staticmethod(fn)
        if name in ('show_in_runtime',):
            if _gizmos_instance:
                return _gizmos_instance.show_in_runtime
            return True
        raise AttributeError(f"Gizmos has no attribute '{name}'")

    def __setattr__(cls, name, value):
        if name in ('show_in_runtime',) and _gizmos_instance is not None:
            _gizmos_instance.show_in_runtime = value
            return
        super().__setattr__(name, value)

class Gizmos(metaclass=_GizmosMeta):
    @staticmethod
    def style(**kw):
        if _gizmos_instance:
            return _gizmos_instance.style(**kw)
        from contextlib import nullcontext
        return nullcontext()

    @staticmethod
    def color_rgb(*a, **kw):
        return GizmosManager.color_rgb(*a, **kw)

    @staticmethod
    def color_hex(*a, **kw):
        return GizmosManager.color_hex(*a, **kw)

    @staticmethod
    def color_lerp(*a, **kw):
        return GizmosManager.color_lerp(*a, **kw)

    @staticmethod
    def color_random(*a, **kw):
        return GizmosManager.color_random(*a, **kw)

    @staticmethod
    def color_hsv(*a, **kw):
        return GizmosManager.color_hsv(*a, **kw)

    SOLID = LineStyle.SOLID
    DASHED = LineStyle.DASHED
    DOTTED = LineStyle.DOTTED
    HIDDEN = LineStyle.HIDDEN
