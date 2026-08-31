# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun
from __future__ import annotations
import math
import time
import numpy as np
from core.ecs.ecs import Component, ComponentRegistry, InstancePrimitive, _get_engine
from core.maths.math3d import Vec3, Vec4
from core.components.inspector_meta import FieldType, InspectorField, ListElementField
from core.gizmo.api import GizmoType, GizmoData, LineStyle, _GIZMO_LINE_BUILDERS, _apply_line_style


def _vec3_to_tuple(v):
    if isinstance(v, Vec3):
        return (v.x, v.y, v.z)
    if isinstance(v, (list, tuple, np.ndarray)):
        return (float(v[0]), float(v[1]), float(v[2]))
    return (0.0, 0.0, 0.0)


def _color_to_tuple(c):
    if isinstance(c, Vec4):
        return (c.x, c.y, c.z, c.w)
    if isinstance(c, (list, tuple, np.ndarray)):
        if len(c) == 3:
            return (float(c[0]), float(c[1]), float(c[2]), 1.0)
        if len(c) >= 4:
            return (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
    return (1.0, 1.0, 1.0, 1.0)


def _sig_points(points):
    out = []
    for p in points:
        if isinstance(p, Vec3):
            out.append((p.x, p.y, p.z))
        else:
            out.append(tuple(float(v) for v in p))
    return tuple(out)


def _v3(v):
    if isinstance(v, Vec3):
        return v
    return Vec3(*_vec3_to_tuple(v))


def _v4(v):
    if isinstance(v, Vec4):
        return v
    t = _color_to_tuple(v)
    return Vec4(*t)


def _v3t(v):
    return _vec3_to_tuple(v)


def _v4t(v):
    return _color_to_tuple(v)


def _is_play_mode():
    try:
        return bool(getattr(_get_engine(), "play_mode", False))
    except Exception:
        return False


def _gizmo_visible(comp):
    try:
        if not bool(getattr(comp, "show_in_play", True)) and _is_play_mode():
            return False
    except Exception:
        pass
    return True


def _wm(tr):
    try:
        if tr is None:
            return np.eye(4, dtype=np.float32)
        d = tr.world_matrix._d
        return np.asarray(d, dtype=np.float32)
    except Exception:
        return np.eye(4, dtype=np.float32)


def _apply_world(starts, ends, colors, tr=None):
    wm = _wm(tr)
    r = wm[:3, :3]
    t = wm[3, :3]
    return (starts @ r + t, ends @ r + t, colors)


def _box_instance(center, size, tr=None):
    c = np.array(_vec3_to_tuple(center), dtype=np.float32)
    h = np.array(_vec3_to_tuple(size), dtype=np.float32) * 0.5
    wm = _wm(tr)
    rs = wm[:3, :3]
    combined = np.eye(4, dtype=np.float32)
    combined[:3, :3] = rs * h
    combined[:3, 3] = rs @ c + wm[3, :3]
    return combined


def _sphere_instance(center, radius, tr=None):
    c = np.array(_vec3_to_tuple(center), dtype=np.float32)
    wm = _wm(tr)
    rs = wm[:3, :3]
    r = float(radius)
    m = max(np.linalg.norm(rs[:, i]) for i in range(3))
    if m > 0.0:
        r *= m
    combined = np.eye(4, dtype=np.float32)
    combined[0, 0] = combined[1, 1] = combined[2, 2] = r
    combined[:3, 3] = rs @ c + wm[3, :3]
    return combined


def _sphere_at(pos, radius, color):
    p = np.array(_vec3_to_tuple(pos), dtype=np.float32)
    r = float(radius)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = m[1, 1] = m[2, 2] = r
    m[:3, 3] = p
    return InstancePrimitive('sphere', m.ravel('F'), list(_v4t(color)))


def _dash_animated(starts, ends, colors, dash_len, gap_len, phase):
    step = dash_len + gap_len
    if step <= 0.0:
        return (starts, ends, colors)
    n0 = starts.shape[0]
    if n0 == 0:
        return (starts, ends, colors)
    dirs = ends - starts
    lens = np.sqrt((dirs * dirs).sum(axis=1))
    ph = phase % step
    out_s = []
    out_e = []
    out_c = []
    for i in range(n0):
        ln = lens[i]
        if ln <= 1e-6:
            continue
        dx = dirs[i]
        col = colors[i]
        kmax = int((ln + step - ph) / step) + 1
        for k in range(-1, kmax):
            t0 = k * step + ph
            t1 = t0 + dash_len
            if t1 <= 0.0 or t0 >= ln:
                continue
            c0 = max(t0, 0.0)
            c1 = min(t1, ln)
            if c1 - c0 < 1e-6:
                continue
            out_s.append(starts[i] + dx * (c0 / ln))
            out_e.append(starts[i] + dx * (c1 / ln))
            out_c.append(col)
    if not out_s:
        return (np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), np.empty((0, 4), dtype=np.float32))
    return (np.stack(out_s), np.stack(out_e), np.stack(out_c))


def _build_primitive(gtype, pos, color, **kwargs):
    data = GizmoData(gizmo_type=gtype, position=_vec3_to_tuple(pos), color=_color_to_tuple(color), **kwargs)
    builder = _GIZMO_LINE_BUILDERS.get(gtype)
    if builder is None:
        return None
    result = builder(data)
    if result is None:
        return None
    starts, ends, colors = result
    style = kwargs.get("line_style", LineStyle.SOLID)
    if isinstance(style, str):
        try:
            style = LineStyle(style)
        except Exception:
            style = LineStyle.SOLID
    if style != LineStyle.SOLID:
        starts, ends, colors = _apply_line_style(starts, ends, colors, style, kwargs.get("dash_length", 0.3), kwargs.get("gap_length", 0.15))
        if starts.shape[0] == 0:
            return None
    return (starts, ends, colors)


@ComponentRegistry.register
class GizmoLine(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("start", "end", "color", "thickness", "show_in_play", "line_style")
    _gizmo_icon_color = (200, 200, 200)
    _gizmo_icon_label = "L"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("start", "Start", FieldType.VEC3),
            InspectorField("end", "End", FieldType.VEC3),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("line_style", "Line Style", FieldType.ENUM, enum_class=LineStyle),
        ]
    def __init__(self):
        super().__init__()
        self.start = Vec3(-1, 0, 0)
        self.end = Vec3(1, 0, 0)
        self.color = Vec4(1, 1, 1, 1)
        self.thickness = 1.0
        self.show_in_play = True
        self.line_style = LineStyle.SOLID
    def serialize(self):
        d = super().serialize()
        d.update({
            "start": _v3t(self.start), "end": _v3t(self.end),
            "color": _v4t(self.color), "thickness": self.thickness,
            "show_in_play": self.show_in_play,
            "line_style": self.line_style.value if isinstance(self.line_style, LineStyle) else str(self.line_style),
        })
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.start = Vec3(*data.get("start", [-1, 0, 0]))
        inst.end = Vec3(*data.get("end", [1, 0, 0]))
        inst.color = Vec4(*data.get("color", [1, 1, 1, 1]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        try:
            inst.line_style = LineStyle(data.get("line_style", "solid"))
        except Exception:
            inst.line_style = LineStyle.SOLID
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.LINE, _v3t(self.start), (*_v4t(self.color),),
                               end_position=_v3t(self.end), thickness=self.thickness, line_style=self.line_style)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoRay(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("direction", "length", "color", "thickness", "show_in_play", "arrow_size")
    _gizmo_icon_color = (200, 200, 50)
    _gizmo_icon_label = "R"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("direction", "Direction", FieldType.VEC3),
            InspectorField("length", "Length", FieldType.FLOAT, min_val=0.1, max_val=100.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("arrow_size", "Arrow Size", FieldType.FLOAT, min_val=0.0, max_val=1.0),
        ]
    def __init__(self):
        super().__init__()
        self.direction = Vec3(0, 1, 0)
        self.length = 2.0
        self.color = Vec4(1, 1, 0, 1)
        self.thickness = 1.0
        self.show_in_play = True
        self.arrow_size = 0.2
    def serialize(self):
        d = super().serialize()
        d.update({"direction": _v3t(self.direction), "length": self.length, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "arrow_size": self.arrow_size})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.direction = Vec3(*data.get("direction", [0, 1, 0]))
        inst.length = data.get("length", 2.0)
        inst.color = Vec4(*data.get("color", [1, 1, 0, 1]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.arrow_size = data.get("arrow_size", 0.2)
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        d = _v3(self.direction).normalized()
        res = _build_primitive(GizmoType.RAY, (0, 0, 0), (*_v4t(self.color),),
                               normal=(d.x, d.y, d.z), size=self.length, arrow_size=self.arrow_size, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoArrow(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("start", "end", "color", "thickness", "show_in_play", "arrow_size")
    _gizmo_icon_color = (50, 200, 50)
    _gizmo_icon_label = "A"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("start", "Start", FieldType.VEC3),
            InspectorField("end", "End", FieldType.VEC3),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("arrow_size", "Arrow Size", FieldType.FLOAT, min_val=0.0, max_val=1.0),
        ]
    def __init__(self):
        super().__init__()
        self.start = Vec3(0, 0, 0)
        self.end = Vec3(0, 2, 0)
        self.color = Vec4(0, 1, 0, 1)
        self.thickness = 1.0
        self.show_in_play = True
        self.arrow_size = 0.2
    def serialize(self):
        d = super().serialize()
        d.update({"start": _v3t(self.start), "end": _v3t(self.end), "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "arrow_size": self.arrow_size})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.start = Vec3(*data.get("start", [0, 0, 0]))
        inst.end = Vec3(*data.get("end", [0, 2, 0]))
        inst.color = Vec4(*data.get("color", [0, 1, 0, 1]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.arrow_size = data.get("arrow_size", 0.2)
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.ARROW, _v3t(self.start), (*_v4t(self.color),),
                               end_position=_v3t(self.end), arrow_size=self.arrow_size, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoCube(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "size", "color", "thickness", "show_in_play", "wireframe", "rotation")
    _gizmo_icon_color = (50, 150, 250)
    _gizmo_icon_label = "C"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("size", "Size", FieldType.VEC3),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("wireframe", "Wireframe", FieldType.BOOL),
            InspectorField("rotation", "Rotation", FieldType.VEC3),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.size = Vec3.one()
        self.color = Vec4(0.2, 0.6, 1.0, 0.6)
        self.thickness = 1.0
        self.show_in_play = True
        self.wireframe = True
        self.rotation = Vec3.zero()
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "size": _v3t(self.size), "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "wireframe": self.wireframe, "rotation": _v3t(self.rotation)})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.size = Vec3(*data.get("size", [1, 1, 1]))
        inst.color = Vec4(*data.get("color", [0.2, 0.6, 1.0, 0.6]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.wireframe = data.get("wireframe", True)
        inst.rotation = Vec3(*data.get("rotation", [0, 0, 0]))
        return inst
    def gizmo_instance_data(self):
        if self.wireframe or not _gizmo_visible(self):
            return None
        tr = self.transform
        if not tr:
            return None
        return InstancePrimitive('box', _box_instance(self.center, self.size, tr).ravel('F'), [*_v4t(self.color)])
    def gizmo_primitives(self):
        if not self.wireframe or not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.BOX, _v3t(self.center), (*_v4t(self.color),),
                               size=(*_v3t(self.size),), rotation=(*_v3t(self.rotation),), thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoSphere(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "radius", "color", "thickness", "show_in_play", "wireframe", "segments")
    _gizmo_icon_color = (250, 150, 50)
    _gizmo_icon_label = "S"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("radius", "Radius", FieldType.FLOAT, min_val=0.01, max_val=10.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("wireframe", "Wireframe", FieldType.BOOL),
            InspectorField("segments", "Segments", FieldType.INT, min_val=4, max_val=64),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.radius = 0.5
        self.color = Vec4(1.0, 0.6, 0.2, 0.6)
        self.thickness = 1.0
        self.show_in_play = True
        self.wireframe = True
        self.segments = 16
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "radius": self.radius, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "wireframe": self.wireframe, "segments": self.segments})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.radius = data.get("radius", 0.5)
        inst.color = Vec4(*data.get("color", [1, 0.6, 0.2, 0.6]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.wireframe = data.get("wireframe", True)
        inst.segments = data.get("segments", 16)
        return inst
    def gizmo_instance_data(self):
        if self.wireframe or not _gizmo_visible(self):
            return None
        tr = self.transform
        if not tr:
            return None
        return InstancePrimitive('sphere', _sphere_instance(self.center, self.radius, tr).ravel('F'), [*_v4t(self.color)])
    def gizmo_primitives(self):
        if not self.wireframe or not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.SPHERE, _v3t(self.center), (*_v4t(self.color),),
                               size=self.radius, segments=self.segments, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoCylinder(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "radius", "height", "color", "thickness", "show_in_play", "segments", "direction")
    _gizmo_icon_color = (100, 200, 100)
    _gizmo_icon_label = "Y"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("radius", "Radius", FieldType.FLOAT, min_val=0.01, max_val=10.0),
            InspectorField("height", "Height", FieldType.FLOAT, min_val=0.01, max_val=10.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("segments", "Segments", FieldType.INT, min_val=4, max_val=64),
            InspectorField("direction", "Direction", FieldType.VEC3),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.radius = 0.5
        self.height = 1.0
        self.color = Vec4(0.5, 1.0, 0.5, 0.6)
        self.thickness = 1.0
        self.show_in_play = True
        self.segments = 16
        self.direction = Vec3(0, 1, 0)
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "radius": self.radius, "height": self.height, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "segments": self.segments, "direction": _v3t(self.direction)})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.radius = data.get("radius", 0.5)
        inst.height = data.get("height", 1.0)
        inst.color = Vec4(*data.get("color", [0.5, 1, 0.5, 0.6]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.segments = data.get("segments", 16)
        inst.direction = Vec3(*data.get("direction", [0, 1, 0]))
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.CYLINDER, _v3t(self.center), (*_v4t(self.color),),
                               size=self.radius, height=self.height, normal=(*_v3t(self.direction),), segments=self.segments, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoCapsule(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "radius", "height", "color", "thickness", "show_in_play", "segments")
    _gizmo_icon_color = (100, 150, 220)
    _gizmo_icon_label = "P"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("radius", "Radius", FieldType.FLOAT, min_val=0.01, max_val=10.0),
            InspectorField("height", "Height", FieldType.FLOAT, min_val=0.01, max_val=10.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("segments", "Segments", FieldType.INT, min_val=4, max_val=64),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.radius = 0.5
        self.height = 1.0
        self.color = Vec4(0.5, 0.5, 1.0, 0.6)
        self.thickness = 1.0
        self.show_in_play = True
        self.segments = 16
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "radius": self.radius, "height": self.height, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "segments": self.segments})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.radius = data.get("radius", 0.5)
        inst.height = data.get("height", 1.0)
        inst.color = Vec4(*data.get("color", [0.5, 0.5, 1, 0.6]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.segments = data.get("segments", 16)
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.CAPSULE, _v3t(self.center), (*_v4t(self.color),),
                               end_position=(self.center.x, self.center.y + self.height, self.center.z), size=self.radius, segments=self.segments, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoPlane(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "normal", "size", "color", "thickness", "show_in_play")
    _gizmo_icon_color = (150, 150, 150)
    _gizmo_icon_label = "PL"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("normal", "Normal", FieldType.VEC3),
            InspectorField("size", "Size", FieldType.FLOAT, min_val=0.1, max_val=100.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.normal = Vec3(0, 1, 0)
        self.size = 5.0
        self.color = Vec4(0.8, 0.8, 0.8, 0.4)
        self.thickness = 1.0
        self.show_in_play = True
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "normal": _v3t(self.normal), "size": self.size, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.normal = Vec3(*data.get("normal", [0, 1, 0]))
        inst.size = data.get("size", 5.0)
        inst.color = Vec4(*data.get("color", [0.8, 0.8, 0.8, 0.4]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.GRID, _v3t(self.center), (*_v4t(self.color),),
                               normal=(*_v3t(self.normal),), size=self.size, segments=10, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoGrid(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "size", "divisions", "color", "thickness", "show_in_play", "normal")
    _gizmo_icon_color = (180, 180, 180)
    _gizmo_icon_label = "G"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("size", "Size", FieldType.FLOAT, min_val=1.0, max_val=100.0),
            InspectorField("divisions", "Divisions", FieldType.INT, min_val=2, max_val=50),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("normal", "Normal", FieldType.VEC3),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.size = 10.0
        self.divisions = 10
        self.color = Vec4(0.5, 0.5, 0.5, 0.3)
        self.thickness = 1.0
        self.show_in_play = True
        self.normal = Vec3(0, 1, 0)
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "size": self.size, "divisions": self.divisions, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "normal": _v3t(self.normal)})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.size = data.get("size", 10.0)
        inst.divisions = data.get("divisions", 10)
        inst.color = Vec4(*data.get("color", [0.5, 0.5, 0.5, 0.3]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.normal = Vec3(*data.get("normal", [0, 1, 0]))
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.GRID, _v3t(self.center), (*_v4t(self.color),),
                               normal=(*_v3t(self.normal),), size=self.size, segments=self.divisions, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoCircle(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "radius", "color", "thickness", "show_in_play", "normal", "segments")
    _gizmo_icon_color = (220, 180, 50)
    _gizmo_icon_label = "O"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("radius", "Radius", FieldType.FLOAT, min_val=0.01, max_val=10.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("normal", "Normal", FieldType.VEC3),
            InspectorField("segments", "Segments", FieldType.INT, min_val=8, max_val=64),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.radius = 1.0
        self.color = Vec4(1, 0.8, 0.2, 1.0)
        self.thickness = 1.0
        self.show_in_play = True
        self.normal = Vec3(0, 1, 0)
        self.segments = 32
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "radius": self.radius, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "normal": _v3t(self.normal), "segments": self.segments})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.radius = data.get("radius", 1.0)
        inst.color = Vec4(*data.get("color", [1, 0.8, 0.2, 1.0]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.normal = Vec3(*data.get("normal", [0, 1, 0]))
        inst.segments = data.get("segments", 32)
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.CIRCLE, _v3t(self.center), (*_v4t(self.color),),
                               normal=(*_v3t(self.normal),), size=self.radius, segments=self.segments, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoDisc(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "radius", "color", "thickness", "show_in_play", "normal")
    _gizmo_icon_color = (220, 180, 50)
    _gizmo_icon_label = "D"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("radius", "Radius", FieldType.FLOAT, min_val=0.01, max_val=10.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("normal", "Normal", FieldType.VEC3),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.radius = 1.0
        self.color = Vec4(1, 0.8, 0.2, 0.4)
        self.thickness = 1.0
        self.show_in_play = True
        self.normal = Vec3(0, 1, 0)
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "radius": self.radius, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "normal": _v3t(self.normal)})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.radius = data.get("radius", 1.0)
        inst.color = Vec4(*data.get("color", [1, 0.8, 0.2, 0.4]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.normal = Vec3(*data.get("normal", [0, 1, 0]))
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.CIRCLE, _v3t(self.center), (*_v4t(self.color),),
                               normal=(*_v3t(self.normal),), size=self.radius, segments=32, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoArc(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "radius", "color", "thickness", "show_in_play", "normal", "angle_start", "angle_end")
    _gizmo_icon_color = (220, 100, 100)
    _gizmo_icon_label = "Arc"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("radius", "Radius", FieldType.FLOAT, min_val=0.01, max_val=10.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("normal", "Normal", FieldType.VEC3),
            InspectorField("angle_start", "Angle Start", FieldType.FLOAT, min_val=0.0, max_val=360.0),
            InspectorField("angle_end", "Angle End", FieldType.FLOAT, min_val=0.0, max_val=360.0),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.radius = 1.0
        self.color = Vec4(1, 0.4, 0.4, 1.0)
        self.thickness = 1.0
        self.show_in_play = True
        self.normal = Vec3(0, 1, 0)
        self.angle_start = 0.0
        self.angle_end = 90.0
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "radius": self.radius, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "normal": _v3t(self.normal), "angle_start": self.angle_start, "angle_end": self.angle_end})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.radius = data.get("radius", 1.0)
        inst.color = Vec4(*data.get("color", [1, 0.4, 0.4, 1.0]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.normal = Vec3(*data.get("normal", [0, 1, 0]))
        inst.angle_start = data.get("angle_start", 0.0)
        inst.angle_end = data.get("angle_end", 90.0)
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.ARC, _v3t(self.center), (*_v4t(self.color),),
                               normal=(*_v3t(self.normal),), size=self.radius, angle_start=self.angle_start, angle_end=self.angle_end, segments=32, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoTorus(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "major_radius", "minor_radius", "color", "thickness", "show_in_play", "segments", "normal")
    _gizmo_icon_color = (150, 100, 220)
    _gizmo_icon_label = "T"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("major_radius", "Major Radius", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("minor_radius", "Minor Radius", FieldType.FLOAT, min_val=0.01, max_val=5.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("segments", "Segments", FieldType.INT, min_val=8, max_val=64),
            InspectorField("normal", "Normal", FieldType.VEC3),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.major_radius = 1.0
        self.minor_radius = 0.2
        self.color = Vec4(0.8, 0.4, 1.0, 0.8)
        self.thickness = 1.0
        self.show_in_play = True
        self.segments = 24
        self.normal = Vec3(0, 1, 0)
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "major_radius": self.major_radius, "minor_radius": self.minor_radius, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "segments": self.segments, "normal": _v3t(self.normal)})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.major_radius = data.get("major_radius", 1.0)
        inst.minor_radius = data.get("minor_radius", 0.2)
        inst.color = Vec4(*data.get("color", [0.8, 0.4, 1.0, 0.8]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.segments = data.get("segments", 24)
        inst.normal = Vec3(*data.get("normal", [0, 1, 0]))
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.TORUS, _v3t(self.center), (*_v4t(self.color),),
                               size=self.major_radius, inner_radius=self.minor_radius, normal=(*_v3t(self.normal),), segments=self.segments, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoCone(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "radius", "height", "color", "thickness", "show_in_play", "direction")
    _gizmo_icon_color = (250, 200, 100)
    _gizmo_icon_label = "Co"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("radius", "Radius", FieldType.FLOAT, min_val=0.01, max_val=10.0),
            InspectorField("height", "Height", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("direction", "Direction", FieldType.VEC3),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.radius = 0.5
        self.height = 1.0
        self.color = Vec4(1, 0.8, 0.3, 0.8)
        self.thickness = 1.0
        self.show_in_play = True
        self.direction = Vec3(0, 1, 0)
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "radius": self.radius, "height": self.height, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "direction": _v3t(self.direction)})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.radius = data.get("radius", 0.5)
        inst.height = data.get("height", 1.0)
        inst.color = Vec4(*data.get("color", [1, 0.8, 0.3, 0.8]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.direction = Vec3(*data.get("direction", [0, 1, 0]))
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.CONE, _v3t(self.center), (*_v4t(self.color),),
                               normal=(*_v3t(self.direction),), size=self.radius, height=self.height, segments=16, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoPyramid(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "base_size", "height", "color", "thickness", "show_in_play")
    _gizmo_icon_color = (200, 180, 80)
    _gizmo_icon_label = "Py"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("base_size", "Base Size", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("height", "Height", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.base_size = 1.0
        self.height = 1.0
        self.color = Vec4(0.9, 0.7, 0.2, 0.8)
        self.thickness = 1.0
        self.show_in_play = True
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "base_size": self.base_size, "height": self.height, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.base_size = data.get("base_size", 1.0)
        inst.height = data.get("height", 1.0)
        inst.color = Vec4(*data.get("color", [0.9, 0.7, 0.2, 0.8]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.PYRAMID, _v3t(self.center), (*_v4t(self.color),),
                               size=self.base_size, height=self.height, segments=4, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoFrustum(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "fov", "aspect", "near_plane", "far_plane", "color", "thickness", "show_in_play")
    _gizmo_icon_color = (100, 200, 220)
    _gizmo_icon_label = "F"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("fov", "FOV", FieldType.FLOAT, min_val=10.0, max_val=120.0),
            InspectorField("aspect", "Aspect", FieldType.FLOAT, min_val=0.1, max_val=3.0),
            InspectorField("near_plane", "Near", FieldType.FLOAT, min_val=0.01, max_val=10.0),
            InspectorField("far_plane", "Far", FieldType.FLOAT, min_val=1.0, max_val=100.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.fov = 60.0
        self.aspect = 1.5
        self.near_plane = 0.3
        self.far_plane = 10.0
        self.color = Vec4(0.4, 0.8, 1.0, 0.6)
        self.thickness = 1.0
        self.show_in_play = True
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "fov": self.fov, "aspect": self.aspect, "near_plane": self.near_plane, "far_plane": self.far_plane, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.fov = data.get("fov", 60.0)
        inst.aspect = data.get("aspect", 1.5)
        inst.near_plane = data.get("near_plane", 0.3)
        inst.far_plane = data.get("far_plane", 10.0)
        inst.color = Vec4(*data.get("color", [0.4, 0.8, 1.0, 0.6]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        base = tr.position if tr else Vec3.zero()
        pos = base + _v3(self.center)
        fwd = tr.forward if tr else Vec3(0, 0, 1)
        return _build_primitive(GizmoType.FRUSTUM, (pos.x, pos.y, pos.z), (*_v4t(self.color),),
                                normal=(fwd.x, fwd.y, fwd.z), fov=self.fov, size=self.aspect, near_plane=self.near_plane, far_plane=self.far_plane, thickness=self.thickness)


@ComponentRegistry.register
class GizmoBounds(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("min_point", "max_point", "color", "thickness", "dash_length", "gap_length", "corner_radius", "show_in_play")
    _gizmo_icon_color = (180, 180, 180)
    _gizmo_icon_label = "B"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("min_point", "Min", FieldType.VEC3),
            InspectorField("max_point", "Max", FieldType.VEC3),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("dash_length", "Dash Length", FieldType.FLOAT, min_val=0.01, max_val=2.0),
            InspectorField("gap_length", "Gap Length", FieldType.FLOAT, min_val=0.01, max_val=2.0),
            InspectorField("corner_radius", "Corner Radius", FieldType.FLOAT, min_val=0.01, max_val=1.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
        ]
    def __init__(self):
        super().__init__()
        self.min_point = Vec3(-1, -1, -1)
        self.max_point = Vec3(1, 1, 1)
        self.color = Vec4(0.25, 0.55, 1.0, 0.85)
        self.thickness = 1.0
        self.dash_length = 0.3
        self.gap_length = 0.15
        self.corner_radius = 0.08
        self.show_in_play = True
    def serialize(self):
        d = super().serialize()
        d.update({"min_point": _v3t(self.min_point), "max_point": _v3t(self.max_point), "color": _v4t(self.color), "thickness": self.thickness, "dash_length": self.dash_length, "gap_length": self.gap_length, "corner_radius": self.corner_radius, "show_in_play": self.show_in_play})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.min_point = Vec3(*data.get("min_point", [-1, -1, -1]))
        inst.max_point = Vec3(*data.get("max_point", [1, 1, 1]))
        inst.color = Vec4(*data.get("color", [0.25, 0.55, 1.0, 0.85]))
        inst.thickness = data.get("thickness", 1.0)
        inst.dash_length = data.get("dash_length", 0.3)
        inst.gap_length = data.get("gap_length", 0.15)
        inst.corner_radius = data.get("corner_radius", 0.08)
        inst.show_in_play = data.get("show_in_play", True)
        return inst
    def gizmo_cache_sig(self):
        return None
    def gizmo_instance_data(self):
        return None
    def gizmo_instances(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        if tr is None:
            return None
        wm = _wm(tr)
        rs = wm[:3, :3]
        t = wm[3, :3]
        mn = _v3(self.min_point)
        mx = _v3(self.max_point)
        corners = np.array([
            [mn.x, mn.y, mn.z], [mx.x, mn.y, mn.z], [mn.x, mx.y, mn.z], [mx.x, mx.y, mn.z],
            [mn.x, mn.y, mx.z], [mx.x, mn.y, mx.z], [mn.x, mx.y, mx.z], [mx.x, mx.y, mx.z],
        ], dtype=np.float32)
        world = corners @ rs + t
        r = max(float(self.corner_radius), 0.001)
        ms = max((np.linalg.norm(rs[:, i]) for i in range(3)), default=1.0)
        if ms <= 0.0:
            ms = 1.0
        radius = r * ms
        return [_sphere_at(w, radius, self.color) for w in world]
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        res = _build_primitive(GizmoType.BBOX, (0, 0, 0), (*_v4t(self.color),),
                               min_point=(*_v3t(self.min_point),), max_point=(*_v3t(self.max_point),), thickness=self.thickness)
        if res is None:
            return None
        s, e, c = _apply_world(*res, self.transform)
        dlen = max(float(self.dash_length), 0.01)
        glen = max(float(self.gap_length), 0.01)
        phase = time.time() * 1.5 * (dlen + glen)
        s, e, c = _dash_animated(s, e, c, dlen, glen, phase)
        if s.shape[0] == 0:
            return None
        return (s, e, c)


@ComponentRegistry.register
class GizmoCross(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "size", "color", "thickness", "show_in_play")
    _gizmo_icon_color = (200, 50, 50)
    _gizmo_icon_label = "X"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("size", "Size", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.size = 1.0
        self.color = Vec4(1, 0.2, 0.2, 1.0)
        self.thickness = 1.0
        self.show_in_play = True
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "size": self.size, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.size = data.get("size", 1.0)
        inst.color = Vec4(*data.get("color", [1, 0.2, 0.2, 1.0]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.CROSS, _v3t(self.center), (*_v4t(self.color),),
                               size=self.size, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoPoint(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "size", "color", "thickness", "show_in_play")
    _gizmo_icon_color = (250, 250, 50)
    _gizmo_icon_label = "P"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("size", "Size", FieldType.FLOAT, min_val=0.01, max_val=5.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.size = 0.1
        self.color = Vec4(1, 1, 0, 1)
        self.thickness = 3.0
        self.show_in_play = True
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "size": self.size, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.size = data.get("size", 0.1)
        inst.color = Vec4(*data.get("color", [1, 1, 0, 1]))
        inst.thickness = data.get("thickness", 3.0)
        inst.show_in_play = data.get("show_in_play", True)
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.CROSS, _v3t(self.center), (*_v4t(self.color),),
                               size=self.size * 2.0, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoPoly(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("color", "thickness", "show_in_play", "closed", "points")
    _gizmo_icon_color = (100, 200, 200)
    _gizmo_icon_label = "Poly"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("points", "Points", FieldType.LIST, element_fields=[ListElementField("point", "Point", FieldType.VEC3)]),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("closed", "Closed", FieldType.BOOL),
        ]
    def __init__(self):
        super().__init__()
        self.points = [Vec3(-1, 0, 0), Vec3(0, 1, 0), Vec3(1, 0, 0)]
        self.color = Vec4(0.5, 0.8, 1.0, 1.0)
        self.thickness = 1.0
        self.show_in_play = True
        self.closed = True
    def serialize(self):
        d = super().serialize()
        d.update({"points": [_v3t(p) for p in self.points], "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "closed": self.closed})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.points = [Vec3(*p) for p in data.get("points", [[-1, 0, 0], [0, 1, 0], [1, 0, 0]])]
        inst.color = Vec4(*data.get("color", [0.5, 0.8, 1.0, 1.0]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.closed = data.get("closed", True)
        return inst
    def gizmo_cache_sig(self):
        tr = self.transform
        if tr is None:
            return None
        try:
            return (tr.world_matrix._d.tobytes(), _sig_points(self.points), tuple(_v4t(self.color)), self.thickness, self.closed)
        except Exception:
            return None
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        pts = [_v3t(p) for p in self.points]
        res = _build_primitive(GizmoType.POLY, (0, 0, 0), (*_v4t(self.color),),
                               points=pts, filled=not self.closed, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoBezier(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("points", "color", "thickness", "show_in_play", "segments")
    _gizmo_icon_color = (180, 100, 220)
    _gizmo_icon_label = "Bz"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("points", "Points", FieldType.LIST, element_fields=[ListElementField("point", "Point", FieldType.VEC3)]),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("segments", "Segments", FieldType.INT, min_val=4, max_val=64),
        ]
    def __init__(self):
        super().__init__()
        self.points = [Vec3(-1, 0, 0), Vec3(-0.5, 1, 0), Vec3(0.5, 1, 0), Vec3(1, 0, 0)]
        self.color = Vec4(0.8, 0.5, 1.0, 1.0)
        self.thickness = 1.0
        self.show_in_play = True
        self.segments = 20
    def serialize(self):
        d = super().serialize()
        d.update({"points": [_v3t(p) for p in self.points], "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "segments": self.segments})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        if "points" in data:
            pts = data.get("points", [])
        else:
            pts = []
            for key in ("p0", "p1", "p2", "p3"):
                if key in data:
                    pts.append(data[key])
            if not pts:
                pts = [[-1, 0, 0], [-0.5, 1, 0], [0.5, 1, 0], [1, 0, 0]]
        inst.points = [Vec3(*p) for p in pts]
        inst.color = Vec4(*data.get("color", [0.8, 0.5, 1.0, 1.0]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.segments = data.get("segments", 20)
        return inst
    def gizmo_cache_sig(self):
        tr = self.transform
        if tr is None:
            return None
        try:
            return (tr.world_matrix._d.tobytes(), _sig_points(self.points), tuple(_v4t(self.color)), self.thickness, self.segments)
        except Exception:
            return None
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        pts = [_v3t(p) for p in self.points]
        res = _build_primitive(GizmoType.BEZIER, (0, 0, 0), (*_v4t(self.color),),
                               points=pts, segments=self.segments, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoStar(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "outer_radius", "inner_radius", "color", "thickness", "show_in_play", "points", "normal")
    _gizmo_icon_color = (255, 215, 0)
    _gizmo_icon_label = "St"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("outer_radius", "Outer", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("inner_radius", "Inner", FieldType.FLOAT, min_val=0.01, max_val=10.0),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("points", "Points", FieldType.INT, min_val=3, max_val=12),
            InspectorField("normal", "Normal", FieldType.VEC3),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.outer_radius = 1.0
        self.inner_radius = 0.4
        self.color = Vec4(1, 0.85, 0.0, 1.0)
        self.thickness = 1.0
        self.show_in_play = True
        self.points = 5
        self.normal = Vec3(0, 1, 0)
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "outer_radius": self.outer_radius, "inner_radius": self.inner_radius, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "points": self.points, "normal": _v3t(self.normal)})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.outer_radius = data.get("outer_radius", 1.0)
        inst.inner_radius = data.get("inner_radius", 0.4)
        inst.color = Vec4(*data.get("color", [1, 0.85, 0.0, 1.0]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.points = data.get("points", 5)
        inst.normal = Vec3(*data.get("normal", [0, 1, 0]))
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.STAR, _v3t(self.center), (*_v4t(self.color),),
                               size=self.outer_radius, inner_radius=self.inner_radius, normal=(*_v3t(self.normal),), segments=self.points, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoIcon(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "icon", "color", "size", "thickness", "show_in_play")
    _gizmo_icon_color = (100, 100, 100)
    _gizmo_icon_label = "Ic"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("icon", "Icon", FieldType.STRING),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("size", "Size", FieldType.FLOAT, min_val=0.1, max_val=5.0),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3.zero()
        self.icon = "star"
        self.color = Vec4(1, 1, 1, 1)
        self.thickness = 1.0
        self.show_in_play = True
        self.size = 1.0
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "icon": self.icon, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "size": self.size})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 0, 0]))
        inst.icon = data.get("icon", "star")
        inst.color = Vec4(*data.get("color", [1, 1, 1, 1]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.size = data.get("size", 1.0)
        return inst
    def gizmo_primitives(self):
        if not _gizmo_visible(self):
            return None
        tr = self.transform
        res = _build_primitive(GizmoType.CROSS, _v3t(self.center), (*_v4t(self.color),),
                               size=self.size, thickness=self.thickness)
        if res is None:
            return None
        return _apply_world(*res, tr)


@ComponentRegistry.register
class GizmoLabel(Component):
    _gizmo_pass = "gizmo"
    _gizmo_cache_attrs = ("center", "text", "color", "font_size", "thickness", "show_in_play")
    _gizmo_icon_color = (200, 200, 200)
    _gizmo_icon_label = "Lb"
    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("text", "Text", FieldType.STRING),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.1, max_val=10.0),
            InspectorField("show_in_play", "Show in Play", FieldType.BOOL),
            InspectorField("font_size", "Font Size", FieldType.INT, min_val=8, max_val=48),
        ]
    def __init__(self):
        super().__init__()
        self.center = Vec3(0, 1, 0)
        self.text = "Label"
        self.color = Vec4(1, 1, 1, 1)
        self.thickness = 1.0
        self.show_in_play = True
        self.font_size = 14
    def serialize(self):
        d = super().serialize()
        d.update({"center": _v3t(self.center), "text": self.text, "color": _v4t(self.color), "thickness": self.thickness, "show_in_play": self.show_in_play, "font_size": self.font_size})
        return d
    @classmethod
    def deserialize(cls, data):
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.center = Vec3(*data.get("center", [0, 1, 0]))
        inst.text = data.get("text", "Label")
        inst.color = Vec4(*data.get("color", [1, 1, 1, 1]))
        inst.thickness = data.get("thickness", 1.0)
        inst.show_in_play = data.get("show_in_play", True)
        inst.font_size = data.get("font_size", 14)
        return inst
    def gizmo_primitives(self):
        return None