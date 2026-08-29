# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional, List
import math

from core.ecs.ecs import Component, ComponentRegistry
from core.maths.math3d import Vec3, Quat
from core.components.inspector_meta import (
    InspectorField,
    FieldType,
    ListElementField,
)
from core.components.animation.animation_component import _resolve_bone_path

_IK_TYPES = ("TwoBoneIK", "FABRIKChain")


def _as_vec3(value):
    if value is None or isinstance(value, Vec3):
        return value
    try:
        return Vec3(float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        return None


def _find_named_descendant(root, name):
    """DFS for an entity named `name` anywhere below `root`."""
    if root is None or not name:
        return None
    stack = [root]
    while stack:
        node = stack.pop()
        children = node._children
        if children:
            stack.extend(children)
        if node is not root and node.name == name:
            return node
    return None


class _IKBase(Component):
    _updates: bool = True
    _gizmo_icon_color: tuple[int, int, int] = (120, 200, 255)
    _gizmo_icon_label: str = "ik"

    def __init__(self):
        super().__init__()
        self.target_entity_id: Optional[str] = None
        self.target_position: Optional[Vec3] = None
        self.pole_entity_id: Optional[str] = None
        self.pole_position: Optional[Vec3] = None
        self.weight: float = 1.0

    # -- entity/path helpers -------------------------------------------------

    def _resolve_bone(self, path: str):
        return _resolve_bone_path(self._entity, path)

    def _resolve_entity(self, ent_id):
        if not ent_id or not self._entity:
            return None
        scene = self._entity._scene
        if scene is None:
            return None
        return scene.get_entity(ent_id)

    def _target_world(self):
        tgt = self._resolve_entity(self.target_entity_id)
        if tgt is not None and tgt.transform is not None:
            return tgt.transform.position
        return _as_vec3(self.target_position)

    def _pole_world(self):
        tgt = self._resolve_entity(self.pole_entity_id)
        if tgt is not None and tgt.transform is not None:
            return tgt.transform.position
        return _as_vec3(self.pole_position)

    # -- paint / gizmo -------------------------------------------------------

    def gizmo_lines(self):
        lines = []
        pts = self._chain_points()
        if len(pts) >= 2:
            base = list(pts)
            for i in range(len(base) - 1):
                lines.append((base[i], base[i + 1], [0.47, 0.78, 1.0, 1.0]))
        target = self._target_world()
        if target is not None and pts:
            lines.append((pts[-1], target, [1.0, 0.65, 0.2, 1.0]))
        return lines

    def _chain_points(self) -> List[Vec3]:
        pts = []
        for e in self._bones():
            if e is not None and e.transform is not None:
                pts.append(e.transform.position)
        return pts

    def _bones(self) -> List:
        return []

    # -- serialization -------------------------------------------------------

    def _base_serialize(self, data: dict) -> dict:
        data["target_entity_id"] = self.target_entity_id
        data["target_position"] = self.target_position.to_list() if self.target_position else None
        data["pole_entity_id"] = self.pole_entity_id
        data["pole_position"] = self.pole_position.to_list() if self.pole_position else None
        data["weight"] = self.weight
        return data

    @classmethod
    def _base_deserialize(cls, inst, data: dict):
        inst.target_entity_id = data.get("target_entity_id")
        inst.target_position = _as_vec3(data.get("target_position"))
        inst.pole_entity_id = data.get("pole_entity_id")
        inst.pole_position = _as_vec3(data.get("pole_position"))
        inst.weight = float(data.get("weight", 1.0))
        return inst


@ComponentRegistry.register
class TwoBoneIK(_IKBase):
    """Analytic two-bone inverse kinematics (arm/leg style).

    root_bone / mid_bone / tip_bone are bone paths relative to the component's
    entity (Unity style, e.g. "Armature/Hips/Spine/Chest"). Empty mid_bone /
    tip_bone auto-derive the chain from the first bone children.
    """

    def __init__(self):
        super().__init__()
        self.root_bone: str = ""
        self.mid_bone: str = ""
        self.tip_bone: str = ""
        self.bend_positive: bool = True
        self.stretch: float = 0.0

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("root_bone", "Root Bone", FieldType.STRING),
            InspectorField("mid_bone", "Mid Bone", FieldType.STRING),
            InspectorField("tip_bone", "Tip Bone", FieldType.STRING),
            InspectorField("target_entity_id", "Target", FieldType.GAMEOBJECT),
            InspectorField("target_position", "Target Pos", FieldType.VEC3),
            InspectorField("pole_entity_id", "Pole", FieldType.GAMEOBJECT),
            InspectorField("pole_position", "Pole Pos", FieldType.VEC3),
            InspectorField("bend_positive", "Bend Positive", FieldType.BOOL),
            InspectorField("stretch", "Stretch", FieldType.FLOAT, 0.0, 1.0, 0.01, 2),
            InspectorField("weight", "Weight", FieldType.FLOAT, 0.0, 1.0, 0.01, 2),
        ]

    def _bones(self):
        ent = self._entity
        if ent is None:
            return []
        root = self._resolve_bone(self.root_bone)
        if root is None:
            return []
        mid = self._resolve_mid_or_tip(root, self.mid_bone)
        if mid is None:
            mid = root._children[0] if root._children else None
        tip = None
        if mid is not None:
            tip = self._resolve_mid_or_tip(root, self.tip_bone)
            if tip is None:
                tip = mid._children[0] if mid._children else None
        return [root, mid, tip]

    def _resolve_mid_or_tip(self, root, path):
        """Mirror Cython _resolve_mid_or_tip: path walk, then descendant DFS."""
        if not path:
            return None
        ent = self._entity
        last = path.split("/")[-1]
        node = self._resolve_bone(path) if ent is not None else None
        if (node is not None and node is not ent and node.name == last
                and last != (ent.name or "")):
            return node
        return _find_named_descendant(root, last) if root is not None else None

    def on_update(self, dt: float):
        try:
            from core._ik import batch_update_ik
            batch_update_ik([self], dt)
            return
        except ImportError:
            pass
        self._solve_python()

    def _solve_python(self):
        bones = self._bones()
        if len(bones) < 3 or any(b is None or b.transform is None for b in bones):
            return
        a = bones[0].transform.position
        b = bones[1].transform.position
        c = bones[2].transform.position
        target = self._target_world()
        if target is None:
            return
        pole = self._pole_world()
        bend = 1.0 if self.bend_positive else -1.0
        b_new, _, _ = _two_bone_solve_py(a, b, c, target, pole, self.stretch, bend)
        # deltas in the world frame, applied with the Cython _apply_world_delta
        # formula local = local_cur * conj(world_cur) * qd * world_cur
        qd_root = _quat_from_two_vecs(
            (b.x - a.x, b.y - a.y, b.z - a.z),
            (b_new[0] - a.x, b_new[1] - a.y, b_new[2] - a.z),
        )
        _apply_world_delta_py(bones[0].transform, qd_root, self.weight)
        # mid delta must come from the CURRENT (post-root) tip direction
        b = bones[1].transform.position
        c = bones[2].transform.position
        qd_mid = _quat_from_two_vecs(
            (c.x - b.x, c.y - b.y, c.z - b.z),
            (target.x - b_new[0], target.y - b_new[1], target.z - b_new[2]),
        )
        _apply_world_delta_py(bones[1].transform, qd_mid, self.weight)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update(self._base_serialize({}))
        d.update({
            "root_bone": self.root_bone,
            "mid_bone": self.mid_bone,
            "tip_bone": self.tip_bone,
            "bend_positive": self.bend_positive,
            "stretch": self.stretch,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> TwoBoneIK:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        _IKBase._base_deserialize(inst, data)
        inst.root_bone = data.get("root_bone", "")
        inst.mid_bone = data.get("mid_bone", "")
        inst.tip_bone = data.get("tip_bone", "")
        inst.bend_positive = data.get("bend_positive", True)
        inst.stretch = float(data.get("stretch", 0.0))
        return inst


@ComponentRegistry.register
class FABRIKChain(_IKBase):
    """Iterative multi-bone IK (FABRIK, full body limbs / tails / spines).

    bones lists explicit bone paths relative to the entity. When empty, the
    chain is auto-walked from root_bone down first children for chain_length
    joints. pole steers the bend plane; per-bone weight via bone_weights.
    """

    def __init__(self):
        super().__init__()
        self.root_bone: str = ""
        self.bones: List[str] = []
        self.chain_length: int = 0
        self.iterations: int = 8
        self.tolerance: float = 1e-4
        self.bone_weights: List[float] = []

    @classmethod
    def _inspector_fields(cls):
        return [
            InspectorField("root_bone", "Root Bone", FieldType.STRING),
            InspectorField(
                "bones",
                "Bones",
                FieldType.LIST,
                element_fields=[
                    ListElementField("", "Bone", FieldType.STRING),
                ],
            ),
            InspectorField("chain_length", "Chain Length", FieldType.INT),
            InspectorField("target_entity_id", "Target", FieldType.GAMEOBJECT),
            InspectorField("target_position", "Target Pos", FieldType.VEC3),
            InspectorField("pole_entity_id", "Pole", FieldType.GAMEOBJECT),
            InspectorField("pole_position", "Pole Pos", FieldType.VEC3),
            InspectorField("iterations", "Iterations", FieldType.INT),
            InspectorField("tolerance", "Tolerance", FieldType.FLOAT),
            InspectorField(
                "bone_weights",
                "Bone Weights",
                FieldType.LIST,
                element_fields=[ListElementField("", "Weight", FieldType.FLOAT, 0.0, 1.0, 0.01, 2)],
            ),
            InspectorField("weight", "Weight", FieldType.FLOAT, 0.0, 1.0, 0.01, 2),
        ]

    def _bones(self):
        ent = self._entity
        if ent is None:
            return []
        root = self._resolve_bone(self.root_bone)
        if root is None:
            return []
        if self.bones:
            return [self._resolve_bone(p) for p in self.bones]
        out = [root]
        cur = root
        for _ in range(1, max(self.chain_length, 1)):
            if not cur._children:
                break
            cur = cur._children[0]
            out.append(cur)
        return out

    def on_update(self, dt: float):
        try:
            from core._ik import batch_update_ik
            batch_update_ik([self], dt)
            return
        except ImportError:
            pass
        self._solve_python()

    def _solve_python(self):
        bones = [b for b in self._bones()]
        if len(bones) < 2 or any(b is None or b.transform is None for b in bones):
            return
        target = self._target_world()
        if target is None:
            return
        pts = np_points([b.transform.position for b in bones])
        res = _fabrik_positions_py(pts, target, self.iterations, self.tolerance)
        pole = self._pole_world()
        _apply_fabrik_py(bones, res, pole, self.weight, self.bone_weights)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update(self._base_serialize({}))
        d.update({
            "root_bone": self.root_bone,
            "bones": list(self.bones),
            "chain_length": self.chain_length,
            "iterations": self.iterations,
            "tolerance": self.tolerance,
            "bone_weights": list(self.bone_weights),
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> FABRIKChain:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        _IKBase._base_deserialize(inst, data)
        inst.root_bone = data.get("root_bone", "")
        inst.bones = list(data.get("bones", []))
        inst.chain_length = int(data.get("chain_length", 0))
        inst.iterations = int(data.get("iterations", 8))
        inst.tolerance = float(data.get("tolerance", 1e-4))
        inst.bone_weights = [float(x) for x in data.get("bone_weights", [])]
        return inst


# ---------------------------------------------------------------------------
# Pure-Python fallback solvers (used when the Cython core._ik is unavailable)
# ---------------------------------------------------------------------------

def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_conj(a):
    return (-a[0], -a[1], -a[2], a[3])


def _rot_vec(v, q):
    qx, qy, qz, qw = q
    tx = 2.0 * (qy * v[2] - qz * v[1])
    ty = 2.0 * (qz * v[0] - qx * v[2])
    tz = 2.0 * (qx * v[1] - qy * v[0])
    return (
        v[0] + qw * tx + (qy * tz - qz * ty),
        v[1] + qw * ty + (qz * tx - qx * tz),
        v[2] + qw * tz + (qx * ty - qy * tx),
    )


def _quat_from_two_vecs(v0, v1):
    n0 = math.sqrt(v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2)
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    if n0 < 1e-10 or n1 < 1e-10:
        return (0.0, 0.0, 0.0, 1.0)
    u = (v0[0] / n0, v0[1] / n0, v0[2] / n0)
    w = (v1[0] / n1, v1[1] / n1, v1[2] / n1)
    c = u[0] * w[0] + u[1] * w[1] + u[2] * w[2]
    if c > 0.99999999:
        return (0.0, 0.0, 0.0, 1.0)
    cr = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])
    den = math.sqrt(2.0 * (1.0 + c))
    if den < 1e-10:
        return (0.0, 0.0, 1.0, 0.0)
    return (cr[0] / den, cr[1] / den, cr[2] / den, den * 0.5)


def _two_bone_solve_py(a, b, c, target, pole, stretch, bend):
    a = (a.x, a.y, a.z)
    b = (b.x, b.y, b.z)
    c = (c.x, c.y, c.z)
    T = (target.x, target.y, target.z)
    P = (pole.x, pole.y, pole.z) if pole is not None else None
    L1 = math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))
    L2 = math.sqrt(sum((c[i] - b[i]) ** 2 for i in range(3)))
    d = math.sqrt(sum((T[i] - a[i]) ** 2 for i in range(3)))
    L1e, L2e = L1, L2
    max_reach = L1 + L2
    min_reach = abs(L1 - L2)
    if d > max_reach and stretch > 0.0:
        extra = d - max_reach
        L1e = L1 + extra * stretch * (L1 / max_reach)
        L2e = L2 + extra * stretch * (L2 / max_reach)
        max_reach = L1e + L2e
    d = min(max(d, min_reach), max_reach)
    if d < 1e-10:
        return (b, (0, 0, 0, 1), (0, 0, 0, 1))
    u = ((T[0] - a[0]) / d, (T[1] - a[1]) / d, (T[2] - a[2]) / d)

    def norm(v):
        n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-9 else (0.0, 0.0, 0.0)

    def cross(x, y):
        return (x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0])

    def dot(x, y):
        return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]

    prev = norm((b[0] - a[0], b[1] - a[1], b[2] - a[2]))
    use_pole = 0
    pd = (0.0, 0.0, 0.0)
    if P is not None:
        pd = norm((P[0] - a[0], P[1] - a[1], P[2] - a[2]))
        if dot(pd, pd) > 1e-6:
            use_pole = 1
    if use_pole:
        axis = cross(u, pd)
        if dot(axis, axis) < 1e-12:
            axis = cross(u, prev)
    else:
        axis = cross(u, prev)
        if dot(axis, axis) < 1e-12:
            axis = cross(u, (0.0, 0.0, 1.0))
    nrm = norm(axis)
    if dot(nrm, nrm) < 1e-12:
        nrm = (0.0, 0.0, 1.0)

    cos_a = (L1e * L1e + d * d - L2e * L2e) / (2.0 * L1e * d)
    cos_a = max(-1.0, min(1.0, cos_a))
    ang = math.acos(cos_a)
    s = math.sin(ang * 0.5)
    qr = (nrm[0] * s, nrm[1] * s, nrm[2] * s, math.cos(ang * 0.5))
    qr_inv = (-qr[0], -qr[1], -qr[2], qr[3])
    c1 = _rot_vec(u, qr)
    c2 = _rot_vec(u, qr_inv)
    if use_pole:
        d1 = dot(c1, pd)
        d2 = dot(c2, pd)
        dirn = c1 if d1 >= d2 else c2
    elif bend >= 0.0:
        dirn = c1
    else:
        dirn = c2
    b_new = (a[0] + L1e * dirn[0], a[1] + L1e * dirn[1], a[2] + L1e * dirn[2])
    qa = _quat_from_two_vecs((b[0] - a[0], b[1] - a[1], b[2] - a[2]),
                             (b_new[0] - a[0], b_new[1] - a[1], b_new[2] - a[2]))
    qb = _quat_from_two_vecs((c[0] - b[0], c[1] - b[1], c[2] - b[2]),
                             norm((T[0] - b_new[0], T[1] - b_new[1], T[2] - b_new[2])))
    return (b_new, qa, qb)


def _world_quat(t):
    wm = t.world_matrix._d
    m = wm
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q = ((m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s, 0.25 / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, 0.25 * s)
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)
    # the engine stores the conjugated quaternion on the wire, so expose it
    # in the same (semantic) basis the solvers and local_rotation use
    return (-q[0], -q[1], -q[2], q[3])


def _parent_world_quat(bone):
    p = bone._parent
    if p is not None and p.transform is not None:
        return _world_quat(p.transform)
    return (0.0, 0.0, 0.0, 1.0)


def _quat_to_local(parent_wq, world_q):
    # engine world = parent * local  =>  local = parent^-1 * world_q
    q = _quat_mul(_quat_conj(parent_wq), world_q)
    n = math.sqrt(sum(v * v for v in q))
    return tuple(v / n for v in q) if n > 1e-9 else (0.0, 0.0, 0.0, 1.0)


def _apply_local_py(t, q, weight):
    if t is None:
        return
    if weight < 0.999999:
        cur = t.local_rotation
        if cur is not None:
            cq = (cur._x, cur._y, cur._z, cur._w)
            d = sum(cq[i] * q[i] for i in range(4))
            s1 = weight if d >= 0 else -weight
            mixed = [cq[i] * (1.0 - weight) + q[i] * s1 for i in range(4)]
            n = math.sqrt(sum(v * v for v in mixed))
            q = tuple(v / n for v in mixed)
    t.local_rotation = Quat._make(q[0], q[1], q[2], q[3])


def _apply_world_delta_py(t, qd, weight):
    # local = local_cur * conj(world_cur) * qd * world_cur   (Cython parity)
    cur = t.local_rotation
    if cur is None:
        return
    ql = (cur._x, cur._y, cur._z, cur._w)
    qw = _world_quat(t)
    qwc = _quat_conj(qw)
    tmp = _quat_mul(qwc, qd)
    tmp2 = _quat_mul(tmp, qw)
    new_local = _quat_mul(ql, tmp2)
    _apply_local_py(t, new_local, weight)


def np_points(vecs):
    return [[v.x, v.y, v.z] for v in vecs]


def _fabrik_positions_py(points, target, iterations, tolerance):
    import numpy as np
    P = np.asarray(points, dtype=np.float64).copy()
    n = P.shape[0]
    if n < 2:
        return P
    T = np.asarray([target.x, target.y, target.z], dtype=np.float64)
    lens = np.sqrt(((P[1:] - P[:-1]) ** 2).sum(axis=1))
    lens = np.maximum(lens, 1e-10)
    base = P[0].copy()
    total = lens.sum()
    d0 = np.linalg.norm(T - base)
    if d0 >= total:
        dirn = (T - base) / d0
        ts = total * np.arange(n) / (n - 1)
        return base[None, :] + dirn[None, :] * ts[:, None]
    for _ in range(iterations):
        P[-1] = T
        for i in range(n - 2, -1, -1):
            d = P[i + 1] - P[i]
            nrm = np.linalg.norm(d)
            d = d / nrm if nrm > 1e-10 else d
            P[i] = P[i + 1] - d * lens[i]
        P[0] = base
        for i in range(n - 1):
            d = P[i + 1] - P[i]
            nrm = np.linalg.norm(d)
            d = d / nrm if nrm > 1e-10 else d
            P[i + 1] = P[i] + d * lens[i]
        if np.linalg.norm(P[-1] - T) < tolerance:
            break
    return P


def _apply_fabrik_py(bones, new_pts, pole, weight, bone_weights):
    n = len(bones)
    # snapshot the original pose so that every local quat is expressed against
    # the pre-solve world frames (mirrors the Cython solver's WQv buffer)
    old_pos = [b.transform.position for b in bones]
    old_wq = [_world_quat(b.transform) for b in bones]
    parent_wq = _parent_world_quat(bones[0])
    for i in range(n - 1):
        d0 = (
            old_pos[i + 1].x - old_pos[i].x,
            old_pos[i + 1].y - old_pos[i].y,
            old_pos[i + 1].z - old_pos[i].z,
        )
        d1 = (
            new_pts[i + 1][0] - new_pts[i][0],
            new_pts[i + 1][1] - new_pts[i][1],
            new_pts[i + 1][2] - new_pts[i][2],
        )
        d0 = _two_denorm(d0)
        d1 = _two_denorm(d1)
        newwq = _quat_mul(_quat_from_two_vecs(d0, d1), old_wq[i])
        local = _quat_to_local(parent_wq, newwq)
        w = weight
        if bone_weights and i < len(bone_weights):
            w = weight * bone_weights[i]
        _apply_local_py(bones[i].transform, local, w)
        parent_wq = newwq


def _two_denorm(v):
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return tuple(x / n for x in v) if n > 1e-10 else v