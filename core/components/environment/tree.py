from __future__ import annotations
import math
import random
import numpy as np
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry, Entity
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Vec3


_GEN_PARAMS = frozenset({
    "height", "trunk_radius", "trunk_segments", "levels", "branch_angle",
    "ratio", "taper", "branches_per_node", "spiral_twist",
    "leaf_size", "leaf_density", "seed", "random_seed",
    "branch_curvature", "leaf_material_path",
    "gnarliness", "gravity",
})


def _normalize(v: tuple) -> tuple:
    l = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if l < 1e-10:
        return (0.0, 1.0, 0.0)
    return (v[0]/l, v[1]/l, v[2]/l)


def _cross(a: tuple, b: tuple) -> tuple:
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _bezier_eval(start: tuple, ctrl: tuple, end: tuple, t: float) -> tuple:
    u = 1.0 - t
    return (u*u*start[0] + 2*u*t*ctrl[0] + t*t*end[0],
            u*u*start[1] + 2*u*t*ctrl[1] + t*t*end[1],
            u*u*start[2] + 2*u*t*ctrl[2] + t*t*end[2])


def _bezier_tangent(start: tuple, ctrl: tuple, end: tuple, t: float) -> tuple:
    return (2*(1-t)*(ctrl[0]-start[0]) + 2*t*(end[0]-ctrl[0]),
            2*(1-t)*(ctrl[1]-start[1]) + 2*t*(end[1]-ctrl[1]),
            2*(1-t)*(ctrl[2]-start[2]) + 2*t*(end[2]-ctrl[2]))


def _bezier_length(start, ctrl, end, samples=12):
    length = 0.0
    prev = start
    for i in range(1, samples + 1):
        t = i / samples
        p = _bezier_eval(start, ctrl, end, t)
        dx = p[0] - prev[0]; dy = p[1] - prev[1]; dz = p[2] - prev[2]
        length += math.sqrt(dx*dx + dy*dy + dz*dz)
        prev = p
    return length


class _BranchNode:
    __slots__ = ("start", "ctrl", "end", "radius_start", "radius_end",
                 "level", "phase_off", "children", "is_terminal", "sub_branches")

    def __init__(self, start: tuple, ctrl: tuple, end: tuple,
                 radius_start: float, radius_end: float,
                 level: int, phase_off: float):
        self.start = start
        self.ctrl = ctrl
        self.end = end
        self.radius_start = radius_start
        self.radius_end = radius_end
        self.level = level
        self.phase_off = phase_off
        self.children: list[_BranchNode] = []
        self.is_terminal = False
        self.sub_branches: list[_BranchNode] = []


class _BranchGen:
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self.branch_curvature = 0.4

    def generate(self, height: float, trunk_radius: float, levels: int,
                 branch_angle: float, ratio: float, taper: float,
                 branches_per_node: int, spiral_twist: float,
                 leaf_size: float, leaf_density: int,
                 branch_curvature: float = 0.4) -> _BranchNode:
        self.branch_curvature = branch_curvature
        trunk_top = (0.0, height, 0.0)
        root = _BranchNode(
            start=(0.0, 0.0, 0.0),
            ctrl=(0.0, height * 0.5, 0.0),
            end=trunk_top,
            radius_start=trunk_radius,
            radius_end=trunk_radius * 0.25,
            level=0,
            phase_off=self._rng.random()
        )
        self._branch(root, height, trunk_radius, levels, branch_angle,
                     ratio, taper, branches_per_node, spiral_twist,
                     leaf_size, leaf_density, 0)
        return root

    def _branch(self, parent: _BranchNode, parent_len: float, parent_radius: float,
                remaining_levels: int, branch_angle: float, ratio: float, taper: float,
                branches_per_node: int, spiral_twist: float,
                leaf_size: float, leaf_density: int, depth: int):
        if remaining_levels <= 0:
            parent.is_terminal = True
            return

        dx = parent.end[0] - parent.start[0]
        dy = parent.end[1] - parent.start[1]
        dz = parent.end[2] - parent.start[2]
        seg_len = math.sqrt(dx*dx + dy*dy + dz*dz)
        if seg_len < 0.01:
            return

        dir_vec = (dx/seg_len, dy/seg_len, dz/seg_len)
        up = (0.0, 1.0, 0.0)
        if abs(dir_vec[1]) > 0.999:
            right = (1.0, 0.0, 0.0)
            forward = (0.0, 0.0, 1.0)
        else:
            right = _normalize(_cross(up, dir_vec))
            forward = _normalize(_cross(dir_vec, right))

        n_branches = max(1, branches_per_node - depth // 2)
        height_ratio = 1.0 - depth / max(1, remaining_levels + depth)
        angle_mult = 0.6 + 0.4 * height_ratio
        base_ang = math.radians(branch_angle) * angle_mult

        child_len = parent_len * ratio
        child_radius = parent_radius * (taper * 0.65)
        attach_offset = 0.25 + 0.4 * (depth / max(1, remaining_levels + depth))

        for i in range(n_branches):
            theta = 2.0 * math.pi * i / n_branches + depth * spiral_twist
            ang_var = base_ang * (0.7 + 0.6 * self._rng.random())
            local_dir = (
                math.sin(ang_var) * math.cos(theta),
                math.cos(ang_var),
                math.sin(ang_var) * math.sin(theta)
            )
            world_dir = (
                local_dir[0]*right[0] + local_dir[1]*dir_vec[0] + local_dir[2]*forward[0],
                local_dir[0]*right[1] + local_dir[1]*dir_vec[1] + local_dir[2]*forward[1],
                local_dir[0]*right[2] + local_dir[1]*dir_vec[2] + local_dir[2]*forward[2]
            )
            wd_len = math.sqrt(world_dir[0]**2 + world_dir[1]**2 + world_dir[2]**2)
            if wd_len > 0.001:
                world_dir = (world_dir[0]/wd_len, world_dir[1]/wd_len, world_dir[2]/wd_len)

            attach_t = attach_offset + 0.25 * self._rng.random()
            attach_point = (
                parent.start[0] + dx * attach_t,
                parent.start[1] + dy * attach_t,
                parent.start[2] + dz * attach_t
            )

            parent_radius_at_t = _lerp(parent.radius_start, parent.radius_end, attach_t)
            rad_off = max(0.0, parent_radius_at_t - child_radius * 0.3)
            horiz_dir = (world_dir[0], 0.0, world_dir[2])
            hd_len = math.sqrt(horiz_dir[0]**2 + horiz_dir[2]**2)
            if hd_len > 0.001:
                horiz_dir = (horiz_dir[0]/hd_len, 0.0, horiz_dir[2]/hd_len)
            else:
                horiz_dir = (right[0], 0.0, right[2])
            attach_point = (
                attach_point[0] + horiz_dir[0] * rad_off,
                attach_point[1],
                attach_point[2] + horiz_dir[2] * rad_off
            )

            child_end = (
                attach_point[0] + world_dir[0] * child_len,
                attach_point[1] + world_dir[1] * child_len,
                attach_point[2] + world_dir[2] * child_len
            )

            horiz = math.sqrt(world_dir[0]**2 + world_dir[2]**2)
            upward_bend = horiz * child_len * self.branch_curvature
            outward_bend = child_len * 0.08
            ctrl_point = (
                attach_point[0] + world_dir[0] * child_len * 0.35 + horiz_dir[0] * outward_bend,
                (attach_point[1] + child_end[1]) * 0.5 + upward_bend,
                attach_point[2] + world_dir[2] * child_len * 0.35 + horiz_dir[2] * outward_bend
            )

            child_phase = self._rng.random()
            child = _BranchNode(
                start=attach_point, ctrl=ctrl_point, end=child_end,
                radius_start=child_radius * 1.15,
                radius_end=child_radius * (taper * 0.85),
                level=parent.level + 1, phase_off=child_phase
            )
            parent.children.append(child)

            self._branch(child, child_len, child_radius, remaining_levels - 1,
                         branch_angle, ratio * 0.9, taper,
                         max(1, branches_per_node - 1), spiral_twist,
                         leaf_size, leaf_density, depth + 1)

            if child_radius > 0.04 and self._rng.random() < 0.5:
                sub_count = self._rng.randint(1, 2)
                for j in range(sub_count):
                    sub_t = 0.6 + 0.3 * self._rng.random()
                    sub_pos = _bezier_eval(attach_point, ctrl_point, child_end, sub_t)
                    sub_len = child_len * 0.25 * (0.5 + 0.5 * self._rng.random())
                    sub_radius = child_radius * 0.3 * (0.5 + 0.5 * self._rng.random())
                    sub_ang = math.radians(50 + self._rng.random() * 30)
                    sub_theta = 2.0 * math.pi * j / sub_count + self._rng.random() * 0.5

                    sub_dir = _bezier_tangent(attach_point, ctrl_point, child_end, sub_t)
                    sub_dir_l = math.sqrt(sub_dir[0]**2 + sub_dir[1]**2 + sub_dir[2]**2)
                    if sub_dir_l > 0.001:
                        sub_dir = (sub_dir[0]/sub_dir_l, sub_dir[1]/sub_dir_l, sub_dir[2]/sub_dir_l)

                    if abs(sub_dir[1]) > 0.999:
                        s_right = (1.0, 0.0, 0.0)
                        s_forward = (0.0, 0.0, 1.0)
                    else:
                        s_right = _normalize(_cross(up, sub_dir))
                        s_forward = _normalize(_cross(sub_dir, s_right))

                    s_local = (
                        math.sin(sub_ang) * math.cos(sub_theta),
                        math.cos(sub_ang),
                        math.sin(sub_ang) * math.sin(sub_theta)
                    )
                    s_world = (
                        s_local[0]*s_right[0] + s_local[1]*sub_dir[0] + s_local[2]*s_forward[0],
                        s_local[0]*s_right[1] + s_local[1]*sub_dir[1] + s_local[2]*s_forward[1],
                        s_local[0]*s_right[2] + s_local[1]*sub_dir[2] + s_local[2]*s_forward[2]
                    )
                    sw_len = math.sqrt(s_world[0]**2 + s_world[1]**2 + s_world[2]**2)
                    if sw_len > 0.001:
                        s_world = (s_world[0]/sw_len, s_world[1]/sw_len, s_world[2]/sw_len)

                    sub_end = (
                        sub_pos[0] + s_world[0] * sub_len,
                        sub_pos[1] + s_world[1] * sub_len,
                        sub_pos[2] + s_world[2] * sub_len
                    )
                    sub_ctrl = (
                        (sub_pos[0] + sub_end[0]) * 0.5 + s_world[0] * sub_len * 0.1,
                        (sub_pos[1] + sub_end[1]) * 0.5 + sub_len * 0.15,
                        (sub_pos[2] + sub_end[2]) * 0.5 + s_world[2] * sub_len * 0.1
                    )
                    sub = _BranchNode(
                        start=sub_pos, ctrl=sub_ctrl, end=sub_end,
                        radius_start=sub_radius, radius_end=0.0,
                        level=parent.level + 2, phase_off=self._rng.random()
                    )
                    sub.is_terminal = True
                    parent.sub_branches.append(sub)

        if depth > 0 and self._rng.random() < 0.2:
            parent.is_terminal = True


def _tube_along_curve(node: _BranchNode, segments: int, rings: int,
                      gnarliness: float = 0.0, gravity: float = 0.0,
                      rng: random.Random = None) -> tuple:
    start, ctrl, end = node.start, node.ctrl, node.end
    dx = end[0] - start[0]; dy = end[1] - start[1]; dz = end[2] - start[2]
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 0.001:
        return [], [], [], [], []
    curve_len = _bezier_length(start, ctrl, end)
    if curve_len < 0.001:
        return [], [], [], [], []

    verts, norms, uvs, colors = [], [], [], []

    init_tangent = _bezier_tangent(start, ctrl, end, 0.0)
    tl = math.sqrt(init_tangent[0]**2 + init_tangent[1]**2 + init_tangent[2]**2)
    if tl > 0.001:
        cur_dir = (init_tangent[0]/tl, init_tangent[1]/tl, init_tangent[2]/tl)
    else:
        cur_dir = (0.0, 1.0, 0.0)
    up_ref = (0.0, 1.0, 0.0)
    if abs(cur_dir[1]) > 0.999:
        cur_basis_x = (1.0, 0.0, 0.0)
    else:
        cur_basis_x = _normalize(_cross(up_ref, cur_dir))
    cur_basis_z = _normalize(_cross(cur_dir, cur_basis_x))

    for ri in range(rings + 1):
        t = ri / max(1, rings)
        pos = _bezier_eval(start, ctrl, end, t)

        r_raw = _lerp(node.radius_start, node.radius_end, t)
        if ri == rings:
            r = 0.001
        elif node.level == 0:
            flare = 1.0 + 0.15 * max(0.0, 1.0 - t * 4.0)
            r = r_raw * flare
        else:
            r = r_raw

        if ri > 0 and gnarliness > 0.0 and r > 0.001 and rng is not None:
            gnarl_amount = max(0.5, 1.0 / math.sqrt(max(0.001, r))) * gnarliness
            pitch = rng.gauss(0, gnarl_amount * 0.25)
            roll = rng.gauss(0, gnarl_amount * 0.25)
            if abs(pitch) > 0.0001:
                c_p = math.cos(pitch); s_p = math.sin(pitch)
                new_bx = (cur_basis_x[0]*c_p + cur_dir[0]*s_p,
                          cur_basis_x[1]*c_p + cur_dir[1]*s_p,
                          cur_basis_x[2]*c_p + cur_dir[2]*s_p)
                new_dir = (-cur_basis_x[0]*s_p + cur_dir[0]*c_p,
                           -cur_basis_x[1]*s_p + cur_dir[1]*c_p,
                           -cur_basis_x[2]*s_p + cur_dir[2]*c_p)
                cur_basis_x = new_bx
                cur_dir = new_dir
            if abs(roll) > 0.0001:
                c_r = math.cos(roll); s_r = math.sin(roll)
                new_bz = (cur_basis_z[0]*c_r + cur_dir[0]*s_r,
                          cur_basis_z[1]*c_r + cur_dir[1]*s_r,
                          cur_basis_z[2]*c_r + cur_dir[2]*s_r)
                new_dir = (-cur_basis_z[0]*s_r + cur_dir[0]*c_r,
                           -cur_basis_z[1]*s_r + cur_dir[1]*c_r,
                           -cur_basis_z[2]*s_r + cur_dir[2]*c_r)
                cur_basis_z = new_bz
                cur_dir = new_dir
            cur_basis_x = _normalize(cur_basis_x)
            cur_dir = _normalize(cur_dir)
            cur_basis_z = _normalize(_cross(cur_dir, cur_basis_x))

        if ri > 0 and gravity > 0.0 and r > 0.001:
            grav_tilt = gravity * 0.02 / max(0.005, r)
            grav_tilt = min(grav_tilt, 0.2)
            c_g = math.cos(grav_tilt); s_g = math.sin(grav_tilt)
            new_dir = (cur_dir[0]*c_g + cur_basis_z[0]*s_g,
                       cur_dir[1]*c_g + cur_basis_z[1]*s_g,
                       cur_dir[2]*c_g + cur_basis_z[2]*s_g)
            new_bz = (-cur_dir[0]*s_g + cur_basis_z[0]*c_g,
                      -cur_dir[1]*s_g + cur_basis_z[1]*c_g,
                      -cur_dir[2]*s_g + cur_basis_z[2]*c_g)
            cur_dir = _normalize(new_dir)
            cur_basis_z = _normalize(new_bz)
            cur_basis_x = _normalize(_cross(cur_basis_z, cur_dir))

        for i in range(segments + 1):
            theta = 2.0 * math.pi * i / segments
            c = math.cos(theta); s = math.sin(theta)
            radial = (cur_basis_x[0]*c + cur_basis_z[0]*s,
                      cur_basis_x[1]*c + cur_basis_z[1]*s,
                      cur_basis_x[2]*c + cur_basis_z[2]*s)
            verts.append(pos[0] + radial[0] * r)
            verts.append(pos[1] + radial[1] * r)
            verts.append(pos[2] + radial[2] * r)
            norms.append(radial[0]); norms.append(radial[1]); norms.append(radial[2])
            uvs.append(float(i) / segments); uvs.append(t)
            branch_lvl = min(1.0, node.level / 5.0)
            stiffness = max(0.05, 1.0 - branch_lvl * 0.7)
            colors.append(branch_lvl); colors.append(stiffness)
            colors.append(node.phase_off); colors.append(0.0)

    idxs = []
    for ri in range(rings):
        for si in range(segments):
            a0 = ri * (segments + 1) + si
            b0 = a0 + 1
            a1 = (ri + 1) * (segments + 1) + si
            b1 = a1 + 1
            idxs.append(a0); idxs.append(b0); idxs.append(a1)
            idxs.append(b0); idxs.append(b1); idxs.append(a1)
    return verts, norms, uvs, colors, idxs


def _make_leaf(position: tuple, size: float, rot_y: float,
               tilt_x: float, tilt_z: float,
               phase_off: float, stiffness: float, branch_lvl: float) -> tuple:
    hw = size * 0.35
    hs = size * 0.5

    def _rotate_y(v, a):
        c = math.cos(a); s = math.sin(a)
        return (v[0]*c - v[2]*s, v[1], v[0]*s + v[2]*c)
    def _rotate_x(v, a):
        c = math.cos(a); s = math.sin(a)
        return (v[0], v[1]*c - v[2]*s, v[1]*s + v[2]*c)
    def _rotate_z(v, a):
        c = math.cos(a); s = math.sin(a)
        return (v[0]*c - v[1]*s, v[0]*s + v[1]*c, v[2])

    all_verts = []
    all_norms = []
    all_uvs = []
    all_colors = []
    all_idxs = []
    base = 0

    for cross_angle in [0.0, math.pi * 0.5]:
        corners = [(-hw, 0.0, -hs), ( hw, 0.0, -hs), ( hw, 0.0,  hs), (-hw, 0.0,  hs)]
        rotated = []
        for c in corners:
            c = _rotate_y(c, cross_angle)
            c = _rotate_y(c, rot_y)
            c = _rotate_x(c, tilt_x)
            c = _rotate_z(c, tilt_z)
            rotated.append((c[0] + position[0], c[1] + position[1] + size * 0.1, c[2] + position[2]))

        nrm = (0.0, 1.0, 0.0)
        nrm = _rotate_y(nrm, cross_angle)
        nrm = _rotate_y(nrm, rot_y)
        nrm = _rotate_x(nrm, tilt_x)
        nrm = _rotate_z(nrm, tilt_z)
        nrm = _normalize(nrm)

        for c in rotated:
            all_verts.extend(c)
        all_norms.extend(nrm * 4)
        all_uvs.extend([0, 0,  1, 0,  1, 1,  0, 1])
        all_colors.extend([branch_lvl, stiffness, phase_off, 1.0] * 4)
        all_idxs.extend([base, base+2, base+1,  base, base+3, base+2])
        base += 4

    return all_verts, all_norms, all_uvs, all_colors, all_idxs


def _leaf_cluster(position: tuple, size: float, count: int,
                  phase_off: float, level: int, rng: random.Random) -> tuple:
    verts, norms, uvs, colors = [], [], [], []
    idxs = []
    base_idx = 0
    branch_lvl = min(1.0, level / 5.0)
    stiffness = max(0.01, 1.0 - branch_lvl * 0.85)

    for i in range(count):
        phi = rng.random() * math.pi * 0.45
        theta = rng.random() * math.pi * 2.0
        r_offset = size * 0.3 * math.sqrt(rng.random())
        lx = r_offset * math.sin(phi) * math.cos(theta)
        ly = r_offset * math.cos(phi) + size * 0.1
        lz = r_offset * math.sin(phi) * math.sin(theta)

        leaf_pos = (position[0] + lx, position[1] + ly, position[2] + lz)
        tilt_x = (rng.random() - 0.5) * math.radians(20)
        tilt_z = (rng.random() - 0.5) * math.radians(20)
        leaf_rot = rng.random() * math.pi * 2.0
        l_scale = size * (0.7 + 0.6 * rng.random())

        lv, ln, lu, lc, li = _make_leaf(
            leaf_pos, l_scale, leaf_rot, tilt_x, tilt_z,
            phase_off, stiffness, branch_lvl
        )
        verts.extend(lv); norms.extend(ln); uvs.extend(lu); colors.extend(lc)
        idxs.extend(base_idx + i for i in li)
        base_idx += 8

    return verts, norms, uvs, colors, idxs


def _collect_branch_geometry(node: _BranchNode, segments: int,
                             leaf_size: float, leaf_density: int,
                             bark: list, leaves: list,
                             rng: random.Random,
                             gnarliness: float = 0.0,
                             gravity: float = 0.0) -> None:
    rings = 6 if node.level == 0 else 5
    bv, bn, bu, bc, bi = _tube_along_curve(node, segments, rings,
                                            gnarliness=gnarliness,
                                            gravity=gravity, rng=rng)
    if bv:
        base = len(bark[0]) // 3
        bark[0].extend(bv); bark[1].extend(bn); bark[2].extend(bu)
        bark[3].extend(bc); bark[4].extend(i + base for i in bi)

    for sub in node.sub_branches:
        sub_gnarl = gnarliness * 1.5
        sv, sn, su, sc, si = _tube_along_curve(sub, max(3, segments // 2), 4,
                                                gnarliness=sub_gnarl,
                                                gravity=gravity, rng=rng)
        if sv:
            base = len(bark[0]) // 3
            bark[0].extend(sv); bark[1].extend(sn); bark[2].extend(su)
            bark[3].extend(sc); bark[4].extend(i + base for i in si)
        if sub.is_terminal and leaf_density > 0:
            lv, ln, lu, lc, li = _leaf_cluster(
                sub.end, leaf_size * 0.6, max(1, leaf_density // 2),
                sub.phase_off, sub.level, rng
            )
            if lv:
                base = len(leaves[0]) // 3
                leaves[0].extend(lv); leaves[1].extend(ln); leaves[2].extend(lu)
                leaves[3].extend(lc); leaves[4].extend(i + base for i in li)

    if node.is_terminal and leaf_density > 0:
        lv, ln, lu, lc, li = _leaf_cluster(
            node.end, leaf_size, leaf_density,
            node.phase_off, node.level + 1, rng
        )
        if lv:
            base = len(leaves[0]) // 3
            leaves[0].extend(lv); leaves[1].extend(ln); leaves[2].extend(lu)
            leaves[3].extend(lc); leaves[4].extend(i + base for i in li)

    for child in node.children:
        _collect_branch_geometry(child, segments, leaf_size, leaf_density,
                                 bark, leaves, rng,
                                 gnarliness=gnarliness, gravity=gravity)


def generate_tree_mesh(height: float = 10.0, trunk_radius: float = 0.3,
                       trunk_segments: int = 8, levels: int = 4,
                       branch_angle: float = 35.0, ratio: float = 0.6,
                       taper: float = 0.8, branches_per_node: int = 3,
                       spiral_twist: float = 1.2, leaf_size: float = 0.35,
                       leaf_density: int = 4, seed: int = 0,
                       branch_curvature: float = 0.4,
                       gnarliness: float = 0.0, gravity: float = 0.0):
    from core.renderer.mesh_data import MeshData
    gen = _BranchGen(seed)
    rng = random.Random(seed + 999)
    root = gen.generate(height, trunk_radius, levels, branch_angle,
                        ratio, taper, branches_per_node, spiral_twist,
                        leaf_size, leaf_density, branch_curvature)

    bark = [[], [], [], [], []]
    leaves = [[], [], [], [], []]

    _collect_branch_geometry(root, trunk_segments, leaf_size, leaf_density,
                             bark, leaves, rng,
                             gnarliness=gnarliness, gravity=gravity)

    all_verts = bark[0] + leaves[0]
    all_norms = bark[1] + leaves[1]
    all_uvs = bark[2] + leaves[2]
    all_colors = bark[3] + leaves[3]

    bark_vert_count = len(bark[0]) // 3
    all_idxs = bark[4] + [i + bark_vert_count for i in leaves[4]]

    if not all_verts:
        md = MeshData()
        h_ = 0.5
        md.vertices = np.array([-h_,0,-h_, h_,0,-h_, h_,0,h_, -h_,0,h_], dtype=np.float32)
        md.normals = np.array([0,1,0, 0,1,0, 0,1,0, 0,1,0], dtype=np.float32)
        md.uvs = np.array([0,0, 1,0, 1,1, 0,1], dtype=np.float32)
        md.indices = np.array([0,1,2, 0,2,3], dtype=np.uint32)
        return md

    mesh = MeshData()
    mesh.vertices = np.array(all_verts, dtype=np.float32)
    mesh.normals = np.array(all_norms, dtype=np.float32)
    mesh.uvs = np.array(all_uvs, dtype=np.float32)
    mesh.colors = np.array(all_colors, dtype=np.float32)
    mesh.indices = np.array(all_idxs, dtype=np.uint32)
    if leaves[4]:
        mesh.sub_mesh_ranges = [(0, len(bark[4])), (len(bark[4]), len(leaves[4]))]
    mesh.compute_aabb()
    return mesh


@ComponentRegistry.register
class Tree(Component):
    _icon = "Tree.png"
    _gizmo_icon_color = (60, 180, 60)
    _gizmo_icon_label = "T"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("height", "Height", FieldType.FLOAT, min_val=1.0, max_val=100.0, step=0.5, decimals=2),
            InspectorField("trunk_radius", "Trunk Radius", FieldType.FLOAT, min_val=0.01, max_val=3.0, step=0.05, decimals=3),
            InspectorField("trunk_segments", "Trunk Segments", FieldType.INT, min_val=4, max_val=24, step=1),
            InspectorField("levels", "Branch Levels", FieldType.INT, min_val=1, max_val=8, step=1),
            InspectorField("branch_angle", "Branch Angle (deg)", FieldType.FLOAT, min_val=5.0, max_val=80.0, step=1.0, decimals=1),
            InspectorField("ratio", "Branch Length Ratio", FieldType.SLIDER, min_val=0.2, max_val=0.95, step=0.01, decimals=3),
            InspectorField("taper", "Branch Taper", FieldType.SLIDER, min_val=0.3, max_val=1.0, step=0.01, decimals=3),
            InspectorField("branches_per_node", "Branches Per Node", FieldType.INT, min_val=1, max_val=6, step=1),
            InspectorField("spiral_twist", "Spiral Twist", FieldType.FLOAT, min_val=0.0, max_val=6.0, step=0.1, decimals=2),
            InspectorField("branch_curvature", "Branch Curvature", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
            InspectorField("gnarliness", "Gnarliness", FieldType.SLIDER, min_val=0.0, max_val=2.0, step=0.05, decimals=2),
            InspectorField("gravity", "Gravity Droop", FieldType.SLIDER, min_val=0.0, max_val=3.0, step=0.05, decimals=2),
            InspectorField("leaf_size", "Leaf Size", FieldType.FLOAT, min_val=0.05, max_val=1.5, step=0.05, decimals=3),
            InspectorField("leaf_density", "Leaf Density", FieldType.INT, min_val=0, max_val=20, step=1),
            InspectorField("seed", "Seed", FieldType.INT, min_val=0, max_val=999999, step=1),
            InspectorField("random_seed", "Random Seed", FieldType.BOOL),
            InspectorField("material_path", "Bark Material", FieldType.RESOURCE_PATH, file_filter="Material (*.mat)"),
            InspectorField("leaf_material_path", "Leaf Material", FieldType.RESOURCE_PATH, file_filter="Material (*.mat)"),
            InspectorField("auto_regenerate", "Auto Regenerate", FieldType.BOOL),
            InspectorField("_btn_regenerate", "Regenerate", FieldType.BUTTON),
        ]

    def __init__(self):
        self._init_done = False
        super().__init__()
        self.height: float = 10.0
        self.trunk_radius: float = 0.3
        self.trunk_segments: int = 8
        self.levels: int = 4
        self.branch_angle: float = 35.0
        self.ratio: float = 0.6
        self.taper: float = 0.8
        self.branches_per_node: int = 3
        self.spiral_twist: float = 1.2
        self.branch_curvature: float = 0.4
        self.gnarliness: float = 0.3
        self.gravity: float = 0.5
        self.leaf_size: float = 0.35
        self.leaf_density: int = 4
        self.seed: int = 0
        self.random_seed: bool = False
        self.material_path: str = ""
        self.leaf_material_path: str = ""
        self.auto_regenerate: bool = True
        self._generated: bool = False
        self._gpu_dirty: bool = True
        self._mesh_data: Optional['MeshData'] = None
        self._mesh_name: str = ""
        self._params_hash: int = 0
        self._init_done = True

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in _GEN_PARAMS and getattr(self, '_init_done', False):
            self._mark_dirty()
            if self.auto_regenerate:
                try:
                    ent = getattr(self, '_entity', None)
                    if ent:
                        sc = getattr(ent, '_scene', None)
                        if sc:
                            sc._render_version += 1
                except Exception:
                    pass

    def _current_params_hash(self) -> int:
        return hash((
            self.height, self.trunk_radius, self.trunk_segments,
            self.levels, self.branch_angle, self.ratio, self.taper,
            self.branches_per_node, self.spiral_twist, self.branch_curvature,
            self.gnarliness, self.gravity,
            self.leaf_size, self.leaf_density, self.seed, self.random_seed,
        ))

    def _mark_dirty(self):
        self._generated = False
        self._gpu_dirty = True

    def _btn_regenerate(self):
        self._mark_dirty()
        self.generate()
        try:
            ent = getattr(self, '_entity', None)
            if ent:
                sc = getattr(ent, '_scene', None)
                if sc:
                    sc._render_version += 1
        except Exception:
            pass

    def generate(self) -> 'MeshData':
        actual_seed = random.randint(0, 999999) if self.random_seed else self.seed
        self._mesh_data = generate_tree_mesh(
            height=self.height, trunk_radius=self.trunk_radius,
            trunk_segments=self.trunk_segments, levels=self.levels,
            branch_angle=self.branch_angle, ratio=self.ratio,
            taper=self.taper, branches_per_node=self.branches_per_node,
            spiral_twist=self.spiral_twist,
            leaf_size=self.leaf_size, leaf_density=self.leaf_density,
            seed=actual_seed, branch_curvature=self.branch_curvature,
            gnarliness=self.gnarliness, gravity=self.gravity,
        )
        self._generated = True
        self._gpu_dirty = True
        self._params_hash = self._current_params_hash()
        return self._mesh_data

    def needs_regenerate(self) -> bool:
        if self._mesh_data is None:
            return True
        if not self._generated:
            return True
        if self.auto_regenerate and self._current_params_hash() != self._params_hash:
            return True
        return False

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "height": self.height, "trunk_radius": self.trunk_radius,
            "trunk_segments": self.trunk_segments, "levels": self.levels,
            "branch_angle": self.branch_angle, "ratio": self.ratio,
            "taper": self.taper, "branches_per_node": self.branches_per_node,
            "spiral_twist": self.spiral_twist, "branch_curvature": self.branch_curvature,
            "gnarliness": self.gnarliness, "gravity": self.gravity,
            "leaf_size": self.leaf_size, "leaf_density": self.leaf_density,
            "seed": self.seed, "random_seed": self.random_seed,
            "material_path": self.material_path,
            "leaf_material_path": self.leaf_material_path,
            "auto_regenerate": self.auto_regenerate,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Tree:
        t = cls()
        t.enabled = data.get("enabled", True)
        t.height = data.get("height", 10.0)
        t.trunk_radius = data.get("trunk_radius", 0.3)
        t.trunk_segments = data.get("trunk_segments", 8)
        t.levels = data.get("levels", 4)
        t.branch_angle = data.get("branch_angle", 35.0)
        t.ratio = data.get("ratio", 0.6)
        t.taper = data.get("taper", 0.8)
        t.branches_per_node = data.get("branches_per_node", 3)
        t.spiral_twist = data.get("spiral_twist", 1.2)
        t.branch_curvature = data.get("branch_curvature", 0.4)
        t.gnarliness = data.get("gnarliness", 0.3)
        t.gravity = data.get("gravity", 0.5)
        t.leaf_size = data.get("leaf_size", 0.35)
        t.leaf_density = data.get("leaf_density", 4)
        t.seed = data.get("seed", 0)
        t.random_seed = data.get("random_seed", False)
        t.material_path = data.get("material_path", "")
        t.leaf_material_path = data.get("leaf_material_path", "")
        t.auto_regenerate = data.get("auto_regenerate", True)
        return t
