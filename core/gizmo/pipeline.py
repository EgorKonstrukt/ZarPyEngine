# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple
from core.math.math3d import Vec3
from core.ecs.ecs import GizmoPrimitive, GizmoStyle
from core.gizmo.api import Gizmos


class GizmoPipeline:
    __slots__ = ('_batches', '_instance_batches')

    def __init__(self):
        self._batches: list[GizmoPrimitive] = []
        self._instance_batches: dict[str, list] = {}

    def add(self, prim: GizmoPrimitive):
        self._batches.append(prim)

    def add_instance(self, shape_type: str, transform_flat: np.ndarray, color: list):
        self._instance_batches.setdefault(shape_type, []).append((transform_flat, color))

    def collect(self, scene, comp_type):
        for entity in scene.get_entities_with_component(comp_type):
            if not entity.active:
                continue
            for comp in entity.get_components(comp_type):
                try:
                    inst = comp.gizmo_instance_data()
                    if inst is not None:
                        self.add_instance(inst.shape_type, inst.transform_flat, inst.color)
                        continue
                    for prim in comp.gizmo():
                        if prim.starts.shape[0] > 0:
                            self._batches.append(prim)
                except Exception:
                    pass

    def get_instance_render_data(self) -> list[Tuple[str, np.ndarray, int]]:
        result = []
        for shape_type, instances in self._instance_batches.items():
            n = len(instances)
            buf = np.empty((n, 20), dtype=np.float32)
            for i, (tf, col) in enumerate(instances):
                buf[i, :16] = tf
                buf[i, 16:20] = col
            result.append((shape_type, buf, n))
        self._instance_batches.clear()
        return result

    def flush(self, time_s: float = 0.0):
        self._render_via(Gizmos.draw_lines, time_s)

    def flush_and_render(self, vp, vp_mat, time_s: float = 0.0):
        fw, fh = vp._get_physical_dims()
        cam_pos = vp._cam.position if vp._cam else Vec3(0, 0, 0)

        def render_func(s, e, c):
            vp._renderer.render_gizmo_arrays(s, e, c, vp_mat, fw, fh, thickness_multiplier=1.0)

        self._render_via(render_func, time_s)

        for shape_type, instance_data, num in self.get_instance_render_data():
            vp._renderer.render_instanced_gizmo_lines(
                shape_type, instance_data, num, vp_mat, fw, fh, thickness_multiplier=1.0, cam_pos=cam_pos)
        self._batches.clear()

    def _render_via(self, draw_fn, time_s: float = 0.0):
        t = time_s
        s_list, e_list, c_list = [], [], []
        glow_list = []
        for prim in self._batches:
            s, e, c = prim.starts, prim.ends, prim.colors
            n = s.shape[0]
            if n == 0:
                continue
            style = prim.style or GizmoStyle.DEFAULT

            if style.pulsating:
                pulse = style.pulse_min_alpha + (1 - style.pulse_min_alpha) * (0.5 + 0.5 * math.sin(t * style.pulse_speed))
                c = c.copy()
                c[:, 3] *= pulse

            if style.color_cycling:
                shift = (math.sin(t * style.cycle_speed) * 0.5 + 0.5) * 0.3
                c = c.copy()
                c[:, 0] += shift; c[:, 1] += shift * 0.5; c[:, 2] -= shift * 0.3
                np.clip(c[:, :3], 0, 1, out=c[:, :3])

            if style.dashed:
                s, e, c = _dash_np(s, e, c, style.dash_length, style.gap_length)
                if s.shape[0] == 0:
                    continue

            s_list.append(s); e_list.append(e); c_list.append(c)

            if style.glow:
                glow_list.append((s, e, c, style.glow_layers, style.glow_intensity))

        if s_list:
            if len(s_list) == 1:
                draw_fn(s_list[0], e_list[0], c_list[0])
            else:
                draw_fn(np.concatenate(s_list), np.concatenate(e_list), np.concatenate(c_list))

        for s, e, c, layers, intensity in glow_list:
            for i in range(layers):
                alpha = intensity * (1.0 - i / layers) / layers
                gc = c.copy()
                gc[:, 3] *= alpha
                draw_fn(s, e, gc)

        self._batches.clear()


def _dash_np(starts: np.ndarray, ends: np.ndarray, colors: np.ndarray,
             dash_len: float, gap_len: float):
    step = dash_len + gap_len
    n = starts.shape[0]
    s_parts, e_parts, c_parts = [], [], []
    for i in range(n):
        sx, sy, sz = starts[i]; ex, ey, ez = ends[i]
        dx, dy, dz = ex - sx, ey - sy, ez - sz
        ln = math.sqrt(dx*dx + dy*dy + dz*dz)
        if ln < 1e-8:
            continue
        nd = max(int(ln / step), 1)
        for j in range(nd):
            t0 = j * step; t1 = min(j * step + dash_len, ln)
            s_parts.append([sx + dx/ln*t0, sy + dy/ln*t0, sz + dz/ln*t0])
            e_parts.append([sx + dx/ln*t1, sy + dy/ln*t1, sz + dz/ln*t1])
            c_parts.append(colors[i])
    if not s_parts:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32))
    return (np.array(s_parts, dtype=np.float32),
            np.array(e_parts, dtype=np.float32),
            np.array(c_parts, dtype=np.float32))



