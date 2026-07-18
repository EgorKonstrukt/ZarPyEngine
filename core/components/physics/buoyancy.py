# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import time
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry
from core.math.math3d import Vec3, Quat
from core.components.inspector_meta import FieldType, InspectorField


_GRAVITY_MAG = 9.81


@ComponentRegistry.register
class Buoyancy(Component):
    _icon = "Buoyancy.png"
    _gizmo_icon_color = (40, 130, 220)
    _gizmo_icon_label = "B"
    _show_gizmo_icon: bool = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("density", "Object Density", FieldType.FLOAT, min_val=1.0, max_val=20000.0, step=0.1, decimals=2),
            InspectorField("volume", "Override Volume", FieldType.FLOAT, min_val=0.0, max_val=100000.0, step=0.001, decimals=4),
            InspectorField("water_density", "Water Density", FieldType.FLOAT, min_val=1.0, max_val=2000.0, step=0.1, decimals=2),
            InspectorField("linear_drag", "Hydro Drag", FieldType.FLOAT, min_val=0.0, max_val=50.0, step=0.01, decimals=3),
            InspectorField("angular_drag", "Hydro Angular Drag", FieldType.FLOAT, min_val=0.0, max_val=50.0, step=0.01, decimals=3),
            InspectorField("flow_influence", "Flow Influence", FieldType.FLOAT, min_val=0.0, max_val=10.0, step=0.01, decimals=3),
            InspectorField("use_waves", "Use Waves", FieldType.BOOL),
            InspectorField("use_flow", "Use Flow", FieldType.BOOL),
            InspectorField("sample_resolution", "Sample Resolution", FieldType.INT, min_val=2, max_val=24, step=1),
            InspectorField("max_acceleration", "Max Accel (x g)", FieldType.FLOAT, min_val=1.0, max_val=200.0, step=0.5, decimals=1),
            InspectorField("enable_debug", "Debug Draw", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.density: float = 500.0
        self.volume: float = 0.0
        self.water_density: float = 1000.0
        self.linear_drag: float = 8.0
        self.angular_drag: float = 6.0
        self.flow_influence: float = 1.0
        self.use_waves: bool = True
        self.use_flow: bool = True
        self.sample_resolution: int = 8
        self.max_acceleration: float = 8.0
        self.enable_debug: bool = False
        self._rigidbody: Optional[object] = None
        self._cached_half_extents: Optional[Vec3] = None
        self._cached_volume: float = 0.0
        self._cached_center_local: Vec3 = Vec3.zero()
        self._submerged_fraction: float = 0.0
        self._applied_force: Vec3 = Vec3.zero()
        self._debug_points: list[Vec3] = []

    @property
    def submerged_fraction(self) -> float:
        return self._submerged_fraction

    def _get_rigidbody(self):
        ent = self._entity
        if ent is None:
            return None
        from core.components.physics.rigidbody import Rigidbody
        return ent.get_component(Rigidbody)

    def _build_collider_shape(self) -> bool:
        ent = self._entity
        if ent is None:
            return False
        from core.components.physics.box_collider import BoxCollider
        from core.components.physics.sphere_collider import SphereCollider
        from core.components.physics.capsule_collider import CapsuleCollider

        tr = self.transform
        if tr is None:
            return False

        half = None
        center = Vec3.zero()
        analytic_volume = 0.0
        bc = ent.get_component(BoxCollider)
        if bc is not None:
            sz = bc.scaled_size
            half = Vec3(abs(sz.x) * 0.5, abs(sz.y) * 0.5, abs(sz.z) * 0.5)
            center = bc.scaled_center
            analytic_volume = sz.x * sz.y * sz.z
        else:
            sc = ent.get_component(SphereCollider)
            if sc is not None:
                r = abs(sc.radius)
                half = Vec3(r, r, r)
                center = sc.center if isinstance(sc.center, Vec3) else Vec3.zero()
                analytic_volume = (4.0 / 3.0) * math.pi * r * r * r
            else:
                cc = ent.get_component(CapsuleCollider)
                if cc is not None:
                    r = abs(cc.radius)
                    h = abs(cc.height)
                    half = Vec3(r, h * 0.5 + r, r)
                    center = cc.center if isinstance(cc.center, Vec3) else Vec3.zero()
                    analytic_volume = (4.0 / 3.0) * math.pi * r * r * r + math.pi * r * r * h

        if half is None:
            return False

        if self.volume > 0.0:
            analytic_volume = self.volume
        if analytic_volume <= 0.0:
            analytic_volume = (half.x * 2.0) * (half.y * 2.0) * (half.z * 2.0)
        self._cached_half_extents = half
        self._cached_center_local = center
        self._cached_volume = analytic_volume
        return True

    def on_start(self):
        self._rigidbody = self._get_rigidbody()
        self._build_collider_shape()

    def on_enable(self):
        self._rigidbody = self._get_rigidbody()
        self._build_collider_shape()

    def on_fixed_update(self, dt: float):
        rb = self._rigidbody
        if rb is None:
            rb = self._get_rigidbody()
            self._rigidbody = rb
        tr = self.transform
        if rb is None or tr is None or rb.is_kinematic:
            return
        if self._cached_half_extents is None:
            if not self._build_collider_shape():
                return

        scene = self._scene_ref()
        if scene is None:
            return
        water_entities = self._water_volumes(scene)
        if not water_entities:
            self._submerged_fraction = 0.0
            return
        from core.components.environment.water_volume import WaterVolume
        water_vols_comps = [ent.get_component(WaterVolume) for ent in water_entities]
        water_vols_comps = [w for w in water_vols_comps if w is not None]

        t = time.time()
        world = tr.world_matrix
        d = world._d
        tx, ty, tz = float(d[3, 0]), float(d[3, 1]), float(d[3, 2])
        cl = self._cached_center_local
        center_world = Vec3(
            d[0, 0] * cl.x + d[1, 0] * cl.y + d[2, 0] * cl.z + tx,
            d[0, 1] * cl.x + d[1, 1] * cl.y + d[2, 1] * cl.z + ty,
            d[0, 2] * cl.x + d[1, 2] * cl.y + d[2, 2] * cl.z + tz,
        )

        half = self._cached_half_extents
        r = max(half.x, half.y, half.z)
        total_volume = self._cached_volume

        res = max(2, int(self.sample_resolution))
        n = res
        step = (2.0 * r) / (n - 1) if n > 1 else 2.0 * r
        sample_vol = total_volume / (n * n * n)

        m00, m10, m20 = float(d[0, 0]), float(d[1, 0]), float(d[2, 0])
        m01, m11, m21 = float(d[0, 1]), float(d[1, 1]), float(d[2, 1])
        m02, m12, m22 = float(d[0, 2]), float(d[1, 2]), float(d[2, 2])

        submerged_volume = 0.0
        buoy_center = Vec3.zero()
        submerged_count = 0
        total_samples = 0
        debug_points: list[Vec3] = []

        cx = center_world.x
        cy = center_world.y
        cz = center_world.z

        for ix in range(n):
            for iy in range(n):
                for iz in range(n):
                    ox = -r + ix * step
                    oy = -r + iy * step
                    oz = -r + iz * step
                    wx = cx + m00 * ox + m01 * oy + m02 * oz
                    wy = cy + m10 * ox + m11 * oy + m12 * oz
                    wz = cz + m20 * ox + m21 * oy + m22 * oz
                    total_samples += 1

                    wl, wden, flow = self._sample_water(water_vols_comps, wx, wy, wz, t)
                    if wl is None or wy >= wl:
                        continue
                    depth = wl - wy
                    if depth <= 0.0:
                        continue
                    submerged_volume += sample_vol
                    buoy_center = buoy_center + Vec3(wx, wy, wz)
                    submerged_count += 1
                    if self.enable_debug:
                        debug_points.append(Vec3(wx, wy, wz))

        self._debug_points = debug_points
        if submerged_count == 0:
            self._submerged_fraction = 0.0
            self._applied_force = Vec3.zero()
            return

        self._submerged_fraction = min(1.0, submerged_volume / total_volume)
        wl, wden, flow = self._sample_water(water_vols_comps, cx, cy, cz, t)

        if wl is None:
            self._submerged_fraction = 0.0
            self._applied_force = Vec3.zero()
            return

        buoy_center = buoy_center * (1.0 / submerged_count)

        rho = wden if wden > 0.0 else self.water_density
        mass = max(1e-4, rb.mass)
        displaced_weight = rho * submerged_volume * _GRAVITY_MAG
        weight = mass * _GRAVITY_MAG
        archimedes = Vec3(0.0, displaced_weight, 0.0)

        vel = rb.velocity
        ang_vel = rb.angular_velocity

        lin_drag_k = (self.linear_drag * mass + rho * submerged_volume * 0.25) * submerged_volume
        lin_visc_k = (self.linear_drag * 0.5 * mass + rho * submerged_volume * 0.5) * submerged_volume
        ang_drag_k = (self.angular_drag * mass + rho * submerged_volume * 0.25) * submerged_volume
        ang_visc_k = (self.angular_drag * 0.5 * mass + rho * submerged_volume * 0.25) * submerged_volume

        v_rel = Vec3(vel.x, vel.y, vel.z)
        if self.use_flow and flow is not None:
            v_rel = v_rel - flow * self.flow_influence

        drag = Vec3(
            -(lin_drag_k * v_rel.x * abs(v_rel.x) + lin_visc_k * v_rel.x),
            -(lin_drag_k * v_rel.y * abs(v_rel.y) + lin_visc_k * v_rel.y),
            -(lin_drag_k * v_rel.z * abs(v_rel.z) + lin_visc_k * v_rel.z),
        )

        ang_drag = Vec3(
            -(ang_drag_k * ang_vel.x * abs(ang_vel.x) + ang_visc_k * ang_vel.x),
            -(ang_drag_k * ang_vel.y * abs(ang_vel.y) + ang_visc_k * ang_vel.y),
            -(ang_drag_k * ang_vel.z * abs(ang_vel.z) + ang_visc_k * ang_vel.z),
        )

        total_force = archimedes + drag

        max_accel = self.max_acceleration * _GRAVITY_MAG
        fx, fy, fz = total_force.x, total_force.y, total_force.z
        nx, ny, nz = fx, fy - weight, fz
        nmag = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nmag > 1e-6:
            nacc = nmag / mass
            if nacc > max_accel:
                s = max_accel / nacc
                nx *= s; ny *= s; nz *= s
                total_force = Vec3(nx, ny + weight, nz)

        rb.add_force(total_force, world_space=True)
        if (abs(ang_drag.x) + abs(ang_drag.y) + abs(ang_drag.z)) > 1e-6:
            rb.add_torque(ang_drag)

        self._applied_force = total_force

    def _sample_water(self, water_vols, x: float, y: float, z: float, t: float):
        best = None
        best_level = None
        for vol in water_vols:
            if not vol.contains(x, y, z):
                continue
            wl = vol.height_at(x, z, t) if self.use_waves else vol.water_level
            if best_level is None or wl > best_level:
                best_level = wl
                best = vol
        if best is None:
            return (None, None, None)
        flow = best.flow_at(x, y, z, t) if self.use_flow else Vec3.zero()
        return (best_level, best.density, flow)

    def _scene_ref(self):
        ent = self._entity
        if ent is None or ent._scene is None:
            return None
        return ent._scene

    def _water_volumes(self, scene) -> list:
        from core.components.environment.water_volume import WaterVolume
        return scene.get_entities_with_component(WaterVolume)

    def gizmo_lines(self) -> list[tuple[Vec3, Vec3, list[float]]]:
        if not self.enable_debug or not self._debug_points:
            return []
        c = [0.2, 0.7, 1.0, 0.9]
        out = []
        for p in self._debug_points:
            out.append((p + Vec3(-0.05, 0.0, 0.0), p + Vec3(0.05, 0.0, 0.0), c))
            out.append((p + Vec3(0.0, 0.0, -0.05), p + Vec3(0.0, 0.0, 0.05), c))
        return out

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "density": self.density,
            "volume": self.volume,
            "water_density": self.water_density,
            "linear_drag": self.linear_drag,
            "angular_drag": self.angular_drag,
            "flow_influence": self.flow_influence,
            "use_waves": self.use_waves,
            "use_flow": self.use_flow,
            "sample_resolution": self.sample_resolution,
            "enable_debug": self.enable_debug,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Buoyancy:
        b = cls()
        b.enabled = data.get("enabled", True)
        b.density = data.get("density", 500.0)
        b.volume = data.get("volume", 0.0)
        b.water_density = data.get("water_density", 1000.0)
        b.linear_drag = data.get("linear_drag", 1.5)
        b.angular_drag = data.get("angular_drag", 1.2)
        b.flow_influence = data.get("flow_influence", 1.0)
        b.use_waves = data.get("use_waves", True)
        b.use_flow = data.get("use_flow", True)
        b.sample_resolution = data.get("sample_resolution", 8)
        b.enable_debug = data.get("enable_debug", False)
        return b
