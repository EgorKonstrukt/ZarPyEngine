# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import os
import tempfile
import uuid
from typing import Optional

import numpy as np

from core.foundation.logger import Logger
from core.physics.physics_solver import IPhysicsSolver


def _sanitize_name(name: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if out and out[0].isdigit():
        out = "_" + out
    return out or "_"


def _euler_to_quat(euler: tuple[float, float, float]) -> tuple[float, float, float, float]:
    ex, ey, ez = euler
    cx, sx = math.cos(ex / 2), math.sin(ex / 2)
    cy, sy = math.cos(ey / 2), math.sin(ey / 2)
    cz, sz = math.cos(ez / 2), math.sin(ez / 2)
    return (
        sx*cy*cz - cx*sy*sz,
        cx*sy*cz + sx*cy*sz,
        cx*cy*sz - sx*sy*cz,
        cx*cy*cz + sx*sy*sz,
    )


def _quat_to_euler(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    qx, qy, qz, qw = q
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def _build_usda(
    gravity: tuple[float, float, float],
    body_specs: list[dict],
    joint_specs: list[dict] | None = None,
) -> str:
    gx, gy, gz = gravity
    lines = [
        '#usda 1.0',
        '(',
        '    defaultPrim = "World"',
        '    metersPerUnit = 1',
        '    upAxis = "Y"',
        ')',
        '',
        'def Xform "World"',
        '{',
    ]

    mag = math.sqrt(gx*gx + gy*gy + gz*gz)
    if mag > 1e-10:
        nd = (gx/mag, gy/mag, gz/mag)
    else:
        nd = (0, -1, 0)
        mag = 9.81

    lines.append('    def PhysicsScene "PhysicsScene"')
    lines.append('    {')
    lines.append(f'        vector3f physics:gravityDirection = ({nd[0]:.6f}, {nd[1]:.6f}, {nd[2]:.6f})')
    lines.append(f'        float physics:gravityMagnitude = {mag:.6f}')
    lines.append('    }')
    lines.append('')

    for b in body_specs:
        bid = b["body_id"]
        eid = b.get("entity_id", "")
        prim_name = _sanitize_name(eid) + f"_{bid}" if eid else f"body_{bid}"
        p = b["position"]
        r = b["rotation"]
        mass = b.get("mass", 1.0)
        is_kinematic = b.get("is_kinematic", False)
        is_trigger = b.get("is_trigger", False)
        shape_type = b["shape_type"]
        shape_params = b.get("shape_params", {})

        is_static = (mass <= 0 and not is_kinematic) or is_trigger

        if is_static:
            api = 'prepend apiSchemas = ["PhysicsCollisionAPI"]'
        else:
            api = 'prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]'

        lines.append(f'    def Xform "{prim_name}" (')
        lines.append(f'        {api}')
        lines.append('    )')
        lines.append('    {')
        lines.append(f'        double3 xformOp:translate = ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})')
        if not is_static:
            qw, qx, qy, qz = _euler_to_quat(r)
            lines.append(f'        quatd xformOp:orient = ({qw:.10f}, {qx:.10f}, {qy:.10f}, {qz:.10f})')
            lines.append('        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]')
        else:
            lines.append('        uniform token[] xformOpOrder = ["xformOp:translate"]')
        lines.append('')

        if not is_static:
            lines.append(f'        float physics:mass = {mass}')
            if is_kinematic:
                lines.append(f'        bool physics:kinematicEnabled = 1')
            lines.append('')

        geom_name = f"{prim_name}_G"
        if shape_type == "box":
            sx, sy, sz = shape_params.get("size", [1, 1, 1])
            cx, cy, cz = shape_params.get("center", [0, 0, 0])
            lines.append(f'        def Cube "{geom_name}" (')
            lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI"]')
            lines.append('        )')
            lines.append('        {')
            lines.append(f'            double size = 2')
            lines.append(f'            double3 xformOp:scale = ({sx/2:.6f}, {sy/2:.6f}, {sz/2:.6f})')
            if cx != 0 or cy != 0 or cz != 0:
                lines.append(f'            double3 xformOp:translate = ({cx:.6f}, {cy:.6f}, {cz:.6f})')
                lines.append(f'            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
            else:
                lines.append(f'            uniform token[] xformOpOrder = ["xformOp:scale"]')
            lines.append('        }')

        elif shape_type == "sphere":
            radius = shape_params.get("radius", 0.5)
            center = shape_params.get("center", [0, 0, 0])
            lines.append(f'        def Sphere "{geom_name}" (')
            lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI"]')
            lines.append('        )')
            lines.append('        {')
            lines.append(f'            float3[] extent = [({-radius:.6f}, {-radius:.6f}, {-radius:.6f}), ({radius:.6f}, {radius:.6f}, {radius:.6f})]')
            lines.append(f'            double radius = {radius:.6f}')
            if center[0] != 0 or center[1] != 0 or center[2] != 0:
                lines.append(f'            double3 xformOp:translate = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})')
                lines.append(f'            uniform token[] xformOpOrder = ["xformOp:translate"]')
            lines.append('        }')

        elif shape_type == "capsule":
            radius = shape_params.get("radius", 0.5)
            height = shape_params.get("height", 2.0)
            center = shape_params.get("center", [0, 0, 0])
            total_h = height + 2 * radius
            lines.append(f'        def Cylinder "{geom_name}" (')
            lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI"]')
            lines.append('        )')
            lines.append('        {')
            lines.append(f'            double height = {total_h:.6f}')
            lines.append(f'            double radius = {radius:.6f}')
            lines.append(f'            float3[] extent = [({-radius:.6f}, {-total_h/2:.6f}, {-radius:.6f}), ({radius:.6f}, {total_h/2:.6f}, {radius:.6f})]')
            if center[0] != 0 or center[1] != 0 or center[2] != 0:
                lines.append(f'            double3 xformOp:translate = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})')
                lines.append(f'            uniform token[] xformOpOrder = ["xformOp:translate"]')
            lines.append('            uniform token axis = "Y"')
            lines.append('        }')

        elif shape_type == "cylinder":
            radius = shape_params.get("radius", 0.5)
            h = shape_params.get("height", 1.0)
            center = shape_params.get("center", [0, 0, 0])
            lines.append(f'        def Cylinder "{geom_name}" (')
            lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI"]')
            lines.append('        )')
            lines.append('        {')
            lines.append(f'            double height = {h:.6f}')
            lines.append(f'            double radius = {radius:.6f}')
            lines.append(f'            float3[] extent = [({-radius:.6f}, {-h/2:.6f}, {-radius:.6f}), ({radius:.6f}, {h/2:.6f}, {radius:.6f})]')
            if center[0] != 0 or center[1] != 0 or center[2] != 0:
                lines.append(f'            double3 xformOp:translate = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})')
                lines.append(f'            uniform token[] xformOpOrder = ["xformOp:translate"]')
            lines.append('            uniform token axis = "Y"')
            lines.append('        }')

        elif shape_type == "plane":
            normal = shape_params.get("normal", [0, 1, 0])
            nx, ny, nz = normal
            axis = "Y" if abs(ny) > 0.5 else ("Z" if abs(nz) > 0.5 else "X")
            lines.append(f'        def Plane "{geom_name}" (')
            lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI"]')
            lines.append('        )')
            lines.append('        {')
            lines.append(f'            uniform token axis = "{axis}"')
            lines.append('            uniform token purpose = "guide"')
            lines.append('        }')

        elif shape_type == "mesh":
            verts = shape_params.get("vertices", [])
            if verts:
                pts_str = ", ".join(f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})" for v in verts)
                lines.append(f'        def Mesh "{geom_name}" (')
                lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]')
                lines.append('        )')
                lines.append('        {')
                lines.append('            uniform token physics:approximation = "convexHull"')
                lines.append('            bool physics:collisionEnabled = 1')
                lines.append(f'            point3f[] points = [{pts_str}]')
                n = len(verts)
                if n >= 3:
                    fc = [3] * (n - 2)
                    fi = []
                    for i in range(1, n - 1):
                        fi.extend([0, i, i + 1])
                    fc_str = ", ".join(str(c) for c in fc)
                    fi_str = ", ".join(str(i) for i in fi)
                    lines.append(f'            int[] faceVertexCounts = [{fc_str}]')
                    lines.append(f'            int[] faceVertexIndices = [{fi_str}]')
                lines.append('        }')
            else:
                file_path = shape_params.get("file", "")
                Logger.warning(f"PhysXSolver: mesh '{file_path}' needs pre-loaded vertices")

        elif shape_type == "heightfield":
            verts = shape_params.get("vertices")
            if verts:
                pts_str = ", ".join(f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})" for v in verts)
                lines.append(f'        def Mesh "{geom_name}" (')
                lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]')
                lines.append('        )')
                lines.append('        {')
                lines.append('            uniform token physics:approximation = "triangleMesh"')
                lines.append('            bool physics:collisionEnabled = 1')
                lines.append(f'            point3f[] points = [{pts_str}]')
                n = len(verts)
                if n >= 3:
                    fc = [3] * (n - 2)
                    fi = []
                    for i in range(1, n - 1):
                        fi.extend([0, i, i + 1])
                    fc_str = ", ".join(str(c) for c in fc)
                    fi_str = ", ".join(str(i) for i in fi)
                    lines.append(f'            int[] faceVertexCounts = [{fc_str}]')
                    lines.append(f'            int[] faceVertexIndices = [{fi_str}]')
                lines.append('        }')
            else:
                Logger.warning("PhysXSolver: heightfield needs pre-built vertices")

        lines.append('')
        lines.append('    }')
        lines.append('')

    # Joints
    if joint_specs:
        lines.append('')
        for j in joint_specs:
            jtype = j["joint_type"]
            jprim = j["prim_name"]
            jbody0 = j["body0_path"]
            jbody1 = j.get("body1_path")
            janchor = j["anchor"]
            jaxis = j.get("axis", (0, 0, 1))
            jlow = j.get("limit_low", -3.14159)
            jhigh = j.get("limit_high", 3.14159)
            jstiff = j.get("stiffness", 10.0)
            jdamp = j.get("damping", 1.0)
            jmax_force = j.get("max_force", 0.0)

            schema_map = {
                "fixed": "PhysicsFixedJoint",
                "revolute": "PhysicsRevoluteJoint",
                "prismatic": "PhysicsPrismaticJoint",
                "spherical": "PhysicsSphericalJoint",
                "d6": "PhysicsD6Joint",
            }
            schema = schema_map.get(jtype, "PhysicsFixedJoint")

            lines.append(f'    def {schema} "{jprim}"')
            lines.append('    {')
            lines.append(f'        rel physics:body0 = <{jbody0}>')
            if jbody1:
                lines.append(f'        rel physics:body1 = <{jbody1}>')
            lines.append(f'        point3f physics:localPos0 = ({janchor[0]:.6f}, {janchor[1]:.6f}, {janchor[2]:.6f})')
            lines.append(f'        point3f physics:localPos1 = ({janchor[0]:.6f}, {janchor[1]:.6f}, {janchor[2]:.6f})')
            lines.append(f'        quatf physics:localRot0 = (1, 0, 0, 0)')
            lines.append(f'        quatf physics:localRot1 = (1, 0, 0, 0)')

            if jtype == "revolute":
                lines.append(f'        float physics:lowerLimit = {jlow:.6f}')
                lines.append(f'        float physics:upperLimit = {jhigh:.6f}')
                lines.append(f'        float3 physics:axis = ({jaxis[0]:.6f}, {jaxis[1]:.6f}, {jaxis[2]:.6f})')
            elif jtype == "prismatic":
                lines.append(f'        float physics:lowerLimit = {jlow:.6f}')
                lines.append(f'        float physics:upperLimit = {jhigh:.6f}')
                lines.append(f'        float3 physics:axis = ({jaxis[0]:.6f}, {jaxis[1]:.6f}, {jaxis[2]:.6f})')
            elif jtype == "spherical":
                lines.append(f'        float physics:coneAngleLimit = {min(jhigh, 3.14159):.6f}')
                lines.append(f'        float3 physics:coneAxis = ({jaxis[0]:.6f}, {jaxis[1]:.6f}, {jaxis[2]:.6f})')
            elif jtype == "d6":
                pass

            if jstiff > 0:
                lines.append(f'        float physics:stiffness = {jstiff:.6f}')
            if jdamp > 0:
                lines.append(f'        float physics:damping = {jdamp:.6f}')
            if jmax_force > 0:
                lines.append(f'        float physics:breakForce = {jmax_force:.6f}')
                lines.append(f'        float physics:breakTorque = {jmax_force:.6f}')

            lines.append('    }')
            lines.append('')

    lines.append('}')
    return "\n".join(lines)


class PhysXSolver(IPhysicsSolver):
    def __init__(self):
        self._initialized = False
        self._ov_physx = None
        self._usd_handle = None
        self._usd_path: str = ""
        self._body_count = 0
        self._next_body_id = 1
        self._body_id_to_prim: dict[int, str] = {}
        self._entity_id_to_body_id: dict[str, int] = {}
        self._body_id_to_entity_id: dict[int, str] = {}
        self._gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
        self._debug_enabled = False
        self._all_body_ids: list[int] = []
        self._device = "auto"
        self._bodies_dirty = False
        self._body_specs: dict[int, dict] = {}
        self._sim_time = 0.0

        # Tensor bindings & buffers
        self._pose_tb = None
        self._vel_tb = None
        self._wrench_tb = None
        self._pose_buf: np.ndarray | None = None
        self._vel_buf: np.ndarray | None = None
        self._wrench_buf: np.ndarray | None = None

        # Dynamic body bookkeeping
        self._dynamic_prim_paths: list[str] = []
        self._prim_path_to_idx: dict[str, int] = {}

        # Pending force / velocity changes (flushed before each step)
        self._pending_vel: dict[int, tuple[float, float, float]] = {}
        self._pending_ang_vel: dict[int, tuple[float, float, float]] = {}
        self._pending_wrench: dict[int, list[float]] = {}

        # Joint bookkeeping
        self._joint_specs: dict[int, dict] = {}
        self._next_joint_id = 1
        self._joint_id_to_prim: dict[int, str] = {}
        self._joint_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, settings: Optional[dict] = None) -> bool:
        if self._initialized:
            return True
        try:
            from ovphysx import PhysX
            opts = settings or {}
            device = opts.get("physx_device", opts.get("device", "cpu"))
            self._device = device
            self._ov_physx = PhysX(device=device)
            gx = opts.get("gravity_x", 0.0)
            gy = opts.get("gravity_y", -9.81)
            gz = opts.get("gravity_z", 0.0)
            self._gravity = (gx, gy, gz)
            self._initialized = True
            Logger.info(f"PhysXSolver initialized (device={device})")
            return True
        except ImportError:
            Logger.error("PhysXSolver: ovphysx not installed. Run: pip install ovphysx")
            return False
        except Exception as e:
            Logger.error(f"PhysXSolver init failed: {e}")
            return False

    def shutdown(self):
        self._destroy_scene()
        if self._ov_physx is not None:
            try:
                self._ov_physx.release()
            except Exception:
                pass
            self._ov_physx = None
        self._initialized = False
        self._body_count = 0
        self._next_body_id = 1
        self._body_id_to_prim.clear()
        self._entity_id_to_body_id.clear()
        self._body_id_to_entity_id.clear()
        self._all_body_ids.clear()
        self._body_specs.clear()
        self._dynamic_prim_paths.clear()
        self._prim_path_to_idx.clear()
        self._pending_vel.clear()
        self._pending_ang_vel.clear()
        self._pending_wrench.clear()
        self._joint_specs.clear()
        self._joint_id_to_prim.clear()
        self._joint_count = 0
        self._next_joint_id = 1
        self._bodies_dirty = False
        Logger.info("PhysXSolver shutdown.")

    @property
    def body_count(self) -> int:
        return self._body_count

    @property
    def debug_draw(self):
        return self._debug_enabled

    @debug_draw.setter
    def debug_draw(self, enabled: bool):
        pass

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def _destroy_tensor_bindings(self):
        for tb in (self._pose_tb, self._vel_tb, self._wrench_tb):
            if tb is not None:
                try:
                    tb.destroy()
                except Exception:
                    pass
        self._pose_tb = None
        self._vel_tb = None
        self._wrench_tb = None
        self._pose_buf = None
        self._vel_buf = None
        self._wrench_buf = None
        self._dynamic_prim_paths.clear()
        self._prim_path_to_idx.clear()
        self._pending_vel.clear()
        self._pending_ang_vel.clear()
        self._pending_wrench.clear()

    def _create_tensor_bindings(self):
        self._destroy_tensor_bindings()
        if self._body_count == 0 or self._ov_physx is None:
            return

        dynamic_paths = sorted(
            prim for body_id, prim in self._body_id_to_prim.items()
            if body_id in self._body_specs
            and self._body_specs[body_id].get("mass", 0) > 0
            and not self._body_specs[body_id].get("is_trigger", False)
        )
        if not dynamic_paths:
            return

        self._dynamic_prim_paths = dynamic_paths
        self._prim_path_to_idx = {p: i for i, p in enumerate(dynamic_paths)}

        try:
            from ovphysx import TensorType
            self._pose_tb = self._ov_physx.create_tensor_binding(
                prim_paths=dynamic_paths,
                tensor_type=TensorType.RIGID_BODY_POSE,
            )
            self._vel_tb = self._ov_physx.create_tensor_binding(
                prim_paths=dynamic_paths,
                tensor_type=TensorType.RIGID_BODY_VELOCITY,
            )
            self._wrench_tb = self._ov_physx.create_tensor_binding(
                prim_paths=dynamic_paths,
                tensor_type=TensorType.RIGID_BODY_WRENCH,
            )
            self._pose_buf = np.zeros(self._pose_tb.shape, dtype=np.float32)
            self._vel_buf = np.zeros(self._vel_tb.shape, dtype=np.float32)
            self._wrench_buf = np.zeros(self._wrench_tb.shape, dtype=np.float32)
        except Exception as e:
            Logger.warning(f"PhysXSolver: could not create tensor bindings ({e})")

    def _destroy_scene(self):
        self._destroy_tensor_bindings()
        if self._usd_handle is not None and self._ov_physx is not None:
            try:
                self._ov_physx.remove_usd(self._usd_handle)
                self._ov_physx.wait_all()
            except Exception:
                pass
            self._usd_handle = None
        try:
            if self._usd_path and os.path.exists(self._usd_path):
                os.unlink(self._usd_path)
        except Exception:
            pass
        self._usd_path = ""

    def _rebuild_scene(self):
        if self._ov_physx is None:
            return
        self._destroy_scene()
        body_list = list(self._body_specs.values())
        joint_list = list(self._joint_specs.values())
        usda = _build_usda(self._gravity, body_list, joint_list)
        new_path = os.path.join(tempfile.gettempdir(), f"zarin_px_{uuid.uuid4().hex}.usda")
        with open(new_path, "w") as f:
            f.write(usda)
        try:
            handle, _ = self._ov_physx.add_usd(new_path)
            self._ov_physx.wait_all()
            self._usd_handle = handle
            self._usd_path = new_path
            self._create_tensor_bindings()
        except Exception as e:
            Logger.error(f"PhysXSolver: scene rebuild failed: {e}")
            try:
                os.unlink(new_path)
            except Exception:
                pass
        self._bodies_dirty = False

    def _ensure_scene(self):
        if self._ov_physx is None:
            return
        if self._usd_handle is None or self._bodies_dirty:
            self._rebuild_scene()

    # ------------------------------------------------------------------
    # Tensor helper
    # ------------------------------------------------------------------

    def _get_body_idx(self, body_id: int) -> int | None:
        prim = self._body_id_to_prim.get(body_id)
        if prim:
            return self._prim_path_to_idx.get(prim)
        return None

    def _is_dynamic(self, body_id: int) -> bool:
        spec = self._body_specs.get(body_id)
        return bool(spec and spec.get("mass", 0) > 0 and not spec.get("is_trigger", False))

    # ------------------------------------------------------------------
    # Pending force/velocity flush
    # ------------------------------------------------------------------

    def _flush_pending(self):
        if self._vel_tb is None and self._wrench_tb is None:
            return
        self._ensure_scene()
        if self._vel_tb is None and self._wrench_tb is None:
            return

        if self._pending_vel or self._pending_ang_vel:
            if self._vel_tb is not None:
                if self._vel_buf is None:
                    self._vel_buf = np.zeros(self._vel_tb.shape, dtype=np.float32)
                self._vel_tb.read(self._vel_buf)
                for body_id, vel in self._pending_vel.items():
                    idx = self._get_body_idx(body_id)
                    if idx is not None and idx < len(self._vel_buf):
                        self._vel_buf[idx, 0] = vel[0]
                        self._vel_buf[idx, 1] = vel[1]
                        self._vel_buf[idx, 2] = vel[2]
                for body_id, av in self._pending_ang_vel.items():
                    idx = self._get_body_idx(body_id)
                    if idx is not None and idx < len(self._vel_buf):
                        self._vel_buf[idx, 3] = av[0]
                        self._vel_buf[idx, 4] = av[1]
                        self._vel_buf[idx, 5] = av[2]
                self._vel_tb.write(self._vel_buf)
            self._pending_vel.clear()
            self._pending_ang_vel.clear()

        if self._pending_wrench:
            if self._wrench_tb is not None:
                if self._wrench_buf is None:
                    self._wrench_buf = np.zeros(self._wrench_tb.shape, dtype=np.float32)
                self._wrench_buf.fill(0.0)
                for body_id, w in self._pending_wrench.items():
                    idx = self._get_body_idx(body_id)
                    if idx is not None and idx < len(self._wrench_buf):
                        self._wrench_buf[idx, 0] = w[0]
                        self._wrench_buf[idx, 1] = w[1]
                        self._wrench_buf[idx, 2] = w[2]
                        self._wrench_buf[idx, 3] = w[3]
                        self._wrench_buf[idx, 4] = w[4]
                        self._wrench_buf[idx, 5] = w[5]
                self._wrench_tb.write(self._wrench_buf)
            self._pending_wrench.clear()

    # ------------------------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------------------------

    def step_simulation(self, dt: float):
        if self._ov_physx is None:
            return
        self._ensure_scene()
        if self._usd_handle is None:
            return
        self._flush_pending()
        try:
            self._sim_time += dt
            self._ov_physx.step(dt, self._sim_time)
            self._ov_physx.wait_all()
            self._pull_transforms()
        except Exception as e:
            Logger.warning(f"PhysXSolver step error: {e}")

    # ------------------------------------------------------------------
    # Transform sync
    # ------------------------------------------------------------------

    def _pull_transforms(self):
        if self._pose_tb is None or self._body_count == 0:
            return
        try:
            if self._pose_buf is None:
                self._pose_buf = np.zeros(self._pose_tb.shape, dtype=np.float32)
            self._pose_tb.read(self._pose_buf)
        except Exception:
            return

        for prim_path, idx in self._prim_path_to_idx.items():
            if idx >= len(self._pose_buf):
                continue
            pose = self._pose_buf[idx]
            px_, py, pz = float(pose[0]), float(pose[1]), float(pose[2])
            qx, qy, qz, qw = float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])
            body_id = next(
                (bid for bid, pp in self._body_id_to_prim.items() if pp == prim_path),
                None,
            )
            if body_id is None:
                continue
            spec = self._body_specs.get(body_id)
            if spec:
                spec["position"] = (px_, py, pz)
                euler = _quat_to_euler((qx, qy, qz, qw))
                spec["rotation"] = euler

    # ------------------------------------------------------------------
    # Gravity
    # ------------------------------------------------------------------

    def set_gravity(self, gravity: tuple[float, float, float]):
        self._gravity = gravity
        self._bodies_dirty = True

    # ------------------------------------------------------------------
    # Rigid body creation / removal
    # ------------------------------------------------------------------

    def create_rigid_body(
        self,
        entity_id: str,
        shape_type: str,
        shape_params: dict,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
        mass: float,
        friction: float = 0.6,
        restitution: float = 0.0,
        is_trigger: bool = False,
        is_kinematic: bool = False,
        collision_layer: int = 0,
        collision_mask: int = 0xFFFF,
    ) -> int:
        body_id = self._next_body_id
        self._next_body_id += 1
        prim_name = _sanitize_name(entity_id) + f"_{body_id}" if entity_id else f"body_{body_id}"
        self._body_id_to_prim[body_id] = f"/World/{prim_name}"
        spec = {
            "body_id": body_id,
            "entity_id": entity_id,
            "shape_type": shape_type,
            "shape_params": dict(shape_params),
            "position": position,
            "rotation": rotation,
            "mass": mass,
            "friction": friction,
            "restitution": restitution,
            "is_trigger": is_trigger,
            "is_kinematic": is_kinematic,
        }
        self._body_specs[body_id] = spec
        self._all_body_ids.append(body_id)
        self._body_count += 1
        if entity_id:
            self._entity_id_to_body_id[entity_id] = body_id
        self._body_id_to_entity_id[body_id] = entity_id
        self._bodies_dirty = True
        return body_id

    def remove_rigid_body(self, body_id: int):
        self._remove_joints_for_body(body_id)
        spec = self._body_specs.pop(body_id, None)
        if spec:
            eid = spec.get("entity_id", "")
            self._entity_id_to_body_id.pop(eid, None)
            self._body_id_to_entity_id.pop(body_id, None)
        self._body_id_to_prim.pop(body_id, None)
        if body_id in self._all_body_ids:
            self._all_body_ids.remove(body_id)
        self._body_count = max(0, self._body_count - 1)
        self._bodies_dirty = True

    def remove_all_bodies(self):
        self._body_specs.clear()
        self._body_id_to_prim.clear()
        self._entity_id_to_body_id.clear()
        self._body_id_to_entity_id.clear()
        self._all_body_ids.clear()
        self._body_count = 0
        self.remove_all_joints()
        self._bodies_dirty = True

    # ------------------------------------------------------------------
    # Transforms (direct tensor read/write for dynamic bodies)
    # ------------------------------------------------------------------

    def set_body_transform(
        self,
        body_id: int,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
    ):
        spec = self._body_specs.get(body_id)
        if not spec:
            return
        if self._is_dynamic(body_id) and self._pose_tb is not None:
            idx = self._get_body_idx(body_id)
            if idx is not None and self._pose_buf is not None:
                qw, qx, qy, qz = _euler_to_quat(rotation)
                self._pose_buf[idx, 0] = position[0]
                self._pose_buf[idx, 1] = position[1]
                self._pose_buf[idx, 2] = position[2]
                self._pose_buf[idx, 3] = qx
                self._pose_buf[idx, 4] = qy
                self._pose_buf[idx, 5] = qz
                self._pose_buf[idx, 6] = qw
                self._pose_tb.write(self._pose_buf)
        spec["position"] = position
        spec["rotation"] = rotation

    def set_body_transform_quat(
        self,
        body_id: int,
        position: tuple[float, float, float],
        quat: tuple[float, float, float, float],
    ):
        spec = self._body_specs.get(body_id)
        if not spec:
            return
        if self._is_dynamic(body_id) and self._pose_tb is not None:
            idx = self._get_body_idx(body_id)
            if idx is not None and self._pose_buf is not None:
                qx, qy, qz, qw = quat
                self._pose_buf[idx, 0] = position[0]
                self._pose_buf[idx, 1] = position[1]
                self._pose_buf[idx, 2] = position[2]
                self._pose_buf[idx, 3] = qx
                self._pose_buf[idx, 4] = qy
                self._pose_buf[idx, 5] = qz
                self._pose_buf[idx, 6] = qw
                self._pose_tb.write(self._pose_buf)
        spec["position"] = position
        import math
        qx, qy, qz, qw = quat
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(max(-1.0, min(1.0, sinp)))
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        spec["rotation"] = (roll, pitch, yaw)

    def get_body_transform(
        self, body_id: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if self._is_dynamic(body_id) and self._pose_tb is not None:
            idx = self._get_body_idx(body_id)
            if idx is not None:
                try:
                    if self._pose_buf is None:
                        self._pose_buf = np.zeros(self._pose_tb.shape, dtype=np.float32)
                    self._pose_tb.read(self._pose_buf)
                    if idx < len(self._pose_buf):
                        p = self._pose_buf[idx]
                        pos = (float(p[0]), float(p[1]), float(p[2]))
                        euler = _quat_to_euler((float(p[3]), float(p[4]), float(p[5]), float(p[6])))
                        return pos, euler
                except Exception:
                    pass
        spec = self._body_specs.get(body_id)
        if spec:
            return spec["position"], spec["rotation"]
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    def get_body_transform_quat(
        self, body_id: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        if self._is_dynamic(body_id) and self._pose_tb is not None:
            idx = self._get_body_idx(body_id)
            if idx is not None:
                try:
                    if self._pose_buf is None:
                        self._pose_buf = np.zeros(self._pose_tb.shape, dtype=np.float32)
                    self._pose_tb.read(self._pose_buf)
                    if idx < len(self._pose_buf):
                        p = self._pose_buf[idx]
                        pos = (float(p[0]), float(p[1]), float(p[2]))
                        quat = (float(p[3]), float(p[4]), float(p[5]), float(p[6]))
                        return pos, quat
                except Exception:
                    pass
        spec = self._body_specs.get(body_id)
        if spec:
            import math
            rx, ry, rz = spec["rotation"]
            hx, hy, hz = rx * 0.5, ry * 0.5, rz * 0.5
            sx, cx = math.sin(hx), math.cos(hx)
            sy, cy = math.sin(hy), math.cos(hy)
            sz, cz = math.sin(hz), math.cos(hz)
            qx = sx * cy * cz - cx * sy * sz
            qy = cx * sy * cz + sx * cy * sz
            qz = cx * cy * sz - sx * sy * cz
            qw = cx * cy * cz + sx * sy * sz
            return spec["position"], (qx, qy, qz, qw)
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------

    def apply_force(self, body_id: int, force: tuple[float, float, float], local: bool = False):
        if not self._is_dynamic(body_id):
            return
        if local:
            spec = self._body_specs.get(body_id)
            if spec:
                euler = spec["rotation"]
                qw, qx, qy, qz = _euler_to_quat(euler)
                v = _rotate_vector_by_quat(force, (qw, qx, qy, qz))
                force = v
        if body_id not in self._pending_wrench:
            self._pending_wrench[body_id] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        w = self._pending_wrench[body_id]
        w[0] += force[0]
        w[1] += force[1]
        w[2] += force[2]

    def apply_torque(self, body_id: int, torque: tuple[float, float, float]):
        if not self._is_dynamic(body_id):
            return
        if body_id not in self._pending_wrench:
            self._pending_wrench[body_id] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        w = self._pending_wrench[body_id]
        w[3] += torque[0]
        w[4] += torque[1]
        w[5] += torque[2]

    def apply_impulse(self, body_id: int, impulse: tuple[float, float, float], local: bool = False):
        if not self._is_dynamic(body_id):
            return
        spec = self._body_specs.get(body_id)
        mass = spec.get("mass", 1.0) if spec else 1.0
        if mass <= 0:
            return
        dv = (impulse[0] / mass, impulse[1] / mass, impulse[2] / mass)
        if local:
            if spec:
                euler = spec["rotation"]
                qw, qx, qy, qz = _euler_to_quat(euler)
                dv = _rotate_vector_by_quat(dv, (qw, qx, qy, qz))
        if body_id in self._pending_vel:
            v = self._pending_vel[body_id]
            self._pending_vel[body_id] = (v[0] + dv[0], v[1] + dv[1], v[2] + dv[2])
        else:
            idx = self._get_body_idx(body_id)
            if idx is not None and self._vel_tb is not None:
                if self._vel_buf is None:
                    self._vel_buf = np.zeros(self._vel_tb.shape, dtype=np.float32)
                self._vel_tb.read(self._vel_buf)
                if idx < len(self._vel_buf):
                    cv = self._vel_buf[idx]
                    self._pending_vel[body_id] = (
                        float(cv[0]) + dv[0],
                        float(cv[1]) + dv[1],
                        float(cv[2]) + dv[2],
                    )
            else:
                self._pending_vel[body_id] = dv

    # ------------------------------------------------------------------
    # Velocity
    # ------------------------------------------------------------------

    def set_velocity(self, body_id: int, velocity: tuple[float, float, float]):
        if self._is_dynamic(body_id):
            self._pending_vel[body_id] = velocity

    def get_velocity(self, body_id: int) -> tuple[float, float, float]:
        if not self._is_dynamic(body_id):
            return (0.0, 0.0, 0.0)
        idx = self._get_body_idx(body_id)
        if idx is None or self._vel_tb is None:
            return (0.0, 0.0, 0.0)
        try:
            if self._vel_buf is None:
                self._vel_buf = np.zeros(self._vel_tb.shape, dtype=np.float32)
            self._vel_tb.read(self._vel_buf)
            if idx < len(self._vel_buf):
                v = self._vel_buf[idx]
                return (float(v[0]), float(v[1]), float(v[2]))
        except Exception:
            pass
        return (0.0, 0.0, 0.0)

    def set_angular_velocity(self, body_id: int, velocity: tuple[float, float, float]):
        if self._is_dynamic(body_id):
            self._pending_ang_vel[body_id] = velocity

    def get_angular_velocity(self, body_id: int) -> tuple[float, float, float]:
        if not self._is_dynamic(body_id):
            return (0.0, 0.0, 0.0)
        idx = self._get_body_idx(body_id)
        if idx is None or self._vel_tb is None:
            return (0.0, 0.0, 0.0)
        try:
            if self._vel_buf is None:
                self._vel_buf = np.zeros(self._vel_tb.shape, dtype=np.float32)
            self._vel_tb.read(self._vel_buf)
            if idx < len(self._vel_buf):
                v = self._vel_buf[idx]
                return (float(v[3]), float(v[4]), float(v[5]))
        except Exception:
            pass
        return (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Ray cast
    # ------------------------------------------------------------------

    def ray_cast(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        max_distance: float = 100.0,
    ) -> Optional[dict]:
        if self._ov_physx is None:
            return None
        try:
            results = self._ov_physx.raycast(origin, direction, max_distance)
        except Exception:
            return None
        if not results:
            return None

        hit = results[0]
        hit_pos = hit["position"]

        best_body_id = None
        best_dist = float("inf")

        for body_id, spec in self._body_specs.items():
            pos = spec["position"]
            dx = hit_pos[0] - pos[0]
            dy = hit_pos[1] - pos[1]
            dz = hit_pos[2] - pos[2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best_dist:
                best_dist = d2
                best_body_id = body_id

        return {
            "body_id": best_body_id,
            "position": hit_pos,
            "normal": hit.get("normal", (0.0, 0.0, 0.0)),
            "distance": hit.get("distance", 0.0),
        }

    # ------------------------------------------------------------------
    # Collision events
    # ------------------------------------------------------------------

    def get_collision_events(self) -> list[dict]:
        if self._ov_physx is None:
            return []
        try:
            cr = self._ov_physx.get_contact_report()
        except Exception:
            return []
        if cr.get("num_headers", 0) == 0 and cr.get("num_points", 0) == 0:
            return []

        events = []
        for i in range(cr.get("num_headers", 0)):
            h = cr["headers"][i]
            try:
                body0 = h.body0 if hasattr(h, "body0") else getattr(h, "actor0", None)
                body1 = h.body1 if hasattr(h, "body1") else getattr(h, "actor1", None)
            except Exception:
                continue
            if body0 is None or body1 is None:
                continue

            ncd = getattr(h, "numContactData", 1)
            pts = []
            base = getattr(h, "baseIndex", 0)
            for j in range(base, min(base + ncd, cr.get("num_points", 0))):
                p = cr["points"][j]
                pts.append({
                    "position": getattr(p, "position", (0, 0, 0)),
                    "normal": getattr(p, "normal", (0, 0, 0)),
                    "impulse": getattr(p, "impulse", 0.0),
                    "separation": getattr(p, "separation", 0.0),
                })

            events.append({
                "body_a": body0,
                "body_b": body1,
                "type": "collision",
                "points": pts,
            })
        return events

    # ------------------------------------------------------------------
    # Planes (helper)
    # ------------------------------------------------------------------

    def add_plane(
        self,
        normal: tuple[float, float, float] = (0, 1, 0),
        distance: float = 0.0,
        friction: float = 0.6,
        restitution: float = 0.0,
    ) -> int:
        body_id = self._next_body_id
        self._next_body_id += 1
        prim_name = f"ground_{body_id}"
        self._body_id_to_prim[body_id] = f"/World/{prim_name}"
        spec = {
            "body_id": body_id,
            "entity_id": "",
            "shape_type": "plane",
            "shape_params": {"normal": list(normal)},
            "position": (0, -distance, 0),
            "rotation": (0, 0, 0),
            "mass": 0.0,
            "friction": friction,
            "restitution": restitution,
            "is_trigger": False,
            "is_kinematic": False,
        }
        self._body_specs[body_id] = spec
        self._all_body_ids.append(body_id)
        self._body_count += 1
        self._bodies_dirty = True
        return body_id

    # ------------------------------------------------------------------
    # Joints
    # ------------------------------------------------------------------

    def _remove_joints_for_body(self, body_id: int):
        to_remove = [
            jid for jid, js in self._joint_specs.items()
            if js["body_a_id"] == body_id or js.get("body_b_id") == body_id
        ]
        for jid in to_remove:
            self._joint_specs.pop(jid, None)
            self._joint_id_to_prim.pop(jid, None)
            self._joint_count = max(0, self._joint_count - 1)

    def create_joint(
        self,
        joint_type: str,
        body_a_id: int,
        body_b_id: int,
        anchor: tuple[float, float, float],
        axis: tuple[float, float, float] = (0, 0, 1),
        limit_low: float = -3.14159,
        limit_high: float = 3.14159,
        stiffness: float = 10.0,
        damping: float = 1.0,
    ) -> int:
        if self._ov_physx is None:
            return -1
        prim_a = self._body_id_to_prim.get(body_a_id)
        if prim_a is None:
            Logger.warning(f"PhysXSolver: body_a {body_a_id} not found for joint")
            return -1
        prim_b = self._body_id_to_prim.get(body_b_id) if body_b_id >= 0 else None
        joint_id = self._next_joint_id
        self._next_joint_id += 1
        jprim = f"joint_{joint_id}"

        # Compute local-space anchor offsets (world-space offset, ignoring rotation)
        spec_a = self._body_specs.get(body_a_id)
        pos_a = spec_a["position"] if spec_a else (0.0, 0.0, 0.0)
        local0 = (anchor[0] - pos_a[0], anchor[1] - pos_a[1], anchor[2] - pos_a[2])

        if prim_b and body_b_id in self._body_specs:
            pos_b = self._body_specs[body_b_id]["position"]
            local1 = (anchor[0] - pos_b[0], anchor[1] - pos_b[1], anchor[2] - pos_b[2])
        else:
            local1 = anchor

        spec = {
            "joint_id": joint_id,
            "joint_type": joint_type,
            "prim_name": jprim,
            "body_a_id": body_a_id,
            "body_b_id": body_b_id if body_b_id >= 0 else None,
            "body0_path": prim_a,
            "body1_path": prim_b if prim_b else None,
            "anchor": local0,
            "local1": local1,
            "axis": axis,
            "limit_low": limit_low,
            "limit_high": limit_high,
            "stiffness": stiffness,
            "damping": damping,
            "max_force": 0.0,
        }
        self._joint_specs[joint_id] = spec
        self._joint_id_to_prim[joint_id] = f"/World/{jprim}"
        self._joint_count += 1
        self._bodies_dirty = True
        Logger.info(f"PhysXSolver: created joint {joint_id} ({joint_type}) between {body_a_id} and {body_b_id}")
        return joint_id

    def remove_joint(self, joint_id: int):
        spec = self._joint_specs.pop(joint_id, None)
        if spec:
            self._joint_id_to_prim.pop(joint_id, None)
            self._joint_count = max(0, self._joint_count - 1)
            self._bodies_dirty = True
            Logger.info(f"PhysXSolver: removed joint {joint_id}")

    def remove_all_joints(self):
        if self._joint_specs:
            self._joint_specs.clear()
            self._joint_id_to_prim.clear()
            self._joint_count = 0
            self._bodies_dirty = True
            Logger.info("PhysXSolver: removed all joints")

    def change_constraint(
        self,
        constraint_id: int,
        pivot: tuple[float, float, float],
        max_force: float = 500,
    ):
        spec = self._joint_specs.get(constraint_id)
        if spec is None:
            Logger.warning(f"PhysXSolver: constraint {constraint_id} not found")
            return
        spec["anchor"] = pivot
        spec["max_force"] = max_force
        self._bodies_dirty = True


def _rotate_vector_by_quat(
    v: tuple[float, float, float],
    q: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    qw, qx, qy, qz = q
    vx, vy, vz = v
    uv_x = qy * vz - qz * vy
    uv_y = qz * vx - qx * vz
    uv_z = qx * vy - qy * vx
    uv2_x = qy * uv_z - qz * uv_y
    uv2_y = qz * uv_x - qx * uv_z
    uv2_z = qx * uv_y - qy * uv_x
    return (
        vx + 2 * (qw * uv_x + uv2_x),
        vy + 2 * (qw * uv_y + uv2_y),
        vz + 2 * (qw * uv_z + uv2_z),
    )
