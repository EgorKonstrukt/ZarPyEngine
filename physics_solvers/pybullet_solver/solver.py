# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional, Tuple, Dict
import os
import numpy as np
from core.foundation.logger import Logger
from core.physics.physics_solver import IPhysicsSolver


class _LazyPybullet:
    _p = None
    _data = None

    def __getattr__(self, name):
        if self._p is None:
            import pybullet
            self._p = pybullet
        return getattr(self._p, name)

p = _LazyPybullet()

def _pybullet_data_path() -> str:
    if _LazyPybullet._data is None:
        import pybullet_data
        _LazyPybullet._data = pybullet_data
    return _LazyPybullet._data.getDataPath() if _LazyPybullet._data else ""


def _decimate_verts(verts: np.ndarray, max_vertices: int) -> np.ndarray:
    n = len(verts)
    if n <= max_vertices or max_vertices < 1:
        return verts
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    extent = maxs - mins
    extent = np.where(extent < 1e-8, 1.0, extent)
    target_cell_vol = extent.prod() / max_vertices
    cell_size = target_cell_vol ** (1.0 / 3.0)
    grid_res = np.maximum(1, np.ceil(extent / cell_size).astype(np.int32))
    indices = np.floor((verts - mins) / extent * grid_res).astype(np.int32)
    indices = np.clip(indices, 0, grid_res - 1)
    cell_ids = indices[:, 0] * grid_res[1] * grid_res[2] + indices[:, 1] * grid_res[2] + indices[:, 2]
    unique_ids, inverse = np.unique(cell_ids, return_inverse=True)
    centroids = np.zeros((len(unique_ids), 3), dtype=np.float32)
    np.add.at(centroids, inverse, verts)
    counts = np.bincount(inverse, minlength=len(unique_ids)).astype(np.float32)
    centroids /= counts[:, None]
    return centroids


def _load_mesh_verts(path: str) -> Optional[np.ndarray]:
    verts, _ = _load_mesh_geometry(path)
    return verts


def _load_mesh_geometry(path: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        from core.components.physics.mesh_collider import load_collision_geometry
        verts, indices, _ = load_collision_geometry(path)
        if verts is not None and len(verts) > 0:
            return verts, indices
    except Exception:
        pass
    return None, None


def _load_mesh_tscale(path: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    try:
        from core.components.physics.mesh_collider import load_collision_geometry
        verts, indices, k = load_collision_geometry(path)
        if verts is not None and len(verts) > 0:
            return verts, indices, k
    except Exception:
        pass
    return None, None, 1.0


def _transform_scale(params_scale, import_scale: float) -> tuple[float, float, float]:
    try:
        k = float(import_scale or 1.0)
    except Exception:
        k = 1.0
    if abs(k) < 1e-9:
        k = 1.0
    try:
        return (float(params_scale[0]) / k, float(params_scale[1]) / k, float(params_scale[2]) / k)
    except Exception:
        return (1.0, 1.0, 1.0)


def _convex_hull_shape(cid: int, points: np.ndarray, max_vertices: int, cache: dict, cache_key: tuple) -> int:
    pts = points
    if len(pts) > 1:
        try:
            pts = np.unique(np.ascontiguousarray(pts, dtype=np.float32), axis=0)
        except Exception:
            pass
    if max_vertices >= 4 and len(pts) > max_vertices:
        pts = _decimate_verts(pts, max_vertices)
    if len(pts) < 4:
        return -1
    try:
        shape_id = p.createCollisionShape(
            p.GEOM_MESH, vertices=pts.tolist(), physicsClientId=cid,
        )
        cache[cache_key] = shape_id
        return shape_id
    except Exception:
        return -1


class PyBulletSolver(IPhysicsSolver):
    """PyBullet implementation of the physics solver interface."""

    def __init__(self):
        self._client: Optional[int] = None
        self._initialized = False
        self._body_count = 0
        self._debug_enabled = False
        self._all_body_ids: list[int] = []
        self._gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
        self._fixed_time_step = 1.0 / 60.0
        self._num_sub_steps = 1
        self._solver_iterations = 50
        self._erp = 0.4
        self._contact_erp = 0.4
        self._friction_erp = 0.0
        self._contact_breaking_threshold = 0.02
        self._restitution = 0.0
        self._linear_damping = 0.04
        self._angular_damping = 0.04
        self._max_contacts_per_body = 64
        self._enable_sleeping = True
        self._mesh_shape_cache: dict[tuple, int] = {}
        self._body_collision_info: dict[int, tuple[int, int]] = {}

    def initialize(self, settings: Optional[dict] = None) -> bool:
        if self._initialized:
            return True
        try:
            opts = settings or {}
            mode = p.DIRECT if opts.get("headless", True) else p.GUI
            self._client = p.connect(mode)
            if self._client < 0:
                Logger.error("PyBullet failed to connect.")
                return False
            p.setAdditionalSearchPath(_pybullet_data_path())

            gx = opts.get("gravity_x", 0.0)
            gy = opts.get("gravity_y", -9.81)
            gz = opts.get("gravity_z", 0.0)
            self._gravity = (gx, gy, gz)
            p.setGravity(gx, gy, gz, physicsClientId=self._client)
            p.setRealTimeSimulation(0, physicsClientId=self._client)

            self._fixed_time_step = max(0.001, opts.get("fixed_time_step", 1.0 / 60.0))
            self._num_sub_steps = max(1, opts.get("num_sub_steps", 1))
            self._solver_iterations = max(1, opts.get("solver_iterations", 50))
            self._erp = opts.get("erp", 0.4)
            self._contact_erp = opts.get("contact_erp", 0.4)
            self._enable_sleeping = opts.get("enable_sleeping", True)

            for param_key, opt_key, default in [
                ("numSolverIterations", "solver_iterations", 50),
                ("numSubSteps", "num_sub_steps", 1),
                ("erp", "erp", 0.4),
                ("defaultContactERP", "contact_erp", 0.4),
                ("frictionERP", "friction_erp", 0.0),
                ("contactBreakingThreshold", "contact_breaking_threshold", 0.02),
                ("fixedTimeStep", "fixed_time_step", 1.0 / 60.0),
            ]:
                v = opts.get(opt_key, default)
                try:
                    p.setPhysicsEngineParameter(**{param_key: v}, physicsClientId=self._client)
                except Exception:
                    pass

            try:
                p.setPhysicsEngineParameter(
                    enableSleeping=1 if self._enable_sleeping else 0,
                    physicsClientId=self._client,
                )
                p.setPhysicsEngineParameter(
                    enableFileCaching=0,
                    physicsClientId=self._client,
                )
            except Exception:
                pass

            self._initialized = True
            Logger.info(f"PyBulletSolver initialized (client={self._client})")
            return True
        except Exception as e:
            Logger.error(f"PyBulletSolver init failed: {e}", e)
            return False

    def shutdown(self):
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None
        self._initialized = False
        self._body_count = 0
        Logger.info("PyBulletSolver shutdown.")

    @property
    def body_count(self) -> int:
        return self._body_count

    @property
    def debug_draw(self):
        return self._debug_enabled

    @debug_draw.setter
    def debug_draw(self, enabled: bool):
        self._debug_enabled = enabled
        if self._client is not None:
            if enabled:
                p.configureDebugVisualizer(
                    p.COV_ENABLE_GUI, 1, physicsClientId=self._client
                )
                p.configureDebugVisualizer(
                    p.COV_ENABLE_RENDERING, 1, physicsClientId=self._client
                )
            else:
                p.configureDebugVisualizer(
                    p.COV_ENABLE_GUI, 0, physicsClientId=self._client
                )

    def _cid(self):
        if self._client is None:
            raise RuntimeError("PyBullet solver not initialized")
        return self._client

    def step_simulation(self, dt: float):
        sub_steps = max(1, self._num_sub_steps)
        internal_dt = dt / sub_steps
        p.setPhysicsEngineParameter(
            fixedTimeStep=internal_dt,
            numSubSteps=1,
            physicsClientId=self._cid(),
        )
        for _ in range(sub_steps):
            p.stepSimulation(physicsClientId=self._cid())

    def set_gravity(self, gravity: tuple[float, float, float]):
        self._gravity = gravity
        p.setGravity(*gravity, physicsClientId=self._cid())

    def _make_shape(self, shape_type: str, shape_params: dict, is_static: bool = True) -> int:
        cid = self._cid()
        if shape_type == "box":
            size = shape_params.get("size", [1, 1, 1])
            center = shape_params.get("center", [0, 0, 0])
            half_extents = [s / 2.0 for s in size]
            return p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                collisionFramePosition=center,
                physicsClientId=cid,
            )
        elif shape_type == "sphere":
            radius = shape_params.get("radius", 0.5)
            center = shape_params.get("center", [0, 0, 0])
            return p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=radius,
                collisionFramePosition=center,
                physicsClientId=cid,
            )
        elif shape_type == "capsule":
            radius = shape_params.get("radius", 0.5)
            height = shape_params.get("height", 2.0)
            center = shape_params.get("center", [0, 0, 0])
            try:
                direction = int(shape_params.get("direction", 1))
            except Exception:
                direction = 1
            try:
                from core.physics.shape_utils import capsule_section_height, _PYBULLET_CAPSULE_EULER
                r, hsec = capsule_section_height(float(radius), float(height))
                frame_rot = p.getQuaternionFromEuler(_PYBULLET_CAPSULE_EULER.get(direction, (0.0, 0.0, 0.0)))
            except Exception:
                r, hsec, frame_rot = radius, height, (0.0, 0.0, 0.0, 1.0)
            return p.createCollisionShape(
                p.GEOM_CAPSULE,
                radius=r,
                height=hsec,
                collisionFramePosition=center,
                collisionFrameOrientation=frame_rot,
                physicsClientId=cid,
            )
        elif shape_type == "cylinder":
            radius = shape_params.get("radius", 0.5)
            height = shape_params.get("height", 1.0)
            center = shape_params.get("center", [0, 0, 0])
            return p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=height,
                collisionFramePosition=center,
                physicsClientId=cid,
            )
        elif shape_type == "plane":
            return p.createCollisionShape(
                p.GEOM_PLANE,
                planeNormal=shape_params.get("normal", [0, 1, 0]),
                physicsClientId=cid,
            )
        elif shape_type == "mesh":
            file_path = shape_params.get("file", "")
            resolved = file_path
            if not os.path.isabs(resolved):
                proj_root = os.path.normpath(
                    os.path.join(os.path.dirname(__file__), "..", "..")
                )
                candidate = os.path.normpath(os.path.join(proj_root, resolved))
                if os.path.exists(candidate):
                    resolved = candidate
            if not os.path.exists(resolved):
                try:
                    from core.components.physics.mesh_collider import _resolve_mesh_path
                    alt = _resolve_mesh_path(file_path)
                    if alt and os.path.exists(alt):
                        resolved = alt
                except Exception:
                    pass
            if not os.path.exists(resolved):
                Logger.warning(f"MeshCollider: file not found: {file_path}")
                return -1

            raw_mode = shape_params.get("collision_mode", "auto")
            try:
                collision_mode = str(raw_mode or "auto").lower()
            except Exception:
                collision_mode = "auto"
            if collision_mode not in ("auto", "mesh", "convex_hull", "box", "sphere"):
                Logger.warning(f"MeshCollider: unknown collision_mode '{raw_mode}', using 'auto'")
                collision_mode = "auto"
            try:
                max_vertices = int(shape_params.get("max_vertices", 0) or 0)
            except Exception:
                max_vertices = 0
            scale = shape_params.get("scale", [1, 1, 1])
            try:
                sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
            except Exception:
                sx = sy = sz = 1.0
            center = shape_params.get("center", [0, 0, 0])
            try:
                cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
            except Exception:
                cx = cy = cz = 0.0
            try:
                from core.components.physics.mesh_collider import _import_sig_of
                meta_sig = _import_sig_of(file_path)
            except Exception:
                meta_sig = "-"
            scale_t = (sx, sy, sz)
            center_t = (cx, cy, cz)
            cache_key = (resolved, scale_t, center_t, collision_mode, max_vertices, meta_sig)
            if cache_key in self._mesh_shape_cache:
                return self._mesh_shape_cache[cache_key]

            if collision_mode == "auto":
                collision_mode = "mesh" if is_static else "convex_hull"
            elif collision_mode == "mesh" and not is_static:
                Logger.warning(f"MeshCollider: concave mesh '{file_path}' requires a static body, using convex hull")
                collision_mode = "convex_hull"

            verts, indices, import_scale = _load_mesh_tscale(resolved)
            if verts is None or len(verts) == 0:
                Logger.warning(f"MeshCollider: could not read mesh '{file_path}'")
                return -1
            tx, ty, tz = _transform_scale(scale, import_scale)
            if not all(np.isfinite((tx, ty, tz))) or min(abs(tx), abs(ty), abs(tz)) < 1e-9:
                Logger.warning(f"MeshCollider: degenerate scale for '{file_path}'")
                return -1
            sv = verts * np.array([tx, ty, tz], dtype=np.float32)

            if collision_mode == "convex_hull":
                shape_id = _convex_hull_shape(cid, sv, max_vertices, self._mesh_shape_cache, cache_key)
                if shape_id >= 0:
                    return shape_id
                Logger.warning(f"MeshCollider: convex hull failed for '{file_path}'")
                return -1

            if collision_mode == "box":
                mins = sv.min(axis=0)
                maxs = sv.max(axis=0)
                half = np.maximum((maxs - mins) * 0.5, 1e-4).tolist()
                frame = ((mins + maxs) * 0.5 + np.array([cx, cy, cz], dtype=np.float32)).tolist()
                try:
                    shape_id = p.createCollisionShape(
                        p.GEOM_BOX,
                        halfExtents=half,
                        collisionFramePosition=frame,
                        physicsClientId=cid,
                    )
                    self._mesh_shape_cache[cache_key] = shape_id
                    return shape_id
                except Exception as e:
                    Logger.warning(f"MeshCollider box approximation failed for '{file_path}': {e}")
                    return -1

            if collision_mode == "sphere":
                bc = sv.mean(axis=0)
                radius = max(float(np.max(np.linalg.norm(sv - bc, axis=1))), 1e-4)
                frame = (bc + np.array([cx, cy, cz], dtype=np.float32)).tolist()
                try:
                    shape_id = p.createCollisionShape(
                        p.GEOM_SPHERE,
                        radius=radius,
                        collisionFramePosition=frame,
                        physicsClientId=cid,
                    )
                    self._mesh_shape_cache[cache_key] = shape_id
                    return shape_id
                except Exception as e:
                    Logger.warning(f"MeshCollider sphere approximation failed for '{file_path}': {e}")
                    return -1

            frame = [cx, cy, cz]
            concave_flags = getattr(p, "GEOM_FORCE_CONCAVE_TRIMESH", 0)
            if indices is not None and len(indices) >= 3 and len(indices) % 3 == 0:
                try:
                    shape_id = p.createCollisionShape(
                        p.GEOM_MESH,
                        vertices=sv.tolist(),
                        indices=indices.reshape(-1).tolist(),
                        collisionFramePosition=frame,
                        flags=concave_flags,
                        physicsClientId=cid,
                    )
                    self._mesh_shape_cache[cache_key] = shape_id
                    return shape_id
                except Exception as e:
                    Logger.warning(f"MeshCollider: failed to build triangle mesh for '{file_path}': {e}")
                    return -1
            if indices is not None and len(indices) % 3 != 0:
                Logger.warning(f"MeshCollider: mesh '{file_path}' has non-triangulated faces, using convex hull")
                shape_id = _convex_hull_shape(cid, sv, max_vertices, self._mesh_shape_cache, cache_key)
                if shape_id < 0:
                    Logger.warning(f"MeshCollider: convex hull failed for '{file_path}'")
                return shape_id
            Logger.warning(f"MeshCollider: mesh '{file_path}' has no triangles for collision")
            return -1
        elif shape_type == "heightfield":
            size = shape_params.get("size", [1000.0, 60.0, 1000.0])
            resolution = int(shape_params.get("resolution", 0))
            height_data = shape_params.get("height_data")
            if height_data is None or resolution < 2:
                Logger.warning("TerrainCollider: no height data available for physics")
                return -1
            try:
                arr = np.asarray(height_data, dtype=np.float32)
                if arr.ndim == 2:
                    rows = arr.shape[0]
                    cols = arr.shape[1]
                    flat = arr.flatten().astype(np.float32)
                else:
                    rows = cols = int(np.sqrt(len(arr)))
                    flat = arr.astype(np.float32)
                hf_rows = rows
                hf_cols = cols
                sx = size[0] / max(1, cols - 1)
                sz = size[2] / max(1, rows - 1)
                shape_id = p.createCollisionShape(
                    p.GEOM_HEIGHTFIELD,
                    numHeightfieldRows=hf_rows,
                    numHeightfieldColumns=hf_cols,
                    heightfieldData=flat.tolist(),
                    heightfieldScaling=[sx, sz, 1.0],
                    physicsClientId=cid,
                )
                return shape_id
            except Exception as e:
                Logger.warning(f"TerrainCollider: heightfield shape failed: {e}")
                return -1
        return -1

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
        cid = self._cid()
        is_static = (mass <= 0.0 or is_trigger or is_kinematic)
        shape_id = self._make_shape(shape_type, shape_params, is_static=is_static)
        if shape_id < 0:
            return -1

        # Visual shape (same as collision for now)
        visual_id = -1
        try:
            if shape_type == "box":
                size = shape_params.get("size", [1, 1, 1])
                half = [s / 2.0 for s in size]
                center = shape_params.get("center", [0, 0, 0])
                visual_id = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half,
                    visualFramePosition=center,
                    rgbaColor=[0.6, 0.6, 0.6, 1.0],
                    physicsClientId=cid,
                )
            elif shape_type == "sphere":
                radius = shape_params.get("radius", 0.5)
                center = shape_params.get("center", [0, 0, 0])
                visual_id = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=radius,
                    visualFramePosition=center,
                    rgbaColor=[0.6, 0.6, 0.6, 1.0],
                    physicsClientId=cid,
                )
            elif shape_type == "capsule":
                radius = shape_params.get("radius", 0.5)
                height = shape_params.get("height", 2.0)
                center = shape_params.get("center", [0, 0, 0])
                visual_id = p.createVisualShape(
                    p.GEOM_CAPSULE,
                    radius=radius,
                    height=height,
                    visualFramePosition=center,
                    rgbaColor=[0.6, 0.6, 0.6, 1.0],
                    physicsClientId=cid,
                )
            elif shape_type == "cylinder":
                radius = shape_params.get("radius", 0.5)
                height = shape_params.get("height", 1.0)
                center = shape_params.get("center", [0, 0, 0])
                visual_id = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    length=height,
                    visualFramePosition=center,
                    rgbaColor=[0.6, 0.6, 0.6, 1.0],
                    physicsClientId=cid,
                )
        except Exception:
            pass

        if mass <= 0.0 or is_trigger or is_kinematic:
            mass = 0.0

        base_visual = visual_id if visual_id >= 0 else -1
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=base_visual,
            basePosition=position,
            baseOrientation=p.getQuaternionFromEuler(rotation),
            physicsClientId=cid,
        )

        if body_id >= 0:
            self._body_count += 1
            self._all_body_ids.append(body_id)
            self._body_collision_info[body_id] = (collision_layer, collision_mask)
            p.changeDynamics(
                body_id,
                -1,
                lateralFriction=friction,
                restitution=restitution,
                activationState=1,
                physicsClientId=cid,
            )
            p.addUserData(body_id, "entity_id", entity_id, physicsClientId=cid)
            self._apply_collision_filters(body_id)

        return body_id

    def create_compound_rigid_body(
        self,
        entity_id: str,
        shapes: list,
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
        cid = self._cid()
        if not shapes:
            return -1
        if len(shapes) == 1:
            first = shapes[0]
            return self.create_rigid_body(
                entity_id=entity_id,
                shape_type=first.get("type", "box"),
                shape_params=first.get("params", {}),
                position=position,
                rotation=rotation,
                mass=mass,
                friction=friction,
                restitution=restitution,
                is_trigger=is_trigger,
                is_kinematic=is_kinematic,
                collision_layer=collision_layer,
                collision_mask=collision_mask,
            )
        try:
            if any(bool(s.get("is_trigger", False)) != bool(is_trigger) for s in shapes):
                Logger.warning("MeshCollider compound: mixed trigger/solid colliders share one body, using the first")
        except Exception:
            pass
        is_static = (mass <= 0.0 or is_trigger or is_kinematic)
        try:
            from core.physics.shape_utils import part_volume
            vols = []
            for s in shapes:
                try:
                    vols.append(max(float(part_volume(s.get("type", "box"), s.get("params", {}))), 1e-9))
                except Exception:
                    vols.append(1.0)
        except Exception:
            vols = [1.0] * len(shapes)
        total_vol = sum(vols) or float(len(shapes))

        base_shape = self._make_shape(shapes[0].get("type", "box"), shapes[0].get("params", {}), is_static=is_static)
        if base_shape < 0:
            return -1
        link_shapes: list[int] = []
        link_masses: list[float] = []
        for s, v in zip(shapes[1:], vols[1:]):
            try:
                sid = self._make_shape(s.get("type", "box"), s.get("params", {}), is_static=is_static)
            except Exception as e:
                Logger.warning(f"Compound link failed, skipping: {e}")
                continue
            if sid < 0:
                Logger.warning("Compound link failed, skipping")
                continue
            link_shapes.append(sid)
            link_masses.append(0.0 if is_static else mass * v / total_vol)
        n_links = len(link_shapes)
        base_mass = 0.0 if (mass <= 0.0 or is_trigger or is_kinematic) else mass * vols[0] / total_vol
        try:
            body_id = p.createMultiBody(
                baseMass=base_mass,
                baseCollisionShapeIndex=base_shape,
                baseVisualShapeIndex=-1,
                basePosition=position,
                baseOrientation=p.getQuaternionFromEuler(rotation),
                linkMasses=link_masses,
                linkCollisionShapeIndices=link_shapes,
                linkVisualShapeIndices=[-1] * n_links,
                linkPositions=[[0.0, 0.0, 0.0]] * n_links,
                linkOrientations=[[0.0, 0.0, 0.0, 1.0]] * n_links,
                linkInertialFramePositions=[[0.0, 0.0, 0.0]] * n_links,
                linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]] * n_links,
                linkParentIndices=[0] * n_links,
                linkJointTypes=[p.JOINT_FIXED] * n_links,
                linkJointAxis=[[0.0, 0.0, 1.0]] * n_links,
                physicsClientId=cid,
            )
        except Exception as e:
            Logger.warning(f"Compound body failed: {e}")
            return -1
        if body_id is None or body_id < 0:
            return -1
        self._body_count += 1
        self._all_body_ids.append(body_id)
        self._body_collision_info[body_id] = (collision_layer, collision_mask)
        try:
            p.changeDynamics(body_id, -1, lateralFriction=friction, restitution=restitution, activationState=1, physicsClientId=cid)
            for li in range(n_links):
                p.changeDynamics(body_id, li, lateralFriction=friction, restitution=restitution, physicsClientId=cid)
        except Exception:
            pass
        try:
            p.addUserData(body_id, "entity_id", entity_id, physicsClientId=cid)
        except Exception:
            pass
        self._apply_collision_filters(body_id)
        return body_id

    def remove_rigid_body(self, body_id: int):
        try:
            p.removeBody(body_id, physicsClientId=self._cid())
            self._body_count -= 1
            if body_id in self._all_body_ids:
                self._all_body_ids.remove(body_id)
            self._body_collision_info.pop(body_id, None)
        except Exception:
            pass

    def remove_all_bodies(self):
        cid = self._cid()
        for bid in list(self._all_body_ids):
            try:
                p.removeBody(bid, physicsClientId=cid)
            except Exception:
                pass
        self._all_body_ids.clear()
        self._body_collision_info.clear()
        self._body_count = 0
        # Clear mesh shape cache when removing all bodies
        self._mesh_shape_cache.clear()
        # Re-apply gravity after removing all bodies
        p.setGravity(*self._gravity, physicsClientId=cid)

    def _apply_collision_filters(self, new_body_id: int):
        cid = self._cid()
        if new_body_id not in self._body_collision_info:
            return
        new_layer, new_mask = self._body_collision_info[new_body_id]
        new_group = 1 << new_layer

        for existing_body_id, (existing_layer, existing_mask) in self._body_collision_info.items():
            if existing_body_id == new_body_id:
                continue
            existing_group = 1 << existing_layer
            should_collide = (new_group & existing_mask) != 0 and (existing_group & new_mask) != 0
            if not should_collide:
                p.setCollisionFilterPair(new_body_id, existing_body_id, -1, -1, 0, physicsClientId=cid)

    def set_body_transform(
        self,
        body_id: int,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
    ):
        cid = self._cid()
        quat = p.getQuaternionFromEuler(rotation)
        p.resetBasePositionAndOrientation(body_id, position, quat, physicsClientId=cid)

    def set_body_transform_quat(
        self,
        body_id: int,
        position: tuple[float, float, float],
        quat: tuple[float, float, float, float],
    ):
        cid = self._cid()
        p.resetBasePositionAndOrientation(body_id, position, quat, physicsClientId=cid)

    def get_body_transform(
        self, body_id: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        cid = self._cid()
        pos, quat = p.getBasePositionAndOrientation(body_id, physicsClientId=cid)
        euler = p.getEulerFromQuaternion(quat)
        return (pos[0], pos[1], pos[2]), (euler[0], euler[1], euler[2])

    def get_body_transform_quat(
        self, body_id: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        cid = self._cid()
        pos, quat = p.getBasePositionAndOrientation(body_id, physicsClientId=cid)
        return (pos[0], pos[1], pos[2]), (quat[0], quat[1], quat[2], quat[3])

    def apply_force(
        self, body_id: int, force: tuple[float, float, float], local: bool = False
    ):
        cid = self._cid()
        flags = p.LINK_FRAME if local else p.WORLD_FRAME
        p.applyExternalForce(body_id, -1, force, (0, 0, 0), flags, physicsClientId=cid)

    def apply_torque(self, body_id: int, torque: tuple[float, float, float]):
        cid = self._cid()
        p.applyExternalTorque(
            body_id, -1, torque, p.WORLD_FRAME, physicsClientId=cid
        )

    def apply_impulse(
        self, body_id: int, impulse: tuple[float, float, float], local: bool = False
    ):
        cid = self._cid()
        flags = p.LINK_FRAME if local else p.WORLD_FRAME
        p.applyExternalForce(body_id, -1, impulse, (0, 0, 0), flags, physicsClientId=cid)

    def set_velocities(
        self, body_id: int,
        linear: Optional[tuple[float, float, float]] = None,
        angular: Optional[tuple[float, float, float]] = None,
    ):
        cid = self._cid()
        kwargs = {"physicsClientId": cid}
        if linear is not None:
            kwargs["linearVelocity"] = linear
        if angular is not None:
            kwargs["angularVelocity"] = angular
        p.resetBaseVelocity(body_id, **kwargs)

    def get_velocities(
        self, body_id: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        cid = self._cid()
        vel, ang = p.getBaseVelocity(body_id, physicsClientId=cid)
        return (vel[0], vel[1], vel[2]), (ang[0], ang[1], ang[2])

    def set_velocity(self, body_id: int, velocity: tuple[float, float, float]):
        self.set_velocities(body_id, linear=velocity)

    def get_velocity(self, body_id: int) -> tuple[float, float, float]:
        return self.get_velocities(body_id)[0]

    def set_angular_velocity(
        self, body_id: int, velocity: tuple[float, float, float]
    ):
        self.set_velocities(body_id, angular=velocity)

    def get_angular_velocity(
        self, body_id: int
    ) -> tuple[float, float, float]:
        return self.get_velocities(body_id)[1]

    def ray_cast(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        max_distance: float = 100.0,
    ) -> Optional[dict]:
        cid = self._cid()
        dx, dy, dz = direction
        mag = (dx * dx + dy * dy + dz * dz) ** 0.5
        if mag < 1e-10:
            return None
        norm_dir = (dx / mag, dy / mag, dz / mag)
        to_pos = (
            origin[0] + norm_dir[0] * max_distance,
            origin[1] + norm_dir[1] * max_distance,
            origin[2] + norm_dir[2] * max_distance,
        )
        result = p.rayTest(origin, to_pos, physicsClientId=cid)
        if result:
            hit_fraction = result[0][2]
            if hit_fraction < 1.0:
                hit_pos = result[0][3]
                return {
                    "body_id": result[0][0],
                    "position": (hit_pos[0], hit_pos[1], hit_pos[2]),
                    "fraction": hit_fraction,
                    "normal": (result[0][4][0], result[0][4][1], result[0][4][2]),
                }
        return None

    def get_collision_events(self) -> list[dict]:
        cid = self._cid()
        points = p.getContactPoints(physicsClientId=cid)
        events = []
        for pt in points:
            events.append(
                {
                    "body_a": pt[1],
                    "body_b": pt[2],
                    "position": (pt[5][0], pt[5][1], pt[5][2]),
                    "normal": (pt[6][0], pt[6][1], pt[6][2]),
                    "distance": pt[7],
                    "force": pt[8],
                }
            )
        return events

    def add_plane(
        self,
        normal: tuple[float, float, float] = (0, 1, 0),
        distance: float = 0.0,
        friction: float = 0.6,
        restitution: float = 0.0,
    ) -> int:
        cid = self._cid()
        shape_id = p.createCollisionShape(
            p.GEOM_PLANE, planeNormal=normal, physicsClientId=cid
        )
        # Use a thin box for visual instead of GEOM_PLANE to avoid URDF warnings
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[50, 0.05, 50],
            rgbaColor=[0.4, 0.4, 0.4, 1.0],
            specularColor=[0.2, 0.2, 0.2],
            physicsClientId=cid,
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=visual_id,
            basePosition=(0, -distance, 0),
            physicsClientId=cid,
        )
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=friction,
            restitution=restitution,
            physicsClientId=cid,
        )
        self._body_count += 1
        return body_id

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
        cid = self._cid()
        joint_id = -1

        if joint_type == "fixed":
            joint_id = p.createConstraint(
                body_a_id, -1,
                body_b_id, -1,
                p.JOINT_FIXED,
                (0, 0, 0),
                anchor,
                childFramePosition=(0, 0, 0),
                parentFrameOrientation=p.getQuaternionFromEuler((0, 0, 0)),
                childFrameOrientation=p.getQuaternionFromEuler((0, 0, 0)),
                physicsClientId=cid,
            )
        else:
            joint_id = p.createConstraint(
                body_a_id, -1,
                body_b_id, -1,
                p.JOINT_POINT2POINT,
                (0, 0, 0),
                anchor,
                childFramePosition=(0, 0, 0),
                parentFrameOrientation=p.getQuaternionFromEuler((0, 0, 0)),
                childFrameOrientation=p.getQuaternionFromEuler((0, 0, 0)),
                physicsClientId=cid,
            )
        if joint_id >= 0:
            kwargs = {"maxForce": 100, "physicsClientId": cid}
            if joint_type == "spring":
                kwargs["erp"] = min(1.0, stiffness * 0.01)
            try:
                p.changeConstraint(joint_id, **kwargs)
            except Exception:
                pass
        return joint_id

    def remove_joint(self, joint_id: int):
        try:
            p.removeConstraint(joint_id, physicsClientId=self._cid())
        except Exception:
            pass

    def remove_all_joints(self):
        cid = self._cid()
        for bid in list(self._all_body_ids):
            try:
                num_cons = p.getNumJoints(bid, physicsClientId=cid)
                for j in range(num_cons):
                    info = p.getJointInfo(bid, j, physicsClientId=cid)
                    p.removeConstraint(info[3], physicsClientId=cid)
            except Exception:
                pass

    def change_constraint(
        self,
        constraint_id: int,
        pivot: tuple[float, float, float],
        max_force: float = 500,
    ):
        cid = self._cid()
        try:
            p.changeConstraint(
                constraint_id,
                jointChildPivot=pivot,
                maxForce=max_force,
                physicsClientId=cid,
            )
        except Exception:
            pass

    @property
    def body_count(self) -> int:
        return self._body_count

    @property
    def debug_draw(self):
        return self._debug_enabled

    @debug_draw.setter
    def debug_draw(self, enabled: bool):
        self._debug_enabled = enabled
