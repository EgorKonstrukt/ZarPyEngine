# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.math.math3d import Vec3


class AABB:
    __slots__ = ("min", "max")
    def __init__(self, min_vec: Vec3 = None, max_vec: Vec3 = None):
        if min_vec is None:
            min_vec = Vec3(-0.5, -0.5, -0.5)
        if max_vec is None:
            max_vec = Vec3(0.5, 0.5, 0.5)
        self.min = min_vec
        self.max = max_vec

    @staticmethod
    def from_center_size(center: Vec3, size: Vec3) -> AABB:
        half = size * 0.5
        return AABB(center - half, center + half)

    def center(self) -> Vec3:
        return Vec3(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5
        )

    def half_extents(self) -> Vec3:
        return Vec3(
            (self.max.x - self.min.x) * 0.5,
            (self.max.y - self.min.y) * 0.5,
            (self.max.z - self.min.z) * 0.5
        )

    def intersects(self, other: AABB) -> bool:
        return (self.min.x <= other.max.x and self.min.y <= other.max.y and self.min.z <= other.max.z and
                self.max.x >= other.min.x and self.max.y >= other.min.y and self.max.z >= other.min.z)

    def contains(self, other: AABB) -> bool:
        return (self.min.x <= other.min.x and self.min.y <= other.min.y and self.min.z <= other.min.z and
                self.max.x >= other.max.x and self.max.y >= other.max.y and self.max.z >= other.max.z)

    def contains_point(self, p: Vec3) -> bool:
        return (self.min.x <= p.x and self.min.y <= p.y and self.min.z <= p.z and
                self.max.x >= p.x and self.max.y >= p.y and self.max.z >= p.z)

    @staticmethod
    def _coord(v: Vec3, i: int) -> float:
        if i == 0: return v.x
        if i == 1: return v.y
        return v.z

    def intersects_ray(self, origin: Vec3, direction: Vec3, max_dist: float) -> float:
        t_near = -1e30
        t_far = 1e30
        for i in range(3):
            d = self._coord(direction, i)
            o = self._coord(origin, i)
            mn = self._coord(self.min, i)
            mx = self._coord(self.max, i)
            if abs(d) < 1e-12:
                if o < mn or o > mx:
                    return -1.0
                continue
            inv = 1.0 / d
            t1 = (mn - o) * inv
            t2 = (mx - o) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_near: t_near = t1
            if t2 < t_far:  t_far = t2
            if t_near > t_far or t_far < 0:
                return -1.0
        if t_near < 0:
            t_near = t_far
        return t_near if t_near <= max_dist else -1.0

    def expanded(self, margin: float) -> AABB:
        m = Vec3(margin, margin, margin)
        return AABB(self.min - m, self.max + m)

    def __repr__(self):
        return f"AABB(min={self.min}, max={self.max})"


class OctreeNode:
    __slots__ = ("center", "half", "depth", "max_depth", "max_objects",
                 "objects", "children", "_is_leaf")

    def __init__(self, center: Vec3, half: Vec3, depth: int = 0,
                 max_depth: int = 8, max_objects: int = 8):
        self.center = center
        self.half = half
        self.depth = depth
        self.max_depth = max_depth
        self.max_objects = max_objects
        self.objects: dict[str, AABB] = {}
        self.children: list[OctreeNode] = []
        self._is_leaf = True

    def aabb(self) -> AABB:
        return AABB(self.center - self.half, self.center + self.half)

    def _get_child_index(self, obj_aabb: AABB) -> int:
        c = self.center
        mx = (obj_aabb.min.x + obj_aabb.max.x) * 0.5
        my = (obj_aabb.min.y + obj_aabb.max.y) * 0.5
        mz = (obj_aabb.min.z + obj_aabb.max.z) * 0.5
        idx = 0
        if mx >= c.x: idx |= 1
        if my >= c.y: idx |= 2
        if mz >= c.z: idx |= 4
        return idx

    def _subdivide(self):
        h = Vec3(self.half.x * 0.5, self.half.y * 0.5, self.half.z * 0.5)
        c = self.center
        offsets = [
            (-1, -1, -1), (1, -1, -1), (-1, 1, -1), (1, 1, -1),
            (-1, -1, 1),  (1, -1, 1),  (-1, 1, 1),  (1, 1, 1),
        ]
        for ox, oy, oz in offsets:
            child_center = Vec3(c.x + ox * h.x, c.y + oy * h.y, c.z + oz * h.z)
            self.children.append(OctreeNode(child_center, h, self.depth + 1,
                                            self.max_depth, self.max_objects))
        self._is_leaf = False
        for eid, aabb in list(self.objects.items()):
            ci = self._get_child_index(aabb)
            self.children[ci].insert(eid, aabb)
        self.objects.clear()

    def insert(self, entity_id: str, aabb: AABB) -> bool:
        node_aabb = self.aabb()
        if not node_aabb.contains(aabb):
            exp = aabb.expanded(0.001)
            if not node_aabb.contains(exp):
                return False
            aabb = exp
        if self._is_leaf:
            self.objects[entity_id] = aabb
            if len(self.objects) > self.max_objects and self.depth < self.max_depth:
                self._subdivide()
            return True
        ci = self._get_child_index(aabb)
        if ci < 8:
            return self.children[ci].insert(entity_id, aabb)
        self.objects[entity_id] = aabb
        return True

    def remove(self, entity_id: str) -> bool:
        if entity_id in self.objects:
            del self.objects[entity_id]
            return True
        if not self._is_leaf:
            for child in self.children:
                if child.remove(entity_id):
                    return True
        return False

    def query(self, aabb: AABB, results: list[str]):
        if not self.aabb().intersects(aabb):
            return
        for eid in self.objects:
            if aabb.intersects(self.objects[eid]):
                results.append(eid)
        if not self._is_leaf:
            for child in self.children:
                child.query(aabb, results)

    def query_point(self, point: Vec3, results: list[str]):
        if not self.aabb().contains_point(point):
            return
        for eid, eaabb in self.objects.items():
            if eaabb.contains_point(point):
                results.append(eid)
        if not self._is_leaf:
            for child in self.children:
                child.query_point(point, results)

    def raycast(self, origin: Vec3, direction: Vec3, max_dist: float,
                results: list[tuple[str, float]]) -> float:
        t = self.aabb().intersects_ray(origin, direction, max_dist)
        if t < 0:
            return -1.0
        closest = max_dist
        for eid, eaabb in self.objects.items():
            t2 = eaabb.intersects_ray(origin, direction, closest)
            if t2 >= 0:
                results.append((eid, t2))
                if t2 < closest:
                    closest = t2
        if not self._is_leaf:
            for child in self.children:
                ct = child.raycast(origin, direction, closest, results)
                if ct >= 0 and ct < closest:
                    closest = ct
        return closest

    def clear(self):
        self.objects.clear()
        self.children.clear()
        self._is_leaf = True


class Octree:
    def __init__(self, world_size: float = 500.0, max_depth: int = 8, max_objects: int = 8):
        half = world_size * 0.5
        self._root = OctreeNode(Vec3(0, 0, 0), Vec3(half, half, half),
                                0, max_depth, max_objects)
        self._object_count = 0

    def insert(self, entity_id: str, aabb: AABB) -> bool:
        r = self._root.insert(entity_id, aabb)
        if r:
            self._object_count += 1
        return r

    def remove(self, entity_id: str) -> bool:
        r = self._root.remove(entity_id)
        if r:
            self._object_count -= 1
        return r

    def query(self, aabb: AABB) -> list[str]:
        results = []
        self._root.query(aabb, results)
        return results

    def query_point(self, point: Vec3) -> list[str]:
        results = []
        self._root.query_point(point, results)
        return results

    def raycast(self, origin: Vec3, direction: Vec3, max_dist: float = 100.0) -> list[tuple[str, float]]:
        results = []
        self._root.raycast(origin, direction, max_dist, results)
        results.sort(key=lambda x: x[1])
        return results

    def clear(self):
        self._root.clear()
        self._object_count = 0

    @property
    def object_count(self) -> int:
        return self._object_count
