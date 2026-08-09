# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import moderngl
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry
from core.maths.math3d import Mat4, Vec3


@ComponentRegistry.register
class GraphicsEffect(Component):
    _registry: list[GraphicsEffect] = []
    _allow_multiple = True
    render_type: str = "additive"
    _skip_rate: int = 0
    _intensity_prop: str = ""
    _use_velocity: bool = False
    _frame_counter: int = 0

    @classmethod
    def increment_frame(cls):
        cls._frame_counter += 1

    def should_skip(self) -> bool:
        if self._skip_rate > 0:
            if (GraphicsEffect._frame_counter % (self._skip_rate + 1)) != 0:
                return True
        prop = self._intensity_prop
        if prop and hasattr(self, prop):
            v = getattr(self, prop)
            if isinstance(v, (int, float)) and v <= 0.001:
                return True
        return False

    def on_awake(self):
        if self not in self._registry:
            self._registry.append(self)

    def on_destroy(self):
        if self in self._registry:
            self._registry.remove(self)
        self._release_gl()

    def on_disable(self):
        if self in self._registry:
            self._registry.remove(self)

    def on_enable(self):
        if self not in self._registry:
            self._registry.append(self)

    def render(self, ctx: moderngl.Context,
               scene_color_tex: moderngl.Texture,
               scene_depth_tex: moderngl.Texture,
               view_mat: Mat4, proj_mat: Mat4,
               cam_pos: Vec3,
               viewport_w: int, viewport_h: int,
               input_tex: Optional[moderngl.Texture] = None,
               output_fbo: Optional[moderngl.Framebuffer] = None,
               velocity_tex: Optional[moderngl.Texture] = None,
               prev_view_proj: Optional[Mat4] = None,
               **kwargs):
        pass

    @classmethod
    def _release_cache_objects(cls, cache: dict):
        """Release all GL objects stored in a cache dict, ignoring non-GL values."""
        for entry in cache.values():
            if isinstance(entry, dict):
                for obj in entry.values():
                    if obj is not None and hasattr(obj, 'release'):
                        try:
                            obj.release()
                        except Exception:
                            pass

    @classmethod
    def clear_res_caches(cls):
        """Walk the subclass tree and release GL resources in all _res_cache / _res_prog_cache dicts."""
        seen = set()
        stack = [cls]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for cache_attr in ('_res_cache', '_res_prog_cache'):
                if hasattr(cur, cache_attr):
                    cache = getattr(cur, cache_attr)
                    cls._release_cache_objects(cache)
                    cache.clear()
            for sub in cur.__subclasses__():
                stack.append(sub)

    @classmethod
    def cleanup_registry(cls):
        """Properly destroy all registered effects (calls on_destroy + releases caches)."""
        for effect in list(cls._registry):
            effect.on_destroy()
        cls._registry.clear()
        cls.clear_res_caches()

    def _release_gl(self):
        pass
