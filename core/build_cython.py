# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_CYTHON = os.path.join(_HERE, "pyx")

def _ext(name, src):
    return Extension(name, sources=[os.path.join(_CYTHON, src)],
                     include_dirs=[numpy.get_include()])

EXTENSIONS = [
    _ext("core._convex_hull", "_convex_hull.pyx"),
    _ext("core._bvh_build", "_bvh_build.pyx"),
    _ext("core._raytracing_data", "_raytracing_data.pyx"),
    _ext("core.math_helpers", "math_helpers.pyx"),
    _ext("core._math_vec", "_math_vec.pyx"),
    _ext("core._culling", "_culling.pyx"),
    _ext("core._transform_batch", "_transform_batch.pyx"),
    _ext("core._render_utils", "_render_utils.pyx"),
    _ext("core._physics_utils", "_physics_utils.pyx"),
    _ext("core._core_batch", "_core_batch.pyx"),
    _ext("core._types", "_types.pyx"),
    _ext("core._ecs_batch", "_ecs_batch.pyx"),
    _ext("core._render_batch", "_render_batch.pyx"),
    _ext("core._octree_batch", "_octree_batch.pyx"),
    _ext("core._constraint_batch", "_constraint_batch.pyx"),
    _ext("core._curve_batch", "_curve_batch.pyx"),
    _ext("core._constraint_update", "_constraint_update.pyx"),
    _ext("core._physics_sync", "_physics_sync.pyx"),
    _ext("core._mesh_import", "_mesh_import.pyx"),
    _ext("core._skinning", "_skinning.pyx"),
    _ext("core._audio_dsp_cy", "_audio_dsp.pyx"),
    _ext("core._raycast", "_raycast.pyx"),
    _ext("core._shadow_batch", "_shadow_batch.pyx"),
]

def build():
    setup(
        name="ZarinEngine-cython-extensions",
        ext_modules=cythonize(EXTENSIONS, language_level="3"),
    )
