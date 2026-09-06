# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

import os
import sys
import platform
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_CYTHON = os.path.join(_HERE, "pyx")

def _compile_args():
    if platform.system() == "Windows":
        return ["/O2", "/fp:fast", "/arch:AVX2", "/GL", "/DNDEBUG"]
    else:
        return ["-O3", "-ffast-math", "-march=native", "-DNDEBUG", "-flto", "-fopenmp"]

def _link_args():
    if platform.system() == "Windows":
        return ["/LTCG"]
    else:
        return ["-fopenmp", "-flto"]

def _ext(name, src, extra=None):
    c_args = _compile_args()
    l_args = _link_args()
    if extra:
        c_args = c_args + extra
    return Extension(name, sources=[os.path.join(_CYTHON, src)],
                     include_dirs=[numpy.get_include()],
                     extra_compile_args=c_args,
                     extra_link_args=l_args,
                     define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"), ("CYTHON_USE_PYLONG_INTERNALS", "1")])

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
    _ext("core._ik", "_ik.pyx"),
    _ext("core._audio_dsp_cy", "_audio_dsp.pyx"),
    _ext("core._raycast", "_raycast.pyx"),
    _ext("core._shadow_batch", "_shadow_batch.pyx"),
    _ext("core._scene_query", "_scene_query.pyx"),
    _ext("core._render_collect", "_render_collect.pyx"),
    _ext("core._math_mat4", "_math_mat4.pyx"),
    _ext("core._vr_batch", "_vr_batch.pyx"),
    _ext("core._nav_batch", "_nav_batch.pyx"),
]

def build():
    setup(
        name="ZarinEngine-cython-extensions",
        ext_modules=cythonize(EXTENSIONS, language_level="3",
            compiler_directives={
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
                "nonecheck": False,
                "initializedcheck": False,
                "embedsignature": False,
                "profile": False,
                "linetrace": False,
                "infer_types": True,
                "overflowcheck": False,
            },
            nthreads=os.cpu_count() or 4,
            annotate=False,
        ),
    )
