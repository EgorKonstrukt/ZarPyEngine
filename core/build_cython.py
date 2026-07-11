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
_CYTHON = os.path.join(_HERE, "cython")

def _ext(name, src):
    return Extension(name, sources=[os.path.join(_CYTHON, src)],
                     include_dirs=[numpy.get_include()])

EXTENSIONS = [
    _ext("core._convex_hull", "_convex_hull.pyx"),
    _ext("core._bvh_build", "_bvh_build.pyx"),
    _ext("core._raytracing_data", "_raytracing_data.pyx"),
    _ext("core.math_helpers", "math_helpers.pyx"),
    _ext("core._culling", "_culling.pyx"),
    _ext("core._transform_batch", "_transform_batch.pyx"),
    _ext("core._render_utils", "_render_utils.pyx"),
    _ext("core._physics_utils", "_physics_utils.pyx"),
]

def build():
    setup(
        name="ZarinEngine-cython-extensions",
        ext_modules=cythonize(EXTENSIONS, language_level="3"),
    )
