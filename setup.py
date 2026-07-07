# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_convex_hull = Extension(
    "core._convex_hull",
    sources=["core/_convex_hull.pyx"],
    include_dirs=[numpy.get_include()],
)

ext_bvh = Extension(
    "core._bvh_build",
    sources=["core/_bvh_build.pyx"],
    include_dirs=[numpy.get_include()],
)

ext_raytrace_data = Extension(
    "core._raytracing_data",
    sources=["core/_raytracing_data.pyx"],
    include_dirs=[numpy.get_include()],
)

ext_math_helpers = Extension(
    "core.math_helpers",
    sources=["core/math_helpers.pyx"],
    include_dirs=[numpy.get_include()],
)

ext_culling = Extension(
    "core._culling",
    sources=["core/_culling.pyx"],
    include_dirs=[numpy.get_include()],
)

ext_transform_batch = Extension(
    "core._transform_batch",
    sources=["core/_transform_batch.pyx"],
    include_dirs=[numpy.get_include()],
)

setup(
    name="ZarinEngine-cython-extensions",
    ext_modules=cythonize([ext_convex_hull, ext_bvh, ext_raytrace_data, ext_math_helpers, ext_culling, ext_transform_batch], language_level="3"),
)
