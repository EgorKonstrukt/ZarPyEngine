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

setup(
    name="ZarinEngine-cython-extensions",
    ext_modules=cythonize([ext_convex_hull, ext_bvh], language_level="3"),
)
