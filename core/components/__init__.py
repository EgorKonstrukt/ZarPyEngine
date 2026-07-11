# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import importlib
import pkgutil
import os

_package_dir = os.path.dirname(__file__)

for _finder, _name, _ispkg in pkgutil.walk_packages(
    path=[_package_dir], prefix=__name__ + "."
):
    _leaf = _name.rsplit(".", 1)[-1]
    if _leaf.startswith("_") or _leaf in ("inspector_meta",):
        continue
    importlib.import_module(_name)

from core.ecs.ecs import ComponentRegistry
from core.components.rendering.cameras.camera import CameraProjection
from core.components.lighting.light import LightType, LightAreaType

__all__ = ["ComponentRegistry", "CameraProjection", "LightType", "LightAreaType"]
for _name, _cls in ComponentRegistry.all().items():
    globals()[_name] = _cls
    __all__.append(_name)
