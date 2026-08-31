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
for _finder, _name, _ispkg in pkgutil.walk_packages(path=[_package_dir], prefix=__name__ + "."):
    _leaf = _name.rsplit(".", 1)[-1]
    if _leaf.startswith("_"):
        continue
    importlib.import_module(_name)
from core.ecs.ecs import _GIZMO_PASS_ORDER
if "gizmo" not in _GIZMO_PASS_ORDER:
    _GIZMO_PASS_ORDER.append("gizmo")
