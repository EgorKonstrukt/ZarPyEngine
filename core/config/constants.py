# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

APP_VERSION = "1.0.0"
APP_VERSION_DISPLAY = f"v{APP_VERSION}"

MAX_FIXED_STEPS = 5

PATH_FIELDS = frozenset({"mesh_path", "material_path", "clip_path", "script_path"})

FRAME_HEADER_SIZE = 4

LOGGER_MAX_ENTRIES = 2000
