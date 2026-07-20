# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os
from pathlib import Path
from PyQt6.QtCore import Qt, QRectF
from core.config.constants import APP_VERSION_DISPLAY
from core.input.constants import (KEY_W, KEY_A, KEY_S, KEY_D, KEY_Q, KEY_E, KEY_R, KEY_F,
                                  KEY_SHIFT, KEY_DELETE, KEY_CTRL, KEY_ALT, KEY_SPACE,
                                  MOUSE_L, MOUSE_R, MOUSE_M, MOUSE_LEFT, MOUSE_RIGHT, MOUSE_MIDDLE)

IPC_HOST = "127.0.0.1"
IPC_PORT = 9101

PROJECTS_DB_PATH = os.path.join(str(Path.home()), ".zarin", "projects.json")

MIN_THUMB = 24
MAX_THUMB = 128
VIEW_ICON = 0
VIEW_LIST = 1
VIEW_DETAILS = 2

THUMB_SIZE = 80
PREVIEW_SIZE = 160

EXTENSIONS = {
    ".zpes": {
        "description": "Zarin Engine Scene",
        "prog_id": "ZarinEngine.Scene",
    },
    ".zpep": {
        "description": "Zarin Engine Prefab",
        "prog_id": "ZarinEngine.Prefab",
    },
    ".zterr": {
        "description": "Zarin Engine Terrain Graph",
        "prog_id": "ZarinEngine.TerrainGraph",
    },
}

SPLASH_WIDTH = 640
SPLASH_HEIGHT = 420
SPLASH_RADIUS = 18
SPLASH_OPACITY = 1.0
SPLASH_WINDOW_FLAGS = (
    Qt.WindowType.FramelessWindowHint |
    Qt.WindowType.WindowStaysOnTopHint |
    Qt.WindowType.SplashScreen
)

LOGO_TARGET_WIDTH = 560
LOGO_VIEWBOX = QRectF(0, 0, 700, 260)

PROGRESS_BAR_WIDTH = 380
PROGRESS_BAR_HEIGHT = 8
PROGRESS_BAR_Y_OFFSET = 60
PROGRESS_BAR_RADIUS = 4
PROGRESS_FILL_RADIUS = 3

STATUS_TEXT_Y_OFFSET = 85
STATUS_TEXT_MARGIN = 30

LOGO_Y = 12
DID_YOU_KNOW_Y = 245
VERSION_Y = 275
ACCENT_BAR_Y = 305

DID_YOU_KNOW_WIDTH_MAX = 540

TIPS = [
    "Did you know? Press Ctrl+Z to undo, Ctrl+Shift+Z to redo any action.",
    "Did you know? Right-click in the viewport to create entities via the context menu.",
    "Did you know? Hold middle mouse and drag to orbit the viewport camera.",
    "Did you know? Scroll to zoom, Ctrl+middle-drag to pan in the viewport.",
    "Did you know? Press F2 to rename any entity in the Hierarchy or Project panel.",
    "Did you know? Press Delete to remove selected entities from the scene.",
    "Did you know? The Inspector supports Vec2, Vec3, Quat, color pickers, and curve editors.",
    "Did you know? The Console groups duplicate messages and supports level filtering.",
    "Did you know? The Profiler shows live frame time breakdowns per system.",
    "Did you know? Prefabs let you reuse complex objects вЂ” drag them from Project into the scene.",
    "Did you know? The Project panel has Icon, List, and Details view modes.",
    "Did you know? Press F5 in the Project panel to refresh the file listing.",
    "Did you know? Press Alt+Up in the Project panel to navigate to the parent directory.",
    "Did you know? Press Home in the Project panel to jump to the project root.",
    "Did you know? The Undo History panel lets you click any point to seek to that state.",
    "Did you know? The gizmo supports Translate (W), Rotate (E), and Scale (R) modes.",
    "Did you know? Hold Shift while using the gizmo to snap to configured grid values.",
    "Did you know? Collider wireframes are drawn directly in the viewport.",
    "Did you know? Camera frustum gizmos show what each camera sees in real-time.",
    "Did you know? Particle emitters display cone, sphere, box, and circle shape gizmos.",
    "Did you know? Audio sources draw min/max distance spheres in the viewport.",
    "Did you know? Scripts can draw custom gizmo lines via the gizmo_lines() API.",
    "Did you know? Physics runs on a separate background thread for smooth performance.",
    "Did you know? The engine supports Box, Sphere, Capsule, Mesh colliders and 2D variants.",
    "Did you know? The Collaboration feature lets you host or join peer-to-peer editing sessions.",
    "Did you know? The Play Window opens a separate viewport for play-mode rendering.",
    "Did you know? The Terminal panel supports PowerShell and Python REPL modes.",
    "Did you know? You can manage plugins via the Plugin Manager вЂ” enable/disable at runtime.",
    "Did you know? The engine includes 10 constraint components like AimConstraint and ParentConstraint.",
    "Did you know? Area-select multiple entities by click-dragging in the viewport.",
    "Did you know? Drop prefab files from Project onto the viewport to instantiate them.",
    "Did you know? The Axis Gizmo in the corner snaps the camera to any axis on click.",
    "Did you know? Component icons float above entities showing camera, light, and audio types.",
    "Did you know? Drag and drop dock panels anywhere to rearrange your workspace layout.",
    "Did you know? The engine has built-in ECS with tags, layers, parenting, and prefab support.",
    "Did you know? You can toggle wireframe overlay mode from the viewport toolbar.",
    "Did you know? The Input system supports Unity-style GetKey, GetButton, and GetAxis.",
    "Did you know? Audio supports 3D positioning, doppler effect, and reverb zones via OpenAL.",
    "Did you know? The engine supports modular shader parsing with auto-recompilation.",
    "Did you know? The build system uses Nuitka to compile standalone executables.",
]

BG_GRADIENT = [
    (0.00, 140, 110, 200, 170),
    (0.20, 110, 85, 165, 172),
    (0.45, 70, 60, 120, 175),
    (0.70, 42, 40, 78, 175),
    (1.00, 24, 24, 48, 175),
]

GLOW_SPOTS = [
    (255, 140, 0, 55),
    (220, 20, 60, 45),
    (139, 195, 74, 40),
    (120, 144, 156, 35),
]

LOGO_GLOW = (200, 150, 255, 50)
CENTER_GLOW = (208, 208, 232, 18)

TEXT_TIP = (180, 195, 220, 255)
TEXT_VERSION = (255, 210, 100, 255)
TEXT_STATUS = (208, 208, 232, 255)

ACCENT_BAR_COLORS = [
    (0.00, 255, 140, 0, 255),
    (0.25, 220, 20, 60, 255),
    (0.50, 139, 195, 74, 255),
    (0.75, 120, 144, 156, 255),
    (1.00, 255, 140, 0, 255),
]

PB_TRACK = (62, 62, 100, 255)
PB_TRACK_FILL = (24, 24, 48, 255)

PB_FILL_COLORS = [
    (0.00, 255, 179, 71, 255),
    (0.40, 220, 20, 60, 255),
    (0.70, 139, 195, 74, 255),
    (1.00, 120, 144, 156, 255),
]
