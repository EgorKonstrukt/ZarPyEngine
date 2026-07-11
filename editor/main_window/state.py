# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import json
import os
import base64

from PyQt6.QtCore import QByteArray, QRect, QSettings
from PyQt6.QtGui import QGuiApplication

from core.foundation.logger import Logger


def save_state(mw):
    path = _window_state_path()
    try:
        fg = mw.geometry()
        data = {
            "geometry": [fg.x(), fg.y(), fg.width(), fg.height()],
            "windowState": base64.b64encode(bytes(mw.saveState())).decode("ascii"),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        Logger.error(f"Failed to save window state: {e}")
    if mw._viewport.camera:
        cam_data = json.dumps(mw._viewport.camera.serialize())
        mw._settings.setValue("sceneCamera", cam_data)


def restore_camera(mw):
    path = _window_state_path()
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            g = data.get("geometry")
            if g and len(g) == 4:
                mw.setGeometry(QRect(*g))
                rect = mw.geometry()
                for screen in QGuiApplication.screens():
                    if screen.availableGeometry().intersects(rect):
                        mw._restored_geometry_once = True
                        break
            ws = data.get("windowState")
            if ws:
                raw = base64.b64decode(ws)
                if mw.restoreState(QByteArray(raw)):
                    mw._layout_restored = True
        except Exception as e:
            Logger.error(f"Failed to restore window state: {e}")
    cam_data = mw._settings.value("sceneCamera")
    if cam_data:
        try:
            mw._viewport.camera.deserialize(json.loads(cam_data))
        except Exception:
            pass


def _window_state_path():
    return os.path.join(str(os.path.expanduser("~")), ".zarin", "window_state.json")


def reset_layout(mw):
    for dock in mw._docks:
        dock.setVisible(True)
