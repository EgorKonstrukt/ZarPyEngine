# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os
import sys

from core.foundation.plugin_manager import PluginBase
from core.components.inspector_meta import FieldType, InspectorField

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_libs_dir = os.path.join(_pkg_dir, "libs")
if _libs_dir not in sys.path:
    sys.path.insert(0, _libs_dir)

from .tracker_audio_source import TrackerAudioSource  # noqa: E402


class TrackerMusicPlugin(PluginBase):
    NAME = "TrackerMusicPlugin"
    VERSION = "1.0.0"
    DESCRIPTION = "Tracker-based music (MOD/XM/S3M) playback and editing"
    SYSTEM = False

    def __init__(self):
        super().__init__()
        self._editor_widget = None

    def initialize(self, engine):
        super().initialize(engine)
        self.register_component(TrackerAudioSource)
        self.register_dock("Tracker Editor", self._create_editor, area="bottom", tab_group="Audio")
        self.add_menu_item("Edit", "Open Tracker Editor", self._open_editor)
        self.add_toolbar_button("Tracker", self._open_editor, tooltip="Open tracker music editor")
        self.extend_file_filter("AudioSource", "clip_path", "Tracker modules (*.mod *.xm *.s3m *.it)")
        self.register_file_opener(
            [".mod", ".xm", ".s3m", ".it"], self.open_module, "Tracker Editor")
        patcher = self._patch_audio_source()
        if patcher is not None:
            self.patches.adopt(patcher)
        try:
            from . import renderer as _r
        except Exception:
            pass

    def _patch_audio_source(self):
        try:
            from .tracker_audio_source import patch_audio_source_play
            return patch_audio_source_play()
        except Exception as e:
            from core.foundation.logger import Logger
            Logger.error(f"TrackerMusicPlugin patch failed: {e}", e)
            return None

    def _create_editor(self):
        if self._editor_widget is not None:
            return self._editor_widget
        try:
            from .editor.tracker_editor_widget import TrackerEditorWidget
            self._editor_widget = TrackerEditorWidget(self._engine, self)
        except Exception as e:
            from core.foundation.logger import Logger
            Logger.error(f"Tracker editor init failed: {e}", e)
            self._editor_widget = None
        return self._editor_widget

    def _open_editor(self):
        self._create_editor()
        self._show_editor_dock()

    def open_module(self, path: str) -> bool:
        try:
            widget = self._create_editor()
            if widget is None:
                return False
            self._show_editor_dock()
            try:
                widget._load_module(path)
            except Exception as e:
                from core.foundation.logger import Logger
                Logger.error(f"TrackerMusicPlugin failed to open '{path}': {e}", e)
                return False
            return True
        except Exception:
            return False

    def _show_editor_dock(self):
        if self._editor_widget is None:
            return
        from PyQt6.QtWidgets import QDockWidget
        dock = None
        cur = self._editor_widget.parent()
        while cur is not None:
            if isinstance(cur, QDockWidget):
                dock = cur
                break
            cur = cur.parent()
        if dock is not None:
            dock.show()
            dock.raise_()

    def shutdown(self):
        if self._editor_widget is not None:
            try:
                self._editor_widget._stop()
            except Exception:
                pass
        super().shutdown()


def get_plugin():
    return TrackerMusicPlugin()