# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QLabel, QLineEdit, QWidget


class ModuleInfoWidget(QWidget):
    metadataEdited = pyqtSignal(str, str)

    _EDITABLE_KEYS = ("songname", "artist")

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._song = None
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(2)
        self._widgets: dict[str, QWidget] = {}

    def load_song(self, song) -> None:
        self._song = song
        info = {}
        try:
            info = dict(song.get_song_info())
        except Exception:
            pass
        keys = [
            ("songname", "Title"),
            ("artist", "Artist"),
            ("format", "Format"),
            ("n_channels", "Channels"),
            ("n_patterns", "Patterns"),
            ("sequence_length", "Sequence"),
            ("speed", "Ticks/Row"),
            ("bpm", "BPM"),
            ("duration_seconds", "Duration (s)"),
        ]
        self._ensure_rows(keys)
        for key, label in keys:
            val = info.get(key, "-")
            if isinstance(val, float):
                val = f"{val:.2f}"
            text = "-" if val is None else str(val)
            w = self._widgets[key]
            if key in self._EDITABLE_KEYS:
                w.blockSignals(True)
                w.setText(text)
                w.blockSignals(False)
            else:
                w.setText(text)

    def _ensure_rows(self, keys) -> None:
        for i, (key, label) in enumerate(keys):
            if key in self._widgets:
                name_item = self._grid.itemAtPosition(i, 0)
                if name_item is not None and isinstance(name_item.widget(), QLabel):
                    name_item.widget().setText(label + ":")
                continue
            self._grid.addWidget(QLabel(label + ":"), i, 0)
            if key in self._EDITABLE_KEYS:
                editor = QLineEdit()
                editor.setMaxLength(64)
                editor.editingFinished.connect(lambda k=key, e=editor: self._on_edited(k, e))
                self._grid.addWidget(editor, i, 1)
                self._widgets[key] = editor
            else:
                val = QLabel("-")
                self._grid.addWidget(val, i, 1)
                self._widgets[key] = val
        self._grid.setColumnStretch(1, 1)

    def _on_edited(self, key: str, editor: QLineEdit) -> None:
        text = editor.text().strip()
        if self._song is not None:
            try:
                if key == "songname":
                    self._song.set_songname(text)
                elif key == "artist":
                    self._song.set_artist(text)
            except Exception:
                return
        self.metadataEdited.emit(key, text)