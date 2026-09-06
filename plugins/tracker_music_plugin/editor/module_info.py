# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from PyQt6.QtWidgets import QGridLayout, QLabel, QWidget


class ModuleInfoWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(2)
        self._values: dict[str, QLabel] = {}

    def load_song(self, song) -> None:
        for label in list(self._values.values()):
            label.setText("-")
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
            self._values[key].setText("-" if val is None else str(val))

    def _ensure_rows(self, keys):
        for i, (key, label) in enumerate(keys):
            if key in self._values:
                name_item = self._grid.itemAtPosition(i, 0)
                if name_item is not None and isinstance(name_item.widget(), QLabel):
                    name_item.widget().setText(label + ":")
                continue
            self._grid.addWidget(QLabel(label + ":"), i, 0)
            val = QLabel("-")
            self._grid.addWidget(val, i, 1)
            self._values[key] = val
        self._grid.setColumnStretch(1, 1)