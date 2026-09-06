# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QWidget


class InstrumentListWidget(QListWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.setWordWrap(False)

    def load_song(self, song) -> None:
        self.clear()
        if song is None:
            return
        fmt = getattr(song, "file_extension", "mod")
        try:
            if fmt == "xm":
                items = []
                for i, inst in enumerate(getattr(song, "instruments", []) or []):
                    name = getattr(inst, "name", "") or ""
                    nsamples = len(getattr(inst, "samples", []) or [])
                    items.append((f"{i + 1:02d}  {name}", f"samples={nsamples}"))
                for label, detail in items:
                    self.addItem(f"{label}   {detail}")
            else:
                items = []
                samples = getattr(song, "samples", []) or []
                for i, smp in enumerate(samples):
                    name = getattr(smp, "name", "") or ""
                    wave = getattr(smp, "waveform", []) or []
                    vol = int(getattr(smp, "volume", 64) or 64)
                    items.append((f"{i + 1:02d}  {name}", f"vol={vol}", f"len={len(wave)}"))
                for label, detail, length in items:
                    self.addItem(f"{label}   {detail}   {length}")
        except Exception:
            self.addItem("(no instrument data)")