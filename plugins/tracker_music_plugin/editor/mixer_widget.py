# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QScrollArea,
                             QSlider, QVBoxLayout, QWidget)


class LevelMeter(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._level = 0.0
        self._peak = 0.0
        self.setMinimumWidth(12)
        self.setMinimumHeight(48)

    def set_level(self, v: float):
        try:
            v = max(0.0, min(1.0, float(v)))
        except Exception:
            v = 0.0
        self._level = v
        self._peak = max(v, self._peak * 0.90)
        self.update()

    def reset(self):
        self._level = 0.0
        self._peak = 0.0
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        rect = self.rect()
        p.fillRect(rect, QColor("#1a1a1a"))
        h = max(1, rect.height())
        w = max(1, rect.width())
        lh = int(h * self._level)
        if lh > 0:
            y0 = h - lh
            if self._level > 0.9:
                col = QColor("#e5484d")
            elif self._level > 0.7:
                col = QColor("#f5a524")
            else:
                col = QColor("#46a758")
            p.fillRect(0, y0, w, lh, col)
        py = h - int(h * max(0.0, min(1.0, self._peak)))
        p.fillRect(0, max(0, py - 1), w, 2, QColor("#ffffff"))


class MixerWidget(QWidget):
    muteChanged = pyqtSignal(int, bool)
    gainChanged = pyqtSignal(int, float)
    masterChanged = pyqtSignal(float)

    MIN_DB = -60.0
    MAX_DB = 6.0

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setMinimumHeight(190)
        self._scroll.setMaximumHeight(230)
        self._strip_host = QWidget()
        self._strips_layout = QHBoxLayout(self._strip_host)
        self._strips_layout.setContentsMargins(2, 2, 2, 2)
        self._strips_layout.setSpacing(4)
        self._scroll.setWidget(self._strip_host)
        outer.addWidget(self._scroll, 1)
        self._strips: list[dict] = []
        self._muted: set[int] = set()
        self._master_db: float = 0.0
        self._master_meter: LevelMeter | None = None
        self._master_db_lbl: QLabel | None = None

    @staticmethod
    def db_text(db: float) -> str:
        return "-inf" if db <= MixerWidget.MIN_DB else f"{db:+.0f} dB"

    def set_channels(self, channels: int) -> None:
        while self._strips:
            strip = self._strips.pop()
            w = strip["widget"]
            self._strips_layout.removeWidget(w)
            w.deleteLater()
        old_master = getattr(self, "_master", None)
        if old_master is not None:
            w = old_master["widget"]
            self._strips_layout.removeWidget(w)
            w.deleteLater()
            self._master = None
        self._muted.clear()
        for ch in range(max(0, channels)):
            self._strips.append(self._make_strip(ch))
        self._master = self._make_master_strip()

    def _make_strip(self, ch: int) -> dict:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(1, 1, 1, 1)
        lay.setSpacing(1)
        name = QLabel(f"Ch {ch}")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(name)
        mid = QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        meter = LevelMeter()
        mid.addWidget(meter)
        slider = QSlider(Qt.Orientation.Vertical)
        slider.setRange(int(self.MIN_DB), int(self.MAX_DB))
        slider.setValue(0)
        slider.setToolTip(f"Channel {ch} gain")
        slider.valueChanged.connect(lambda v, c=ch: self._on_gain(c, float(v)))
        mid.addWidget(slider)
        mid_wrap = QWidget()
        mid_wrap.setLayout(mid)
        lay.addWidget(mid_wrap)
        db_lbl = QLabel(self.db_text(0.0))
        db_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(db_lbl)
        mute = QCheckBox("M")
        mute.setToolTip(f"Mute channel {ch}")
        mute.stateChanged.connect(lambda state, c=ch: self._on_toggled(c, state))
        lay.addWidget(mute, alignment=Qt.AlignmentFlag.AlignCenter)
        wrap.setMinimumWidth(56)
        self._strips_layout.addWidget(wrap)
        return {"widget": wrap, "meter": meter, "slider": slider,
                "db_lbl": db_lbl, "mute": mute, "db": 0.0}

    def _make_master_strip(self) -> dict:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(1, 1, 1, 1)
        lay.setSpacing(1)
        name = QLabel("Master")
        name.setStyleSheet("font-weight: bold;")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(name)
        mid = QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        self._master_meter = LevelMeter()
        mid.addWidget(self._master_meter)
        slider = QSlider(Qt.Orientation.Vertical)
        slider.setRange(int(self.MIN_DB), int(self.MAX_DB))
        slider.setValue(int(self._master_db))
        slider.setToolTip("Master gain")
        slider.valueChanged.connect(lambda v: self._on_master(float(v)))
        mid.addWidget(slider)
        mid_wrap = QWidget()
        mid_wrap.setLayout(mid)
        lay.addWidget(mid_wrap)
        self._master_db_lbl = QLabel(self.db_text(self._master_db))
        self._master_db_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._master_db_lbl)
        lay.addStretch()
        wrap.setMinimumWidth(64)
        self._strips_layout.addWidget(wrap)
        self._master_slider = slider
        return {"widget": wrap}

    def muted_channels(self) -> set[int]:
        return set(self._muted)

    def set_muted(self, ch: int, muted: bool) -> None:
        if 0 <= ch < len(self._strips):
            self._strips[ch]["mute"].setChecked(muted)

    def channel_db(self, ch: int) -> float:
        if 0 <= ch < len(self._strips):
            return float(self._strips[ch]["db"])
        return 0.0

    def all_gains_db(self) -> list[float]:
        return [float(s["db"]) for s in self._strips]

    def set_channel_db(self, ch: int, db: float) -> None:
        if 0 <= ch < len(self._strips):
            slider = self._strips[ch]["slider"]
            slider.blockSignals(True)
            slider.setValue(int(max(self.MIN_DB, min(self.MAX_DB, db))))
            slider.blockSignals(False)
            self._strips[ch]["db"] = float(db)
            self._strips[ch]["db_lbl"].setText(self.db_text(db))

    def master_db(self) -> float:
        return float(self._master_db)

    def set_master_db(self, db: float) -> None:
        self._master_db = float(max(self.MIN_DB, min(self.MAX_DB, db)))
        if hasattr(self, "_master_slider"):
            self._master_slider.blockSignals(True)
            self._master_slider.setValue(int(self._master_db))
            self._master_slider.blockSignals(False)
        if self._master_db_lbl is not None:
            self._master_db_lbl.setText(self.db_text(self._master_db))

    def set_levels(self, levels) -> None:
        for ch, strip in enumerate(self._strips):
            try:
                v = float(levels[ch]) if ch < len(levels) else 0.0
            except Exception:
                v = 0.0
            strip["meter"].set_level(v)

    def set_master_level(self, v: float) -> None:
        if self._master_meter is not None:
            try:
                self._master_meter.set_level(float(v))
            except Exception:
                pass

    def reset_levels(self) -> None:
        for strip in self._strips:
            strip["meter"].reset()
        if self._master_meter is not None:
            self._master_meter.reset()

    def _on_toggled(self, ch: int, state: int) -> None:
        muted = state != 0
        if muted:
            self._muted.add(ch)
        else:
            self._muted.discard(ch)
        self.muteChanged.emit(ch, muted)

    def _on_gain(self, ch: int, db: float) -> None:
        if 0 <= ch < len(self._strips):
            self._strips[ch]["db"] = db
            self._strips[ch]["db_lbl"].setText(self.db_text(db))
        self.gainChanged.emit(ch, db)

    def _on_master(self, db: float) -> None:
        self._master_db = db
        if self._master_db_lbl is not None:
            self._master_db_lbl.setText(self.db_text(db))
        self.masterChanged.emit(db)
