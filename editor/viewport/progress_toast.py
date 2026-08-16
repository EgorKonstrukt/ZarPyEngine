# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import time

from PyQt6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
)
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from core.foundation import progress

_MARGIN = 16
_MAX_ROWS = 3
_MAX_HISTORY = 3
_POLL_MS = 60
_HISTORY_TTL = 6.0
_ERROR_TTL = 8.0
_SLIDE_X = 64
_FADE_IN_MS = 260
_FADE_OUT_MS = 200
_SLIDE_IN_MS = 360
_SLIDE_OUT_MS = 200
_RESIZE_MS = 220

_CARD_QSS = """
QFrame#ProgressCard {
    background-color: rgba(28, 28, 30, 235);
    border: 1px solid #3d3d3f;
    border-radius: 6px;
}
QLabel {
    color: #d4d4d4;
    font-size: 11px;
}
QProgressBar {
    background-color: #26262a;
    border: 1px solid #444;
    border-radius: 3px;
}
QProgressBar::chunk {
    background-color: #4a9cf5;
    border-radius: 2px;
}
"""


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m{secs:02d}s"


def _pct(fraction: float | None) -> str:
    if fraction is None:
        return ""
    return f"{max(0, min(100, round(fraction * 100)))}%"


def _fmt_count(n: float) -> str:
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{int(v)}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{int(v)}k" if v == int(v) else f"{v:.1f}k"
    return f"{n:.0f}"


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def _fmt_rate(rate: float, units: str | None) -> str:
    if units == "bytes":
        return f"{_fmt_bytes(rate)}/s"
    return f"{_fmt_count(rate)}/s"


class ProgressToast(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAutoFillBackground(False)

        self._rows: dict[str, tuple[QWidget, QLabel, QLabel, QLabel, QProgressBar]] = {}
        self._row_ids: list[str] = []
        self._history: list[tuple[str, float, float]] = []

        self._anim_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._anim_effect)
        self._anim_effect.setOpacity(0.0)

        self._fade_in = QPropertyAnimation(self._anim_effect, b"opacity", self)
        self._fade_in.setDuration(_FADE_IN_MS)
        self._fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)

        self._fade_out = QPropertyAnimation(self._anim_effect, b"opacity", self)
        self._fade_out.setDuration(_FADE_OUT_MS)
        self._fade_out.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._fade_out.setStartValue(1.0)
        self._fade_out.setEndValue(0.0)
        self._fade_out.finished.connect(self._hide_done)

        self._slide = QPropertyAnimation(self, b"pos", self)
        self._slide.setDuration(_SLIDE_IN_MS)
        self._slide.setEasingCurve(QEasingCurve.Type.OutBack)

        self._resize = QPropertyAnimation(self, b"geometry", self)
        self._resize.setDuration(_RESIZE_MS)
        self._resize.setEasingCurve(QEasingCurve.Type.OutCubic)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._card = QFrame(self)
        self._card.setObjectName("ProgressCard")
        self._card.setStyleSheet(_CARD_QSS)
        outer.addWidget(self._card)

        self._layout = QVBoxLayout(self._card)
        self._layout.setContentsMargins(12, 10, 12, 10)
        self._layout.setSpacing(8)

        self._more_lbl = QLabel("", self._card)
        self._more_lbl.setStyleSheet("color: #999; font-size: 10px;")
        self._more_lbl.hide()
        self._layout.addWidget(self._more_lbl)

        self._history_lbl = QLabel("", self._card)
        self._history_lbl.setStyleSheet("color: #7a7a80; font-size: 10px;")
        self._history_lbl.hide()
        self._layout.addWidget(self._history_lbl)

        self._errors_lbl = QLabel("", self._card)
        self._errors_lbl.setStyleSheet("color: #e06c6c; font-size: 10px;")
        self._errors_lbl.hide()
        self._layout.addWidget(self._errors_lbl)

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self.hide()

    @staticmethod
    def _set_bar(bar: QProgressBar, fraction) -> None:
        if fraction is None:
            bar.setRange(0, 0)
        else:
            bar.setRange(0, 100)
            bar.setValue(max(0, min(100, round(fraction * 100))))

    def _make_row(self, task: dict) -> tuple[QWidget, QLabel, QLabel, QLabel, QProgressBar]:
        row = QWidget(self._card)
        v = QVBoxLayout(row)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        title = QLabel(task["title"], row)
        info = QLabel("", row)
        info.setStyleSheet("color: #9db8e8;")
        header.addWidget(title, 1)
        header.addWidget(info, 0)

        detail = QLabel("", row)
        detail.setStyleSheet("color: #8a8a8f; font-size: 10px;")
        detail.hide()

        bar = QProgressBar(row)
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        self._set_bar(bar, task["fraction"])

        v.addLayout(header)
        v.addWidget(detail)
        v.addWidget(bar)
        return row, title, info, detail, bar

    def _rebuild(self, tasks) -> None:
        for row_widget, *_ in self._rows.values():
            self._layout.removeWidget(row_widget)
            row_widget.deleteLater()
        self._rows.clear()
        self._row_ids.clear()
        self._layout.removeWidget(self._more_lbl)
        self._layout.removeWidget(self._history_lbl)
        self._layout.removeWidget(self._errors_lbl)
        for task in tasks[:_MAX_ROWS]:
            tid = task["id"]
            row_widget, title, info, detail, bar = self._make_row(task)
            self._layout.addWidget(row_widget)
            self._rows[tid] = (row_widget, title, info, detail, bar)
            self._row_ids.append(tid)
        if len(tasks) > _MAX_ROWS:
            self._more_lbl.setText(f"+{len(tasks) - _MAX_ROWS} more")
            self._layout.addWidget(self._more_lbl)
            self._more_lbl.show()
        else:
            self._more_lbl.hide()
        self._layout.addWidget(self._history_lbl)
        self._layout.addWidget(self._errors_lbl)

    def _update_row(self, row, task: dict, now: float) -> None:
        _, title, info, detail, bar = row
        title.setText(task["title"])
        elapsed = now - task["started"]
        fraction = task["fraction"]
        rate = None
        total = task["total"]
        if total is not None and elapsed > 0.0:
            if fraction is not None:
                rate = total * fraction / elapsed
            else:
                rate = total / elapsed
        parts = []
        if fraction is not None:
            parts.append(_pct(fraction))
            parts.append(_fmt_time(elapsed))
            if fraction > 0.0:
                parts.append(f"~{_fmt_time(elapsed * (1.0 - fraction) / fraction)} left")
        else:
            parts.append(_fmt_time(elapsed))
        info.setText(" · ".join(parts))
        self._set_bar(bar, fraction)

        detail_parts = [task["detail"]] if task["detail"] else []
        if rate is not None:
            detail_parts.append(_fmt_rate(rate, task["units"]))
        detail_text = " · ".join(detail_parts)
        detail.setText(detail_text)
        detail.setVisible(bool(detail_text))

    def _tick(self):
        now = time.monotonic()
        tasks = progress.snapshot()
        completed = progress.snapshot_completed()
        self._history = [
            (c["title"], c["duration"], c["time"])
            for c in sorted(completed, key=lambda c: c["time"], reverse=True)
            if now - c["time"] < _HISTORY_TTL
        ][:_MAX_HISTORY]
        errors = [n for n in progress.snapshot_notifications() if now - n["time"] < _ERROR_TTL][-3:]

        if [t["id"] for t in tasks][:_MAX_ROWS] != self._row_ids:
            self._rebuild(tasks)
        for task in tasks:
            row = self._rows.get(task["id"])
            if row is not None:
                self._update_row(row, task, now)

        hist_text = "\n".join(f"{title} · {_fmt_time(dur)}" for title, dur, _ in self._history)
        self._history_lbl.setText(hist_text)
        self._history_lbl.setVisible(bool(hist_text))
        err_text = "\n".join(n["message"] for n in errors)
        self._errors_lbl.setText(err_text)
        self._errors_lbl.setVisible(bool(err_text))

        if tasks or self._history or errors:
            if self.isHidden():
                self.adjustSize()
            elif self._slide.state() != QAbstractAnimation.State.Running \
                    and self._resize.state() != QAbstractAnimation.State.Running:
                if self.size() != self.sizeHint():
                    self._animate_resize()
                else:
                    self.reposition()
            self._show_animated()
        elif self.isVisible():
            self._hide_animated()

    def _animate_resize(self):
        target = self._target_rect(self.sizeHint())
        self._resize.stop()
        self._resize.setStartValue(self.geometry())
        self._resize.setEndValue(target)
        self._resize.start()

    def _show_animated(self):
        if self.isHidden():
            self.show()
            self._fade_out.stop()
            self._slide.stop()
            self._resize.stop()
            self._anim_effect.setOpacity(0.0)
            start = self._target_pos() + QPoint(_SLIDE_X, 0)
            self.move(start)
            self._slide.setDuration(_SLIDE_IN_MS)
            self._slide.setEasingCurve(QEasingCurve.Type.OutBack)
            self._slide.setStartValue(start)
            self._slide.setEndValue(self._target_pos())
            self._slide.start()
            self._fade_in.start()
        else:
            if self._fade_out.state() == QAbstractAnimation.State.Running:
                self._fade_out.stop()
                self._slide.stop()
                self._resize.stop()
                self._anim_effect.setOpacity(1.0)
                self.reposition()

    def _hide_animated(self):
        if self._fade_out.state() != QAbstractAnimation.State.Running:
            self._fade_in.stop()
            self._resize.stop()
            start = self.pos()
            self._slide.setDuration(_SLIDE_OUT_MS)
            self._slide.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self._slide.setStartValue(start)
            self._slide.setEndValue(start + QPoint(_SLIDE_X, 0))
            self._slide.start()
            self._fade_out.start()

    def _hide_done(self):
        self.hide()
        self._slide.stop()
        self._resize.stop()
        self._anim_effect.setOpacity(1.0)

    def _target_rect(self, size) -> QRect:
        parent = self.parentWidget()
        if parent is None:
            return QRect(QPoint(0, 0), size)
        x = max(0, parent.width() - size.width() - _MARGIN)
        y = max(0, parent.height() - size.height() - _MARGIN)
        return QRect(x, y, size.width(), size.height())

    def _target_pos(self) -> QPoint:
        return self._target_rect(self.size()).topLeft()

    def reposition(self) -> None:
        self.move(self._target_pos())
