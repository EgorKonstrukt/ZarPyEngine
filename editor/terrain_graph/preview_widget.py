# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from editor.NodeGraphQt.widgets.node_widgets import NodeBaseWidget


class NodePreviewWidget(NodeBaseWidget):
    PREVIEW_SIZE = 64

    def __init__(self, parent=None, name="_preview", label="Preview"):
        super(NodePreviewWidget, self).__init__(parent, name, label)
        self._img_label = QtWidgets.QLabel()
        self._img_label.setFixedSize(self.PREVIEW_SIZE, self.PREVIEW_SIZE)
        self._img_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self._img_label.setStyleSheet(
            "background: #1a1a1a; border: 1px solid #333; border-radius: 2px;"
        )
        self._img_label.setText("--")
        placeholder = self._make_placeholder()
        self._img_label.setPixmap(placeholder)
        self.set_custom_widget(self._img_label)
        self.widget().setMaximumWidth(self.PREVIEW_SIZE + 8)

    @property
    def type_(self):
        return "PreviewNodeWidget"

    def get_value(self):
        return ""

    def set_value(self, text):
        pass

    def set_preview(self, heightfield: np.ndarray | None):
        if heightfield is None or heightfield.size == 0:
            self._img_label.setPixmap(self._make_placeholder())
            return
        hmin = float(heightfield.min())
        hmax = float(heightfield.max())
        if hmax - hmin < 1e-8:
            normalized = np.zeros_like(heightfield, dtype=np.float32)
        else:
            normalized = (heightfield - hmin) / (hmax - hmin)
        normalized = np.clip(normalized, 0.0, 1.0)
        gray = (normalized * 255).astype(np.uint8)
        h, w = gray.shape
        bytes_per_line = w
        qimg = QtGui.QImage(gray.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.PREVIEW_SIZE, self.PREVIEW_SIZE,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._img_label.setPixmap(scaled)

    def _make_placeholder(self) -> QtGui.QPixmap:
        pm = QtGui.QPixmap(self.PREVIEW_SIZE, self.PREVIEW_SIZE)
        pm.fill(QtGui.QColor("#1a1a1a"))
        painter = QtGui.QPainter(pm)
        painter.setPen(QtGui.QColor("#555"))
        painter.drawText(pm.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "?")
        painter.end()
        return pm
