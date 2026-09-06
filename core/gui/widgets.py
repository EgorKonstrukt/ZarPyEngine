# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional, Any, Callable
from PyQt6.QtWidgets import (
    QWidget, QFrame, QPushButton, QLabel, QSlider, QLineEdit,
    QCheckBox, QProgressBar, QComboBox, QScrollArea,
    QRadioButton, QListWidget, QTableWidget, QTreeWidget,
    QTabWidget, QGroupBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QDial, QTextBrowser,
    QSplitter, QStackedWidget, QToolBox, QCalendarWidget,
    QLCDNumber, QPlainTextEdit, QScrollBar, QToolButton,
    QFontComboBox, QMdiArea, QMdiSubWindow, QVBoxLayout,
)
from PyQt6.QtCore import Qt, QRect, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen


ANCHOR_TOP_LEFT = 0
ANCHOR_TOP_CENTER = 1
ANCHOR_TOP_RIGHT = 2
ANCHOR_CENTER_LEFT = 3
ANCHOR_CENTER = 4
ANCHOR_CENTER_RIGHT = 5
ANCHOR_BOTTOM_LEFT = 6
ANCHOR_BOTTOM_CENTER = 7
ANCHOR_BOTTOM_RIGHT = 8
ANCHOR_STRETCH_ALL = 9
ANCHOR_STRETCH_WIDTH = 10
ANCHOR_STRETCH_HEIGHT = 11


def _set_widget_bounds(w: QWidget, x: float, y: float, ww: float, h: float):
    w.setGeometry(int(x), int(y), max(1, int(ww)), max(1, int(h)))


def _update_anchor(w: QWidget, anchor: int, canvas_w: float, canvas_h: float,
                   offset_x: float, offset_y: float, gw: float, gh: float) -> tuple[float, float, float, float]:
    x, y, ww, hh = w.x(), w.y(), gw, gh
    if anchor == ANCHOR_TOP_LEFT:
        pass
    elif anchor == ANCHOR_TOP_CENTER:
        x = (canvas_w - ww) / 2 + offset_x
    elif anchor == ANCHOR_TOP_RIGHT:
        x = canvas_w - ww - offset_x
    elif anchor == ANCHOR_CENTER_LEFT:
        y = (canvas_h - hh) / 2 + offset_y
    elif anchor == ANCHOR_CENTER:
        x = (canvas_w - ww) / 2 + offset_x
        y = (canvas_h - hh) / 2 + offset_y
    elif anchor == ANCHOR_CENTER_RIGHT:
        x = canvas_w - ww - offset_x
        y = (canvas_h - hh) / 2 + offset_y
    elif anchor == ANCHOR_BOTTOM_LEFT:
        y = canvas_h - hh - offset_y
    elif anchor == ANCHOR_BOTTOM_CENTER:
        x = (canvas_w - ww) / 2 + offset_x
        y = canvas_h - hh - offset_y
    elif anchor == ANCHOR_BOTTOM_RIGHT:
        x = canvas_w - ww - offset_x
        y = canvas_h - hh - offset_y
    elif anchor == ANCHOR_STRETCH_ALL:
        x = offset_x
        y = offset_y
        ww = canvas_w - offset_x * 2
        hh = canvas_h - offset_y * 2
    elif anchor == ANCHOR_STRETCH_WIDTH:
        x = offset_x
        ww = canvas_w - offset_x * 2
    elif anchor == ANCHOR_STRETCH_HEIGHT:
        y = offset_y
        hh = canvas_h - offset_y * 2
    return x, y, ww, hh


class Panel(QFrame):
    def __init__(self, x: float = 0, y: float = 0, w: float = 200, h: float = 200, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiPanel")
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "Panel",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> Panel:
        p = Panel(data.get("x", 0), data.get("y", 0), data.get("w", 200), data.get("h", 200))
        p._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        p._z_index = data.get("z_index", 0)
        p._tag = data.get("tag", "")
        return p


class Label(QLabel):
    def __init__(self, x: float = 0, y: float = 0, w: float = 100, h: float = 30, text: str = "Label", parent=None):
        super().__init__(text, parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiLabel")
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "Label",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "text": self.text(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> Label:
        l = Label(data.get("x", 0), data.get("y", 0), data.get("w", 100), data.get("h", 30), data.get("text", "Label"))
        l._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        l._z_index = data.get("z_index", 0)
        l._tag = data.get("tag", "")
        return l


class Button(QPushButton):
    def __init__(self, x: float = 0, y: float = 0, w: float = 100, h: float = 40, text: str = "Button", parent=None):
        super().__init__(text, parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiButton")
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "Button",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "text": self.text(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> Button:
        b = Button(data.get("x", 0), data.get("y", 0), data.get("w", 100), data.get("h", 40), data.get("text", "Button"))
        b._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        b._z_index = data.get("z_index", 0)
        b._tag = data.get("tag", "")
        return b


class Slider(QSlider):
    def __init__(self, x: float = 0, y: float = 0, w: float = 200, h: float = 30, min_val: int = 0, max_val: int = 100, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiSlider")
        self.setRange(min_val, max_val)
        self.setValue(0)
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "Slider",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "min": self.minimum(), "max": self.maximum(), "value": self.value(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> Slider:
        s = Slider(data.get("x", 0), data.get("y", 0), data.get("w", 200), data.get("h", 30),
                   data.get("min", 0), data.get("max", 100))
        s.setValue(data.get("value", 0))
        s._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        s._z_index = data.get("z_index", 0)
        s._tag = data.get("tag", "")
        return s


class TextInput(QLineEdit):
    def __init__(self, x: float = 0, y: float = 0, w: float = 200, h: float = 36, placeholder: str = "", parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiTextInput")
        self.setPlaceholderText(placeholder)
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "TextInput",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "text": self.text(), "placeholder": self.placeholderText(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> TextInput:
        t = TextInput(data.get("x", 0), data.get("y", 0), data.get("w", 200), data.get("h", 36),
                      data.get("placeholder", ""))
        t.setText(data.get("text", ""))
        t._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        t._z_index = data.get("z_index", 0)
        t._tag = data.get("tag", "")
        return t


class Toggle(QCheckBox):
    def __init__(self, x: float = 0, y: float = 0, w: float = 100, h: float = 30, text: str = "Toggle", parent=None):
        super().__init__(text, parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiToggle")
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "Toggle",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "text": self.text(), "checked": self.isChecked(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> Toggle:
        t = Toggle(data.get("x", 0), data.get("y", 0), data.get("w", 100), data.get("h", 30), data.get("text", "Toggle"))
        t.setChecked(data.get("checked", False))
        t._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        t._z_index = data.get("z_index", 0)
        t._tag = data.get("tag", "")
        return t


class ProgressBar(QProgressBar):
    def __init__(self, x: float = 0, y: float = 0, w: float = 200, h: float = 24, min_val: int = 0, max_val: int = 100, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiProgressBar")
        self.setRange(min_val, max_val)
        self.setValue(0)
        self.setTextVisible(True)
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "ProgressBar",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "min": self.minimum(), "max": self.maximum(), "value": self.value(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> ProgressBar:
        p = ProgressBar(data.get("x", 0), data.get("y", 0), data.get("w", 200), data.get("h", 24),
                        data.get("min", 0), data.get("max", 100))
        p.setValue(data.get("value", 0))
        p._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        p._z_index = data.get("z_index", 0)
        p._tag = data.get("tag", "")
        return p


class Dropdown(QComboBox):
    def __init__(self, x: float = 0, y: float = 0, w: float = 200, h: float = 32, items: Optional[list[str]] = None, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiDropdown")
        if items:
            self.addItems(items)
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "Dropdown",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "items": [self.itemText(i) for i in range(self.count())],
            "current": self.currentIndex(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> Dropdown:
        d = Dropdown(data.get("x", 0), data.get("y", 0), data.get("w", 200), data.get("h", 32),
                     data.get("items"))
        d.setCurrentIndex(data.get("current", 0))
        d._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        d._z_index = data.get("z_index", 0)
        d._tag = data.get("tag", "")
        return d


class ScrollPanel(QScrollArea):
    def __init__(self, x: float = 0, y: float = 0, w: float = 200, h: float = 200, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiScrollPanel")
        self.setWidgetResizable(True)
        _set_widget_bounds(self, x, y, w, h)
        self._scroll_content = QWidget()
        self.setWidget(self._scroll_content)

    @property
    def scroll_content(self) -> QWidget:
        return self._scroll_content

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "ScrollPanel",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> ScrollPanel:
        s = ScrollPanel(data.get("x", 0), data.get("y", 0), data.get("w", 200), data.get("h", 200))
        s._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        s._z_index = data.get("z_index", 0)
        s._tag = data.get("tag", "")
        return s


class Image(QWidget):
    def __init__(self, x: float = 0, y: float = 0, w: float = 100, h: float = 100, source: str = "", parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiImage")
        _set_widget_bounds(self, x, y, w, h)
        self._pixmap: Optional[QPixmap] = None
        if source:
            self._pixmap = QPixmap(source)

    @property
    def source(self) -> str:
        return getattr(self, '_source', "")

    @source.setter
    def source(self, v: str):
        self._source = v
        self._pixmap = QPixmap(v) if v else None
        self.update()

    def paintEvent(self, event):
        if self._pixmap and not self._pixmap.isNull():
            painter = QPainter(self)
            painter.drawPixmap(self.rect(), self._pixmap)
            painter.end()
        else:
            super().paintEvent(event)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "Image",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "source": getattr(self, '_source', ""),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> Image:
        i = Image(data.get("x", 0), data.get("y", 0), data.get("w", 100), data.get("h", 100),
                  data.get("source", ""))
        i._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        i._z_index = data.get("z_index", 0)
        i._tag = data.get("tag", "")
        return i


class RadioButton(QRadioButton):
    def __init__(self, x: float = 0, y: float = 0, w: float = 100, h: float = 30, text: str = "Radio", parent=None):
        super().__init__(text, parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiRadioButton")
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "RadioButton",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "text": self.text(), "checked": self.isChecked(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> RadioButton:
        r = RadioButton(data.get("x", 0), data.get("y", 0), data.get("w", 100), data.get("h", 30), data.get("text", "Radio"))
        r.setChecked(data.get("checked", False))
        r._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        r._z_index = data.get("z_index", 0)
        r._tag = data.get("tag", "")
        return r


class ListWidget(QListWidget):
    def __init__(self, x: float = 0, y: float = 0, w: float = 200, h: float = 200, items: Optional[list[str]] = None, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiListWidget")
        if items:
            self.addItems(items)
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "ListWidget",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "items": [self.item(i).text() for i in range(self.count())],
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> ListWidget:
        lw = ListWidget(data.get("x", 0), data.get("y", 0), data.get("w", 200), data.get("h", 200), data.get("items"))
        lw._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        lw._z_index = data.get("z_index", 0)
        lw._tag = data.get("tag", "")
        return lw


class TableWidget(QTableWidget):
    def __init__(self, x: float = 0, y: float = 0, w: float = 300, h: float = 200,
                 rows: int = 3, cols: int = 3, parent=None):
        super().__init__(rows, cols, parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiTableWidget")
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "TableWidget",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "rows": self.rowCount(), "cols": self.columnCount(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> TableWidget:
        tw = TableWidget(data.get("x", 0), data.get("y", 0), data.get("w", 300), data.get("h", 200),
                         data.get("rows", 3), data.get("cols", 3))
        tw._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        tw._z_index = data.get("z_index", 0)
        tw._tag = data.get("tag", "")
        return tw


class TreeWidget(QTreeWidget):
    def __init__(self, x: float = 0, y: float = 0, w: float = 200, h: float = 200, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiTreeWidget")
        self.setHeaderLabel("Tree")
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "TreeWidget",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> TreeWidget:
        tw = TreeWidget(data.get("x", 0), data.get("y", 0), data.get("w", 200), data.get("h", 200))
        tw._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        tw._z_index = data.get("z_index", 0)
        tw._tag = data.get("tag", "")
        return tw


class TabWidget(QTabWidget):
    def __init__(self, x: float = 0, y: float = 0, w: float = 300, h: float = 200, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiTabWidget")
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "TabWidget",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "tab_count": self.count(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> TabWidget:
        tw = TabWidget(data.get("x", 0), data.get("y", 0), data.get("w", 300), data.get("h", 200))
        tw._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        tw._z_index = data.get("z_index", 0)
        tw._tag = data.get("tag", "")
        return tw


class GroupBox(QGroupBox):
    def __init__(self, x: float = 0, y: float = 0, w: float = 200, h: float = 200, title: str = "Group", parent=None):
        super().__init__(title, parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiGroupBox")
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "GroupBox",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "title": self.title(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> GroupBox:
        gb = GroupBox(data.get("x", 0), data.get("y", 0), data.get("w", 200), data.get("h", 200),
                      data.get("title", "Group"))
        gb._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        gb._z_index = data.get("z_index", 0)
        gb._tag = data.get("tag", "")
        return gb


class SpinBox(QSpinBox):
    def __init__(self, x: float = 0, y: float = 0, w: float = 100, h: float = 32,
                 min_val: int = 0, max_val: int = 100, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiSpinBox")
        self.setRange(min_val, max_val)
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "SpinBox",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "min": self.minimum(), "max": self.maximum(), "value": self.value(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> SpinBox:
        sb = SpinBox(data.get("x", 0), data.get("y", 0), data.get("w", 100), data.get("h", 32),
                     data.get("min", 0), data.get("max", 100))
        sb.setValue(data.get("value", 0))
        sb._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        sb._z_index = data.get("z_index", 0)
        sb._tag = data.get("tag", "")
        return sb


class DoubleSpinBox(QDoubleSpinBox):
    def __init__(self, x: float = 0, y: float = 0, w: float = 120, h: float = 32,
                 min_val: float = 0.0, max_val: float = 100.0, decimals: int = 2, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiDoubleSpinBox")
        self.setRange(min_val, max_val)
        self.setDecimals(decimals)
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "DoubleSpinBox",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "min": self.minimum(), "max": self.maximum(),
            "value": self.value(), "decimals": self.decimals(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> DoubleSpinBox:
        dsb = DoubleSpinBox(data.get("x", 0), data.get("y", 0), data.get("w", 120), data.get("h", 32),
                           data.get("min", 0.0), data.get("max", 100.0), data.get("decimals", 2))
        dsb.setValue(data.get("value", 0.0))
        dsb._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        dsb._z_index = data.get("z_index", 0)
        dsb._tag = data.get("tag", "")
        return dsb


class TextEdit(QTextEdit):
    def __init__(self, x: float = 0, y: float = 0, w: float = 200, h: float = 150, text: str = "", parent=None):
        super().__init__(text, parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiTextEdit")
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "TextEdit",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "text": self.toPlainText(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> TextEdit:
        te = TextEdit(data.get("x", 0), data.get("y", 0), data.get("w", 200), data.get("h", 150), data.get("text", ""))
        te._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        te._z_index = data.get("z_index", 0)
        te._tag = data.get("tag", "")
        return te


class HtmlView(QTextBrowser):
    def __init__(self, x: float = 0, y: float = 0, w: float = 300, h: float = 200, text: str = "", parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiHtmlView")
        self.setOpenExternalLinks(True)
        self.setOpenLinks(True)
        _set_widget_bounds(self, x, y, w, h)
        if text:
            self.setHtml(text)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "HtmlView",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "html": self.toHtml(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> HtmlView:
        hv = HtmlView(data.get("x", 0), data.get("y", 0), data.get("w", 300), data.get("h", 200), data.get("html", ""))
        hv._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        hv._z_index = data.get("z_index", 0)
        hv._tag = data.get("tag", "")
        return hv


class Dial(QDial):
    def __init__(self, x: float = 0, y: float = 0, w: float = 80, h: float = 80,
                 min_val: int = 0, max_val: int = 100, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._anchor_offset_x = 0.0
        self._anchor_offset_y = 0.0
        self._z_index = 0
        self._tag = ""
        self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None
        self._canvas_size = (800, 600)
        self._orig_w = w
        self._orig_h = h
        self.setObjectName("GuiDial")
        self.setRange(min_val, max_val)
        self.setWrapping(False)
        self.setNotchesVisible(True)
        _set_widget_bounds(self, x, y, w, h)

    def update_anchor(self, canvas_w: float, canvas_h: float):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h,
                                         self._anchor_offset_x, self._anchor_offset_y,
                                         self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self) -> dict:
        return {
            "type": "Dial",
            "x": self.x(), "y": self.y(),
            "w": self.width(), "h": self.height(),
            "min": self.minimum(), "max": self.maximum(), "value": self.value(),
            "anchor": self._anchor,
            "z_index": self._z_index,
            "tag": self._tag,
        }

    @staticmethod
    def deserialize(data: dict) -> Dial:
        d = Dial(data.get("x", 0), data.get("y", 0), data.get("w", 80), data.get("h", 80),
                data.get("min", 0), data.get("max", 100))
        d.setValue(data.get("value", 0))
        d._anchor = data.get("anchor", ANCHOR_TOP_LEFT)
        d._z_index = data.get("z_index", 0)
        d._tag = data.get("tag", "")
        return d


class Splitter(QSplitter):
    def __init__(self, x: float = 0, y: float = 0, w: float = 200, h: float = 200, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT; self._anchor_offset_x = 0.0; self._anchor_offset_y = 0.0
        self._z_index = 0; self._tag = ""; self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None; self._canvas_size = (800, 600)
        self._orig_w = w; self._orig_h = h; self.setObjectName("GuiSplitter")
        _set_widget_bounds(self, x, y, w, h)
    def update_anchor(self, canvas_w, canvas_h):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h, self._anchor_offset_x, self._anchor_offset_y, self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)
    def serialize(self):
        return {"type": "Splitter", "x": self.x(), "y": self.y(), "w": self.width(), "h": self.height(), "anchor": self._anchor, "z_index": self._z_index, "tag": self._tag}
    @staticmethod
    def deserialize(data):
        s = Splitter(data.get("x", 0), data.get("y", 0), data.get("w", 200), data.get("h", 200))
        s._anchor = data.get("anchor", ANCHOR_TOP_LEFT); s._z_index = data.get("z_index", 0); s._tag = data.get("tag", ""); return s


class StackedWidget(QStackedWidget):
    def __init__(self, x: float = 0, y: float = 0, w: float = 200, h: float = 200, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT; self._anchor_offset_x = 0.0; self._anchor_offset_y = 0.0
        self._z_index = 0; self._tag = ""; self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None; self._canvas_size = (800, 600)
        self._orig_w = w; self._orig_h = h; self.setObjectName("GuiStackedWidget")
        _set_widget_bounds(self, x, y, w, h)
    def update_anchor(self, canvas_w, canvas_h):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h, self._anchor_offset_x, self._anchor_offset_y, self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)
    def serialize(self):
        return {"type": "StackedWidget", "x": self.x(), "y": self.y(), "w": self.width(), "h": self.height(), "anchor": self._anchor, "z_index": self._z_index, "tag": self._tag}
    @staticmethod
    def deserialize(data):
        s = StackedWidget(data.get("x", 0), data.get("y", 0), data.get("w", 200), data.get("h", 200))
        s._anchor = data.get("anchor", ANCHOR_TOP_LEFT); s._z_index = data.get("z_index", 0); s._tag = data.get("tag", ""); return s


class ToolBox(QToolBox):
    def __init__(self, x: float = 0, y: float = 0, w: float = 200, h: float = 300, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT; self._anchor_offset_x = 0.0; self._anchor_offset_y = 0.0
        self._z_index = 0; self._tag = ""; self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None; self._canvas_size = (800, 600)
        self._orig_w = w; self._orig_h = h; self.setObjectName("GuiToolBox")
        _set_widget_bounds(self, x, y, w, h)
    def update_anchor(self, canvas_w, canvas_h):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h, self._anchor_offset_x, self._anchor_offset_y, self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)
    def serialize(self):
        return {"type": "ToolBox", "x": self.x(), "y": self.y(), "w": self.width(), "h": self.height(), "anchor": self._anchor, "z_index": self._z_index, "tag": self._tag}
    @staticmethod
    def deserialize(data):
        t = ToolBox(data.get("x", 0), data.get("y", 0), data.get("w", 200), data.get("h", 300))
        t._anchor = data.get("anchor", ANCHOR_TOP_LEFT); t._z_index = data.get("z_index", 0); t._tag = data.get("tag", ""); return t


class Calendar(QCalendarWidget):
    def __init__(self, x: float = 0, y: float = 0, w: float = 250, h: float = 200, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT; self._anchor_offset_x = 0.0; self._anchor_offset_y = 0.0
        self._z_index = 0; self._tag = ""; self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None; self._canvas_size = (800, 600)
        self._orig_w = w; self._orig_h = h; self.setObjectName("GuiCalendar")
        _set_widget_bounds(self, x, y, w, h)
    def update_anchor(self, canvas_w, canvas_h):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h, self._anchor_offset_x, self._anchor_offset_y, self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)
    def serialize(self):
        return {"type": "Calendar", "x": self.x(), "y": self.y(), "w": self.width(), "h": self.height(), "anchor": self._anchor, "z_index": self._z_index, "tag": self._tag}
    @staticmethod
    def deserialize(data):
        c = Calendar(data.get("x", 0), data.get("y", 0), data.get("w", 250), data.get("h", 200))
        c._anchor = data.get("anchor", ANCHOR_TOP_LEFT); c._z_index = data.get("z_index", 0); c._tag = data.get("tag", ""); return c


class LCDNumber(QLCDNumber):
    def __init__(self, x: float = 0, y: float = 0, w: float = 120, h: float = 40, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT; self._anchor_offset_x = 0.0; self._anchor_offset_y = 0.0
        self._z_index = 0; self._tag = ""; self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None; self._canvas_size = (800, 600)
        self._orig_w = w; self._orig_h = h; self.setObjectName("GuiLCDNumber")
        _set_widget_bounds(self, x, y, w, h)
    def update_anchor(self, canvas_w, canvas_h):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h, self._anchor_offset_x, self._anchor_offset_y, self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)
    def serialize(self):
        return {"type": "LCDNumber", "x": self.x(), "y": self.y(), "w": self.width(), "h": self.height(), "anchor": self._anchor, "z_index": self._z_index, "tag": self._tag}
    @staticmethod
    def deserialize(data):
        l = LCDNumber(data.get("x", 0), data.get("y", 0), data.get("w", 120), data.get("h", 40))
        l._anchor = data.get("anchor", ANCHOR_TOP_LEFT); l._z_index = data.get("z_index", 0); l._tag = data.get("tag", ""); return l


class PlainText(QPlainTextEdit):
    def __init__(self, x: float = 0, y: float = 0, w: float = 200, h: float = 150, text: str = "", parent=None):
        super().__init__(text, parent)
        self._anchor = ANCHOR_TOP_LEFT; self._anchor_offset_x = 0.0; self._anchor_offset_y = 0.0
        self._z_index = 0; self._tag = ""; self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None; self._canvas_size = (800, 600)
        self._orig_w = w; self._orig_h = h; self.setObjectName("GuiPlainText")
        _set_widget_bounds(self, x, y, w, h)
    def update_anchor(self, canvas_w, canvas_h):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h, self._anchor_offset_x, self._anchor_offset_y, self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)
    def serialize(self):
        return {"type": "PlainText", "x": self.x(), "y": self.y(), "w": self.width(), "h": self.height(), "text": self.toPlainText(), "anchor": self._anchor, "z_index": self._z_index, "tag": self._tag}
    @staticmethod
    def deserialize(data):
        p = PlainText(data.get("x", 0), data.get("y", 0), data.get("w", 200), data.get("h", 150), data.get("text", ""))
        p._anchor = data.get("anchor", ANCHOR_TOP_LEFT); p._z_index = data.get("z_index", 0); p._tag = data.get("tag", ""); return p


class ScrollBar(QScrollBar):
    def __init__(self, x: float = 0, y: float = 0, w: float = 120, h: float = 20, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT; self._anchor_offset_x = 0.0; self._anchor_offset_y = 0.0
        self._z_index = 0; self._tag = ""; self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None; self._canvas_size = (800, 600)
        self._orig_w = w; self._orig_h = h; self.setObjectName("GuiScrollBar")
        _set_widget_bounds(self, x, y, w, h)
    def update_anchor(self, canvas_w, canvas_h):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h, self._anchor_offset_x, self._anchor_offset_y, self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)
    def serialize(self):
        return {"type": "ScrollBar", "x": self.x(), "y": self.y(), "w": self.width(), "h": self.height(), "anchor": self._anchor, "z_index": self._z_index, "tag": self._tag}
    @staticmethod
    def deserialize(data):
        s = ScrollBar(data.get("x", 0), data.get("y", 0), data.get("w", 120), data.get("h", 20))
        s._anchor = data.get("anchor", ANCHOR_TOP_LEFT); s._z_index = data.get("z_index", 0); s._tag = data.get("tag", ""); return s


class ToolButton(QToolButton):
    def __init__(self, x: float = 0, y: float = 0, w: float = 30, h: float = 30, text: str = "", parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT; self._anchor_offset_x = 0.0; self._anchor_offset_y = 0.0
        self._z_index = 0; self._tag = ""; self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None; self._canvas_size = (800, 600)
        self._orig_w = w; self._orig_h = h; self.setObjectName("GuiToolButton")
        _set_widget_bounds(self, x, y, w, h)
        if text: self.setText(text)
    def update_anchor(self, canvas_w, canvas_h):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h, self._anchor_offset_x, self._anchor_offset_y, self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)
    def serialize(self):
        return {"type": "ToolButton", "x": self.x(), "y": self.y(), "w": self.width(), "h": self.height(), "text": self.text(), "anchor": self._anchor, "z_index": self._z_index, "tag": self._tag}
    @staticmethod
    def deserialize(data):
        t = ToolButton(data.get("x", 0), data.get("y", 0), data.get("w", 30), data.get("h", 30), data.get("text", ""))
        t._anchor = data.get("anchor", ANCHOR_TOP_LEFT); t._z_index = data.get("z_index", 0); t._tag = data.get("tag", ""); return t


class FontCombo(QFontComboBox):
    def __init__(self, x: float = 0, y: float = 0, w: float = 160, h: float = 28, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT; self._anchor_offset_x = 0.0; self._anchor_offset_y = 0.0
        self._z_index = 0; self._tag = ""; self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None; self._canvas_size = (800, 600)
        self._orig_w = w; self._orig_h = h; self.setObjectName("GuiFontCombo")
        _set_widget_bounds(self, x, y, w, h)
    def update_anchor(self, canvas_w, canvas_h):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h, self._anchor_offset_x, self._anchor_offset_y, self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)
    def serialize(self):
        return {"type": "FontCombo", "x": self.x(), "y": self.y(), "w": self.width(), "h": self.height(), "anchor": self._anchor, "z_index": self._z_index, "tag": self._tag}
    @staticmethod
    def deserialize(data):
        f = FontCombo(data.get("x", 0), data.get("y", 0), data.get("w", 160), data.get("h", 28))
        f._anchor = data.get("anchor", ANCHOR_TOP_LEFT); f._z_index = data.get("z_index", 0); f._tag = data.get("tag", ""); return f


class MdiSubWindow(QMdiSubWindow):
    def __init__(self, x: float = 0, y: float = 0, w: float = 300, h: float = 200, title: str = "Window", parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT; self._anchor_offset_x = 0.0; self._anchor_offset_y = 0.0
        self._z_index = 0; self._tag = ""; self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None; self._canvas_size = (800, 600)
        self._orig_w = w; self._orig_h = h; self.setObjectName("GuiMdiSubWindow")
        self.setWindowTitle(title)
        self.setWidget(QWidget(self))
        _set_widget_bounds(self, x, y, w, h)
    def update_anchor(self, canvas_w, canvas_h):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h, self._anchor_offset_x, self._anchor_offset_y, self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)
    def serialize(self):
        return {"type": "MdiSubWindow", "x": self.x(), "y": self.y(), "w": self.width(), "h": self.height(), "title": self.windowTitle(), "anchor": self._anchor, "z_index": self._z_index, "tag": self._tag}
    @staticmethod
    def deserialize(data):
        m = MdiSubWindow(data.get("x", 0), data.get("y", 0), data.get("w", 300), data.get("h", 200), data.get("title", "Window"))
        m._anchor = data.get("anchor", ANCHOR_TOP_LEFT); m._z_index = data.get("z_index", 0); m._tag = data.get("tag", ""); return m


class MdiArea(QMdiArea):
    def __init__(self, x: float = 0, y: float = 0, w: float = 400, h: float = 300, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT; self._anchor_offset_x = 0.0; self._anchor_offset_y = 0.0
        self._z_index = 0; self._tag = ""; self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None; self._canvas_size = (800, 600)
        self._orig_w = w; self._orig_h = h; self.setObjectName("GuiMdiArea")
        _set_widget_bounds(self, x, y, w, h)
    def add_sub_window(self, widget: QWidget, title: str = "Window") -> MdiSubWindow:
        sw = QMdiSubWindow(self)
        sw.setWidget(widget)
        sw.setWindowTitle(title)
        sw.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.addSubWindow(sw)
        return sw
    def tile_windows(self):
        self.tileSubWindows()
    def cascade_windows(self):
        self.cascadeSubWindows()
    def close_all(self):
        self.closeAllSubWindows()
    def update_anchor(self, canvas_w, canvas_h):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h, self._anchor_offset_x, self._anchor_offset_y, self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)
    def serialize(self):
        return {"type": "MdiArea", "x": self.x(), "y": self.y(), "w": self.width(), "h": self.height(), "anchor": self._anchor, "z_index": self._z_index, "tag": self._tag}
    @staticmethod
    def deserialize(data):
        m = MdiArea(data.get("x", 0), data.get("y", 0), data.get("w", 400), data.get("h", 300))
        m._anchor = data.get("anchor", ANCHOR_TOP_LEFT); m._z_index = data.get("z_index", 0); m._tag = data.get("tag", ""); return m


GuiWidget = QWidget

WIDGET_REGISTRY = {
    "Panel": Panel,
    "Label": Label,
    "Button": Button,
    "Slider": Slider,
    "TextInput": TextInput,
    "Toggle": Toggle,
    "ProgressBar": ProgressBar,
    "Dropdown": Dropdown,
    "ScrollPanel": ScrollPanel,
    "Image": Image,
    "RadioButton": RadioButton,
    "ListWidget": ListWidget,
    "TableWidget": TableWidget,
    "TreeWidget": TreeWidget,
    "TabWidget": TabWidget,
    "GroupBox": GroupBox,
    "SpinBox": SpinBox,
    "DoubleSpinBox": DoubleSpinBox,
    "TextEdit": TextEdit,
    "HtmlView": HtmlView,
    "Dial": Dial,
    "Splitter": Splitter,
    "StackedWidget": StackedWidget,
    "ToolBox": ToolBox,
    "Calendar": Calendar,
    "LCDNumber": LCDNumber,
    "PlainText": PlainText,
    "ScrollBar": ScrollBar,
    "ToolButton": ToolButton,
    "FontCombo": FontCombo,
"MdiArea": MdiArea,
    "MdiSubWindow": MdiSubWindow,
}


class ChartPlotter(QWidget):
    def __init__(self, x: float = 0, y: float = 0, w: float = 320, h: float = 200, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT; self._anchor_offset_x = 0.0; self._anchor_offset_y = 0.0
        self._z_index = 0; self._tag = ""; self._widget_id: Optional[str] = None
        self._canvas_ref: Optional[Any] = None; self._canvas_size = (800, 600)
        self._orig_w = w; self._orig_h = h; self.setObjectName("GuiChart")
        self._chart = None
        self._show_toolbar = False
        self._show_sidebar = False
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        _set_widget_bounds(self, x, y, w, h)

    def chart(self):
        if self._chart is not None:
            return self._chart
        from plugins.plotter_plugin.plotter import ChartWidget
        chart = ChartWidget(parent=self, show_toolbar=self._show_toolbar,
                            show_legend=True, show_sidebar=self._show_sidebar)
        self._lay.addWidget(chart)
        self._chart = chart
        return chart

    def setChartConfig(self, show_toolbar: bool = False, show_sidebar: bool = False):
        if show_toolbar != self._show_toolbar or show_sidebar != self._show_sidebar:
            self._show_toolbar = show_toolbar
            self._show_sidebar = show_sidebar
            chart = self._chart
            if chart is not None:
                chart.setToolbarVisible(show_toolbar)
                chart.setSidebarVisible(show_sidebar)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        chart = self.__dict__.get("_chart")
        if chart is not None:
            try:
                return getattr(chart, name)
            except RuntimeError:
                pass
        raise AttributeError(name)

    def update_anchor(self, canvas_w, canvas_h):
        nx, ny, nw, nh = _update_anchor(self, self._anchor, canvas_w, canvas_h, self._anchor_offset_x, self._anchor_offset_y, self._orig_w, self._orig_h)
        _set_widget_bounds(self, nx, ny, nw, nh)

    def serialize(self):
        return {"type": "Chart", "x": self.x(), "y": self.y(), "w": self.width(), "h": self.height(), "anchor": self._anchor, "z_index": self._z_index, "tag": self._tag}
    @staticmethod
    def deserialize(data):
        c = ChartPlotter(data.get("x", 0), data.get("y", 0), data.get("w", 320), data.get("h", 200))
        c._anchor = data.get("anchor", ANCHOR_TOP_LEFT); c._z_index = data.get("z_index", 0); c._tag = data.get("tag", ""); return c


WIDGET_REGISTRY["Chart"] = ChartPlotter


_WIDGET_BASE_STYLES: dict[str, str] = {
    "GuiPanel": "background: rgba(55, 55, 65, 90); border: 1px solid #555;",
    "GuiLabel": "color: #ddd; background: transparent; border: none; padding: 2px; font-size: 13px;",
    "GuiButton": "background-color: #4a7ab5; color: #fff; border: 1px solid #5a8ac5; border-radius: 4px; padding: 4px 12px; font-size: 13px;",
    "GuiSlider": "border: 1px solid #555; border-radius: 3px; background: #3a3a3a;",
    "GuiTextInput": "background-color: #1e1e1e; color: #fff; border: 1px solid #555; border-radius: 3px; padding: 4px 6px; font-size: 13px;",
    "GuiToggle": "color: #ddd; spacing: 6px; font-size: 13px;",
    "GuiProgressBar": "border: 1px solid #555; border-radius: 3px; text-align: center; color: #ddd; background: #333; font-size: 11px;",
    "GuiDropdown": "background-color: #1e1e1e; color: #ddd; border: 1px solid #555; border-radius: 3px; padding: 4px 6px; font-size: 13px;",
    "GuiScrollPanel": "border: 1px solid #555; border-radius: 4px; background: #2d2d2d;",
    "GuiImage": "background: transparent; border: none;",
    "GuiRadioButton": "color: #ddd; spacing: 6px; font-size: 13px;",
    "GuiListWidget": "background: #1e1e1e; color: #ddd; border: 1px solid #555;",
    "GuiTableWidget": "background: #1e1e1e; color: #ddd; border: 1px solid #555;",
    "GuiTreeWidget": "background: #1e1e1e; color: #ddd; border: 1px solid #555;",
    "GuiTabWidget": "background: #2d2d2d;",
    "GuiGroupBox": "color: #ddd; border: 1px solid #555; border-radius: 4px; font-size: 13px;",
    "GuiSpinBox": "background: #1e1e1e; color: #ddd; border: 1px solid #555; border-radius: 3px; font-size: 13px;",
    "GuiDoubleSpinBox": "background: #1e1e1e; color: #ddd; border: 1px solid #555; border-radius: 3px; font-size: 13px;",
    "GuiTextEdit": "background: #1e1e1e; color: #ddd; border: 1px solid #555; border-radius: 3px; font-size: 13px;",
    "GuiHtmlView": "background: #1e1e1e; color: #ddd; border: 1px solid #555; border-radius: 3px; font-size: 13px;",
    "GuiDial": "background: transparent;",
    "GuiSplitter": "border: 1px solid #555; background: #2d2d2d;",
    "GuiStackedWidget": "background: #2d2d2d; border: 1px solid #555;",
    "GuiToolBox": "background: #2d2d2d; border: 1px solid #555;",
    "GuiCalendar": "background: #1e1e1e; color: #ddd; border: 1px solid #555;",
    "GuiLCDNumber": "color: #55ff55; background: #1a1a1a; border: 1px solid #555; padding: 4px; font-size: 20px;",
    "GuiPlainText": "background: #1e1e1e; color: #ddd; border: 1px solid #555; border-radius: 3px; font-size: 13px;",
    "GuiScrollBar": "background: #2d2d2d; border: 1px solid #555;",
    "GuiToolButton": "background-color: #4a4a4a; color: #ddd; border: 1px solid #666; border-radius: 3px; padding: 2px 6px; font-size: 12px;",
    "GuiFontCombo": "background: #1e1e1e; color: #ddd; border: 1px solid #555; border-radius: 3px; padding: 2px 4px; font-size: 12px;",
"GuiMdiArea": "background: #1e1e1e; border: 1px solid #555;",
    "GuiMdiSubWindow": "background: #2d2d2d; color: #ddd;",
    "GuiChart": "background: #1e1e1e; border: 1px solid #555;",
}


_WIDGET_EXTRA_RULES: dict[str, list[str]] = {
    "GuiButton": [
        "QPushButton#GuiButton:hover { background-color: #5a8ac5; }",
        "QPushButton#GuiButton:pressed { background-color: #3a6aa5; }",
    ],
    "GuiSlider": [
        "QSlider::groove:horizontal { border: 1px solid #555; height: 6px; background: #3a3a3a; border-radius: 3px; }",
        "QSlider::handle:horizontal { background: #ddd; border: 1px solid #aaa; width: 12px; margin: -4px 0; border-radius: 4px; }",
        "QSlider::sub-page:horizontal { background: #4a7ab5; border-radius: 3px; }",
    ],
    "GuiDropdown": [
        "QComboBox#GuiDropdown::drop-down { border: none; width: 20px; }",
        "QComboBox#GuiDropdown QAbstractItemView { background: #2d2d2d; color: #ddd; selection-background-color: #4a7ab5; }",
    ],
    "GuiProgressBar": [
        "QProgressBar::chunk { background-color: #4a7ab5; border-radius: 2px; }",
    ],
    "GuiRadioButton": [
        "QRadioButton#GuiRadioButton { color: #ddd; spacing: 6px; font-size: 13px; }",
    ],
    "GuiListWidget": [
        "QListWidget#GuiListWidget { background: #1e1e1e; color: #ddd; border: 1px solid #555; outline: none; }",
        "QListWidget#GuiListWidget::item:selected { background: #4a7ab5; }",
    ],
    "GuiTableWidget": [
        "QTableWidget#GuiTableWidget { background: #1e1e1e; color: #ddd; border: 1px solid #555; gridline-color: #444; }",
        "QTableWidget#GuiTableWidget::item:selected { background: #4a7ab5; }",
        "QHeaderView::section { background: #2d2d2d; color: #ccc; border: 1px solid #444; padding: 3px; }",
    ],
    "GuiTreeWidget": [
        "QTreeWidget#GuiTreeWidget { background: #1e1e1e; color: #ddd; border: 1px solid #555; outline: none; }",
        "QTreeWidget#GuiTreeWidget::item:selected { background: #4a7ab5; }",
    ],
    "GuiTabWidget": [
        "QTabWidget#GuiTabWidget::pane { border: 1px solid #555; background: #2d2d2d; }",
        "QTabBar::tab { background: #333; color: #aaa; border: 1px solid #555; padding: 6px 12px; }",
        "QTabBar::tab:selected { background: #4a7ab5; color: #fff; }",
    ],
    "GuiGroupBox": [
        "QGroupBox#GuiGroupBox { color: #ddd; border: 1px solid #555; border-radius: 4px; margin-top: 8px; padding-top: 12px; font-size: 13px; }",
        "QGroupBox#GuiGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }",
    ],
    "GuiSpinBox": [
        "QSpinBox#GuiSpinBox { background: #1e1e1e; color: #ddd; border: 1px solid #555; border-radius: 3px; padding: 2px 4px; font-size: 13px; }",
        "QSpinBox#GuiSpinBox::up-button { border-left: 1px solid #555; width: 16px; }",
        "QSpinBox#GuiSpinBox::down-button { border-left: 1px solid #555; width: 16px; }",
    ],
    "GuiDoubleSpinBox": [
        "QDoubleSpinBox#GuiDoubleSpinBox { background: #1e1e1e; color: #ddd; border: 1px solid #555; border-radius: 3px; padding: 2px 4px; font-size: 13px; }",
        "QDoubleSpinBox#GuiDoubleSpinBox::up-button { border-left: 1px solid #555; width: 16px; }",
        "QDoubleSpinBox#GuiDoubleSpinBox::down-button { border-left: 1px solid #555; width: 16px; }",
    ],
    "GuiTextEdit": [
        "QTextEdit#GuiTextEdit { background: #1e1e1e; color: #ddd; border: 1px solid #555; border-radius: 3px; padding: 4px; font-size: 13px; }",
    ],
    "GuiDial": [
        "QDial#GuiDial { background: transparent; }",
    ],
    "GuiHtmlView": [
        "QTextBrowser#GuiHtmlView { background: #1e1e1e; color: #ddd; border: 1px solid #555; border-radius: 3px; padding: 4px; font-size: 13px; }",
    ],
    "GuiSplitter": [
        "QSplitter::handle { background: #444; }",
        "QSplitter::handle:horizontal { width: 4px; }",
        "QSplitter::handle:vertical { height: 4px; }",
    ],
    "GuiToolBox": [
        "QToolBox::tab { background: #333; color: #ddd; border: 1px solid #555; padding: 4px 8px; }",
        "QToolBox::tab:selected { background: #4a7ab5; color: #fff; }",
    ],
    "GuiCalendar": [
        "QCalendarWidget QWidget { background: #1e1e1e; color: #ddd; }",
        "QCalendarWidget QToolButton { color: #ddd; background: #333; border: 1px solid #555; padding: 4px; }",
    ],
    "GuiScrollBar": [
        "QScrollBar:horizontal { border: none; background: #2d2d2d; height: 12px; }",
        "QScrollBar::handle:horizontal { background: #555; min-width: 20px; border-radius: 3px; }",
        "QScrollBar:vertical { border: none; background: #2d2d2d; width: 12px; }",
        "QScrollBar::handle:vertical { background: #555; min-height: 20px; border-radius: 3px; }",
    ],
    "GuiToolButton": [
        "QToolButton#GuiToolButton:hover { background-color: #5a5a5a; }",
        "QToolButton#GuiToolButton:pressed { background-color: #3a3a3a; }",
    ],
    "GuiFontCombo": [
        "QFontComboBox#GuiFontCombo { background: #1e1e1e; color: #ddd; border: 1px solid #555; border-radius: 3px; padding: 2px 4px; }",
        "QFontComboBox#GuiFontCombo::drop-down { border: none; width: 18px; }",
        "QFontComboBox#GuiFontCombo QAbstractItemView { background: #2d2d2d; color: #ddd; selection-background-color: #4a7ab5; }",
    ],
    "GuiMdiArea": [
        "QMdiArea#GuiMdiArea { background: #1e1e1e; border: none; }",
    ],
    "GuiMdiSubWindow": [
        "QMdiSubWindow#GuiMdiSubWindow { background: #2d2d2d; color: #ddd; }",
    ],
}


def get_widget_base_style(object_name: str) -> str:
    return _WIDGET_BASE_STYLES.get(object_name, "")


def get_widget_extra_rules(object_name: str) -> list[str]:
    return _WIDGET_EXTRA_RULES.get(object_name, [])


def get_widget_type_for_object(obj_name: str) -> str:
    TYPE_MAP = {
        "GuiPanel": "QWidget", "GuiLabel": "QLabel", "GuiButton": "QPushButton",
        "GuiSlider": "QSlider", "GuiTextInput": "QLineEdit", "GuiToggle": "QCheckBox",
        "GuiProgressBar": "QProgressBar", "GuiDropdown": "QComboBox",
        "GuiScrollPanel": "QScrollArea", "GuiImage": "QWidget",
        "GuiRadioButton": "QRadioButton", "GuiListWidget": "QListWidget",
        "GuiTableWidget": "QTableWidget", "GuiTreeWidget": "QTreeWidget",
        "GuiTabWidget": "QTabWidget", "GuiGroupBox": "QGroupBox",
        "GuiSpinBox": "QSpinBox", "GuiDoubleSpinBox": "QDoubleSpinBox",
        "GuiTextEdit": "QTextEdit", "GuiHtmlView": "QTextBrowser", "GuiDial": "QDial",
        "GuiSplitter": "QSplitter", "GuiStackedWidget": "QStackedWidget",
        "GuiToolBox": "QToolBox", "GuiCalendar": "QCalendarWidget",
        "GuiLCDNumber": "QLCDNumber", "GuiPlainText": "QPlainTextEdit",
        "GuiScrollBar": "QScrollBar", "GuiToolButton": "QToolButton",
        "GuiFontCombo": "QFontComboBox",
"GuiMdiArea": "QMdiArea", "GuiMdiSubWindow": "QMdiSubWindow",
        "GuiChart": "QWidget",
    }
    return TYPE_MAP.get(obj_name, "QWidget")


def apply_fusion_style(widget: QWidget):
    """Apply Fusion-compatible dark stylesheet to a widget tree."""
    widget.setStyleSheet("""
        QWidget#GuiPanel {
            background: rgba(55, 55, 65, 90);
            border: 1px solid #555;
        }
        QLabel#GuiLabel {
            color: #ddd;
            background: transparent;
            border: none;
            padding: 2px;
        }
        QPushButton#GuiButton {
            background-color: #4a7ab5;
            color: #fff;
            border: 1px solid #5a8ac5;
            border-radius: 4px;
            padding: 4px 12px;
            font-size: 13px;
        }
        QPushButton#GuiButton:hover {
            background-color: #5a8ac5;
        }
        QPushButton#GuiButton:pressed {
            background-color: #3a6aa5;
        }
        QSlider::groove:horizontal {
            border: 1px solid #555;
            height: 6px;
            background: #3a3a3a;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #ddd;
            border: 1px solid #aaa;
            width: 12px;
            margin: -4px 0;
            border-radius: 4px;
        }
        QSlider::sub-page:horizontal {
            background: #4a7ab5;
            border-radius: 3px;
        }
        QLineEdit#GuiTextInput {
            background-color: #1e1e1e;
            color: #fff;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 4px 6px;
        }
        QCheckBox#GuiToggle {
            color: #ddd;
            spacing: 6px;
        }
        QProgressBar#GuiProgressBar {
            border: 1px solid #555;
            border-radius: 3px;
            text-align: center;
            color: #ddd;
            background: #333;
        }
        QProgressBar#GuiProgressBar::chunk {
            background-color: #4a7ab5;
            border-radius: 2px;
        }
        QComboBox#GuiDropdown {
            background-color: #1e1e1e;
            color: #ddd;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 4px 6px;
        }
        QComboBox#GuiDropdown::drop-down {
            border: none;
            width: 20px;
        }
        QComboBox#GuiDropdown QAbstractItemView {
            background: #2d2d2d;
            color: #ddd;
            selection-background-color: #4a7ab5;
        }
        QScrollArea#GuiScrollPanel {
            border: 1px solid #555;
            border-radius: 4px;
            background: #2d2d2d;
        }
        QRadioButton#GuiRadioButton {
            color: #ddd;
            spacing: 6px;
            font-size: 13px;
        }
        QListWidget#GuiListWidget {
            background: #1e1e1e;
            color: #ddd;
            border: 1px solid #555;
            outline: none;
        }
        QListWidget#GuiListWidget::item:selected {
            background: #4a7ab5;
        }
        QTableWidget#GuiTableWidget {
            background: #1e1e1e;
            color: #ddd;
            border: 1px solid #555;
            gridline-color: #444;
        }
        QTableWidget#GuiTableWidget::item:selected {
            background: #4a7ab5;
        }
        QHeaderView::section {
            background: #2d2d2d;
            color: #ccc;
            border: 1px solid #444;
            padding: 3px;
        }
        QTreeWidget#GuiTreeWidget {
            background: #1e1e1e;
            color: #ddd;
            border: 1px solid #555;
            outline: none;
        }
        QTreeWidget#GuiTreeWidget::item:selected {
            background: #4a7ab5;
        }
        QTabWidget#GuiTabWidget::pane {
            border: 1px solid #555;
            background: #2d2d2d;
        }
        QTabBar::tab {
            background: #333;
            color: #aaa;
            border: 1px solid #555;
            padding: 6px 12px;
        }
        QTabBar::tab:selected {
            background: #4a7ab5;
            color: #fff;
        }
        QGroupBox#GuiGroupBox {
            color: #ddd;
            border: 1px solid #555;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 12px;
            font-size: 13px;
        }
        QGroupBox#GuiGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
        QSpinBox#GuiSpinBox {
            background: #1e1e1e;
            color: #ddd;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 2px 4px;
            font-size: 13px;
        }
        QSpinBox#GuiSpinBox::up-button {
            border-left: 1px solid #555;
            width: 16px;
        }
        QSpinBox#GuiSpinBox::down-button {
            border-left: 1px solid #555;
            width: 16px;
        }
        QDoubleSpinBox#GuiDoubleSpinBox {
            background: #1e1e1e;
            color: #ddd;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 2px 4px;
            font-size: 13px;
        }
        QDoubleSpinBox#GuiDoubleSpinBox::up-button {
            border-left: 1px solid #555;
            width: 16px;
        }
        QDoubleSpinBox#GuiDoubleSpinBox::down-button {
            border-left: 1px solid #555;
            width: 16px;
        }
        QTextEdit#GuiTextEdit {
            background: #1e1e1e;
            color: #ddd;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 4px;
            font-size: 13px;
        }
        QDial#GuiDial {
            background: transparent;
        }
    """)


_ANCHOR_NAMES = [
    "top-left", "top-center", "top-right",
    "center-left", "center", "center-right",
    "bottom-left", "bottom-center", "bottom-right",
    "stretch-all", "stretch-width", "stretch-height",
]

_ANCHOR_LABELS: dict[int, str] = {
    ANCHOR_TOP_LEFT: "Top Left",
    ANCHOR_TOP_CENTER: "Top Center",
    ANCHOR_TOP_RIGHT: "Top Right",
    ANCHOR_CENTER_LEFT: "Center Left",
    ANCHOR_CENTER: "Center",
    ANCHOR_CENTER_RIGHT: "Center Right",
    ANCHOR_BOTTOM_LEFT: "Bottom Left",
    ANCHOR_BOTTOM_CENTER: "Bottom Center",
    ANCHOR_BOTTOM_RIGHT: "Bottom Right",
    ANCHOR_STRETCH_ALL: "Stretch All",
    ANCHOR_STRETCH_WIDTH: "Stretch Width",
    ANCHOR_STRETCH_HEIGHT: "Stretch Height",
}


class AnchorPresetSelector(QWidget):
    anchor_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._anchor = ANCHOR_TOP_LEFT
        self._cell_size = 18
        self._gap = 2
        w = self._cell_size * 3 + self._gap * 2
        h = self._cell_size * 4 + self._gap * 3
        self.setFixedSize(w + 12, h + 12)
        self.setMouseTracking(True)
        self._hovered = -1

    def set_anchor(self, value: int):
        if value != self._anchor:
            self._anchor = value
            self.update()

    @property
    def anchor(self) -> int:
        return self._anchor

    def _cell_rect(self, idx: int) -> tuple[int, int, int, int]:
        col = idx % 3
        row = idx // 3
        x = 6 + col * (self._cell_size + self._gap)
        y = 6 + row * (self._cell_size + self._gap)
        return x, y, self._cell_size, self._cell_size

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QColor("#2d2d2d")
        selected_color = QColor("#4a7ab5")
        hover_color = QColor("#3a5a8a")
        empty_color = QColor("#444")
        text_color = QColor("#aaa")
        for i in range(9):
            rx, ry, rw, rh = self._cell_rect(i)
            if i == self._anchor:
                p.setBrush(selected_color)
                p.setPen(Qt.PenStyle.NoPen)
            elif i == self._hovered:
                p.setBrush(hover_color)
                p.setPen(Qt.PenStyle.NoPen)
            else:
                p.setBrush(empty_color)
                p.setPen(QPen(QColor("#555"), 1))
            p.drawRoundedRect(rx, ry, rw, rh, 2, 2)
        for i in range(9, 12):
            rx, ry, rw, rh = self._cell_rect(i)
            if i == self._anchor:
                p.setBrush(selected_color)
            elif i == self._hovered:
                p.setBrush(hover_color)
            else:
                p.setBrush(empty_color)
            p.setPen(QPen(QColor("#555"), 1))
            p.drawRoundedRect(rx, ry, rw, rh, 2, 2)
            label = _ANCHOR_NAMES[i].split("-")[-1]
            p.setPen(text_color)
            p.setFont(self.font())
            p.drawText(rx + 2, ry, rw - 2, rh, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
        label = _ANCHOR_LABELS.get(self._anchor, "")
        if label:
            p.setPen(QColor("#ddd"))
            p.setFont(self.font())
            p.drawText(6, self.height() - 14, self.width() - 12, 14,
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"Anchor: {label}")
        p.end()

    def mousePressEvent(self, event):
        mx, my = event.position().x(), event.position().y()
        for i in range(12):
            rx, ry, rw, rh = self._cell_rect(i)
            if rx <= mx <= rx + rw and ry <= my <= ry + rh:
                self._anchor = i
                self.anchor_changed.emit(i)
                self.update()
                return

    def mouseMoveEvent(self, event):
        mx, my = event.position().x(), event.position().y()
        old = self._hovered
        self._hovered = -1
        for i in range(12):
            rx, ry, rw, rh = self._cell_rect(i)
            if rx <= mx <= rx + rw and ry <= my <= ry + rh:
                self._hovered = i
                break
        if old != self._hovered:
            self.update()
