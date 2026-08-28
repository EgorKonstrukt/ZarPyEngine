# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.config.editor_scale import scale, scale_xy
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton,
                             QLabel, QScrollArea, QFrame, QLineEdit, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor


_CATEGORIES: dict[str, list[tuple[str, str, str, str]]] = {
    "Containers": [
        ("Panel", "panel", "#3a3a3a", "Empty container for grouping widgets"),
        ("Scroll Panel", "scrollpanel", "#3a3a5a", "Scrollable container"),
        ("Group Box", "groupbox", "#4a4a3a", "Grouped container with title"),
        ("Tab Widget", "tabwidget", "#5a3a4a", "Multi-tab container"),
        ("Splitter", "splitter", "#4a3a3a", "Resizable splitter container"),
        ("Stacked Widget", "stackedwidget", "#3a4a3a", "Stacked/page switcher"),
        ("Tool Box", "toolbox", "#3a3a4a", "Collapsible section container"),
        ("Mdi Area", "mdiarea", "#3a4a5a", "Multiple document interface area"),
    ],
    "Display": [
        ("Label", "label", "#2d5a3a", "Static text display"),
        ("Image", "image", "#5a3a5a", "Image display widget"),
        ("Progress Bar", "progressbar", "#5a7a3a", "Progress indicator"),
        ("List Widget", "listwidget", "#3a5a5a", "Item list display"),
        ("Table Widget", "tablewidget", "#3a5a7a", "Grid/table display"),
        ("Tree Widget", "treewidget", "#4a5a3a", "Hierarchical tree display"),
        ("Text Edit", "textedit", "#5a5a5a", "Multi-line text editor"),
        ("Plain Text", "plaintext", "#5a5a5a", "Plain text editor"),
        ("Html View", "html", "#5a5a5a", "HTML rich text viewer"),
        ("LCD Number", "lcdnumber", "#3a6a3a", "Digital number display"),
        ("Calendar", "calendar", "#5a5a3a", "Date picker calendar"),
        ("Chart", "chart", "#3a5a7a", "Interactive chart/plot widget"),
    ],
    "Layout": [
        ("Horizontal Layout", "horizontallayout", "#2a4a3a", "Arranges children left to right"),
        ("Vertical Layout", "verticallayout", "#2a4a3a", "Arranges children top to bottom"),
        ("Grid Layout", "gridlayout", "#2a4a3a", "Arranges children in a grid"),
        ("Layout Element", "layoutelement", "#3a5a3a", "Per-child layout sizing control"),
    ],
    "Input": [
        ("Button", "button", "#4a7ab5", "Clickable button"),
        ("Text Input", "textinput", "#5a5a7a", "Single-line text input"),
        ("Slider", "slider", "#7a5a3a", "Horizontal slider control"),
        ("Toggle", "toggle", "#3a6a5a", "On/off toggle switch"),
        ("Dropdown", "dropdown", "#6a5a3a", "Dropdown selection menu"),
        ("Radio Button", "radiobutton", "#7a4a4a", "Single-select radio button"),
        ("Spin Box", "spinbox", "#4a6a4a", "Integer number input"),
        ("Double Spin Box", "doublespinbox", "#4a7a6a", "Float number input"),
        ("Dial", "dial", "#6a5a5a", "Rotary dial control"),
        ("Scroll Bar", "scrollbar", "#7a6a4a", "Scrollable value control"),
        ("Tool Button", "toolbutton", "#5a6a7a", "Toolbar-style button"),
        ("Font Combo", "fontcombo", "#5a5a6a", "Font selection dropdown"),
    ],
}


class WidgetPaletteButton(QPushButton):
    def __init__(self, text: str, widget_type: str, color: str, tooltip: str, parent=None):
        super().__init__(text, parent)
        self._widget_type = widget_type
        self._color = color
        self.setFixedHeight(scale(28))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 10px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {_lighten(color)};
                border: 1px solid #4a7ab5;
            }}
        """)

    @property
    def widget_type(self) -> str:
        return self._widget_type


def _lighten(hex_color: str) -> str:
    c = QColor(hex_color)
    return f"#{min(255, c.red() + 35):02x}{min(255, c.green() + 35):02x}{min(255, c.blue() + 35):02x}"


class WidgetPalette(QFrame):
    widget_added = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(150)
        self.setMaximumWidth(200)
        self._all_items: list[tuple[str, str, str, str, str]] = []
        self._widgets: list[QWidget] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        title = QLabel("Widget Palette")
        title.setStyleSheet("color: #bbb; font-weight: bold; font-size: 11px; padding: 2px 4px;")
        layout.addWidget(title)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search widgets...")
        self._search.setStyleSheet("""
            QLineEdit {
                background-color: #1e1e1e; color: #ccc; border: 1px solid #444;
                border-radius: 3px; padding: 3px 6px; font-size: 10px;
            }
        """)
        self._search.textChanged.connect(self._filter)
        layout.addWidget(self._search)
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._scroll_content = QWidget()
        self._scroll_content.setStyleSheet("background: transparent;")
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setSpacing(2)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_area.setWidget(self._scroll_content)
        layout.addWidget(self._scroll_area)
        self._build()

    def _build(self):
        for cat_name, items in _CATEGORIES.items():
            section = QLabel(cat_name)
            section.setStyleSheet("color: #777; font-size: 9px; padding: 4px 4px 1px 4px;")
            self._scroll_layout.addWidget(section)
            self._widgets.append(section)
            for name, wtype, color, desc in items:
                self._all_items.append((name, wtype, color, desc, cat_name))
                btn = WidgetPaletteButton(name, wtype, color, desc)
                btn.clicked.connect(lambda checked=False, wt=wtype: self.widget_added.emit(wt))
                self._scroll_layout.addWidget(btn)
                self._widgets.append(btn)
        self._scroll_layout.addStretch()

    def _filter(self, text: str):
        text = text.lower()
        for i, item in enumerate(self._all_items):
            name, wtype, color, desc, cat = item
            btn = self._widgets[i * 2 + 1]
            show = not text or text in name.lower() or text in desc.lower() or text in wtype.lower()
            btn.setVisible(show)
        section_visible: dict[str, bool] = {}
        for item, w in zip(self._all_items, [w for w in self._widgets if isinstance(w, WidgetPaletteButton)]):
            name, wtype, color, desc, cat = item
            section_visible.setdefault(cat, False)
            if w.isVisible():
                section_visible[cat] = True
        idx = 0
        for cat_name in _CATEGORIES:
            label = self._widgets[idx]
            label.setVisible(section_visible.get(cat_name, False))
            idx += 1 + len(_CATEGORIES[cat_name])
