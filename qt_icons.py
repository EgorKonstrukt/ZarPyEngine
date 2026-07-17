# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QStyle
from PyQt6.QtGui import QGuiApplication

COL_SIZE = 4

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Standard Icons')
        layout = QGridLayout(self)
        count = 0
        clipboard = QGuiApplication.clipboard()
        for attr in dir(QStyle.StandardPixmap):
            if attr.startswith('SP_'):
                icon_attr = getattr(QStyle.StandardPixmap, attr)
                btn = QPushButton(attr)
                btn.setIcon(self.style().standardIcon(icon_attr))
                def clip_copy_fn(text):
                    def clip_copy():
                        clipboard.setText(text)
                    return clip_copy
                btn.clicked.connect(clip_copy_fn(attr))
                layout.addWidget(btn, count // COL_SIZE, count % COL_SIZE)
                count += 1


if __name__ == '__main__':
    app = QApplication([])
    # app.setStyle('Windows')
    w = Widget()
    w.show()
    app.exec()