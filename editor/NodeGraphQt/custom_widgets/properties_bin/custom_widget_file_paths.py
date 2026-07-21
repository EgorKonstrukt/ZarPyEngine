# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

#!/usr/bin/python
import qtawesome as qta
from PyQt6 import QtWidgets, QtCore

from editor.NodeGraphQt.widgets.dialogs import FileDialog
from .prop_widgets_abstract import BaseProperty


class PropFilePath(BaseProperty):
    """
    Displays a node property as a "QFileDialog" open widget in the
    PropertiesBin.
    """

    def __init__(self, parent=None):
        super(PropFilePath, self).__init__(parent)
        self._ledit = QtWidgets.QLineEdit()
        self._ledit.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self._ledit.editingFinished.connect(self._on_value_change)
        self._ledit.clearFocus()

        icon = qta.icon("fa5s.folder-open", color="#d4d4d4")
        _button = QtWidgets.QPushButton()
        _button.setIcon(icon)
        _button.clicked.connect(self._on_select_file)

        hbox = QtWidgets.QHBoxLayout(self)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self._ledit)
        hbox.addWidget(_button)

        self._ext = '*'
        self._file_directory = None

    def _on_select_file(self):
        file_path = FileDialog.getOpenFileName(self,
                                               file_dir=self._file_directory,
                                               ext_filter=self._ext)
        file = file_path[0] or None
        if file:
            self.set_value(file)

    def _on_value_change(self, value=None):
        if value is None:
            value = self._ledit.text()
        self.set_file_directory(value)
        self.value_changed.emit(self.get_name(), value)

    def set_file_ext(self, ext=None):
        self._ext = ext or '*'

    def set_file_directory(self, directory):
        self._file_directory = directory

    def get_value(self):
        return self._ledit.text()

    def set_value(self, value):
        _value = str(value)
        if _value != self.get_value():
            self._ledit.setText(_value)
            self._on_value_change(_value)


class PropFileSavePath(PropFilePath):
    """
    Displays a node property as a "QFileDialog" save widget in the
    PropertiesBin.
    """

    def _on_select_file(self):
        file_path = FileDialog.getSaveFileName(self,
                                               file_dir=self._file_directory,
                                               ext_filter=self._ext)
        file = file_path[0] or None
        if file:
            self.set_value(file)
