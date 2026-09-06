# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFileDialog, QMessageBox,
                             QComboBox, QLineEdit, QDialogButtonBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication

from tools.build_zplugin import build_zplugin, load_source_manifest


class PluginPackageDialog(QDialog):
    def __init__(self, parent=None, plugin_src: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Package .zplugin")
        self.setMinimumWidth(460)
        self._result_path: str | None = None
        self._setup_ui(plugin_src)

    def _setup_ui(self, plugin_src: str):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Plugin package directory:"))
        src_row = QHBoxLayout()
        self._src_edit = QLineEdit(plugin_src)
        src_row.addWidget(self._src_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_src)
        src_row.addWidget(browse_btn)
        layout.addLayout(src_row)

        self._meta_label = QLabel("No manifest loaded.")
        self._meta_label.setWordWrap(True)
        layout.addWidget(self._meta_label)

        layout.addWidget(QLabel("Output directory:"))
        out_row = QHBoxLayout()
        self._out_edit = QLineEdit("dist")
        out_row.addWidget(self._out_edit)
        out_browse = QPushButton("Browse...")
        out_browse.clicked.connect(self._browse_out)
        out_row.addWidget(out_browse)
        layout.addLayout(out_row)

        layout.addWidget(QLabel("Build mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["source", "cython", "nuitka"])
        self._mode_combo.setToolTip(
            "source: ship .py files (any platform). "
            "cython/nuitka: compile modules to extensions for this machine.")
        layout.addWidget(self._mode_combo)

        self._build_btn = QPushButton("Build .zplugin")
        self._build_btn.clicked.connect(self._build)
        layout.addWidget(self._build_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._src_edit.textChanged.connect(self._refresh_meta)
        self._refresh_meta()

    def _browse_src(self):
        d = QFileDialog.getExistingDirectory(self, "Select Plugin Package Directory", "")
        if d:
            self._src_edit.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", "")
        if d:
            self._out_edit.setText(d)

    def _refresh_meta(self):
        src = self._src_edit.text().strip()
        try:
            meta = load_source_manifest(src)
            self._meta_label.setText(
                f"{meta.get('name', '?')} v{meta.get('version', '?')} "
                f"(module: {meta.get('module', meta.get('name', '?'))})")
            self._build_btn.setEnabled(True)
        except Exception as e:
            self._meta_label.setText(f"No manifest loaded: {e}")
            self._build_btn.setEnabled(False)

    def _build(self):
        src = self._src_edit.text().strip()
        out = self._out_edit.text().strip() or "dist"
        mode = self._mode_combo.currentText()
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        err = ""
        try:
            result = build_zplugin(src, out, mode)
        except Exception as e:
            result = None
            err = str(e)
        finally:
            QGuiApplication.restoreOverrideCursor()
        if result:
            self._result_path = result
            QMessageBox.information(self, "Build Complete",
                                    f"Package built successfully.\nOutput: {result}")
        else:
            QMessageBox.critical(self, "Build Failed",
                                 f"Packaging failed{(': ' + err) if err else ''}. "
                                 "Check the console for details.")

    @property
    def result_path(self) -> str | None:
        return self._result_path


def show_package_dialog(parent=None, plugin_src: str = "") -> str | None:
    dlg = PluginPackageDialog(parent, plugin_src)
    dlg.exec()
    return dlg.result_path
