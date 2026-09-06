# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
from typing import Optional
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                              QTableWidget, QTableWidgetItem, QHeaderView,
                              QPushButton, QFileDialog, QMessageBox,
                              QLabel, QCheckBox, QFrame, QDialog,
                              QDialogButtonBox, QComboBox, QLineEdit)
from PyQt6.QtCore import Qt
from core.foundation.plugin_manager import PluginBase
from tools.build_plugin import build_plugin


class PluginManagerPanel(QDockWidget):
    def __init__(self, engine, parent=None):
        super().__init__("Plugin Manager", parent)
        self._engine = engine
        self._setup_ui()
        self._refresh()

    def _setup_ui(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        hdr = QHBoxLayout()
        title = QLabel("Installed Plugins")
        title.setStyleSheet("font-weight: bold;")
        hdr.addWidget(title)
        hdr.addStretch()
        self._count_label = QLabel("0 plugins")
        hdr.addWidget(self._count_label)
        layout.addLayout(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(["", "Name", "Version", "Description", "System", "Status"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 30)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setStyleSheet("""
            QTableWidget::item { padding: 4px; }
            QHeaderView::section { padding: 4px; }
        """)
        self._table.cellChanged.connect(self._on_cell_changed)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._load_btn = QPushButton("Load Plugin...")
        self._load_btn.clicked.connect(self._load_plugin)
        self._load_btn.setStyleSheet("padding: 4px 12px;")
        btn_row.addWidget(self._load_btn)
        self._build_btn = QPushButton("Build Plugin...")
        self._build_btn.clicked.connect(self._build_plugin)
        self._build_btn.setStyleSheet("padding: 4px 12px;")
        btn_row.addWidget(self._build_btn)
        self._package_btn = QPushButton("Package .zplugin...")
        self._package_btn.clicked.connect(self._package_zplugin)
        self._package_btn.setStyleSheet("padding: 4px 12px;")
        btn_row.addWidget(self._package_btn)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh)
        self._refresh_btn.setStyleSheet("padding: 4px 12px;")
        btn_row.addWidget(self._refresh_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        self.setWidget(w)

    def _refresh(self):
        manager = self._engine.plugin_manager
        plugins = manager.get_all()
        shown = {p.NAME for p in plugins}
        extra = [(n, s) for n, s in manager.all_plugin_statuses().items()
                 if n not in shown and s.get("state", "") != "loaded"]
        self._table.blockSignals(True)
        self._table.setRowCount(len(plugins) + len(extra))
        for i, p in enumerate(plugins):
            cb = QCheckBox()
            cb.setChecked(p.enabled)
            cb.stateChanged.connect(lambda state, idx=i: self._toggle_plugin(idx, state))
            cw = QWidget()
            layout = QHBoxLayout(cw)
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(cb)
            self._table.setCellWidget(i, 0, cw)

            name_item = QTableWidgetItem(p.NAME)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 1, name_item)

            ver_item = QTableWidgetItem(p.VERSION)
            ver_item.setFlags(ver_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 2, ver_item)

            desc_item = QTableWidgetItem(p.DESCRIPTION)
            desc_item.setFlags(desc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 3, desc_item)

            sys_item = QTableWidgetItem("Yes" if p.SYSTEM else "")
            sys_item.setFlags(sys_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            sys_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 4, sys_item)

            self._table.setItem(i, 5, self._status_item(manager.plugin_status(p.NAME)))
        for j, (n, s) in enumerate(extra):
            i = len(plugins) + j
            name_item = QTableWidgetItem(n)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 1, name_item)

            desc_item = QTableWidgetItem(s.get("reason", ""))
            desc_item.setFlags(desc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 3, desc_item)

            self._table.setItem(i, 5, self._status_item(s))
        self._table.blockSignals(False)
        total = len(plugins) + len(extra)
        self._count_label.setText(f"{total} plugin{'s' if total != 1 else ''}")

    @staticmethod
    def _status_item(status: dict):
        item = QTableWidgetItem(status.get("state", ""))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        reason = status.get("reason", "")
        if reason:
            item.setToolTip(reason)
        return item

    def _toggle_plugin(self, row: int, state: int):
        plugins = self._engine.plugin_manager.get_all()
        if row < len(plugins):
            plugins[row].enabled = (state == Qt.CheckState.Checked.value)

    def _on_cell_changed(self, row: int, col: int):
        pass

    def _load_plugin(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Plugin", "", "Python Plugin (*.py);;Native Plugin (*.dll *.so);;Plugin Package (*.zplugin)"
        )
        if path:
            try:
                self._engine.plugin_manager.load_from_file(path)
                self._refresh()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load plugin:\n{e}")

    def _build_plugin(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Plugin to Build", "",
            "Python Plugin (*.py);;Plugin Directory (select __init__.py)"
        )
        if not path:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Build Plugin")
        dlg.setMinimumWidth(400)
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel("Output directory:"))
        out_edit = QLineEdit("dist")
        layout.addWidget(out_edit)

        layout.addWidget(QLabel("Build backend:"))
        backend_combo = QComboBox()
        backend_combo.addItems(["auto", "nuitka", "cython"])
        layout.addWidget(backend_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        output_dir = out_edit.text().strip() or "dist"
        backend = backend_combo.currentText()

        from PyQt6.QtGui import QGuiApplication
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            success = build_plugin(path, output_dir, backend)
        finally:
            QGuiApplication.restoreOverrideCursor()

        if success:
            out_path = os.path.abspath(output_dir)
            QMessageBox.information(self, "Build Complete",
                                    f"Plugin built successfully.\nOutput: {out_path}")
        else:
            QMessageBox.critical(self, "Build Failed",
                                 "Plugin build failed. Check the console for details.")

    def _package_zplugin(self):
        from editor.panels.plugin_package_panel import show_package_dialog
        show_package_dialog(self)
