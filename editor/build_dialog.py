# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

import sys
import os
import subprocess
import threading
from pathlib import Path
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QPushButton, QLabel, QLineEdit, QCheckBox,
                             QGroupBox, QTextEdit, QProgressBar, QFileDialog,
                             QMessageBox, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
from core.config.editor_scale import scale

ROOT = Path(__file__).resolve().parent.parent


class BuildDialog(QDialog):
    _log_signal = pyqtSignal(str)
    _done_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_process = None
        self._build_thread = None
        self._cancelled = False

        self.setWindowTitle("Build Zarin Engine")
        self.setMinimumSize(600, 500)
        self.resize(700, 550)

        self._setup_ui()
        self._log_signal.connect(self._append_log)
        self._done_signal.connect(self._on_build_done)

    def _setup_ui(self):
        main = QVBoxLayout(self)
        main.setSpacing(8)

        # Settings
        settings_group = QGroupBox("Build Settings")
        form = QFormLayout(settings_group)

        self._target_combo = QComboBox()
        self._target_combo.addItem("Player (player.py)", "player")
        self._target_combo.addItem("Editor (main.py)", "editor")
        form.addRow("Target:", self._target_combo)

        self._output_dir_edit = QLineEdit()
        self._output_dir_edit.setText(str(ROOT / "build_output"))
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(scale(32))
        browse_btn.clicked.connect(self._browse_output_dir)
        h = QHBoxLayout()
        h.addWidget(self._output_dir_edit, 1)
        h.addWidget(browse_btn)
        form.addRow("Output Dir:", h)

        self._onefile_cb = QCheckBox("Single file (--onefile)")
        form.addRow("Mode:", self._onefile_cb)

        self._console_cb = QCheckBox("Show console window")
        self._console_cb.setChecked(True)
        form.addRow("Console:", self._console_cb)

        self._strip_cb = QCheckBox("Strip unused assets (scans scenes for references)")
        self._strip_cb.setChecked(True)
        form.addRow("Assets:", self._strip_cb)

        self._winrt_cb = QCheckBox("Include Windows Runtime DLLs (--include-windows-runtime-dlls)")
        self._winrt_cb.setChecked(True)
        self._winrt_cb.setToolTip("Disable to reduce distribution size. Only affects Windows builds.")
        form.addRow("WinRT:", self._winrt_cb)

        main.addWidget(settings_group)

        # Output
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(4, 4, 4, 4)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.hide()
        output_layout.addWidget(self._progress_bar)

        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setFont(QFont("Consolas", 9))
        self._log_output.setMinimumHeight(150)
        output_layout.addWidget(self._log_output)

        btn_layout = QHBoxLayout()
        self._build_btn = QPushButton("Build")
        self._build_btn.setMinimumHeight(32)
        self._build_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 4px 20px; }")
        btn_layout.addStretch()
        btn_layout.addWidget(self._build_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setMinimumHeight(32)
        self._cancel_btn.setEnabled(False)
        btn_layout.addWidget(self._cancel_btn)

        self._open_output_btn = QPushButton("Open Output Folder")
        self._open_output_btn.setEnabled(False)
        btn_layout.addWidget(self._open_output_btn)

        close_btn = QPushButton("Close")
        close_btn.setMinimumHeight(32)
        btn_layout.addWidget(close_btn)
        output_layout.addLayout(btn_layout)

        main.addWidget(output_group)

        close_btn.clicked.connect(self.close)
        self._build_btn.clicked.connect(self._start_build)
        self._cancel_btn.clicked.connect(self._cancel_build)
        self._open_output_btn.clicked.connect(self._open_output_folder)

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self._output_dir_edit.setText(path)

    def _append_log(self, text: str):
        self._log_output.moveCursor(QTextCursor.MoveOperation.End)
        self._log_output.insertPlainText(text)
        self._log_output.moveCursor(QTextCursor.MoveOperation.End)

    def _collect_args(self):
        args = [sys.executable, str(ROOT / "build_nuitka.py")]
        if self._target_combo.currentData() == "editor":
            args.append("--editor")
        args.append(f"--output-dir={self._output_dir_edit.text().strip()}")
        if not self._console_cb.isChecked():
            args.append("--no-console")
        if self._onefile_cb.isChecked():
            args.append("--onefile")
        if self._strip_cb.isChecked():
            args.append("--strip-unused")
        else:
            args.append("--no-strip-unused")
        if not self._winrt_cb.isChecked():
            args.append("--no-winrt")
        return args

    def _start_build(self):
        args = self._collect_args()

        self._log_output.clear()
        self._build_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._open_output_btn.setEnabled(False)
        self._progress_bar.show()
        self._cancelled = False

        self._append_log(f">>> {' '.join(args)}\n\n")

        self._build_thread = threading.Thread(
            target=self._run_build, args=(args,), daemon=True
        )
        self._build_thread.start()

    def _run_build(self, args):
        try:
            self._build_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(ROOT),
            )
            for line in self._build_process.stdout:
                if self._cancelled:
                    self._build_process.terminate()
                    break
                self._log_signal.emit(line)
            returncode = self._build_process.wait()
        except Exception as e:
            self._log_signal.emit(f"\nError: {e}\n")
            returncode = -1
        finally:
            self._build_process = None
            self._done_signal.emit(returncode)

    def _on_build_done(self, returncode):
        self._build_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress_bar.hide()

        if self._cancelled:
            self._append_log("\n=== BUILD CANCELLED ===\n")
            return

        if returncode == 0:
            self._append_log("\n=== BUILD SUCCEEDED ===\n")
            self._open_output_btn.setEnabled(True)
        else:
            self._append_log(f"\n=== BUILD FAILED (code {returncode}) ===\n")
            QMessageBox.warning(self, "Build Failed",
                                f"Build exited with code {returncode}.\nCheck the log for details.")

    def _cancel_build(self):
        if self._build_process:
            self._cancelled = True
            self._build_process.terminate()
            self._append_log("\nCancelling...\n")
        self._cancel_btn.setEnabled(False)

    def _open_output_folder(self):
        out_dir = self._output_dir_edit.text().strip()
        if os.path.isdir(out_dir):
            os.startfile(out_dir)

    def closeEvent(self, event):
        if self._build_process:
            reply = QMessageBox.question(self, "Build in Progress",
                                         "A build is running. Cancel and exit?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self._cancelled = True
                self._build_process.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def show_build_dialog(parent=None):
    dialog = BuildDialog(parent)
    dialog.exec()


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Zarin Engine Builder")
    app.setStyle("Fusion")
    dialog = BuildDialog()
    dialog.show()
    sys.exit(app.exec())
