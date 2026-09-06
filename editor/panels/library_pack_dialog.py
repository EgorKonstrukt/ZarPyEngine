# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from tools.build_zplugin import (
    ARCH_CHOICES,
    create_library_pack,
    current_architecture,
    list_installed_distributions,
)

TARGET_PYTHONS = ["current", "3.10", "3.11", "3.12", "3.13", "3.14", "3.15"]
PIP_PLATFORMS = {"win_x86_64": "win_amd64", "win_arm64": "win_arm64",
                 "linux_x86_64": "manylinux2014_x86_64", "linux_aarch64": "manylinux2014_aarch64",
                 "macos_x86_64": "macosx_10_9_x86_64", "macos_arm64": "macosx_11_0_arm64"}


class LibraryPackDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Library Plugin Pack")
        self.setMinimumSize(560, 640)
        self._result = None
        self._dists = []
        self._setup_ui()
        self._reload_dists()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addWidget(QLabel("1. Pick libraries to vendor into libs/:"))
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self._source = QComboBox()
        self._source.addItems(["Installed packages", "Download wheels for target"])
        self._source.currentTextChanged.connect(self._update_source_note)
        src_row.addWidget(self._source)
        src_row.addStretch(1)
        layout.addLayout(src_row)
        self._source_note = QLabel("")
        self._source_note.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._source_note)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search installed distributions...")
        self._search.textChanged.connect(self._apply_filter)
        layout.addWidget(self._search)
        self._list = QListWidget()
        self._list.setMinimumHeight(160)
        self._list.itemChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list)

        tgt_row = QHBoxLayout()
        tgt_row.addWidget(QLabel("Download for Python:"))
        self._tpy = QComboBox()
        self._tpy.addItems(TARGET_PYTHONS)
        self._tpy.setCurrentText("current")
        self._tpy.currentTextChanged.connect(self._sync_abi)
        tgt_row.addWidget(self._tpy)
        tgt_row.addWidget(QLabel("ABI:"))
        self._tabi = QLineEdit("")
        self._tabi.setPlaceholderText("auto")
        tgt_row.addWidget(self._tabi)
        tgt_row.addStretch(1)
        layout.addLayout(tgt_row)
        self._update_source_note()

        meta_row = QHBoxLayout()
        meta_row.addWidget(QLabel("Pack name:"))
        self._name = QLineEdit("mylib_pack")
        meta_row.addWidget(self._name)
        meta_row.addWidget(QLabel("Version:"))
        self._version = QLineEdit("1.0.0")
        meta_row.addWidget(self._version)
        layout.addLayout(meta_row)

        desc_row = QHBoxLayout()
        desc_row.addWidget(QLabel("Description:"))
        self._desc = QLineEdit("")
        desc_row.addWidget(self._desc)
        layout.addLayout(desc_row)

        layout.addWidget(QLabel("2. Target platform:"))
        plat_row = QHBoxLayout()
        plat_row.addWidget(QLabel("Architecture:"))
        self._arch = QComboBox()
        self._arch.addItems(ARCH_CHOICES)
        plat_row.addWidget(self._arch)
        plat_row.addWidget(QLabel("Python:"))
        self._pyreq = QLineEdit(">=3.10")
        plat_row.addWidget(self._pyreq)
        plat_row.addWidget(QLabel("Engine API:"))
        self._api = QLineEdit(">=1")
        plat_row.addWidget(self._api)
        layout.addLayout(plat_row)
        self._arch_note = QLabel("")
        self._arch_note.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._arch_note)
        self._arch.currentTextChanged.connect(self._update_arch_note)
        self._update_arch_note()

        layout.addWidget(QLabel("3. Signing (optional):"))
        sign_row = QHBoxLayout()
        self._sign_key = QLineEdit()
        self._sign_key.setPlaceholderText("ed25519 private key (hex), empty = unsigned")
        sign_row.addWidget(self._sign_key)
        gen_btn = QPushButton("Generate")
        gen_btn.clicked.connect(self._gen_key)
        sign_row.addWidget(gen_btn)
        layout.addLayout(sign_row)

        layout.addWidget(QLabel("4. Output:"))
        out_row = QHBoxLayout()
        self._outdir = QLineEdit("dist")
        out_row.addWidget(self._outdir)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_outdir)
        out_row.addWidget(browse_btn)
        self._install_cb = QCheckBox("Install into plugins/ after build")
        self._install_cb.setChecked(True)
        out_row.addWidget(self._install_cb)
        layout.addLayout(out_row)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(120)
        layout.addWidget(self._log)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        build_btn = QPushButton("Build Pack")
        build_btn.setDefault(True)
        build_btn.clicked.connect(self._build)
        buttons.addButton(build_btn, buttons.ButtonRole.ActionRole)
        layout.addWidget(buttons)

    def _reload_dists(self):
        try:
            self._dists = list_installed_distributions()
        except Exception as e:
            self._log.appendPlainText(f"Could not list distributions: {e}")
            self._dists = []
        self._apply_filter()

    def _apply_filter(self):
        q = self._search.text().strip().lower()
        self._list.blockSignals(True)
        self._list.clear()
        for d in self._dists:
            if q and q not in d["name"].lower():
                continue
            label = f"{d['name']} {d['version']}  ({', '.join(d['modules'][:4])})"
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, d)
            self._list.addItem(item)
        self._list.blockSignals(False)

    def _selected_modules(self):
        mods = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                d = item.data(Qt.ItemDataRole.UserRole)
                mods.extend(d.get("modules", []))
        return sorted(set(mods))

    def _on_selection_changed(self):
        mods = self._selected_modules()
        if len(mods) == 1 and self._name.text().strip() in ("mylib_pack", ""):
            self._name.setText(f"{mods[0]}_pack")
        try:
            if mods:
                from importlib import metadata as _md
                self._version.setText(_md.version(mods[0]))
        except Exception:
            pass

    def _update_source_note(self):
        if self._source.currentText().startswith("Download"):
            self._source_note.setText("Checked distributions are pinned (name==version) and downloaded "
                                      "from PyPI for the target triple below, with all dependencies.")
        else:
            self._source_note.setText("Checked distributions are copied from this interpreter as-is.")

    def _sync_abi(self):
        txt = self._tpy.currentText()
        if txt == "current":
            import sys
            self._tabi.setPlaceholderText(f"cp{sys.version_info[0]}{sys.version_info[1]}")
        else:
            self._tabi.setPlaceholderText("cp" + txt.replace(".", ""))

    def _selected_dists(self):
        out = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def _target_triple(self):
        import sys
        py = self._tpy.currentText()
        if py == "current":
            py = f"{sys.version_info[0]}.{sys.version_info[1]}"
        arch = self._arch.currentText()
        if arch == "current":
            arch = current_architecture()
        triple = {"python_version": py,
                  "abi": self._tabi.text().strip() or ("cp" + py.replace(".", ""))}
        plat = PIP_PLATFORMS.get(arch, "")
        if plat:
            triple["platform"] = plat
        return triple

    def _download_requirements(self):
        reqs = []
        for d in self._selected_dists():
            ver = d.get("version", "")
            if ver and ver != "unknown":
                reqs.append(f"{d['name']}=={ver}")
            else:
                reqs.append(d["name"])
        return reqs

    def _update_arch_note(self):
        cur = current_architecture()
        if self._arch.currentText() in ("any", "current"):
            self._arch_note.setText(f"Records '{cur}'. 'any' fits every machine, a fixed tag blocks other platforms.")
        else:
            self._arch_note.setText(f"Current machine is '{cur}'. Fixed tags refuse to load elsewhere.")

    def _gen_key(self):
        try:
            from core.foundation.ed25519 import generate_hex
            priv, pub = generate_hex()
            self._sign_key.setText(priv)
            self._log.appendPlainText(f"New keypair. PUBLIC (share for verification):\n{pub}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Key generation failed:\n{e}")

    def _browse_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Output Directory", self._outdir.text().strip() or ".")
        if d:
            self._outdir.setText(d)

    def _build(self):
        mods = self._selected_modules()
        if not mods:
            QMessageBox.warning(self, "No Libraries", "Check at least one distribution first.")
            return
        name = self._name.text().strip()
        version = self._version.text().strip()
        if not name or not version:
            QMessageBox.warning(self, "Missing Data", "Pack name and version are required.")
            return
        outdir = self._outdir.text().strip() or "dist"
        install = os.path.join(os.getcwd(), "plugins") if self._install_cb.isChecked() else None
        use_download = self._source.currentText().startswith("Download")
        reqs = self._download_requirements() if use_download else []
        triple = self._target_triple() if use_download else None
        what = ", ".join(reqs) if use_download else ", ".join(mods)
        self._log.appendPlainText(f"Building '{name}' v{version} with: {what} ...")
        try:
            result = create_library_pack(
                name, version, [] if use_download else mods, output_dir=outdir,
                arch=self._arch.currentText(),
                python_requires=self._pyreq.text().strip(),
                engine_api=self._api.text().strip() or ">=1",
                description=self._desc.text().strip(),
                sign_key=self._sign_key.text().strip(),
                install_dir=install, download=use_download,
                download_requires=reqs or None, target=triple)
        except Exception as e:
            QMessageBox.critical(self, "Build Failed", f"Pack build failed:\n{e}")
            return
        if not result:
            QMessageBox.critical(self, "Build Failed", "Pack build failed. Check the log.")
            return
        self._result = result
        self._log.appendPlainText(f"OK: {result}")
        QMessageBox.information(self, "Build Complete", f"Library pack built:\n{result}")

    @staticmethod
    def show(parent=None):
        dlg = LibraryPackDialog(parent)
        dlg.exec()
        return dlg._result


def show_library_pack_dialog(parent=None):
    return LibraryPackDialog.show(parent)