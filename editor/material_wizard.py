# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import re
from typing import Optional
import qtawesome as qta
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QListWidget, QFileDialog, QLineEdit, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QMessageBox, QGroupBox, QWidget, QSizePolicy, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from core.config.editor_scale import scale, scale_xy
from core.assets.material import Material, _parse_shader_file, _project_root

_ALBEDO_KEYWORDS = ("albedo", "diffuse", "base_color", "basecolor", "color", "basecolor",
                     "base_map", "basemap", "tint", "bedo")
_NORMAL_KEYWORDS = ("normal", "nrm", "nrm_map", "normal_map", "norm", "ntex")
_METALLIC_KEYWORDS = ("metallic", "metal", "met", "metalness")
_ROUGHNESS_KEYWORDS = ("roughness", "rough", "rgh")
_SMOOTHNESS_KEYWORDS = ("smoothness", "smooth", "gloss", "glossiness", "specular")
_AO_KEYWORDS = ("ao", "ambient_occlusion", "occlusion", "occ", "lightmap")
_EMISSION_KEYWORDS = ("emission", "emissive", "emit", "glow", "self_illum")
_HEIGHT_KEYWORDS = ("height", "displacement", "disp", "depth", "parallax", "hgt")
_DETAIL_ALBEDO_KEYWORDS = ("detail_albedo", "detail_color", "detail_diffuse", "detail_base")
_DETAIL_NORMAL_KEYWORDS = ("detail_normal", "detail_nrm")
_CLEARCOAT_KEYWORDS = ("clearcoat", "clear_coat", "clear")
_SUBSURFACE_KEYWORDS = ("subsurface", "sss", "translucency", "backlight")
_THIN_FILM_KEYWORDS = ("thin_film", "thin film", "iridescence")

_CHANNEL_HINTS = {
    "metallic": "_Metallic",
    "roughness": "_Smoothness",
    "smoothness": "_Smoothness",
    "ao": "_OcclusionMap",
}

_PRIORITY_ORDER = [
    (_DETAIL_ALBEDO_KEYWORDS, "_DetailAlbedoMap"),
    (_DETAIL_NORMAL_KEYWORDS, "_DetailNormalMap"),
    (_ALBEDO_KEYWORDS, "_BaseMap"),
    (_NORMAL_KEYWORDS, "_NormalMap"),
    (_METALLIC_KEYWORDS, "_Metallic"),
    (_ROUGHNESS_KEYWORDS, "_Smoothness"),
    (_SMOOTHNESS_KEYWORDS, "_Smoothness"),
    (_AO_KEYWORDS, "_OcclusionMap"),
    (_EMISSION_KEYWORDS, "_EmissionMap"),
    (_HEIGHT_KEYWORDS, "_HeightMap"),
    (_CLEARCOAT_KEYWORDS, "_ClearCoat"),
    (_SUBSURFACE_KEYWORDS, "_SubsurfaceColor"),
    (_THIN_FILM_KEYWORDS, "_ThinFilmThickness"),
]


def _normalize_name(name: str) -> str:
    name = os.path.splitext(name)[0].lower()
    name = re.sub(r'[\s\-_.]+', '_', name)
    return name


def _guess_map_type(filename: str) -> Optional[str]:
    base = _normalize_name(filename)
    for keywords, prop in _PRIORITY_ORDER:
        for kw in keywords:
            if kw in base:
                return prop
    for hint_key, prop in _CHANNEL_HINTS.items():
        if hint_key in base:
            return prop
    return None


def _collect_shaders(project_root: str) -> list[tuple[str, str]]:
    shaders = []
    engine_root = _project_root()
    dirs_to_scan = []
    shaders_dir = os.path.join(engine_root, "core", "shaders")
    if os.path.isdir(shaders_dir):
        dirs_to_scan.append(shaders_dir)
    if project_root:
        for root, _, files in os.walk(project_root):
            for f in files:
                if f.endswith(".shader"):
                    full = os.path.normpath(os.path.join(root, f))
                    shaders.append((f, full))
            break
    for d in dirs_to_scan:
        for f in os.listdir(d):
            if f.endswith(".shader"):
                full = os.path.normpath(os.path.join(d, f))
                shaders.append((f, full))
    seen = set()
    unique = []
    for name, path in shaders:
        if path not in seen:
            seen.add(path)
            unique.append((name, path))
    unique.sort(key=lambda x: x[0].lower())
    return unique


def _extract_shader_name(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read(2048)
        m = re.search(r'Shader\s+"([^"]+)"', text)
        if m:
            return m.group(1)
    except Exception:
        pass
    return os.path.splitext(os.path.basename(path))[0]


class _TextureDropList(QListWidget):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSpacing(2)
        self.setMinimumHeight(120)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            paths = []
            for url in event.mimeData().urls():
                p = url.toLocalFile()
                if os.path.isfile(p):
                    paths.append(p)
            if paths:
                self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class MaterialWizardDialog(QDialog):
    materials_created = pyqtSignal(list)

    def __init__(self, project_root: str = "", parent=None, submesh_names: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Material Wizard")
        self.setMinimumSize(700, 620)
        self.resize(780, 680)
        self._project_root = project_root or os.getcwd()
        self._shader_data: list[tuple[str, str]] = _collect_shaders(self._project_root)
        self._assignments: dict[str, str] = {}
        self._submesh_names: list[str] = list(submesh_names) if submesh_names else []
        self._setup_ui()
        self._populate_shaders()
        self._refresh_preview()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        shader_group = QGroupBox("Shader")
        sg_layout = QVBoxLayout(shader_group)
        sg_layout.setContentsMargins(8, 12, 8, 8)
        sg_layout.setSpacing(4)
        shader_row = QHBoxLayout()
        shader_lbl = QLabel("Shader:")
        shader_lbl.setMinimumWidth(70)
        shader_row.addWidget(shader_lbl)
        self._shader_combo = QComboBox()
        self._shader_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._shader_combo.currentIndexChanged.connect(self._on_shader_changed)
        shader_row.addWidget(self._shader_combo, 1)
        sg_layout.addLayout(shader_row)
        layout.addWidget(shader_group)

        tex_group = QGroupBox("Textures")
        tg_layout = QVBoxLayout(tex_group)
        tg_layout.setContentsMargins(8, 12, 8, 8)
        tg_layout.setSpacing(4)

        btn_row = QHBoxLayout()
        add_btn = QPushButton(qta.icon("fa5s.plus", color="#9ccc65"), " Add Textures")
        add_btn.clicked.connect(self._on_add_textures)
        btn_row.addWidget(add_btn)
        auto_btn = QPushButton(qta.icon("fa5s.magic", color="#5a9cf5"), " Auto-Map")
        auto_btn.setToolTip("Automatically map textures based on filenames")
        auto_btn.clicked.connect(self._on_auto_map)
        btn_row.addWidget(auto_btn)
        clear_btn = QPushButton(qta.icon("fa5s.trash", color="#f44747"), " Clear All")
        clear_btn.clicked.connect(self._on_clear_textures)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        tg_layout.addLayout(btn_row)

        self._texture_table = QTableWidget()
        self._texture_table.setColumnCount(3)
        self._texture_table.setHorizontalHeaderLabels(["Texture", "Maps To", ""])
        self._texture_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._texture_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._texture_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._texture_table.setColumnWidth(2, 30)
        self._texture_table.verticalHeader().setVisible(False)
        self._texture_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._texture_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._texture_table.setMinimumHeight(180)
        tg_layout.addWidget(self._texture_table)

        drop_label = QLabel("Drag & drop texture files here")
        drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_label.setStyleSheet("color: #888; font-size: 11px; padding: 4px;")
        tg_layout.addWidget(drop_label)

        layout.addWidget(tex_group, 1)

        if self._submesh_names:
            sub_group = QGroupBox("Submeshes")
            sg2_layout = QVBoxLayout(sub_group)
            sg2_layout.setContentsMargins(8, 12, 8, 8)
            sg2_layout.setSpacing(2)
            for sn in self._submesh_names:
                row = QLabel(f"  {sn}")
                row.setStyleSheet("font-size: 11px; color: #aaa;")
                sg2_layout.addWidget(row)
            layout.addWidget(sub_group)

        output_group = QGroupBox("Output")
        og_layout = QVBoxLayout(output_group)
        og_layout.setContentsMargins(8, 12, 8, 8)
        og_layout.setSpacing(4)

        name_row = QHBoxLayout()
        name_lbl = QLabel("Name Pattern:")
        name_lbl.setMinimumWidth(100)
        name_row.addWidget(name_lbl)
        self._name_edit = QLineEdit("{material_name}")
        self._name_edit.setPlaceholderText("{material_name}")
        self._name_edit.setToolTip(
            "Variables: {material_name}, {shader_name}, {texture_name}"
        )
        name_row.addWidget(self._name_edit, 1)
        og_layout.addLayout(name_row)

        dir_row = QHBoxLayout()
        dir_lbl = QLabel("Output Folder:")
        dir_lbl.setMinimumWidth(100)
        dir_row.addWidget(dir_lbl)
        self._dir_edit = QLineEdit("")
        self._dir_edit.setPlaceholderText("Materials/ (relative to project)")
        dir_row.addWidget(self._dir_edit, 1)
        dir_browse = QPushButton(qta.icon("fa5s.folder-open", color="#d4d4d4"), "")
        dir_browse.setFixedSize(*scale_xy(22, 22))
        dir_browse.clicked.connect(self._on_browse_output_dir)
        dir_row.addWidget(dir_browse)
        og_layout.addLayout(dir_row)

        create_per_mesh_cb = QCheckBox("Create one material per submesh")
        create_per_mesh_cb.setChecked(True)
        create_per_mesh_cb.setToolTip(
            "When enabled, creates a separate material for each texture group.\n"
            "When disabled, creates a single material with all textures combined."
        )
        self._per_mesh_cb = create_per_mesh_cb
        og_layout.addWidget(create_per_mesh_cb)

        layout.addWidget(output_group)

        preview_group = QGroupBox("Preview")
        pg_layout = QVBoxLayout(preview_group)
        pg_layout.setContentsMargins(8, 12, 8, 8)
        self._preview_label = QLabel()
        self._preview_label.setWordWrap(True)
        self._preview_label.setStyleSheet("font-size: 11px; color: #aaa;")
        self._preview_label.setMinimumHeight(60)
        pg_layout.addWidget(self._preview_label)
        layout.addWidget(preview_group)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        bottom_row.addWidget(cancel_btn)
        create_btn = QPushButton(qta.icon("fa5s.check", color="#9ccc65"), " Create Materials")
        create_btn.setStyleSheet(
            "QPushButton { background: #2d5a2d; border: 1px solid #4ec9b0; "
            "border-radius: 4px; padding: 6px 16px; font-weight: bold; color: #4ec9b0; }"
            "QPushButton:hover { background: #3a7a3a; }"
        )
        create_btn.clicked.connect(self._on_create)
        bottom_row.addWidget(create_btn)
        layout.addLayout(bottom_row)

        self._drop_list = _TextureDropList()
        self._drop_list.files_dropped.connect(self._on_files_dropped)
        self._drop_list.hide()

    def _populate_shaders(self):
        self._shader_combo.clear()
        self._shader_combo.addItem("-- Select Shader --", "")
        for name, path in self._shader_data:
            display = _extract_shader_name(path)
            self._shader_combo.addItem(f"{display}  ({name})", path)
        if len(self._shader_data) == 1:
            self._shader_combo.setCurrentIndex(1)

    def _on_shader_changed(self, index: int):
        if index <= 0:
            return
        self._on_auto_map()

    def _get_current_shader_path(self) -> str:
        idx = self._shader_combo.currentIndex()
        if idx <= 0:
            return ""
        return self._shader_combo.currentData()

    def _get_texture_files(self) -> list[str]:
        paths = []
        for row in range(self._texture_table.rowCount()):
            item = self._texture_table.item(row, 0)
            if item:
                paths.append(item.data(Qt.ItemDataRole.UserRole) or item.text())
        return paths

    def _on_add_textures(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Textures", "",
            "Images (*.png *.jpg *.jpeg *.tga *.bmp *.dds *.hdr *.exr);;All Files (*)"
        )
        if files:
            self._add_texture_files(files)

    def _on_files_dropped(self, paths: list[str]):
        self._add_texture_files(paths)

    def _add_texture_files(self, files: list[str]):
        existing = set(self._get_texture_files())
        for f in files:
            if f in existing:
                continue
            existing.add(f)
            self._insert_texture_row(f)
        self._on_auto_map()
        self._refresh_preview()

    def _insert_texture_row(self, filepath: str):
        row = self._texture_table.rowCount()
        self._texture_table.insertRow(row)

        name_item = QTableWidgetItem(os.path.basename(filepath))
        name_item.setData(Qt.ItemDataRole.UserRole, filepath)
        name_item.setToolTip(filepath)
        self._texture_table.setItem(row, 0, name_item)

        combo = QComboBox()
        combo.addItem("Auto-detect", "")
        shader_path = self._get_current_shader_path()
        shader_props = self._get_texture_properties(shader_path)
        for prop_name, display in shader_props:
            combo.addItem(display, prop_name)
        combo.addItem("Custom...", "custom")
        combo.currentIndexChanged.connect(lambda _, r=row: self._on_map_changed(r))
        self._texture_table.setCellWidget(row, 1, combo)

        remove_btn = QPushButton(qta.icon("fa5s.times", color="#f44747"), "")
        remove_btn.setFixedSize(*scale_xy(22, 22))
        remove_btn.clicked.connect(lambda _, r=row: self._remove_texture_row(r))
        self._texture_table.setCellWidget(row, 2, remove_btn)

    def _remove_texture_row(self, row: int):
        if 0 <= row < self._texture_table.rowCount():
            self._texture_table.removeRow(row)
            self._refresh_preview()

    def _on_map_changed(self, row: int):
        self._refresh_preview()

    def _get_texture_properties(self, shader_path: str) -> list[tuple[str, str]]:
        if not shader_path:
            return []
        result = _parse_shader_file(shader_path)
        if not result:
            return []
        props = result[0]
        tex_props = [(p.name, p.display_name) for p in props if p.prop_type in ("2D", "cube")]
        return tex_props

    def _on_auto_map(self):
        shader_path = self._get_current_shader_path()
        shader_props = self._get_texture_properties(shader_path)
        prop_names = [p[0] for p in shader_props]

        used = set()
        for row in range(self._texture_table.rowCount()):
            name_item = self._texture_table.item(row, 0)
            combo = self._texture_table.cellWidget(row, 1)
            if not name_item or not isinstance(combo, QComboBox):
                continue
            filepath = name_item.data(Qt.ItemDataRole.UserRole) or name_item.text()
            filename = os.path.basename(filepath)
            guessed = _guess_map_type(filename)

            if guessed and guessed in prop_names and guessed not in used:
                for i in range(combo.count()):
                    if combo.itemData(i) == guessed:
                        combo.setCurrentIndex(i)
                        used.add(guessed)
                        break
            else:
                combo.setCurrentIndex(0)

        self._refresh_preview()

    def _on_clear_textures(self):
        self._texture_table.setRowCount(0)
        self._assignments.clear()
        self._refresh_preview()

    def _on_browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self._project_root
        )
        if d:
            try:
                rel = os.path.relpath(d, self._project_root)
            except ValueError:
                rel = d
            self._dir_edit.setText(rel.replace("\\", "/"))

    def _build_assignment_map(self) -> dict[str, str]:
        assignments = {}
        for row in range(self._texture_table.rowCount()):
            name_item = self._texture_table.item(row, 0)
            combo = self._texture_table.cellWidget(row, 1)
            if not name_item or not isinstance(combo, QComboBox):
                continue
            filepath = name_item.data(Qt.ItemDataRole.UserRole) or name_item.text()
            prop = combo.currentData()
            if prop and prop != "custom" and prop != "":
                assignments[prop] = filepath
            elif prop == "" or prop == "Auto-detect":
                filename = os.path.basename(filepath)
                guessed = _guess_map_type(filename)
                if guessed:
                    assignments[guessed] = filepath
        return assignments

    def _refresh_preview(self):
        assignments = self._build_assignment_map()
        shader_path = self._get_current_shader_path()
        shader_name = _extract_shader_name(shader_path) if shader_path else "None"
        output_dir = self._dir_edit.text().strip() or "Materials/"
        name_pattern = self._name_edit.text().strip() or "{material_name}"

        lines = [f"Shader: {shader_name}"]
        lines.append(f"Output: {output_dir}/")
        lines.append(f"Name: {name_pattern}")
        if self._submesh_names:
            lines.append(f"Submeshes: {', '.join(self._submesh_names)}")
        lines.append("")
        if assignments:
            lines.append("Texture Assignments:")
            for prop, path in sorted(assignments.items()):
                fname = os.path.basename(path)
                lines.append(f"  {prop} <- {fname}")
        else:
            lines.append("No textures assigned.")
        self._preview_label.setText("\n".join(lines))

    def _on_create(self):
        shader_path = self._get_current_shader_path()
        if not shader_path:
            QMessageBox.warning(self, "Material Wizard", "Please select a shader.")
            return

        assignments = self._build_assignment_map()
        if not assignments:
            QMessageBox.warning(self, "Material Wizard", "Please add and map at least one texture.")
            return

        output_dir = self._dir_edit.text().strip() or "Materials"
        if not os.path.isabs(output_dir):
            output_dir = os.path.normpath(os.path.join(self._project_root, output_dir))
        os.makedirs(output_dir, exist_ok=True)

        name_pattern = self._name_edit.text().strip() or "{material_name}"
        shader_name = _extract_shader_name(shader_path).split("/")[-1].replace(" ", "_")

        created_paths = []

        if self._per_mesh_cb.isChecked():
            if self._submesh_names:
                groups = self._group_textures_by_pattern()
                group_map = {}
                for gname, gprops in groups:
                    group_map[gname.lower()] = (gname, gprops)
                matched_groups = {}
                unmatched_textures = {}
                used_groups = set()
                for sn in self._submesh_names:
                    sn_lower = sn.lower()
                    if sn_lower in group_map:
                        matched_groups[sn] = group_map[sn_lower]
                        used_groups.add(sn_lower)
                    else:
                        unmatched_textures[sn] = dict(assignments)
                for gname, gprops in groups:
                    if gname.lower() not in used_groups:
                        for sn in self._submesh_names:
                            if sn not in matched_groups and sn not in unmatched_textures:
                                matched_groups[sn] = (gname, gprops)
                                break
                for sn in self._submesh_names:
                    if sn in matched_groups:
                        _, group_props = matched_groups[sn]
                    elif sn in unmatched_textures:
                        group_props = unmatched_textures[sn]
                    else:
                        group_props = dict(assignments)
                    mat_name = name_pattern.replace("{material_name}", sn)
                    mat_name = mat_name.replace("{shader_name}", shader_name)
                    mat_name = re.sub(r'[<>:"/\\|?*]', '_', mat_name)
                    if not mat_name.endswith(".mat"):
                        mat_name += ".mat"
                    mat_path = os.path.join(output_dir, mat_name)
                    mat = self._create_material(sn, shader_path, group_props)
                    mat.save(mat_path, self._project_root)
                    created_paths.append(mat_path)
            else:
                groups = self._group_textures_by_pattern()
                if not groups:
                    groups = [("Material", assignments)]
                for group_name, group_props in groups:
                    mat_name = name_pattern.replace("{material_name}", group_name)
                    mat_name = mat_name.replace("{shader_name}", shader_name)
                    mat_name = re.sub(r'[<>:"/\\|?*]', '_', mat_name)
                    if not mat_name.endswith(".mat"):
                        mat_name += ".mat"
                    mat_path = os.path.join(output_dir, mat_name)
                    mat = self._create_material(
                        os.path.splitext(os.path.basename(mat_name))[0],
                        shader_path, group_props
                    )
                    mat.save(mat_path, self._project_root)
                    created_paths.append(mat_path)
        else:
            all_props = dict(assignments)
            mat_name = name_pattern.replace("{material_name}", "Material")
            mat_name = mat_name.replace("{shader_name}", shader_name)
            mat_name = re.sub(r'[<>:"/\\|?*]', '_', mat_name)
            if not mat_name.endswith(".mat"):
                mat_name += ".mat"
            mat_path = os.path.join(output_dir, mat_name)
            mat = self._create_material(
                os.path.splitext(os.path.basename(mat_name))[0],
                shader_path, all_props
            )
            mat.save(mat_path, self._project_root)
            created_paths.append(mat_path)

        self.materials_created.emit(created_paths)
        self.accept()

    def _group_textures_by_pattern(self) -> list[tuple[str, dict[str, str]]]:
        all_files = self._get_texture_files()
        if not all_files:
            return []

        groups: dict[str, dict[str, str]] = {}
        for filepath in all_files:
            filename = os.path.basename(filepath)
            base = _normalize_name(filename)
            group_name = self._extract_group_name(base)
            prop = _guess_map_type(filename)
            if not prop:
                shader_path = self._get_current_shader_path()
                props = self._get_texture_properties(shader_path)
                if props:
                    prop = props[0][0]
                else:
                    prop = "_BaseMap"
            if group_name not in groups:
                groups[group_name] = {}
            groups[group_name][prop] = filepath

        result = [(k, v) for k, v in groups.items()]
        result.sort(key=lambda x: x[0])
        return result

    def _extract_group_name(self, normalized_base: str) -> str:
        suffixes = []
        suffixes.extend(_ALBEDO_KEYWORDS)
        suffixes.extend(_NORMAL_KEYWORDS)
        suffixes.extend(_METALLIC_KEYWORDS)
        suffixes.extend(_ROUGHNESS_KEYWORDS)
        suffixes.extend(_SMOOTHNESS_KEYWORDS)
        suffixes.extend(_AO_KEYWORDS)
        suffixes.extend(_EMISSION_KEYWORDS)
        suffixes.extend(_HEIGHT_KEYWORDS)
        suffixes.extend(_DETAIL_ALBEDO_KEYWORDS)
        suffixes.extend(_DETAIL_NORMAL_KEYWORDS)
        suffixes.extend(_CLEARCOAT_KEYWORDS)
        suffixes.extend(_SUBSURFACE_KEYWORDS)
        suffixes.extend(_THIN_FILM_KEYWORDS)
        suffixes.sort(key=len, reverse=True)

        name = normalized_base
        for suffix in suffixes:
            for sep in ("_", "-", " "):
                token = sep + suffix
                idx = name.rfind(token)
                if idx >= 0:
                    candidate = name[:idx].rstrip("_- ")
                    if candidate:
                        return candidate.replace("_", " ").replace("-", " ").title()
                    break

        name = re.sub(r'_?(color|albedo|diffuse|normal|nrm|metallic|metal|roughness|rough|smoothness|smooth|ao|ambient|emission|emissive|height|displacement|map|tex|texture)\b.*$', '', name)
        name = name.rstrip("_- ")
        if name:
            return name.replace("_", " ").replace("-", " ").title()
        return "Material"

    def _create_material(self, name: str, shader_path: str, texture_map: dict[str, str]) -> Material:
        mat = Material(name)
        mat.shader_path = shader_path
        mat.load_shader_properties(shader_path, self._project_root)

        for prop in mat._shader_properties:
            if prop.name not in mat.properties:
                mat.properties[prop.name] = prop.default_value

        for prop_name, tex_path in texture_map.items():
            mat.properties[prop_name] = tex_path

        return mat

    def get_created_paths(self) -> list[str]:
        return []
