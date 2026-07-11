# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json, os
import numpy as np
from typing import Optional, TYPE_CHECKING
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QLabel, QLineEdit, QPushButton, QCheckBox, QDoubleSpinBox,
    QSpinBox, QComboBox, QGroupBox, QFrame, QMenu, QDialog, QTextEdit,
    QPlainTextEdit)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor
from core.config.editor_scale import scale, scale_xy
from core.foundation.logger import Logger
from core.foundation.commands import SetComponentCommand, CompoundCommand, get_history
from core.config.config import get_project_config
from core.physics.collision_layers import MAX_LAYERS, DEFAULT_LAYER_NAMES
from core.components.animation.animator_controller import (
    AnimatorController, AnimatorState, AnimatorTransition,
    AnimatorCondition, AnimatorConditionMode,
)
from editor.inspector.constants import (_FUSION_ACCENT_GREEN, _FUSION_ACCENT_RED, _FUSION_ACCENT_ORANGE,
    _FUSION_CARD_RADIUS, _FUSION_INPUT_RADIUS, _accent)
from editor.inspector.helpers import make_resource_picker, make_asset_picker
from editor.inspector.component_widget import ComponentWidget
from editor.inspector.component_picker import ComponentPickerDialog

if TYPE_CHECKING:
    from core.ecs.ecs import Entity, Scene
    from core.engine.engine import Engine


class InspectorPanel(QDockWidget):
    _clipboard = None
    open_prefab_editor = pyqtSignal(str)

    def __init__(self, engine: Engine, parent=None):
        super().__init__("Inspector", parent)
        self._engine = engine
        self._entity: Optional[Entity] = None
        self._selected_entities: list = []
        self._comp_widgets: list[ComponentWidget] = []
        self._asset_widgets: list[QWidget] = []
        self._updating: bool = False
        self._locked: bool = False
        self._asset_path: Optional[str] = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_transform)
        self._refresh_timer.start(300)
        self._animator_mode = False
        self._animator_state: Optional[AnimatorState] = None
        self._animator_transition: Optional[AnimatorTransition] = None
        self._animator_controller: Optional[AnimatorController] = None
        self._saved_entity: Optional[Entity] = None
        self._saved_entities: list[Entity] = []
        self._setup_ui()

    def load_config(self, config) -> None:
        refresh_interval = config.get("inspector.refresh_interval", 300)
        self._refresh_timer.setInterval(refresh_interval)

    def _setup_ui(self):
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self._header_widget = QWidget()
        header_layout = QHBoxLayout(self._header_widget)
        header_layout.setContentsMargins(6, 4, 6, 4)
        header_layout.setSpacing(4)
        self._lock_btn = QPushButton("\U0001F512")
        self._lock_btn.setFixedSize(*scale_xy(22, 22))
        self._lock_btn.setCheckable(True)
        self._lock_btn.setChecked(False)
        self._lock_btn.setToolTip("Lock Inspector")
        self._lock_btn.toggled.connect(self._on_lock_toggled)
        header_layout.addWidget(self._lock_btn)
        self._active_cb = QCheckBox()
        self._active_cb.toggled.connect(self._on_active_changed)
        self._active_cb.setStyleSheet(f"background: transparent;")
        header_layout.addWidget(self._active_cb)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Entity name")
        self._name_edit.setStyleSheet(f"""
            QLineEdit {{
                selection-background-color: {_accent()};
            }}
            QLineEdit:focus {{ border-color: {_accent()}; }}
        """)
        self._name_edit.textChanged.connect(self._on_name_changed)
        header_layout.addWidget(self._name_edit, 1)
        self._tag_edit = QLineEdit()
        self._tag_edit.setPlaceholderText("Tag")
        self._tag_edit.setFixedWidth(scale(80))
        self._tag_edit.setStyleSheet(f"QLineEdit:focus {{ border-color: {_accent()}; }}")
        self._tag_edit.textChanged.connect(self._on_tag_changed)
        header_layout.addWidget(self._tag_edit)
        layer_lbl = QLabel("Layer")
        header_layout.addWidget(layer_lbl)
        self._layer_sb = QSpinBox()
        self._layer_sb.setRange(0, 31)
        self._layer_sb.setFixedWidth(scale(46))
        self._layer_sb.valueChanged.connect(self._on_layer_changed)
        header_layout.addWidget(self._layer_sb)
        outer_layout.addWidget(self._header_widget)
        self._prefab_bar_widget = QWidget()
        prefab_bar_layout = QHBoxLayout(self._prefab_bar_widget)
        prefab_bar_layout.setContentsMargins(6, 2, 6, 2)
        self._prefab_label = QLabel()
        self._prefab_label.setStyleSheet(f"color: {_accent()}; font-weight: 600; font-size: 11px; background: transparent;")
        prefab_bar_layout.addWidget(self._prefab_label)
        self._override_label = QLabel()
        self._override_label.setStyleSheet(f"color: {_FUSION_ACCENT_ORANGE}; font-size: 10px; background: transparent;")
        prefab_bar_layout.addWidget(self._override_label)
        prefab_bar_layout.addStretch()
        _prefab_btn_style = f"""
            QPushButton {{
                padding: 2px 8px; font-size: 10px;
            }}
        """
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setFixedHeight(scale(22))
        self._apply_btn.setStyleSheet(_prefab_btn_style)
        self._apply_btn.clicked.connect(self._on_apply_prefab)
        prefab_bar_layout.addWidget(self._apply_btn)
        self._revert_btn = QPushButton("Revert")
        self._revert_btn.setFixedHeight(scale(22))
        self._revert_btn.setStyleSheet(_prefab_btn_style)
        self._revert_btn.clicked.connect(self._on_revert_prefab)
        prefab_bar_layout.addWidget(self._revert_btn)
        self._select_prefab_btn = QPushButton("Select")
        self._select_prefab_btn.setFixedHeight(scale(22))
        self._select_prefab_btn.setStyleSheet(_prefab_btn_style)
        self._select_prefab_btn.clicked.connect(self._on_select_prefab_asset)
        prefab_bar_layout.addWidget(self._select_prefab_btn)
        self._open_prefab_btn = QPushButton("Open Prefab")
        self._open_prefab_btn.setFixedHeight(scale(22))
        self._open_prefab_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 2px 10px; font-size: 10px; font-weight: 700;
                color: {_accent()};
                border: 1px solid {_accent()};
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background: {_accent()};
                color: #fff;
            }}
        """)
        self._open_prefab_btn.clicked.connect(self._on_open_prefab_editor)
        prefab_bar_layout.addWidget(self._open_prefab_btn)
        outer_layout.addWidget(self._prefab_bar_widget)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        outer_layout.addWidget(sep)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._content_widget = QWidget()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(4, 4, 4, 4)
        self._content_layout.setSpacing(4)
        self._content_layout.addStretch()
        self._scroll.setWidget(self._content_widget)
        outer_layout.addWidget(self._scroll, 1)
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(6, 4, 6, 6)
        self._add_comp_btn = QPushButton("+ Add Component")
        self._add_comp_btn.setFixedHeight(scale(24))
        self._add_comp_btn.setStyleSheet("QPushButton { background: #2e7d32; color: #fff; }")
        self._add_comp_btn.clicked.connect(self._show_add_component_menu)
        bottom_layout.addWidget(self._add_comp_btn)
        outer_layout.addWidget(bottom)
        self._prefab_bar_widget.setVisible(False)
        self._header_widget.setVisible(False)
        self._add_comp_btn.setVisible(False)
        self.setWidget(outer)

    def _on_lock_toggled(self, checked: bool):
        self._locked = checked
        self._lock_btn.setToolTip("Unlock Inspector" if checked else "Lock Inspector")

    def set_entity(self, entity: Optional[Entity]):
        if self._locked:
            return
        self._animator_mode = False
        self._animator_state = None
        self._animator_transition = None
        self._animator_controller = None
        self._entity = entity
        self._selected_entities = [entity] if entity else []
        self._asset_path = None
        self._rebuild()

    def set_selected_entities(self, entities: list):
        if self._locked:
            return
        self._animator_mode = False
        self._animator_state = None
        self._animator_transition = None
        self._animator_controller = None
        self._selected_entities = list(entities)
        self._entity = entities[0] if entities else None
        self._asset_path = None
        self._rebuild()

    def show_import_settings(self, path: str):
        self._entity = None
        self._asset_path = path
        self._rebuild()

    def show_animator_state(self, state: AnimatorState, controller: AnimatorController):
        self._saved_entity = self._entity
        self._saved_entities = list(self._selected_entities)
        self._entity = None
        self._asset_path = None
        self._animator_mode = True
        self._animator_state = state
        self._animator_transition = None
        self._animator_controller = controller
        self._rebuild()

    def show_animator_transition(self, trans: AnimatorTransition, controller: AnimatorController):
        self._saved_entity = self._entity
        self._saved_entities = list(self._selected_entities)
        self._entity = None
        self._asset_path = None
        self._animator_mode = True
        self._animator_transition = trans
        self._animator_state = None
        self._animator_controller = controller
        self._rebuild()

    def clear_animator_mode(self):
        if self._animator_mode:
            self._animator_mode = False
            self._animator_state = None
            self._animator_transition = None
            self._entity = self._saved_entity
            self._selected_entities = self._saved_entities
            self._animator_controller = None
            self._rebuild()

    def _add_asset_widget(self, w: QWidget):
        self._asset_widgets.append(w)
        self._content_layout.addWidget(w)

    def _build_import_settings(self):
        self._updating = True
        name = os.path.basename(self._asset_path)
        ext = os.path.splitext(name)[1].lower()
        title = QLabel(f"<b>{name}</b>")
        self._add_asset_widget(title)
        info = QLabel(f"Path: {self._asset_path}")
        info.setWordWrap(True)
        self._add_asset_widget(info)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        self._add_asset_widget(sep)
        if ext in (".obj", ".fbx", ".stl", ".usdz", ".gltf", ".glb"):
            self._build_mesh_import_settings()
        elif ext in (".png", ".jpg", ".jpeg", ".svg"):
            self._build_texture_import_settings()
        elif ext in (".wav", ".mp3", ".ogg"):
            self._build_audio_import_settings()
        elif ext == ".zpem" or ext == ".mat":
            self._build_material_editor()
        else:
            lbl = QLabel("No import settings for this file type.")
            self._add_asset_widget(lbl)
        self._updating = False

    def _build_labeled_field(self, label: str, widget: QWidget):
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 2, 0, 2)
        rl.setSpacing(4)
        lbl = QLabel(label)
        lbl.setFixedWidth(scale(120))
        rl.addWidget(lbl)
        rl.addWidget(widget)
        self._add_asset_widget(row)

    def _build_mesh_import_settings(self):
        from editor.model_preview import ModelPreviewWidget
        from core.assets.asset_importer import load_obj_async, load_mesh_async
        preview = ModelPreviewWidget()
        preview.setFixedHeight(200)
        self._add_asset_widget(preview)
        from PyQt6.QtCore import QTimer
        def _on_mesh_loaded(data):
            if data is not None and len(data.vertices) >= 3 and len(data.indices) >= 3:
                s = settings.get("scale", 1.0)
                verts = data.vertices.reshape(-1, 3).astype(np.float32) * s
                QTimer.singleShot(0, lambda: preview.set_mesh(verts, data.indices, normals=data.normals))
        ext = os.path.splitext(self._asset_path)[1].lower()
        if ext == ".obj":
            load_obj_async(self._asset_path, _on_mesh_loaded)
        else:
            load_mesh_async(self._asset_path, _on_mesh_loaded)
        from core.config.config import get_global_config
        cache_path = self._asset_path + ".import"
        settings = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    settings = json.load(f)
            except: pass
        scale_sb = QDoubleSpinBox()
        scale_sb.setRange(0.001, 1000.0)
        scale_sb.setSingleStep(0.1)
        scale_sb.setValue(settings.get("scale", 1.0))
        scale_sb.valueChanged.connect(lambda v: self._save_import_setting("scale", v))
        self._build_labeled_field("Scale Factor", scale_sb)
        pivot_cb = QCheckBox()
        pivot_cb.setChecked(settings.get("center_pivot", False))
        pivot_cb.toggled.connect(lambda v: self._save_import_setting("center_pivot", v))
        self._build_labeled_field("Center Pivot", pivot_cb)
        flip_uv_cb = QCheckBox()
        flip_uv_cb.setChecked(settings.get("flip_uvs", False))
        flip_uv_cb.toggled.connect(lambda v: self._save_import_setting("flip_uvs", v))
        self._build_labeled_field("Flip UVs", flip_uv_cb)
        smooth_sb = QDoubleSpinBox()
        smooth_sb.setRange(0.0, 180.0)
        smooth_sb.setSingleStep(1.0)
        smooth_sb.setDecimals(1)
        smooth_sb.setValue(settings.get("smooth_angle", 30.0))
        smooth_sb.valueChanged.connect(lambda v: self._save_import_setting("smooth_angle", v))
        self._build_labeled_field("Smooth Angle", smooth_sb)
        gen_nrm = QCheckBox()
        gen_nrm.setChecked(settings.get("gen_normals", True))
        gen_nrm.toggled.connect(lambda v: self._save_import_setting("gen_normals", v))
        self._build_labeled_field("Generate Normals", gen_nrm)
        gen_uv = QCheckBox()
        gen_uv.setChecked(settings.get("gen_uvs", True))
        gen_uv.toggled.connect(lambda v: self._save_import_setting("gen_uvs", v))
        self._build_labeled_field("Generate UVs", gen_uv)

    def _build_texture_import_settings(self):
        from core.assets.texture_import_settings import DEFAULT_SETTINGS
        cache_path = self._asset_path + ".import"
        settings = dict(DEFAULT_SETTINGS)
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    settings.update(json.load(f))
            except: pass
        type_cb = QComboBox()
        type_cb.addItems(["albedo", "normal", "metallic", "roughness", "ao", "emission", "sprite"])
        type_cb.setCurrentText(settings.get("type", "albedo"))
        type_cb.currentTextChanged.connect(lambda v: self._save_import_setting("type", v))
        self._build_labeled_field("Texture Type", type_cb)
        srgb = QCheckBox()
        srgb.setChecked(settings.get("srgb", True))
        srgb.toggled.connect(lambda v: self._save_import_setting("srgb", v))
        self._build_labeled_field("sRGB", srgb)
        filter_cb = QComboBox()
        filter_cb.addItems(["point", "bilinear", "trilinear"])
        filter_cb.setCurrentText(settings.get("filter_mode", "trilinear"))
        filter_cb.currentTextChanged.connect(lambda v: self._save_import_setting("filter_mode", v))
        self._build_labeled_field("Filter Mode", filter_cb)
        aniso_cb = QComboBox()
        aniso_cb.addItems(["1 (Off)", "2", "4", "8", "16"])
        aniso_val = settings.get("anisotropic", 1)
        aniso_cb.setCurrentIndex({1:0, 2:1, 4:2, 8:3, 16:4}.get(aniso_val, 0))
        def _on_aniso(v):
            mapping = [1, 2, 4, 8, 16]
            self._save_import_setting("anisotropic", mapping[aniso_cb.currentIndex()])
        aniso_cb.currentIndexChanged.connect(_on_aniso)
        self._build_labeled_field("Aniso Level", aniso_cb)
        max_sb = QSpinBox()
        max_sb.setRange(32, 8192)
        max_sb.setSingleStep(2)
        max_sb.setValue(settings.get("max_size", 2048))
        max_sb.valueChanged.connect(lambda v: self._save_import_setting("max_size", v))
        self._build_labeled_field("Max Size", max_sb)
        wrap_cb = QComboBox()
        wrap_cb.addItems(["clamp", "repeat", "mirrored_repeat"])
        wrap_cb.setCurrentText(settings.get("wrap_mode", "clamp"))
        wrap_cb.currentTextChanged.connect(lambda v: self._save_import_setting("wrap_mode", v))
        self._build_labeled_field("Wrap Mode", wrap_cb)
        comp_cb = QComboBox()
        comp_cb.addItems(["none", "low", "normal", "high"])
        comp_cb.setCurrentText(settings.get("compression", "none"))
        comp_cb.currentTextChanged.connect(lambda v: self._save_import_setting("compression", v))
        self._build_labeled_field("Compression", comp_cb)

    def _build_audio_import_settings(self):
        cache_path = self._asset_path + ".import"
        settings = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    settings = json.load(f)
            except: pass
        qual_sb = QSpinBox()
        qual_sb.setRange(0, 100)
        qual_sb.setValue(settings.get("quality", 80))
        qual_sb.valueChanged.connect(lambda v: self._save_import_setting("quality", v))
        self._build_labeled_field("Quality", qual_sb)
        stream_cb = QCheckBox()
        stream_cb.setChecked(settings.get("stream", False))
        stream_cb.toggled.connect(lambda v: self._save_import_setting("stream", v))
        self._build_labeled_field("Stream", stream_cb)

    def _save_import_setting(self, key: str, value):
        if not self._asset_path: return
        cache_path = self._asset_path + ".import"
        settings = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    settings = json.load(f)
            except: pass
        settings[key] = value
        try:
            with open(cache_path, "w") as f:
                json.dump(settings, f, indent=2)
        except: pass

    def _build_material_editor(self):
        from core.assets.material import Material
        from editor.material_preview import MaterialPreviewWidget
        try:
            mat = Material.load(self._asset_path, self._engine.project_root if self._engine else "")
        except Exception:
            mat = None
        if mat is None:
            lbl = QLabel("Failed to load material.")
            lbl.setStyleSheet(f"color: {_FUSION_ACCENT_RED};")
            self._add_asset_widget(lbl)
            return
        props = mat.properties
        shader_props = mat._shader_properties
        preview = MaterialPreviewWidget()
        preview.setFixedHeight(200)
        preview_vals = {}
        ac = props.get("albedo_color") or props.get("_BaseColor")
        if ac:
            preview_vals["albedo"] = ac[:3] if len(ac) >= 3 else [1.0, 1.0, 1.0]
        m = props.get("metallic") if "metallic" in props else props.get("_Metallic")
        if m is not None:
            preview_vals["metallic"] = m
        s = props.get("smoothness") if "smoothness" in props else props.get("_Smoothness")
        if s is not None:
            preview_vals["smoothness"] = s
        ec = props.get("emission_color") or props.get("_EmissionColor")
        if ec:
            preview_vals["emission"] = ec
        ei = props.get("emission_intensity") if "emission_intensity" in props else props.get("_EmissionIntensity")
        if ei is not None:
            preview_vals["emit_intensity"] = ei
        for tex_key in ("albedo_texture", "_BaseMap", "_BaseTex"):
            if tex_key in props and props[tex_key]:
                preview_vals["albedo_tex"] = props[tex_key]
                break
        if preview_vals:
            preview.set_properties(**preview_vals)
        self._add_asset_widget(preview)
        def _save():
            mat.save(self._asset_path, self._engine.project_root)
        def _update_preview():
            pv = {}
            ac = props.get("albedo_color") or props.get("_BaseColor")
            if ac:
                pv["albedo"] = ac[:3] if len(ac) >= 3 else [1.0, 1.0, 1.0]
            m = props.get("metallic") if "metallic" in props else props.get("_Metallic")
            if m is not None:
                pv["metallic"] = m
            s = props.get("smoothness") if "smoothness" in props else props.get("_Smoothness")
            if s is not None:
                pv["smoothness"] = s
            ec = props.get("emission_color") or props.get("_EmissionColor")
            if ec:
                pv["emission"] = ec
            ei = props.get("emission_intensity") if "emission_intensity" in props else props.get("_EmissionIntensity")
            if ei is not None:
                pv["emit_intensity"] = ei
            for tex_key in ("albedo_texture", "_BaseMap", "_BaseTex"):
                if tex_key in props and props[tex_key]:
                    pv["albedo_tex"] = props[tex_key]
                    break
            if pv:
                preview.set_properties(**pv)
        shader_row = QWidget()
        shader_rl = QHBoxLayout(shader_row)
        shader_rl.setContentsMargins(0, 2, 0, 2)
        shader_lbl = QLabel("Shader")
        shader_lbl.setFixedWidth(scale(120))
        shader_rl.addWidget(shader_lbl)
        def _on_shader_pick(p):
            mat.shader_path = p
            mat.load_shader_properties(p, self._engine.project_root)
            props.clear()
            for sp in mat._shader_properties:
                props[sp.name] = sp.default_value
            _save()
            self._rebuild()
        shader_picker = make_resource_picker(mat.shader_path, "Shaders (*.shader *.vert *.frag)", _on_shader_pick)
        shader_rl.addWidget(shader_picker, 1)
        self._add_asset_widget(shader_row)
        if shader_props:
            tex_props = [p for p in shader_props if p.prop_type in ("2D", "cube")]
            non_tex_props = [p for p in shader_props if p.prop_type not in ("2D", "cube")]
            for sp in non_tex_props:
                self._add_shader_property_widget(sp, props, _save, _update_preview)
            if tex_props:
                sep = QLabel("<b>Textures</b>")
                self._add_asset_widget(sep)
                for sp in tex_props:
                    self._add_shader_property_widget(sp, props, _save, _update_preview)
        else:
            known_keys = {
                "albedo_color": {"label": "Albedo", "widget": "color"},
                "albedo_texture": {"label": "Albedo Map", "widget": "texture"},
                "metallic": {"label": "Metallic", "widget": "slider", "min": 0, "max": 1, "step": 0.01},
                "smoothness": {"label": "Smoothness", "widget": "slider", "min": 0, "max": 1, "step": 0.01},
                "emission_color": {"label": "Emission", "widget": "color"},
                "emission_intensity": {"label": "Emission Intensity", "widget": "slider", "min": 0, "max": 100, "step": 0.1},
                "normal_texture": {"label": "Normal Map", "widget": "texture"},
                "roughness_texture": {"label": "Roughness Map", "widget": "texture"},
            }
            tex_seen = False
            for key, cfg in known_keys.items():
                if cfg["widget"] == "texture" and not tex_seen:
                    sep = QLabel("<b>Textures</b>")
                    self._add_asset_widget(sep)
                    tex_seen = True
                self._add_fallback_widget(key, cfg, props, _save, _update_preview)
        self._add_asset_widget(QLabel(""))

    def _add_shader_property_widget(self, sp, props, _save, _update_preview):
        from editor.resource_picker import pick_resource
        from PyQt6.QtGui import QColor
        prop_type = sp.prop_type
        key = sp.name
        label = sp.display_name
        if prop_type in ("2D", "cube"):
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 2, 0, 2)
            lbl = QLabel(label)
            lbl.setFixedWidth(scale(120))
            rl.addWidget(lbl)
            def _on_pick(p):
                props[key] = p
                _save()
                _update_preview()
            picker = make_resource_picker(props.get(key, ""), "Images (*.png *.jpg *.jpeg)", _on_pick)
            rl.addWidget(picker, 1)
            self._add_asset_widget(row)
        elif prop_type == "Color":
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 2, 0, 2)
            lbl = QLabel(label)
            lbl.setFixedWidth(scale(120))
            rl.addWidget(lbl)
            from editor.color_picker import ColorLineEdit
            cl = ColorLineEdit(props.get(key, [1.0, 1.0, 1.0, 1.0]))
            def _on_color(_, _key=key):
                props[_key] = cl.get_color_rgba()
                _save()
                _update_preview()
            cl.colorChanged.connect(_on_color)
            rl.addWidget(cl, 1)
            self._add_asset_widget(row)
        elif prop_type == "Range":
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 2, 0, 2)
            lbl = QLabel(label)
            lbl.setFixedWidth(scale(120))
            rl.addWidget(lbl)
            sb = QDoubleSpinBox()
            sb.setRange(sp.range_min, sp.range_max)
            sb.setSingleStep((sp.range_max - sp.range_min) / 100.0)
            sb.setValue(props.get(key, 0.0))
            def _on_change(v, _key=key):
                props[_key] = v
                _save()
                _update_preview()
            sb.valueChanged.connect(_on_change)
            rl.addWidget(sb, 1)
            self._add_asset_widget(row)
        elif prop_type in ("Float", "Int"):
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 2, 0, 2)
            lbl = QLabel(label)
            lbl.setFixedWidth(scale(120))
            rl.addWidget(lbl)
            if prop_type == "Int":
                sb = QSpinBox()
                sb.setRange(-999999, 999999)
            else:
                sb = QDoubleSpinBox()
                sb.setRange(-999999.0, 999999.0)
                sb.setSingleStep(0.1)
            sb.setValue(props.get(key, 0))
            def _on_change(v, _key=key):
                props[_key] = v
                _save()
                _update_preview()
            sb.valueChanged.connect(_on_change)
            rl.addWidget(sb, 1)
            self._add_asset_widget(row)
        elif prop_type == "Gradient":
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 2, 0, 2)
            lbl = QLabel(label)
            lbl.setFixedWidth(scale(120))
            rl.addWidget(lbl)
            from editor.gradient_editor import GradientLineEdit
            gle = GradientLineEdit(props.get(key, None))
            def _on_gradient(_, _key=key):
                props[_key] = gle.get_stops()
                _save()
                _update_preview()
            gle.gradientChanged.connect(_on_gradient)
            rl.addWidget(gle, 1)
            self._add_asset_widget(row)

    def _add_fallback_widget(self, key, cfg, props, _save, _update_preview):
        from editor.resource_picker import pick_resource
        from PyQt6.QtGui import QColor
        label = cfg["label"]
        widget_type = cfg["widget"]
        if widget_type == "texture":
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 2, 0, 2)
            lbl = QLabel(label)
            lbl.setFixedWidth(scale(120))
            rl.addWidget(lbl)
            def _on_pick(p, _key=key):
                props[_key] = p
                _save()
                _update_preview()
            picker = make_resource_picker(props.get(key, ""), "Images (*.png *.jpg *.jpeg)", _on_pick)
            rl.addWidget(picker, 1)
            self._add_asset_widget(row)
        elif widget_type == "color":
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 2, 0, 2)
            lbl = QLabel(label)
            lbl.setFixedWidth(scale(120))
            rl.addWidget(lbl)
            from editor.color_picker import ColorLineEdit
            cl = ColorLineEdit(props.get(key, [1.0, 1.0, 1.0, 1.0]))
            def _on_color(_, _key=key):
                props[_key] = cl.get_color_rgba()
                _save()
                _update_preview()
            cl.colorChanged.connect(_on_color)
            rl.addWidget(cl, 1)
            self._add_asset_widget(row)
        elif widget_type == "slider":
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 2, 0, 2)
            lbl = QLabel(label)
            lbl.setFixedWidth(scale(120))
            rl.addWidget(lbl)
            sb = QDoubleSpinBox()
            sb.setRange(cfg.get("min", 0.0), cfg.get("max", 1.0))
            sb.setSingleStep(cfg.get("step", 0.01))
            sb.setValue(props.get(key, 0.0))
            def _on_change(v, _key=key):
                props[_key] = v
                _save()
                _update_preview()
            sb.valueChanged.connect(_on_change)
            rl.addWidget(sb, 1)
            self._add_asset_widget(row)
        elif widget_type == "gradient":
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 2, 0, 2)
            lbl = QLabel(label)
            lbl.setFixedWidth(scale(120))
            rl.addWidget(lbl)
            from editor.gradient_editor import GradientLineEdit
            gle = GradientLineEdit(props.get(key, None))
            def _on_gradient(_, _key=key):
                props[_key] = gle.get_stops()
                _save()
                _update_preview()
            gle.gradientChanged.connect(_on_gradient)
            rl.addWidget(gle, 1)
            self._add_asset_widget(row)

    def _build_animator_state_inspector(self, state: AnimatorState):
        self._build_section_title(f"State: {state.name}")
        name_edit = QLineEdit(state.name)
        name_edit.setStyleSheet(self._animator_input_style())
        name_edit.textChanged.connect(lambda t: self._on_animator_state_name_changed(state, t))
        self._content_layout.addWidget(QLabel("Name"))
        self._content_layout.addWidget(name_edit)
        speed_sb = QDoubleSpinBox()
        speed_sb.setRange(-100.0, 100.0)
        speed_sb.setValue(state.speed)
        speed_sb.valueChanged.connect(lambda v: setattr(state, 'speed', v))
        speed_sb.setStyleSheet(self._animator_input_style())
        self._content_layout.addWidget(QLabel("Speed"))
        self._content_layout.addWidget(speed_sb)
        tag_edit = QLineEdit(state.tag)
        tag_edit.setStyleSheet(self._animator_input_style())
        tag_edit.textChanged.connect(lambda t: setattr(state, 'tag', t))
        self._content_layout.addWidget(QLabel("Tag"))
        self._content_layout.addWidget(tag_edit)

    def _build_animator_transition_inspector(self, trans: AnimatorTransition):
        self._build_section_title("Transition")
        dur_sb = QDoubleSpinBox()
        dur_sb.setRange(0.0, 10.0)
        dur_sb.setSingleStep(0.05)
        dur_sb.setValue(trans.transition_duration)
        dur_sb.valueChanged.connect(lambda v: setattr(trans, 'transition_duration', v))
        dur_sb.setStyleSheet(self._animator_input_style())
        self._content_layout.addWidget(QLabel("Duration"))
        self._content_layout.addWidget(dur_sb)
        exit_sb = QDoubleSpinBox()
        exit_sb.setRange(0.0, 1.0)
        exit_sb.setSingleStep(0.05)
        exit_sb.setValue(trans.exit_time)
        exit_sb.valueChanged.connect(lambda v: setattr(trans, 'exit_time', v))
        exit_sb.setStyleSheet(self._animator_input_style())
        self._content_layout.addWidget(QLabel("Exit Time"))
        self._content_layout.addWidget(exit_sb)
        has_exit_cb = QCheckBox("Has Exit Time")
        has_exit_cb.setChecked(trans.has_exit_time)
        has_exit_cb.toggled.connect(lambda v: setattr(trans, 'has_exit_time', v))
        has_exit_cb.setStyleSheet("font-size: 10px;")
        self._content_layout.addWidget(has_exit_cb)
        fixed_dur_cb = QCheckBox("Fixed Duration")
        fixed_dur_cb.setChecked(trans.has_fixed_duration)
        fixed_dur_cb.toggled.connect(lambda v: setattr(trans, 'has_fixed_duration', v))
        fixed_dur_cb.setStyleSheet("font-size: 10px;")
        self._content_layout.addWidget(fixed_dur_cb)
        self._build_section_title("Conditions")
        for cond in trans.conditions:
            self._build_animator_condition_row(trans, cond)
        add_cond_btn = QPushButton("+ Add Condition")
        add_cond_btn.clicked.connect(lambda: self._add_animator_condition(trans))
        add_cond_btn.setStyleSheet(self._animator_btn_style())
        self._content_layout.addWidget(add_cond_btn)

    def _build_animator_condition_row(self, trans, cond):
        row = QHBoxLayout()
        param_combo = QComboBox()
        ctrl = self._animator_controller
        if ctrl:
            for p in ctrl.parameters:
                param_combo.addItem(p.name)
        param_combo.setCurrentText(cond.parameter)
        param_combo.textActivated.connect(lambda text, c=cond: setattr(c, 'parameter', text))
        param_combo.setStyleSheet(self._animator_input_style())
        row.addWidget(param_combo)
        mode_combo = QComboBox()
        modes = ["if", "if_not", "greater", "less", "equals", "not_equal"]
        mode_combo.addItems(modes)
        mode_combo.setCurrentText(cond.mode.value)
        mode_combo.textActivated.connect(lambda text, c=cond: setattr(c, 'mode', AnimatorConditionMode(text)))
        mode_combo.setStyleSheet(self._animator_input_style())
        row.addWidget(mode_combo)
        thresh_sb = QDoubleSpinBox()
        thresh_sb.setRange(-99999.0, 99999.0)
        thresh_sb.setValue(cond.threshold)
        thresh_sb.valueChanged.connect(lambda v, c=cond: setattr(c, 'threshold', v))
        thresh_sb.setStyleSheet(self._animator_input_style())
        row.addWidget(thresh_sb)
        del_btn = QPushButton("x")
        del_btn.setFixedSize(*scale_xy(20, 20))
        del_btn.clicked.connect(lambda: self._remove_animator_condition(trans, cond))
        del_btn.setStyleSheet("QPushButton { color: #c66; border: none; font-size: 10px; } QPushButton:hover { color: #f88; }")
        row.addWidget(del_btn)
        self._content_layout.addLayout(row)

    def _add_animator_condition(self, trans):
        trans.conditions.append(AnimatorCondition())
        self._rebuild()

    def _remove_animator_condition(self, trans, cond):
        if cond in trans.conditions:
            trans.conditions.remove(cond)
        self._rebuild()

    def _on_animator_state_name_changed(self, state, new_name):
        old_name = state.name
        state.name = new_name
        self._rebuild()

    def _build_section_title(self, text: str):
        lbl = QLabel(text)
        self._content_layout.addWidget(lbl)

    def _animator_input_style(self) -> str:
        return ""

    def _animator_btn_style(self) -> str:
        return ""

    def _rebuild(self):
        self._content_widget.setVisible(False)
        try:
            self._updating = True
            for w in self._comp_widgets:
                w.hide()
                w.deleteLater()
            self._comp_widgets.clear()
            while self._content_layout.count():
                item = self._content_layout.takeAt(0)
                if item and item.widget():
                    item.widget().hide()
                    item.widget().deleteLater()
            self._asset_widgets.clear()
            stretch = self._content_layout.takeAt(self._content_layout.count() - 1)
            if self._animator_mode:
                self._header_widget.setVisible(False)
                self._add_comp_btn.setVisible(False)
                if self._animator_state:
                    self._build_animator_state_inspector(self._animator_state)
                elif self._animator_transition:
                    self._build_animator_transition_inspector(self._animator_transition)
                self._content_layout.addStretch()
                return
            if self._asset_path:
                self._header_widget.setVisible(False)
                self._add_comp_btn.setVisible(False)
                self._build_import_settings()
                self._content_layout.addStretch()
                return
            if not self._entity:
                self._header_widget.setVisible(False)
                self._add_comp_btn.setVisible(False)
                self._content_layout.addStretch()
                return
            self._header_widget.setVisible(True)
            self._add_comp_btn.setVisible(True)
            if self._entity.is_prefab_instance:
                from core.ecs.prefab import Prefab, PrefabLibrary
                prefab_path = PrefabLibrary.path_for_guid(self._entity._prefab_guid)
                overrides = Prefab.compute_all_overrides([self._entity])
                prefab_name = prefab_path.replace("\\", "/").split("/")[-1] if prefab_path else "Prefab"
                self._prefab_label.setText(f"Prefab: {prefab_name}")
                if overrides:
                    self._override_label.setText(f"({len(overrides)} override{'s' if len(overrides) != 1 else ''})")
                    self._override_label.setVisible(True)
                else:
                    self._override_label.setVisible(False)
                self._prefab_bar_widget.setVisible(True)
            else:
                self._prefab_bar_widget.setVisible(False)
            for i in range(self._header_widget.layout().count() - 1, -1, -1):
                item = self._header_widget.layout().itemAt(i)
                if item and item.widget() and item.widget().property("is_multi_label"):
                    item.widget().deleteLater()
            if len(self._selected_entities) > 1:
                multi_label = QLabel(f"({len(self._selected_entities)} selected)")
                multi_label.setProperty("is_multi_label", True)
                multi_label.setStyleSheet(f"color: {_accent()}; font-size: 11px; padding: 2px 0; background: transparent;")
                self._header_widget.layout().insertWidget(0, multi_label)
            self._active_cb.setChecked(self._entity.active)
            self._name_edit.setText(self._entity.name)
            tags = ", ".join(self._entity.tags)
            self._tag_edit.setText(tags)
            self._layer_sb.setValue(self._entity.layer)
            rev_map = {id(c): k for k, c in self._entity._components.items()}
            comps = self._entity.get_all_components()
            for idx, comp in enumerate(comps):
                try:
                    key = rev_map.get(id(comp), "")
                    cw = ComponentWidget(comp, self._entity, self._selected_entities, self._content_widget, component_key=key)
                    cw.remove_requested.connect(self._remove_component)
                    cw.move_up_requested.connect(self._move_component_up)
                    cw.move_down_requested.connect(self._move_component_down)
                    cw.reorder_requested.connect(self._on_reorder_component)
                    cw._move_up_btn.setEnabled(idx > 0)
                    cw._move_down_btn.setEnabled(idx < len(comps) - 1)
                    self._content_layout.addWidget(cw)
                    self._comp_widgets.append(cw)
                except Exception as e:
                    Logger.error(f"Inspector build error for {type(comp).__name__}: {e}", e)
            self._content_layout.addStretch()
        finally:
            self._updating = False
            self._content_widget.setVisible(True)

    def _refresh_transform(self):
        for cw in self._comp_widgets:
            ctype = type(cw._component).__name__
            if ctype == "Transform":
                try: cw.refresh_transform()
                except: pass
            else:
                fields = getattr(type(cw._component), "_inspector_fields", lambda: [])()
                for f in fields:
                    if f.field_type.value == "vec2":
                        try: cw.refresh_vec2_field(f.name)
                        except: pass
                    elif f.field_type.value == "vec3":
                        try: cw.refresh_vec3_field(f.name)
                        except: pass
                    elif f.field_type.value == "vec4":
                        try: cw.refresh_vec4_field(f.name)
                        except: pass
                    elif f.field_type.value in ("vec2_slider", "vec3_slider"):
                        try: cw.refresh_vec2_field(f.name) if f.field_type.value == "vec2_slider" else cw.refresh_vec3_field(f.name)
                        except: pass

    def _on_active_changed(self, checked: bool):
        if self._updating or not self._entity: return
        old = not checked
        self._entity.active = checked
        get_history().execute(SetComponentCommand(self._entity, type(self._entity), "active", old, checked))

    def _on_name_changed(self, text: str):
        if self._updating or not self._entity: return
        old = self._entity.name
        self._entity.name = text
        get_history().execute(SetComponentCommand(self._entity, type(self._entity), "name", old, text))
        if self._engine.scene: self._engine.scene.mark_dirty()

    def _on_tag_changed(self, text: str):
        if self._updating or not self._entity: return
        old = set(self._entity.tags)
        self._entity._tags = set(t.strip() for t in text.split(",") if t.strip())
        get_history().execute(SetComponentCommand(self._entity, type(self._entity), "tags", old, set(self._entity.tags)))

    def _on_layer_changed(self, val: int):
        if self._updating or not self._entity: return
        old = self._entity.layer
        self._entity.layer = val
        get_history().execute(SetComponentCommand(self._entity, type(self._entity), "layer", old, val))

    def _remove_component(self, comp_name: str, comp_key: str = ""):
        if not self._entity: return
        from core.ecs.ecs import ComponentRegistry
        from core.foundation.commands import RemoveComponentCommand
        cls = ComponentRegistry.get(comp_name)
        if cls:
            cmd = RemoveComponentCommand(self._entity, cls, component_key=comp_key)
            get_history().execute(cmd)
        self._rebuild()
        self._send_collab_component_remove(comp_key or comp_name)

    def _on_reorder_component(self, source_eid: str, dragged_key: str, target_key: str):
        if not self._entity or self._entity.id != source_eid:
            return
        keys = list(self._entity._components.keys())
        if dragged_key not in keys or target_key not in keys:
            return
        dragged_idx = keys.index(dragged_key)
        target_idx = keys.index(target_key)
        if dragged_idx == target_idx:
            return
        keys.remove(dragged_key)
        new_pos = keys.index(target_key)
        keys.insert(new_pos + 1 if dragged_idx < target_idx else new_pos, dragged_key)
        self._entity._components = {k: self._entity._components[k] for k in keys}
        self._rebuild()

    def _move_component_up(self, comp_key: str):
        if not self._entity: return
        self._entity.move_component(comp_key, -1)
        self._rebuild()

    def _move_component_down(self, comp_key: str):
        if not self._entity: return
        self._entity.move_component(comp_key, 1)
        self._rebuild()

    def _show_add_component_menu(self):
        if not self._entity: return
        dlg = ComponentPickerDialog(self._entity, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            result = dlg.selected_result()
            if result and result["type"] == "component":
                self._add_component(result["name"])
            elif result and result["type"] == "script":
                self._add_script_component(result["path"])

    def _send_collab_component_add(self, comp_name: str, added_key: str):
        if not self._entity or not hasattr(self, '_engine') or not self._engine:
            return
        collab = self._engine.collab_manager if hasattr(self._engine, 'collab_manager') else None
        if not collab or not collab.connected:
            return
        from core.ecs.ecs import ComponentRegistry
        cls = ComponentRegistry.get(comp_name)
        if not cls:
            return
        comp_key = added_key or next((k for k in self._entity._components if k == comp_name or k.startswith(comp_name + ".")), None)
        comp = self._entity._components.get(comp_key) if comp_key else None
        if not comp:
            return
        comp_data = comp.serialize() if hasattr(comp, 'serialize') else {}
        if not comp_data:
            comp_data = {}
            for attr in dir(comp):
                if attr.startswith('_'): continue
                v = getattr(comp, attr)
                if callable(v): continue
                comp_data[attr] = v
        collab.send_component_add(self._entity.id, comp_name, comp_data)

    def _send_collab_component_remove(self, comp_key: str):
        if not self._entity or not hasattr(self, '_engine') or not self._engine:
            return
        collab = self._engine.collab_manager if hasattr(self._engine, 'collab_manager') else None
        if not collab or not collab.connected:
            return
        collab.send_component_remove(self._entity.id, comp_key)

    def _add_component(self, comp_name: str):
        if not self._entity: return
        from core.ecs.ecs import ComponentRegistry
        from core.foundation.commands import AddComponentCommand
        cls = ComponentRegistry.get(comp_name)
        if cls:
            can_multiple = getattr(cls, '_allow_multiple', False)
            if not can_multiple and self._entity.has_component(cls):
                return
            cmd = AddComponentCommand(self._entity, cls)
            get_history().execute(cmd)
            self._rebuild()
            self._send_collab_component_add(comp_name, cmd._added_key)

    def _add_script_component(self, script_path: str):
        if not self._entity: return
        from core.components.scripting.script_component import ScriptComponent
        from core.foundation.commands import AddComponentCommand
        cmd = AddComponentCommand(self._entity, ScriptComponent)
        get_history().execute(cmd)
        found = self._entity.get_components(ScriptComponent)
        if found:
            root = self._engine.project_root
            try:
                rel = os.path.relpath(script_path, root)
                found[-1].script_path = rel.replace("\\", "/") if not rel.startswith("..") else os.path.abspath(script_path)
            except ValueError:
                found[-1].script_path = os.path.abspath(script_path)
            self._rebuild()
        self._send_collab_component_add("ScriptComponent", cmd._added_key)

    def _on_apply_prefab(self):
        if not self._entity or not self._entity.is_prefab_instance:
            return
        from core.ecs.prefab import Prefab, PrefabLibrary
        from core.foundation.logger import Logger
        prefab_path = PrefabLibrary.path_for_guid(self._entity._prefab_guid)
        if not prefab_path:
            Logger.warning("Cannot find prefab asset for this instance.")
            return
        roots = Prefab.get_prefab_roots([self._entity])
        all_entities = []
        def collect(e):
            all_entities.append(e)
            for c in e.children:
                collect(c)
        for r in roots:
            collect(r)
        current_data = {}
        for e in all_entities:
            current_data[e.id] = e.serialize()
        pref = Prefab(self._entity.name, self._entity._prefab_guid)
        pref.roots_data = [current_data[r.id] for r in roots]
        pref.save(prefab_path)
        PrefabLibrary.invalidate(prefab_path)
        self._rebuild()

    def _on_revert_prefab(self):
        if not self._entity or not self._entity.is_prefab_instance:
            return
        from core.foundation.commands import RevertPrefabInstanceCommand, get_history
        from core.ecs.prefab import Prefab
        scene = self._engine.scene if hasattr(self._engine, 'scene') else None
        if not scene:
            return
        roots = Prefab.get_prefab_roots([self._entity])
        cmd = RevertPrefabInstanceCommand(scene, roots)
        get_history().execute(cmd)
        self._rebuild()

    def _on_select_prefab_asset(self):
        if not self._entity or not self._entity.is_prefab_instance:
            return
        from core.ecs.prefab import PrefabLibrary
        prefab_path = PrefabLibrary.path_for_guid(self._entity._prefab_guid)
        if prefab_path:
            self.show_import_settings(prefab_path)

    def _on_open_prefab_editor(self):
        if not self._entity or not self._entity.is_prefab_instance:
            return
        from core.ecs.prefab import PrefabLibrary
        prefab_path = PrefabLibrary.path_for_guid(self._entity._prefab_guid)
        if prefab_path:
            self.open_prefab_editor.emit(prefab_path)
