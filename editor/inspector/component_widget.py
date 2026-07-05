# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, QDoubleSpinBox, QSpinBox, \
    QSlider, QComboBox, QFrame, QMenu, QDialog, QPlainTextEdit, QApplication, QLineEdit
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData, QSize, QEvent
from PyQt6.QtGui import QAction, QPixmap, QIcon, QDrag, QCursor, QColor
from core.editor_scale import scale, scale_xy
from core.math3d import Vec2, Vec3, Vec4, Quat
from core.logger import Logger
from core.commands import SetComponentCommand, CompoundCommand, get_history
from core.curve import Curve
from editor.curve_editor import CurvePreview, CurveEditorDialog
from core.gui.widgets import AnchorPresetSelector
from core.components.animation.animator_controller import (
    AnimatorController, AnimatorState, AnimatorTransition,
    AnimatorCondition, AnimatorConditionMode,
)
from core.physics.collision_layers import MAX_LAYERS, DEFAULT_LAYER_NAMES
from core.config import get_project_config
from editor.inspector.constants import (_FUSION_ACCENT_GREEN, _FUSION_ACCENT_RED,
    _FUSION_CARD_RADIUS, _FUSION_INPUT_RADIUS,
    _COMPONENT_MIME, _accent)
from editor.inspector.helpers import (make_spinbox, make_clickable_label, get_component_icon_pixmap,
    get_component_source_path, get_property_line_number, collapse_value, make_resource_picker,
    make_gameobject_picker, make_resource_type_picker, make_asset_picker, make_vec2_row,
    make_vec3_row, make_vec4_row, make_vec2_slider_row, make_vec3_slider_row)

class ComponentWidget(QWidget):
    remove_requested = pyqtSignal(str, str)
    move_up_requested = pyqtSignal(str)
    move_down_requested = pyqtSignal(str)
    reorder_requested = pyqtSignal(str, str, str)

    def __init__(self, component, entity=None, selected_entities=None, parent=None, component_key: str = ""):
        super().__init__(parent)
        self._component = component
        self._entity = entity
        self._selected_entities = list(selected_entities if selected_entities else [])
        self._component_key = component_key
        self._updating = False
        self._collapsed = False
        self.setStyleSheet(f"""
            ComponentWidget {{
                border-radius: {_FUSION_CARD_RADIUS};
            }}
        """)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self._header_widget = QWidget()
        self._header_widget.setObjectName("compHeader")
        header_layout = QHBoxLayout(self._header_widget)
        header_layout.setContentsMargins(6, 3, 6, 3)
        header_layout.setSpacing(4)
        self._collapse_btn = QPushButton("\u25bc")
        self._collapse_btn.setFixedSize(*scale_xy(14, 14))
        self._collapse_btn.setFlat(True)
        self._collapse_btn.clicked.connect(self._toggle_collapse)
        header_layout.addWidget(self._collapse_btn)
        self._icon_label = QLabel()
        self._icon_label.setFixedSize(*scale_xy(16, 16))
        comp_cls = type(component)
        pix = get_component_icon_pixmap(comp_cls, 16)
        self._icon_label.setPixmap(pix)
        self._icon_label.setToolTip(comp_cls.__name__)
        self._icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        header_layout.addWidget(self._icon_label)
        self._name_label = QLabel(type(component).__name__)
        self._name_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        header_layout.addWidget(self._name_label, 1)
        self._drag_start_pos = None
        self._enabled_cb = QCheckBox()
        self._enabled_cb.setChecked(component.enabled)
        self._enabled_cb.toggled.connect(self._on_enabled_toggled)
        self._enabled_cb.setStyleSheet(f"background: transparent;")
        header_layout.addWidget(self._enabled_cb)
        self._move_up_btn = QPushButton("^")
        self._move_up_btn.setFixedSize(*scale_xy(16, 16))
        self._move_up_btn.setFlat(True)
        self._move_up_btn.clicked.connect(lambda: self.move_up_requested.emit(self._component_key))
        header_layout.addWidget(self._move_up_btn)
        self._move_down_btn = QPushButton("v")
        self._move_down_btn.setFixedSize(*scale_xy(16, 16))
        self._move_down_btn.setFlat(True)
        self._move_down_btn.clicked.connect(lambda: self.move_down_requested.emit(self._component_key))
        header_layout.addWidget(self._move_down_btn)
        self._header_widget.installEventFilter(self)
        main_layout.addWidget(self._header_widget)
        self._content_widget = QWidget()
        self._content_widget.setObjectName("compBody")
        self._content_widget.setStyleSheet(f"""
            #compBody {{
                border-bottom-left-radius: {_FUSION_CARD_RADIUS};
                border-bottom-right-radius: {_FUSION_CARD_RADIUS};
            }}
        """)
        self._layout = QVBoxLayout(self._content_widget)
        self._layout.setContentsMargins(8, 6, 8, 6)
        self._layout.setSpacing(3)
        main_layout.addWidget(self._content_widget)
        self.setAcceptDrops(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self._build_fields()
        self._update_appearance()

    def eventFilter(self, obj, event):
        if obj is self._header_widget:
            if event.type() == event.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_start_pos = event.position().toPoint()
                self._drag_started = False
            elif event.type() == event.Type.MouseMove and (event.buttons() & Qt.MouseButton.LeftButton):
                if self._drag_start_pos is not None and not self._drag_started:
                    delta = event.position().toPoint() - self._drag_start_pos
                    if delta.manhattanLength() >= QApplication.startDragDistance():
                        self._start_drag()
                return True
            elif event.type() == event.Type.MouseButtonRelease:
                self._drag_start_pos = None
                self._drag_started = False
        return super().eventFilter(obj, event)

    def _start_drag(self):
        self._drag_started = True
        comp_data = self._component.serialize()
        data = {
            "entity_id": self._entity.id if self._entity else "",
            "component_key": self._component_key,
            "component_type": type(self._component).__name__,
            "component_data": comp_data,
        }
        mime = QMimeData()
        mime.setData(_COMPONENT_MIME, json.dumps(data).encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        pixmap = self._header_widget.grab()
        drag.setPixmap(pixmap)
        drag.setHotSpot(self._header_widget.mapFromGlobal(QCursor.pos()))
        drag.exec(Qt.DropAction.MoveAction | Qt.DropAction.CopyAction)
        self._drag_start_pos = None
        self._drag_started = False

    def _toggle_collapse(self):
        self._collapsed = not self._collapsed
        self._collapse_btn.setText("\u25b6" if self._collapsed else "\u25bc")
        self._content_widget.setVisible(not self._collapsed)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(_COMPONENT_MIME):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(_COMPONENT_MIME):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasFormat(_COMPONENT_MIME):
            raw = bytes(event.mimeData().data(_COMPONENT_MIME)).decode("utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                event.ignore()
                return
            dragged_key = data.get("component_key", "")
            target_key = self._component_key
            source_eid = data.get("entity_id", "")
            if dragged_key and target_key and source_eid == (self._entity.id if self._entity else ""):
                self.reorder_requested.emit(source_eid, dragged_key, target_key)
                event.acceptProposedAction()
                return
            event.ignore()
        else:
            super().dropEvent(event)

    def _undo_setter(self, prop_name):
        c = self._component
        def _set_and_sync(v):
            get_history().execute(SetComponentCommand(self._entity, type(c), prop_name, getattr(c, prop_name), v))
            try:
                from core.engine import Engine
                collab = Engine.instance().collab_manager
                if collab and collab.connected and self._entity:
                    val = collapse_value(v)
                    print(f"COLLAB SYNC: entity={self._entity.id}, key={self._component_key}, prop={prop_name}, val={val}")
                    collab.send_component_update(self._entity.id, self._component_key, prop_name, val)
                else:
                    print(f"COLLAB SKIP: collab={collab}, connected={collab.connected if collab else 'N/A'}, entity={self._entity}")
            except Exception as e:
                import traceback
                traceback.print_exc()
        return _set_and_sync

    def _undo_setter_all(self, comp_type, prop_name):
        entities = [e for e in self._selected_entities if e.has_component(comp_type)]
        if len(entities) <= 1:
            return self._undo_setter(prop_name)
        old_values = []
        for ent in entities:
            comp = ent.get_component(comp_type)
            if comp and hasattr(comp, prop_name):
                old_values.append(getattr(comp, prop_name))
            else:
                old_values.append(None)
        def _set_all(v):
            cmds = []
            for i, ent in enumerate(entities):
                comp = ent.get_component(comp_type)
                if comp and hasattr(comp, prop_name):
                    old_v = old_values[i]
                    if old_v is not None:
                        cmds.append(SetComponentCommand(ent, comp_type, prop_name, old_v, v))
            if cmds:
                get_history().execute(CompoundCommand(cmds, f"Set {prop_name} on {len(entities)} entities"))
        return _set_all

    def _undo_setter_int(self, prop_name):
        return self._undo_setter(prop_name)

    def _on_layer_mask_toggle(self, prop_name, bit, btn, layer_names, menu, all_act, nothing_act):
        mask = int(getattr(self._component, prop_name))
        if mask & (1 << bit):
            mask &= ~(1 << bit)
        else:
            mask |= 1 << bit
        setattr(self._component, prop_name, mask)
        self._update_layer_mask_text(btn, mask, layer_names)
        all_act.setChecked(mask == 0xFFFF)
        nothing_act.setChecked(mask == 0)

    def _on_layer_mask_set_all(self, prop_name, state, btn, layer_names, menu):
        mask = 0xFFFF if state else 0
        setattr(self._component, prop_name, mask)
        self._update_layer_mask_text(btn, mask, layer_names)
        for act in menu.actions():
            if act.isCheckable() and act.text() not in ("Everything", "Nothing"):
                act.setChecked(state)
            elif act.text() == "Everything":
                act.setChecked(state)
            elif act.text() == "Nothing":
                act.setChecked(not state)

    def _update_layer_mask_text(self, btn, mask, layer_names):
        if mask == 0:
            btn.setText("Nothing")
        elif mask == 0xFFFF:
            btn.setText("Everything")
        else:
            selected = []
            for i in range(MAX_LAYERS):
                if mask & (1 << i):
                    name = layer_names[i] if i < len(layer_names) else f"Layer{i}"
                    selected.append(name)
            if len(selected) <= 3:
                btn.setText(", ".join(selected))
            else:
                btn.setText(f"{', '.join(selected[:3])}... (+{len(selected)-3})")

    def _update_appearance(self):
        self._content_widget.setEnabled(self._component.enabled)

    def _on_enabled_toggled(self, checked: bool):
        old = not checked
        self._component.enabled = checked
        if checked: self._component.on_enable()
        else: self._component.on_disable()
        if self._entity:
            get_history().execute(SetComponentCommand(self._entity, type(self._component), "enabled", old, checked))
        self._update_appearance()

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        from editor.inspector.panel import InspectorPanel
        copy_comp = QAction("Copy Component", self)
        copy_comp.triggered.connect(self._copy_component)
        menu.addAction(copy_comp)
        copy_vals = QAction("Copy Values", self)
        copy_vals.triggered.connect(self._copy_values)
        menu.addAction(copy_vals)
        paste_vals = QAction("Paste Component Values", self)
        paste_vals.setEnabled(InspectorPanel._clipboard is not None)
        paste_vals.triggered.connect(self._paste_values)
        menu.addAction(paste_vals)
        menu.addSeparator()
        rem = QAction("Remove Component", self)
        rem.triggered.connect(lambda: self.remove_requested.emit(type(self._component).__name__, self._component_key))
        menu.addAction(rem)
        menu.exec(self.mapToGlobal(pos))

    def _copy_component(self):
        from editor.inspector.panel import InspectorPanel
        InspectorPanel._clipboard = {
            "mode": "component",
            "type": type(self._component).__name__,
            "data": self._component.serialize(),
        }

    def _copy_values(self):
        from editor.inspector.panel import InspectorPanel
        data = self._component.serialize()
        InspectorPanel._clipboard = {
            "mode": "values",
            "type": type(self._component).__name__,
            "data": data,
        }

    def _paste_values(self):
        from editor.inspector.panel import InspectorPanel
        if InspectorPanel._clipboard is None:
            return
        from core.ecs import ComponentRegistry
        cb = InspectorPanel._clipboard
        target_type_name = cb["type"]
        target_cls = ComponentRegistry.get(target_type_name)
        if not target_cls:
            return
        source_data = cb["data"]
        current_type_name = type(self._component).__name__
        if target_type_name != current_type_name:
            return
        cmds = []
        entities = [e for e in self._selected_entities if e.has_component(target_cls)]
        if not entities:
            entities = [self._entity] if self._entity else []
            if not entities:
                return
        for ent in entities:
            comp = ent.get_component(target_cls)
            if comp:
                for key, val in source_data.items():
                    if key in ("type", "_key", "enabled"):
                        continue
                    old_val = getattr(comp, key, None)
                    if old_val is not None:
                        setattr(comp, key, val)
                        cmds.append(SetComponentCommand(ent, target_cls, key, old_val, val))
        if cmds:
            get_history().execute(CompoundCommand(cmds, f"Paste {target_type_name}"))

    def _build_fields(self):
        comp = self._component
        ctype = type(comp).__name__
        if ctype == "Transform":
            self._build_transform()
        elif ctype == "ScriptComponent":
            self._build_script_fields(comp)
        else:
            self._toggle_checkboxes: dict[str, QCheckBox] = {}
            self._toggle_rows: dict[str, list[QWidget]] = {}
            fields = getattr(type(comp), "_inspector_fields", lambda: [])()
            for field in fields:
                self._build_field_from_meta(field)
            for toggle_name, cb in self._toggle_checkboxes.items():
                rows = self._toggle_rows.get(toggle_name, [])
                if rows:
                    cb.toggled.connect(lambda v, rs=rows: self._on_toggle_changed(v, rs))
                    for r in rows:
                        r.setVisible(cb.isChecked())

    def _on_toggle_changed(self, v: bool, rows: list[QWidget]):
        for r in rows:
            r.setVisible(v)

    def _add_field(self, label: str, widget: QWidget, prop_name: str = "", toggle_field: str = ""):
        row = QWidget()
        row.setStyleSheet(f"background: transparent;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        lbl = QLabel(label)
        lbl.setFixedWidth(scale(100))
        rl.addWidget(lbl)
        rl.addWidget(widget, 1)
        if prop_name:
            comp_type = type(self._component).__name__
            src_path = get_component_source_path(type(self._component))
            line_num = get_property_line_number(type(self._component), prop_name)
            source_lbl = make_clickable_label("src", lambda sp=src_path, ln=line_num: self._show_source(sp, ln, comp_type, prop_name))
            rl.addWidget(source_lbl)
        self._layout.addWidget(row)
        if toggle_field:
            self._toggle_rows.setdefault(toggle_field, []).append(row)

    def _build_field_from_meta(self, field):
        c = self._component
        prop_name = field.name
        if field.field_type.value == "header":
            header = QLabel(f"  {field.label}")
            header.setStyleSheet(f"""
                QLabel {{
                    color: {_accent()};
                    font-weight: 600;
                    font-size: 11px;
                    padding: 5px 0 3px 0;
                }}
            """)
            self._layout.addWidget(header)
            return
        value = getattr(c, prop_name)
        if field.field_type.value == "float":
            sb = make_spinbox(value, field.min_val, field.max_val, field.step, field.decimals)
            comp_cls = type(c)
            sb.valueChanged.connect(self._undo_setter_all(comp_cls, prop_name))
            self._add_field(field.label, sb, prop_name, field.toggle_field)
        elif field.field_type.value == "int":
            if field.readonly:
                lbl = QLabel(str(value))
                self._add_field(field.label, lbl)
            else:
                sb = QSpinBox()
                min_i = max(-2147483648, min(2147483647, int(field.min_val)))
                max_i = max(-2147483648, min(2147483647, int(field.max_val)))
                sb.setRange(min_i, max_i)
                sb.setValue(max(min_i, min(max_i, int(value))))
                sb.setMinimumWidth(60)
                comp_cls = type(c)
                sb.valueChanged.connect(self._undo_setter_all(comp_cls, prop_name))
                self._add_field(field.label, sb, prop_name, field.toggle_field)
        elif field.field_type.value == "slider":
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)
            _slider_scale = 1000.0
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(field.min_val * _slider_scale), int(field.max_val * _slider_scale))
            slider.setValue(int(value * _slider_scale))
            slider.setSingleStep(max(1, int(field.step * _slider_scale)))
            slider.setStyleSheet(f"""
                QSlider::groove:horizontal {{
                    border: none;
                    height: 4px;
                    background: {self.palette().mid().color().name()};
                    border-radius: 2px;
                }}
                QSlider::handle:horizontal {{
                    background: {_accent()};
                    border: none;
                    width: 10px;
                    height: 10px;
                    margin: -3px 0;
                    border-radius: 5px;
                }}
                QSlider::handle:horizontal:hover {{
                    background: #7bb5ff;
                }}
                QSlider::sub-page:horizontal {{
                    background: {_accent()};
                    border-radius: 2px;
                }}
            """)
            sb = make_spinbox(value, field.min_val, field.max_val, field.step, field.decimals)
            comp_cls = type(c)
            sb.valueChanged.connect(self._undo_setter_all(comp_cls, prop_name))
            _updating = [False]
            _slider_scale_val = _slider_scale
            def _on_slider(v):
                if _updating[0]: return
                _updating[0] = True
                sb.setValue(v / _slider_scale_val)
                _updating[0] = False
            def _on_spinbox(v):
                if _updating[0]: return
                _updating[0] = True
                slider.setValue(int(v * _slider_scale_val))
                _updating[0] = False
            slider.valueChanged.connect(_on_slider)
            sb.valueChanged.connect(_on_spinbox)
            rl.addWidget(slider, 1)
            rl.addWidget(sb)
            self._add_field(field.label, row, prop_name, field.toggle_field)
        elif field.field_type.value == "int_slider":
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)
            min_i = max(-2147483648, min(2147483647, int(field.min_val)))
            max_i = max(-2147483648, min(2147483647, int(field.max_val)))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_i, max_i)
            slider.setValue(max(min_i, min(max_i, int(value))))
            slider.setSingleStep(max(1, int(field.step)))
            slider.setStyleSheet(f"""
                QSlider::groove:horizontal {{
                    border: none;
                    height: 4px;
                    border-radius: 2px;
                }}
                QSlider::handle:horizontal {{
                    background: {_accent()};
                    border: none;
                    width: 10px;
                    height: 10px;
                    margin: -3px 0;
                    border-radius: 5px;
                }}
                QSlider::handle:horizontal:hover {{
                    background: #7bb5ff;
                }}
                QSlider::sub-page:horizontal {{
                    background: {_accent()};
                    border-radius: 2px;
                }}
            """)
            sb = QSpinBox()
            sb.setRange(min_i, max_i)
            sb.setValue(max(min_i, min(max_i, int(value))))
            sb.setMinimumWidth(50)
            comp_cls = type(c)
            sb.valueChanged.connect(self._undo_setter_all(comp_cls, prop_name))
            _updating_int = [False]
            def _on_slider_int(v):
                if _updating_int[0]: return
                _updating_int[0] = True
                sb.setValue(v)
                _updating_int[0] = False
            def _on_spinbox_int(v):
                if _updating_int[0]: return
                _updating_int[0] = True
                slider.setValue(v)
                _updating_int[0] = False
            slider.valueChanged.connect(_on_slider_int)
            sb.valueChanged.connect(_on_spinbox_int)
            rl.addWidget(slider, 1)
            rl.addWidget(sb)
            self._add_field(field.label, row, prop_name, field.toggle_field)
        elif field.field_type.value == "bool":
            cb = QCheckBox()
            cb.setChecked(value)
            
            comp_cls = type(c)
            cb.toggled.connect(self._undo_setter_all(comp_cls, prop_name))
            self._add_field(field.label, cb, prop_name, field.toggle_field)
            self._toggle_checkboxes[prop_name] = cb
        elif field.field_type.value == "button":
            btn = QPushButton(field.label)
            btn.setStyleSheet(f"""
                QPushButton:hover {{
                    border-color: {_accent()};
                }}
            """)
            btn.clicked.connect(lambda: getattr(c, prop_name, lambda: None)())
            self._add_field("", btn)
        elif field.field_type.value == "color":
            from editor.color_picker import ColorLineEdit
            initial = None
            if isinstance(value, (list, tuple)):
                initial = QColor.fromRgbF(*value[:4])
            elif isinstance(value, Vec3):
                initial = QColor.fromRgbF(value.x, value.y, value.z)
            elif isinstance(value, Vec4):
                initial = QColor.fromRgbF(value.x, value.y, value.z, value.w)
            elif isinstance(value, QColor):
                initial = value
            color_edit = ColorLineEdit(initial)
            comp_cls = type(c)
            def _on_color_changed(col, _pn=prop_name, _val=value, _cls=comp_cls):
                rgb = [col.redF(), col.greenF(), col.blueF()]
                if isinstance(_val, Vec4):
                    rgb.append(col.alphaF())
                if isinstance(_val, Vec3):
                    setattr(c, _pn, Vec3(*rgb))
                elif isinstance(_val, Vec4):
                    setattr(c, _pn, Vec4(*rgb))
                elif isinstance(_val, list):
                    setattr(c, _pn, rgb)
                elif isinstance(_val, tuple):
                    setattr(c, _pn, tuple(rgb))
                elif isinstance(_val, QColor):
                    setattr(c, _pn, col)
                get_history().execute(SetComponentCommand(self._entity, _cls, _pn, _val, rgb))
            color_edit.colorChanged.connect(_on_color_changed)
            self._add_field(field.label, color_edit, prop_name, field.toggle_field)
        elif field.field_type.value == "enum":
            combo = QComboBox()
            options = field.enum_options or []
            for opt in options:
                combo.addItem(opt)
            try:
                combo.setCurrentText(str(value))
            except Exception:
                pass
            comp_cls = type(c)
            def _on_enum_change(t):
                setattr(c, prop_name, t)
                get_history().execute(SetComponentCommand(self._entity, comp_cls, prop_name, value, t))
            combo.currentTextChanged.connect(_on_enum_change)
            self._add_field(field.label, combo, prop_name, field.toggle_field)
        elif field.field_type.value == "resource":
            pw = make_resource_picker(value, field.file_filter or "All Files (*)", self._undo_setter(prop_name))
            self._add_field(field.label, pw, prop_name, field.toggle_field)
        elif field.field_type.value == "gameobject":
            from core.engine import Engine
            scene = Engine.instance().scene
            gw = make_gameobject_picker(value, scene, self._undo_setter(prop_name))
            self._add_field(field.label, gw, prop_name, field.toggle_field)
        elif field.field_type.value == "resource_type":
            pw = make_resource_type_picker(value, field.resource_type or "", self._undo_setter(prop_name))
            self._add_field(field.label, pw, prop_name, field.toggle_field)
        elif field.field_type.value == "asset":
            pw = make_asset_picker(value, field.resource_type or "", self._undo_setter(prop_name))
            self._add_field(field.label, pw, prop_name, field.toggle_field)
        elif field.field_type.value == "curve":
            preview = CurvePreview()
            preview.setFixedSize(*scale_xy(60, 20))
            if isinstance(value, Curve):
                preview.set_curve(value)
            from PyQt6.QtWidgets import QColorDialog
            def _open_editor(*, _prop_name=prop_name, _field=field):
                dlg = CurveEditorDialog(c, _prop_name, self)
                if dlg.exec() == QDialog.DialogCode.Accepted:
                    preview.set_curve(getattr(c, _prop_name, None))
            preview.mousePressEvent = lambda ev, pe=_open_editor: pe() if ev.button() == Qt.MouseButton.LeftButton else None
            self._add_field(field.label, preview, prop_name, field.toggle_field)
        elif field.field_type.value == "anchor":
            ctrl = AnchorPresetSelector()
            if hasattr(c, prop_name):
                current = getattr(c, prop_name)
                if isinstance(current, int):
                    ctrl.set_anchor(current)
            def _on_anchor_change(v):
                setattr(c, prop_name, v)
                get_history().execute(SetComponentCommand(self._entity, type(c), prop_name, value, v))
            ctrl.anchor_changed.connect(_on_anchor_change)
            self._add_field(field.label, ctrl, prop_name, field.toggle_field)
        elif field.field_type.value == "text":
            if field.multiline:
                te = QPlainTextEdit()
                te.setPlainText(str(value) if value else "")
                te.setFixedHeight(scale(60))
                te.setStyleSheet(f"""
                    QPlainTextEdit {{
                        border-radius: {_FUSION_INPUT_RADIUS};
                        padding: 2px 4px;
                        font-size: 11px;
                        selection-background-color: {_accent()};
                    }}
                    QPlainTextEdit:focus {{ border-color: {_accent()}; }}
                """)
                te.textChanged.connect(lambda: setattr(c, prop_name, te.toPlainText()))
                comp_cls = type(c)
                te.focusOutEvent = lambda ev: (get_history().execute(SetComponentCommand(self._entity, comp_cls, prop_name, value, te.toPlainText())), QPlainTextEdit.focusOutEvent(te, ev))
            else:
                te = QLineEdit()
                te.setText(str(value) if value else "")
                te.setStyleSheet(f"""
                    QLineEdit {{
                        border-radius: {_FUSION_INPUT_RADIUS};
                        padding: 2px 4px;
                        font-size: 11px;
                        selection-background-color: {_accent()};
                    }}
                    QLineEdit:focus {{ border-color: {_accent()}; }}
                """)
                te.textChanged.connect(lambda: setattr(c, prop_name, te.text()))
                comp_cls = type(c)
                te.editingFinished.connect(lambda: get_history().execute(SetComponentCommand(self._entity, comp_cls, prop_name, value, te.text())))
            self._add_field(field.label, te, prop_name, field.toggle_field)
        elif field.field_type.value == "vec2":
            w, sbs = make_vec2_row(field.label, value, lambda: None)
            comp_cls = type(c)
            for sb in sbs:
                sb.valueChanged.connect(lambda v, pn=prop_name, sbs_box=sbs, cls=comp_cls: self._on_vec2_changed(pn, sbs_box))
            for i, sb in enumerate(sbs):
                def make_setter(idx):
                    def setter(v):
                        vec = getattr(c, prop_name)
                        lst = [vec.x, vec.y]
                        lst[idx] = v
                        setattr(c, prop_name, type(vec)(*lst))
                    return setter
                sb.valueChanged.connect(make_setter(i))
            self._layout.addWidget(w)
        elif field.field_type.value == "vec3":
            w, sbs = make_vec3_row(field.label, value, lambda: None)
            comp_cls = type(c)
            for i, sb in enumerate(sbs):
                def make_setter(idx):
                    def setter(v):
                        vec = getattr(c, prop_name)
                        lst = [vec.x, vec.y, vec.z]
                        lst[idx] = v
                        setattr(c, prop_name, type(vec)(*lst))
                    return setter
                sb.valueChanged.connect(make_setter(i))
            self._layout.addWidget(w)
        elif field.field_type.value == "list":
            self._build_list_field_standalone(field, prop_name)
        elif field.field_type.value == "layer_mask":
            btn = QPushButton("Everything")
            btn.setStyleSheet(f"""
                QPushButton {{
                    border-radius: {_FUSION_INPUT_RADIUS};
                    padding: 2px 6px;
                    font-size: 10px;
                    text-align: left;
                }}
            """)
            btn.setMinimumHeight(22)
            from core.engine import Engine
            cfg = get_project_config(Engine.instance().project_root)
            layer_names = cfg.get("physics.layer_names", DEFAULT_LAYER_NAMES) if cfg else DEFAULT_LAYER_NAMES
            menu = QMenu(self)

            all_act = menu.addAction("Everything")
            all_act.setCheckable(True)
            all_act.setChecked(True)
            all_act.triggered.connect(lambda checked: self._on_layer_mask_set_all(prop_name, checked, btn, layer_names, menu))
            nothing_act = menu.addAction("Nothing")
            nothing_act.setCheckable(True)
            nothing_act.triggered.connect(lambda checked: self._on_layer_mask_set_all(prop_name, not checked, btn, layer_names, menu))
            menu.addSeparator()
            for i in range(MAX_LAYERS):
                name = layer_names[i] if i < len(layer_names) else f"Layer{i}"
                act = menu.addAction(name)
                act.setCheckable(True)
                mask = int(getattr(self._component, prop_name))
                act.setChecked(bool(mask & (1 << i)))
                act.triggered.connect(lambda checked, b=i, b0=btn, lm=layer_names, m=menu, al=all_act, nt=nothing_act: self._on_layer_mask_toggle(prop_name, b, b0, lm, m, al, nt))
            _mask = int(getattr(self._component, prop_name))
            self._update_layer_mask_text(btn, _mask, layer_names)
            self._add_field(field.label, btn, prop_name, field.toggle_field)
        elif field.field_type.value == "vec4":
            w, sbs = make_vec4_row(field.label, value, lambda: None)
            comp_cls = type(c)
            for i, sb in enumerate(sbs):
                def make_setter(idx):
                    def setter(v):
                        vec = getattr(c, prop_name)
                        lst = [vec.x, vec.y, vec.z, vec.w]
                        lst[idx] = v
                        setattr(c, prop_name, type(vec)(*lst))
                    return setter
                sb.valueChanged.connect(make_setter(i))
            self._layout.addWidget(w)
        elif field.field_type.value == "keybinding":
            te = QLineEdit()
            te.setText(str(value) if value else "")
            te.setStyleSheet(f"""
                QLineEdit {{
                    border-radius: {_FUSION_INPUT_RADIUS};
                    padding: 2px 4px;
                    font-size: 11px;
                    selection-background-color: {_accent()};
                }}
                QLineEdit:focus {{ border-color: {_accent()}; }}
            """)
            te.textChanged.connect(lambda: setattr(c, prop_name, te.text()))
            comp_cls = type(c)
            te.editingFinished.connect(lambda: get_history().execute(SetComponentCommand(self._entity, comp_cls, prop_name, value, te.text())))
            self._add_field(field.label, te, prop_name, field.toggle_field)
        elif field.field_type.value == "vec2_slider":
            w, sbs = make_vec2_slider_row(field.label, value, lambda: None, field.min_val, field.max_val)
            comp_cls = type(c)
            for i, sb in enumerate(sbs):
                def make_setter(idx):
                    def setter(v):
                        vec = getattr(c, prop_name)
                        lst = [vec.x, vec.y]
                        lst[idx] = v
                        setattr(c, prop_name, type(vec)(*lst))
                    return setter
                sb.valueChanged.connect(make_setter(i))
            self._layout.addWidget(w)
        elif field.field_type.value == "vec3_slider":
            w, sbs = make_vec3_slider_row(field.label, value, lambda: None, field.min_val, field.max_val)
            comp_cls = type(c)
            for i, sb in enumerate(sbs):
                def make_setter(idx):
                    def setter(v):
                        vec = getattr(c, prop_name)
                        lst = [vec.x, vec.y, vec.z]
                        lst[idx] = v
                        setattr(c, prop_name, type(vec)(*lst))
                    return setter
                sb.valueChanged.connect(make_setter(i))
            self._layout.addWidget(w)
        elif field.field_type.value == "gradient":
            grad_preview = QPushButton()
            grad_preview.setFixedHeight(22)
            grad_preview.setMinimumWidth(120)
            grad_preview.setToolTip("Click to edit gradient")
            def _paint_gradient(btn, grad_data=value):
                if not grad_data:
                    return
                stops = []
                for pos, rgba in grad_data:
                    r, g, b, a = (int(c * 255) for c in rgba[:4])
                    stops.append((pos, f"rgba({r},{g},{b},{a})"))
                if stops:
                    css = ", ".join(f"stop:{p} {c}" for p, c in stops)
                    btn.setStyleSheet(
                        f"QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, {css}); "
                        f"border-radius: {_FUSION_INPUT_RADIUS}; border: 1px solid; }}"
                    )
            _paint_gradient(grad_preview)
            def _open_gradient_editor():
                from editor.gradient_editor import GradientEditorDialog
                dlg = GradientEditorDialog(value, "Edit Gradient", self)
                if dlg.exec() == QDialog.DialogCode.Accepted:
                    new_gradient = dlg.get_stops()
                    setattr(c, prop_name, new_gradient)
                    _paint_gradient(grad_preview, new_gradient)
                    get_history().execute(SetComponentCommand(self._entity, type(c), prop_name, value, new_gradient))
            grad_preview.clicked.connect(_open_gradient_editor)
            self._add_field(field.label, grad_preview, prop_name, field.toggle_field)
        elif field.field_type.value == "resource_path":
            pw = make_resource_picker(value, field.file_filter or "All Files (*)", self._undo_setter(prop_name))
            self._add_field(field.label, pw, prop_name, field.toggle_field)
        elif field.field_type.value == "string":
            te = QLineEdit()
            te.setText(str(value) if value else "")
            te.setStyleSheet(f"""
                QLineEdit {{
                    border-radius: {_FUSION_INPUT_RADIUS};
                    padding: 2px 4px;
                    font-size: 11px;
                    selection-background-color: {_accent()};
                }}
                QLineEdit:focus {{ border-color: {_accent()}; }}
            """)
            te.textChanged.connect(lambda: setattr(c, prop_name, te.text()))
            comp_cls = type(c)
            te.editingFinished.connect(lambda: get_history().execute(SetComponentCommand(self._entity, comp_cls, prop_name, value, te.text())))
            self._add_field(field.label, te, prop_name, field.toggle_field)
        elif field.field_type.value == "textarea":
            te = QPlainTextEdit()
            te.setPlainText(str(value) if value else "")
            te.setFixedHeight(scale(60))
            te.setStyleSheet(f"""
                QPlainTextEdit {{
                    border-radius: {_FUSION_INPUT_RADIUS};
                    padding: 2px 4px;
                    font-size: 11px;
                    selection-background-color: {_accent()};
                }}
                QPlainTextEdit:focus {{ border-color: {_accent()}; }}
            """)
            te.textChanged.connect(lambda: setattr(c, prop_name, te.toPlainText()))
            comp_cls = type(c)
            te.focusOutEvent = lambda ev: (get_history().execute(SetComponentCommand(self._entity, comp_cls, prop_name, value, te.toPlainText())), QPlainTextEdit.focusOutEvent(te, ev))
            self._add_field(field.label, te, prop_name, field.toggle_field)
        elif field.field_type.value == "layer":
            sb = QSpinBox()
            sb.setRange(0, 31)
            sb.setValue(int(value) if value is not None else 0)
            sb.setMinimumWidth(60)
            comp_cls = type(c)
            sb.valueChanged.connect(self._undo_setter_all(comp_cls, prop_name))
            self._add_field(field.label, sb, prop_name, field.toggle_field)

    def _build_list_field_standalone(self, field, prop_name):
        c = self._component
        old = getattr(self, "_list_container", None)
        if old is not None:
            old.setParent(None)
            old.deleteLater()
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(2)
        header = QWidget()
        hl = QHBoxLayout(header)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(2)
        lbl = QLabel(field.label)
        
        hl.addWidget(lbl)
        add_btn = QPushButton("+")
        add_btn.setFixedSize(*scale_xy(18, 18))
        add_btn.setStyleSheet(f"""
            QPushButton {{ color: {_FUSION_ACCENT_GREEN}; font-size: 10px; background: transparent; border: none; }}
        """)
        add_btn.clicked.connect(lambda: self._list_add_item(c, prop_name))
        hl.addStretch()
        hl.addWidget(add_btn)
        cl.addWidget(header)
        self._list_container = container
        def rebuild_list():
            for i in reversed(range(cl.count())):
                w = cl.itemAt(i).widget()
                if w is not header and w is not None:
                    w.deleteLater()
            items = getattr(c, prop_name, [])
            element_fields = self._get_list_element_fields(prop_name)
            for idx, elem in enumerate(items):
                row = self._build_list_row_standalone(prop_name, idx, elem, element_fields)
                cl.addWidget(row)
        rebuild_list()
        self._layout.addWidget(container)

    def _build_list_row_standalone(self, prop_name, index, elem_data, element_fields):
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)
        remove_btn = QPushButton("\u2212")
        remove_btn.setFixedSize(*scale_xy(16, 16))
        remove_btn.setStyleSheet(f"""
            QPushButton {{ color: {_FUSION_ACCENT_RED}; font-size: 10px; background: transparent; border: none; }}
        """)
        remove_btn.clicked.connect(lambda: self._list_remove_item(self._component, prop_name, index))
        rl.addWidget(remove_btn)
        elem_widget = QWidget()
        elem_widget.setStyleSheet("background: transparent;")
        el = QHBoxLayout(elem_widget)
        el.setContentsMargins(0, 0, 0, 0)
        el.setSpacing(2)
        for ef in element_fields:
            w = self._build_list_element_widget_standalone(prop_name, index, ef, elem_data)
            if w:
                el.addWidget(w)
        rl.addWidget(elem_widget, 1)
        return row

    def _build_list_element_widget_standalone(self, prop_name, index, ef, val):
        if ef.field_type.value == "float":
            sb = make_spinbox(val.get(ef.name, 0.0) if isinstance(val, dict) else 0.0)
            sb.setStyleSheet(f"""
                QDoubleSpinBox {{
                    background: palette(base); color: palette(text);
                    border: 1px solid palette(mid); border-radius: 2px; padding: 2px;
                }}
            """)
            def on_change(v, idx=index, pn=prop_name, fn=ef.name):
                items = list(getattr(self._component, pn))
                if idx < len(items) and isinstance(items[idx], dict):
                    items[idx][fn] = v
                    setattr(self._component, pn, items)
            sb.valueChanged.connect(on_change)
            return sb
        elif ef.field_type.value == "int":
            sb = QSpinBox()
            sb.setRange(-2147483648, 2147483647)
            sb.setValue(val.get(ef.name, 0) if isinstance(val, dict) else 0)
            sb.setStyleSheet(f"""
                QSpinBox {{
                    background: palette(base); color: palette(text);
                    border: 1px solid palette(mid); border-radius: 2px; padding: 2px;
                }}
            """)
            def on_change(v, idx=index, pn=prop_name, fn=ef.name):
                items = list(getattr(self._component, pn))
                if idx < len(items) and isinstance(items[idx], dict):
                    items[idx][fn] = v
                    setattr(self._component, pn, items)
            sb.valueChanged.connect(on_change)
            return sb
        elif ef.field_type.value == "bool":
            cb = QCheckBox()
            cb.setChecked(val.get(ef.name, False) if isinstance(val, dict) else False)
            cb.setStyleSheet("background: transparent;")
            def on_toggle(v, idx=index, pn=prop_name, fn=ef.name):
                items = list(getattr(self._component, pn))
                if idx < len(items) and isinstance(items[idx], dict):
                    items[idx][fn] = v
                    setattr(self._component, pn, items)
            cb.toggled.connect(on_toggle)
            return cb
        elif ef.field_type.value == "gameobject":
            from core.engine import Engine
            scene = Engine.instance().scene
            eid = val.get(ef.name, "") if isinstance(val, dict) else ""
            def on_entity(eid, idx=index, pn=prop_name, fn=ef.name):
                items = list(getattr(self._component, pn))
                if idx < len(items) and isinstance(items[idx], dict):
                    items[idx][fn] = eid
                    setattr(self._component, pn, items)
            return make_gameobject_picker(eid, scene, on_entity)
        elif ef.field_type.value == "enum":
            combo = QComboBox()
            options = ef.enum_options or []
            for opt in options:
                combo.addItem(opt)
            current = val.get(ef.name, "") if isinstance(val, dict) else ""
            try: combo.setCurrentText(str(current))
            except: pass
            def on_change(t, idx=index, pn=prop_name, fn=ef.name):
                items = list(getattr(self._component, pn))
                if idx < len(items) and isinstance(items[idx], dict):
                    items[idx][fn] = t
                    setattr(self._component, pn, items)
            combo.currentTextChanged.connect(on_change)
            return combo
        elif ef.field_type.value == "resource":
            path = val.get(ef.name, "") if isinstance(val, dict) else ""
            filter_str = ef.file_filter or "All Files (*)"
            def on_change(t, idx=index, pn=prop_name, fn=ef.name):
                items = list(getattr(self._component, pn))
                if idx < len(items) and isinstance(items[idx], dict):
                    items[idx][fn] = t
                    setattr(self._component, pn, items)
            return make_resource_picker(path, filter_str, on_change)
        elif ef.field_type.value == "resource_type":
            path = val.get(ef.name, "") if isinstance(val, dict) else ""
            rt = ef.resource_type or ""
            def on_change(t, idx=index, pn=prop_name, fn=ef.name):
                items = list(getattr(self._component, pn))
                if idx < len(items) and isinstance(items[idx], dict):
                    items[idx][fn] = t
                    setattr(self._component, pn, items)
            return make_resource_type_picker(path, rt, on_change)
        elif ef.field_type.value == "string" or ef.field_type.value == "text":
            te = QLineEdit()
            te.setText(str(val.get(ef.name, "")) if isinstance(val, dict) else "")
            te.setStyleSheet(f"""
                QLineEdit {{
                    background: palette(base); color: palette(text);
                    border: 1px solid palette(mid); border-radius: 2px; padding: 2px 4px; font-size: 11px;
                }}
            """)
            def on_change(t, idx=index, pn=prop_name, fn=ef.name):
                items = list(getattr(self._component, pn))
                if idx < len(items) and isinstance(items[idx], dict):
                    items[idx][fn] = t
                    setattr(self._component, pn, items)
            te.textChanged.connect(on_change)
            return te
        elif ef.field_type.value == "vec3":
            val_vec = val.get(ef.name, Vec3(0, 0, 0)) if isinstance(val, dict) else Vec3(0, 0, 0)
            w, sbs = make_vec3_row(ef.label or "", val_vec, lambda: None)
            def on_change(idx=index, pn=prop_name, fn=ef.name, boxes=sbs):
                items = list(getattr(self._component, pn))
                if idx < len(items) and isinstance(items[idx], dict):
                    items[idx][fn] = Vec3(boxes[0].value(), boxes[1].value(), boxes[2].value())
                    setattr(self._component, pn, items)
            for sb in sbs:
                sb.valueChanged.connect(on_change)
            return w
        elif ef.field_type.value == "color":
            from editor.color_picker import ColorLineEdit
            rgb = val.get(ef.name, [1, 1, 1]) if isinstance(val, dict) else [1, 1, 1]
            initial = QColor.fromRgbF(*rgb[:3])
            color_edit = ColorLineEdit(initial)
            color_edit.setFixedSize(*scale_xy(60, 20))
            def on_change(col, idx=index, pn=prop_name, fn=ef.name):
                items = list(getattr(self._component, pn))
                if idx < len(items) and isinstance(items[idx], dict):
                    items[idx][fn] = [col.redF(), col.greenF(), col.blueF()]
                    setattr(self._component, pn, items)
            color_edit.colorChanged.connect(on_change)
            return color_edit
        return None

    def _list_add_item(self, component, prop_name):
        items = list(getattr(component, prop_name, []))
        items.append({})
        setattr(component, prop_name, items)
        self._build_list_field_standalone(
            [f for f in getattr(type(component), "_inspector_fields", lambda: [])() if f.name == prop_name][0],
            prop_name
        )

    def _list_remove_item(self, component, prop_name, index):
        items = list(getattr(component, prop_name, []))
        if 0 <= index < len(items):
            items.pop(index)
            setattr(component, prop_name, items)
        self._build_list_field_standalone(
            [f for f in getattr(type(component), "_inspector_fields", lambda: [])() if f.name == prop_name][0],
            prop_name
        )

    def _list_set_entity(self, component, prop_name, index, field_name, entity_id):
        items = list(getattr(component, prop_name, []))
        if index < len(items) and isinstance(items[index], dict):
            items[index][field_name] = entity_id
            setattr(component, prop_name, items)

    def _get_list_element_fields(self, prop_name):
        from core.components.inspector_meta import InspectorField
        fields = getattr(type(self._component), "_inspector_fields", lambda: [])()
        for f in fields:
            if f.name == prop_name:
                return f.element_fields or []
        return []

    def _on_vec2_changed(self, prop_name: str, spinboxes: list):
        self._updating = True
        self._updating = False

    def refresh_vec2_field(self, prop_name: str):
        pass

    def _on_vec3_changed(self, prop_name: str, spinboxes: list):
        self._updating = True
        self._updating = False

    def _on_vec4_changed(self, prop_name: str, spinboxes: list):
        self._updating = True
        self._updating = False

    def refresh_vec4_field(self, prop_name: str):
        pass

    def _on_gradient_changed(self, comp, prop_name, stops):
        pass

    def refresh_vec3_field(self, prop_name: str):
        pass

    def _show_source(self, file_path: str, line_number: int, comp_type: str, prop_name: str):
        from editor.inspector.source_viewer import SourceViewerDialog
        dlg = SourceViewerDialog(file_path, line_number, f"{comp_type}.{prop_name}", self)
        dlg.exec()

    def _build_script_fields(self, comp):
        for field in comp._cached_fields:
            self._build_script_field_from_meta(field, comp)

    def _build_script_field_from_meta(self, field, comp):
        prop_name = field.name
        value = getattr(comp, prop_name, None)
        if field.field_type.value == "float":
            sb = make_spinbox(value or 0.0)
            def _on_float_changed(v, n=prop_name):
                setattr(comp, n, v)
            sb.valueChanged.connect(_on_float_changed)
            self._add_field(field.label or prop_name, sb)
        elif field.field_type.value == "int":
            sb = QSpinBox()
            sb.setRange(-2147483648, 2147483647)
            sb.setValue(int(value or 0))
            def _on_int_changed(v, n=prop_name):
                setattr(comp, n, v)
            sb.valueChanged.connect(_on_int_changed)
            self._add_field(field.label or prop_name, sb)
        elif field.field_type.value == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(value))
            def _on_bool_changed(v, n=prop_name):
                setattr(comp, n, v)
            cb.toggled.connect(_on_bool_changed)
            self._add_field(field.label or prop_name, cb)
        elif field.field_type.value == "str":
            te = QLineEdit()
            te.setText(str(value or ""))
            te.setStyleSheet(f"""
                QLineEdit {{
                    border-radius: {_FUSION_INPUT_RADIUS};
                    padding: 2px 4px;
                    font-size: 11px;
                    selection-background-color: {_accent()};
                }}
                QLineEdit:focus {{ border-color: {_accent()}; }}
            """)
            def _on_str_changed(v, n=prop_name):
                setattr(comp, n, v)
            te.textChanged.connect(_on_str_changed)
            self._add_field(field.label or prop_name, te)
        elif field.field_type.value == "vec2":
            v = value or Vec2()
            w, sbs = make_vec2_row(field.label or "", v, lambda: None)
            def _on_vec2_changed(n=prop_name, sbs_box=sbs):
                setattr(comp, n, Vec2(sbs_box[0].value(), sbs_box[1].value()))
            for sb in sbs:
                sb.valueChanged.connect(_on_vec2_changed)
            self._layout.addWidget(w)
        elif field.field_type.value == "vec3":
            v = value or Vec3()
            w, sbs = make_vec3_row(field.label or "", v, lambda: None)
            def _on_vec3_changed(n=prop_name, sbs_box=sbs):
                setattr(comp, n, Vec3(sbs_box[0].value(), sbs_box[1].value(), sbs_box[2].value()))
            for sb in sbs:
                sb.valueChanged.connect(_on_vec3_changed)
            self._layout.addWidget(w)
        elif field.field_type.value == "enum":
            combo = QComboBox()
            options = field.enum_options or []
            for opt in options:
                combo.addItem(opt)
            try: combo.setCurrentText(str(value or ""))
            except: pass
            def _on_enum_changed(v, n=prop_name):
                setattr(comp, n, v)
            combo.currentTextChanged.connect(_on_enum_changed)
            self._add_field(field.label or prop_name, combo)
        elif field.field_type.value == "resource":
            filter_str = field.file_filter or ""
            def _on_click_r(m=prop_name, cc=comp):
                from editor.resource_picker import pick_resource
                p = pick_resource(self, "Select Resource", filter_str, getattr(cc, m, ""))
                if p:
                    setattr(cc, m, p)
            from PyQt6.QtWidgets import QPushButton, QSizePolicy
            btn = QPushButton(value or f"Pick {field.label}")
            btn.setStyleSheet(f"""
                QPushButton {{
                    border-radius: {_FUSION_INPUT_RADIUS};
                    padding: 2px 6px; font-size: 10px; text-align: left;
                }}
            """)
            btn.clicked.connect(_on_click_r)
            self._add_field(field.label or prop_name, btn)
        elif field.field_type.value == "resource_type":
            from core.components.scripting.script_component import RESOURCE_TYPE_FILTERS
            filter_str = RESOURCE_TYPE_FILTERS.get(field.resource_type, "")
            def _on_click_rt(m=prop_name, cc=comp):
                from editor.resource_picker import pick_resource
                p = pick_resource(self, "Select Resource", filter_str, getattr(cc, m, ""))
                if p:
                    setattr(cc, m, p)
            from PyQt6.QtWidgets import QPushButton, QSizePolicy
            btn = QPushButton(value or f"Pick {field.label}")
            btn.setStyleSheet(f"""
                QPushButton {{
                    border-radius: {_FUSION_INPUT_RADIUS};
                    padding: 2px 6px; font-size: 10px; text-align: left;
                }}
            """)
            btn.clicked.connect(_on_click_rt)
            self._add_field(field.label or prop_name, btn)

    def _on_script_gameobject_changed(self, comp, prop_name, value):
        pass

    def _on_script_resource_changed(self, comp, prop_name, value):
        pass

    def _build_transform(self):
        from editor.inspector.panel import InspectorPanel
        self._transform_refresh = True
        tr = self._component
        w_pos, sbs_pos = make_vec3_row("Position", tr.local_position, lambda: None,
            reset_to=[0.0, 0.0, 0.0])
        for sb in sbs_pos:
            sb.valueChanged.connect(lambda v, sbs_box=sbs_pos: self._update_transform_from_spinboxes())
        self._layout.addWidget(w_pos)
        w_rot, sbs_rot = make_vec3_row("Rotation", tr.local_euler_angles, lambda: None,
            reset_to=[0.0, 0.0, 0.0])
        for sb in sbs_rot:
            sb.valueChanged.connect(lambda v, sbs_box=sbs_rot: self._update_transform_from_spinboxes())
        self._layout.addWidget(w_rot)
        w_scale, sbs_scale = make_vec3_row("Scale", tr.local_scale, lambda: None,
            reset_to=[1.0, 1.0, 1.0])
        for sb in sbs_scale:
            sb.valueChanged.connect(lambda v, sbs_box=sbs_scale: self._update_transform_from_spinboxes())
        self._layout.addWidget(w_scale)

    def _update_transform_from_spinboxes(self):
        from editor.inspector.panel import InspectorPanel
        if self._updating: return
        self._updating = True
        try:
            pass
        finally:
            self._updating = False
