# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, QDoubleSpinBox, QSpinBox, \
    QSlider, QComboBox, QFrame, QMenu, QDialog, QPlainTextEdit, QApplication, QLineEdit, QSizePolicy, QFormLayout, QGroupBox
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData, QSize, QEvent
import qtawesome as qta
from PyQt6.QtGui import QAction, QPixmap, QIcon, QDrag, QCursor, QColor
from core.config.editor_scale import scale, scale_xy
from core.maths.math3d import Vec2, Vec3, Vec4, Quat
from core.foundation.logger import Logger
from core.foundation.commands import SetComponentCommand, CompoundCommand, get_history
from core.foundation.curve import Curve
from editor.curve_editor import CurvePreview, CurveEditorDialog
from core.gui.widgets import AnchorPresetSelector
from core.components.animation.animator_controller import (
    AnimatorController, AnimatorState, AnimatorTransition,
    AnimatorCondition, AnimatorConditionMode,
)
from core.physics.collision_layers import MAX_LAYERS, DEFAULT_LAYER_NAMES
from core.config.config import get_project_config
from editor.inspector.constants import (_FUSION_CARD_RADIUS, _FUSION_INPUT_RADIUS,
    _COMPONENT_MIME, _accent, _window_text, _base, _mid, _alternate)
from editor.inspector.helpers import (make_spinbox, make_clickable_label, get_component_icon_pixmap,
    get_component_source_path, get_property_line_number, collapse_value, make_resource_picker,
    make_gameobject_picker, make_resource_type_picker, make_asset_picker, make_vec2_row,
    make_vec3_row, make_vec4_row, make_vec2_slider_row, make_vec3_slider_row)

class ComponentWidget(QWidget):
    remove_requested = pyqtSignal(str, str)
    move_up_requested = pyqtSignal(str)
    move_down_requested = pyqtSignal(str)
    reorder_requested = pyqtSignal(str, str, str)

    def __init__(self, component, entity=None, selected_entities=None, parent=None, component_key: str = "", overridden_props: set = None):
        super().__init__(parent)
        self._component = component
        self._entity = entity
        self._selected_entities = list(selected_entities if selected_entities else [])
        self._component_key = component_key
        self._overridden_props = set(overridden_props) if overridden_props else set()
        self._updating = False
        self._collapsed = False
        self._field_rows: list[dict] = []
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self._header_widget = QWidget()
        self._header_widget.setObjectName("compHeader")
        header_layout = QHBoxLayout(self._header_widget)
        header_layout.setContentsMargins(6, 3, 6, 3)
        header_layout.setSpacing(4)
        self._collapse_btn = QPushButton(qta.icon("fa5s.caret-down", color="#d4d4d4"), "")
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
        header_layout.addWidget(self._enabled_cb)
        self._move_up_btn = QPushButton(qta.icon("fa5s.chevron-up", color="#d4d4d4"), "")
        self._move_up_btn.setFixedSize(*scale_xy(16, 16))
        self._move_up_btn.setFlat(True)
        self._move_up_btn.clicked.connect(lambda: self.move_up_requested.emit(self._component_key))
        header_layout.addWidget(self._move_up_btn)
        self._move_down_btn = QPushButton(qta.icon("fa5s.chevron-down", color="#d4d4d4"), "")
        self._move_down_btn.setFixedSize(*scale_xy(16, 16))
        self._move_down_btn.setFlat(True)
        self._move_down_btn.clicked.connect(lambda: self.move_down_requested.emit(self._component_key))
        header_layout.addWidget(self._move_down_btn)
        self._header_widget.installEventFilter(self)
        main_layout.addWidget(self._header_widget)
        self._content_widget = QWidget()
        self._content_widget.setObjectName("compBody")
        self._layout = QFormLayout(self._content_widget)
        self._layout.setContentsMargins(8, 6, 8, 6)
        self._layout.setSpacing(3)
        self._layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._layout.setHorizontalSpacing(8)
        self._group_form: QFormLayout | None = None
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
        self._collapse_btn.setIcon(qta.icon("fa5s.caret-right" if self._collapsed else "fa5s.caret-down", color="#d4d4d4"))
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
                from core.engine.engine import Engine
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
        from core.ecs.ecs import ComponentRegistry
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
        self._wire_validation()

    def _on_toggle_changed(self, v: bool, rows: list[QWidget]):
        for r in rows:
            r.setVisible(v)

    def _on_group_toggled(self, gform: QFormLayout, visible: bool):
        for i in range(gform.count()):
            item = gform.itemAt(i)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.setVisible(visible)

    def _resolve_default(self, prop_name: str, suggested):
        if suggested is not None:
            return suggested
        f = self._current_field
        if f is not None and f.field_type.value in ("float", "int", "slider", "int_slider"):
            return 0
        if f is not None and f.field_type.value == "bool":
            return False
        if f is not None and f.field_type.value in ("string", "text", "textarea", "keybinding"):
            return ""
        return None

    def _reset_to_default(self, prop_name: str, suggested=None):
        default = self._resolve_default(prop_name, suggested)
        if default is None:
            return
        c = self._component
        old = getattr(c, prop_name, None)
        setattr(c, prop_name, default)
        get_history().execute(SetComponentCommand(self._entity, type(c), prop_name, old, default))
        self._rebuild_fields()

    def _copy_value(self, prop_name: str):
        try:
            val = getattr(self._component, prop_name)
            from core.foundation.commands import collapse_value as _cv
            QApplication.clipboard().setText(str(_cv(val)))
        except Exception:
            pass

    def _paste_value(self, prop_name: str):
        text = QApplication.clipboard().text()
        if not text:
            return
        c = self._component
        old = getattr(c, prop_name, None)
        f = self._current_field
        try:
            if f is not None and f.field_type.value == "int":
                v = int(float(text))
            elif f is not None and f.field_type.value in ("float", "slider", "int_slider"):
                v = float(text)
            elif f is not None and f.field_type.value == "bool":
                v = text.strip().lower() in ("1", "true", "yes")
            else:
                v = text
            setattr(c, prop_name, v)
            get_history().execute(SetComponentCommand(self._entity, type(c), prop_name, old, v))
            self._rebuild_fields()
        except Exception:
            pass

    def _show_field_context_menu(self, pos, prop_name: str):
        menu = QMenu(self)
        copy_act = QAction("Copy Value", self)
        copy_act.triggered.connect(lambda: self._copy_value(prop_name))
        menu.addAction(copy_act)
        paste_act = QAction("Paste Value", self)
        paste_act.triggered.connect(lambda: self._paste_value(prop_name))
        menu.addAction(paste_act)
        reset_act = QAction("Reset to Default", self)
        reset_act.triggered.connect(lambda: self._reset_to_default(prop_name))
        menu.addAction(reset_act)
        f = self._current_field
        if f is not None and f.field_type.value in ("float", "vec2", "vec3", "vec4"):
            menu.addSeparator()
            key_act = QAction("Add Keyframe", self)
            key_act.triggered.connect(lambda: Logger.info(f"Keyframe requested for {prop_name}"))
            menu.addAction(key_act)
        menu.exec(self.sender().mapToGlobal(pos))

    def set_filter(self, text: str):
        text = text.strip().lower()
        for row in self._field_rows:
            if not text:
                row["label_widget"].setVisible(True)
                row["field_widget"].setVisible(True)
                continue
            match = text in row["label"].lower() or (row["prop_name"] and text in row["prop_name"].lower())
            row["label_widget"].setVisible(match)
            row["field_widget"].setVisible(match)

    def _validate_field(self, prop_name: str, widget: QWidget, value):
        f = self._current_field
        bad = False
        if f is not None and f.field_type.value in ("float", "int", "slider", "int_slider"):
            try:
                if value is None or (isinstance(value, float) and value != value):
                    bad = True
                elif hasattr(f, "min_val") and f.min_val > -1e15 and value < f.min_val:
                    bad = True
                elif hasattr(f, "max_val") and f.max_val < 1e15 and value > f.max_val:
                    bad = True
            except Exception:
                pass
        if bad:
            widget.setStyleSheet("border: 1px solid #f44747; border-radius: 2px;")
            widget.setToolTip("Value out of range or invalid")
        else:
            widget.setStyleSheet("")
            widget.setToolTip("")

    def _wire_validation(self):
        from PyQt6.QtWidgets import QAbstractSpinBox, QSlider
        for row in self._field_rows:
            f = row.get("field_meta")
            if f is None or f.field_type.value not in ("float", "int", "slider", "int_slider"):
                continue
            w = row["field_widget"]
            target = w._field_child if hasattr(w, "_field_child") else w
            prop = row["prop_name"]
            if isinstance(target, QAbstractSpinBox):
                target.valueChanged.connect(lambda v, p=prop, tw=target: self._validate_field(p, tw, v))
            elif isinstance(target, QSlider):
                target.valueChanged.connect(lambda v, p=prop, tw=target: self._validate_field(p, tw, v))

    def _rebuild_fields(self):
        self._group_form = None
        self._field_rows.clear()
        self._toggle_rows.clear()
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._build_fields()

    def _sync_int_field_values(self):
        c = self._component
        for r in self._field_rows:
            pn = r.get("prop_name", "")
            if not pn or not hasattr(c, pn):
                continue
            cell = r.get("field_widget")
            if cell is None:
                continue
            child = getattr(cell, "_field_child", None)
            if isinstance(child, QSpinBox):
                child.blockSignals(True)
                child.setValue(getattr(c, pn, child.value()))
                child.blockSignals(False)

    def _target_layout(self) -> QFormLayout:
        return self._group_form if self._group_form is not None else self._layout

    def _make_field_cell(self, widget: QWidget, prop_name: str, field_meta=None):
        cell = QWidget()
        fl = QHBoxLayout(cell)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(4)
        fl.addWidget(widget, 1)
        cell._field_child = widget
        if prop_name and field_meta is not None and field_meta.default_value is not None:
            reset_btn = QPushButton("\u21ba")
            reset_btn.setFixedSize(*scale_xy(18, 18))
            reset_btn.setToolTip("Reset to default")
            reset_btn.setStyleSheet("QPushButton { color: #bbb; background: transparent; border: 1px solid #444; border-radius: 3px; font-size: 10px; } QPushButton:hover { color: #fff; border-color: #777; }")
            reset_btn.clicked.connect(lambda _, pn=prop_name, dv=field_meta.default_value: self._reset_to_default(pn, dv))
            fl.addWidget(reset_btn)
            cell._reset_btn = reset_btn
        if prop_name:
            comp_type = type(self._component).__name__
            src_path = get_component_source_path(type(self._component))
            line_num = get_property_line_number(type(self._component), prop_name)
            source_lbl = make_clickable_label("src", lambda sp=src_path, ln=line_num: self._show_source(sp, ln, comp_type, prop_name))
            fl.addWidget(source_lbl)
        return cell

    def _add_field(self, label: str, widget: QWidget, prop_name: str = "", toggle_field: str = "", field_meta=None):
        if field_meta is None:
            field_meta = getattr(self, "_current_field", None)
        target = self._target_layout()
        lbl = QLabel(label)
        lbl.setWordWrap(True)
        if prop_name:
            lbl.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            lbl.customContextMenuRequested.connect(lambda pos, pn=prop_name: self._show_field_context_menu(pos, pn))
        if field_meta is not None and field_meta.description:
            lbl.setToolTip(field_meta.description)
        if prop_name and prop_name in self._overridden_props:
            lbl.setStyleSheet("color: #e0a93b; font-weight: 600;")
        if prop_name and len(self._selected_entities) > 1:
            vals = []
            for e in self._selected_entities:
                comp = e.get_component(type(self._component))
                if comp is not None and hasattr(comp, prop_name):
                    vals.append(getattr(comp, prop_name))
            if len(vals) > 1 and any(v != vals[0] for v in vals[1:]):
                base_tip = field_meta.description if field_meta is not None and field_meta.description else ""
                lbl.setToolTip((base_tip + "\n" if base_tip else "") + "Mixed values across selection")
                style = lbl.styleSheet() or ""
                if prop_name not in self._overridden_props:
                    lbl.setStyleSheet(style + " color: #7fb0ff;")
        cell = self._make_field_cell(widget, prop_name, field_meta) if (prop_name or field_meta is not None) else None
        if label:
            target.addRow(lbl, cell if cell is not None else widget)
        else:
            target.addRow(cell if cell is not None else widget)
        self._field_rows.append({
            "label": label,
            "label_widget": lbl,
            "field_widget": cell if cell is not None else widget,
            "prop_name": prop_name,
            "field_meta": field_meta,
            "toggle_field": toggle_field,
        })
        if toggle_field:
            self._toggle_rows.setdefault(toggle_field, []).append(cell if cell is not None else widget)

    def _enum_spec(self, field, value):
        options = list(field.enum_options or [])
        enum_cls = field.enum_class
        if not options and enum_cls is not None:
            options = [e.name for e in enum_cls]
        current = ""
        if value is not None:
            if enum_cls is not None and isinstance(value, enum_cls):
                current = value.name
            elif enum_cls is not None and isinstance(value, str):
                if value in enum_cls.__members__:
                    current = enum_cls[value].name
                else:
                    for e in enum_cls:
                        if e.value == value:
                            current = e.name
                            break
                    else:
                        current = value
            else:
                current = str(value)
        return options, enum_cls, current

    def _enum_value(self, enum_cls, text):
        return enum_cls[text] if enum_cls is not None else text

    def _build_field_from_meta(self, field):
        c = self._component
        self._current_field = field
        prop_name = field.name
        if field.field_type.value == "header":
            group = QGroupBox(field.label)
            group.setObjectName("inspectorGroup")
            group.setCheckable(True)
            group.setChecked(True)
            group.setStyleSheet(
                "QGroupBox#inspectorGroup { color: #ccc; border: 1px solid #3a3a3a; "
                "border-radius: 4px; margin-top: 10px; padding-top: 16px; font-size: 11px; "
                "font-weight: 600; background: rgba(255,255,255,0.02); } "
                "QGroupBox#inspectorGroup::title { subcontrol-origin: margin; left: 8px; "
                "padding: 0 4px; } "
                "QGroupBox#inspectorGroup::indicator { width: 10px; height: 10px; }"
            )
            gform = QFormLayout(group)
            gform.setContentsMargins(8, 4, 8, 6)
            gform.setSpacing(3)
            gform.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            gform.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            gform.setHorizontalSpacing(8)
            group.toggled.connect(lambda v, gf=gform: self._on_group_toggled(gf, v))
            self._group_form = gform
            self._layout.addWidget(group)
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
                natural_min = min_i
                natural_max = max_i
                if field.on_set and hasattr(c, field.on_set):
                    min_i = max(-2147483648, min_i - 1)
                    max_i = min(2147483647, max_i + 1)
                sb.setRange(min_i, max_i)
                sb.setValue(max(natural_min, min(natural_max, int(value))))
                sb.setMinimumWidth(60)
                comp_cls = type(c)
                if field.on_set and hasattr(c, field.on_set):
                    def _on_int_value(v, n=prop_name, m=field.on_set, nmn=natural_min, nmx=natural_max):
                        if nmn <= v <= nmx:
                            self._undo_setter_all(comp_cls, n)(v)
                        else:
                            fn = getattr(c, m, None)
                            if callable(fn) and fn(n, v):
                                self._sync_int_field_values()
                    sb.valueChanged.connect(_on_int_value)
                else:
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
            def _on_button():
                fn = getattr(c, prop_name, None)
                if callable(fn):
                    fn()
                self._sync_int_field_values()
            btn.clicked.connect(_on_button)
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
            options, enum_cls, current = self._enum_spec(field, value)
            combo = QComboBox()
            for opt in options:
                combo.addItem(opt)
            try:
                combo.setCurrentText(current)
            except Exception:
                pass
            comp_cls = type(c)
            def _on_enum_change(t):
                nv = self._enum_value(enum_cls, t)
                setattr(c, prop_name, nv)
                get_history().execute(SetComponentCommand(self._entity, comp_cls, prop_name, value, nv))
            combo.currentTextChanged.connect(_on_enum_change)
            self._add_field(field.label, combo, prop_name, field.toggle_field)
        elif field.field_type.value == "resource":
            pw = make_resource_picker(value, field.file_filter or "All Files (*)", self._undo_setter(prop_name))
            self._add_field(field.label, pw, prop_name, field.toggle_field)
        elif field.field_type.value == "gameobject":
            from core.engine.engine import Engine
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
            preview.setMinimumHeight(scale(56))
            preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if isinstance(value, Curve):
                preview.set_curve(value)
            from PyQt6.QtWidgets import QColorDialog
            def _open_editor(*, _prop_name=prop_name, _field=field):
                dlg = CurveEditorDialog(getattr(c, _prop_name, None), field.label, self)
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
                te.textChanged.connect(lambda: setattr(c, prop_name, te.toPlainText()))
                comp_cls = type(c)
                te.focusOutEvent = lambda ev: (get_history().execute(SetComponentCommand(self._entity, comp_cls, prop_name, value, te.toPlainText())), QPlainTextEdit.focusOutEvent(te, ev))
            else:
                te = QLineEdit()
                te.setText(str(value) if value else "")
                te.textChanged.connect(lambda: setattr(c, prop_name, te.text()))
                comp_cls = type(c)
                te.editingFinished.connect(lambda: get_history().execute(SetComponentCommand(self._entity, comp_cls, prop_name, value, te.text())))
            self._add_field(field.label, te, prop_name, field.toggle_field)
        elif field.field_type.value == "vec2":
            w, sbs = make_vec2_row(field.label, value if value is not None else Vec2(0.0, 0.0), lambda: None)
            comp_cls = type(c)
            for sb in sbs:
                sb.valueChanged.connect(lambda v, pn=prop_name, sbs_box=sbs, cls=comp_cls: self._on_vec2_changed(pn, sbs_box))
            for i, sb in enumerate(sbs):
                def make_setter(idx):
                    def setter(v):
                        vec = getattr(c, prop_name)
                        if vec is None:
                            vec = Vec2(0.0, 0.0)
                        if isinstance(vec, (list, tuple)):
                            lst = [float(vec[0]), float(vec[1])]
                        else:
                            lst = [vec.x, vec.y]
                        lst[idx] = v
                        if isinstance(vec, (list, tuple)):
                            setattr(c, prop_name, list(lst))
                        else:
                            setattr(c, prop_name, type(vec)(*lst))
                    return setter
                sb.valueChanged.connect(make_setter(i))
            self._target_layout().addWidget(w)
        elif field.field_type.value == "vec3":
            w, sbs = make_vec3_row(field.label, value if value is not None else Vec3(0.0, 0.0, 0.0), lambda: None)
            for i, sb in enumerate(sbs):
                def make_setter(idx):
                    def setter(v):
                        vec = getattr(c, prop_name)
                        if vec is None:
                            vec = Vec3(0.0, 0.0, 0.0)
                        if isinstance(vec, (list, tuple)):
                            lst = [float(vec[0]), float(vec[1]), float(vec[2])]
                        else:
                            lst = [vec.x, vec.y, vec.z]
                        lst[idx] = v
                        if isinstance(vec, (list, tuple)):
                            setattr(c, prop_name, list(lst))
                        else:
                            setattr(c, prop_name, type(vec)(*lst))
                    return setter
                sb.valueChanged.connect(make_setter(i))
            self._target_layout().addWidget(w)
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
            from core.engine.engine import Engine
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
            w, sbs = make_vec4_row(field.label, value if value is not None else Vec4(0.0, 0.0, 0.0, 0.0), lambda: None)
            for i, sb in enumerate(sbs):
                def make_setter(idx):
                    def setter(v):
                        vec = getattr(c, prop_name)
                        if vec is None:
                            vec = Vec4(0.0, 0.0, 0.0, 0.0)
                        lst = [vec.x, vec.y, vec.z, vec.w]
                        lst[idx] = v
                        setattr(c, prop_name, type(vec)(*lst))
                    return setter
                sb.valueChanged.connect(make_setter(i))
            self._target_layout().addWidget(w)
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
            self._target_layout().addWidget(w)
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
            self._target_layout().addWidget(w)
        elif field.field_type.value == "gradient":
            grad_preview = QPushButton()
            grad_preview.setFixedHeight(22)
            grad_preview.setMinimumWidth(120)
            grad_preview.setToolTip("Click to edit gradient")
            def _paint_gradient(btn, grad_data=value):
                if not grad_data:
                    return
                if isinstance(grad_data, dict):
                    alpha_keys = grad_data.get("alpha_keys", [])
                    color_keys = grad_data.get("color_keys", [])
                    alpha_map = {k[0]: k[1] for k in alpha_keys}
                    color_map = {k[0]: list(k[1]) for k in color_keys}
                    positions = sorted(set(list(alpha_map.keys()) + list(color_map.keys())))
                    grad_data = []
                    for p in positions:
                        c = color_map.get(p, [1, 1, 1])
                        a = alpha_map.get(p, 1.0)
                        grad_data.append((p, [c[0], c[1], c[2], a]))
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
                editor_stops = value
                if isinstance(value, dict):
                    alpha_keys = value.get("alpha_keys", [])
                    color_keys = value.get("color_keys", [])
                    alpha_map = {k[0]: k[1] for k in alpha_keys}
                    color_map = {k[0]: list(k[1]) for k in color_keys}
                    positions = sorted(set(list(alpha_map.keys()) + list(color_map.keys())))
                    editor_stops = []
                    for p in positions:
                        c = color_map.get(p, [1, 1, 1])
                        a = alpha_map.get(p, 1.0)
                        editor_stops.append((p, [c[0], c[1], c[2], a]))
                dlg = GradientEditorDialog(editor_stops, "Edit Gradient", self)
                if dlg.exec() == QDialog.DialogCode.Accepted:
                    new_gradient = dlg.get_stops()
                    if isinstance(value, dict):
                        alpha_keys = []
                        color_keys = []
                        for p, rgba in new_gradient:
                            color_keys.append([p, [rgba[0], rgba[1], rgba[2]]])
                            alpha_keys.append([p, rgba[3]])
                        new_gradient = {"alpha_keys": alpha_keys, "color_keys": color_keys}
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
        self._list_submesh_names = self._get_submesh_names()
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
            QPushButton {{ color: {_accent()}; font-size: 10px; background: transparent; border: none; }}
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
        self._target_layout().addWidget(container)

    def _list_item_label(self, prop_name, index):
        if type(self._component).__name__ == "MeshRenderer" and prop_name == "materials":
            names = getattr(self, "_list_submesh_names", None)
            if names is None:
                names = self._get_submesh_names()
            if names and len(names) > 1:
                return names[index] if (index < len(names) and names[index]) else f"Submesh {index}"
        return ""

    def _list_min_length(self, prop_name):
        if type(self._component).__name__ == "MeshRenderer" and prop_name == "materials":
            names = getattr(self, "_list_submesh_names", None)
            if names is None:
                names = self._get_submesh_names()
            if names and len(names) > 1:
                return len(names)
        return 0

    def _get_submesh_names(self):
        if self._entity is None:
            return []
        try:
            from core.components.rendering.renderers.mesh_filter import MeshFilter
            from core.engine.engine import Engine
            eng = Engine.instance()
            r = getattr(eng, '_renderer', None)
            if r is None:
                vp = getattr(eng, 'viewport', None)
                r = getattr(vp, '_renderer', None)
            if r is None or r._mesh_loader is None:
                return []
            mf = self._entity.get_component(MeshFilter)
            if mf is None:
                return []
            key = mf.mesh_path
            if not key:
                return []
            meshes = r._mesh_loader._meshes
            mesh = meshes.get(key)
            if mesh is None:
                prefix = key + "|"
                for k, v in meshes.items():
                    if k.startswith(prefix):
                        mesh = v
                        break
            if mesh is not None:
                names = getattr(mesh, 'sub_mesh_names', None)
                if names:
                    return list(names)
        except Exception:
            pass
        return []

    def _build_list_row_standalone(self, prop_name, index, elem_data, element_fields):
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)
        remove_btn = QPushButton("\u2212")
        remove_btn.setFixedSize(*scale_xy(16, 16))
        remove_btn.setStyleSheet(f"""
            QPushButton {{ color: {_window_text()}; font-size: 10px; background: transparent; border: none; }}
        """)
        remove_btn.clicked.connect(lambda: self._list_remove_item(self._component, prop_name, index))
        rl.addWidget(remove_btn)
        item_label = self._list_item_label(prop_name, index)
        if item_label:
            lbl = QLabel(item_label)
            lbl.setWordWrap(True)
            lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            rl.addWidget(lbl, 0)
        elem_widget = QWidget()
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
            from core.engine.engine import Engine
            scene = Engine.instance().scene
            eid = val.get(ef.name, "") if isinstance(val, dict) else ""
            def on_entity(eid, idx=index, pn=prop_name, fn=ef.name):
                items = list(getattr(self._component, pn))
                if idx < len(items) and isinstance(items[idx], dict):
                    items[idx][fn] = eid
                    setattr(self._component, pn, items)
            return make_gameobject_picker(eid, scene, on_entity)
        elif ef.field_type.value == "enum":
            raw = val.get(ef.name, "") if isinstance(val, dict) else ""
            options, enum_cls, current = self._enum_spec(ef, raw)
            combo = QComboBox()
            for opt in options:
                combo.addItem(opt)
            try: combo.setCurrentText(current)
            except: pass
            def on_change(t, idx=index, pn=prop_name, fn=ef.name):
                nv = self._enum_value(enum_cls, t)
                items = list(getattr(self._component, pn))
                if idx < len(items) and isinstance(items[idx], dict):
                    items[idx][fn] = nv
                    setattr(self._component, pn, items)
            combo.currentTextChanged.connect(on_change)
            return combo
        elif ef.field_type.value == "resource" or ef.field_type.value == "resource_path":
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
        self._current_field = field
        value = comp.get_field_value(prop_name)
        if field.field_type.value == "float":
            sb = make_spinbox(value if isinstance(value, (int, float)) else 0.0)
            def _on_float_changed(v, n=prop_name):
                comp.set_field_value(n, v)
            sb.valueChanged.connect(_on_float_changed)
            self._add_field(field.label or prop_name, sb)
        elif field.field_type.value == "int":
            sb = QSpinBox()
            sb.setRange(-2147483648, 2147483647)
            sb.setValue(int(value) if isinstance(value, (int, float)) else 0)
            def _on_int_changed(v, n=prop_name):
                comp.set_field_value(n, v)
            sb.valueChanged.connect(_on_int_changed)
            self._add_field(field.label or prop_name, sb)
        elif field.field_type.value == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(value))
            def _on_bool_changed(v, n=prop_name):
                comp.set_field_value(n, v)
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
                comp.set_field_value(n, v)
            te.textChanged.connect(_on_str_changed)
            self._add_field(field.label or prop_name, te)
        elif field.field_type.value == "vec2":
            v = value if isinstance(value, Vec2) else Vec2()
            w, sbs = make_vec2_row(field.label or "", v, lambda: None)
            def _on_vec2_changed(n=prop_name, sbs_box=sbs):
                comp.set_field_value(n, Vec2(sbs_box[0].value(), sbs_box[1].value()))
            for sb in sbs:
                sb.valueChanged.connect(_on_vec2_changed)
            self._target_layout().addWidget(w)
        elif field.field_type.value == "vec3":
            v = value if isinstance(value, Vec3) else Vec3()
            w, sbs = make_vec3_row(field.label or "", v, lambda: None)
            def _on_vec3_changed(n=prop_name, sbs_box=sbs):
                comp.set_field_value(n, Vec3(sbs_box[0].value(), sbs_box[1].value(), sbs_box[2].value()))
            for sb in sbs:
                sb.valueChanged.connect(_on_vec3_changed)
            self._target_layout().addWidget(w)
        elif field.field_type.value == "enum":
            options, enum_cls, current = self._enum_spec(field, value)
            combo = QComboBox()
            for opt in options:
                combo.addItem(opt)
            try: combo.setCurrentText(current)
            except: pass
            def _on_enum_changed(v, n=prop_name):
                nv = self._enum_value(enum_cls, v)
                comp.set_field_value(n, nv)
            combo.currentTextChanged.connect(_on_enum_changed)
            self._add_field(field.label or prop_name, combo)
        elif field.field_type.value == "gameobject":
            btn = QPushButton(str(value or "") or f"Pick {field.label}")
            btn.setStyleSheet(f"""
                QPushButton {{
                    border-radius: {_FUSION_INPUT_RADIUS};
                    padding: 2px 6px; font-size: 10px; text-align: left;
                }}
            """)
            def _on_click_go(m=prop_name, cc=comp):
                from editor.entity_picker import pick_entity
                eid = pick_entity(self, cc.get_field_value(m) or "")
                if eid:
                    cc.set_field_value(m, eid)
                    btn.setText(str(eid))
            btn.clicked.connect(_on_click_go)
            self._add_field(field.label or prop_name, btn)
        elif field.field_type.value == "resource":
            filter_str = field.file_filter or ""
            btn = QPushButton(str(value or "") or f"Pick {field.label}")
            btn.setStyleSheet(f"""
                QPushButton {{
                    border-radius: {_FUSION_INPUT_RADIUS};
                    padding: 2px 6px; font-size: 10px; text-align: left;
                }}
            """)
            def _on_click_r(m=prop_name, cc=comp, fs=filter_str):
                from editor.resource_picker import pick_resource
                p = pick_resource(self, "Select Resource", fs, cc.get_field_value(m) or "")
                if p:
                    cc.set_field_value(m, p)
                    btn.setText(str(p))
            btn.clicked.connect(_on_click_r)
            self._add_field(field.label or prop_name, btn)
        elif field.field_type.value == "resource_type":
            from core.components.scripting.script_component import RESOURCE_TYPE_FILTERS
            filter_str = RESOURCE_TYPE_FILTERS.get(field.resource_type, "")
            btn = QPushButton(str(value or "") or f"Pick {field.label}")
            btn.setStyleSheet(f"""
                QPushButton {{
                    border-radius: {_FUSION_INPUT_RADIUS};
                    padding: 2px 6px; font-size: 10px; text-align: left;
                }}
            """)
            def _on_click_rt(m=prop_name, cc=comp, fs=filter_str):
                from editor.resource_picker import pick_resource
                p = pick_resource(self, "Select Resource", fs, cc.get_field_value(m) or "")
                if p:
                    cc.set_field_value(m, p)
                    btn.setText(str(p))
            btn.clicked.connect(_on_click_rt)
            self._add_field(field.label or prop_name, btn)
        elif field.field_type.value == "curve":
            from editor.curve_editor import CurvePreview, CurveEditorDialog
            preview = CurvePreview()
            preview.setMinimumHeight(scale(56))
            preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            cv = comp.get_field_value(prop_name)
            if isinstance(cv, Curve):
                preview.set_curve(cv)
            def _open_curve_editor(_pn=prop_name, _preview=preview, _label=field.label):
                cur = comp.get_field_value(_pn)
                edit_curve = cur.copy() if isinstance(cur, Curve) else Curve()
                dlg = CurveEditorDialog(edit_curve, _label, self)
                if dlg.exec() == QDialog.DialogCode.Accepted:
                    comp.set_field_value(_pn, dlg.get_curve())
                    _preview.set_curve(dlg.get_curve())
            preview.mousePressEvent = lambda ev, pe=_open_curve_editor: pe() if ev.button() == Qt.MouseButton.LeftButton else None
            self._add_field(field.label or prop_name, preview)

    def _on_script_gameobject_changed(self, comp, prop_name, value):
        pass

    def _on_script_resource_changed(self, comp, prop_name, value):
        pass

    def _build_transform(self):
        tr = self._component
        w_pos, sbs_pos = make_vec3_row("Position", tr.local_position,
            lambda v: self._update_transform_from_spinboxes("local_position", sbs_pos),
            reset_to=[0.0, 0.0, 0.0])
        self._tr_pos_sbs = sbs_pos
        self._target_layout().addWidget(w_pos)
        w_rot, sbs_rot = make_vec3_row("Rotation", tr.local_euler_angles,
            lambda v: self._update_transform_from_spinboxes("local_euler_angles", sbs_rot),
            reset_to=[0.0, 0.0, 0.0])
        self._tr_rot_sbs = sbs_rot
        self._target_layout().addWidget(w_rot)
        w_scale, sbs_scale = make_vec3_row("Scale", tr.local_scale,
            lambda v: self._update_transform_from_spinboxes("local_scale", sbs_scale),
            reset_to=[1.0, 1.0, 1.0])
        self._tr_scale_sbs = sbs_scale
        self._target_layout().addWidget(w_scale)

    def _update_transform_from_spinboxes(self, attr, sbs):
        if self._updating: return
        tr = self._component
        new = Vec3(sbs[0].value(), sbs[1].value(), sbs[2].value())
        old = getattr(tr, attr)
        get_history().execute(SetComponentCommand(self._entity, type(tr), attr, old, new))
        self._redraw_viewport()

    def refresh_transform(self):
        if self._updating: return
        tr = self._component
        pos = tr.local_position
        rot = tr.local_euler_angles
        scl = tr.local_scale
        self._updating = True
        try:
            self._set_transform_sb(self._tr_pos_sbs, pos.x, pos.y, pos.z)
            self._set_transform_sb(self._tr_rot_sbs, rot.x, rot.y, rot.z)
            self._set_transform_sb(self._tr_scale_sbs, scl.x, scl.y, scl.z)
        finally:
            self._updating = False

    @staticmethod
    def _set_transform_sb(sbs, x, y, z):
        for sb, val in zip(sbs, (x, y, z)):
            if sb is None:
                continue
            if sb.hasFocus() or (sb.lineEdit() is not None and sb.lineEdit().hasFocus()):
                continue
            sb.setValue(float(val))

    def _redraw_viewport(self):
        try:
            from core.engine.engine import Engine
            vp = Engine.instance().viewport
            if vp is not None:
                vp.update_scene()
        except Exception:
            pass
