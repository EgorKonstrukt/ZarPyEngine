# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
from typing import Optional, Callable
from PyQt6.QtWidgets import QLabel, QWidget, QHBoxLayout, QPushButton, QDoubleSpinBox, QSlider, QDialog, QFileDialog, QInputDialog, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont, QPainter, QColor, QBrush, QPen, QFont as QF
from core.editor_scale import scale, scale_xy
from editor.inspector.constants import _FUSION_BG, _FUSION_BG_INPUT, _FUSION_BORDER, _FUSION_BORDER_LIGHT, _FUSION_TEXT, _FUSION_TEXT_DIM, _FUSION_TEXT_BRIGHT, _FUSION_ACCENT_RED, _FUSION_BG_HOVER, _XYZ_COLORS, _FUSION_INPUT_RADIUS, _accent
from editor.inspector.widgets import _FocusSpinBox, _DragLabel, _ResourceDropLabel, _EntityDropLabel, EntityPickerDialog
from core.math3d import Vec2, Vec3, Vec4

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

_FUSION_SPINBOX_STYLE = f"""
    QDoubleSpinBox, QSpinBox {{
        background: {_FUSION_BG_INPUT};
        color: {_FUSION_TEXT_BRIGHT};
        border: 1px solid {_FUSION_BORDER};
        border-radius: {_FUSION_INPUT_RADIUS};
        padding: 1px 2px 1px 4px;
        font-size: 11px;
        min-height: 20px;
        selection-background-color: {_accent()};
    }}
    QDoubleSpinBox:hover, QSpinBox:hover {{
        border-color: {_FUSION_BORDER_LIGHT};
    }}
    QDoubleSpinBox:focus, QSpinBox:focus {{
        border-color: {_accent()};
    }}
    QDoubleSpinBox::up-button, QSpinBox::up-button {{
        border: none;
        background: transparent;
        width: 12px;
        subcontrol-origin: border;
        subcontrol-position: top right;
    }}
    QDoubleSpinBox::down-button, QSpinBox::down-button {{
        border: none;
        background: transparent;
        width: 12px;
        subcontrol-origin: border;
        subcontrol-position: bottom right;
    }}
    QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
        width: 6px;
        height: 6px;
        border: none;
    }}
    QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
        width: 6px;
        height: 6px;
        border: none;
    }}
"""

def make_spinbox(val: float, lo: float = -1e9, hi: float = 1e9, step: float = 0.1, decimals: int = 4) -> QDoubleSpinBox:
    sb = _FocusSpinBox()
    sb.setRange(lo, hi)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    sb.setValue(val)
    sb.setMinimumWidth(60)
    sb.setStyleSheet(_FUSION_SPINBOX_STYLE)
    return sb

def make_clickable_label(text: str, on_click: Callable[[], None]) -> QLabel:
    lbl = QLabel(f"  {text}")
    lbl.setStyleSheet(f"""
        QLabel {{
            color: {_accent()};
            font-size: 9px;
            padding: 0px;
        }}
        QLabel:hover {{
            color: #8abbff;
        }}
    """)
    lbl.setCursor(Qt.CursorShape.PointingHandCursor)
    lbl.setToolTip("Click to view source code")
    lbl.mousePressEvent = lambda e: on_click()
    return lbl

def get_component_icon_pixmap(cls, size: int = 16) -> QPixmap:
    icon_name = getattr(cls, '_icon', None)
    if icon_name:
        icons_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'core', 'components', 'icons')
        icon_path = os.path.join(icons_dir, icon_name)
        if os.path.exists(icon_path):
            return QPixmap(icon_path).scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    gizmo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gizmo_icons')
    icon_path = os.path.join(gizmo_dir, f'{cls.__name__}.png')
    if os.path.exists(icon_path):
        return QPixmap(icon_path).scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    r, g, b = getattr(cls, '_gizmo_icon_color', (140, 60, 200))
    label = getattr(cls, '_gizmo_icon_label', '?')
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    from PyQt6.QtGui import QPainter, QColor as QC, QFont as QF, QBrush as QB, QPen
    from PyQt6.QtCore import QRect
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QB(QC(r, g, b)))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(0, 0, size, size, 3, 3)
    if label:
        p.setPen(QC(255, 255, 255))
        f = QF("Segoe UI", size // 2, QF.Weight.Bold)
        p.setFont(f)
        p.drawText(QRect(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, label[0].upper())
    p.end()
    return pix

def update_resource_icon(icon_lbl: QLabel, path: str, size: int):
    if path and os.path.exists(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tga"):
            pix = QPixmap(path).scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            icon_lbl.setPixmap(pix)
            return
    icon_lbl.clear()
    icon_lbl.setText("")

def make_resource_picker(path: str, filter_str: str, callback: Callable[[str], None]) -> QWidget:
    from editor.resource_picker import pick_resource
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    name = os.path.basename(path) if path else ""
    icon_lbl = QLabel()
    icon_lbl.setFixedSize(*scale_xy(20, 20))
    icon_lbl.setStyleSheet(f"border: 1px solid {_FUSION_BORDER_LIGHT}; border-radius: 2px; background: {_FUSION_BG};")
    update_resource_icon(icon_lbl, path, 20)
    layout.addWidget(icon_lbl)
    def _on_resource_drop(p: str):
        _update_display(p)
    name_lbl = _ResourceDropLabel(_on_resource_drop, name if name else "None")
    name_lbl.setStyleSheet(
        f"color: {_FUSION_TEXT}; background: {_FUSION_BG_INPUT}; border: 1px solid {_FUSION_BORDER}; border-radius: {_FUSION_INPUT_RADIUS}; padding: 2px 6px;"
    )
    name_lbl.setMinimumHeight(22)
    name_lbl.setToolTip(path if path else "No resource selected")
    layout.addWidget(name_lbl, 1)
    def _update_display(p: str):
        nonlocal name
        new_name = os.path.basename(p) if p else ""
        name_lbl.setText(new_name if new_name else "None")
        name_lbl.setToolTip(p if p else "No resource selected")
        update_resource_icon(icon_lbl, p, 20)
        clear_btn.setVisible(bool(p))
        callback(p)
    btn = QPushButton("\u25CB")
    btn.setFixedSize(*scale_xy(22, 22))
    btn.setToolTip("Pick Resource")
    btn.setStyleSheet(f"""
        QPushButton {{ color: {_FUSION_TEXT_DIM}; border: 1px solid {_FUSION_BORDER_LIGHT}; border-radius: 11px;
        background: {_FUSION_BG_INPUT}; font-size: 14px; }}
        QPushButton:hover {{ background: {_FUSION_BG_HOVER}; color: {_FUSION_TEXT_BRIGHT}; }}
    """)
    def _pick():
        p = pick_resource(w, "Select Resource", filter_str, path)
        if p:
            _update_display(p)
    btn.clicked.connect(_pick)
    layout.addWidget(btn)
    clear_btn = QPushButton("x")
    clear_btn.setFixedSize(*scale_xy(20, 20))
    clear_btn.setToolTip("Clear")
    clear_btn.setStyleSheet(f"""
        QPushButton {{ color: {_FUSION_TEXT_DIM}; border: none; border-radius: {_FUSION_INPUT_RADIUS}; font-size: 10px; background: transparent; }}
        QPushButton:hover {{ color: {_FUSION_ACCENT_RED}; background: #3a1a1a; }}
    """)
    def _clear():
        _update_display("")
    clear_btn.clicked.connect(_clear)
    clear_btn.setVisible(bool(path))
    layout.addWidget(clear_btn)
    return w

def make_gameobject_picker(entity_id: str, scene, callback: Callable[[str], None]) -> QWidget:
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    target_entity = scene.get_entity(entity_id) if scene and entity_id else None
    name = target_entity.name if target_entity else ""
    icon_lbl = QLabel()
    icon_lbl.setFixedSize(*scale_xy(20, 20))
    icon_lbl.setStyleSheet(f"border: 1px solid {_FUSION_BORDER_LIGHT}; border-radius: 2px; background: {_FUSION_BG};")
    icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if target_entity:
        for c in target_entity.get_all_components():
            if getattr(type(c), '_show_gizmo_icon', True) and type(c).__name__ != "Transform":
                pix = get_component_icon_pixmap(type(c), 18)
                icon_lbl.setPixmap(pix)
                break
    layout.addWidget(icon_lbl)
    def _on_entity_drop(eid: str):
        _update_entity_display(eid)
    name_lbl = _EntityDropLabel(_on_entity_drop, name if name else "None")
    name_lbl.setStyleSheet(
        f"color: {_FUSION_TEXT}; background: {_FUSION_BG_INPUT}; border: 1px solid {_FUSION_BORDER}; border-radius: {_FUSION_INPUT_RADIUS}; padding: 2px 6px;"
    )
    name_lbl.setMinimumHeight(22)
    name_lbl.setToolTip(entity_id if entity_id else "No entity selected")
    layout.addWidget(name_lbl, 1)
    def _update_entity_display(eid: str):
        nonlocal target_entity
        target_entity = scene.get_entity(eid) if scene and eid else None
        new_name = target_entity.name if target_entity else ""
        name_lbl.setText(new_name if new_name else "None")
        name_lbl.setToolTip(eid if eid else "No entity selected")
        icon_lbl.clear()
        if target_entity:
            for c in target_entity.get_all_components():
                if getattr(type(c), '_show_gizmo_icon', True) and type(c).__name__ != "Transform":
                    pix = get_component_icon_pixmap(type(c), 18)
                    icon_lbl.setPixmap(pix)
                    break
        clear_btn.setVisible(bool(eid))
        callback(eid)
    btn = QPushButton("\u25CB")
    btn.setFixedSize(*scale_xy(22, 22))
    btn.setToolTip("Pick Entity")
    btn.setStyleSheet(f"""
        QPushButton {{ color: {_FUSION_TEXT_DIM}; border: 1px solid {_FUSION_BORDER_LIGHT}; border-radius: 11px;
        background: {_FUSION_BG_INPUT}; font-size: 14px; }}
        QPushButton:hover {{ background: {_FUSION_BG_HOVER}; color: {_FUSION_TEXT_BRIGHT}; }}
    """)
    def _pick():
        dlg = EntityPickerDialog(scene, w)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            picked_id = dlg.selected_id()
            if picked_id is not None:
                _update_entity_display(picked_id)
    btn.clicked.connect(_pick)
    layout.addWidget(btn)
    clear_btn = QPushButton("x")
    clear_btn.setFixedSize(*scale_xy(20, 20))
    clear_btn.setToolTip("Clear")
    clear_btn.setStyleSheet(f"""
        QPushButton {{ color: {_FUSION_TEXT_DIM}; border: none; border-radius: {_FUSION_INPUT_RADIUS}; font-size: 10px; background: transparent; }}
        QPushButton:hover {{ color: {_FUSION_ACCENT_RED}; background: #3a1a1a; }}
    """)
    def _clear():
        _update_entity_display("")
    clear_btn.clicked.connect(_clear)
    clear_btn.setVisible(bool(entity_id))
    layout.addWidget(clear_btn)
    return w

def make_resource_type_picker(path: str, resource_type: str, callback: Callable[[str], None]) -> QWidget:
    from editor.resource_picker import pick_resource
    from core.components.scripting.script_component import RESOURCE_TYPE_FILTERS
    filter_str = RESOURCE_TYPE_FILTERS.get(resource_type, "All Files (*)")
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    name = os.path.basename(path) if path else ""
    icon_lbl = QLabel()
    icon_lbl.setFixedSize(*scale_xy(20, 20))
    icon_lbl.setStyleSheet(f"border: 1px solid {_FUSION_BORDER_LIGHT}; border-radius: 2px; background: {_FUSION_BG};")
    update_resource_icon(icon_lbl, path, 20)
    layout.addWidget(icon_lbl)
    def _on_resource_drop(p: str):
        _update_display(p)
    name_lbl = _ResourceDropLabel(_on_resource_drop, name if name else f"None ({resource_type})")
    name_lbl.setStyleSheet(
        f"color: {_FUSION_TEXT}; background: {_FUSION_BG_INPUT}; border: 1px solid {_FUSION_BORDER}; border-radius: {_FUSION_INPUT_RADIUS}; padding: 2px 6px;"
    )
    name_lbl.setMinimumHeight(22)
    name_lbl.setToolTip(path if path else f"No {resource_type} selected")
    layout.addWidget(name_lbl, 1)
    def _update_display(p: str):
        new_name = os.path.basename(p) if p else ""
        name_lbl.setText(new_name if new_name else f"None ({resource_type})")
        name_lbl.setToolTip(p if p else f"No {resource_type} selected")
        update_resource_icon(icon_lbl, p, 20)
        clear_btn.setVisible(bool(p))
        callback(p)
    btn = QPushButton("\u25CB")
    btn.setFixedSize(*scale_xy(22, 22))
    btn.setToolTip(f"Pick {resource_type}")
    btn.setStyleSheet(f"""
        QPushButton {{ color: {_FUSION_TEXT_DIM}; border: 1px solid {_FUSION_BORDER_LIGHT}; border-radius: 11px;
        background: {_FUSION_BG_INPUT}; font-size: 14px; }}
        QPushButton:hover {{ background: {_FUSION_BG_HOVER}; color: {_FUSION_TEXT_BRIGHT}; }}
    """)
    def _pick():
        p = pick_resource(w, f"Select {resource_type}", filter_str, path)
        if p:
            _update_display(p)
    btn.clicked.connect(_pick)
    layout.addWidget(btn)
    clear_btn = QPushButton("x")
    clear_btn.setFixedSize(*scale_xy(20, 20))
    clear_btn.setToolTip("Clear")
    clear_btn.setStyleSheet(f"""
        QPushButton {{ color: {_FUSION_TEXT_DIM}; border: none; border-radius: {_FUSION_INPUT_RADIUS}; font-size: 10px; background: transparent; }}
        QPushButton:hover {{ color: {_FUSION_ACCENT_RED}; background: #3a1a1a; }}
    """)
    def _clear():
        _update_display("")
    clear_btn.clicked.connect(_clear)
    clear_btn.setVisible(bool(path))
    layout.addWidget(clear_btn)
    return w

def make_asset_picker(path: str, asset_type: str, callback: Callable[[str], None]) -> QWidget:
    from editor.resource_picker import pick_resource
    from core.components.scripting.script_component import RESOURCE_TYPE_FILTERS
    filter_str = RESOURCE_TYPE_FILTERS.get(asset_type, "All Files (*)")
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    name = os.path.basename(path) if path else ""
    icon_lbl = QLabel()
    icon_lbl.setFixedSize(*scale_xy(20, 20))
    icon_lbl.setStyleSheet(f"border: 1px solid {_FUSION_BORDER_LIGHT}; border-radius: 2px; background: {_FUSION_BG};")
    update_resource_icon(icon_lbl, path, 20)
    layout.addWidget(icon_lbl)
    def _on_asset_drop(p: str):
        _update_display(p)
    name_lbl = _ResourceDropLabel(_on_asset_drop, name if name else f"None ({asset_type})")
    name_lbl.setStyleSheet(
        f"color: {_FUSION_TEXT}; background: {_FUSION_BG_INPUT}; border: 1px solid {_FUSION_BORDER}; border-radius: {_FUSION_INPUT_RADIUS}; padding: 2px 6px;"
    )
    name_lbl.setMinimumHeight(22)
    name_lbl.setToolTip(path if path else f"No {asset_type} selected")
    layout.addWidget(name_lbl, 1)
    def _update_display(p: str):
        new_name = os.path.basename(p) if p else ""
        name_lbl.setText(new_name if new_name else f"None ({asset_type})")
        name_lbl.setToolTip(p if p else f"No {asset_type} selected")
        update_resource_icon(icon_lbl, p, 20)
        clear_btn.setVisible(bool(p))
        callback(p)
    btn = QPushButton("\u25CB")
    btn.setFixedSize(*scale_xy(22, 22))
    btn.setToolTip(f"Pick {asset_type}")
    btn.setStyleSheet(f"""
        QPushButton {{ color: {_FUSION_TEXT_DIM}; border: 1px solid {_FUSION_BORDER_LIGHT}; border-radius: 11px;
        background: {_FUSION_BG_INPUT}; font-size: 14px; }}
        QPushButton:hover {{ background: {_FUSION_BG_HOVER}; color: {_FUSION_TEXT_BRIGHT}; }}
    """)
    def _pick():
        p = pick_resource(w, f"Select {asset_type}", filter_str, path)
        if p:
            _update_display(p)
    btn.clicked.connect(_pick)
    layout.addWidget(btn)
    create_btn = QPushButton("+")
    create_btn.setFixedSize(*scale_xy(22, 22))
    create_btn.setToolTip(f"Create new {asset_type}")
    create_btn.setStyleSheet(f"""
        QPushButton {{ color: {_FUSION_TEXT_DIM}; border: 1px solid {_FUSION_BORDER_LIGHT}; border-radius: 11px;
        background: {_FUSION_BG_INPUT}; font-size: 14px; }}
        QPushButton:hover {{ background: {_FUSION_BG_HOVER}; color: #4ec9b0; }}
    """)
    def _create():
        _create_asset_dialog(w, asset_type, _update_display)
    create_btn.clicked.connect(_create)
    layout.addWidget(create_btn)
    clear_btn = QPushButton("x")
    clear_btn.setFixedSize(*scale_xy(20, 20))
    clear_btn.setToolTip("Clear")
    clear_btn.setStyleSheet(f"""
        QPushButton {{ color: {_FUSION_TEXT_DIM}; border: none; border-radius: {_FUSION_INPUT_RADIUS}; font-size: 10px; background: transparent; }}
        QPushButton:hover {{ color: {_FUSION_ACCENT_RED}; background: #3a1a1a; }}
    """)
    def _clear():
        _update_display("")
    clear_btn.clicked.connect(_clear)
    clear_btn.setVisible(bool(path))
    layout.addWidget(clear_btn)
    return w

def _create_asset_dialog(parent, asset_type: str, callback: Callable[[str], None]):
    from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
    from core.components.scripting.script_component import RESOURCE_TYPE_FILTERS
    project_root = getattr(parent, '_project_root', os.getcwd())
    name, ok = QInputDialog.getText(parent, f"New {asset_type}", f"Asset name:")
    if not ok or not name.strip():
        return
    fname = name.strip()
    exts = {"animclip": ".animclip", "animcontroller": ".animcontroller"}
    ext = exts.get(asset_type, ".asset")
    if not fname.endswith(ext):
        fname += ext
    default_path = os.path.join(project_root, "Assets", fname)
    path, _ = QFileDialog.getSaveFileName(parent, f"Save {asset_type}", default_path,
                                          RESOURCE_TYPE_FILTERS.get(asset_type, "All Files (*)"))
    if not path:
        return
    from core.components.animation.animation_clip import AnimationClip
    if asset_type == "animclip":
        clip = AnimationClip(name.strip())
        clip.save(path)
    elif asset_type == "animcontroller":
        from core.components.animation.animator_controller import AnimatorController
        ctrl = AnimatorController(name.strip())
        ctrl.save(path)
    callback(path)

def make_vec2_row(label: str, vec: Vec2, callback) -> tuple[QWidget, list[QDoubleSpinBox]]:
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    lbl = QLabel(label)
    lbl.setFixedWidth(scale(80))
    layout.addWidget(lbl)
    spinboxes = []
    for val, comp_label in [(vec.x, "X"), (vec.y, "Y")]:
        lbl_c = QLabel(comp_label)
        lbl_c.setFixedWidth(scale(14))
        color = _XYZ_COLORS.get(comp_label, "#aaa")
        lbl_c.setStyleSheet(f"color: {color}; font-weight: bold;")
        sb = make_spinbox(val)
        sb.valueChanged.connect(callback)
        layout.addWidget(lbl_c)
        layout.addWidget(sb)
        spinboxes.append(sb)
    return w, spinboxes

def make_vec3_row(label: str, vec: Vec3, callback, reset_to: Optional[list] = None) -> tuple[QWidget, list[QDoubleSpinBox]]:
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    lbl = QLabel(label)
    lbl.setFixedWidth(scale(80))
    layout.addWidget(lbl)
    spinboxes = []
    for val, comp_label in [(vec.x, "X"), (vec.y, "Y"), (vec.z, "Z")]:
        sb = make_spinbox(val)
        sb.valueChanged.connect(callback)
        lbl_c = _DragLabel(comp_label, _XYZ_COLORS.get(comp_label, "#aaa"), sb)
        layout.addWidget(lbl_c)
        layout.addWidget(sb)
        spinboxes.append(sb)
    if reset_to is not None:
        btn = QPushButton()
        btn.setText("\u21ba")
        btn.setFixedSize(*scale_xy(18, 18))
        btn.setToolTip(f"Reset {label}")
        btn.setStyleSheet(f"""
            QPushButton {{ font-size: 12px; color: {_FUSION_TEXT_DIM}; border: 1px solid {_FUSION_BORDER}; border-radius: {_FUSION_INPUT_RADIUS}; background: transparent; }}
            QPushButton:hover {{ color: {_accent()}; border-color: {_accent()}; background: {_FUSION_BG_HOVER}; }}
        """)
        def _reset():
            for sb, v in zip(spinboxes, reset_to):
                sb.setValue(v)
        btn.clicked.connect(_reset)
        layout.addWidget(btn)
    return w, spinboxes

def make_vec4_row(label: str, vec: Vec4, callback) -> tuple[QWidget, list[QDoubleSpinBox]]:
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    lbl = QLabel(label)
    lbl.setFixedWidth(scale(80))
    layout.addWidget(lbl)
    spinboxes = []
    for val, comp_label in [(vec.x, "X"), (vec.y, "Y"), (vec.z, "Z"), (vec.w, "W")]:
        sb = make_spinbox(val)
        sb.valueChanged.connect(callback)
        lbl_c = _DragLabel(comp_label, _XYZ_COLORS.get(comp_label, "#aaa"), sb)
        layout.addWidget(lbl_c)
        layout.addWidget(sb)
        spinboxes.append(sb)
    return w, spinboxes

def make_vec2_slider_row(label: str, vec: Vec2, callback, lo=0.0, hi=1.0) -> tuple[QWidget, list[QDoubleSpinBox]]:
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    if label:
        lbl = QLabel(label)
        lbl.setFixedWidth(scale(80))
        lbl.setStyleSheet(f"color: {_FUSION_TEXT}; font-size: 11px; background: transparent;")
        layout.addWidget(lbl)
    spinboxes = []
    for val, comp_label in [(vec.x, "X"), (vec.y, "Y")]:
        lbl_c = _DragLabel(comp_label, _XYZ_COLORS.get(comp_label, "#aaa"), None)
        layout.addWidget(lbl_c)
        sb = make_spinbox(val, lo, hi, (hi - lo) / 100.0)
        sb.setMinimumWidth(60)
        lbl_c._spinbox = sb
        sb.valueChanged.connect(callback)
        layout.addWidget(sb)
        spinboxes.append(sb)
    return w, spinboxes

def make_vec3_slider_row(label: str, vec: Vec3, callback, lo=0.0, hi=1.0) -> tuple[QWidget, list[QDoubleSpinBox]]:
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    if label:
        lbl = QLabel(label)
        lbl.setFixedWidth(scale(80))
        lbl.setStyleSheet(f"color: {_FUSION_TEXT}; font-size: 11px; background: transparent;")
        layout.addWidget(lbl)
    spinboxes = []
    for val, comp_label in [(vec.x, "X"), (vec.y, "Y"), (vec.z, "Z")]:
        lbl_c = _DragLabel(comp_label, _XYZ_COLORS.get(comp_label, "#aaa"), None)
        layout.addWidget(lbl_c)
        sb = make_spinbox(val, lo, hi, (hi - lo) / 100.0)
        sb.setMinimumWidth(60)
        lbl_c._spinbox = sb
        sb.valueChanged.connect(callback)
        layout.addWidget(sb)
        spinboxes.append(sb)
    return w, spinboxes

def get_component_source_path(comp_cls: type) -> str:
    import inspect
    try:
        file_path = inspect.getfile(comp_cls)
        rel = os.path.relpath(file_path, _PROJECT_ROOT)
        return rel.replace(os.sep, "/")
    except Exception:
        return ""

def get_property_line_number(comp_cls: type, prop_name: str) -> int:
    import inspect
    try:
        lines, start_line = inspect.getsourcelines(comp_cls)
        for i, line in enumerate(lines):
            if prop_name in line and ("self." + prop_name) in line:
                return start_line + i
        for i, line in enumerate(lines):
            if f"self.{prop_name}" in line or f": {prop_name}" in line:
                return start_line + i
    except Exception:
        pass
    return 1

def collapse_value(v):
    if hasattr(v, 'to_list'):
        return v.to_list()
    if hasattr(v, '__iter__') and not isinstance(v, (str, bytes, dict)):
        return list(v)
    return v
