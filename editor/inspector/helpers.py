# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
from typing import Optional, Callable
import qtawesome as qta
from PyQt6.QtWidgets import QLabel, QWidget, QHBoxLayout, QPushButton, QDoubleSpinBox, QSlider, QDialog, QFileDialog, \
    QInputDialog, QMessageBox, QFrame, QGraphicsOpacityEffect, QSizePolicy
from PyQt6.QtCore import Qt, QObject, QEvent, QPropertyAnimation, QEasingCurve, QRect
from PyQt6.QtGui import QPixmap, QFont, QPainter, QColor, QBrush, QPen, QFont as QF
from core.config.editor_scale import scale, scale_xy
from editor.inspector.constants import _XYZ_COLORS, _accent
from editor.inspector.widgets import _FocusSpinBox, _DragLabel, _ResourceDropLabel, _EntityDropLabel, EntityPickerDialog
from core.maths.math3d import Vec2, Vec3, Vec4

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

class _NavFilter(QObject):
    def __init__(self, single_cb, double_cb=None, parent=None):
        super().__init__(parent)
        self._single = single_cb
        self._double = double_cb
    def eventFilter(self, obj, event):
        t = event.type()
        if t == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton and self._double:
            self._double()
            return True
        if t == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton and self._single:
            self._single()
        return super().eventFilter(obj, event)

def _navigate_resource(label: QLabel):
    p = label.toolTip()
    if p and "No " not in p:
        mw = label.window()
        if hasattr(mw, '_project'):
            mw._project.reveal_resource(p)
            mw._project.open_resource(p)

def _flash_resource(label: QLabel):
    p = label.toolTip()
    if p and "No " not in p:
        mw = label.window()
        if hasattr(mw, '_project'):
            mw._project.flash_resource(p)

def _navigate_entity(label: QLabel):
    eid = label.toolTip()
    if eid and "No " not in eid:
        mw = label.window()
        if hasattr(mw, '_hierarchy'):
            mw._hierarchy.reveal_entity(eid)

def _flash_entity(label: QLabel):
    eid = label.toolTip()
    if eid and "No " not in eid:
        mw = label.window()
        if hasattr(mw, '_hierarchy'):
            mw._hierarchy.flash_entity(eid)

def _expanded_rect(rect: QRect, factor: float) -> QRect:
    w = int(rect.width() * factor)
    h = int(rect.height() * factor)
    x = rect.center().x() - w // 2
    y = rect.center().y() - h // 2
    return QRect(x, y, w, h)

def _flash_overlay(viewport, item_rect: QRect, duration: int = 800):
    main_window = viewport.window()
    if not main_window:
        return
    global_tl = viewport.mapToGlobal(item_rect.topLeft())
    item_global = QRect(global_tl, item_rect.size())
    start_global = _expanded_rect(item_global, 1.5)

    overlay = QFrame(main_window)
    overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    overlay.setStyleSheet("background: rgba(255, 255, 0, 180); border-radius: 4px;")
    overlay.setGeometry(QRect(main_window.mapFromGlobal(start_global.topLeft()), start_global.size()))
    overlay.show()
    overlay.raise_()

    op = QGraphicsOpacityEffect(overlay)
    overlay.setGraphicsEffect(op)
    opacity_anim = QPropertyAnimation(op, b"opacity")
    opacity_anim.setStartValue(0.8)
    opacity_anim.setEndValue(0.0)
    opacity_anim.setDuration(duration)

    geo_anim = QPropertyAnimation(overlay, b"geometry")
    geo_anim.setStartValue(QRect(main_window.mapFromGlobal(start_global.topLeft()), start_global.size()))
    geo_anim.setEndValue(QRect(main_window.mapFromGlobal(item_global.topLeft()), item_global.size()))
    geo_anim.setDuration(duration // 2)
    geo_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _cleanup():
        overlay.deleteLater()
    opacity_anim.finished.connect(_cleanup)
    opacity_anim.start()
    geo_anim.start()
    overlay._opacity_anim = opacity_anim
    overlay._geo_anim = geo_anim

_PICKER_BTN_STYLE = """
    QPushButton {
        background: transparent; border: 1px solid palette(mid);
        border-radius: 3px; padding: 0px; font-size: 10px;
    }
    QPushButton:hover { background: palette(light); border-color: palette(highlight); }
    QPushButton:pressed { background: palette(dark); }
"""

def _style_picker_btn(btn):
    btn.setStyleSheet(_PICKER_BTN_STYLE)

def make_spinbox(val: float, lo: float = -1e9, hi: float = 1e9, step: float = 0.1, decimals: int = 4) -> QDoubleSpinBox:
    sb = _FocusSpinBox()
    sb.setRange(lo, hi)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    sb.setValue(val)
    sb.setMinimumWidth(60)
    return sb

def make_clickable_label(text: str, on_click: Callable[[], None]) -> QLabel:
    lbl = QLabel(f"  {text}")
    lbl.setStyleSheet(f"""
        QLabel {{
            color: {_accent()};
            font-size: 9px;
            padding: 0px;
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

_COMMON_EXT = (".fbx", ".obj", ".stl", ".gltf", ".glb", ".usdz",
               ".png", ".jpg", ".jpeg", ".bmp", ".tga",
               ".wav", ".mp3", ".ogg",
               ".py",
               ".mat",
               ".shader", ".vert", ".frag")

def _is_path_valid(p: str) -> bool:
    if not p:
        return False
    p = os.path.normpath(p)
    if os.path.exists(p):
        return True
    for ext in _COMMON_EXT:
        if os.path.exists(p + ext):
            return True
    try:
        from core.engine.engine import Engine
        eng = Engine.instance()
        if eng and eng.project_root:
            root = eng.project_root
            for ext in ("",) + _COMMON_EXT:
                if os.path.exists(os.path.normpath(os.path.join(root, p + ext))):
                    return True
    except Exception:
        pass
    return False

_ERROR_STYLE = """
    QLabel {
        background: #3d1a1a; color: #f44747;
        border: 1px solid #f44747; border-radius: 2px;
        padding: 2px 4px; font-size: 11px;
    }
"""

_EMPTY_STYLE = """
    QLabel {
        background: #3d3410; color: #e0c040;
        border: 1px solid #b89320; border-radius: 2px;
        padding: 2px 4px; font-size: 11px;
    }
"""

def make_resource_picker(path: str, filter_str: str, callback: Callable[[str], None]) -> QWidget:
    from editor.resource_picker import pick_resource
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    name = os.path.basename(path) if path else ""
    icon_lbl = QLabel()
    icon_lbl.setFixedSize(*scale_xy(20, 20))
    layout.addWidget(icon_lbl)
    def _on_resource_drop(p: str):
        _update_display(p)
    name_lbl = _ResourceDropLabel(_on_resource_drop, name if name else "None")
    name_lbl.setMinimumHeight(22)
    name_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
    name_lbl.installEventFilter(_NavFilter(lambda: _flash_resource(name_lbl), lambda: _navigate_resource(name_lbl), name_lbl))
    layout.addWidget(name_lbl, 1)
    def _apply_state(p: str):
        name_lbl.setText(os.path.basename(p) if p else "None")
        name_lbl.setToolTip(p if p else "No resource selected")
        if _is_path_valid(p):
            name_lbl.setStyleSheet(name_lbl._BASE_STYLE)
            update_resource_icon(icon_lbl, p, 20)
        elif p:
            name_lbl.setStyleSheet(_ERROR_STYLE)
            icon_lbl.setPixmap(qta.icon("fa5s.exclamation-triangle", color="#e74c3c").pixmap(20, 20))
        else:
            name_lbl.setStyleSheet(_EMPTY_STYLE)
            update_resource_icon(icon_lbl, p, 20)
    def _update_display(p: str):
        _apply_state(p)
        clear_btn.setVisible(bool(p))
        callback(p)
    _apply_state(path)
    btn = QPushButton(qta.icon("fa5s.folder-open", color="#d4d4d4"), "")
    btn.setFixedSize(*scale_xy(22, 22))
    btn.setToolTip("Pick Resource")
    _style_picker_btn(btn)
    def _pick():
        p = pick_resource(w, "Select Resource", filter_str, path)
        if p:
            _update_display(p)
    btn.clicked.connect(_pick)
    layout.addWidget(btn)
    clear_btn = QPushButton(qta.icon("fa5s.times", color="#d4d4d4"), "")
    clear_btn.setFixedSize(*scale_xy(20, 20))
    clear_btn.setToolTip("Clear")
    _style_picker_btn(clear_btn)
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
    name_lbl.setMinimumHeight(22)
    name_lbl.setToolTip(entity_id if entity_id else "No entity selected")
    name_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
    name_lbl.installEventFilter(_NavFilter(lambda: _flash_entity(name_lbl), lambda: _navigate_entity(name_lbl), name_lbl))
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
            name_lbl.setStyleSheet(name_lbl._BASE_STYLE)
        elif eid:
            name_lbl.setStyleSheet(_ERROR_STYLE)
        else:
            name_lbl.setStyleSheet(_EMPTY_STYLE)
        clear_btn.setVisible(bool(eid))
        callback(eid)
    btn = QPushButton(qta.icon("fa5s.crosshairs", color="#d4d4d4"), "")
    btn.setFixedSize(*scale_xy(22, 22))
    btn.setToolTip("Pick Entity")
    _style_picker_btn(btn)
    def _pick():
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QCursor
        dlg = EntityPickerDialog(scene, None)
        dlg.adjustSize()
        screen = QApplication.primaryScreen().availableGeometry()
        dw, dh = dlg.width(), dlg.height()
        cursor_pos = QCursor.pos()
        x = cursor_pos.x() - dw // 2
        y = cursor_pos.y() + 4
        if y + dh > screen.bottom():
            y = cursor_pos.y() - dh - 4
        if y < screen.top():
            y = screen.top()
        if x + dw > screen.right():
            x = screen.right() - dw
        if x < screen.left():
            x = screen.left()
        dlg.move(x, y)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            picked_id = dlg.selected_id()
            if picked_id is not None:
                _update_entity_display(picked_id)
    btn.clicked.connect(_pick)
    layout.addWidget(btn)
    clear_btn = QPushButton(qta.icon("fa5s.times", color="#d4d4d4"), "")
    clear_btn.setFixedSize(*scale_xy(20, 20))
    clear_btn.setToolTip("Clear")
    _style_picker_btn(clear_btn)
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
    update_resource_icon(icon_lbl, path, 20)
    layout.addWidget(icon_lbl)
    def _on_resource_drop(p: str):
        _update_display(p)
    name_lbl = _ResourceDropLabel(_on_resource_drop, name if name else f"None ({resource_type})")
    name_lbl.setMinimumHeight(22)
    name_lbl.setToolTip(path if path else f"No {resource_type} selected")
    name_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
    name_lbl.installEventFilter(_NavFilter(lambda: _flash_resource(name_lbl), lambda: _navigate_resource(name_lbl), name_lbl))
    layout.addWidget(name_lbl, 1)
    def _update_display(p: str):
        new_name = os.path.basename(p) if p else ""
        name_lbl.setText(new_name if new_name else f"None ({resource_type})")
        name_lbl.setToolTip(p if p else f"No {resource_type} selected")
        update_resource_icon(icon_lbl, p, 20)
        if p:
            if _is_path_valid(p):
                name_lbl.setStyleSheet(name_lbl._BASE_STYLE)
            else:
                name_lbl.setStyleSheet(_ERROR_STYLE)
                icon_lbl.setPixmap(qta.icon("fa5s.exclamation-triangle", color="#e74c3c").pixmap(20, 20))
        else:
            name_lbl.setStyleSheet(_EMPTY_STYLE)
        clear_btn.setVisible(bool(p))
        callback(p)
    btn = QPushButton(qta.icon("fa5s.folder-open", color="#d4d4d4"), "")
    btn.setFixedSize(*scale_xy(22, 22))
    btn.setToolTip(f"Pick {resource_type}")
    _style_picker_btn(btn)
    def _pick():
        p = pick_resource(w, f"Select {resource_type}", filter_str, path)
        if p:
            _update_display(p)
    btn.clicked.connect(_pick)
    layout.addWidget(btn)
    clear_btn = QPushButton(qta.icon("fa5s.times", color="#d4d4d4"), "")
    clear_btn.setFixedSize(*scale_xy(20, 20))
    clear_btn.setToolTip("Clear")
    _style_picker_btn(clear_btn)
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
    update_resource_icon(icon_lbl, path, 20)
    layout.addWidget(icon_lbl)
    def _on_asset_drop(p: str):
        _update_display(p)
    name_lbl = _ResourceDropLabel(_on_asset_drop, name if name else f"None ({asset_type})")
    name_lbl.setMinimumHeight(22)
    name_lbl.setToolTip(path if path else f"No {asset_type} selected")
    name_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
    name_lbl.installEventFilter(_NavFilter(lambda: _flash_resource(name_lbl), lambda: _navigate_resource(name_lbl), name_lbl))
    layout.addWidget(name_lbl, 1)
    def _update_display(p: str):
        new_name = os.path.basename(p) if p else ""
        name_lbl.setText(new_name if new_name else f"None ({asset_type})")
        name_lbl.setToolTip(p if p else f"No {asset_type} selected")
        update_resource_icon(icon_lbl, p, 20)
        if p:
            if _is_path_valid(p):
                name_lbl.setStyleSheet(name_lbl._BASE_STYLE)
            else:
                name_lbl.setStyleSheet(_ERROR_STYLE)
                icon_lbl.setPixmap(qta.icon("fa5s.exclamation-triangle", color="#e74c3c").pixmap(20, 20))
        else:
            name_lbl.setStyleSheet(_EMPTY_STYLE)
        clear_btn.setVisible(bool(p))
        callback(p)
    btn = QPushButton(qta.icon("fa5s.folder-open", color="#d4d4d4"), "")
    btn.setFixedSize(*scale_xy(22, 22))
    btn.setToolTip(f"Pick {asset_type}")
    _style_picker_btn(btn)
    def _pick():
        p = pick_resource(w, f"Select {asset_type}", filter_str, path)
        if p:
            _update_display(p)
    btn.clicked.connect(_pick)
    layout.addWidget(btn)
    create_btn = QPushButton(qta.icon("fa5s.plus", color="#9ccc65"), "")
    create_btn.setFixedSize(*scale_xy(22, 22))
    create_btn.setToolTip(f"Create new {asset_type}")
    _style_picker_btn(create_btn)
    def _create():
        _create_asset_dialog(w, asset_type, _update_display)
    create_btn.clicked.connect(_create)
    layout.addWidget(create_btn)
    clear_btn = QPushButton(qta.icon("fa5s.times", color="#d4d4d4"), "")
    clear_btn.setFixedSize(*scale_xy(20, 20))
    clear_btn.setToolTip("Clear")
    _style_picker_btn(clear_btn)
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
    lbl.setWordWrap(True)
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
    lbl.setWordWrap(True)
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
        btn = QPushButton(qta.icon("fa5s.undo-alt", color="#d4d4d4"), "")
        btn.setFixedSize(*scale_xy(18, 18))
        btn.setToolTip(f"Reset {label}")
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
    lbl.setWordWrap(True)
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
        lbl.setWordWrap(True)
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
        lbl.setWordWrap(True)
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

_SOURCE_PATH_CACHE: dict[type, str] = {}
_LINE_NUM_CACHE: dict[tuple, int] = {}


def get_component_source_path(comp_cls: type) -> str:
    cached = _SOURCE_PATH_CACHE.get(comp_cls)
    if cached is not None:
        return cached
    import inspect
    result = ""
    try:
        file_path = inspect.getfile(comp_cls)
        rel = os.path.relpath(file_path, _PROJECT_ROOT)
        result = rel.replace(os.sep, "/")
    except Exception:
        pass
    _SOURCE_PATH_CACHE[comp_cls] = result
    return result


def get_property_line_number(comp_cls: type, prop_name: str) -> int:
    key = (comp_cls, prop_name)
    cached = _LINE_NUM_CACHE.get(key)
    if cached is not None:
        return cached
    import inspect
    result = 1
    try:
        lines, start_line = inspect.getsourcelines(comp_cls)
        for i, line in enumerate(lines):
            if prop_name in line and ("self." + prop_name) in line:
                result = start_line + i
                break
        else:
            for i, line in enumerate(lines):
                if f"self.{prop_name}" in line or f": {prop_name}" in line:
                    result = start_line + i
                    break
    except Exception:
        pass
    _LINE_NUM_CACHE[key] = result
    return result

def collapse_value(v):
    if hasattr(v, 'to_list'):
        return v.to_list()
    if hasattr(v, '__iter__') and not isinstance(v, (str, bytes, dict)):
        return list(v)
    return v


