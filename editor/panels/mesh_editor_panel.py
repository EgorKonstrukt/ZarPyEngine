# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QDoubleSpinBox,
                             QSpinBox, QGroupBox, QGridLayout, QScrollArea,
                             QFrame, QCheckBox, QSlider, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from core.ecs import ComponentRegistry
from core.commands import AddComponentCommand, RemoveComponentCommand, get_history
from core.components.inspector_meta import FieldType, InspectorField
from core.components.mesh_editor import (ProBuilderMesh, SelectionMode,
    generate_box, generate_sphere, generate_cylinder, generate_plane,
    generate_torus, generate_cone, generate_pipe, create_primitive,
    extrude_faces, bevel_edges, subdivide_faces, weld_vertices,
    flip_normals, collapse_edges, bridge_edges, smart_optimize,
    get_primitive_names)
from core.components.rendering.mesh_filter import MeshFilter
from core.components.rendering.mesh_renderer import MeshRenderer
from core.logger import Logger
if TYPE_CHECKING:
    from core.ecs import Entity
    from core.engine import Engine
from core.editor_scale import scale, scale_xy

_STYLE_BTN = ("QPushButton { border-radius: 3px; padding: 4px 8px; font-size: 11px; }")
_STYLE_BTN_ACCENT = ("QPushButton { border-radius: 3px; padding: 4px 8px; font-size: 11px; }")
_STYLE_LABEL = "font-size: 10px; padding: 2px 0;"
_STYLE_GROUP = ("QGroupBox { font-size: 11px; font-weight: bold; "
                "border: 1px solid; border-radius: 4px; margin-top: 8px; padding-top: 14px; } "
                "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }")


def _make_btn(text: str, tooltip: str = "", accent: bool = False) -> QPushButton:
    btn = QPushButton(text)
    btn.setToolTip(tooltip)
    btn.setStyleSheet(_STYLE_BTN_ACCENT if accent else _STYLE_BTN)
    btn.setMinimumHeight(24)
    return btn


def _make_spin(val: float, lo: float = -1e9, hi: float = 1e9, step: float = 0.1, decimals: int = 4) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    sb.setValue(val)
    sb.setMinimumWidth(60)
    return sb


def _make_int_spin(val: int, lo: int = 1, hi: int = 128) -> QSpinBox:
    sb = QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(val)
    sb.setMinimumWidth(50)
    return sb


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFrameShadow(QFrame.Shadow.Sunken)
    return f


class MeshEditorPanel(QDockWidget):
    def __init__(self, engine: Engine, parent=None):
        super().__init__("Mesh Editor", parent)
        self._engine = engine
        self._entity: Optional[Entity] = None
        self._mesh_comp: Optional[ProBuilderMesh] = None
        self._updating = False

        self.setObjectName("MeshEditorDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable)

        root = QWidget()
        self.setWidget(root)
        self._layout = QVBoxLayout(root)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        self._scroll_layout = QVBoxLayout(scroll_content)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(2)
        scroll.setWidget(scroll_content)
        self._layout.addWidget(scroll)

        self._build_primitive_section()
        self._build_tools_section()
        self._build_selection_section()
        self._build_transform_section()
        self._build_info_section()
        self._scroll_layout.addStretch()

        self._update_mesh_ui(False)

    def _add_group(self, title: str, store_as: str = "") -> QVBoxLayout:
        gb = QGroupBox(title)
        gb.setStyleSheet(_STYLE_GROUP)
        layout = QVBoxLayout(gb)
        layout.setContentsMargins(6, 12, 6, 6)
        layout.setSpacing(3)
        self._scroll_layout.addWidget(gb)
        if store_as:
            setattr(self, store_as, gb)
        return layout

    def _build_primitive_section(self):
        layout = self._add_group("New Shape")
        row1 = QHBoxLayout()
        row1.setSpacing(3)
        self._primitive_combo = QComboBox()
        for name in get_primitive_names():
            self._primitive_combo.addItem(name)
        self._primitive_combo.setStyleSheet("border: 1px solid; padding: 2px 4px;")
        row1.addWidget(self._primitive_combo, 1)
        create_btn = _make_btn("Create", "Create primitive as new mesh", accent=True)
        create_btn.clicked.connect(self._on_create_primitive)
        row1.addWidget(create_btn)
        layout.addLayout(row1)

        size_row = QHBoxLayout()
        size_row.setSpacing(3)
        size_row.addWidget(QLabel("Size"))
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setRange(1, 100)
        self._size_slider.setValue(50)
        self._size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._size_label = QLabel("1.0")
        self._size_label.setFixedWidth(scale(28))
        self._size_label.setStyleSheet("")
        self._size_slider.valueChanged.connect(lambda v: self._size_label.setText(f"{v/50:.1f}"))
        size_row.addWidget(self._size_slider)
        size_row.addWidget(self._size_label)
        layout.addLayout(size_row)

        detail_row = QHBoxLayout()
        detail_row.setSpacing(3)
        detail_row.addWidget(QLabel("Detail"))
        self._detail_slider = QSlider(Qt.Orientation.Horizontal)
        self._detail_slider.setRange(1, 100)
        self._detail_slider.setValue(50)
        self._detail_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._detail_label = QLabel("32")
        self._detail_label.setFixedWidth(scale(28))
        self._detail_label.setStyleSheet("")
        self._detail_slider.valueChanged.connect(lambda v: self._detail_label.setText(f"{max(4, v*32//50)}"))
        detail_row.addWidget(self._detail_slider)
        detail_row.addWidget(self._detail_label)
        layout.addLayout(detail_row)

    def _build_tools_section(self):
        layout = self._add_group("Tools", "_tools_group")
        r1 = QHBoxLayout()
        r1.setSpacing(3)
        extrude_btn = _make_btn("Extrude", "Extrude selected faces")
        extrude_btn.clicked.connect(self._on_extrude)
        r1.addWidget(extrude_btn)
        self._extrude_dist = _make_spin(0.25, -10, 10, 0.05, 3)
        r1.addWidget(self._extrude_dist)
        layout.addLayout(r1)

        r2 = QHBoxLayout()
        r2.setSpacing(3)
        bevel_btn = _make_btn("Bevel", "Bevel selected edges")
        bevel_btn.clicked.connect(self._on_bevel)
        r2.addWidget(bevel_btn)
        self._bevel_amount = _make_spin(0.1, 0.001, 5, 0.01, 3)
        r2.addWidget(self._bevel_amount)
        layout.addLayout(r2)

        r3 = QHBoxLayout()
        r3.setSpacing(3)
        subd_btn = _make_btn("Subdivide", "Subdivide selected faces")
        subd_btn.clicked.connect(self._on_subdivide)
        r3.addWidget(subd_btn)
        layout.addLayout(r3)

        r4 = QHBoxLayout()
        r4.setSpacing(3)
        weld_btn = _make_btn("Weld Verts", "Merge vertices within threshold")
        weld_btn.clicked.connect(self._on_weld)
        r4.addWidget(weld_btn)
        self._weld_thresh = _make_spin(0.001, 0.0001, 1, 0.001, 4)
        r4.addWidget(self._weld_thresh)
        layout.addLayout(r4)

        r5 = QHBoxLayout()
        r5.setSpacing(3)
        flip_btn = _make_btn("Flip Normals", "Flip normals of selected faces")
        flip_btn.clicked.connect(self._on_flip)
        r5.addWidget(flip_btn)
        layout.addLayout(r5)

        r6 = QHBoxLayout()
        r6.setSpacing(3)
        collapse_btn = _make_btn("Collapse Edges", "Collapse selected edges to midpoint")
        collapse_btn.clicked.connect(self._on_collapse)
        r6.addWidget(collapse_btn)
        layout.addLayout(r6)

        r7 = QHBoxLayout()
        r7.setSpacing(3)
        bridge_btn = _make_btn("Bridge Edges", "Bridge two selected edges")
        bridge_btn.clicked.connect(self._on_bridge)
        r7.addWidget(bridge_btn)
        layout.addLayout(r7)

        r8 = QHBoxLayout()
        r8.setSpacing(3)
        optimize_btn = _make_btn("Optimize", "Remove degenerate faces and weld coincident verts")
        optimize_btn.clicked.connect(self._on_optimize)
        r8.addWidget(optimize_btn)
        layout.addLayout(r8)

    def _build_selection_section(self):
        layout = self._add_group("Selection", "_sel_group")
        r = QHBoxLayout()
        r.setSpacing(3)
        self._sel_obj = _make_btn("Object", "Object selection mode")
        self._sel_obj.clicked.connect(lambda: self._set_sel_mode(SelectionMode.OBJECT))
        r.addWidget(self._sel_obj)
        self._sel_vert = _make_btn("Vertex", "Vertex selection mode")
        self._sel_vert.clicked.connect(lambda: self._set_sel_mode(SelectionMode.VERTEX))
        r.addWidget(self._sel_vert)
        self._sel_edge = _make_btn("Edge", "Edge selection mode")
        self._sel_edge.clicked.connect(lambda: self._set_sel_mode(SelectionMode.EDGE))
        r.addWidget(self._sel_edge)
        self._sel_face = _make_btn("Face", "Face selection mode")
        self._sel_face.clicked.connect(lambda: self._set_sel_mode(SelectionMode.FACE))
        r.addWidget(self._sel_face)
        layout.addLayout(r)
        clear_sel = _make_btn("Clear Selection", "Deselect all")
        clear_sel.clicked.connect(self._on_clear_selection)
        layout.addWidget(clear_sel)

    def _build_transform_section(self):
        layout = self._add_group("Transform Tools", "_xform_group")
        r1 = QHBoxLayout()
        r1.setSpacing(3)
        center_btn = _make_btn("Center Pivot", "Center pivot to geometry")
        center_btn.clicked.connect(self._on_center_pivot)
        r1.addWidget(center_btn)
        layout.addLayout(r1)

    def _build_info_section(self):
        layout = self._add_group("Mesh Info", "_info_group")
        self._vert_label = QLabel("Verts: 0")
        self._vert_label.setStyleSheet(_STYLE_LABEL)
        layout.addWidget(self._vert_label)
        self._tri_label = QLabel("Tris: 0")
        self._tri_label.setStyleSheet(_STYLE_LABEL)
        layout.addWidget(self._tri_label)
        self._edge_label = QLabel("Edges: 0")
        self._edge_label.setStyleSheet(_STYLE_LABEL)
        layout.addWidget(self._edge_label)
        smooth_row = QHBoxLayout()
        smooth_row.setSpacing(3)
        smooth_row.addWidget(QLabel("Smooth Angle"))
        self._smooth_angle = QSlider(Qt.Orientation.Horizontal)
        self._smooth_angle.setRange(0, 180)
        self._smooth_angle.setValue(45)
        self._smooth_angle.valueChanged.connect(self._on_smooth_changed)
        smooth_row.addWidget(self._smooth_angle)
        self._smooth_label = QLabel("45")
        self._smooth_label.setFixedWidth(scale(24))
        self._smooth_label.setStyleSheet("")
        smooth_row.addWidget(self._smooth_label)
        layout.addLayout(smooth_row)

    def _update_mesh_ui(self, enabled: bool):
        for w in [self._tools_group, self._sel_group, self._xform_group, self._info_group]:
            w.setEnabled(enabled)

    def set_entity(self, entity: Optional[Entity]):
        self._entity = entity
        if entity is None:
            self._mesh_comp = None
            self._update_mesh_ui(False)
            self._update_info()
            return
        comp = entity.get_component(ProBuilderMesh)
        self._mesh_comp = comp
        if comp is None:
            self._update_mesh_ui(False)
        else:
            self._update_mesh_ui(True)
            self._update_info()
            self._update_sel_buttons()

    def _get_selected_entity(self) -> Optional[Entity]:
        from core.engine import Engine
        eng = Engine.instance()
        if not eng:
            return None
        vp = getattr(eng, 'viewport', None)
        if not vp:
            return None
        sel = getattr(vp, '_selected_entities', None)
        if sel and len(sel) > 0:
            return sel[0]
        return None

    def _on_create_primitive(self):
        name = self._primitive_combo.currentText()
        size_norm = self._size_slider.value() / 50.0
        detail_norm = max(4, self._detail_slider.value() * 32 // 50)
        entity = self._get_selected_entity()
        if entity is None:
            from core.ecs import Entity as EcsEntity
            scene = self._engine.scene
            if not scene:
                return
            entity = scene.create_entity(f"{name}")
            from core.components.transform import Transform
            entity.add_component(Transform())
        pb: ProBuilderMesh = entity.get_component(ProBuilderMesh)
        if pb is None:
            pb = ProBuilderMesh()
            entity.add_component(pb)
        if name == "Box":
            s = size_norm * 0.5
            positions, indices = create_primitive(name, width=s, height=s, depth=s)
        elif name == "Sphere":
            positions, indices = create_primitive(name, radius=size_norm * 0.5, segments=detail_norm)
        elif name == "Cylinder":
            positions, indices = create_primitive(name, radius=size_norm * 0.25, height=size_norm * 0.5, segments=detail_norm)
        elif name == "Plane":
            positions, indices = create_primitive(name, width=size_norm, depth=size_norm)
        elif name == "Torus":
            positions, indices = create_primitive(name, major_radius=size_norm * 0.3, minor_radius=size_norm * 0.08,
                                                   major_segments=detail_norm, minor_segments=max(6, detail_norm // 2))
        elif name == "Cone":
            positions, indices = create_primitive(name, radius=size_norm * 0.25, height=size_norm * 0.5, segments=detail_norm)
        elif name == "Pipe":
            positions, indices = create_primitive(name, radius=size_norm * 0.3, thickness=size_norm * 0.06,
                                                   height=size_norm * 0.5, segments=detail_norm)
        elif name == "Stairs":
            positions, indices = create_primitive(name, width=size_norm * 0.8, height=size_norm * 0.5,
                                                   depth=size_norm, steps=max(2, detail_norm))
        else:
            positions, indices = create_primitive(name)
        pb.set_mesh_data(positions, indices)
        pb._gpu_dirty = True
        mf = entity.get_component(MeshFilter)
        if mf is None:
            mf = MeshFilter()
            entity.add_component(mf)
            mr = MeshRenderer()
            entity.add_component(mr)
        mf.mesh_name = f"ProBuilder_{entity.id[:6]}"
        mf.mesh_path = ""
        mr.material_path = "assets/materials/ProBuilderPrototype.mat"
        self._mesh_comp = pb
        self._update_info()
        if self._engine.scene:
            self._engine.scene._render_version += 1

    def _set_sel_mode(self, mode: SelectionMode):
        if self._mesh_comp is None:
            return
        self._mesh_comp.selection_mode = mode
        self._mesh_comp.clear_selection()
        self._update_sel_buttons()

    def _update_sel_buttons(self):
        if self._mesh_comp is None:
            return
        mode = self._mesh_comp.selection_mode
        for btn, m in [(self._sel_obj, SelectionMode.OBJECT),
                        (self._sel_vert, SelectionMode.VERTEX),
                        (self._sel_edge, SelectionMode.EDGE),
                        (self._sel_face, SelectionMode.FACE)]:
            if m == mode:
                btn.setStyleSheet(_STYLE_BTN_ACCENT)
            else:
                btn.setStyleSheet(_STYLE_BTN)

    def _update_info(self):
        if self._mesh_comp is None:
            self._vert_label.setText("Verts: 0")
            self._tri_label.setText("Tris: 0")
            self._edge_label.setText("Edges: 0")
            return
        self._vert_label.setText(f"Verts: {self._mesh_comp.vertex_count}")
        self._tri_label.setText(f"Tris: {self._mesh_comp.triangle_count}")
        self._edge_label.setText(f"Edges: {self._mesh_comp.edge_count}")
        self._smooth_angle.setValue(int(self._mesh_comp.smooth_angle))
        self._smooth_label.setText(str(int(self._mesh_comp.smooth_angle)))

    def _on_extrude(self):
        if self._mesh_comp is None:
            return
        if self._mesh_comp.selection_mode != SelectionMode.FACE or not self._mesh_comp.selected_faces:
            Logger.warning("Mesh Editor: Select faces in Face mode to extrude")
            return
        dist = self._extrude_dist.value()
        extrude_faces(self._mesh_comp, self._mesh_comp.selected_faces, dist)
        self._sync_to_renderer()
        self._update_info()

    def _on_bevel(self):
        if self._mesh_comp is None:
            return
        if self._mesh_comp.selection_mode != SelectionMode.EDGE or not self._mesh_comp.selected_edges:
            Logger.warning("Mesh Editor: Select edges in Edge mode to bevel")
            return
        amount = self._bevel_amount.value()
        bevel_edges(self._mesh_comp, self._mesh_comp.selected_edges, amount)
        self._sync_to_renderer()
        self._update_info()

    def _on_subdivide(self):
        if self._mesh_comp is None:
            return
        faces = self._mesh_comp.selected_faces if self._mesh_comp.selection_mode == SelectionMode.FACE and self._mesh_comp.selected_faces else None
        if faces is None:
            faces = set(range(self._mesh_comp.triangle_count))
        subdivide_faces(self._mesh_comp, faces)
        self._sync_to_renderer()
        self._update_info()

    def _on_weld(self):
        if self._mesh_comp is None:
            return
        thresh = self._weld_thresh.value()
        weld_vertices(self._mesh_comp, thresh)
        self._sync_to_renderer()
        self._update_info()

    def _on_flip(self):
        if self._mesh_comp is None:
            return
        faces = self._mesh_comp.selected_faces if self._mesh_comp.selection_mode == SelectionMode.FACE and self._mesh_comp.selected_faces else None
        flip_normals(self._mesh_comp, faces)
        self._sync_to_renderer()
        self._update_info()

    def _on_collapse(self):
        if self._mesh_comp is None:
            return
        if self._mesh_comp.selection_mode != SelectionMode.EDGE or not self._mesh_comp.selected_edges:
            Logger.warning("Mesh Editor: Select edges in Edge mode to collapse")
            return
        collapse_edges(self._mesh_comp, self._mesh_comp.selected_edges)
        self._sync_to_renderer()
        self._update_info()

    def _on_bridge(self):
        if self._mesh_comp is None:
            return
        if self._mesh_comp.selection_mode != SelectionMode.EDGE or len(self._mesh_comp.selected_edges) < 2:
            Logger.warning("Mesh Editor: Select at least 2 edges in Edge mode to bridge")
            return
        edges = sorted(self._mesh_comp.selected_edges, key=lambda x: (x[0], x[1]))
        bridge_edges(self._mesh_comp, edges[0], edges[1])
        self._sync_to_renderer()
        self._update_info()

    def _on_optimize(self):
        if self._mesh_comp is None:
            return
        smart_optimize(self._mesh_comp)
        self._sync_to_renderer()
        self._update_info()

    def _on_center_pivot(self):
        if self._mesh_comp is None:
            return
        pos = self._mesh_comp.positions
        if pos.size == 0:
            return
        center = pos.mean(axis=0)
        pos[:] = pos - center
        self._mesh_comp.rebuild_normals()
        self._sync_to_renderer()
        self._update_info()
        self._mesh_comp.rebuild_normals()
        self._sync_to_renderer()

    def _on_clear_selection(self):
        if self._mesh_comp is None:
            return
        self._mesh_comp.clear_selection()

    def _on_smooth_changed(self, val: int):
        if self._mesh_comp is None or self._updating:
            return
        self._smooth_label.setText(str(val))
        self._mesh_comp.smooth_angle = float(val)
        self._mesh_comp.rebuild_normals()
        self._sync_to_renderer()

    def _sync_to_renderer(self):
        if self._mesh_comp is None:
            return
        self._mesh_comp._gpu_dirty = True
        if self._engine and self._engine.scene:
            self._engine.scene._render_version += 1
