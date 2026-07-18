# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                              QToolBar, QToolButton, QLabel, QFileDialog,
                              QFrame, QMessageBox)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPalette
if TYPE_CHECKING:
    from core.engine.engine import Engine

from core.config.editor_scale import scale

_DARK_STYLE = ""

class _ShaderGraphWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._graph = None
        self._view = None
        self._current_file = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QToolBar()
        toolbar.setIconSize(QSize(16, 16))
        toolbar.setStyleSheet(f"""
        """)

        self._new_btn = QToolButton()
        self._new_btn.setText("+ New")
        self._new_btn.clicked.connect(self._new_shader)
        toolbar.addWidget(self._new_btn)

        self._open_btn = QToolButton()
        self._open_btn.setText("Open")
        self._open_btn.clicked.connect(self._open_shader)
        toolbar.addWidget(self._open_btn)

        self._save_btn = QToolButton()
        self._save_btn.setText("Save")
        self._save_btn.clicked.connect(self._save_shader)
        toolbar.addWidget(self._save_btn)

        self._save_as_btn = QToolButton()
        self._save_as_btn.setText("Save As")
        self._save_as_btn.clicked.connect(self._save_shader_as)
        toolbar.addWidget(self._save_as_btn)

        toolbar.addSeparator()

        self._compile_btn = QToolButton()
        self._compile_btn.setText("Generate Code")
        self._compile_btn.clicked.connect(self._generate_and_preview)
        toolbar.addWidget(self._compile_btn)

        self._file_label = QLabel("  No file")
        toolbar.addWidget(self._file_label)

        layout.addWidget(toolbar)

        self._node_palette = self._create_node_palette()
        layout.addWidget(self._node_palette)

        self._create_graph_view()

    def _create_node_palette(self):
        from editor.NodeGraphQt import BaseNode
        palette = QWidget()
        palette.setMaximumHeight(120)
        palette.setStyleSheet(f"""
                           border-radius: 3px; padding: 4px 8px; font-size: 10px; }}
        """)
        layout = QVBoxLayout(palette)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        categories = {
            "Input": [
                ("Vertex Position", "VertexPosition"),
                ("UV", "UV"),
                ("Normal", "Normal"),
                ("Time", "Time"),
                ("Color", "Color"),
                ("Float", "Float"),
            ],
            "Math": [
                ("Add", "Add"),
                ("Multiply", "Multiply"),
                ("Subtract", "Subtract"),
                ("Lerp", "Lerp"),
                ("Dot Product", "DotProduct"),
                ("Normalize", "Normalize"),
                ("Clamp", "Clamp"),
                ("Step", "Step"),
                ("Fresnel", "Fresnel"),
            ],
            "Texture": [
                ("Texture 2D", "Texture2D"),
            ],
            "Output": [
                ("Vertex Output", "VertexOutput"),
                ("Fragment Output", "FragmentOutput"),
            ],
        }

        from editor.shader_graph.nodes import ALL_NODES
        node_map = {}
        for i, cls in enumerate(ALL_NODES):
            name = cls.NODE_NAME if hasattr(cls, 'NODE_NAME') else cls.__name__
            node_map[name] = cls
        self._node_classes = node_map

        cat_layout = QHBoxLayout()
        cat_layout.setSpacing(4)
        for cat_name, nodes in categories.items():
            cat_frame = QFrame()
            cat_layout_f = QVBoxLayout(cat_frame)
            cat_layout_f.setContentsMargins(4, 2, 4, 2)
            cat_layout_f.setSpacing(2)
            cat_label = QLabel(cat_name)
            cat_layout_f.addWidget(cat_label)
            btn_layout = QHBoxLayout()
            btn_layout.setSpacing(2)
            for display_name, key in nodes:
                btn = QToolButton()
                btn.setText(display_name)
                btn.setToolTip(f"Add {display_name} node")
                btn.clicked.connect(lambda checked=False, k=key: self._add_node(k))
                btn_layout.addWidget(btn)
            cat_layout_f.addLayout(btn_layout)
            cat_layout.addWidget(cat_frame)
        layout.addLayout(cat_layout)

        return palette

    def _create_graph_view(self):
        from editor.NodeGraphQt import NodeGraph
        self._graph = NodeGraph()
        self._graph.register_nodes([cls for cls in self._node_classes.values()])

        viewer = self._graph.viewer()
        self._view = viewer
        self.layout().addWidget(viewer)

    def _add_node(self, key):
        cls = self._node_classes.get(key)
        if cls:
            node = cls()
            self._graph.add_node(node)
            node.set_pos(0, 0)

    def _new_shader(self):
        self._graph.clear()
        self._current_file = None
        self._file_label.setText("  Untitled Shader")
        self._add_node('VertexOutput')
        self._add_node('FragmentOutput')

    def _open_shader(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Shader", "",
            "Shader Files (*.shader);;All Files (*)")
        if path:
            self._load_shader_file(path)

    def _load_shader_file(self, path: str):
        self._current_file = path
        self._file_label.setText(f"  {os.path.basename(path)}")
        self._graph.clear()
        self._add_node('VertexOutput')
        self._add_node('FragmentOutput')
        vo = None
        fo = None
        for n in self._graph.all_nodes():
            cn = type(n).__name__
            if 'VertexOutput' in cn:
                vo = n
                vo.set_pos(-200, 0)
            elif 'FragmentOutput' in cn:
                fo = n
                fo.set_pos(200, 0)

    def _save_shader(self):
        if self._current_file:
            self._compile_and_save(self._current_file)
        else:
            self._save_shader_as()

    def _save_shader_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Shader", "",
            "Shader Files (*.shader)")
        if path:
            if not path.endswith('.shader'):
                path += '.shader'
            self._current_file = path
            self._file_label.setText(f"  {os.path.basename(path)}")
            self._compile_and_save(path)

    def _compile_and_save(self, path: str):
        from editor.shader_graph.code_generator import generate_shader_code
        code = generate_shader_code(self._graph, self._get_shader_name(path))
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(code)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")

    def _get_shader_name(self, path: str):
        basename = os.path.splitext(os.path.basename(path))[0]
        return f"Zarin/{basename}"

    def _generate_and_preview(self):
        from PyQt6.QtWidgets import QPlainTextEdit
        from editor.shader_graph.code_generator import generate_shader_code
        code = generate_shader_code(self._graph, "Preview/Shader")
        preview = QPlainTextEdit()
        preview.setReadOnly(True)
        preview.setPlainText(code)
        preview.resize(700, 500)
        preview.setWindowTitle("Generated Shader Code")
        preview.setStyleSheet(f"""
            QPlainTextEdit {{ background: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace; font-size: 12px; }}
        """)
        preview.show()

class ScriptsPanel(QDockWidget):
    def __init__(self, engine: Engine, parent=None):
        super().__init__("Shaders", parent)
        self._engine = engine
        self.setStyleSheet(_DARK_STYLE)
        self.setObjectName("ShadersDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.setMinimumWidth(200)
        self._shader_widget = _ShaderGraphWidget()
        self.setWidget(self._shader_widget)
