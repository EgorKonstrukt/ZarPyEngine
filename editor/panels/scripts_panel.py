# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import re
from typing import Optional, TYPE_CHECKING
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                              QSplitter, QTabWidget, QToolBar, QToolButton,
                              QLabel, QLineEdit, QFileDialog, QMenu,
                              QPlainTextEdit, QFrame, QMessageBox,
                              QComboBox, QSizePolicy)
from PyQt6.QtCore import Qt, QSize, QRegularExpression, QThreadPool, QRunnable, pyqtSignal, QObject
from PyQt6.QtGui import (QFont, QColor, QPainter, QSyntaxHighlighter,
                          QTextCharFormat, QKeySequence, QTextCursor,
                          QFontMetrics, QPalette)
if TYPE_CHECKING:
    from core.engine.engine import Engine

from core.config.editor_scale import scale

_DARK_STYLE = ""

class _PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rules = []

        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor("#569cd6"))
        kw_fmt.setFontWeight(QFont.Weight.Bold)
        keywords = (r'\b(def|class|return|if|elif|else|for|while|import|from|'
                    r'try|except|finally|with|as|yield|lambda|pass|break|'
                    r'continue|and|or|not|in|is|True|False|None|self|print|'
                    r'raise|del|global|nonlocal|assert|async|await)\b')
        self._rules.append((QRegularExpression(keywords), kw_fmt))

        builtins_fmt = QTextCharFormat()
        builtins_fmt.setForeground(QColor("#4ec9b0"))
        builtins = r'\b(int|float|str|bool|list|dict|tuple|set|type|len|range|enumerate|zip|map|filter|super|property|staticmethod|classmethod|isinstance|issubclass|hasattr|getattr|setattr|open|abs|min|max|sum|sorted|reversed|any|all|iter|next|object|Exception|ValueError|TypeError|KeyError|IndexError)\b'
        self._rules.append((QRegularExpression(builtins), builtins_fmt))

        string_fmt = QTextCharFormat()
        string_fmt.setForeground(QColor("#ce9178"))
        self._rules.append((QRegularExpression(r'""".*?"""|\'\'\'.*?\'\'\'|".*?"|\'.*?\''), string_fmt))

        number_fmt = QTextCharFormat()
        number_fmt.setForeground(QColor("#b5cea8"))
        self._rules.append((QRegularExpression(r'\b\d+\.?\d*\b'), number_fmt))

        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor("#6a9955"))
        comment_fmt.setFontItalic(True)
        self._rules.append((QRegularExpression(r'#[^\n]*'), comment_fmt))

        decorator_fmt = QTextCharFormat()
        decorator_fmt.setForeground(QColor("#dcdcaa"))
        self._rules.append((QRegularExpression(r'@\w+'), decorator_fmt))

        func_fmt = QTextCharFormat()
        func_fmt.setForeground(QColor("#dcdcaa"))
        self._rules.append((QRegularExpression(r'\b\w+(?=\()'), func_fmt))

    def highlightBlock(self, text):
        for pattern, fmt in self._rules:
            match_it = pattern.globalMatch(text)
            while match_it.hasNext():
                match = match_it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)

class _CodeEditor(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabStopDistance(QFontMetrics(QFont("Consolas", 10)).horizontalAdvance(' ') * 4)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Tab:
            cursor = self.textCursor()
            cursor.insertText("    ")
            return
        super().keyPressEvent(event)

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
        from editor.shader_graph.code_generator import generate_shader_code
        code = generate_shader_code(self._graph, "Preview/Shader")
        preview = _CodeEditor()
        preview.setReadOnly(True)
        preview.setPlainText(code)
        preview.resize(700, 500)
        preview.setWindowTitle("Generated Shader Code")
        preview.setStyleSheet(f"""
            QPlainTextEdit {{ background: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace; font-size: 12px; }}
        """)
        preview.show()

class _ScriptEditorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_file = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QToolBar()
        toolbar.setStyleSheet(f"""
        """)

        new_btn = QToolButton()
        new_btn.setText("+ New")
        new_btn.clicked.connect(self._new_script)
        toolbar.addWidget(new_btn)

        open_btn = QToolButton()
        open_btn.setText("Open")
        open_btn.clicked.connect(self._open_script)
        toolbar.addWidget(open_btn)

        save_btn = QToolButton()
        save_btn.setText("Save")
        save_btn.clicked.connect(self._save_script)
        toolbar.addWidget(save_btn)

        save_as_btn = QToolButton()
        save_as_btn.setText("Save As")
        save_as_btn.clicked.connect(self._save_script_as)
        toolbar.addWidget(save_as_btn)

        toolbar.addSeparator()

        self._file_label = QLabel("  No file")
        toolbar.addWidget(self._file_label)

        layout.addWidget(toolbar)

        self._editor = _CodeEditor()
        self._highlighter = _PythonHighlighter(self._editor.document())
        self._editor.setStyleSheet(f"""
            QPlainTextEdit {{
                background: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                font-size: 12px;
                border: none;
                selection-background-color: #264f78;
            }}
        """)
        layout.addWidget(self._editor)

    def _new_script(self):
        self._editor.clear()
        self._current_file = None
        self._file_label.setText("  Untitled Script")

    def _open_script(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Script", "",
            "Python Files (*.py);;All Files (*)")
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self._editor.setPlainText(content)
                self._current_file = path
                self._file_label.setText(f"  {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "Open Error", f"Failed to open:\n{e}")

    def _save_script(self):
        if self._current_file:
            try:
                with open(self._current_file, 'w', encoding='utf-8') as f:
                    f.write(self._editor.toPlainText())
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")
        else:
            self._save_script_as()

    def _save_script_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Script", "",
            "Python Files (*.py)")
        if path:
            if not path.endswith('.py'):
                path += '.py'
            self._current_file = path
            self._file_label.setText(f"  {os.path.basename(path)}")
            self._save_script()

class ScriptsPanel(QDockWidget):
    def __init__(self, engine: Engine, parent=None):
        super().__init__("Scripts", parent)
        self._engine = engine
        self.setStyleSheet(_DARK_STYLE)
        self.setObjectName("ScriptsDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.setMinimumWidth(200)

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)

        self._shader_widget = _ShaderGraphWidget()
        tabs.addTab(self._shader_widget, "Shader Graph")

        self._script_widget = _ScriptEditorWidget()
        tabs.addTab(self._script_widget, "Script Editor")

        self.setWidget(tabs)
