# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import random
import os
from typing import Optional, TYPE_CHECKING
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QToolBar, QToolButton, QLabel,
                             QDoubleSpinBox, QSpinBox,
                             QPushButton, QPlainTextEdit, QFrame, QFileDialog)
from PyQt6.QtCore import Qt, QTimer, QSize
from core.foundation.commands import AddComponentCommand, get_history
from core.components.rendering.terrain import Terrain
from core.components.physics.terrain_collider import TerrainCollider
from core.components.transform import Transform
from core.foundation.logger import Logger
if TYPE_CHECKING:
    from core.ecs.ecs import Entity
    from core.engine.engine import Engine


class _TerrainNodeGraphWidget(QWidget):
    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self._panel = panel
        self._graph = None
        self._view = None
        self._node_classes = {}
        self._graph_path = ""
        self._loading = False
        self._collab_bridge = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QToolBar()
        toolbar.setIconSize(QSize(16, 16))

        save_btn = QToolButton()
        save_btn.setText("Save")
        save_btn.setToolTip("Save graph to .zterr file")
        save_btn.clicked.connect(lambda: self._save_graph())
        toolbar.addWidget(save_btn)

        save_as_btn = QToolButton()
        save_as_btn.setText("Save As")
        save_as_btn.clicked.connect(lambda: self._save_graph_as())
        toolbar.addWidget(save_as_btn)

        open_btn = QToolButton()
        open_btn.setText("Open")
        open_btn.setToolTip("Open .zterr graph file")
        open_btn.clicked.connect(lambda: self._open_graph())
        toolbar.addWidget(open_btn)

        toolbar.addSeparator()

        new_btn = QToolButton()
        new_btn.setText("+ New")
        new_btn.clicked.connect(self._new_graph)
        toolbar.addWidget(new_btn)

        delete_btn = QToolButton()
        delete_btn.setText("Delete")
        delete_btn.setToolTip("Delete selected nodes (Del)")
        delete_btn.clicked.connect(self._delete_selected)
        toolbar.addWidget(delete_btn)

        toolbar.addSeparator()

        preview_btn = QToolButton()
        preview_btn.setText("Preview")
        preview_btn.clicked.connect(self._on_preview)
        toolbar.addWidget(preview_btn)

        self._live_btn = QToolButton()
        self._live_btn.setText("Live")
        self._live_btn.setCheckable(True)
        self._live_btn.toggled.connect(self._on_live_toggle)
        toolbar.addWidget(self._live_btn)

        toolbar.addSeparator()

        res_label = QLabel("Res:")
        toolbar.addWidget(res_label)
        self._res_spin = QSpinBox()
        self._res_spin.setRange(32, 2048)
        self._res_spin.setSingleStep(16)
        self._res_spin.setValue(512)
        self._res_spin.setMaximumWidth(80)
        toolbar.addWidget(self._res_spin)

        world_label = QLabel("World:")
        toolbar.addWidget(world_label)
        self._world_spin = QDoubleSpinBox()
        self._world_spin.setRange(10.0, 100000.0)
        self._world_spin.setSingleStep(10.0)
        self._world_spin.setDecimals(1)
        self._world_spin.setValue(1000.0)
        self._world_spin.setMaximumWidth(100)
        toolbar.addWidget(self._world_spin)

        self._shader_preview_btn = QToolButton()
        self._shader_preview_btn.setText("View GLSL")
        self._shader_preview_btn.clicked.connect(self._show_shader_code)
        toolbar.addWidget(self._shader_preview_btn)

        layout.addWidget(toolbar)

        self._node_palette = self._create_node_palette()
        layout.addWidget(self._node_palette)

        self._create_graph_view()
        viewer = self._graph.viewer()
        self._view = viewer
        self._view.setMinimumHeight(200)
        layout.addWidget(self._view)

        self._graph.node_created.connect(self._on_graph_changed)
        self._graph.nodes_deleted.connect(self._on_graph_changed)
        self._graph.port_connected.connect(self._on_graph_changed)
        self._graph.port_disconnected.connect(self._on_graph_changed)
        self._graph.property_changed.connect(self._on_graph_changed)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(600)
        self._preview_timer.timeout.connect(self._do_update_previews)

        self._view.installEventFilter(self)

        self._collab_bridge = None

    def set_collaboration_manager(self, mgr):
        if self._collab_bridge is None:
            from editor.terrain_graph.collab_bridge import TerrainGraphCollabBridge
            self._collab_bridge = TerrainGraphCollabBridge(self, mgr)
        else:
            self._collab_bridge.set_collaboration_manager(mgr)

    def eventFilter(self, obj, event):
        if obj is self._view and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_Delete:
                self._delete_selected()
                return True
        return super().eventFilter(obj, event)

    def _create_node_palette(self):
        from editor.terrain_graph.nodes import ALL_NODES
        palette = QWidget()
        palette.setMaximumHeight(110)
        palette.setStyleSheet("""
            QToolButton { border-radius: 3px; padding: 3px 6px; font-size: 10px; }
        """)
        main_layout = QVBoxLayout(palette)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(2)

        node_map = {}
        for cls in ALL_NODES:
            name = cls.NODE_NAME if hasattr(cls, 'NODE_NAME') else cls.__name__
            node_map[name] = cls
        self._node_classes = node_map

        categories = {}
        for cls in ALL_NODES:
            name = cls.NODE_NAME if hasattr(cls, 'NODE_NAME') else cls.__name__
            nt = getattr(cls, 'NODE_TYPE', 'value')
            cat = nt.replace("_", " ").title()
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((name, name))

        cat_layout = QHBoxLayout()
        cat_layout.setSpacing(4)
        for cat_name, nodes in categories.items():
            cat_frame = QFrame()
            cat_frame_layout = QVBoxLayout(cat_frame)
            cat_frame_layout.setContentsMargins(4, 2, 4, 2)
            cat_frame_layout.setSpacing(2)
            cat_label = QLabel(cat_name)
            cat_frame_layout.addWidget(cat_label)
            btn_layout = QHBoxLayout()
            btn_layout.setSpacing(2)
            for display_name, key in nodes:
                btn = QToolButton()
                btn.setText(display_name)
                btn.setToolTip(f"Add {display_name} node")
                btn.clicked.connect(lambda checked=False, k=key: self._add_node(k))
                btn_layout.addWidget(btn)
            cat_frame_layout.addLayout(btn_layout)
            cat_layout.addWidget(cat_frame)
        main_layout.addLayout(cat_layout)
        return palette

    def _create_graph_view(self):
        from editor.NodeGraphQt import NodeGraph
        from editor.terrain_graph.nodes import ALL_NODES
        self._graph = NodeGraph()
        self._graph.register_nodes(ALL_NODES)

    def _add_node(self, key):
        for cls in self._node_classes.values():
            if cls.NODE_NAME == key:
                node = cls()
                self._graph.add_node(node)
                node.set_pos(0, 0)
                break

    def _delete_selected(self):
        selected = self._graph.selected_nodes()
        if not selected:
            return
        self._graph.delete_nodes(selected)

    def _new_graph(self):
        self._graph.clear()
        self._graph_path = ""
        self._update_title()

    def _update_title(self):
        name = os.path.splitext(os.path.basename(self._graph_path))[0] if self._graph_path else "untitled"
        self._panel.setWindowTitle(f"Terrain Editor — {name}")

    def _save_graph(self):
        if self._graph_path:
            self._do_save(self._graph_path)
        else:
            self._save_graph_as()

    def _save_graph_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Terrain Graph", self._graph_path or "",
            "Terrain Graphs (*.zterr)")
        if not path:
            return
        if not path.endswith(".zterr"):
            path += ".zterr"
        self._do_save(path)
        self._graph_path = path
        self._update_title()
        if self._panel._terrain is not None:
            self._panel._terrain.graph_path = path

    def _do_save(self, path):
        from editor.terrain_graph.graph_serializer import save_graph
        if save_graph(self._graph, path):
            Logger.info(f"TerrainGraph: saved to {path}")
        else:
            Logger.warning(f"TerrainGraph: failed to save to {path}")

    def _open_graph(self, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Terrain Graph", "",
                "Terrain Graphs (*.zterr)")
            if not path:
                return
        self._loading = True
        from editor.terrain_graph.graph_serializer import load_graph
        ok = load_graph(self._graph, path)
        self._loading = False
        if ok:
            self._graph_path = path
            self._update_title()
            Logger.info(f"TerrainGraph: loaded from {path}")
            if self._panel._terrain is not None:
                self._panel._terrain.graph_path = path
        else:
            Logger.warning(f"TerrainGraph: failed to load from {path}")

    def _on_graph_changed(self, *args):
        if self._loading:
            return
        if self._panel._live:
            if self._panel._terrain is not None or self._panel._find_or_create_terrain():
                self._panel._on_generate()
        self._preview_timer.start()

    def _on_preview(self):
        self._panel._on_generate()
        self._do_update_previews()

    def _do_update_previews(self):
        try:
            from editor.terrain_graph.node_preview import update_all_previews
            res = self._res_spin.value() if self._res_spin else 64
            update_all_previews(self._graph, resolution=res)
        except Exception as e:
            Logger.warning(f"TerrainGraph: preview update failed: {e}")

    def _on_live_toggle(self, enabled):
        self._panel.set_live(enabled)

    def _show_shader_code(self):
        from editor.terrain_graph.code_generator import generate_shader
        res = self._res_spin.value()
        source, uniforms, height_scale = generate_shader(self._graph, res)
        if not source:
            Logger.warning("TerrainGraph: empty graph, cannot generate shader")
            return
        self._preview_window = QPlainTextEdit()
        self._preview_window.setReadOnly(True)
        self._preview_window.setPlainText(source)
        self._preview_window.resize(700, 500)
        self._preview_window.setWindowTitle("Generated Terrain Compute Shader")
        self._preview_window.setStyleSheet("""
            QPlainTextEdit { background: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace; font-size: 12px; }
        """)
        self._preview_window.show()


class TerrainPanel(QDockWidget):
    def __init__(self, engine: Engine, parent=None):
        super().__init__("Terrain Editor", parent)
        self._engine = engine
        self._entity: Optional[Entity] = None
        self._terrain: Optional[Terrain] = None
        self._live = False

        self.setObjectName("TerrainEditorDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable)

        root = QWidget()
        self.setWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        action_bar = QHBoxLayout()
        create_btn = QPushButton("Create Terrain")
        create_btn.setToolTip("Create a new terrain on the selected object")
        create_btn.clicked.connect(self._on_create)
        action_bar.addWidget(create_btn)
        add_col_btn = QPushButton("Add Collider")
        add_col_btn.setToolTip("Add a TerrainCollider to the current terrain")
        add_col_btn.clicked.connect(self._on_add_collider)
        action_bar.addWidget(add_col_btn)
        random_btn = QPushButton("Randomize Seed")
        random_btn.clicked.connect(self._on_randomize)
        action_bar.addWidget(random_btn)
        layout.addLayout(action_bar)

        self._graph_widget = _TerrainNodeGraphWidget(self)
        layout.addWidget(self._graph_widget)

    @property
    def _graph(self):
        return self._graph_widget._graph if self._graph_widget else None

    @property
    def _res_spin(self):
        return self._graph_widget._res_spin if self._graph_widget else None

    @property
    def _world_spin(self):
        return self._graph_widget._world_spin if self._graph_widget else None

    def set_collaboration_manager(self, mgr):
        if self._graph_widget:
            self._graph_widget.set_collaboration_manager(mgr)

    def set_live(self, enabled: bool):
        self._live = enabled
        if enabled and (self._terrain is not None or self._find_or_create_terrain()):
            self._on_generate()

    def load_graph(self, path: str):
        if self._graph_widget:
            self._graph_widget._open_graph(path)
        if self._live and (self._terrain is not None or self._find_or_create_terrain()):
            self._on_generate()

    def _find_or_create_terrain(self):
        if self._terrain is not None:
            return True
        if not self._engine or not self._engine.scene:
            return False
        scene = self._engine.scene
        for e in scene.get_all_entities():
            tc = e.get_component(Terrain)
            if tc is not None:
                self._entity = e
                self._terrain = tc
                Logger.info("TerrainGraph: auto-selected terrain entity '{}'".format(getattr(e, '_name', '?')))
                return True
        return False

    def _on_generate(self):
        if self._graph is None:
            Logger.info("TerrainGraph: no graph, skipping generate")
            return
        if self._terrain is None:
            if not self._find_or_create_terrain():
                Logger.info("TerrainGraph: no terrain component in scene, skipping generate")
                return
        nodes = self._graph.all_nodes()
        if not nodes:
            Logger.info("TerrainGraph: graph has no nodes")
            return
        from editor.terrain_graph.code_generator import generate_shader
        from editor.terrain_graph.gpu_runner import run_shader, clear_cache
        clear_cache()
        res = int(self._res_spin.value()) if self._res_spin else 512
        seed = random.randint(0, 100000)
        source, uniforms, height_scale = generate_shader(self._graph, res)
        if not source:
            Logger.warning("TerrainGraph: code generator returned empty (need Height Output node connected)")
            return
        hf = run_shader(source, res, uniforms)
        if hf is None:
            Logger.warning("TerrainGraph: GPU shader execution failed (check GLSL)")
            return
        Logger.info(f"TerrainGraph: generated {hf.shape[0]}x{hf.shape[1]} heightfield, hscale={height_scale:.1f}")
        self._apply_heightfield(hf, height_scale)

    def _apply_heightfield(self, hf, height_scale=1.0):
        if self._terrain is None:
            return
        ws = float(self._world_spin.value()) if self._world_spin else 1000.0
        self._terrain.world_size = ws
        from core.terrain.terrain_generator import get_generator
        mesh = get_generator().mesh_from_heightfield(hf, ws)
        if mesh is None:
            Logger.warning("TerrainGraph: mesh_from_heightfield returned None")
            return
        self._terrain.set_heightfield(hf, mesh)
        tc = self._entity.get_component(TerrainCollider) if self._entity else None
        if tc is not None:
            tc.set_height_data(hf)
            tc.resolution = hf.shape[0]
            tc.size = self._terrain_size_vec(height_scale)
            tc.height_scale = height_scale
        if self._engine and self._engine.scene:
            self._engine.scene._render_version += 1
        self._collab_sync_terrain()

    def _collab_sync_terrain(self):
        mgr = getattr(self._engine, "collab_manager", None)
        if not mgr or not mgr.connected or not self._entity or not self._terrain:
            return
        eid = self._entity.id
        data = {
            "world_size": self._terrain.world_size,
            "graph_path": self._graph_widget._graph_path if self._graph_widget else "",
        }
        mgr.send_component_sync(eid, "Terrain", data)
        tc = self._entity.get_component(TerrainCollider)
        if tc is not None:
            cd = tc.serialize()
            cd.pop("height_data", None)
            mgr.send_component_sync(eid, "TerrainCollider", cd)

    def _terrain_size_vec(self, height_scale=1.0):
        from core.math.math3d import Vec3
        ws = float(self._world_spin.value()) if self._world_spin else 1000.0
        return Vec3(ws, height_scale, ws)

    def set_entity(self, entity: Optional[Entity]):
        self._entity = entity
        self._terrain = None
        if entity is None:
            return
        comp = entity.get_component(Terrain)
        self._terrain = comp

    def _on_create(self):
        entity = self._get_selected_entity()
        created_entity = False
        if entity is None:
            scene = self._engine.scene
            if not scene:
                return
            entity = scene.create_entity("Terrain")
            entity.add_component(Transform())
            created_entity = True
        if entity.get_component(Terrain) is None:
            terrain = Terrain()
            entity.add_component(terrain)
            self._terrain = terrain
        else:
            self._terrain = entity.get_component(Terrain)
        self.set_entity(entity)
        self._prompt_save_graph()
        self._collab_sync_create(entity, created_entity)
        if self._live:
            self._on_generate()

    def _collab_sync_create(self, entity, created_entity: bool):
        mgr = getattr(self._engine, "collab_manager", None)
        if not mgr or not mgr.connected:
            return
        if created_entity:
            mgr.send_entity_create(entity.serialize())
        else:
            terrain = entity.get_component(Terrain)
            if terrain is not None:
                data = terrain.serialize()
                data.pop("heightfield", None)
                mgr.send_component_add(entity.id, "Terrain", data)

    def _prompt_save_graph(self):
        if self._graph is None:
            return
        nodes = self._graph.all_nodes()
        if not nodes:
            return
        gw = self._graph_widget
        if gw._graph_path:
            gw._do_save(gw._graph_path)
        else:
            gw._save_graph_as()

    def _on_add_collider(self):
        entity = self._get_selected_entity()
        if entity is None and self._entity is not None:
            entity = self._entity
        if entity is None:
            return
        if entity.get_component(Terrain) is None:
            Logger.warning("Terrain Editor: select a terrain object first")
            return
        added_now = False
        if entity.get_component(TerrainCollider) is None:
            history = get_history()
            if history is not None:
                history.execute(AddComponentCommand(entity, TerrainCollider()))
            else:
                entity.add_component(TerrainCollider())
            added_now = True
        tc = entity.get_component(TerrainCollider)
        if tc is not None:
            hs = 1.0
            if self._terrain and self._terrain.heightfield is not None:
                hs = self._terrain.height_scale
            tc.size = self._terrain_size_vec(hs)
            tc.height_scale = hs
            tc.resolution = int(self._res_spin.value()) if self._res_spin else 256
            self._collab_sync_collider(entity, tc, added_now)

    def _collab_sync_collider(self, entity, tc, added_now: bool):
        mgr = getattr(self._engine, "collab_manager", None)
        if not mgr or not mgr.connected:
            return
        eid = entity.id
        if added_now:
            mgr.send_component_add(eid, "TerrainCollider", tc.serialize())
        else:
            cd = tc.serialize()
            cd.pop("height_data", None)
            mgr.send_component_sync(eid, "TerrainCollider", cd)

    def _on_randomize(self):
        if self._graph is None:
            return
        for n in self._graph.all_nodes():
            params = getattr(n, '_PARAMS', {})
            if 'seed' in params:
                n.seed = float(random.randint(0, 100000))
        if self._terrain is not None:
            self._on_generate()

    def _get_selected_entity(self) -> Optional[Entity]:
        from core.engine.engine import Engine
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
