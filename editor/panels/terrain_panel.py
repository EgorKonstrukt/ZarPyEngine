# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import time
import numpy as np
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QDoubleSpinBox,
                             QSpinBox, QGroupBox, QGridLayout, QScrollArea,
                             QFrame, QCheckBox, QSlider, QTabWidget, QLineEdit)
from PyQt6.QtCore import Qt, QTimer
from core.ecs.ecs import ComponentRegistry
from core.foundation.commands import AddComponentCommand, get_history
from core.components.rendering.terrain import Terrain
from core.components.physics.terrain_collider import TerrainCollider
from core.components.transform import Transform
from core.terrain.terrain_generator import TerrainSettings, _DEFAULTS
from core.terrain.terrain_worker import get_worker
from core.foundation.logger import Logger
if TYPE_CHECKING:
    from core.ecs.ecs import Entity
    from core.engine.engine import Engine

_FLOAT_KEYS = [
    ("baseFrequency", "Base Frequency", 0.0, 0.2, 0.001, 4),
    ("lacunarity", "Lacunarity", 0.5, 4.0, 0.01, 3),
    ("persistence", "Persistence", 0.1, 1.0, 0.01, 3),
    ("heightScale", "Height Scale", 0.0, 500.0, 0.5, 2),
    ("offset", "Height Offset", -200.0, 200.0, 0.5, 2),
    ("warpeness", "Domain Warp", 0.0, 1.0, 0.01, 3),
    ("warpFrequency", "Warp Frequency", 0.0, 0.2, 0.001, 4),
    ("warpIterations", "Warp Iterations", 1.0, 4.0, 1.0, 0),
    ("ridge", "Ridged Mix", 0.0, 1.0, 0.01, 3),
    ("ridgePower", "Ridge Power", 0.5, 8.0, 0.1, 2),
    ("ridgeSharpness", "Ridge Sharpness", 0.5, 1.0, 0.01, 3),
    ("billow", "Billow Mix", 0.0, 1.0, 0.01, 3),
    ("billowPower", "Billow Power", 0.5, 6.0, 0.1, 2),
    ("continentMask", "Continent Mask", 0.0, 1.0, 0.01, 3),
    ("continentScale", "Continent Scale", 0.0005, 0.05, 0.0005, 5),
    ("continentFalloff", "Continent Falloff", 0.2, 4.0, 0.01, 3),
    ("detail", "Detail Strength", 0.0, 1.0, 0.01, 3),
    ("detailFrequency", "Detail Frequency", 1.0, 16.0, 0.1, 2),
    ("strata", "Strata Strength", 0.0, 1.0, 0.01, 3),
    ("strataScale", "Strata Scale", 1.0, 40.0, 0.5, 2),
    ("plateau", "Plateau Sharpness", 0.0, 1.0, 0.01, 3),
    ("plateauLevel", "Plateau Level", -1.0, 1.0, 0.01, 3),
    ("slopeMask", "Slope Mask", 0.0, 1.0, 0.01, 3),
    ("slopeMin", "Slope Min", 0.0, 1.5, 0.01, 3),
    ("thermalErosion", "Thermal Erosion", 0.0, 1.0, 0.01, 3),
    ("hydraulicErosion", "Hydraulic Erosion", 0.0, 1.0, 0.01, 3),
    ("erosionIterations", "Erosion Iterations", 1.0, 24.0, 1.0, 0),
    ("talus", "Talus Angle", 0.001, 0.2, 0.001, 4),
    ("peakSmoothing", "Peak Smoothing", 0.0, 1.0, 0.01, 3),
    ("valleyDepth", "Valley Depth", 0.0, 1.0, 0.01, 3),
    ("riverStrength", "River Carve", 0.0, 1.0, 0.01, 3),
    ("terrace", "Terrace Mix", 0.0, 1.0, 0.01, 3),
    ("terraceSteps", "Terrace Steps", 2.0, 32.0, 1.0, 0),
    ("dune", "Dune Mix", 0.0, 1.0, 0.01, 3),
    ("duneDir", "Dune Direction", 0.0, 6.283, 0.01, 3),
    ("fractalTwist", "Fractal Twist", 0.0, 1.0, 0.01, 3),
    ("sharpen", "Sharpen", 0.0, 2.0, 0.01, 3),
    ("heightBias", "Height Bias", -1.0, 1.0, 0.01, 3),
    ("noiseSeed", "Seed", 0.0, 100000.0, 1.0, 0),
]

_INT_KEYS = [
    ("resolution", "Resolution", 32, 1024, 16),
    ("octaves", "Octaves", 1, 16, 1),
    ("flipY", "Flip Y", 0, 1, 1),
]


def _make_spin(val, lo, hi, step, decimals):
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    sb.setValue(val)
    sb.setMinimumWidth(70)
    return sb


class TerrainPanel(QDockWidget):
    def __init__(self, engine: Engine, parent=None):
        super().__init__("Terrain Editor", parent)
        self._engine = engine
        self._entity: Optional[Entity] = None
        self._terrain: Optional[Terrain] = None
        self._float_widgets: dict = {}
        self._int_widgets: dict = {}
        self._live = False
        self._dragging = False
        self._last_request = 0.0
        self._pending_token = 0
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(40)
        self._live_timer.timeout.connect(self._poll_worker)

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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(scroll_content)
        self._scroll_layout.setContentsMargins(4, 4, 4, 4)
        self._scroll_layout.setSpacing(4)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        self._build_action_section()
        self._build_tabs()
        self._connect_live_signals()
        self._scroll_layout.addStretch()
        self._refresh_enabled(False)

    def _build_action_section(self):
        gb = QGroupBox("Terrain")
        v = QVBoxLayout(gb)
        row = QHBoxLayout()
        create_btn = QPushButton("Create Terrain")
        create_btn.setToolTip("Create a new terrain on the selected object (or a new object)")
        create_btn.clicked.connect(self._on_create)
        row.addWidget(create_btn)
        add_col_btn = QPushButton("Add Collider")
        add_col_btn.setToolTip("Add a TerrainCollider to the current terrain object")
        add_col_btn.clicked.connect(self._on_add_collider)
        row.addWidget(add_col_btn)
        v.addLayout(row)
        gen_btn = QPushButton("Generate")
        gen_btn.setToolTip("Generate terrain from current settings")
        gen_btn.clicked.connect(self._on_generate)
        v.addWidget(gen_btn)
        rand_btn = QPushButton("Randomize Seed")
        rand_btn.clicked.connect(self._on_randomize)
        v.addWidget(rand_btn)
        self._live_check = QCheckBox("Live Preview")
        self._live_check.setToolTip("Real-time progressive generation in a background thread (adaptive resolution)")
        self._live_check.toggled.connect(self.set_live)
        v.addWidget(self._live_check)
        self._scroll_layout.addWidget(gb)

    def _build_tabs(self):
        tabs = QTabWidget()
        self._scroll_layout.addWidget(tabs)
        tab_noise = QWidget()
        tab_erosion = QWidget()
        tab_shapes = QWidget()
        tab_world = QWidget()
        tabs.addTab(tab_noise, "Noise")
        tabs.addTab(tab_erosion, "Erosion")
        tabs.addTab(tab_shapes, "Shapes")
        tabs.addTab(tab_world, "World")
        self._build_noise_tab(tab_noise)
        self._build_erosion_tab(tab_erosion)
        self._build_shapes_tab(tab_shapes)
        self._build_world_tab(tab_world)

    def _add_float_controls(self, container_layout, keys):
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        r = 0
        for key, label, lo, hi, step, dec in keys:
            lab = QLabel(label)
            lab.setToolTip(f"{label} ({key})")
            val = _DEFAULTS.get(key, 0.0)
            sb = _make_spin(val, lo, hi, step, dec)
            grid.addWidget(lab, r, 0)
            grid.addWidget(sb, r, 1)
            self._float_widgets[key] = sb
            r += 1
        container_layout.addLayout(grid)

    def _build_noise_tab(self, tab):
        v = QVBoxLayout(tab)
        gb = QGroupBox("Fractal Noise")
        gv = QVBoxLayout(gb)
        self._add_float_controls(gv, [
            ("baseFrequency", "Base Frequency", 0.0, 0.2, 0.001, 4),
            ("lacunarity", "Lacunarity", 0.5, 4.0, 0.01, 3),
            ("persistence", "Persistence", 0.1, 1.0, 0.01, 3),
            ("octaves", "Octaves", 1.0, 16.0, 1.0, 0),
            ("fractalTwist", "Fractal Twist", 0.0, 1.0, 0.01, 3),
            ("noiseSeed", "Seed", 0.0, 100000.0, 1.0, 0),
        ])
        v.addWidget(gb)

        gb2 = QGroupBox("Domain Warp")
        gv2 = QVBoxLayout(gb2)
        self._add_float_controls(gv2, [
            ("warpeness", "Warp Strength", 0.0, 1.0, 0.01, 3),
            ("warpFrequency", "Warp Frequency", 0.0, 0.2, 0.001, 4),
            ("warpIterations", "Warp Iterations", 1.0, 4.0, 1.0, 0),
            ("detail", "Detail Strength", 0.0, 1.0, 0.01, 3),
            ("detailFrequency", "Detail Frequency", 1.0, 16.0, 0.1, 2),
        ])
        v.addWidget(gb2)

        gb3 = QGroupBox("Blend Modes")
        gv3 = QVBoxLayout(gb3)
        self._add_float_controls(gv3, [
            ("ridge", "Ridged Mix", 0.0, 1.0, 0.01, 3),
            ("ridgePower", "Ridge Power", 0.5, 8.0, 0.1, 2),
            ("ridgeSharpness", "Ridge Sharpness", 0.5, 1.0, 0.01, 3),
            ("billow", "Billow Mix", 0.0, 1.0, 0.01, 3),
            ("billowPower", "Billow Power", 0.5, 6.0, 0.1, 2),
        ])
        v.addWidget(gb3)
        v.addStretch()

    def _build_erosion_tab(self, tab):
        v = QVBoxLayout(tab)
        gb = QGroupBox("Thermal Erosion")
        gv = QVBoxLayout(gb)
        self._add_float_controls(gv, [
            ("thermalErosion", "Strength", 0.0, 1.0, 0.01, 3),
            ("talus", "Talus Angle", 0.001, 0.2, 0.001, 4),
            ("erosionIterations", "Iterations", 1.0, 24.0, 1.0, 0),
        ])
        v.addWidget(gb)
        gb2 = QGroupBox("Hydraulic Erosion")
        gv2 = QVBoxLayout(gb2)
        self._add_float_controls(gv2, [
            ("hydraulicErosion", "Strength", 0.0, 1.0, 0.01, 3),
            ("sedimentCapacity", "Sediment Capacity", 0.0, 12.0, 0.1, 2),
            ("erosionStrength", "Erosion Strength", 0.0, 1.0, 0.01, 3),
        ])
        v.addWidget(gb2)
        gb3 = QGroupBox("Relief Shaping")
        gv3 = QVBoxLayout(gb3)
        self._add_float_controls(gv3, [
            ("peakSmoothing", "Peak Smoothing", 0.0, 1.0, 0.01, 3),
            ("valleyDepth", "Valley Depth", 0.0, 1.0, 0.01, 3),
            ("riverStrength", "River Carve", 0.0, 1.0, 0.01, 3),
            ("sharpen", "Sharpen", 0.0, 2.0, 0.01, 3),
        ])
        v.addWidget(gb3)
        v.addStretch()

    def _build_shapes_tab(self, tab):
        v = QVBoxLayout(tab)
        gb = QGroupBox("Continents")
        gv = QVBoxLayout(gb)
        self._add_float_controls(gv, [
            ("continentMask", "Mask Strength", 0.0, 1.0, 0.01, 3),
            ("continentScale", "Continent Scale", 0.0005, 0.05, 0.0005, 5),
            ("continentFalloff", "Continent Falloff", 0.2, 4.0, 0.01, 3),
        ])
        v.addWidget(gb)
        gb2 = QGroupBox("Plateaus & Terraces")
        gv2 = QVBoxLayout(gb2)
        self._add_float_controls(gv2, [
            ("plateau", "Plateau Sharpness", 0.0, 1.0, 0.01, 3),
            ("plateauLevel", "Plateau Level", -1.0, 1.0, 0.01, 3),
            ("terrace", "Terrace Mix", 0.0, 1.0, 0.01, 3),
            ("terraceSteps", "Terrace Steps", 2.0, 32.0, 1.0, 0),
            ("strata", "Strata Strength", 0.0, 1.0, 0.01, 3),
            ("strataScale", "Strata Scale", 1.0, 40.0, 0.5, 2),
        ])
        v.addWidget(gb2)
        gb3 = QGroupBox("Dunes & Slopes")
        gv3 = QVBoxLayout(gb3)
        self._add_float_controls(gv3, [
            ("dune", "Dune Mix", 0.0, 1.0, 0.01, 3),
            ("duneDir", "Dune Direction", 0.0, 6.283, 0.01, 3),
            ("slopeMask", "Slope Mask", 0.0, 1.0, 0.01, 3),
            ("slopeMin", "Slope Min", 0.0, 1.5, 0.01, 3),
        ])
        v.addWidget(gb3)
        v.addStretch()

    def _build_world_tab(self, tab):
        v = QVBoxLayout(tab)
        gb = QGroupBox("World")
        gv = QVBoxLayout(gb)
        self._world_size = _make_spin(1000.0, 10.0, 100000.0, 10.0, 1)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("World Size"))
        size_row.addWidget(self._world_size)
        gv.addLayout(size_row)
        self._height_scale = self._add_named_spin(gv, "Height Scale", "heightScale", 1000.0, 0.0, 500.0, 0.5, 2)
        self._offset = self._add_named_spin(gv, "Height Offset", "offset", 0.0, -200.0, 200.0, 0.5, 2)
        self._height_bias = self._add_named_spin(gv, "Height Bias", "heightBias", 0.0, -1.0, 1.0, 0.01, 3)
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution"))
        self._res_spin = QSpinBox()
        self._res_spin.setRange(32, 2048)
        self._res_spin.setSingleStep(16)
        self._res_spin.setValue(512)
        res_row.addWidget(self._res_spin)
        gv.addLayout(res_row)
        self._flip = QCheckBox("Flip Y")
        gv.addWidget(self._flip)
        v.addWidget(gb)
        v.addStretch()

    def _add_named_spin(self, parent_layout, label, key, default, lo, hi, step, dec):
        sb = _make_spin(default, lo, hi, step, dec)
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(sb)
        parent_layout.addLayout(row)
        self._float_widgets[key] = sb
        return sb

    def _connect_live_signals(self):
        for sb in self._float_widgets.values():
            sb.valueChanged.connect(self._on_live_changed)
            sb.installEventFilter(self)
        if hasattr(self, "_res_spin"):
            self._res_spin.valueChanged.connect(self._on_live_changed)
            self._res_spin.installEventFilter(self)
        if hasattr(self, "_flip"):
            self._flip.stateChanged.connect(lambda _: self._on_live_changed())
        self._world_size.installEventFilter(self)
        self._world_size.valueChanged.connect(self._on_live_changed)

    def eventFilter(self, obj, event):
        if obj in self._float_widgets.values() or obj is self._world_size or (hasattr(self, "_res_spin") and obj is self._res_spin):
            if event.type() == event.Type.MouseButtonPress or event.type() == event.Type.Wheel:
                self._on_drag_start()
                QTimer.singleShot(250, self._on_drag_end)
            elif event.type() == event.Type.KeyPress:
                self._on_drag_start()
                QTimer.singleShot(250, self._on_drag_end)
        return super().eventFilter(obj, event)

    def set_live(self, enabled: bool):
        self._live = enabled
        if enabled:
            if not self._live_timer.isActive():
                self._live_timer.start()
            if self._terrain is not None:
                self._request_now()
        else:
            self._live_timer.stop()

    def _on_drag_start(self):
        self._dragging = True

    def _on_drag_end(self):
        self._dragging = False
        self._request_now()

    def _on_live_changed(self):
        if not self._live or self._terrain is None:
            return
        now = time.time()
        if now - self._last_request < 0.03:
            return
        self._last_request = now
        if not self._dragging:
            self._request_now()
        else:
            QTimer.singleShot(30, self._request_now)

    def _adaptive_resolution(self) -> int:
        if self._dragging:
            return 128
        return int(self._res_spin.value()) if hasattr(self, "_res_spin") else 512

    def _request_now(self):
        if self._terrain is None:
            return
        settings = self._collect_settings()
        self._terrain.settings = settings
        if hasattr(self, "_world_size"):
            self._terrain.world_size = float(self._world_size.value())
        res = self._adaptive_resolution()
        self._pending_token = get_worker().request(settings.to_dict(), res)

    def _poll_worker(self):
        worker = get_worker()
        while True:
            res = worker.consume_result()
            if res is None:
                break
            token, hf = res
            if token != self._pending_token:
                continue
            self._apply_heightfield(hf)

    def _apply_heightfield(self, hf: np.ndarray):
        if self._terrain is None:
            return
        from core.terrain.terrain_generator import get_generator
        size = float(self._world_size.value()) if hasattr(self, "_world_size") else 1000.0
        mesh = get_generator().mesh_from_heightfield(hf, size)
        if mesh is None:
            return
        self._terrain.set_heightfield(hf, mesh)
        tc = self._entity.get_component(TerrainCollider) if self._entity else None
        if tc is not None:
            tc.set_height_data(hf)
            tc.resolution = hf.shape[0]
            tc.size = self._terrain_size_vec()
            tc.height_scale = self._float_widgets["heightScale"].value()
        if self._engine and self._engine.scene:
            self._engine.scene._render_version += 1

    def _refresh_enabled(self, enabled: bool):
        for w in list(self._float_widgets.values()) + list(self._int_widgets.values()):
            try:
                w.setEnabled(enabled)
            except Exception:
                pass
        if hasattr(self, "_world_size"):
            self._world_size.setEnabled(enabled)
        if hasattr(self, "_res_spin"):
            self._res_spin.setEnabled(enabled)
        if hasattr(self, "_flip"):
            self._flip.setEnabled(enabled)

    def set_entity(self, entity: Optional[Entity]):
        self._entity = entity
        self._terrain = None
        if entity is None:
            self._refresh_enabled(False)
            return
        comp = entity.get_component(Terrain)
        self._terrain = comp
        if comp is None:
            self._refresh_enabled(False)
            return
        self._refresh_enabled(True)
        self._push_to_ui(comp)

    def _push_to_ui(self, terrain: Terrain):
        d = terrain.settings.data
        for key, sb in self._float_widgets.items():
            if key in d:
                try:
                    sb.setValue(float(d[key]))
                except Exception:
                    pass
        if hasattr(self, "_world_size"):
            self._world_size.setValue(float(terrain.world_size))
        if hasattr(self, "_res_spin"):
            self._res_spin.setValue(int(terrain.settings.get("resolution")))
        if hasattr(self, "_flip"):
            self._flip.setChecked(int(terrain.settings.get("flipY")) == 1)

    def _collect_settings(self) -> TerrainSettings:
        ts = TerrainSettings()
        for key, sb in self._float_widgets.items():
            try:
                ts.set(key, float(sb.value()))
            except Exception:
                pass
        if hasattr(self, "_res_spin"):
            ts.set("resolution", int(self._res_spin.value()))
        if hasattr(self, "_flip"):
            ts.set("flipY", 1 if self._flip.isChecked() else 0)
        return ts

    def _on_create(self):
        from core.ecs.ecs import Entity as EcsEntity
        entity = self._get_selected_entity()
        if entity is None:
            scene = self._engine.scene
            if not scene:
                return
            entity = scene.create_entity("Terrain")
            entity.add_component(Transform())
        if entity.get_component(Terrain) is None:
            terrain = Terrain()
            entity.add_component(terrain)
            self._terrain = terrain
        else:
            self._terrain = entity.get_component(Terrain)
        self._terrain.world_size = self._world_size.value() if hasattr(self, "_world_size") else 1000.0
        self._terrain.settings = self._collect_settings()
        self._refresh_enabled(True)
        self.set_entity(entity)
        if self._live_check.isChecked():
            self._request_now()
        else:
            self._on_generate()
        if self._engine.scene:
            self._engine.scene._render_version += 1

    def _on_add_collider(self):
        entity = self._get_selected_entity()
        if entity is None and self._entity is not None:
            entity = self._entity
        if entity is None:
            return
        if entity.get_component(Terrain) is None:
            Logger.warning("Terrain Editor: select a terrain object first")
            return
        if entity.get_component(TerrainCollider) is None:
            history = get_history()
            if history is not None:
                history.execute(AddComponentCommand(entity, TerrainCollider()))
            else:
                entity.add_component(TerrainCollider())
        tc = entity.get_component(TerrainCollider)
        if tc is not None:
            tc.size = self._terrain_size_vec()
            tc.height_scale = self._float_widgets["heightScale"].value()
            tc.resolution = int(self._res_spin.value()) if hasattr(self, "_res_spin") else 256

    def _terrain_size_vec(self):
        from core.math.math3d import Vec3
        hs = self._float_widgets["heightScale"].value()
        ws = self._world_size.value() if hasattr(self, "_world_size") else 1000.0
        return Vec3(ws, hs, ws)

    def _on_generate(self):
        if self._terrain is None:
            Logger.warning("Terrain Editor: no terrain selected")
            return
        settings = self._collect_settings()
        self._terrain.settings = settings
        if hasattr(self, "_world_size"):
            self._terrain.world_size = float(self._world_size.value())
        if self._live:
            self._dragging = False
            self._request_now()
            return
        ok = self._terrain.generate()
        if not ok:
            Logger.warning("Terrain Editor: generation failed (no GL context?)")
            return
        tc = self._entity.get_component(TerrainCollider) if self._entity else None
        if tc is not None:
            tc.set_height_data(self._terrain.heightfield)
            tc.resolution = self._terrain.heightfield.shape[0]
            tc.size = self._terrain_size_vec()
            tc.height_scale = self._float_widgets["heightScale"].value()
        self._terrain._gpu_dirty = True
        if self._engine and self._engine.scene:
            self._engine.scene._render_version += 1

    def _on_randomize(self):
        import random
        self._float_widgets["noiseSeed"].setValue(float(random.randint(0, 100000)))
        if self._terrain is not None:
            self._terrain.settings.set("noiseSeed", float(random.randint(0, 100000)))
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
