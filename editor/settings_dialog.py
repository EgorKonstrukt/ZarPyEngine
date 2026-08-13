# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional, Any

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget,
                             QWidget, QFormLayout, QLineEdit, QDoubleSpinBox,
                             QSpinBox, QCheckBox, QPushButton, QListWidgetItem,
                             QStackedWidget, QFrame, QScrollArea, QLabel,
                             QSlider, QApplication, QGroupBox,
                             QGridLayout, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal
from core.config.config import Config
from core.physics.collision_layers import MAX_LAYERS, DEFAULT_LAYER_NAMES
from core.config.editor_scale import scale, scale_xy

try:
    import qtawesome as qta
except ImportError:
    qta = None

SECTION_ICONS = {
    "editor": "fa5s.edit",
    "camera": "fa5s.camera",
    "rendering": "fa5s.palette",
    "gizmo": "fa5s.arrows-alt",
    "viewport": "fa5s.desktop",
    "console": "fa5s.terminal",
    "terminal": "fa5s.window-maximize",
    "profiler": "fa5s.chart-bar",
    "hierarchy": "fa5s.sitemap",
    "inspector": "fa5s.search",
    "project": "fa5s.folder-open",
    "engine": "fa5s.cogs",
    "collab": "fa5s.users",
    "undo": "fa5s.undo",
    "input": "fa5s.keyboard",
    "physics": "fa5s.atom",
    "audio": "fa5s.music",
    "toolbar": "fa5s.wrench",
    "file_assoc": "fa5s.file-code",
    "mesh_preview": "fa5s.cube",
}

SECTION_DESCRIPTIONS = {
    "editor": "Editor appearance and behavior",
    "camera": "Viewport camera settings",
    "rendering": "Graphics and rendering quality",
    "gizmo": "Transform gizmo appearance and behavior",
    "viewport": "Viewport display settings",
    "console": "Console log settings",
    "terminal": "Embedded terminal settings",
    "profiler": "Performance profiler settings",
    "hierarchy": "Hierarchy panel settings",
    "inspector": "Inspector panel settings",
    "project": "Project browser settings",
    "engine": "Engine time and physics settings",
    "collab": "Collaboration network settings",
    "undo": "Undo history settings",
    "input": "Input axis and sensitivity",
    "physics": "Physics simulation settings",
    "audio": "Audio volume settings",
    "toolbar": "Toolbar toggle states",
    "file_assoc": "File type associations in Windows",
    "mesh_preview": "3D model thumbnail preview appearance",
}

FIELD_TOOLTIPS = {
    "physics.multi_threaded": "Run physics simulation in a separate process for better performance",
    "editor.theme": "Editor color theme (restart required)",
    "editor.font_size": "Base font size in the editor (restart required)",
    "editor.ui_scale": "Global UI scale percentage (50-200, restart required)",
    "editor.language": "Editor UI language (restart required)",
    "editor.auto_save": "Automatically save the current scene",
    "editor.auto_save_interval": "Auto-save interval in seconds",
    "editor.thumb_cache_mode": "Thumbnail cache key: metadata (fast, uses mtime/size) or content (full file hash, slower but exact)",
    "editor.thumb_resolution": "Thumbnail render resolution in pixels (higher = sharper but slower)",

    "mesh_preview.camera_rot_x": "Camera X rotation angle in degrees for mesh thumbnails",
    "mesh_preview.camera_rot_y": "Camera Y rotation angle in degrees for mesh thumbnails",
    "mesh_preview.bg": "Background color [R, G, B, A] (0-1, A=0 is transparent)",
    "mesh_preview.tri": "Triangle fill color [R, G, B, A] (0-1)",
    "mesh_preview.wire": "Wireframe color [R, G, B, A] (0-1)",
    "mesh_preview.wire_width": "Wireframe line width in pixels",

    "camera.fov": "Camera field of view in degrees",
    "camera.near": "Near clipping plane distance",
    "camera.far": "Far clipping plane distance",
    "camera.move_speed": "Camera movement speed",
    "camera.fast_mult": "Speed multiplier when holding Shift",
    "camera.rotate_speed": "Camera orbit rotation speed",
    "camera.zoom_speed": "Camera zoom speed",
    "camera.pan_speed": "Camera pan speed",
    "camera.zoom_strength": "Zoom strength multiplier",
    "camera.damping": "Camera smooth movement damping",
    "camera.acceleration": "Camera movement acceleration",
    "camera.transition_speed": "Camera transition animation speed",
    "camera.zoom_smooth_speed": "Zoom smooth interpolation speed",
    "camera.use_ortho_in_2d": "Use orthographic projection in 2D mode",
    "camera.speed_boost_enabled": "Enable camera speed boost",
    "camera.speed_boost_mult": "Speed boost multiplier",
    "camera.speed_boost_ramp_time": "Time to reach full boost speed (seconds)",
    "rendering.vsync": "Enable vertical synchronization",
    "rendering.target_fps": "Target frame rate (0 = unlimited)",
    "rendering.shadow_resolution": "Shadow map resolution",
    "rendering.show_grid": "Show the reference grid in the viewport",
    "rendering.grid_size": "Reference grid cell size",
    "rendering.grid_world_size": "Reference grid total world size",
    "rendering.exposure": "HDR exposure in EV applied before tonemapping",
    "rendering.ambient": "Ambient light color [R, G, B] (0-1)",
    "rendering.selection_outline": "Selection outline color [R, G, B, A] (0-1)",
    "rendering.selection_outline_thickness": "Selection outline thickness",
    "rendering.tick_rate": "Game logic update rate (ticks per second)",
    "rendering.fixed_tick_rate": "Physics fixed update rate (ticks per second)",
    "rendering.max_lights": "Maximum number of dynamic lights",
    "gizmo.handle_size": "Gizmo handle size in world units",
    "gizmo.base_axis_length": "Length of gizmo axis arrows",
    "gizmo.plane_handle_size": "Size of plane translation handles",
    "gizmo.pick_threshold": "Gizmo click pick threshold in pixels",
    "gizmo.arrow_size_ratio": "Size ratio of arrowheads",
    "gizmo.center_handle_size": "Size of the center (scale all) handle",
    "gizmo.screen_axis_length": "Length of screen-space axis overlay",
    "gizmo.line_width": "Gizmo line width",
    "gizmo.show_delta_label": "Show delta value label during drag",
    "gizmo.smooth_snap": "Enable smooth snap interpolation",
    "gizmo.smooth_snap_speed": "Smooth snap interpolation speed",
    "gizmo.show_icons": "Show component icons in the viewport",
    "gizmo.icon_scale": "Scale of component icons",
    "gizmo.selection_bounds": "Show animated bounding box around selected entities",
    "gizmo.selection_bounds_speed": "Smooth animation speed for selection bounds",
    "gizmo.selection_bounds_color": "Selection bounds line color [R, G, B] (0-1)",
    "viewport.clear": "Viewport clear color [R, G, B] (0-1)",
    "viewport.no_scene": "No-scene background color [R, G, B] (0-1)",
    "viewport.update_interval": "Viewport update interval (ms)",
    "viewport.grid_step": "Grid step size for snapping",
    "console.font_size": "Console font size",
    "console.font_family": "Console font family name",
    "console.max_blocks": "Maximum console log blocks",
    "console.refresh_interval": "Console refresh interval (ms)",
    "terminal.font_size": "Terminal font size",
    "terminal.font_family": "Terminal font family name",
    "profiler.enabled": "Enable the performance profiler",
    "profiler.update_interval": "Profiler update interval (seconds)",
    "profiler.max_samples": "Maximum profiler samples to keep",
    "profiler.refresh_interval": "Profiler UI refresh interval (ms)",
    "hierarchy.refresh_interval": "Hierarchy panel refresh interval (ms)",
    "inspector.refresh_interval": "Inspector panel refresh interval (ms)",
    "project.thumb_size": "Project browser thumbnail size",
    "engine.time_scale": "Global time scale factor",
    "engine.fixed_update_dt": "Fixed update timestep (seconds)",
    "collab.cursor_rate": "Collaboration cursor sync rate (Hz)",
    "collab.camera_rate": "Collaboration camera sync rate (Hz)",
    "collab.transform_rate": "Collaboration transform sync rate (Hz)",
    "collab.gizmo_rate": "Collaboration gizmo sync rate (Hz)",
    "collab.ping_interval": "Collaboration ping interval (seconds)",
    "collab.poll_interval": "Collaboration poll interval (seconds)",
    "undo.max_stack": "Maximum undo history steps",
    "input.control_scheme": "Control scheme: fps = first person, tps = third person",
    "input.horizontal": "Horizontal movement axis (positive,negative key names, comma separated)",
    "input.vertical": "Vertical movement axis (positive,negative key names, comma separated)",
    "input.jump": "Jump button binding",
    "input.fire": "Fire / primary action button binding",
    "input.crouch": "Crouch button binding",
    "input.sprint": "Sprint button binding",
    "input.interact": "Interact button binding",
    "input.reload": "Reload button binding",
    "input.mouse_axis_x": "Name of the mouse X axis (defaults to Mouse X)",
    "input.mouse_axis_y": "Name of the mouse Y axis (defaults to Mouse Y)",
    "input.mouse_sensitivity": "Mouse look sensitivity multiplier",
    "input.invert_mouse_x": "Invert horizontal mouse look",
    "input.invert_mouse_y": "Invert vertical mouse look",
    "input.axis_gravity": "Axis return-to-zero speed when released",
    "input.axis_sensitivity": "Axis ramp-up speed while held",
    "input.axis_dead": "Axis dead zone threshold",
    "physics.gravity_x": "Global gravity X component",
    "physics.gravity_y": "Global gravity Y component",
    "physics.gravity_z": "Global gravity Z component",
    "physics.fixed_time_step": "Physics fixed timestep (seconds)",
    "physics.num_sub_steps": "Physics solver sub-steps per frame",
    "physics.solver_iterations": "Physics constraint solver iterations",
    "physics.erp": "Physics error reduction parameter",
    "physics.contact_erp": "Physics contact error reduction parameter",
    "physics.friction_erp": "Physics friction error reduction parameter",
    "physics.contact_breaking_threshold": "Contact breaking threshold",
    "physics.restitution": "Default restitution (bounciness)",
    "physics.linear_damping": "Default linear damping",
    "physics.angular_damping": "Default angular damping",
    "physics.max_contacts_per_body": "Maximum contacts per rigid body",
    "physics.simulation_mode": "How physics simulation is executed. single = main thread, multi_threaded = one separate process, per_layer_process = one process per collision layer (for extreme parallel workloads)",
    "physics.solver": "Physics solver backend (culverin, pybullet, or physx)",
    "audio.enable_audio": "Enable the audio system on startup",
    "audio.device_name": "Audio output device name (leave empty for Windows system default)",
    "audio.sample_rate": "Audio sample rate in Hz (44100 or 48000 recommended)",
    "audio.master_volume": "Master audio volume",
    "audio.sfx_volume": "Sound effects volume",
    "audio.music_volume": "Music volume",
    "audio.voice_volume": "Voice / dialogue volume",
    "audio.ambient_volume": "Ambient / environment volume",
    "audio.max_sources": "Maximum number of simultaneous audio sources",
    "audio.stream_buffer_size": "Streaming buffer size in bytes",
    "audio.distance_model": "3D audio distance attenuation model",
    "audio.doppler_factor": "Doppler effect intensity factor",
    "audio.speed_of_sound": "Speed of sound in world units per second",
    "audio.priority_threshold": "Volume threshold below which sounds can be culled",
    "audio.enable_spatialization": "Enable 3D spatial audio positioning",
    "audio.enable_reverb": "Enable reverb / environmental audio effects",
    "audio.enable_occlusion": "Enable audio occlusion simulation",
    "rendering.render_pipeline": "Active render pipeline",
    "rendering.anti_aliasing": "Anti-aliasing mode",
    "rendering.shadow_distance": "Maximum shadow rendering distance",
    "toolbar.grid": "Show grid in viewport",
    "toolbar.snap": "Enable snapping",
    "toolbar.snap_translate": "Translate snap increment",
    "toolbar.snap_rotate": "Rotate snap increment (degrees)",
    "toolbar.snap_scale": "Scale snap increment",
    "toolbar.skybox": "Show skybox in viewport",
    "toolbar.effects": "Enable post-processing effects",
}

_FIELD_RANGES = {
    "editor.font_size": (8, 72),
    "editor.ui_scale": (50, 200),
    "editor.auto_save_interval": (10, 600),
    "editor.thumb_resolution": (64, 2048),
    "camera.fov": (1, 179),
    "camera.near": (0.001, 10.0),
    "camera.far": (10.0, 50000.0),
    "camera.move_speed": (0.1, 100.0),
    "camera.fast_mult": (1.0, 20.0),
    "camera.rotate_speed": (0.01, 5.0),
    "camera.zoom_speed": (0.1, 50.0),
    "camera.pan_speed": (0.0001, 1.0),
    "camera.zoom_strength": (0.01, 5.0),
    "camera.damping": (0.1, 50.0),
    "camera.acceleration": (0.1, 50.0),
    "camera.transition_speed": (0.1, 20.0),
    "camera.zoom_smooth_speed": (0.1, 50.0),
    "camera.speed_boost_mult": (1.0, 20.0),
    "camera.speed_boost_ramp_time": (0.1, 10.0),
    "rendering.target_fps": (0, 360),
    "rendering.shadow_resolution": (256, 4096),
    "rendering.show_grid": None,
    "rendering.grid_size": (0.1, 100.0),
    "rendering.grid_world_size": (10.0, 10000.0),
    "rendering.ambient": None,
    "rendering.selection_outline": None,
    "rendering.selection_outline_thickness": (0.0, 0.5),
    "rendering.tick_rate": (30.0, 1000.0),
    "rendering.fixed_tick_rate": (10.0, 500.0),
    "rendering.max_lights": (1, 64),
    "gizmo.handle_size": (0.01, 2.0),
    "gizmo.base_axis_length": (0.1, 10.0),
    "gizmo.plane_handle_size": (0.02, 2.0),
    "gizmo.pick_threshold": (1.0, 100.0),
    "gizmo.arrow_size_ratio": (0.01, 1.0),
    "gizmo.center_handle_size": (0.01, 2.0),
    "gizmo.screen_axis_length": (10.0, 500.0),
    "gizmo.line_width": (0.5, 10.0),
    "gizmo.smooth_snap_speed": (0.01, 2.0),
    "gizmo.selection_bounds_speed": (0.1, 50.0),
    "gizmo.selection_bounds_color": None,
    "gizmo.icon_scale": (0.5, 20.0),
    "viewport.clear": None,
    "viewport.no_scene": None,
    "viewport.update_interval": (1, 500),
    "viewport.grid_step": (0.1, 100.0),
    "console.font_size": (6, 72),
    "console.max_blocks": (100, 10000),
    "console.refresh_interval": (16, 2000),
    "terminal.font_size": (6, 72),
    "profiler.update_interval": (0.05, 5.0),
    "profiler.max_samples": (50, 5000),
    "profiler.refresh_interval": (16, 2000),
    "hierarchy.refresh_interval": (50, 5000),
    "inspector.refresh_interval": (16, 2000),
    "project.thumb_size": (16, 256),
    "engine.time_scale": (0.0, 10.0),
    "engine.fixed_update_dt": (0.001, 1.0),
    "collab.cursor_rate": (1.0, 120.0),
    "collab.camera_rate": (1.0, 120.0),
    "collab.transform_rate": (1.0, 120.0),
    "collab.gizmo_rate": (1.0, 120.0),
    "collab.ping_interval": (0.5, 60.0),
    "collab.poll_interval": (1, 120),
    "undo.max_stack": (10, 2000),
    "input.mouse_sensitivity": (0.01, 10.0),
    "input.axis_gravity": (0.01, 50.0),
    "input.axis_sensitivity": (0.01, 20.0),
    "input.axis_dead": (0.0, 1.0),
    "physics.gravity_x": (-100.0, 100.0),
    "physics.gravity_y": (-100.0, 100.0),
    "physics.gravity_z": (-100.0, 100.0),
    "physics.fixed_time_step": (0.001, 1.0),
    "physics.num_sub_steps": (1, 16),
    "physics.solver_iterations": (1, 50),
    "physics.erp": (0.0, 1.0),
    "physics.contact_erp": (0.0, 1.0),
    "physics.friction_erp": (0.0, 1.0),
    "physics.contact_breaking_threshold": (0.0, 1.0),
    "physics.restitution": (0.0, 1.0),
    "physics.linear_damping": (0.0, 10.0),
    "physics.angular_damping": (0.0, 10.0),
    "physics.max_contacts_per_body": (1, 256),
    "audio.enable_audio": None,
    "audio.sample_rate": (8000, 192000),
    "audio.master_volume": (0.0, 1.0),
    "audio.sfx_volume": (0.0, 1.0),
    "audio.music_volume": (0.0, 1.0),
    "audio.voice_volume": (0.0, 1.0),
    "audio.ambient_volume": (0.0, 1.0),
    "audio.max_sources": (8, 256),
    "audio.stream_buffer_size": (1024, 65536),
    "audio.distance_model": None,
    "audio.doppler_factor": (0.0, 10.0),
    "audio.speed_of_sound": (0.1, 10000.0),
    "audio.priority_threshold": (0.0, 1.0),
    "audio.enable_spatialization": None,
    "audio.enable_reverb": None,
    "audio.enable_occlusion": None,
    "rendering.shadow_distance": (1.0, 500.0),
    "toolbar.snap_translate": (0.001, 100.0),
    "toolbar.snap_rotate": (0.1, 360.0),
    "toolbar.snap_scale": (0.001, 100.0),
    "mesh_preview.camera_rot_x": (-180.0, 180.0),
    "mesh_preview.camera_rot_y": (-180.0, 180.0),
    "mesh_preview.bg": None,
    "mesh_preview.tri": None,
    "mesh_preview.wire": None,
    "mesh_preview.wire_width": (0.5, 5.0),
}


class SliderSpinBox(QWidget):
    valueChanged = pyqtSignal(object)

    def __init__(self, key: str, value, is_float: bool, slider_range, parent=None):
        super().__init__(parent)
        self._key = key
        self._is_float = is_float
        self._updating = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimumWidth(120)
        if slider_range:
            lo, hi = slider_range
            if is_float:
                self._slider.setRange(int(lo * 1000), int(hi * 1000))
            else:
                self._slider.setRange(lo, hi)
        else:
            if is_float:
                self._slider.setRange(-999999000, 999999000)
            else:
                self._slider.setRange(-999999, 999999)
        if is_float:
            self._spin = QDoubleSpinBox()
            self._spin.setDecimals(4)
            self._spin.setRange(-999999.0, 999999.0)
            self._spin.setValue(float(value))
        else:
            self._spin = QSpinBox()
            self._spin.setRange(-999999, 999999)
            self._spin.setValue(int(value))
        self._spin.setFixedWidth(scale(100))
        if slider_range:
            lo, hi = slider_range
            if is_float:
                self._slider.setValue(int(float(value) * 1000))
            else:
                self._slider.setValue(int(value))
        else:
            if is_float:
                self._slider.setValue(int(float(value) * 1000))
            else:
                self._slider.setValue(int(value))
        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)
        layout.addWidget(self._slider)
        layout.addWidget(self._spin)

    def _on_slider(self, val: int):
        if self._updating:
            return
        self._updating = True
        if self._is_float:
            v = val / 1000.0
            self._spin.setValue(v)
        else:
            self._spin.setValue(val)
        self._updating = False
        self.valueChanged.emit(self._spin.value())

    def _on_spin(self, val):
        if self._updating:
            return
        self._updating = True
        if self._is_float:
            self._slider.setValue(int(val * 1000))
        else:
            self._slider.setValue(int(val))
        self._updating = False
        self.valueChanged.emit(val)

    def setValue(self, value):
        if self._updating:
            return
        self._updating = True
        if self._is_float:
            v = float(value)
            self._spin.setValue(v)
            self._slider.setValue(int(v * 1000))
        else:
            v = int(value)
            self._spin.setValue(v)
            self._slider.setValue(v)
        self._updating = False
        self.valueChanged.emit(self._spin.value())


class SettingsDialog(QDialog):
    config_changed = pyqtSignal(str, object)

    def __init__(self, title: str, config: Config, parent=None):
        super().__init__(parent)
        self._config = config
        self.setWindowTitle(title)
        self.setMinimumSize(800, 520)
        self.resize(900, 580)
        self._field_resets: dict[str, tuple[QPushButton, Any]] = {}
        self._field_widgets: dict[str, QWidget] = {}
        self._config.on_changed(self._on_config_changed)
        self._setup_ui()
        if self._list_widget.count() > 0:
            self._list_widget.setCurrentRow(0)

    def _on_config_changed(self, key: str, value: Any):
        self.config_changed.emit(key, value)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        title_label = QLabel(self.windowTitle())
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search settings...")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.setFixedWidth(200)
        self._search_edit.textChanged.connect(self._on_search)
        header_layout.addWidget(self._search_edit)
        main_layout.addWidget(header)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self._list_widget = QListWidget()
        self._list_widget.setFixedWidth(scale(180))
        self._list_widget.setFrameShape(QFrame.Shape.StyledPanel)
        self._list_widget.currentRowChanged.connect(self._on_list_row_changed)
        self._list_widget.installEventFilter(self)

        sections = [(k, v) for k, v in self._config.to_dict().items() if isinstance(v, dict)]
        sections.sort(key=lambda x: x[0])

        self._sections = [s[0] for s in sections]
        self._pages: dict[str, QWidget] = {}

        self._stack = QStackedWidget()

        for section, values in sections:
            icon_name = SECTION_ICONS.get(section)
            item = QListWidgetItem()
            if icon_name is not None and qta is not None:
                item.setIcon(qta.icon(icon_name, color="#d4d4d4"))
            item.setText(section.replace("_", " ").title())
            self._list_widget.addItem(item)
            scroll = self._build_page(section, values)
            self._pages[section] = scroll
            self._stack.addWidget(scroll)

        content_layout.addWidget(self._list_widget)
        content_layout.addWidget(self._stack, 1)
        main_layout.addWidget(content, 1)

        bottom_bar = QWidget()
        bar_layout = QHBoxLayout(bottom_bar)
        bar_layout.setContentsMargins(8, 4, 8, 4)
        restore_btn = QPushButton("Restore Defaults")
        restore_btn.clicked.connect(self._on_restore)
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(scale(80))
        close_btn.clicked.connect(self.close)
        bar_layout.addWidget(restore_btn)
        bar_layout.addStretch()
        bar_layout.addWidget(close_btn)
        main_layout.addWidget(bottom_bar)

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if obj is self._list_widget and event.type() == QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            row = self._list_widget.currentRow()
            if delta > 0 and row > 0:
                self._list_widget.setCurrentRow(row - 1)
            elif delta < 0 and row < self._list_widget.count() - 1:
                self._list_widget.setCurrentRow(row + 1)
            return True
        return super().eventFilter(obj, event)

    def _on_list_row_changed(self, row: int):
        if 0 <= row < len(self._sections):
            self._stack.setCurrentIndex(row)

    def _on_search(self, text: str):
        text = text.lower().strip()
        for i in range(self._stack.count()):
            self._stack.widget(i).setVisible(True)

        for i, section in enumerate(self._sections):
            item = self._list_widget.item(i)
            if not text:
                item.setHidden(False)
                self._pages[section].setVisible(True)
            else:
                matches = text in section.lower() or text in SECTION_DESCRIPTIONS.get(section, "").lower()
                matches = matches or any(
                    text in k.lower() or text in FIELD_TOOLTIPS.get(f"{section}.{k}", "").lower()
                    for k in self._config.to_dict().get(section, {}) if not isinstance(self._config.to_dict().get(section, {})[k], dict)
                )
                item.setHidden(not matches)
                self._pages[section].setVisible(matches)

    def _build_page(self, prefix: str, data: dict) -> QWidget:
        if prefix == "file_assoc":
            return self._build_file_assoc_page(prefix)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        outer_layout = QVBoxLayout(container)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        header_text = prefix.replace("_", " ").title()
        desc_text = SECTION_DESCRIPTIONS.get(prefix, "")
        if desc_text:
            header_text += f"  \u2014  {desc_text}"
        hlabel = QLabel(header_text)
        hlabel.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px 16px;")
        outer_layout.addWidget(hlabel)

        form_container = QWidget()
        form = QFormLayout(form_container)
        form.setContentsMargins(16, 8, 16, 8)
        form.setSpacing(6)

        keys = sorted(data.keys())
        for key in keys:
            value = data[key]
            full_key = f"{prefix}.{key}"
            if isinstance(value, dict):
                continue
            widget = self._create_widget(full_key, value)
            if widget:
                label_text = key.replace("_", " ").title()
                restart = self._config.is_restart_key(full_key)
                if restart:
                    label_text += " [restart]"
                label = QLabel(label_text)
                tooltip = FIELD_TOOLTIPS.get(full_key, "")
                if restart:
                    label.setStyleSheet("color: orange;")
                    label.setToolTip("Restart required" + ("\n" + tooltip if tooltip else ""))
                if tooltip:
                    label.setToolTip(tooltip)
                    widget.setToolTip(tooltip)
                default_val = self._config._defaults.get(prefix, {}).get(key)
                self._field_widgets[full_key] = widget
                if default_val is not None and not isinstance(default_val, dict):
                    row_widget = QWidget()
                    row_layout = QHBoxLayout(row_widget)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.setSpacing(4)
                    row_layout.addWidget(widget, 1)
                    reset_btn = QPushButton()
                    reset_btn.setIcon(qta.icon("fa5s.undo", color="#888") if qta else QIcon())
                    reset_btn.setFixedSize(scale(20), scale(20))
                    reset_btn.setToolTip("Reset to default")
                    reset_btn.setStyleSheet("QPushButton { border: none; } QPushButton:hover { background: rgba(255,255,255,0.1); border-radius: 3px; }")
                    reset_btn.clicked.connect(lambda _, k=full_key: self._reset_field(k))
                    row_layout.addWidget(reset_btn)
                    self._field_resets[full_key] = (reset_btn, default_val)
                    self._update_reset_btn(full_key)
                    form.addRow(label, row_widget)
                else:
                    form.addRow(label, widget)

        outer_layout.addWidget(form_container)

        if prefix == "physics":
            self._add_collision_layers_editor(outer_layout)

        outer_layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_file_assoc_page(self, prefix: str) -> QWidget:
        from editor.file_associations import register, unregister, status
        from editor.constants import EXTENSIONS
        import os

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        outer_layout = QVBoxLayout(container)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        header_text = "File Associations"
        desc_text = SECTION_DESCRIPTIONS.get(prefix, "")
        if desc_text:
            header_text += f"  \u2014  {desc_text}"
        hlabel = QLabel(header_text)
        hlabel.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px 16px;")
        outer_layout.addWidget(hlabel)

        info_label = QLabel(
            "Register Zarin Engine file extensions with Windows Explorer.\n"
            "This allows opening .zpes and .zpep files by double-clicking."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("padding: 4px 16px 8px 16px; color: #aaa;")
        outer_layout.addWidget(info_label)

        status_group = QWidget()
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(16, 8, 16, 8)
        status_layout.setSpacing(4)

        self._assoc_status_labels = {}
        for ext, info in EXTENSIONS.items():
            row = QHBoxLayout()
            lbl = QLabel(f"{ext}  \u2014  {info['description']}")
            lbl.setStyleSheet("font-size: 13px;")
            status_lbl = QLabel("unknown")
            status_lbl.setStyleSheet("color: gray; font-weight: bold;")
            self._assoc_status_labels[ext] = status_lbl
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(status_lbl)
            status_layout.addLayout(row)

        outer_layout.addWidget(status_group)

        btn_group = QWidget()
        btn_layout = QHBoxLayout(btn_group)
        btn_layout.setContentsMargins(16, 8, 16, 8)
        btn_layout.setSpacing(12)

        register_btn = QPushButton("Register File Associations")
        register_btn.setFixedHeight(scale(32))
        register_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 4px 16px; }")
        register_btn.clicked.connect(self._on_register_assoc)
        btn_layout.addWidget(register_btn)

        unregister_btn = QPushButton("Unregister File Associations")
        unregister_btn.setFixedHeight(scale(32))
        unregister_btn.setStyleSheet("QPushButton { padding: 4px 16px; }")
        unregister_btn.clicked.connect(self._on_unregister_assoc)
        btn_layout.addWidget(unregister_btn)

        btn_layout.addStretch()
        outer_layout.addWidget(btn_group)

        self._assoc_refresh_btn = QPushButton("Refresh Status")
        self._assoc_refresh_btn.setFixedWidth(scale(120))
        self._assoc_refresh_btn.clicked.connect(self._refresh_assoc_status)
        br_layout = QHBoxLayout()
        br_layout.setContentsMargins(16, 4, 16, 12)
        br_layout.addWidget(self._assoc_refresh_btn)
        br_layout.addStretch()
        outer_layout.addLayout(br_layout)

        outer_layout.addStretch()
        scroll.setWidget(container)
        self._refresh_assoc_status()
        return scroll

    def _refresh_assoc_status(self):
        from editor.file_associations import status
        try:
            st = status()
        except Exception:
            st = {}
        for ext, lbl in self._assoc_status_labels.items():
            registered = st.get(ext, False)
            if registered:
                lbl.setText("Registered")
                lbl.setStyleSheet("color: #4CAF50; font-weight: bold;")
            else:
                lbl.setText("Not registered")
                lbl.setStyleSheet("color: #f44336; font-weight: bold;")

    def _on_register_assoc(self):
        from editor.file_associations import register
        import os
        svg_path = os.path.join(os.path.dirname(__file__), "..", "zarin_icon.svg")
        svg_path = os.path.abspath(svg_path)
        if not os.path.exists(svg_path):
            svg_path = os.path.join(os.path.dirname(__file__), "..", "assets", "zarin_icon.svg")
            svg_path = os.path.abspath(svg_path)
        if not os.path.exists(svg_path):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "zarin_icon.svg not found")
            return
        result = register(svg_path)
        self._refresh_assoc_status()
        from PyQt6.QtWidgets import QMessageBox
        if result:
            QMessageBox.information(self, "Success",
                f"Registered: {', '.join(result)}")
        else:
            QMessageBox.warning(self, "Error", "Failed to register file associations")

    def _on_unregister_assoc(self):
        from editor.file_associations import unregister
        result = unregister()
        self._refresh_assoc_status()
        from PyQt6.QtWidgets import QMessageBox
        if result:
            QMessageBox.information(self, "Success",
                f"Unregistered: {', '.join(result)}")

    def _create_widget(self, key: str, value) -> Optional[QWidget]:
        if key == "input.control_scheme":
            cb = QComboBox()
            cb.addItems(["fps", "tps"])
            cb.setCurrentText(self._config.get(key, "fps"))
            cb.currentTextChanged.connect(lambda t, k=key: self._on_value_changed(k, t))
            return cb
        if key == "physics.simulation_mode":
            container = QWidget()
            vl = QVBoxLayout(container)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(2)
            cb = QComboBox()
            cb.addItems(["single", "multi_threaded", "per_layer_process"])
            cb.setCurrentText(self._config.get(key, value))
            cb.currentTextChanged.connect(lambda t, k=key: self._on_value_changed(k, t))
            vl.addWidget(cb)
            info = QLabel(
                "single = всё в основном потоке, отладка\n"
                "multi_threaded = один отдельный процесс, стандарт\n"
                "per_layer_process = свой процесс на каждый слой коллизии, для безумных симуляций"
            )
            info.setStyleSheet("color: #888; font-size: 11px; padding-left: 4px;")
            info.setWordWrap(True)
            vl.addWidget(info)
            return container
        if key == "audio.distance_model":
            cb = QComboBox()
            cb.addItems([
                "none", "inverse_distance", "inverse_distance_clamped",
                "linear_distance", "linear_distance_clamped",
                "exponent_distance", "exponent_distance_clamped",
            ])
            cb.setCurrentText(self._config.get(key, "inverse_distance_clamped"))
            cb.currentTextChanged.connect(lambda t, k=key: self._on_value_changed(k, t))
            return cb
        if key == "audio.device_name":
            container = QWidget()
            hl = QHBoxLayout(container)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(4)
            le = QLineEdit(self._config.get(key, ""))
            le.setPlaceholderText("System default")
            le.setFixedWidth(220)
            le.textChanged.connect(lambda t, k=key: self._on_value_changed(k, t))
            hl.addWidget(le)
            from core.audio.audio_system import AudioSystem
            devices = AudioSystem.get_available_devices()
            if devices:
                detect_btn = QPushButton("Detect")
                detect_btn.setFixedWidth(scale(56))
                def _detect(_checked=False, _btn=detect_btn, edit=le, k=key):
                    from core.audio.audio_system import AudioSystem
                    devs = AudioSystem.get_available_devices()
                    if devs:
                        current = edit.text()
                        if current and current in devs:
                            idx = devs.index(current)
                        else:
                            idx = 0
                        from PyQt6.QtWidgets import QInputDialog
                        item, ok = QInputDialog.getItem(_btn, "Audio Devices", "Select device:", devs, idx, False)
                        if ok:
                            edit.setText(item)
                            self._on_value_changed(k, item)
                detect_btn.clicked.connect(_detect)
                hl.addWidget(detect_btn)
            return container
        if key == "editor.thumb_cache_mode":
            cb = QComboBox()
            cb.addItems(["metadata", "content"])
            cb.setCurrentText(self._config.get(key, "metadata"))
            cb.currentTextChanged.connect(lambda t, k=key: self._on_value_changed(k, t))
            return cb
        if isinstance(value, list):
            from editor.color_picker import ColorLineEdit
            current = self._config.get(key, value)
            cl = ColorLineEdit(current)
            cl.colorChanged.connect(lambda _, k=key, w=cl: self._on_value_changed(k, w.get_color_rgba()))
            return cl
        if isinstance(value, bool):
            cb = QCheckBox()
            cb.setChecked(self._config.get(key, value))
            cb.toggled.connect(lambda checked, k=key: self._on_value_changed(k, checked))
            return cb
        elif isinstance(value, (int, float)):
            val = self._config.get(key, value)
            slider_range = _FIELD_RANGES.get(key)
            is_float = isinstance(value, float)
            w = SliderSpinBox(key, val, is_float, slider_range)
            w.valueChanged.connect(lambda v, k=key: self._on_value_changed(k, v))
            return w
        elif isinstance(value, str):
            le = QLineEdit(self._config.get(key, value))
            le.setFixedWidth(220)
            le.textChanged.connect(lambda t, k=key: self._on_value_changed(k, t))
            return le
        return None

    def _add_collision_layers_editor(self, parent_layout):
        layer_names = self._config.get("physics.layer_names", list(DEFAULT_LAYER_NAMES))
        collision_matrix = self._config.get("physics.collision_matrix", [0xFFFF] * MAX_LAYERS)

        gb_layers = QGroupBox("Collision Layers")
        gl = QGridLayout(gb_layers)
        gl.setSpacing(4)
        self._layer_edits = []
        for i in range(MAX_LAYERS):
            name = layer_names[i] if i < len(layer_names) else DEFAULT_LAYER_NAMES[i]
            lbl = QLabel(f"{i}:")
            lbl.setFixedWidth(scale(24))
            le = QLineEdit(name)
            le.setFixedWidth(scale(160))
            le.textChanged.connect(lambda _, idx=i: self._on_layer_name_changed(idx))
            self._layer_edits.append(le)
            gl.addWidget(lbl, i, 0)
            gl.addWidget(le, i, 1)
        parent_layout.addWidget(gb_layers)

        gb_matrix = QGroupBox("Collision Matrix")
        gb_layout = QVBoxLayout(gb_matrix)
        open_btn = QPushButton("Edit Collision Matrix...")
        open_btn.clicked.connect(lambda: self._open_collision_matrix_dialog())
        gb_layout.addWidget(open_btn)
        parent_layout.addWidget(gb_matrix)

    def _on_layer_name_changed(self, idx):
        names = [le.text() for le in self._layer_edits]
        self._config.set("physics.layer_names", names, notify=True)
        self._config.save()

    def _open_collision_matrix_dialog(self):
        layer_names = self._config.get("physics.layer_names", list(DEFAULT_LAYER_NAMES))
        collision_matrix = list(self._config.get("physics.collision_matrix", [0xFFFF] * MAX_LAYERS))
        dialog = CollisionMatrixDialog(layer_names, collision_matrix, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._config.set("physics.collision_matrix", dialog.get_matrix(), notify=True)
            self._config.save()

    def _on_value_changed(self, key: str, value):
        self._config.set(key, value, notify=True)
        self._config.save()
        self._update_reset_btn(key)

    def _update_reset_btn(self, key: str):
        if key not in self._field_resets:
            return
        btn, default_val = self._field_resets[key]
        current = self._config.get(key)
        btn.setVisible(current != default_val)

    def _reset_field(self, key: str):
        if key not in self._field_resets:
            return
        _, default_val = self._field_resets[key]
        self._config.reset(key)
        self._config.save()
        self._config._notify(key, default_val)
        self._update_reset_btn(key)
        widget = self._field_widgets.get(key)
        if widget is None:
            return
        if hasattr(widget, 'setValue'):
            widget.setValue(default_val if isinstance(default_val, (int, float)) else 0)
        elif hasattr(widget, 'setCurrentText'):
            widget.setCurrentText(str(default_val))
        elif hasattr(widget, 'setChecked'):
            widget.setChecked(bool(default_val))
        elif hasattr(widget, 'setText'):
            widget.setText(str(default_val))
        elif hasattr(widget, 'set_color'):
            widget.set_color(default_val)

    def _on_restore(self):
        row = self._list_widget.currentRow()
        if row < 0 or row >= len(self._sections):
            return
        section = self._sections[row]
        prefix = section + "."
        for k in [k for k in self._field_resets if k.startswith(prefix)]:
            self._field_resets.pop(k, None)
            self._field_widgets.pop(k, None)
        self._config.reset(section)
        self._config.save()
        old = self._pages[section]
        self._stack.removeWidget(old)
        old.deleteLater()
        values = self._config.to_dict().get(section, {})
        scroll = self._build_page(section, values)
        self._pages[section] = scroll
        self._stack.insertWidget(row, scroll)
        self._stack.setCurrentIndex(row)


class CollisionMatrixDialog(QDialog):
    def __init__(self, layer_names, matrix, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Collision Matrix")
        self.setMinimumSize(600, 500)
        self._matrix = list(matrix)
        self._layer_names = list(layer_names)
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(2)
        info = QLabel("Check which layers collide with each other:")
        info.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(info)
        self._checks = []
        for i in range(MAX_LAYERS):
            lbl = QLabel(f"{i}:{layer_names[i] if i < len(layer_names) else ''}")
            lbl.setFixedWidth(scale(140))
            grid.addWidget(lbl, i + 1, 0)
            lbl_top = QLabel(str(i))
            lbl_top.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_top.setFixedWidth(scale(24))
            grid.addWidget(lbl_top, 0, i + 1)
            row_checks = []
            for j in range(MAX_LAYERS):
                chk = QCheckBox()
                chk.setChecked(bool(self._matrix[i] & (1 << j)))
                chk.stateChanged.connect(lambda _, ri=i, cj=j: self._on_toggle(ri, cj))
                chk.setFixedWidth(scale(24))
                grid.addWidget(chk, i + 1, j + 1, Qt.AlignmentFlag.AlignCenter)
                row_checks.append(chk)
            self._checks.append(row_checks)
        scroll.setWidget(container)
        layout.addWidget(scroll)
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _on_toggle(self, i, j):
        chk = self._checks[i][j]
        if chk.isChecked():
            self._matrix[i] |= 1 << j
        else:
            self._matrix[i] &= ~(1 << j)

    def get_matrix(self):
        return self._matrix
