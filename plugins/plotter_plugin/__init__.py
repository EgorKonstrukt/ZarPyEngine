# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from core.foundation.logger import Logger
from core.foundation.plugin_manager import PluginBase


class PlotterPlugin(PluginBase):
    NAME = "PlotterPlugin"
    VERSION = "1.2.0"
    DESCRIPTION = "Real-time property plotting dock and Chart widgets"
    SYSTEM = False

    def __init__(self):
        super().__init__()
        self._panel = None
        self._extras: list = []

    def initialize(self, engine):
        super().initialize(engine)
        self.register_dock("Plotter", self._create_panel, area="left", icon="fa5s.chart-line")
        self.add_menu_item("Tools", "Open Plotter", self._open_panel, icon="fa5s.chart-line")
        self.add_menu_item("Tools", "New Plotter Window", self._new_plotter, icon="fa5s.plus")
        self.add_toolbar_button("New Plotter", self._new_plotter,
                                icon="fa5s.chart-line", tooltip="Create a new plotter window")

    def _create_panel(self):
        if self._panel is not None:
            return self._panel
        try:
            from .panel import PlotterPanel

            self._panel = PlotterPanel(self._engine, plugin=self)
        except Exception as e:
            Logger.error(f"Plotter panel init failed: {e}", e)
            self._panel = None
        return self._panel

    def _open_panel(self):
        w = getattr(self, "_panel", None)
        if w is None:
            return
        dock = w.parent()
        if dock is not None:
            dock.show()
            dock.raise_()
            dock.setFocus()

    def _new_plotter(self):
        self._cleanup_closed_extras()
        idx = len(self._extras) + 1

        def _factory():
            from .panel import PlotterPanel

            panel = PlotterPanel(self._engine, plugin=self, persist=False)
            self._extras.append(panel)
            return panel

        self.register_dock(f"Plotter {idx}", _factory, area="left", icon="fa5s.chart-line")

    def _cleanup_closed_extras(self):
        keep = []
        for panel in self._extras:
            try:
                panel.parent()
                keep.append(panel)
            except RuntimeError:
                continue
        self._extras = keep

    def shutdown(self):
        for panel in self._extras:
            try:
                panel.set_active(False)
            except Exception as e:
                Logger.warning(f"[{self.NAME}] Could not stop extra plotter: {e}")
        self._extras = []
        if self._panel is not None:
            try:
                self._panel.set_active(False)
            except Exception as e:
                Logger.warning(f"[{self.NAME}] Could not stop plotter: {e}")
        super().shutdown()


def get_plugin():
    return PlotterPlugin()