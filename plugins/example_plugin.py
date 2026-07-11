# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from core.foundation.plugin_manager import PluginBase
from core.foundation.logger import Logger

class ExamplePlugin(PluginBase):
    NAME = "ExamplePlugin"
    VERSION = "1.0.0"
    DESCRIPTION = "Example user plugin demonstrating the plugin API."
    SYSTEM = False

    def __init__(self):
        super().__init__()
        self._frame = 0

    def initialize(self, engine):
        super().initialize(engine)
        self.add_menu_item("ExamplePlugin", "Say Hello", lambda: Logger.info("Hello from ExamplePlugin!"), "Ctrl+Shift+H")
        self.add_toolbar_button("Hello", lambda: Logger.info("Hello from toolbar!"), tooltip="Say hello")
        Logger.info(f"[{self.NAME}] Initialized! Counter: {self.get_config('counter', 0)}")

    def on_scene_loaded(self, scene):
        Logger.debug(f"[{self.NAME}] Scene loaded: {scene.name}")

    def on_play_start(self):
        Logger.debug(f"[{self.NAME}] Play started!")
        self._frame = 0

    def on_play_stop(self):
        Logger.debug(f"[{self.NAME}] Play stopped after {self._frame} frames.")
        counter = self.get_config("counter", 0) + 1
        self.set_config("counter", counter)
        Logger.info(f"[{self.NAME}] Run counter saved: {counter}")

    def shutdown(self):
        Logger.info(f"[{self.NAME}] Shutdown.")

def get_plugin():
    return ExamplePlugin()
