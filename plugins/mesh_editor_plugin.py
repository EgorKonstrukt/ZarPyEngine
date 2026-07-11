# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.foundation.plugin_manager import PluginBase


class MeshEditorPlugin(PluginBase):
    NAME = "MeshEditorPlugin"
    VERSION = "1.0.0"
    DESCRIPTION = "ProBuilder-style mesh editor integration"
    SYSTEM = False

    def initialize(self, engine):
        super().initialize(engine)
        self.add_menu_item("Mesh Editor", "Open Mesh Editor", self._open_mesh_editor, "Ctrl+Shift+M")

    def _open_mesh_editor(self):
        eng = self._engine
        vp = getattr(eng, 'viewport', None)
        mw = vp.parent() if vp else None
        if mw and hasattr(mw, '_mesh_editor'):
            mw._mesh_editor.show()
            mw._mesh_editor.raise_()
            sel = getattr(vp, '_selected_entities', None)
            if sel and len(sel) > 0:
                mw._mesh_editor.set_entity(sel[0])


def get_plugin():
    return MeshEditorPlugin()
