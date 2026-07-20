# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import json


def save_graph(graph, path: str) -> bool:
    try:
        data = graph.serialize_session()
        path = path.strip()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        graph._model.session = path
        return True
    except Exception as e:
        from core.foundation.logger import Logger
        Logger.error(f"TerrainGraph: failed to save graph to {path}: {e}", e)
        return False


def load_graph(graph, path: str) -> bool:
    try:
        path = path.strip()
        if not os.path.isfile(path):
            return False
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        graph.deserialize_session(data, clear_session=True, clear_undo_stack=True)
        graph._model.session = path
        _sync_widgets(graph)
        return True
    except Exception as e:
        from core.foundation.logger import Logger
        Logger.error(f"TerrainGraph: failed to load graph from {path}: {e}", e)
        return False


def _sync_widgets(graph):
    for node in graph.all_nodes():
        for name in node.model.properties.keys():
            val = node.get_property(name)
            widgets = getattr(node.view, "widgets", None)
            if widgets and name in widgets:
                try:
                    widgets[name].set_value(val)
                except Exception:
                    pass
