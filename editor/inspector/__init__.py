# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from editor.inspector.source_viewer import SourceViewerDialog
from editor.inspector.widgets import _FocusSpinBox, _DragLabel, _DropLabelMixin, _ResourceDropLabel, _EntityDropLabel, EntityPickerDialog
from editor.inspector.helpers import (
    make_spinbox, make_clickable_label, get_component_icon_pixmap,
    update_resource_icon, make_resource_picker, make_gameobject_picker,
    make_resource_type_picker, make_asset_picker, _create_asset_dialog,
    make_vec2_row, make_vec3_row, make_vec4_row,
    make_vec2_slider_row, make_vec3_slider_row,
    get_component_source_path, get_property_line_number, collapse_value,
)
from editor.inspector.component_widget import ComponentWidget
from editor.inspector.component_picker import _CategoryIconWidget, ComponentPickerDialog
from editor.inspector.panel import InspectorPanel
from editor.inspector.constants import _COMPONENT_MIME

__all__ = [
    "SourceViewerDialog",
    "_FocusSpinBox", "_DragLabel", "_DropLabelMixin", "_ResourceDropLabel",
    "_EntityDropLabel", "EntityPickerDialog",
    "make_spinbox", "make_clickable_label", "get_component_icon_pixmap",
    "update_resource_icon", "make_resource_picker", "make_gameobject_picker",
    "make_resource_type_picker", "make_asset_picker", "_create_asset_dialog",
    "make_vec2_row", "make_vec3_row", "make_vec4_row",
    "make_vec2_slider_row", "make_vec3_slider_row",
    "get_component_source_path", "get_property_line_number", "collapse_value",
    "ComponentWidget",
    "_CategoryIconWidget", "ComponentPickerDialog",
    "InspectorPanel",
    "_COMPONENT_MIME",
]
