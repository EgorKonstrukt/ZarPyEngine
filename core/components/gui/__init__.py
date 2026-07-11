# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.components.gui.layout.base_widget_component import GuiWidgetComponentBase
    from core.components.gui.containers.panel_component import PanelComponent
    from core.components.gui.controls.label_component import LabelComponent
    from core.components.gui.controls.button_component import ButtonComponent
    from core.components.gui.controls.slider_component import SliderComponent
    from core.components.gui.controls.textinput_component import TextInputComponent
    from core.components.gui.controls.toggle_component import ToggleComponent
    from core.components.gui.controls.progressbar_component import ProgressBarComponent
    from core.components.gui.controls.dropdown_component import DropdownComponent
    from core.components.gui.containers.scrollpanel_component import ScrollPanelComponent
    from core.components.gui.controls.image_component import ImageComponent
    from core.components.gui.controls.radiobutton_component import RadioButtonComponent
    from core.components.gui.containers.listwidget_component import ListWidgetComponent
    from core.components.gui.containers.tablewidget_component import TableWidgetComponent
    from core.components.gui.containers.treewidget_component import TreeWidgetComponent
    from core.components.gui.containers.tabwidget_component import TabWidgetComponent
    from core.components.gui.containers.groupbox_component import GroupBoxComponent
    from core.components.gui.controls.spinbox_component import SpinBoxComponent
    from core.components.gui.controls.doublespinbox_component import DoubleSpinBoxComponent
    from core.components.gui.controls.textedit_component import TextEditComponent
    from core.components.gui.controls.dial_component import DialComponent
    from core.components.gui.controls.html_component import HtmlComponent
    from core.components.gui.containers.splitter_component import SplitterComponent
    from core.components.gui.containers.stacked_component import StackedComponent
    from core.components.gui.containers.toolbox_component import ToolBoxComponent
    from core.components.gui.controls.calendar_component import CalendarComponent
    from core.components.gui.controls.lcd_component import LCDComponent
    from core.components.gui.controls.plaintext_component import PlainTextComponent
    from core.components.gui.containers.scrollbar_component import ScrollBarComponent
    from core.components.gui.controls.toolbutton_component import ToolButtonComponent
    from core.components.gui.controls.fontcombo_component import FontComboComponent
    from core.components.gui.containers.mdiarea_component import MdiAreaComponent
    from core.components.gui.controls.tooltip_component import TooltipComponent
    from core.components.gui.layout.layout_element_component import LayoutElementComponent
    from core.components.gui.layout.horizontal_layout_component import (
        HorizontalLayoutComponent, VerticalLayoutComponent, GridLayoutComponent,
    )

_LAZY_IMPORTS: dict[str, str] = {
    "GuiWidgetComponentBase": "core.components.gui.layout.base_widget_component",
    "PanelComponent": "core.components.gui.containers.panel_component",
    "LabelComponent": "core.components.gui.controls.label_component",
    "ButtonComponent": "core.components.gui.controls.button_component",
    "SliderComponent": "core.components.gui.controls.slider_component",
    "TextInputComponent": "core.components.gui.controls.textinput_component",
    "ToggleComponent": "core.components.gui.controls.toggle_component",
    "ProgressBarComponent": "core.components.gui.controls.progressbar_component",
    "DropdownComponent": "core.components.gui.controls.dropdown_component",
    "ScrollPanelComponent": "core.components.gui.containers.scrollpanel_component",
    "ImageComponent": "core.components.gui.controls.image_component",
    "RadioButtonComponent": "core.components.gui.controls.radiobutton_component",
    "ListWidgetComponent": "core.components.gui.containers.listwidget_component",
    "TableWidgetComponent": "core.components.gui.containers.tablewidget_component",
    "TreeWidgetComponent": "core.components.gui.containers.treewidget_component",
    "TabWidgetComponent": "core.components.gui.containers.tabwidget_component",
    "GroupBoxComponent": "core.components.gui.containers.groupbox_component",
    "SpinBoxComponent": "core.components.gui.controls.spinbox_component",
    "DoubleSpinBoxComponent": "core.components.gui.controls.doublespinbox_component",
    "TextEditComponent": "core.components.gui.controls.textedit_component",
    "DialComponent": "core.components.gui.controls.dial_component",
    "HtmlComponent": "core.components.gui.controls.html_component",
    "SplitterComponent": "core.components.gui.containers.splitter_component",
    "StackedComponent": "core.components.gui.containers.stacked_component",
    "ToolBoxComponent": "core.components.gui.containers.toolbox_component",
    "CalendarComponent": "core.components.gui.controls.calendar_component",
    "LCDComponent": "core.components.gui.controls.lcd_component",
    "PlainTextComponent": "core.components.gui.controls.plaintext_component",
    "ScrollBarComponent": "core.components.gui.containers.scrollbar_component",
    "ToolButtonComponent": "core.components.gui.controls.toolbutton_component",
    "FontComboComponent": "core.components.gui.controls.fontcombo_component",
    "MdiAreaComponent": "core.components.gui.containers.mdiarea_component",
    "TooltipComponent": "core.components.gui.controls.tooltip_component",
    "LayoutElementComponent": "core.components.gui.layout.layout_element_component",
    "HorizontalLayoutComponent": "core.components.gui.layout.horizontal_layout_component",
    "VerticalLayoutComponent": "core.components.gui.layout.horizontal_layout_component",
    "GridLayoutComponent": "core.components.gui.layout.horizontal_layout_component",
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return list(_LAZY_IMPORTS.keys())


def _build_component_map() -> dict:
    from core.components.gui.containers.panel_component import PanelComponent
    from core.components.gui.controls.label_component import LabelComponent
    from core.components.gui.controls.button_component import ButtonComponent
    from core.components.gui.controls.slider_component import SliderComponent
    from core.components.gui.controls.textinput_component import TextInputComponent
    from core.components.gui.controls.toggle_component import ToggleComponent
    from core.components.gui.controls.progressbar_component import ProgressBarComponent
    from core.components.gui.controls.dropdown_component import DropdownComponent
    from core.components.gui.containers.scrollpanel_component import ScrollPanelComponent
    from core.components.gui.controls.image_component import ImageComponent
    from core.components.gui.controls.radiobutton_component import RadioButtonComponent
    from core.components.gui.containers.listwidget_component import ListWidgetComponent
    from core.components.gui.containers.tablewidget_component import TableWidgetComponent
    from core.components.gui.containers.treewidget_component import TreeWidgetComponent
    from core.components.gui.containers.tabwidget_component import TabWidgetComponent
    from core.components.gui.containers.groupbox_component import GroupBoxComponent
    from core.components.gui.controls.spinbox_component import SpinBoxComponent
    from core.components.gui.controls.doublespinbox_component import DoubleSpinBoxComponent
    from core.components.gui.controls.textedit_component import TextEditComponent
    from core.components.gui.controls.dial_component import DialComponent
    from core.components.gui.controls.html_component import HtmlComponent
    from core.components.gui.containers.splitter_component import SplitterComponent
    from core.components.gui.containers.stacked_component import StackedComponent
    from core.components.gui.containers.toolbox_component import ToolBoxComponent
    from core.components.gui.controls.calendar_component import CalendarComponent
    from core.components.gui.controls.lcd_component import LCDComponent
    from core.components.gui.controls.plaintext_component import PlainTextComponent
    from core.components.gui.containers.scrollbar_component import ScrollBarComponent
    from core.components.gui.controls.toolbutton_component import ToolButtonComponent
    from core.components.gui.controls.fontcombo_component import FontComboComponent
    from core.components.gui.containers.mdiarea_component import MdiAreaComponent
    return {
        "panel": PanelComponent,
        "label": LabelComponent,
        "button": ButtonComponent,
        "slider": SliderComponent,
        "textinput": TextInputComponent,
        "toggle": ToggleComponent,
        "progressbar": ProgressBarComponent,
        "dropdown": DropdownComponent,
        "scrollpanel": ScrollPanelComponent,
        "image": ImageComponent,
        "radiobutton": RadioButtonComponent,
        "listwidget": ListWidgetComponent,
        "tablewidget": TableWidgetComponent,
        "treewidget": TreeWidgetComponent,
        "tabwidget": TabWidgetComponent,
        "groupbox": GroupBoxComponent,
        "spinbox": SpinBoxComponent,
        "doublespinbox": DoubleSpinBoxComponent,
        "textedit": TextEditComponent,
        "dial": DialComponent,
        "html": HtmlComponent,
        "splitter": SplitterComponent,
        "stackedwidget": StackedComponent,
        "toolbox": ToolBoxComponent,
        "calendar": CalendarComponent,
        "lcdnumber": LCDComponent,
        "plaintext": PlainTextComponent,
        "scrollbar": ScrollBarComponent,
        "toolbutton": ToolButtonComponent,
        "fontcombo": FontComboComponent,
        "mdiarea": MdiAreaComponent,
    }

GUI_COMPONENT_MAP = None

def _ensure_component_map() -> dict:
    global GUI_COMPONENT_MAP
    if GUI_COMPONENT_MAP is None:
        GUI_COMPONENT_MAP = _build_component_map()
    return GUI_COMPONENT_MAP

LAYOUT_COMP_NAMES = ["HorizontalLayoutComponent", "VerticalLayoutComponent", "GridLayoutComponent"]
LAYOUT_ELEMENT_NAME = "LayoutElementComponent"

__all__ = [
    "GuiWidgetComponentBase",
    "PanelComponent", "LabelComponent", "ButtonComponent", "SliderComponent",
    "TextInputComponent", "ToggleComponent", "ProgressBarComponent",
    "DropdownComponent", "ScrollPanelComponent", "ImageComponent",
    "RadioButtonComponent", "ListWidgetComponent", "TableWidgetComponent",
    "TreeWidgetComponent", "TabWidgetComponent", "GroupBoxComponent",
    "SpinBoxComponent", "DoubleSpinBoxComponent", "TextEditComponent",
    "DialComponent", "HtmlComponent", "SplitterComponent", "StackedComponent",
    "ToolBoxComponent", "CalendarComponent", "LCDComponent", "PlainTextComponent",
    "ScrollBarComponent", "ToolButtonComponent", "FontComboComponent",
    "MdiAreaComponent", "TooltipComponent", "LayoutElementComponent",
    "HorizontalLayoutComponent", "VerticalLayoutComponent", "GridLayoutComponent",
    "GUI_COMPONENT_MAP", "_ensure_component_map",
]
