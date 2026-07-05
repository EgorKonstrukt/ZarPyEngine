# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from typing import Optional
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QListWidgetItem, QSplitter, QScrollArea,
    QFrame, QDoubleSpinBox, QLineEdit, QComboBox, QMenu,
    QInputDialog, QMessageBox, QGraphicsView, QGraphicsScene,
    QGraphicsItem, QGraphicsObject, QSizePolicy, QCheckBox,
    QGraphicsEllipseItem, QGraphicsTextItem, QSpinBox, QGraphicsLineItem,
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QRectF, QPointF, QLineF, QTimer,
    QUuid, QObject,
)
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QMouseEvent,
    QKeyEvent, QAction, QPainterPath, QFontMetrics, QPolygonF,
    QTransform,
)
from core.components.animation.animator_controller import (
    AnimatorController, AnimatorState, AnimatorTransition,
    AnimatorParameter, AnimatorParameterType, AnimatorCondition,
    AnimatorConditionMode,
)
from core.editor_scale import scale, scale_xy

_STATE_WIDTH = 160
_STATE_HEIGHT = 50
_STATE_RADIUS = 8
_ENTRY_SIZE = 30
_PORT_RADIUS = 6

_BG_COLOR = QColor(37, 37, 37)
_GRID_COLOR = QColor(50, 50, 50)
_STATE_BG = QColor(55, 55, 55)
_STATE_BORDER = QColor(120, 120, 120)
_STATE_SEL_BORDER = QColor(90, 156, 245)
_ENTRY_COLOR = QColor(60, 180, 60)
_ANY_STATE_COLOR = QColor(180, 160, 60)
_TEXT_COLOR = QColor(200, 200, 200)
_TEXT_DIM = QColor(140, 140, 140)
_TRANSITION_COLOR = QColor(150, 150, 150)
_TRANSITION_SEL_COLOR = QColor(90, 156, 245)
_CONDITION_TRUE = QColor(60, 180, 60)
_CONDITION_FALSE = QColor(200, 80, 80)


class StateNodeItem(QGraphicsObject):
    PORT_TOP = 0
    PORT_BOTTOM = 1
    PORT_LEFT = 2
    PORT_RIGHT = 3

    def __init__(self, state: AnimatorState, is_entry: bool = False, is_any: bool = False):
        super().__init__()
        self._state = state
        self._is_entry = is_entry
        self._is_any = is_any
        self._selected = False
        self._ports: dict[int, QGraphicsEllipseItem] = {}
        self._setup_visuals()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not is_entry)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setCacheMode(QGraphicsItem.CacheMode.ItemCoordinateCache)
        if is_entry:
            self.setPos(state.x, state.y) if (state.x or state.y) else self.setPos(-200, 0)
        elif is_any:
            self.setPos(state.x, state.y) if (state.x or state.y) else self.setPos(-200, 100)
        else:
            self.setPos(state.x, state.y)

    def _setup_visuals(self):
        if self._is_entry:
            w = h = _ENTRY_SIZE
        elif self._is_any:
            w = 120
            h = _STATE_HEIGHT
        else:
            w = _STATE_WIDTH
            h = _STATE_HEIGHT
        self._w = w
        self._h = h
        if not self._is_entry:
            for side in [self.PORT_TOP, self.PORT_BOTTOM]:
                port = QGraphicsEllipseItem(-_PORT_RADIUS, -_PORT_RADIUS, _PORT_RADIUS * 2, _PORT_RADIUS * 2, self)
                port.setBrush(QBrush(QColor(100, 100, 100)))
                port.setPen(QPen(QColor(80, 80, 80), 1))
                port.setVisible(False)
                self._ports[side] = port
        self._update_label()

    def _update_label(self):
        children = [c for c in self.childItems() if isinstance(c, QGraphicsTextItem)]
        for c in children:
            c.deleteLater()
        if self._is_entry:
            label = QGraphicsTextItem("Entry", self)
            label.setDefaultTextColor(_TEXT_COLOR)
            font = QFont("Segoe UI", 8, QFont.Weight.Bold)
            label.setFont(font)
            label.setPos(-label.boundingRect().width() / 2, -label.boundingRect().height() / 2)
            return
        display_name = self._state.name
        if self._is_any:
            display_name = "Any State"
        label = QGraphicsTextItem(display_name, self)
        label.setDefaultTextColor(_TEXT_COLOR)
        font = QFont("Segoe UI", 9)
        label.setFont(font)
        lw = label.boundingRect().width()
        lh = label.boundingRect().height()
        label.setPos((self._w - lw) / 2, (self._h - lh) / 2 - 4)
        if self._state.clip:
            clip_label = QGraphicsTextItem(self._state.clip.name, self)
            clip_label.setDefaultTextColor(_TEXT_DIM)
            font2 = QFont("Segoe UI", 7)
            clip_label.setFont(font2)
            clw = clip_label.boundingRect().width()
            clip_label.setPos((self._w - clw) / 2, self._h / 2 + 2)

    def boundingRect(self):
        if self._is_entry:
            return QRectF(-_ENTRY_SIZE / 2, -_ENTRY_SIZE / 2, _ENTRY_SIZE, _ENTRY_SIZE)
        return QRectF(0, 0, self._w, self._h)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._is_entry:
            rect = QRectF(-_ENTRY_SIZE / 2, -_ENTRY_SIZE / 2, _ENTRY_SIZE, _ENTRY_SIZE)
            painter.setBrush(QBrush(_ENTRY_COLOR))
            painter.setPen(QPen(QColor(40, 160, 40), 2))
            painter.drawEllipse(rect)
            return
        rect = QRectF(0, 0, self._w, self._h)
        border_color = _STATE_SEL_BORDER if self._selected else _STATE_BORDER
        border_w = 2 if self._selected else 1
        bg = _ANY_STATE_COLOR if self._is_any else _STATE_BG
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border_color, border_w))
        painter.drawRoundedRect(rect, _STATE_RADIUS, _STATE_RADIUS)

    def set_selected(self, sel: bool):
        self._selected = sel
        self.update()
        for port in self._ports.values():
            port.setVisible(sel)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.set_selected(bool(value))
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if self._state:
                self._state.x = self.pos().x()
                self._state.y = self.pos().y()
        return super().itemChange(change, value)

    def port_pos(self, side: int) -> QPointF:
        if side == self.PORT_TOP:
            return self.mapToScene(QPointF(self._w / 2, 0))
        elif side == self.PORT_BOTTOM:
            return self.mapToScene(QPointF(self._w / 2, self._h))
        elif side == self.PORT_LEFT:
            return self.mapToScene(QPointF(0, self._h / 2))
        elif side == self.PORT_RIGHT:
            return self.mapToScene(QPointF(self._w, self._h / 2))
        return self.mapToScene(QPointF(self._w / 2, self._h / 2))

    def closest_port(self, target: QPointF) -> tuple[int, QPointF]:
        best_side = self.PORT_BOTTOM
        best_dist = float("inf")
        for side in [self.PORT_TOP, self.PORT_BOTTOM, self.PORT_LEFT, self.PORT_RIGHT]:
            pt = self.port_pos(side)
            d = (pt - target).manhattanLength()
            if d < best_dist:
                best_dist = d
                best_side = side
        return best_side, self.port_pos(best_side)


class TransitionArrowItem(QGraphicsObject):
    def __init__(self, source: StateNodeItem, target: StateNodeItem, transition: AnimatorTransition):
        super().__init__()
        self._source = source
        self._target = target
        self._transition = transition
        self._selected = False
        self.setZValue(-1)
        self._update_path()

    def _update_path(self):
        if not self._source or not self._target:
            return
        s_side, s_pt = self._source.closest_port(self._target.pos())
        t_side, t_pt = self._target.closest_port(self._source.pos())
        self._start = s_pt
        self._end = t_pt
        dx = self._end.x() - self._start.x()
        dy = self._end.y() - self._start.y()
        dist = math.sqrt(dx * dx + dy * dy)
        ctrl_offset = dist * 0.4
        if abs(dx) > abs(dy):
            self._cp1 = QPointF(self._start.x() + dx * 0.5, self._start.y())
            self._cp2 = QPointF(self._end.x() - dx * 0.5, self._end.y())
        else:
            self._cp1 = QPointF(self._start.x(), self._start.y() + dy * 0.5)
            self._cp2 = QPointF(self._end.x(), self._end.y() - dy * 0.5)

    def boundingRect(self):
        return QRectF(min(self._start.x(), self._end.x()) - 10,
                      min(self._start.y(), self._end.y()) - 10,
                      abs(self._end.x() - self._start.x()) + 20,
                      abs(self._end.y() - self._start.y()) + 20)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = _TRANSITION_SEL_COLOR if self._selected else _TRANSITION_COLOR
        painter.setPen(QPen(color, 2))
        path = QPainterPath()
        path.moveTo(self._start)
        path.cubicTo(self._cp1, self._cp2, self._end)
        painter.drawPath(path)
        angle = math.atan2(self._end.y() - self._cp2.y(), self._end.x() - self._cp2.x())
        arrow_size = 10
        p1 = QPointF(self._end.x() - arrow_size * math.cos(angle - 0.4),
                     self._end.y() - arrow_size * math.sin(angle - 0.4))
        p2 = QPointF(self._end.x() - arrow_size * math.cos(angle + 0.4),
                     self._end.y() - arrow_size * math.sin(angle + 0.4))
        painter.setBrush(QBrush(color))
        arrow = QPolygonF([self._end, p1, p2])
        painter.drawPolygon(arrow)

    def set_selected(self, sel: bool):
        self._selected = sel
        self.update()

    def contains_point(self, pt: QPointF) -> bool:
        return self.boundingRect().contains(pt)

    def source(self) -> StateNodeItem:
        return self._source

    def target(self) -> StateNodeItem:
        return self._target

    def transition(self) -> AnimatorTransition:
        return self._transition


class AnimatorScene(QGraphicsScene):
    transition_created = pyqtSignal(object, object)
    node_selected = pyqtSignal(object)
    transition_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._controller: Optional[AnimatorController] = None
        self._nodes: dict[str, StateNodeItem] = {}
        self._arrows: list[TransitionArrowItem] = []
        self._entry_node: Optional[StateNodeItem] = None
        self._any_node: Optional[StateNodeItem] = None
        self._drag_line: Optional[QGraphicsLineItem] = None
        self._drag_source: Optional[StateNodeItem] = None
        self._selected_node: Optional[StateNodeItem] = None
        self._selected_arrow: Optional[TransitionArrowItem] = None
        self.setBackgroundBrush(_BG_COLOR)

    def set_controller(self, ctrl: Optional[AnimatorController]):
        self._controller = ctrl
        self.clear()
        self._nodes.clear()
        self._arrows.clear()
        self._entry_node = None
        self._any_node = None
        self._drag_line = None
        self._drag_source = None
        self._selected_node = None
        self._selected_arrow = None
        if not ctrl:
            return
        self._entry_node = StateNodeItem(AnimatorState(name="Entry"), is_entry=True)
        self.addItem(self._entry_node)
        any_state = AnimatorState(name="Any State")
        self._any_node = StateNodeItem(any_state, is_any=True)
        self.addItem(self._any_node)
        if ctrl.layers:
            layer = ctrl.layers[0]
            for state in layer.states:
                self._add_state_node(state)
            for state in layer.states:
                for trans in state.transitions:
                    self._add_transition_arrow(state.name, trans.destination_state, trans)
        self._rebuild_entry_transitions()

    def _add_state_node(self, state: AnimatorState):
        node = StateNodeItem(state)
        self._nodes[state.name] = node
        self.addItem(node)
        return node

    def _add_transition_arrow(self, from_name: str, to_name: str, trans: AnimatorTransition):
        src = self._nodes.get(from_name) or self._any_node
        tgt = self._nodes.get(to_name)
        if src and tgt:
            arrow = TransitionArrowItem(src, tgt, trans)
            self._arrows.append(arrow)
            self.addItem(arrow)

    def _rebuild_entry_transitions(self):
        for arrow in self._arrows[:]:
            self.removeItem(arrow)
        self._arrows.clear()
        if not self._controller or not self._controller.layers:
            return
        layer = self._controller.layers[0]
        for state in layer.states:
            for trans in state.transitions:
                self._add_transition_arrow(state.name, trans.destination_state, trans)

    def add_state(self, name: str = "New State") -> Optional[AnimatorState]:
        if not self._controller:
            return None
        state = self._controller.add_state(0, AnimatorState(name=name))
        node = self._add_state_node(state)
        node.setPos(state.x, state.y)
        return state

    def remove_state(self, name: str):
        if not self._controller:
            return
        self._controller.remove_state(0, name)
        node = self._nodes.pop(name, None)
        if node:
            self.removeItem(node)
        self._rebuild_entry_transitions()

    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), QTransform())
        if isinstance(item, StateNodeItem):
            self._selected_node = item
            self._selected_arrow = None
            self.node_selected.emit(item._state if not item._is_entry and not item._is_any else None)
            super().mousePressEvent(event)
            return
        if isinstance(item, TransitionArrowItem):
            self._selected_arrow = item
            self._selected_node = None
            self.transition_selected.emit(item.transition())
            super().mousePressEvent(event)
            return
        if isinstance(item, QGraphicsEllipseItem) and item.parentItem() and isinstance(item.parentItem(), StateNodeItem):
            parent_node = item.parentItem()
            self._drag_source = parent_node
            self._drag_line = self.addLine(QLineF(event.scenePos(), event.scenePos()), QPen(_TRANSITION_COLOR, 2, Qt.PenStyle.DashLine))
            return
        self._selected_node = None
        self._selected_arrow = None
        self.node_selected.emit(None)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_line and self._drag_source:
            line = QLineF(self._drag_source.port_pos(StateNodeItem.PORT_BOTTOM), event.scenePos())
            self._drag_line.setLine(line)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_line:
            self.removeItem(self._drag_line)
            self._drag_line = None
            target_item = self.itemAt(event.scenePos(), QTransform())
            if target_item and isinstance(target_item, (StateNodeItem, QGraphicsEllipseItem)):
                if isinstance(target_item, QGraphicsEllipseItem):
                    target_node = target_item.parentItem()
                else:
                    target_node = target_item
                if isinstance(target_node, StateNodeItem) and target_node != self._drag_source:
                    src_state = self._drag_source._state
                    tgt_state = target_node._state
                    if src_state and tgt_state:
                        trans = AnimatorTransition(destination_state=tgt_state.name)
                        src_state.transitions.append(trans)
                        self._add_transition_arrow(src_state.name, tgt_state.name, trans)
                        self.transition_created.emit(src_state, trans)
                        self._drag_source = None
                        return
            self._drag_source = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            if self._selected_node and self._selected_node._state:
                self.remove_state(self._selected_node._state.name)
                self._selected_node = None
                self.node_selected.emit(None)
            elif self._selected_arrow:
                self._remove_transition(self._selected_arrow)
                self._selected_arrow = None
                self.transition_selected.emit(None)
        super().keyPressEvent(event)

    def _remove_transition(self, arrow: TransitionArrowItem):
        src = arrow.source()._state
        trans = arrow.transition()
        if trans in src.transitions:
            src.transitions.remove(trans)
        if arrow in self._arrows:
            self._arrows.remove(arrow)
        self.removeItem(arrow)

    def contextMenuEvent(self, event):
        menu = QMenu()
        add_state_action = QAction("Create State", None)
        add_state_action.triggered.connect(lambda: self.add_state())
        menu.addAction(add_state_action)
        if self._selected_node and self._selected_node._state:
            delete_action = QAction("Delete State", None)
            delete_action.triggered.connect(lambda: self.remove_state(self._selected_node._state.name))
            menu.addAction(delete_action)
            rename_action = QAction("Rename", None)
            rename_action.triggered.connect(lambda: self._rename_state(self._selected_node._state))
            menu.addAction(rename_action)
        if self._selected_arrow:
            del_trans_action = QAction("Delete Transition", None)
            del_trans_action.triggered.connect(lambda: self._remove_transition(self._selected_arrow))
            menu.addAction(del_trans_action)
        menu.exec(event.screenPos())

    def _rename_state(self, state: AnimatorState):
        name, ok = QInputDialog.getText(None, "Rename State", "Name:", text=state.name)
        if ok and name:
            old_name = state.name
            state.name = name
            if old_name in self._nodes:
                node = self._nodes.pop(old_name)
                self._nodes[name] = node
                node._update_label()
                node.update()
                for arrow in self._arrows[:]:
                    if arrow.source()._state is state or arrow.transition().destination_state == old_name:
                        if arrow.transition().destination_state == old_name:
                            arrow.transition().destination_state = name
                        arrow._update_path()

    def find_node(self, state_name: str) -> Optional[StateNodeItem]:
        return self._nodes.get(state_name)


class AnimatorPanel(QDockWidget):
    state_selected_signal = pyqtSignal(object, object)
    transition_selected_signal = pyqtSignal(object, object)
    selection_cleared = pyqtSignal()

    def __init__(self, engine, main_window):
        super().__init__("Animator", main_window)
        self._engine = engine
        self._main_window = main_window
        self._controller: Optional[AnimatorController] = None
        self._selected_state: Optional[AnimatorState] = None
        self._selected_transition: Optional[AnimatorTransition] = None
        self.setObjectName("AnimatorDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.setMinimumSize(300, 200)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        toolbar = self._build_toolbar()
        main_layout.addWidget(toolbar)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._params_panel = self._build_params_panel()
        splitter.addWidget(self._params_panel)
        self._scene = AnimatorScene()
        self._scene.node_selected.connect(self._on_node_selected)
        self._scene.transition_selected.connect(self._on_transition_selected)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self._view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._view.setStyleSheet("background: transparent; border: none;")
        splitter.addWidget(self._view)
        splitter.setSizes([200, 600])
        main_layout.addWidget(splitter, 1)

    def _build_toolbar(self) -> QWidget:
        tb = QWidget()
        layout = QHBoxLayout(tb)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)
        self._add_state_btn = QPushButton("+ State")
        self._add_state_btn.setFixedHeight(scale(22))
        self._add_state_btn.clicked.connect(self._add_state)
        layout.addWidget(self._add_state_btn)
        self._add_param_btn = QPushButton("+ Parameter")
        self._add_param_btn.setFixedHeight(scale(22))
        self._add_param_btn.clicked.connect(self._add_parameter)
        layout.addWidget(self._add_param_btn)
        self._ctrl_name_label = QLabel("No Controller")
        self._ctrl_name_label.setStyleSheet("font-size: 11px; padding: 0 8px;")
        layout.addWidget(self._ctrl_name_label)
        layout.addStretch()
        self._live_link_cb = QCheckBox("Live Link")
        self._live_link_cb.setStyleSheet("font-size: 10px;")
        self._live_link_cb.setChecked(True)
        layout.addWidget(self._live_link_cb)
        return tb

    def _btn_style(self) -> str:
        return ""

    def _build_params_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QLabel("Parameters")
        header.setStyleSheet("font-size: 10px; font-weight: bold; padding: 4px 6px; "
                             "border-bottom: 1px solid;")
        layout.addWidget(header)
        self._param_list = QListWidget()
        self._param_list.setStyleSheet("border: none; font-size: 10px;")
        self._param_list.itemClicked.connect(self._on_param_selected)
        self._param_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._param_list.customContextMenuRequested.connect(self._param_context_menu)
        layout.addWidget(self._param_list, 1)
        add_row = QWidget()
        ar_layout = QHBoxLayout(add_row)
        ar_layout.setContentsMargins(4, 2, 4, 2)
        self._param_name_edit = QLineEdit()
        self._param_name_edit.setPlaceholderText("param name")
        self._param_name_edit.setStyleSheet("border: 1px solid; "
                                            "border-radius: 3px; padding: 2px 4px; font-size: 10px;")
        ar_layout.addWidget(self._param_name_edit)
        self._param_type_combo = QComboBox()
        self._param_type_combo.addItems(["Float", "Int", "Bool", "Trigger"])
        self._param_type_combo.setStyleSheet("border: 1px solid; "
                                             "border-radius: 3px; font-size: 10px;")
        ar_layout.addWidget(self._param_type_combo)
        add_btn = QPushButton("+")
        add_btn.setFixedSize(*scale_xy(22, 22))
        add_btn.clicked.connect(self._add_parameter)
        ar_layout.addWidget(add_btn)
        layout.addWidget(add_row)
        return w

    def load_controller(self, ctrl: Optional[AnimatorController]):
        self._controller = ctrl
        self._ctrl_name_label.setText(ctrl.name if ctrl else "No Controller")
        self._scene.set_controller(ctrl)
        self._update_param_list()
        self._selected_state = None
        self._selected_transition = None
        self.selection_cleared.emit()

    def _add_state(self):
        if not self._controller:
            QMessageBox.warning(self, "No Controller", "Create an Animator Controller first.")
            return
        name, ok = QInputDialog.getText(self, "New State", "State name:", text="New State")
        if ok and name:
            self._scene.add_state(name)

    def _add_parameter(self):
        if not self._controller:
            QMessageBox.warning(self, "No Controller", "Create an Animator Controller first.")
            return
        name = self._param_name_edit.text().strip() or "NewParam"
        ptype_str = self._param_type_combo.currentText()
        ptype_map = {"Float": AnimatorParameterType.FLOAT, "Int": AnimatorParameterType.INT,
                     "Bool": AnimatorParameterType.BOOL, "Trigger": AnimatorParameterType.TRIGGER}
        param = AnimatorParameter(name=name, param_type=ptype_map[ptype_str])
        self._controller.add_parameter(param)
        self._update_param_list()
        self._param_name_edit.clear()

    def _update_param_list(self):
        self._param_list.clear()
        if not self._controller:
            return
        for p in self._controller.parameters:
            item = QListWidgetItem(f"{p.name} ({p.param_type.value})")
            item.setData(Qt.ItemDataRole.UserRole, p.name)
            self._param_list.addItem(item)

    def _param_context_menu(self, pos):
        item = self._param_list.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        del_action = QAction("Delete Parameter", self)
        param_name = item.data(Qt.ItemDataRole.UserRole)
        del_action.triggered.connect(lambda: self._delete_param(param_name))
        menu.addAction(del_action)
        menu.exec(self._param_list.mapToGlobal(pos))

    def _delete_param(self, name: str):
        if self._controller:
            self._controller.remove_parameter(name)
            self._update_param_list()

    def _on_param_selected(self, item):
        pass

    def _on_node_selected(self, state):
        self._selected_state = state
        self._selected_transition = None
        if state:
            self.state_selected_signal.emit(state, self._controller)
        else:
            self.selection_cleared.emit()

    def _on_transition_selected(self, transition):
        self._selected_transition = transition
        self._selected_state = None
        if transition:
            self.transition_selected_signal.emit(transition, self._controller)
        else:
            self.selection_cleared.emit()

    def load_config(self, config):
        pass
