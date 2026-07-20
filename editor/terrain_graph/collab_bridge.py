# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import threading
from collections import deque
from typing import Optional, TYPE_CHECKING
from PyQt6.QtCore import Qt, QObject, QTimer
from PyQt6.QtGui import QColor, QBrush, QPen, QFont
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem
from core.network.protocol import MessageType
from core.foundation.logger import Logger

if TYPE_CHECKING:
    from core.network.collaboration import CollaborationManager


class TerrainGraphCollabBridge(QObject):
    CURSOR_RADIUS = 6
    CURSOR_COLORS = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (255, 180, 100), (180, 100, 255),
    ]

    def __init__(self, graph_widget, collab_mgr: Optional[CollaborationManager] = None):
        super().__init__(graph_widget)
        self._widget = graph_widget
        self._graph = graph_widget._graph
        self._view = graph_widget._view
        self._collab = collab_mgr
        self._suppress = False
        self._cursor_items: dict[str, list] = {}
        self._own_peer_id = ""
        self._sync_requested = False

        self._cursor_timer = QTimer(self)
        self._cursor_timer.setInterval(33)
        self._cursor_timer.timeout.connect(self._send_cursor)
        self._last_cursor_pos: Optional[tuple[float, float]] = None

        self._remote_queue: deque[tuple[int, dict]] = deque()
        self._remote_lock = threading.Lock()
        self._apply_timer = QTimer(self)
        self._apply_timer.setInterval(0)
        self._apply_timer.timeout.connect(self._process_remote_queue)

        self._sync_check_timer = QTimer(self)
        self._sync_check_timer.setInterval(1000)
        self._sync_check_timer.timeout.connect(self._check_sync_on_connect)

        self._setup()
        if self._collab:
            self.set_collaboration_manager(self._collab)

    def set_collaboration_manager(self, mgr):
        self._collab = mgr
        self._own_peer_id = mgr.own_peer_id or ""
        for t in (
            MessageType.GRAPH_NODE_MOVE,
            MessageType.GRAPH_PORT_CONNECT,
            MessageType.GRAPH_PORT_DISCONNECT,
            MessageType.GRAPH_PARAM_CHANGE,
            MessageType.GRAPH_NODE_ADD,
            MessageType.GRAPH_NODE_DELETE,
            MessageType.GRAPH_CURSOR,
            MessageType.GRAPH_SYNC,
            MessageType.GRAPH_SYNC_REQ,
        ):
            mgr.register_handler(t, lambda d, _t=t: self._queue_remote(_t, d))
        self._cursor_timer.start()
        self._apply_timer.start()
        self._sync_check_timer.start()
        if mgr.connected:
            self._sync_requested = True
            self._sync_check_timer.stop()
            self._request_sync()

    def _check_sync_on_connect(self):
        if not self._sync_requested and self._collab and self._collab.connected:
            self._sync_requested = True
            self._own_peer_id = self._collab.own_peer_id or ""
            self._sync_check_timer.stop()
            self._request_sync()

    def _setup(self):
        g = self._graph
        v = self._view
        g.node_created.connect(self._on_node_created)
        g.nodes_deleted.connect(self._on_nodes_deleted)
        g.port_connected.connect(self._on_port_connected)
        g.port_disconnected.connect(self._on_port_disconnected)
        g.property_changed.connect(self._on_property_changed)
        v.moved_nodes.connect(self._on_moved_nodes)
        v.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self._view and event.type() == event.Type.MouseMove:
            sp = self._view.mapToScene(event.pos())
            self._last_cursor_pos = (sp.x(), sp.y())
        return super().eventFilter(obj, event)

    # --- Cursor sending ---

    def _send_cursor(self):
        if self._collab and self._collab.connected and self._last_cursor_pos:
            x, y = self._last_cursor_pos
            self._collab._client.send(MessageType.GRAPH_CURSOR, {"x": x, "y": y})

    # --- Queue remote operations (called from poll thread) ---

    def _queue_remote(self, msg_type: int, data: dict):
        with self._remote_lock:
            self._remote_queue.append((msg_type, data))

    def _process_remote_queue(self):
        with self._remote_lock:
            items = list(self._remote_queue)
            self._remote_queue.clear()
        for msg_type, data in items:
            self._dispatch_remote(msg_type, data)

    def _request_sync(self):
        if self._collab and self._collab.connected:
            self._collab._client.send(MessageType.GRAPH_SYNC_REQ, {})

    def _send_full_sync(self):
        if self._collab and self._collab.connected:
            session = self._graph.serialize_session()
            self._collab._client.send(MessageType.GRAPH_SYNC, {"session": session})

    def _dispatch_remote(self, msg_type: int, data: dict):
        pid = data.get("id", "")
        if pid == self._own_peer_id:
            return
        self._suppress = True
        try:
            if msg_type == MessageType.GRAPH_NODE_MOVE:
                self._apply_node_move(data)
            elif msg_type == MessageType.GRAPH_PORT_CONNECT:
                self._apply_port_connect(data)
            elif msg_type == MessageType.GRAPH_PORT_DISCONNECT:
                self._apply_port_disconnect(data)
            elif msg_type == MessageType.GRAPH_PARAM_CHANGE:
                self._apply_param_change(data)
            elif msg_type == MessageType.GRAPH_NODE_ADD:
                self._apply_node_add(data)
            elif msg_type == MessageType.GRAPH_NODE_DELETE:
                self._apply_node_delete(data)
            elif msg_type == MessageType.GRAPH_CURSOR:
                self._apply_cursor(data)
            elif msg_type == MessageType.GRAPH_SYNC:
                self._apply_graph_sync(data)
            elif msg_type == MessageType.GRAPH_SYNC_REQ:
                self._send_full_sync()
        except Exception as e:
            Logger.warning(f"GraphCollab: remote apply error: {e}")
        finally:
            self._suppress = False

    # --- Local signal handlers (send to peers) ---

    def _should_send(self):
        return not self._suppress and self._collab and self._collab.connected

    def _send(self, msg_type: int, data: dict):
        if self._collab and self._collab.connected:
            self._collab._client.send(msg_type, data)

    def _on_node_created(self, node):
        if not self._should_send():
            return
        self._send(MessageType.GRAPH_NODE_ADD, {
            "type": node.type_(),
            "node_id": node.id,
            "x": node.pos()[0],
            "y": node.pos()[1],
        })

    def _on_nodes_deleted(self, node_ids):
        if not self._should_send():
            return
        self._send(MessageType.GRAPH_NODE_DELETE, {"node_ids": node_ids})

    def _on_moved_nodes(self, node_data):
        if not self._should_send():
            return
        for nv, _ in node_data.items():
            node = self._graph.get_node_by_id(nv.id)
            if node:
                x, y = node.pos()
                self._send(MessageType.GRAPH_NODE_MOVE, {
                    "node_id": nv.id, "x": x, "y": y
                })

    def _on_port_connected(self, in_port, out_port):
        if not self._should_send():
            return
        self._send(MessageType.GRAPH_PORT_CONNECT, {
            "from_node": out_port.node().id,
            "from_port": out_port.name(),
            "to_node": in_port.node().id,
            "to_port": in_port.name(),
        })

    def _on_port_disconnected(self, in_port, out_port):
        if not self._should_send():
            return
        self._send(MessageType.GRAPH_PORT_DISCONNECT, {
            "from_node": out_port.node().id,
            "from_port": out_port.name(),
            "to_node": in_port.node().id,
            "to_port": in_port.name(),
        })

    def _on_property_changed(self, node, prop_name, prop_value):
        if not self._should_send():
            return
        if prop_name == "pos":
            return
        self._send(MessageType.GRAPH_PARAM_CHANGE, {
            "node_id": node.id,
            "prop": prop_name,
            "value": prop_value,
        })

    # --- Remote apply handlers ---

    def _apply_node_move(self, data: dict):
        nid = data.get("node_id", "")
        x = float(data.get("x", 0))
        y = float(data.get("y", 0))
        node = self._graph.get_node_by_id(nid)
        if node:
            node.set_property("pos", [x, y], push_undo=False)

    def _apply_port_connect(self, data: dict):
        from_n = self._graph.get_node_by_id(data.get("from_node", ""))
        to_n = self._graph.get_node_by_id(data.get("to_node", ""))
        if not from_n or not to_n:
            return
        fp = from_n.outputs().get(data.get("from_port", ""))
        tp = to_n.inputs().get(data.get("to_port", ""))
        if fp and tp:
            fp.connect_to(tp, push_undo=False, emit_signal=True)

    def _apply_port_disconnect(self, data: dict):
        from_n = self._graph.get_node_by_id(data.get("from_node", ""))
        to_n = self._graph.get_node_by_id(data.get("to_node", ""))
        if not from_n or not to_n:
            return
        fp = from_n.outputs().get(data.get("from_port", ""))
        tp = to_n.inputs().get(data.get("to_port", ""))
        if fp and tp:
            fp.disconnect_from(tp, push_undo=False, emit_signal=True)

    def _apply_param_change(self, data: dict):
        node = self._graph.get_node_by_id(data.get("node_id", ""))
        if node:
            prop = data.get("prop", "")
            value = data.get("value")
            if prop:
                node.set_property(prop, value, push_undo=False)

    def _apply_node_add(self, data: dict):
        ntype = data.get("type", "")
        nid = data.get("node_id", "")
        x = float(data.get("x", 0))
        y = float(data.get("y", 0))
        node = self._graph.create_node(ntype, push_undo=False)
        if node:
            old_id = node.id
            if old_id != nid:
                self._graph.model.nodes.pop(old_id, None)
                node.model.id = nid
                node.view.id = nid
                self._graph.model.nodes[nid] = node
            node.set_property("pos", [x, y], push_undo=False)

    def _apply_node_delete(self, data: dict):
        node_ids = data.get("node_ids", [])
        nodes = []
        for nid in node_ids:
            n = self._graph.get_node_by_id(nid)
            if n:
                nodes.append(n)
        if nodes:
            self._graph.delete_nodes(nodes, push_undo=False)

    def _apply_cursor(self, data: dict):
        pid = data.get("id", "")
        if pid == self._own_peer_id:
            return
        if pid not in self._collab.peers:
            self.remove_peer_cursor(pid)
            return
        x = float(data.get("x", 0))
        y = float(data.get("y", 0))
        peer = self._collab.peers[pid]
        self._update_cursor_item(pid, peer, x, y)

    def _apply_graph_sync(self, data: dict):
        session = data.get("session", {})
        if session:
            self._graph.deserialize_session(session, clear_session=True, clear_undo_stack=True)

    # --- Cursor rendering ---

    def _update_cursor_item(self, peer_id: str, peer, x: float, y: float):
        items = self._cursor_items.get(peer_id)
        if items is None:
            scene = self._view.scene()
            dot = QGraphicsEllipseItem(-self.CURSOR_RADIUS, -self.CURSOR_RADIUS,
                                       self.CURSOR_RADIUS * 2, self.CURSOR_RADIUS * 2)
            dot.setZValue(9999)
            label = QGraphicsTextItem(peer.name)
            label.setDefaultTextColor(QColor(255, 255, 255))
            f = QFont("Segoe UI", 9)
            label.setFont(f)
            label.setZValue(9999)
            scene.addItem(dot)
            scene.addItem(label)
            items = [dot, label]
            self._cursor_items[peer_id] = items
        dot, label = items
        c = self._cursor_color_for(peer_id)
        dot.setBrush(QBrush(QColor(*c, 160)))
        dot.setPen(QPen(QColor(*c, 220), 1.5))
        dot.setPos(x, y)
        label.setPos(x + self.CURSOR_RADIUS + 2, y - 8)
        label.setPlainText(peer.name)

    def _cursor_color_for(self, peer_id: str) -> tuple:
        idx = abs(hash(peer_id)) % len(self.CURSOR_COLORS)
        return self.CURSOR_COLORS[idx]

    def remove_peer_cursor(self, peer_id: str):
        items = self._cursor_items.pop(peer_id, None)
        if items:
            scene = self._view.scene()
            for item in items:
                scene.removeItem(item)

    def clear_all_cursors(self):
        for pid in list(self._cursor_items.keys()):
            self.remove_peer_cursor(pid)
