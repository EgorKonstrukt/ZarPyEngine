# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import shutil
import datetime
import subprocess
from typing import Optional, TYPE_CHECKING
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QTreeWidget, QTreeWidgetItem, QPushButton,
                             QLabel, QLineEdit, QMenu, QFileDialog, QSplitter,
                             QListWidget, QListWidgetItem, QAbstractItemView,
                             QToolButton, QSlider, QFileIconProvider,
                             QStackedWidget, QFrame, QHeaderView, QSizePolicy,
                             QGraphicsDropShadowEffect, QStyledItemDelegate,
                             QAbstractItemDelegate, QApplication)
from PyQt6.QtCore import Qt, QEvent, pyqtSignal, QMimeData, QByteArray, QSize, QFileInfo, QUrl, QTimer
from PyQt6.QtGui import (QAction, QDrag, QIcon, QWheelEvent, QKeyEvent, QGuiApplication,
                          QShortcut, QKeySequence, QColor, QPainter, QPainterPath, QFont,
                          QPen, QBrush, QPixmap, QFontMetrics, QPalette)

try:
    import qtawesome as qta
except ImportError:
    qta = None

_QTA_COLORS = {
    "folder": "#d4d4d4",
    "file": "#d4d4d4",
    "nav": "#d4d4d4",
    "create": "#9ccc65",
    "refresh": "#d4d4d4",
    "dual": "#d4d4d4",
    "view": "#d4d4d4",
}


def _qta_icon(name: str, color: str | None = None) -> QIcon:
    if qta is None:
        return QIcon()
    c = color if color else _QTA_COLORS.get(name.split(".")[-1] if "." in name else name, "#d4d4d4")
    return qta.icon(name, color=c)

if TYPE_CHECKING:
    from core.engine.engine import Engine
from editor.resource_picker import _get_thumbnail, _format_size, _thumb_resolution

from editor.constants import MIN_THUMB, MAX_THUMB, VIEW_ICON, VIEW_LIST, VIEW_DETAILS
from editor.inspector.helpers import _flash_overlay

_ENTITY_MIME = "application/x-zpe-entity"
from core.config.editor_scale import scale, scale_xy

_file_clipboard: list[str] = []
_clipboard_is_cut: bool = False

_VCS_COLORS = {
    "untracked": QColor("#3C9B3C"),
    "added":     QColor("#3C9B3C"),
    "modified":  QColor("#3C6E9B"),
    "staged":    QColor("#3C6E9B"),
    "unstaged":  QColor("#3C6E9B"),
    "deleted":   QColor("#9B3C3C"),
    "conflict":  QColor("#9B3C3C"),
    "renamed":   QColor("#9B6E3C"),
}


def _parse_vcs_status(project_path: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for candidate in ["git.exe", "git"]:
        try:
            r = subprocess.run([candidate, "--version"], capture_output=True,
                               text=True, timeout=5)
            if r.returncode == 0:
                break
        except FileNotFoundError:
            continue
    else:
        return result
    try:
        r = subprocess.run([candidate, "status", "--porcelain", "-u"],
                           capture_output=True, text=True,
                           cwd=project_path, timeout=15)
    except Exception:
        return result
    for line in r.stdout.strip().split("\n"):
        line = line.rstrip()
        if not line or len(line) < 3:
            continue
        xy = line[:2]
        path = line[3:]
        if xy[0] == "?" and xy[1] == "?":
            result[path.replace("\\", "/")] = "untracked"
        elif xy[0] == "A" or xy[0] == "A ":
            result[path.replace("\\", "/")] = "added"
        elif xy[0] == "M" or xy[1] == "M":
            result[path.replace("\\", "/")] = "modified"
        elif xy[0] == "D" or xy[1] == "D":
            result[path.replace("\\", "/")] = "deleted"
        elif xy[0] == "R" or xy[1] == "R":
            result[path.replace("\\", "/")] = "renamed"
        elif xy[0] == "U" or xy[1] == "U":
            result[path.replace("\\", "/")] = "conflict"
    return result


class _NavButton(QToolButton):
    def __init__(self, icon_name="", tooltip="", parent=None):
        super().__init__(parent)
        if icon_name and qta is not None:
            self.setIcon(qta.icon(icon_name, color="#d4d4d4"))
            self.setIconSize(QSize(*scale_xy(14, 14)))
        self.setToolTip(tooltip)
        self.setFixedSize(scale(28), scale(28))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QToolButton {{
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 0;
            }}
            QToolButton:hover {{
            }}
            QToolButton:pressed {{
                background: #444;
            }}
            QToolButton:disabled {{
                color: #3a3a3a;
            }}
        """)

class _AddressBar(QLineEdit):
    path_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet(f"""
            QLineEdit {{
                border-radius: 2px;
                padding: 2px 8px;
                font-size: 12px;
                min-height: 22px;
            }}
            QLineEdit:focus {{
            }}
        """)

class _SearchBar(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Search...")
        self.setClearButtonEnabled(True)
        self.setMaximumWidth(scale(200))
        self.setStyleSheet(f"""
            QLineEdit {{
                border-radius: 2px;
                padding: 2px 24px 2px 8px;
                font-size: 11px;
                min-height: 22px;
            }}
            QLineEdit:focus {{
            }}
            QLineEdit::placeholder {{
            }}
        """)

class _GroupHeader(QWidget):
    def __init__(self, text, count=0, parent=None):
        super().__init__(parent)
        self._text = text
        self._count = count
        self.setFixedHeight(scale(26))
        self.setStyleSheet(f"""
        """)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = p.font()
        font.setPointSize(9)
        font.setBold(False)
        p.setFont(font)
        x = scale(8)
        y = (self.height() - p.fontMetrics().height()) // 2 + p.fontMetrics().ascent()
        p.drawText(x, y, self._text)
        if self._count > 0:
            count_text = f" ({self._count})"
            fm = QFontMetrics(font)
            text_w = fm.horizontalAdvance(self._text)
            p.drawText(x + text_w + scale(4), y, count_text)
        p.end()

class FileListWidget(QListWidget):
    def __init__(self, panel: ProjectPanel, parent=None):
        super().__init__(parent)
        self._panel = panel
        self.setAcceptDrops(True)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta != 0:
                step = 8 if abs(delta) > 120 else 4
                new_val = self._panel._thumb_size + (step if delta > 0 else -step)
                new_val = max(MIN_THUMB, min(new_val, MAX_THUMB))
                self._panel._thumb_size = new_val
                self._panel._zoom_slider.setValue(new_val)
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_F2:
            items = self.selectedItems()
            if items:
                self.editItem(items[0])
            return
        if key == Qt.Key.Key_Delete:
            self._panel._delete_selected()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            items = self.selectedItems()
            if items:
                self._panel._open_item(items[0])
            return
        if key == Qt.Key.Key_Backspace or (event.modifiers() == Qt.KeyboardModifier.AltModifier and key == Qt.Key.Key_Up):
            self._panel._go_to_parent()
            return
        if key == Qt.Key.Key_Home:
            self._panel._go_to_root()
            return
        if key == Qt.Key.Key_F5:
            self._panel._refresh()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(_ENTITY_MIME) or event.mimeData().hasUrls() or event.mimeData().hasText():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(_ENTITY_MIME) or event.mimeData().hasUrls() or event.mimeData().hasText():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasFormat(_ENTITY_MIME):
            self._panel._on_entity_drop(event)
        elif event.mimeData().hasUrls() or event.mimeData().hasText():
            self._panel._on_file_list_drop(event)
        else:
            super().dropEvent(event)

    def startDrag(self, supportedActions):
        self._panel._start_drag_list(supportedActions)

    def _open_item(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self._panel._open_path(path)

class FileDetailWidget(QTreeWidget):
    def __init__(self, panel: ProjectPanel, parent=None):
        super().__init__(parent)
        self._panel = panel
        self.setAcceptDrops(True)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta != 0:
                step = 8 if abs(delta) > 120 else 4
                new_val = self._panel._thumb_size + (step if delta > 0 else -step)
                new_val = max(MIN_THUMB, min(new_val, MAX_THUMB))
                self._panel._thumb_size = new_val
                self._panel._zoom_slider.setValue(new_val)
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_F2:
            items = self.selectedItems()
            if items:
                self.editItem(items[0], 0)
            return
        if key == Qt.Key.Key_Delete:
            self._panel._delete_selected()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            items = self.selectedItems()
            if items:
                self._panel._open_item_by_path(items[0].data(0, Qt.ItemDataRole.UserRole))
            return
        if key == Qt.Key.Key_Backspace or (event.modifiers() == Qt.KeyboardModifier.AltModifier and key == Qt.Key.Key_Up):
            self._panel._go_to_parent()
            return
        if key == Qt.Key.Key_Home:
            self._panel._go_to_root()
            return
        if key == Qt.Key.Key_F5:
            self._panel._refresh()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(_ENTITY_MIME) or event.mimeData().hasUrls() or event.mimeData().hasText():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(_ENTITY_MIME) or event.mimeData().hasUrls() or event.mimeData().hasText():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasFormat(_ENTITY_MIME):
            self._panel._on_entity_drop(event)
        elif event.mimeData().hasUrls() or event.mimeData().hasText():
            self._panel._on_detail_tree_drop(event)
        else:
            super().dropEvent(event)

    def startDrag(self, supportedActions):
        self._panel._start_drag_detail(supportedActions)

class FolderTreeWidget(QTreeWidget):
    def __init__(self, panel: ProjectPanel, parent=None):
        super().__init__(parent)
        self._panel = panel
        self.setAcceptDrops(True)
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setIndentation(scale(16))
        self.setStyleSheet(f"""
            QTreeWidget {{
                border: none;
                outline: none;
                font-size: 12px;
                padding: 4px 2px;
            }}
            QTreeWidget::item {{
                padding: 3px 4px;
                border: 1px solid transparent;
                border-radius: 2px;
                min-height: 22px;
            }}
            QTreeWidget::item:selected {{
            }}
            QTreeWidget::item:hover:!selected {{
            }}
            QTreeWidget::branch {{
                background: transparent;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 8px;
            }}
            QScrollBar::handle:vertical {{
                min-height: 30px;
                border-radius: 4px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #666;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(_ENTITY_MIME) or event.mimeData().hasUrls() or event.mimeData().hasText():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(_ENTITY_MIME) or event.mimeData().hasUrls() or event.mimeData().hasText():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasFormat(_ENTITY_MIME):
            self._panel._on_entity_drop(event)
        elif event.mimeData().hasUrls() or event.mimeData().hasText():
            self._panel._on_folder_tree_drop(event)
        else:
            super().dropEvent(event)

class _AddressBreadcrumb(QLineEdit):
    def __init__(self, pane: _FilePane, parent=None):
        super().__init__(parent)
        self._pane = pane
        self._editing = False
        self.setReadOnly(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QLineEdit {{
                background: transparent;
                border: 1px solid transparent;
                border-radius: 2px;
                padding: 2px 6px;
                font-size: 12px;
                min-height: 20px;
            }}
            QLineEdit:focus {{
                padding: 2px 6px;
            }}
        """)

    def mousePressEvent(self, event):
        if not self._editing:
            self._enter_edit_mode()
        super().mousePressEvent(event)

    def focusInEvent(self, event):
        if not self._editing:
            self._enter_edit_mode()
        super().focusInEvent(event)

    def _enter_edit_mode(self):
        self._editing = True
        self.setReadOnly(False)
        self.setText(self._pane._current_dir)
        self.selectAll()
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self.setStyleSheet(f"""
            QLineEdit {{
                border-radius: 2px;
                padding: 2px 6px;
                font-size: 12px;
                min-height: 20px;
            }}
        """)

    def _exit_edit_mode(self):
        self._editing = False
        self.setReadOnly(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pane.rebuild_breadcrumb()
        self.setStyleSheet(f"""
            QLineEdit {{
                background: transparent;
                border: 1px solid transparent;
                border-radius: 2px;
                padding: 2px 6px;
                font-size: 12px;
                min-height: 20px;
            }}
            QLineEdit:focus {{
                padding: 2px 6px;
            }}
        """)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            path = self.text().strip()
            if path and os.path.isdir(path):
                self._pane.populate_files(path)
                self._pane._panel._push_history(path)
            self._exit_edit_mode()
            return
        if event.key() == Qt.Key.Key_Escape:
            self._exit_edit_mode()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        self._exit_edit_mode()
        super().focusOutEvent(event)

class _FilePane(QWidget):
    def __init__(self, panel: ProjectPanel, parent=None):
        super().__init__(parent)
        self._panel = panel
        self._current_dir = panel._project_root
        self._active = False
        self._view_mode_cache = panel._view_mode

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._breadcrumb_bar = _AddressBreadcrumb(self)
        layout.addWidget(self._breadcrumb_bar)

        self._stack = QStackedWidget()

        self._file_list = FileListWidget(panel)
        self._file_list.setDragEnabled(True)
        self._file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._file_list.setIconSize(QSize(scale(20), scale(20)))
        self._file_list.setSpacing(1)
        from PyQt6.QtWidgets import QStyledItemDelegate
        self._file_list.setItemDelegate(_FileListDelegate(panel))
        self._file_list.itemClicked.connect(panel._on_file_single_click)
        self._file_list.itemDoubleClicked.connect(panel._on_file_double_click)
        self._file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._file_list.customContextMenuRequested.connect(panel._show_file_context_menu)
        self._file_list.itemChanged.connect(panel._on_list_item_changed)
        self._stack.addWidget(self._file_list)

        self._detail_tree = FileDetailWidget(panel)
        self._detail_tree.setHeaderHidden(False)
        self._detail_tree.setColumnCount(4)
        self._detail_tree.setHeaderLabels(["Name", "Size", "Type", "Date Modified"])
        self._detail_tree.setRootIsDecorated(False)
        self._detail_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._detail_tree.setDragEnabled(True)
        self._detail_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._detail_tree.customContextMenuRequested.connect(panel._show_file_context_menu)
        self._detail_tree.itemClicked.connect(panel._on_file_single_click)
        self._detail_tree.itemDoubleClicked.connect(panel._on_file_double_click)
        self._detail_tree.itemChanged.connect(panel._on_tree_item_changed)
        self._detail_tree.setSortingEnabled(True)
        self._detail_tree.setIndentation(0)
        header = self._detail_tree.header()
        header.setStretchLastSection(True)
        header.setSectionsClickable(True)
        header.setSectionsMovable(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(0, scale(280))
        header.resizeSection(1, scale(80))
        header.resizeSection(2, scale(140))
        header.resizeSection(3, scale(120))
        self._detail_tree.setIconSize(QSize(scale(16), scale(16)))
        self._stack.addWidget(self._detail_tree)

        layout.addWidget(self._stack, 1)

    def set_active(self, active: bool):
        self._active = active
        if self._panel._dual_pane:
            self._stack.setStyleSheet(f"border: 1px solid {self.palette().color(QPalette.ColorRole.Highlight).name()};")
        else:
            self._stack.setStyleSheet("")

    def rebuild_breadcrumb(self):
        if self._breadcrumb_bar._editing:
            return
        nav_bar = self._panel._nav_bar
        if nav_bar:
            nav_bar._back_btn.setEnabled(len(self._panel._history[0]) > 1 if hasattr(self._panel, '_history') else False)
            nav_bar._forward_btn.setEnabled(len(self._panel._history[1]) > 0 if hasattr(self._panel, '_history') else False)
        display = self._current_dir.replace(self._panel._project_root, os.path.basename(self._panel._project_root))
        display = display.replace("\\", " \u25B8 ").replace("/", " \u25B8 ")
        self._breadcrumb_bar.setText(display)

    def _on_search(self, text: str):
        if text:
            self._search_all(text)
        else:
            self.populate_files(self._current_dir)

    def _search_all(self, text: str):
        self._file_list.clear()
        if self._panel._view_mode == VIEW_DETAILS:
            self._detail_tree.setSortingEnabled(False)
            self._detail_tree.clear()
        for root, dirs, files in os.walk(self._panel._project_root):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in sorted(files):
                if text.lower() in f.lower() and not f.startswith("."):
                    full = os.path.join(root, f)
                    if self._panel._view_mode == VIEW_DETAILS:
                        item = QTreeWidgetItem()
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                        try:
                            fi = QFileInfo(full)
                            icon = self._panel._icon_provider.icon(fi)
                            if icon and not icon.isNull():
                                item.setIcon(0, icon)
                        except Exception:
                            pass
                        item.setText(0, f)
                        try:
                            item.setText(1, _format_size(os.path.getsize(full)))
                        except OSError:
                            item.setText(1, "")
                        ext = os.path.splitext(f)[1].lower()
                        type_map = {
                            ".py": "Python Script", ".zpes": "Zarin Scene",
                            ".zpep": "Zarin Prefab",
                            ".mat": "Material", ".obj": "OBJ Model", ".fbx": "FBX Model",
                            ".stl": "3D Model", ".gltf": "3D Model", ".glb": "3D Model", ".usdz": "3D Model",
                            ".png": "PNG Image", ".jpg": "JPEG Image", ".jpeg": "JPEG Image",
                            ".wav": "WAV Audio", ".mp3": "MP3 Audio", ".ogg": "OGG Audio",
                            ".txt": "Text Document", ".json": "JSON File",
                            ".animclip": "Animation Clip", ".animcontroller": "Animator Controller",
                            ".zterr": "Terrain Graph",
                        }
                        item.setText(2, type_map.get(ext, f"{ext.upper()} File" if ext else "File"))
                        item.setData(0, Qt.ItemDataRole.UserRole, full)
                        self._detail_tree.addTopLevelItem(item)
                    else:
                        item = QListWidgetItem()
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                        std_icon = self._panel._get_file_icon(full)
                        if std_icon:
                            item.setIcon(std_icon)
                        else:
                            pm = _get_thumbnail(full, _thumb_resolution())
                            item.setIcon(QIcon(pm))
                        item.setText(f)
                        item.setData(Qt.ItemDataRole.UserRole, full)
                        self._file_list.addItem(item)
        if self._panel._view_mode == VIEW_DETAILS:
            self._detail_tree.setSortingEnabled(True)

    def populate_files(self, dirpath: str, filter_text: str = ""):
        self._current_dir = dirpath
        self.rebuild_breadcrumb()
        try:
            entries = sorted(os.listdir(dirpath))
        except (PermissionError, FileNotFoundError) as e:
            if isinstance(e, FileNotFoundError):
                os.makedirs(dirpath, exist_ok=True)
                entries = []
            else:
                return
        visible = [e for e in entries if not e.startswith(".") and not e.endswith(".import")]
        if filter_text:
            visible = [e for e in visible if filter_text.lower() in e.lower()]
        if self._panel._view_mode == VIEW_DETAILS:
            self._populate_detail_tree(dirpath, visible, filter_text)
        else:
            self._populate_list_view(dirpath, visible, filter_text)
        if self._panel._status_bar:
            self._panel._status_bar.update_counts(len(visible))
        if self._panel._nav_bar:
            self._panel._nav_bar.update_address(dirpath)
        if self._panel._status_bar:
            self._panel._status_bar.update_path(dirpath)
        self._panel._apply_vcs_colors()

    def _populate_list_view(self, dirpath: str, entries: list[str], filter_text: str):
        widget = self._file_list
        widget.blockSignals(True)
        widget.clear()
        is_icon = self._panel._view_mode == VIEW_ICON
        folders = []
        files = []
        for entry in entries:
            full = os.path.join(dirpath, entry)
            if entry.startswith("."):
                continue
            if filter_text and filter_text.lower() not in entry.lower():
                continue
            if os.path.isdir(full):
                folders.append((entry, full))
            else:
                files.append((entry, full))

        if is_icon and len(folders) > 0:
            hdr = _GroupHeader(f"Folders", len(folders))
            item = QListWidgetItem()
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            item.setSizeHint(QSize(0, hdr.height() + 4))
            item.setData(Qt.ItemDataRole.UserRole, None)
            item.setData(Qt.ItemDataRole.UserRole + 1, "group_header")
            item.setData(Qt.ItemDataRole.UserRole + 2, "Folders")
            item.setData(Qt.ItemDataRole.UserRole + 3, len(folders))
            widget.addItem(item)

        for entry, full in folders:
            item = QListWidgetItem()
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            try:
                item.setIcon(self._panel._icon_provider.icon(QFileIconProvider.IconType.Folder))
            except Exception:
                item.setIcon(QIcon())
            item.setText(entry)
            item.setData(Qt.ItemDataRole.UserRole, full)
            item.setToolTip(f"Folder: {full}")
            widget.addItem(item)

        if is_icon and len(files) > 0:
            grouped = self._group_files_by_date(files)
            for group_name, group_files in grouped:
                hdr = _GroupHeader(group_name, len(group_files))
                item = QListWidgetItem()
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                item.setSizeHint(QSize(0, hdr.height() + 4))
                item.setData(Qt.ItemDataRole.UserRole, None)
                item.setData(Qt.ItemDataRole.UserRole + 1, "group_header")
                item.setData(Qt.ItemDataRole.UserRole + 2, group_name)
                item.setData(Qt.ItemDataRole.UserRole + 3, len(group_files))
                widget.addItem(item)

                for entry, full in group_files:
                    item = QListWidgetItem()
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                    std_icon = self._panel._get_file_icon(full)
                    if std_icon:
                        item.setIcon(std_icon)
                    else:
                        pm = _get_thumbnail(full, _thumb_resolution())
                        item.setIcon(QIcon(pm))
                    item.setText(entry)
                    item.setData(Qt.ItemDataRole.UserRole, full)
                    try:
                        sz = os.path.getsize(full)
                    except OSError:
                        sz = 0
                    item.setToolTip(f"{entry}\n{_format_size(sz)}")
                    widget.addItem(item)
        else:
            for entry, full in files:
                item = QListWidgetItem()
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                if is_icon:
                    std_icon = self._panel._get_file_icon(full)
                    if std_icon:
                        item.setIcon(std_icon)
                    else:
                        pm = _get_thumbnail(full, _thumb_resolution())
                        item.setIcon(QIcon(pm))
                else:
                    try:
                        fi = QFileInfo(full)
                        item.setIcon(self._panel._icon_provider.icon(fi))
                    except Exception:
                        item.setIcon(QIcon())
                item.setText(entry)
                item.setData(Qt.ItemDataRole.UserRole, full)
                try:
                    sz = os.path.getsize(full)
                except OSError:
                    sz = 0
                item.setToolTip(f"{entry}\n{_format_size(sz)}")
                widget.addItem(item)

        widget.blockSignals(False)

    def _group_files_by_date(self, files: list[tuple]) -> list[tuple[str, list]]:
        now = datetime.datetime.now()
        today = now.date()
        yesterday = today - datetime.timedelta(days=1)

        groups = {}
        for entry, full in files:
            try:
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(full)).date()
            except OSError:
                mtime = today
            if mtime == today:
                group = "Today"
            elif mtime == yesterday:
                group = "Yesterday"
            elif (today - mtime).days <= 7:
                group = "Last week"
            elif mtime.month == now.month and mtime.year == now.year:
                group = "Earlier this month"
            elif mtime.year == now.year:
                group = mtime.strftime("%B %Y")
            else:
                group = str(mtime.year)
            if group not in groups:
                groups[group] = []
            groups[group].append((entry, full))

        order = ["Today", "Yesterday", "Last week", "Earlier this month"]
        result = []
        for g in order:
            if g in groups:
                result.append((g, groups.pop(g)))
        for g in sorted(groups.keys()):
            result.append((g, groups[g]))
        return result

    def _populate_detail_tree(self, dirpath: str, entries: list[str], filter_text: str):
        widget = self._detail_tree
        widget.setSortingEnabled(False)
        widget.blockSignals(True)
        widget.clear()
        use_thumbs = self._panel._thumb_size >= 20
        for entry in entries:
            full = os.path.join(dirpath, entry)
            if entry.startswith("."):
                continue
            if filter_text and filter_text.lower() not in entry.lower():
                continue
            item = QTreeWidgetItem()
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            if os.path.isdir(full):
                try:
                    item.setIcon(0, self._panel._icon_provider.icon(QFileIconProvider.IconType.Folder))
                except Exception:
                    pass
                item.setText(0, entry)
                item.setText(1, "")
                item.setText(2, "File folder")
                item.setToolTip(0, f"Folder: {full}")
            else:
                if use_thumbs:
                    pm = _get_thumbnail(full, _thumb_resolution())
                    if pm and not pm.isNull():
                        item.setIcon(0, QIcon(pm))
                    else:
                        try:
                            fi = QFileInfo(full)
                            icon = self._panel._icon_provider.icon(fi)
                            if icon and not icon.isNull():
                                item.setIcon(0, icon)
                        except Exception:
                            pass
                else:
                    try:
                        fi = QFileInfo(full)
                        icon = self._panel._icon_provider.icon(fi)
                        if icon and not icon.isNull():
                            item.setIcon(0, icon)
                    except Exception:
                        pass
                item.setText(0, entry)
                try:
                    sz = os.path.getsize(full)
                    item.setText(1, _format_size(sz))
                except OSError:
                    item.setText(1, "")
                ext = os.path.splitext(entry)[1].lower()
                type_map = {
                    ".py": "Python Script", ".zpes": "Zarin Scene",
                    ".zpep": "Zarin Prefab",
                    ".mat": "Material", ".obj": "OBJ Model", ".fbx": "FBX Model",
                    ".stl": "3D Model", ".gltf": "3D Model", ".glb": "3D Model", ".usdz": "3D Model",
                    ".png": "PNG Image", ".jpg": "JPEG Image", ".jpeg": "JPEG Image",
                    ".wav": "WAV Audio", ".mp3": "MP3 Audio", ".ogg": "OGG Audio",
                    ".txt": "Text Document", ".json": "JSON File",
                    ".xml": "XML File", ".csv": "CSV File",
                    ".toml": "TOML File", ".yaml": "YAML File", ".yml": "YAML File",
                    ".ini": "INI File", ".cfg": "Configuration",
                    ".vert": "Vertex Shader", ".frag": "Fragment Shader",
                    ".animclip": "Animation Clip", ".animcontroller": "Animator Controller",
                    ".zterr": "Terrain Graph",
                }
                item.setText(2, type_map.get(ext, f"{ext.upper()} File" if ext else "File"))
                try:
                    mtime = os.path.getmtime(full)
                    dt = datetime.datetime.fromtimestamp(mtime)
                    item.setText(3, dt.strftime("%d.%m.%Y %H:%M"))
                except OSError:
                    item.setText(3, "")
            item.setData(0, Qt.ItemDataRole.UserRole, full)
            widget.addTopLevelItem(item)
        widget.blockSignals(False)
        widget.setSortingEnabled(True)

    def active_widget(self):
        return self._detail_tree if self._panel._view_mode == VIEW_DETAILS else self._file_list

    def refresh(self):
        self.populate_files(self._current_dir)

class _NavBar(QWidget):
    def __init__(self, panel: ProjectPanel, parent=None):
        super().__init__(parent)
        self._panel = panel
        self.setFixedHeight(scale(26))
        self.setStyleSheet(f"""
            QWidget {{
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(scale(4), scale(1), scale(4), scale(1))
        layout.setSpacing(scale(1))

        self._back_btn = _NavButton("fa5s.arrow-left", "Back (Alt+Left)")
        self._back_btn.setEnabled(False)
        self._back_btn.clicked.connect(self._panel._go_back)
        layout.addWidget(self._back_btn)

        self._forward_btn = _NavButton("fa5s.arrow-right", "Forward (Alt+Right)")
        self._forward_btn.setEnabled(False)
        self._forward_btn.clicked.connect(self._panel._go_forward)
        layout.addWidget(self._forward_btn)

        self._up_btn = _NavButton("fa5s.arrow-up", "Up (Alt+Up)")
        self._up_btn.clicked.connect(self._panel._go_to_parent)
        layout.addWidget(self._up_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedWidth(1)
        layout.addWidget(sep)

        self._address_bar = QLineEdit()
        self._address_bar.setReadOnly(True)
        self._address_bar.setStyleSheet(f"""
            QLineEdit {{
                border-radius: 2px;
                padding: 2px 8px;
                font-size: 11px;
                min-height: 18px;
            }}
            QLineEdit:focus {{
            }}
        """)
        layout.addWidget(self._address_bar, 1)

        self._search_bar = QLineEdit()
        self._search_bar.setPlaceholderText(" Search...")
        self._search_bar.setClearButtonEnabled(True)
        self._search_bar.setMaximumWidth(scale(120))
        self._search_bar.textChanged.connect(self._on_search)
        self._search_bar.setStyleSheet(f"""
            QLineEdit {{
                border-radius: 2px;
                padding: 2px 24px 2px 8px;
                font-size: 11px;
                min-height: 18px;
            }}
            QLineEdit:focus {{
            }}
            QLineEdit::placeholder {{
            }}
        """)
        layout.addWidget(self._search_bar)

    def _on_search(self, text):
        self._panel._active_pane()._on_search(text)

    def update_address(self, path: str):
        display = path.replace(self._panel._project_root, os.path.basename(self._panel._project_root))
        display = display.replace("\\", " \u25B8 ").replace("/", " \u25B8 ")
        self._address_bar.setText(display)

class _StatusBar(QWidget):
    def __init__(self, panel: ProjectPanel, parent=None):
        super().__init__(parent)
        self._panel = panel
        self.setFixedHeight(scale(18))
        self.setStyleSheet(f"""
            QWidget {{
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(scale(8), 0, scale(8), 0)
        layout.setSpacing(scale(12))

        self._count_label = QLabel("0 items")
        layout.addWidget(self._count_label)

        sep = QLabel("|")
        layout.addWidget(sep)

        self._selected_label = QLabel("")
        layout.addWidget(self._selected_label)

        layout.addStretch()

        self._path_label = QLabel("")
        layout.addWidget(self._path_label)

    def update_counts(self, total: int, selected: int = 0):
        self._count_label.setText(f"{total} item{'s' if total != 1 else ''}")
        if selected > 0:
            self._selected_label.setText(f"{selected} selected")
        else:
            self._selected_label.setText("")

    def update_path(self, path: str):
        short = path
        if len(short) > 60:
            short = "..." + short[-57:]
        self._path_label.setText(short)

class _FileListDelegate(QStyledItemDelegate):
    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self._panel = panel

    def paint(self, painter, option, index):
        group_type = index.data(Qt.ItemDataRole.UserRole + 1)
        if group_type == "group_header":
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            rect = option.rect
            painter.setPen(QPen(QColor(80, 80, 80), 1))
            painter.drawLine(rect.x(), rect.y(), rect.x() + rect.width(), rect.y())
            painter.drawLine(rect.x(), rect.y() + rect.height() - 1, rect.x() + rect.width(), rect.y() + rect.height() - 1)
            font = painter.font()
            font.setPointSize(9)
            font.setBold(False)
            painter.setFont(font)
            text = index.data(Qt.ItemDataRole.UserRole + 2) or ""
            count = index.data(Qt.ItemDataRole.UserRole + 3) or 0
            fm = QFontMetrics(font)
            x = rect.x() + scale(8)
            y = rect.y() + (rect.height() + fm.ascent() - fm.descent()) // 2
            painter.drawText(x, y, text)
            if count > 0:
                count_text = f" ({count})"
                text_w = fm.horizontalAdvance(text)
                painter.drawText(x + text_w + scale(2), y, count_text)
            painter.restore()
        else:
            super().paint(painter, option, index)

    def sizeHint(self, option, index):
        group_type = index.data(Qt.ItemDataRole.UserRole + 1)
        if group_type == "group_header":
            return QSize(0, scale(24))
        return super().sizeHint(option, index)

class ProjectPanel(QDockWidget):
    file_double_clicked = pyqtSignal(str)
    prefab_drag_started = pyqtSignal(str)
    import_model_requested = pyqtSignal(str)
    file_selected = pyqtSignal(str)

    def __init__(self, engine: Engine, project_root: str = "assets", parent=None):
        super().__init__("Project", parent)
        self._engine = engine
        self._project_root = os.path.abspath(project_root)
        self._thumb_size = 64
        self._icon_provider = QFileIconProvider()
        self._view_mode = VIEW_DETAILS
        self._dual_pane = False
        self._history = [[], []]
        self._history_index = 0
        self._nav_bar = None
        self._status_bar = None
        self._vcs_status: dict[str, str] = {}
        self._vcs_timer = QTimer(self)
        self._vcs_timer.timeout.connect(self._refresh_vcs_status)
        self._vcs_timer.start(3000)
        self._refresh_vcs_status()
        self._setup_ui()
        self._populate_tree()
        self._push_history(self._project_root)

    def load_config(self, config) -> None:
        thumb_size = config.get("project.thumb_size", self._thumb_size)
        self._thumb_size = max(MIN_THUMB, min(thumb_size, MAX_THUMB))
        self._dual_pane = config.get("project.dual_pane", False)
        if self._dual_pane != self._dual_pane_btn.isChecked():
            self._dual_pane_btn.setChecked(self._dual_pane)

    def save_config(self, config) -> None:
        config.set("project.thumb_size", self._thumb_size)
        config.set("project.dual_pane", self._dual_pane)

    def _get_file_icon(self, path: str) -> QIcon:
        ext = os.path.splitext(path)[1].lower()
        ext_map = {
            ".py": "fa5s.file-code", ".txt": "fa5s.file-alt", ".json": "fa5s.file-code",
            ".xml": "fa5s.file-code", ".csv": "fa5s.file-csv", ".ini": "fa5s.file-code",
            ".cfg": "fa5s.file-code", ".toml": "fa5s.file-code", ".yaml": "fa5s.file-code",
            ".yml": "fa5s.file-code",
        }
        if ext in ext_map:
            return _qta_icon(ext_map[ext], "#d4d4d4")
        if ext == ".zpes":
            from editor.resource_picker import _draw_scene_icon
            return QIcon(_draw_scene_icon(self._thumb_size))
        if ext == ".zpep":
            from editor.resource_picker import _draw_prefab_icon
            return QIcon(_draw_prefab_icon(self._thumb_size))
        if ext == ".mat":
            from editor.resource_picker import _get_material_thumbnail, _draw_material_icon
            pm = _get_material_thumbnail(path, self._thumb_size)
            if pm:
                return QIcon(pm)
            return QIcon(_draw_material_icon(self._thumb_size))
        if ext in (".animclip", ".animcontroller"):
            from editor.resource_picker import _draw_file_icon
            return QIcon(_draw_file_icon(self._thumb_size))
        return None

    def _active_pane(self) -> _FilePane:
        if self._dual_pane:
            if self._pane_b and self._pane_b._active:
                return self._pane_b
        return self._pane_a

    def _push_history(self, path: str):
        if self._history_index < len(self._history[0]):
            self._history[0] = self._history[0][:self._history_index + 1]
        if self._history[0] and self._history[0][-1] == path:
            return
        self._history[0].append(path)
        self._history[1].clear()
        self._update_nav_buttons()

    def _go_back(self):
        if len(self._history[0]) > 1:
            self._history[1].append(self._history[0].pop())
            self._history_index = len(self._history[0]) - 1
            path = self._history[0][-1]
            self._active_pane().populate_files(path)
            self._update_nav_buttons()

    def _go_forward(self):
        if self._history[1]:
            path = self._history[1].pop()
            self._history[0].append(path)
            self._history_index = len(self._history[0]) - 1
            self._active_pane().populate_files(path)
            self._update_nav_buttons()

    def _update_nav_buttons(self):
        if self._nav_bar:
            self._nav_bar._back_btn.setEnabled(len(self._history[0]) > 1)
            self._nav_bar._forward_btn.setEnabled(len(self._history[1]) > 0)

    def _setup_ui(self):
        w = QWidget()
        main_layout = QVBoxLayout(w)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        toolbar = QWidget()
        toolbar.setFixedHeight(scale(24))
        toolbar.setStyleSheet(f"""
            QWidget {{
            }}
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(scale(4), scale(1), scale(4), scale(1))
        toolbar_layout.setSpacing(scale(2))

        create_btn = QToolButton()
        if qta is not None:
            create_btn.setIcon(qta.icon("fa5s.plus", color="#9ccc65"))
            create_btn.setIconSize(QSize(*scale_xy(12, 12)))
        create_btn.setText(" New")
        create_btn.setToolTip("Create new asset")
        create_btn.setFixedHeight(scale(20))
        create_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        create_btn.setStyleSheet(f"""
            QToolButton {{
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                font-size: 10px;
                padding: 0 6px;
            }}
            QToolButton:hover {{
            }}
            QToolButton:pressed {{
                background: #444;
            }}
        """)
        create_menu = QMenu(self)
        for name, cb in [
            ("Folder", self._create_new_folder),
            ("Scene", self._create_new_scene),
            ("Material", self._create_new_material),
            ("Animation Clip", self._create_new_animclip),
            ("Animator Controller", self._create_new_animcontroller),
            ("Python Script", self._create_new_script),
        ]:
            a = QAction(name, self)
            a.triggered.connect(cb)
            create_menu.addAction(a)
        create_btn.setMenu(create_menu)
        toolbar_layout.addWidget(create_btn)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setFixedWidth(1)
        toolbar_layout.addWidget(sep1)

        self._view_mode_btn = QToolButton()
        self._view_mode_btn.setFixedSize(scale(22), scale(22))
        self._view_mode_btn.setToolTip("View Mode")
        self._view_mode_btn.clicked.connect(self._toggle_view_mode)
        self._update_view_mode_icon()
        self._view_mode_btn.setStyleSheet(f"""
            QToolButton {{
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                font-size: 13px;
            }}
            QToolButton:hover {{
            }}
        """)
        toolbar_layout.addWidget(self._view_mode_btn)

        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setFixedWidth(scale(50))
        self._zoom_slider.setRange(MIN_THUMB, MAX_THUMB)
        self._zoom_slider.setValue(self._thumb_size)
        self._zoom_slider.valueChanged.connect(self._on_zoom_changed)
        toolbar_layout.addWidget(self._zoom_slider)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setFixedWidth(1)
        toolbar_layout.addWidget(sep2)

        self._dual_pane_btn = QToolButton()
        self._dual_pane_btn.setFixedSize(scale(22), scale(22))
        self._dual_pane_btn.setToolTip("Dual Pane")
        self._dual_pane_btn.setCheckable(True)
        self._dual_pane_btn.setChecked(False)
        if qta is not None:
            self._dual_pane_btn.setIcon(qta.icon("fa5s.columns", color="#d4d4d4"))
            self._dual_pane_btn.setIconSize(QSize(*scale_xy(14, 14)))
        self._dual_pane_btn.toggled.connect(self._toggle_dual_pane)
        self._dual_pane_btn.setStyleSheet(f"""
            QToolButton {{
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                font-size: 11px;
            }}
            QToolButton:hover {{
            }}
            QToolButton:checked {{
            }}
        """)
        toolbar_layout.addWidget(self._dual_pane_btn)

        refresh_btn = QToolButton()
        refresh_btn.setFixedSize(scale(22), scale(22))
        refresh_btn.setToolTip("Refresh (F5)")
        if qta is not None:
            refresh_btn.setIcon(qta.icon("fa5s.sync-alt", color="#d4d4d4"))
            refresh_btn.setIconSize(QSize(*scale_xy(14, 14)))
        refresh_btn.clicked.connect(self._refresh)
        refresh_btn.setStyleSheet(f"""
            QToolButton {{
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                font-size: 14px;
            }}
            QToolButton:hover {{
            }}
        """)
        toolbar_layout.addWidget(refresh_btn)

        toolbar_layout.addStretch()
        main_layout.addWidget(toolbar)

        self._nav_bar = _NavBar(self)
        main_layout.addWidget(self._nav_bar)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self._folder_tree = FolderTreeWidget(self)
        self._folder_tree.setMinimumWidth(scale(100))
        self._folder_tree.setMaximumWidth(scale(240))
        self._folder_tree.itemClicked.connect(self._on_folder_selected)
        self._folder_tree.setDragEnabled(True)
        content_layout.addWidget(self._folder_tree)

        tree_separator = QFrame()
        tree_separator.setFrameShape(QFrame.Shape.VLine)
        tree_separator.setFixedWidth(1)
        content_layout.addWidget(tree_separator)

        self._pane_a = _FilePane(self)
        self._pane_a.set_active(True)
        self._install_pane_focus(self._pane_a)

        self._pane_b = None
        self._pane_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._pane_splitter.setHandleWidth(1)
        self._pane_splitter.setStyleSheet(f"""
            QSplitter::handle {{
            }}
            QSplitter::handle:horizontal {{
                width: 1px;
            }}
        """)
        self._pane_splitter.addWidget(self._pane_a)

        self._setup_shortcuts()
        self._apply_view_mode()
        content_layout.addWidget(self._pane_splitter, 1)
        main_layout.addWidget(content, 1)

        self._status_bar = _StatusBar(self)
        main_layout.addWidget(self._status_bar)

        self.setWidget(w)

        from editor.resource_picker import _get_thumb_service
        from editor.thumb_cache import cache_root_for_project
        _get_thumb_service().set_cache_dir(cache_root_for_project(self._project_root))
        self._connect_mesh_loader()
        self._pane_a.populate_files(self._project_root)

    def _refresh_vcs_status(self):
        self._vcs_status = _parse_vcs_status(self._project_root)
        if hasattr(self, "_pane_a"):
            self._apply_vcs_colors()

    def _vcs_status_of(self, full_path: str) -> str:
        if not self._vcs_status or not full_path:
            return ""
        try:
            rel = os.path.relpath(full_path, self._project_root)
        except ValueError:
            return ""
        rel = rel.replace("\\", "/")
        return self._vcs_status.get(rel, "")

    def _apply_vcs_colors(self):
        if not self._vcs_status:
            return
        default = QBrush()
        for pane in (self._pane_a, self._pane_b):
            if not pane:
                continue
            for i in range(pane._file_list.count()):
                item = pane._file_list.item(i)
                path = item.data(Qt.ItemDataRole.UserRole)
                if path and os.path.isfile(path):
                    status = self._vcs_status_of(path)
                    item.setForeground(_VCS_COLORS.get(status, default))
            for i in range(pane._detail_tree.topLevelItemCount()):
                item = pane._detail_tree.topLevelItem(i)
                path = item.data(0, Qt.ItemDataRole.UserRole)
                if path and os.path.isfile(path):
                    status = self._vcs_status_of(path)
                    item.setForeground(0, _VCS_COLORS.get(status, default))

    def _git_run(self, *args: str) -> tuple[int, str, str]:
        for candidate in ["git.exe", "git"]:
            try:
                r = subprocess.run([candidate, "--version"], capture_output=True,
                                   text=True, timeout=5)
                if r.returncode == 0:
                    break
            except FileNotFoundError:
                continue
        else:
            return -1, "", "git not found"
        try:
            r = subprocess.run([candidate] + list(args), capture_output=True,
                               text=True, cwd=self._project_root, timeout=30)
            return r.returncode, r.stdout, r.stderr
        except Exception as e:
            return -1, "", str(e)

    def _git_add_file(self, path: str):
        if not isinstance(path, str) or not path:
            return
        try:
            rel = os.path.relpath(path, self._project_root)
        except ValueError:
            return
        rc, _, err = self._git_run("add", "--", rel)
        if rc == 0:
            self._refresh_vcs_status()
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Git Add Failed", err[:200])

    def _git_unstage_file(self, path: str):
        if not isinstance(path, str) or not path:
            return
        try:
            rel = os.path.relpath(path, self._project_root)
        except ValueError:
            return
        rc, _, err = self._git_run("restore", "--staged", "--", rel)
        if rc == 0:
            self._refresh_vcs_status()
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Git Unstage Failed", err[:200])

    def _git_ignore_file(self, path: str):
        if not isinstance(path, str) or not path:
            return
        try:
            rel = os.path.relpath(path, self._project_root)
        except ValueError:
            return
        gi_path = os.path.join(self._project_root, ".gitignore")
        try:
            with open(gi_path, "a", encoding="utf-8") as f:
                f.write(f"\n/{rel}\n")
            self._refresh_vcs_status()
        except OSError as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Could not update .gitignore:\n{e}")

    def _connect_mesh_loader(self):
        from editor.resource_picker import _get_thumb_service
        loader = _get_thumb_service()
        loader.thumbnail_ready.connect(self._on_mesh_data_ready, Qt.ConnectionType.QueuedConnection)

    def _on_mesh_data_ready(self, path: str, size: int, pixmap):
        if pixmap is None or pixmap.isNull():
            return
        icon = QIcon(pixmap)
        for pane in (self._pane_a, self._pane_b):
            if not pane:
                continue
            if self._view_mode == VIEW_DETAILS:
                tree = pane._detail_tree
                for i in range(tree.topLevelItemCount()):
                    item = tree.topLevelItem(i)
                    item_path = item.data(0, Qt.ItemDataRole.UserRole)
                    if item_path == path:
                        item.setIcon(0, icon)
            else:
                fl = pane._file_list
                for i in range(fl.count()):
                    item = fl.item(i)
                    item_path = item.data(Qt.ItemDataRole.UserRole)
                    if item_path == path:
                        item.setIcon(icon)

    def _toggle_dual_pane(self, checked):
        self._dual_pane = checked
        if checked:
            if not self._pane_b:
                self._pane_b = _FilePane(self)
                self._pane_splitter.addWidget(self._pane_b)
                self._pane_splitter.setSizes([self._pane_splitter.width() // 2] * 2)
                self._pane_b.populate_files(self._pane_a._current_dir)
                self._install_pane_focus(self._pane_b)
                self._apply_view_mode()
            self._pane_a.set_active(not self._pane_b._active)
            self._pane_b.set_active(not self._pane_a._active)
        else:
            if self._pane_b:
                self._pane_b.setParent(None)
                self._pane_b.deleteLater()
                self._pane_b = None
            self._pane_a.set_active(True)

    def _install_pane_focus(self, pane):
        pane._file_list.installEventFilter(self)
        pane._detail_tree.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.FocusIn:
            for pane in (self._pane_a, self._pane_b):
                if pane and (pane._file_list is obj or pane._detail_tree is obj):
                    pane.set_active(True)
                    other = self._pane_b if pane is self._pane_a else self._pane_a
                    if other:
                        other.set_active(False)
                    if self._nav_bar:
                        self._nav_bar.update_address(pane._current_dir)
                    if self._status_bar:
                        self._status_bar.update_path(pane._current_dir)
                    break
        return super().eventFilter(obj, event)

    def _setup_shortcuts(self):
        sc = Qt.ShortcutContext.WidgetWithChildrenShortcut
        mapping = {
            "Ctrl+C": self._copy_selected,
            "Ctrl+X": self._cut_selected,
            "Ctrl+V": self._paste_clipboard,
            "Ctrl+D": self._duplicate_selected,
            "Ctrl+A": lambda: self._active_widget().selectAll(),
            "Ctrl+Shift+N": self._create_new_folder,
            "Alt+Left": self._go_back,
            "Alt+Right": self._go_forward,
            "Alt+Up": self._go_to_parent,
        }
        for seq_str, cb in mapping.items():
            s = QShortcut(QKeySequence(seq_str), self)
            s.activated.connect(cb)
            s.setContext(sc)

    def _active_widget(self):
        return self._active_pane().active_widget()

    def _update_view_mode_icon(self):
        icon_map = {
            VIEW_ICON: ("fa5s.th", "Icon View"),
            VIEW_DETAILS: ("fa5s.list", "Details View"),
        }
        icon_name, tip = icon_map.get(self._view_mode, ("fa5s.th", "View Mode"))
        if qta is not None:
            self._view_mode_btn.setIcon(qta.icon(icon_name, color="#d4d4d4"))
            self._view_mode_btn.setIconSize(QSize(*scale_xy(14, 14)))
            self._view_mode_btn.setText("")
        self._view_mode_btn.setToolTip(tip)

    def _toggle_view_mode(self):
        if self._view_mode == VIEW_ICON:
            self._view_mode = VIEW_DETAILS
        else:
            self._view_mode = VIEW_ICON
        self._apply_view_mode()
        self._update_view_mode_icon()
        self._pane_a.refresh()
        if self._pane_b:
            self._pane_b.refresh()

    def _apply_view_mode(self):
        is_detail = self._view_mode == VIEW_DETAILS
        for pane in (self._pane_a, self._pane_b):
            if not pane:
                continue
            pane._stack.setCurrentIndex(1 if is_detail else 0)
            fl = pane._file_list
            if not is_detail:
                fl.setViewMode(QListWidget.ViewMode.IconMode)
                fl.setIconSize(QSize(self._thumb_size, self._thumb_size))
                fl.setGridSize(QSize(self._thumb_size + scale(20), self._thumb_size + scale(36)))
                fl.setWordWrap(True)
                fl.setSpacing(2)
                fl.setUniformItemSizes(True)
                fl.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._zoom_slider.setVisible(True)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta != 0:
                step = 8 if abs(delta) > 120 else 4
                new_val = self._thumb_size + (step if delta > 0 else -step)
                new_val = max(MIN_THUMB, min(MAX_THUMB, new_val))
                self._thumb_size = new_val
                self._zoom_slider.setValue(new_val)
            event.accept()
        else:
            super().wheelEvent(event)

    def _on_zoom_changed(self, val: int):
        self._thumb_size = val
        if self._view_mode == VIEW_ICON:
            for pane in (self._pane_a, self._pane_b):
                if not pane:
                    continue
                pane._file_list.setIconSize(QSize(val, val))
                pane._file_list.setGridSize(QSize(val + scale(20), val + scale(36)))
        else:
            pt = max(7, min(20, int(val / 6)))
            icon_sz = max(16, min(64, int(val * 0.5)))
            for pane in (self._pane_a, self._pane_b):
                if not pane:
                    continue
                pane._detail_tree.setIconSize(QSize(icon_sz, icon_sz))

    def _open_path(self, path: str):
        pane = self._active_pane()
        if os.path.isdir(path):
            pane.populate_files(path)
            self._push_history(path)
            if self._nav_bar:
                self._nav_bar.update_address(path)
            if self._status_bar:
                self._status_bar.update_path(path)
        else:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".zpes":
                self._engine.load_scene(path)
            elif ext == ".zpep":
                self.file_double_clicked.emit(path)
            elif ext in (".obj", ".fbx", ".stl", ".usdz", ".gltf", ".glb"):
                self.import_model_requested.emit(path)
            elif ext in (".animclip", ".animcontroller"):
                self.file_double_clicked.emit(path)
            else:
                self.file_double_clicked.emit(path)

    def _open_item_by_path(self, path: str):
        if path:
            self._open_path(path)

    def _go_to_parent(self):
        pane = self._active_pane()
        parent = os.path.dirname(pane._current_dir)
        if parent and os.path.commonpath([parent, self._project_root]) == self._project_root:
            pane.populate_files(parent)
            self._push_history(parent)
            if self._nav_bar:
                self._nav_bar.update_address(parent)
            if self._status_bar:
                self._status_bar.update_path(parent)

    def _go_to_root(self):
        self._active_pane().populate_files(self._project_root)
        self._push_history(self._project_root)
        if self._nav_bar:
            self._nav_bar.update_address(self._project_root)
        if self._status_bar:
            self._status_bar.update_path(self._project_root)

    def _populate_tree(self):
        self._folder_tree.clear()
        root_item = QTreeWidgetItem(self._folder_tree)
        root_item.setText(0, os.path.basename(self._project_root))
        root_item.setData(0, Qt.ItemDataRole.UserRole, self._project_root)
        root_item.setIcon(0, self._icon_provider.icon(QFileIconProvider.IconType.Folder))
        self._add_subfolders(root_item, self._project_root)
        root_item.setExpanded(True)

    def _add_subfolders(self, parent_item: QTreeWidgetItem, dirpath: str):
        try:
            entries = sorted(os.listdir(dirpath))
        except PermissionError:
            return
        for entry in entries:
            full = os.path.join(dirpath, entry)
            if os.path.isdir(full) and not entry.startswith(".") and entry != "__pycache__":
                item = QTreeWidgetItem(parent_item)
                item.setText(0, entry)
                item.setData(0, Qt.ItemDataRole.UserRole, full)
                item.setIcon(0, self._icon_provider.icon(QFileIconProvider.IconType.Folder))
                status = self._vcs_status_of(full)
                c = _VCS_COLORS.get(status)
                if c:
                    item.setForeground(0, c)
                self._add_subfolders(item, full)

    def _on_list_item_changed(self, item: QListWidgetItem):
        old_path = item.data(Qt.ItemDataRole.UserRole)
        if not old_path or not os.path.exists(old_path):
            return
        new_name = item.text()
        if not new_name:
            return
        old_name = os.path.basename(old_path)
        if new_name == old_name:
            return
        new_path = os.path.join(os.path.dirname(old_path), new_name)
        try:
            os.rename(old_path, new_path)
            item.setData(Qt.ItemDataRole.UserRole, new_path)
            old_tip = item.toolTip()
            if "\n" in old_tip:
                item.setToolTip(new_name + "\n" + old_tip.split("\n", 1)[1])
            else:
                item.setToolTip(new_name)
        except OSError as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Could not rename:\n{e}")
            item.blockSignals(True)
            item.setText(old_name)
            item.blockSignals(False)

    def _on_tree_item_changed(self, item: QTreeWidgetItem, col: int):
        if col != 0:
            return
        old_path = item.data(0, Qt.ItemDataRole.UserRole)
        if not old_path or not os.path.exists(old_path):
            return
        new_name = item.text(0)
        if not new_name:
            return
        old_name = os.path.basename(old_path)
        if new_name == old_name:
            return
        new_path = os.path.join(os.path.dirname(old_path), new_name)
        try:
            os.rename(old_path, new_path)
            item.setData(0, Qt.ItemDataRole.UserRole, new_path)
            if not os.path.isdir(new_path):
                ext = os.path.splitext(new_name)[1].lower()
                if ext:
                    type_map = {
                        ".py": "Python Script", ".zpes": "Zarin Scene",
                        ".zpep": "Zarin Prefab",
                        ".mat": "Material", ".obj": "OBJ Model", ".fbx": "FBX Model",
                        ".stl": "3D Model", ".gltf": "3D Model", ".glb": "3D Model", ".usdz": "3D Model",
                        ".png": "PNG Image", ".jpg": "JPEG Image", ".jpeg": "JPEG Image",
                        ".wav": "WAV Audio", ".mp3": "MP3 Audio", ".ogg": "OGG Audio",
                        ".txt": "Text Document", ".json": "JSON File",
                        ".animclip": "Animation Clip", ".animcontroller": "Animator Controller",
                        ".zterr": "Terrain Graph",
                    }
                    item.setText(1, type_map.get(ext, "Renaming..."))
                else:
                    item.setText(1, "")
            path = item.data(Qt.ItemDataRole.UserRole)
            if path and os.path.isfile(path):
                self.file_selected.emit(path)
        except OSError:
            pass

    def _on_file_single_click(self, item):
        if isinstance(item, QTreeWidgetItem):
            path = item.data(0, Qt.ItemDataRole.UserRole)
        elif hasattr(item, 'data'):
            path = item.data(Qt.ItemDataRole.UserRole)
        else:
            path = None
        if path and os.path.isfile(path):
            self.file_selected.emit(path)

    def _on_file_double_click(self, item):
        if isinstance(item, QTreeWidgetItem):
            path = item.data(0, Qt.ItemDataRole.UserRole)
        else:
            path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        if os.path.isdir(path):
            pane = self._active_pane()
            pane.populate_files(path)
            self._push_history(path)
            self._sync_tree_selection(path)
            if self._nav_bar:
                self._nav_bar.update_address(path)
            if self._status_bar:
                self._status_bar.update_path(path)
        else:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".zpes":
                self._engine.load_scene(path)
            elif ext == ".zpep":
                self.file_double_clicked.emit(path)
            elif ext in (".obj", ".fbx", ".stl", ".usdz", ".gltf", ".glb"):
                self.import_model_requested.emit(path)
            elif ext in (".animclip", ".animcontroller"):
                self.file_double_clicked.emit(path)
            elif ext == ".zterr":
                self.file_double_clicked.emit(path)
            else:
                self._open_path_with_default_app(path)

    def _sync_tree_selection(self, dirpath: str):
        root = self._folder_tree.topLevelItem(0)
        if not root:
            return
        stack = [(root, root.data(0, Qt.ItemDataRole.UserRole))]
        while stack:
            item, path = stack.pop()
            if path == dirpath:
                self._folder_tree.setCurrentItem(item)
                self._folder_tree.scrollToItem(item)
                return
            for i in range(item.childCount()):
                child = item.child(i)
                stack.append((child, child.data(0, Qt.ItemDataRole.UserRole)))

    def _refresh(self):
        self._refresh_vcs_status()
        self._populate_tree()
        self._pane_a.refresh()
        if self._pane_b:
            self._pane_b.refresh()

    def _force_refresh_thumbnails(self):
        from editor.resource_picker import _get_thumb_service
        _get_thumb_service().invalidate_thumbnails()
        from editor.thumb_cache import cache_root_for_project
        _get_thumb_service().set_cache_dir(cache_root_for_project(self._project_root))
        self._refresh()

    def _on_folder_selected(self, item, col):
        dirpath = item.data(0, Qt.ItemDataRole.UserRole)
        if dirpath and os.path.isdir(dirpath):
            self._active_pane().populate_files(dirpath)
            self._push_history(dirpath)
            if self._nav_bar:
                self._nav_bar.update_address(dirpath)
            if self._status_bar:
                self._status_bar.update_path(dirpath)

    def _show_file_context_menu(self, pos):
        widget = self.sender()
        item = widget.itemAt(pos) if isinstance(widget, (QTreeWidget, QListWidget)) else None
        menu = QMenu(self)
        if item:
            path = (item.data(0, Qt.ItemDataRole.UserRole) if isinstance(item, QTreeWidgetItem)
                    else item.data(Qt.ItemDataRole.UserRole))
            if not path:
                pass
            elif os.path.isdir(path):
                open_act = QAction("Open", self)
                open_act.triggered.connect(lambda p=path: self._active_pane().populate_files(p))
                menu.addAction(open_act)
            else:
                ext = os.path.splitext(path)[1].lower() if path else ""
                if ext == ".zpes":
                    act = QAction("Open Scene", self)
                    act.triggered.connect(lambda: self._engine.load_scene(path))
                    menu.addAction(act)
                elif ext == ".zpep":
                    act = QAction("Instantiate Prefab", self)
                    act.triggered.connect(lambda: self._instantiate_prefab(path))
                    menu.addAction(act)
                elif ext in (".obj", ".fbx", ".stl", ".usdz", ".gltf", ".glb"):
                    act = QAction("Add to Scene", self)
                    act.triggered.connect(lambda p=path: self.import_model_requested.emit(p))
                    menu.addAction(act)
                elif ext in (".py",):
                    act = QAction("Run Script", self)
                    act.triggered.connect(lambda: self.file_double_clicked.emit(path))
                    menu.addAction(act)
                elif ext in (".wav", ".mp3", ".ogg"):
                    act = QAction("Play", self)
                    act.triggered.connect(lambda: self.file_double_clicked.emit(path))
                    menu.addAction(act)
                elif ext in (".png", ".jpg", ".jpeg"):
                    act = QAction("View Image", self)
                    act.triggered.connect(lambda: self.file_double_clicked.emit(path))
                    menu.addAction(act)
                elif ext in (".animclip", ".animcontroller"):
                    act = QAction("Open", self)
                    act.triggered.connect(lambda: self.file_double_clicked.emit(path))
                    menu.addAction(act)
                elif ext == ".zterr":
                    act = QAction("Open in Terrain Editor", self)
                    act.triggered.connect(lambda: self.file_double_clicked.emit(path))
                    menu.addAction(act)
                menu.addSeparator()
                cut_act = QAction("Cut\tCtrl+X", self)
                cut_act.triggered.connect(lambda: self._cut_selected())
                menu.addAction(cut_act)
                copy_act = QAction("Copy\tCtrl+C", self)
                copy_act.triggered.connect(lambda: self._copy_selected())
                menu.addAction(copy_act)
                paste_act = QAction("Paste\tCtrl+V", self)
                paste_act.triggered.connect(self._paste_clipboard)
                menu.addAction(paste_act)
                menu.addSeparator()
                rename_act = QAction("Rename\tF2", self)
                rename_act.triggered.connect(lambda: self._rename_item(widget, item))
                menu.addAction(rename_act)
                delete_act = QAction("Delete\tDel", self)
                delete_act.triggered.connect(lambda: self._delete_selected())
                menu.addAction(delete_act)
                menu.addSeparator()
                dup_act = QAction("Duplicate\tCtrl+D", self)
                dup_act.triggered.connect(lambda: self._duplicate_selected())
                menu.addAction(dup_act)
                menu.addSeparator()
                reveal_act = QAction("Show in File Manager", self)
                reveal_act.triggered.connect(lambda: self._reveal_file(path))
                menu.addAction(reveal_act)
                menu.addSeparator()
                status = self._vcs_status_of(path) if path else ""
                if status:
                    label = {"untracked": "Untracked", "added": "Added",
                             "modified": "Modified", "deleted": "Deleted",
                             "conflict": "Conflict", "renamed": "Renamed"}
                    info = QAction(f"VCS: {label.get(status, status)}", self)
                    info.setEnabled(False)
                    menu.addAction(info)
                if status in ("untracked", "modified", "deleted"):
                    add_act = QAction("Add to VCS", self)
                    add_act.triggered.connect(lambda p=path: self._git_add_file(p))
                    menu.addAction(add_act)
                if status == "added":
                    unstage_act = QAction("Unstage", self)
                    unstage_act.triggered.connect(lambda p=path: self._git_unstage_file(p))
                    menu.addAction(unstage_act)
                ignore_act = QAction("Ignore in Git", self)
                ignore_act.triggered.connect(lambda p=path: self._git_ignore_file(p))
                menu.addAction(ignore_act)
        menu.addSeparator()
        copy_path_act = QAction("Copy Path", self)
        if item:
            p = (item.data(0, Qt.ItemDataRole.UserRole) if isinstance(item, QTreeWidgetItem)
                 else item.data(Qt.ItemDataRole.UserRole))
            copy_path_act.triggered.connect(lambda: self._copy_path(p))
        else:
            copy_path_act.triggered.connect(lambda: self._copy_path(self._active_pane()._current_dir))
        menu.addAction(copy_path_act)
        menu.exec(widget.mapToGlobal(pos))

    def _rename_item(self, widget, item):
        if isinstance(item, QTreeWidgetItem):
            widget.editItem(item, 0)
        else:
            widget.editItem(item)

    def _start_drag_list(self, supported_actions):
        fl = self._active_pane()._file_list
        items = fl.selectedItems()
        if not items:
            return
        paths = [i.data(Qt.ItemDataRole.UserRole) for i in items if i.data(Qt.ItemDataRole.UserRole)]
        if not paths:
            return
        drag = QDrag(fl)
        mime = QMimeData()
        uri_paths = ["file:///" + os.path.abspath(p).replace("\\", "/") for p in paths]
        mime.setUrls([QUrl(u) for u in uri_paths])
        mime.setText(paths[0])
        if len(paths) == 1:
            ext = os.path.splitext(paths[0])[1].lower()
            if ext == ".zpep":
                mime.setData("application/x-zpep", QByteArray(paths[0].encode()))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction)

    def _on_file_list_drop(self, event):
        paths = self._extract_drop_paths(event)
        if not paths:
            return
        self._move_or_copy_files(paths, dest_dir=self._active_pane()._current_dir)
        event.acceptProposedAction()

    def _on_detail_tree_drop(self, event):
        paths = self._extract_drop_paths(event)
        if not paths:
            return
        self._move_or_copy_files(paths, dest_dir=self._active_pane()._current_dir)
        event.acceptProposedAction()

    def _on_folder_tree_drop(self, event):
        paths = self._extract_drop_paths(event)
        if not paths:
            return
        target_item = self._folder_tree.itemAt(event.position().toPoint())
        if target_item:
            target_path = target_item.data(0, Qt.ItemDataRole.UserRole)
        else:
            root = self._folder_tree.topLevelItem(0)
            target_path = root.data(0, Qt.ItemDataRole.UserRole)
        self._move_or_copy_files(paths, dest_dir=target_path)
        event.acceptProposedAction()

    def _on_entity_drop(self, event):
        if not event.mimeData().hasFormat(_ENTITY_MIME):
            return
        raw = bytes(event.mimeData().data(_ENTITY_MIME)).decode("utf-8")
        eids = [x.strip() for x in raw.split(",") if x.strip()]
        if not eids or not self._engine.scene:
            return
        from core.ecs.prefab import Prefab
        entity = self._engine.scene.get_entity(eids[0])
        if not entity:
            return
        prefab = Prefab(entity.name)
        prefab.capture([entity])
        ext = ".zpep"
        name = entity.name.replace(" ", "_").replace("/", "_")
        dest_dir = self._active_pane()._current_dir
        path = os.path.join(dest_dir, f"{name}{ext}")
        counter = 1
        while os.path.exists(path):
            path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
            counter += 1
        prefab.save(path)
        self._refresh()
        event.acceptProposedAction()

    def _extract_drop_paths(self, event):
        if event.mimeData().hasUrls():
            return [u.toLocalFile() for u in event.mimeData().urls()]
        if event.mimeData().hasText():
            paths = []
            for line in event.mimeData().text().splitlines():
                line = line.strip()
                if line:
                    paths.append(line)
            return paths
        return []

    def _move_or_copy_files(self, paths, dest_dir=None):
        if not dest_dir:
            dest_dir = self._active_pane()._current_dir
        errors = []
        for src in paths:
            if not os.path.exists(src):
                continue
            name = os.path.basename(src)
            dst = os.path.join(dest_dir, name)
            if src == dst or src.startswith(dst + os.sep):
                continue
            if os.path.exists(dst):
                base, ext = os.path.splitext(name)
                counter = 1
                while os.path.exists(os.path.join(dest_dir, f"{base} ({counter}){ext}")):
                    counter += 1
                dst = os.path.join(dest_dir, f"{base} ({counter}){ext}")
            try:
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            except OSError as e:
                errors.append(f"{name}: {e}")
        self._refresh()
        if errors:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Import Errors", "\n".join(errors))

    def _open_path_with_default_app(self, path):
        import subprocess, sys
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _start_drag_detail(self, supported_actions):
        dt = self._active_pane()._detail_tree
        items = dt.selectedItems()
        if not items:
            return
        paths = [i.data(0, Qt.ItemDataRole.UserRole) for i in items]
        if not paths:
            return
        drag = QDrag(dt)
        mime = QMimeData()
        uri_paths = ["file:///" + os.path.abspath(p).replace("\\", "/") for p in paths]
        mime.setUrls([QUrl(u) for u in uri_paths])
        mime.setText(paths[0])
        ext = os.path.splitext(paths[0])[1].lower()
        if len(paths) == 1 and ext == ".zpep":
            mime.setData("application/x-zpep", QByteArray(paths[0].encode()))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction)

    def _instantiate_prefab(self, path: str):
        if not self._engine.scene:
            return
        from core.ecs.prefab import Prefab
        from core.engine.engine import Engine
        pref = Prefab.load(path)
        if pref:
            e = pref.instantiate(self._engine.scene, Engine.instance()._component_registry)
            if e:
                self._engine._emit_event("entity_created", e)

    def _reveal_file(self, path: str):
        import subprocess, sys
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])

    def _copy_path(self, path: str):
        QGuiApplication.clipboard().setText(path)

    def _get_selected_paths(self):
        widget = self._active_widget()
        paths = []
        for item in widget.selectedItems():
            d = item.data(0, Qt.ItemDataRole.UserRole) if isinstance(item, QTreeWidgetItem) else item.data(Qt.ItemDataRole.UserRole)
            if d:
                paths.append(d)
        return paths

    def _copy_selected(self):
        global _file_clipboard, _clipboard_is_cut
        paths = self._get_selected_paths()
        if paths:
            _file_clipboard = list(paths)
            _clipboard_is_cut = False

    def _cut_selected(self):
        global _file_clipboard, _clipboard_is_cut
        paths = self._get_selected_paths()
        if paths:
            _file_clipboard = list(paths)
            _clipboard_is_cut = True

    def _paste_clipboard(self):
        global _file_clipboard, _clipboard_is_cut
        if not _file_clipboard:
            return
        target = self._active_pane()._current_dir
        errors = []
        for src in _file_clipboard:
            if not os.path.exists(src):
                continue
            name = os.path.basename(src)
            dst = os.path.join(target, name)
            if src == dst:
                continue
            if os.path.exists(dst):
                base, ext = os.path.splitext(name)
                counter = 1
                while os.path.exists(os.path.join(target, f"{base} ({counter}){ext}")):
                    counter += 1
                dst = os.path.join(target, f"{base} ({counter}){ext}")
            try:
                if _clipboard_is_cut:
                    shutil.move(src, dst)
                else:
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
            except OSError as e:
                errors.append(f"{name}: {e}")
        if _clipboard_is_cut:
            _file_clipboard = []
            _clipboard_is_cut = False
        self._refresh()
        if errors:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Paste Errors", "\n".join(errors))

    def _duplicate_selected(self):
        paths = self._get_selected_paths()
        if not paths:
            return
        errors = []
        for src in paths:
            if not os.path.exists(src):
                continue
            name = os.path.basename(src)
            base, ext = os.path.splitext(name)
            counter = 1
            while os.path.exists(os.path.join(os.path.dirname(src), f"{base} ({counter}){ext}")):
                counter += 1
            dst = os.path.join(os.path.dirname(src), f"{base} ({counter}){ext}")
            try:
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            except OSError as e:
                errors.append(f"{name}: {e}")
        self._refresh()
        if errors:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Duplicate Errors", "\n".join(errors))

    def _delete_selected(self):
        paths = self._get_selected_paths()
        if not paths:
            return
        from PyQt6.QtWidgets import QMessageBox
        names = "\n".join(os.path.basename(p) for p in paths[:10])
        if len(paths) > 10:
            names += f"\n... and {len(paths) - 10} more"
        reply = QMessageBox.question(self, "Delete",
            f"Delete {len(paths)} item(s)?\n{names}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            errors = []
            for p in paths:
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                    else:
                        os.remove(p)
                except OSError as e:
                    errors.append(f"{os.path.basename(p)}: {e}")
            self._refresh()
            if errors:
                QMessageBox.warning(self, "Delete Errors", "\n".join(errors))

    def _current_dir(self):
        return self._active_pane()._current_dir

    def _create_new_folder(self):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Create Folder", "Folder name:")
        if ok and name:
            try:
                os.makedirs(os.path.join(self._current_dir(), name), exist_ok=True)
                self._refresh()
            except OSError as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Error", f"Could not create folder:\n{e}")

    def _create_new_scene(self):
        path, _ = QFileDialog.getSaveFileName(self, "Create Scene", self._current_dir(), "Scenes (*.zpes)")
        if not path:
            return
        if not path.endswith(".zpes"):
            path += ".zpes"
        import json
        data = {"name": os.path.splitext(os.path.basename(path))[0], "entities": {}}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._refresh()

    def _create_new_material(self):
        path, _ = QFileDialog.getSaveFileName(self, "Create Material", self._current_dir(), "Materials (*.mat)")
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext != ".mat":
            path += ".mat"
        from core.assets.material import Material
        mat = Material(os.path.splitext(os.path.basename(path))[0])
        mat.save(path, self._engine.project_root)
        self._refresh()

    def _create_new_animclip(self):
        path, _ = QFileDialog.getSaveFileName(self, "Create Animation Clip", self._current_dir(), "Animation Clips (*.animclip)")
        if not path:
            return
        if not path.endswith(".animclip"):
            path += ".animclip"
        from core.components.animation.animation_clip import AnimationClip
        clip = AnimationClip(os.path.splitext(os.path.basename(path))[0])
        clip.save(path)
        self._refresh()

    def _create_new_animcontroller(self):
        path, _ = QFileDialog.getSaveFileName(self, "Create Animator Controller", self._current_dir(), "Animator Controllers (*.animcontroller)")
        if not path:
            return
        if not path.endswith(".animcontroller"):
            path += ".animcontroller"
        from core.components.animation.animator_controller import AnimatorController
        ctrl = AnimatorController(os.path.splitext(os.path.basename(path))[0])
        ctrl.save(path)
        self._refresh()

    def _create_new_script(self):
        path, _ = QFileDialog.getSaveFileName(self, "Create Script", self._current_dir(), "Python Scripts (*.py)")
        if not path:
            return
        if not path.endswith(".py"):
            path += ".py"
        template = '''from __future__ import annotations
from core.ecs.ecs import Component, ComponentRegistry
@ComponentRegistry.register
class NewScript(Component):
    def __init__(self):
        super().__init__()
        self.my_var: float = 0.0
    def update(self, dt: float):
        pass
'''
        with open(path, "w") as f:
            f.write(template)
        self._refresh()

    def open_resource(self, path: str):
        if os.path.isdir(path):
            return
        ext = os.path.splitext(path)[1].lower()
        if ext == ".zpes":
            self._engine.load_scene(path)
        elif ext == ".zpep":
            self.file_double_clicked.emit(path)
        elif ext in (".obj", ".fbx", ".stl", ".usdz", ".gltf", ".glb"):
            self.import_model_requested.emit(path)
        elif ext in (".animclip", ".animcontroller"):
            self.file_double_clicked.emit(path)
        elif ext == ".zterr":
            self.file_double_clicked.emit(path)
        else:
            self._open_path_with_default_app(path)

    def reveal_resource(self, path: str):
        norm = os.path.normpath(path)
        if not os.path.exists(norm):
            return
        self._sync_tree_selection(os.path.dirname(norm))
        pane = self._active_pane()
        pane.populate_files(os.path.dirname(norm))
        item = self._find_item_by_path(norm)
        if item:
            pane.active_widget().setCurrentItem(item)
            pane.active_widget().scrollToItem(item)
            self._flash_item(item)

    def flash_resource(self, path: str):
        norm = os.path.normpath(path)
        if not os.path.exists(norm):
            return
        self._sync_tree_selection(os.path.dirname(norm))
        pane = self._active_pane()
        pane.populate_files(os.path.dirname(norm))
        item = self._find_item_by_path(norm)
        if item:
            pane.active_widget().scrollToItem(item)
            self._flash_item(item)

    def _find_item_by_path(self, norm):
        pane = self._active_pane()
        if self._view_mode == VIEW_DETAILS:
            tree = pane._detail_tree
            for i in range(tree.topLevelItemCount()):
                it = tree.topLevelItem(i)
                if it and it.data(0, Qt.ItemDataRole.UserRole) == norm:
                    return it
        else:
            lst = pane._file_list
            for i in range(lst.count()):
                it = lst.item(i)
                if it and it.data(Qt.ItemDataRole.UserRole) == norm:
                    return it
        return None

    @staticmethod
    def _flash_item(item):
        try:
            if isinstance(item, QTreeWidgetItem):
                tree = item.treeWidget()
                if not tree:
                    return
                _flash_overlay(tree.viewport(), tree.visualItemRect(item))
            else:
                lst = item.listWidget()
                if not lst:
                    return
                _flash_overlay(lst.viewport(), lst.visualItemRect(item))
        except RuntimeError:
            pass

    def set_project_root(self, path: str):
        self._project_root = os.path.abspath(path)
        self._pane_a._current_dir = self._project_root
        if self._pane_b:
            self._pane_b._current_dir = self._project_root
        self._refresh_vcs_status()
        self._populate_tree()
