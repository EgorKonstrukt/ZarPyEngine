# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import subprocess
import time
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QPlainTextEdit, QListWidget,
    QListWidgetItem, QPushButton, QToolBar, QLineEdit, QLabel,
    QCheckBox, QTabWidget, QComboBox, QMenu, QProgressBar,
    QHeaderView, QMessageBox, QInputDialog, QApplication,
    QStyledItemDelegate,
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QThread, QSize, QPoint,
)
from PyQt6.QtGui import (
    QFont, QColor, QTextCharFormat, QSyntaxHighlighter,
    QPainter, QPen, QTextDocument,
)
from core.editor_scale import scale


def _find_git() -> str:
    for candidate in ["git.exe", "git"]:
        try:
            r = subprocess.run([candidate, "--version"], capture_output=True,
                               text=True, timeout=5)
            if r.returncode == 0:
                return candidate
        except FileNotFoundError:
            continue
    return ""


def _git_run(git: str, args: list[str], cwd: str | None = None,
             timeout: int = 30) -> tuple[int, str, str]:
    try:
        r = subprocess.run([git] + args, capture_output=True, text=True,
                           cwd=cwd, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError:
        return -2, "", "git not found"
    except Exception as e:
        return -3, "", str(e)


class _GitProcess(QThread):
    finished = pyqtSignal(int, str, str)

    def __init__(self, git: str, args: list[str], cwd: str | None = None):
        super().__init__()
        self._git = git
        self._args = args
        self._cwd = cwd

    def run(self):
        rc, out, err = _git_run(self._git, self._args, self._cwd, timeout=120)
        self.finished.emit(rc, out, err)


class _Git:
    def __init__(self):
        self._path = _find_git()
        self._repo_root: str = ""
        self._cwd: str = ""

    @property
    def available(self) -> bool:
        return bool(self._path)

    @property
    def repo_root(self) -> str:
        return self._repo_root

    def detect(self, path: str) -> bool:
        if not self._path:
            return False
        rc, out, _ = _git_run(self._path, ["rev-parse", "--show-toplevel"],
                               cwd=path)
        if rc == 0:
            self._repo_root = out.strip()
            self._cwd = self._repo_root
            return True
        self._repo_root = ""
        self._cwd = ""
        return False

    def run_sync(self, *args: str, timeout: int = 30) -> tuple[int, str, str]:
        return _git_run(self._path, list(args), self._cwd, timeout)

    def run_async(self, *args: str) -> _GitProcess:
        t = _GitProcess(self._path, list(args), self._cwd)
        t.start()
        return t

    def status(self) -> tuple[int, str, str]:
        return self.run_sync("status", "--porcelain", "-u")

    def branch_info(self) -> tuple[int, str, str]:
        return self.run_sync("rev-parse", "--abbrev-ref", "HEAD")

    def branch_list(self) -> tuple[int, str, str]:
        return self.run_sync("branch", "-a")

    def current_branch(self) -> str:
        rc, out, _ = self.branch_info()
        if rc == 0:
            return out.strip() or "HEAD"
        return ""

    def ahead_behind(self, branch: str = "") -> str:
        if not branch:
            branch = self.current_branch()
        if not branch:
            return ""
        upstream = f"{branch}@{{upstream}}"
        rc, out, _ = self.run_sync("rev-list", "--left-right",
                                    "--count", f"{upstream}...{branch}",
                                    timeout=10)
        if rc == 0:
            parts = out.strip().split()
            if len(parts) == 2:
                behind, ahead = parts[0], parts[1]
                if behind != "0" or ahead != "0":
                    return f" [+{ahead}/-{behind}]"
        return ""

    def staged_diff(self) -> str:
        rc, out, _ = self.run_sync("diff", "--cached", "--no-color", timeout=15)
        return out if rc == 0 else ""

    def unstaged_diff(self) -> str:
        rc, out, _ = self.run_sync("diff", "--no-color", timeout=15)
        return out if rc == 0 else ""

    def file_diff(self, path: str, staged: bool = False) -> str:
        args = ["diff", "--no-color"]
        if staged:
            args.append("--cached")
        args.append("--")
        args.append(path)
        rc, out, _ = self.run_sync(*args, timeout=15)
        return out if rc == 0 else ""

    def add(self, paths: list[str]) -> tuple[int, str, str]:
        return self.run_sync("add", "--", *paths)

    def unstage(self, paths: list[str]) -> tuple[int, str, str]:
        return self.run_sync("restore", "--staged", "--", *paths)

    def restore(self, paths: list[str]) -> tuple[int, str, str]:
        return self.run_sync("restore", "--", *paths)

    def commit(self, message: str, amend: bool = False,
               signoff: bool = False) -> tuple[int, str, str]:
        args = ["commit"]
        if amend:
            args.append("--amend")
        if signoff:
            args.append("--signoff")
        args.extend(["-m", message])
        return self.run_sync(*args)

    def log(self, count: int = 100) -> list[dict]:
        fmt = "%H%n%h%n%an%n%ae%n%at%n%s%n%P%n"
        rc, out, _ = self.run_sync("log", f"--max-count={count}",
                                    f"--format={fmt}", "--date=short",
                                    "--decorate=short", "--simplify-merges",
                                    timeout=15)
        if rc != 0:
            return []
        commits = []
        for block in out.strip().split("\n\n"):
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n")
            if len(lines) < 6:
                continue
            c = {
                "hash": lines[0],
                "short": lines[1],
                "author": lines[2],
                "email": lines[3],
                "time": int(lines[4]) if lines[4].isdigit() else 0,
                "message": lines[5],
                "parents": lines[6].split() if len(lines) > 6 else [],
            }
            commits.append(c)
        return commits

    def log_graph(self, count: int = 100) -> str:
        rc, out, _ = self.run_sync("log", f"--max-count={count}",
                                    "--oneline", "--graph",
                                    "--decorate=short", "--all",
                                    timeout=15)
        return out if rc == 0 else ""

    def create_branch(self, name: str) -> tuple[int, str, str]:
        return self.run_sync("branch", name)

    def switch_branch(self, name: str) -> tuple[int, str, str]:
        return self.run_sync("checkout", name)

    def delete_branch(self, name: str, force: bool = False) -> tuple[int, str, str]:
        args = ["branch"]
        if force:
            args.append("-D")
        else:
            args.append("-d")
        args.append(name)
        return self.run_sync(*args)

    def merge_branch(self, name: str) -> tuple[int, str, str]:
        return self.run_sync("merge", name)

    def stash_list(self) -> list[dict]:
        rc, out, _ = self.run_sync("stash", "list",
                                    "--format=%gd%n%gs%n%at%n")
        if rc != 0 or not out.strip():
            return []
        stashes = []
        for block in out.strip().split("\n\n"):
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n")
            if len(lines) < 2:
                continue
            s = {
                "ref": lines[0],
                "message": lines[1],
                "time": int(lines[2]) if len(lines) > 2 and lines[2].isdigit() else 0,
            }
            stashes.append(s)
        return stashes

    def stash_push(self, message: str = "") -> tuple[int, str, str]:
        args = ["stash", "push"]
        if message:
            args.extend(["-m", message])
        return self.run_sync(*args)

    def stash_pop(self, ref: str = "stash@{0}") -> tuple[int, str, str]:
        return self.run_sync("stash", "pop", ref)

    def stash_drop(self, ref: str = "stash@{0}") -> tuple[int, str, str]:
        return self.run_sync("stash", "drop", ref)

    def remotes(self) -> list[dict]:
        rc, out, _ = self.run_sync("remote", "-v")
        if rc != 0:
            return []
        seen = set()
        result = []
        for line in out.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                url = parts[1]
                if name not in seen:
                    seen.add(name)
                    result.append({"name": name, "url": url})
        return result

    def push(self, remote: str = "origin",
             branch: str = "") -> _GitProcess:
        args = ["push"]
        if remote:
            args.append(remote)
        if branch:
            args.append(branch)
        return self.run_async(*args)

    def pull(self, remote: str = "origin",
             branch: str = "") -> _GitProcess:
        args = ["pull"]
        if remote:
            args.append(remote)
        if branch:
            args.append(branch)
        return self.run_async(*args)

    def fetch(self, remote: str = "") -> _GitProcess:
        args = ["fetch"]
        if remote:
            args.append(remote)
        return self.run_async(*args)

    def tags(self) -> list[str]:
        rc, out, _ = self.run_sync("tag", "-l", timeout=10)
        if rc != 0:
            return []
        return [t for t in out.strip().split("\n") if t]

    def blame(self, path: str) -> str:
        rc, out, _ = self.run_sync("blame", "--date=short", "--", path,
                                    timeout=15)
        return out if rc == 0 else ""

    def show_commit(self, rev: str) -> str:
        rc, out, _ = self.run_sync("show", "--no-color", "--stat", rev,
                                    timeout=15)
        return out if rc == 0 else ""


_parse_cache: dict[str, list[dict]] = {}


def _parse_porcelain(status_output: str) -> list[dict]:
    if status_output in _parse_cache:
        return _parse_cache[status_output]
    entries: list[dict] = []
    for line in status_output.strip().split("\n"):
        line = line.rstrip()
        if not line or len(line) < 3:
            continue
        xy = line[:2]
        path = line[3:]
        if xy[1] == " ":
            staged = xy[0] != " "
        else:
            staged = xy[0] != " "
        unstaged = xy[1] != " "
        if xy[0] == "?" and xy[1] == "?":
            staged = False
            unstaged = False
            status = "untracked"
        elif xy[0] == "!" and xy[1] == "!":
            continue
        elif xy[0] == "U" or xy[1] == "U":
            status = "conflict"
        elif staged and unstaged:
            status = "modified"
        elif staged:
            status = "staged"
        elif unstaged:
            status = "unstaged"
        else:
            status = "unknown"
        entries.append({
            "path": path,
            "status": status,
            "xy": xy,
            "staged": staged,
            "unstaged": unstaged,
        })
    _parse_cache[status_output] = entries
    return entries


class _DiffHighlighter(QSyntaxHighlighter):
    def __init__(self, doc: QTextDocument):
        super().__init__(doc)
        self._file_hdr_fmt = QTextCharFormat()
        self._file_hdr_fmt.setForeground(QColor("#569cd6"))
        self._file_hdr_fmt.setFontWeight(700)
        self._add_fmt = QTextCharFormat()
        self._add_fmt.setForeground(QColor("#6a9955"))
        self._del_fmt = QTextCharFormat()
        self._del_fmt.setForeground(QColor("#f44747"))
        self._hunk_fmt = QTextCharFormat()
        self._hunk_fmt.setForeground(QColor("#dcdcaa"))

    def highlightBlock(self, text: str) -> None:
        if not text:
            return
        if text.startswith("+++") or text.startswith("---"):
            self.setFormat(0, len(text), self._file_hdr_fmt)
        elif text.startswith("@@"):
            self.setFormat(0, len(text), self._hunk_fmt)
        elif text.startswith("+"):
            self.setFormat(0, len(text), self._add_fmt)
        elif text.startswith("-"):
            self.setFormat(0, len(text), self._del_fmt)


class _DiffView(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Courier New", 10))
        self.setLineWrapMode(self.LineWrapMode.NoWrap)
        self.setTabStopDistance(scale(24))
        self._highlighter = _DiffHighlighter(self.document())

    def show_diff(self, diff_text: str):
        self.setPlainText(diff_text)

    def clear_diff(self):
        self.clear()


class _CommitPanel(QWidget):
    commit_requested = pyqtSignal(str, bool, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._subject = QLineEdit()
        self._subject.setPlaceholderText("Commit message (Ctrl+Enter to commit)")
        layout.addWidget(self._subject)

        opt_row = QHBoxLayout()
        opt_row.setSpacing(8)
        self._amend_cb = QCheckBox("Amend")
        opt_row.addWidget(self._amend_cb)
        self._signoff_cb = QCheckBox("Sign-off")
        opt_row.addWidget(self._signoff_cb)
        opt_row.addStretch()
        self._commit_btn = QPushButton("Commit")
        self._commit_btn.clicked.connect(self._on_commit)
        opt_row.addWidget(self._commit_btn)
        layout.addLayout(opt_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

    def _on_commit(self):
        msg = self._subject.text().strip()
        if not msg:
            self._status_label.setText("Commit message is required")
            return
        self.commit_requested.emit(msg, self._amend_cb.isChecked(),
                                   self._signoff_cb.isChecked())

    def clear_message(self):
        self._subject.clear()
        self._amend_cb.setChecked(False)
        self._status_label.setText("")

    def set_status(self, text: str):
        self._status_label.setText(text)


class _FileItemDelegate(QStyledItemDelegate):
    _status_colors = {
        "staged": ("#6a9955", "\u2713 "),
        "unstaged": ("#dcdcaa", "~ "),
        "untracked": ("#f44747", "? "),
        "conflict": ("#ce9178", "! "),
        "modified": ("#569cd6", "M "),
    }

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if index.column() != 0:
            return
        status = index.data(Qt.ItemDataRole.UserRole + 1)
        if status not in self._status_colors:
            return
        color_str, prefix = self._status_colors[status]
        painter.save()
        painter.setPen(QColor(color_str))
        font = option.font
        font.setBold(True)
        painter.setFont(font)
        text_rect = option.rect.adjusted(4, 0, 0, 0)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, prefix)
        painter.restore()


class _StatusTree(QTreeWidget):
    file_selected = pyqtSignal(str, bool)
    file_action_requested = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Path", "Status"])
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.setAlternatingRowColors(True)
        self.setRootIsDecorated(True)
        self.setAnimated(True)
        self.setItemDelegate(_FileItemDelegate(self))
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        self.itemClicked.connect(self._on_item_clicked)
        self._entries: list[dict] = []

    def set_entries(self, entries: list[dict]):
        self._entries = entries
        self.clear()
        sections: dict[str, list[dict]] = {
            "staged": [],
            "unstaged": [],
            "untracked": [],
            "conflict": [],
        }
        for e in entries:
            s = e["status"]
            if s not in sections:
                s = "unstaged"
            sections[s].append(e)

        section_labels = {
            "staged": "Staged",
            "conflict": "Conflicts",
            "unstaged": "Unstaged",
            "untracked": "Untracked",
        }
        for key in ["staged", "conflict", "unstaged", "untracked"]:
            items = sections[key]
            if not items:
                continue
            section_item = QTreeWidgetItem([f"{section_labels[key]} ({len(items)})", ""])
            section_item.setFlags(section_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            bold_font = self.font()
            bold_font.setBold(True)
            section_item.setFont(0, bold_font)
            self.addTopLevelItem(section_item)
            for e in items:
                child = QTreeWidgetItem([e["path"], e["status"]])
                child.setData(0, Qt.ItemDataRole.UserRole + 1, e["status"])
                child.setData(0, Qt.ItemDataRole.UserRole, e["path"])
                section_item.addChild(child)

        self.expandAll()

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path:
            staged = False
            parent = item.parent()
            if parent:
                parent_text = parent.text(0)
                if parent_text.startswith("Staged"):
                    staged = True
                elif parent_text.startswith("Conflict"):
                    staged = False
            self.file_selected.emit(path, staged)

    def _on_context_menu(self, pos: QPoint):
        item = self.itemAt(pos)
        if not item:
            return
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if not path:
            return
        parent = item.parent()
        section = parent.text(0) if parent else ""
        menu = QMenu(self)
        if section.startswith("Staged"):
            a1 = menu.addAction("Unstage")
            a1.triggered.connect(lambda: self.file_action_requested.emit(path, "unstage"))
        elif section.startswith("Unstaged"):
            a2 = menu.addAction("Stage")
            a2.triggered.connect(lambda: self.file_action_requested.emit(path, "stage"))
            a3 = menu.addAction("Discard Changes")
            a3.triggered.connect(lambda: self.file_action_requested.emit(path, "discard"))
        elif section.startswith("Conflict"):
            a4 = menu.addAction("Stage (Mark Resolved)")
            a4.triggered.connect(lambda: self.file_action_requested.emit(path, "stage"))
            a5 = menu.addAction("Open File")
            a5.triggered.connect(lambda: self.file_action_requested.emit(path, "open"))
        elif section.startswith("Untracked"):
            a6 = menu.addAction("Stage (Add)")
            a6.triggered.connect(lambda: self.file_action_requested.emit(path, "stage"))
            a7 = menu.addAction("Add to .gitignore")
            a7.triggered.connect(lambda: self.file_action_requested.emit(path, "gitignore"))
        menu.addSeparator()
        a8 = menu.addAction("Copy Path")
        a8.triggered.connect(lambda: QApplication.clipboard().setText(path))
        menu.exec(self.viewport().mapToGlobal(pos))


class _LogTree(QTreeWidget):
    commit_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Graph", "Commit", "Author", "Date", "Message"])
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.header().resizeSection(0, scale(120))
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.header().resizeSection(1, scale(70))
        self.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.header().resizeSection(3, scale(100))
        self.header().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.setAlternatingRowColors(True)
        self.setRootIsDecorated(False)
        self.setFont(QFont("Courier New", 9))
        self.itemClicked.connect(self._on_click)
        self._commits: list[dict] = []

    def set_log(self, graph_text: str, commits: list[dict]):
        self._commits = commits
        self.clear()
        g_lines = graph_text.strip().split("\n") if graph_text else []
        for i, c in enumerate(commits):
            graph_part = g_lines[i] if i < len(g_lines) else ""
            date_str = time.strftime("%Y-%m-%d %H:%M",
                                     time.gmtime(c["time"])) if c["time"] else ""
            items = [
                graph_part,
                c["short"],
                c["author"],
                date_str,
                c["message"],
            ]
            item = QTreeWidgetItem(items)
            item.setData(1, Qt.ItemDataRole.UserRole, c["hash"])
            item.setToolTip(0, graph_part)
            item.setToolTip(1, c["hash"])
            item.setToolTip(4, c["message"])
            self.addTopLevelItem(item)

    def _on_click(self, item: QTreeWidgetItem, column: int):
        h = item.data(1, Qt.ItemDataRole.UserRole)
        if h:
            self.commit_selected.emit(h)


class _BranchList(QListWidget):
    branch_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.itemDoubleClicked.connect(self._on_double_click)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    def set_branches(self, branches_text: str, current: str):
        self.clear()
        for line in branches_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            is_current = line.startswith("*")
            name = line[2:] if line.startswith("* ") else line
            name = name.strip()
            item = QListWidgetItem(line)
            item.setData(Qt.ItemDataRole.UserRole, name)
            if is_current:
                bold = self.font()
                bold.setBold(True)
                item.setFont(bold)
            self.addItem(item)

    def _on_double_click(self, item: QListWidgetItem):
        name = item.data(Qt.ItemDataRole.UserRole)
        if name:
            self.branch_selected.emit(name)

    def _on_context_menu(self, pos):
        item = self.itemAt(pos)
        if not item:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        if not name:
            return
        menu = QMenu(self)
        a1 = menu.addAction("Switch")
        a1.triggered.connect(lambda: self.branch_selected.emit(name))
        a2 = menu.addAction("Delete")
        a2.triggered.connect(lambda: self._request_delete(name))
        a3 = menu.addAction("Merge into current")
        a3.triggered.connect(lambda: self._request_merge(name))
        menu.exec(self.viewport().mapToGlobal(pos))

    def _request_delete(self, name: str):
        self.parent().parent().parent()._delete_branch(name)

    def _request_merge(self, name: str):
        self.parent().parent().parent()._merge_branch(name)


class _StashList(QListWidget):
    stash_action = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    def set_stashes(self, stashes: list[dict]):
        self.clear()
        for s in stashes:
            msg = s["message"]
            date_str = time.strftime("%Y-%m-%d %H:%M",
                                     time.gmtime(s["time"])) if s["time"] else ""
            item = QListWidgetItem(f"{s['ref']}: {msg} ({date_str})")
            item.setData(Qt.ItemDataRole.UserRole, s["ref"])
            self.addItem(item)

    def _on_context_menu(self, pos):
        item = self.itemAt(pos)
        if not item:
            return
        ref = item.data(Qt.ItemDataRole.UserRole)
        if not ref:
            return
        menu = QMenu(self)
        a1 = menu.addAction("Pop")
        a1.triggered.connect(lambda: self.stash_action.emit(ref, "pop"))
        a2 = menu.addAction("Drop")
        a2.triggered.connect(lambda: self.stash_action.emit(ref, "drop"))
        menu.exec(self.viewport().mapToGlobal(pos))


class _RemoteWidget(QWidget):
    push_requested = pyqtSignal(str, str)
    pull_requested = pyqtSignal(str, str)
    fetch_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._list = QListWidget()
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._remote_combo = QComboBox()
        btn_row.addWidget(self._remote_combo)
        self._branch_combo = QComboBox()
        btn_row.addWidget(self._branch_combo)
        layout.addLayout(btn_row)

        btn_row2 = QHBoxLayout()
        self._push_btn = QPushButton("Push")
        self._push_btn.clicked.connect(self._on_push)
        btn_row2.addWidget(self._push_btn)
        self._pull_btn = QPushButton("Pull")
        self._pull_btn.clicked.connect(self._on_pull)
        btn_row2.addWidget(self._pull_btn)
        self._fetch_btn = QPushButton("Fetch")
        self._fetch_btn.clicked.connect(self._on_fetch)
        btn_row2.addWidget(self._fetch_btn)
        layout.addLayout(btn_row2)

    def set_remotes(self, remotes: list[dict], branches: list[str],
                    current_branch: str = ""):
        self._list.clear()
        for r in remotes:
            self._list.addItem(f"{r['name']}  {r['url']}")
        self._remote_combo.clear()
        self._remote_combo.addItems([r["name"] for r in remotes])
        self._branch_combo.clear()
        self._branch_combo.addItem("(current)")
        for b in branches:
            self._branch_combo.addItem(b)
        idx = self._branch_combo.findText(current_branch)
        if idx >= 0:
            self._branch_combo.setCurrentIndex(idx)

    def _on_push(self):
        remote = self._remote_combo.currentText()
        branch = self._branch_combo.currentText()
        if branch == "(current)":
            branch = ""
        self.push_requested.emit(remote, branch)

    def _on_pull(self):
        remote = self._remote_combo.currentText()
        branch = self._branch_combo.currentText()
        if branch == "(current)":
            branch = ""
        self.pull_requested.emit(remote, branch)

    def _on_fetch(self):
        remote = self._remote_combo.currentText()
        self.fetch_requested.emit(remote)


class VcsPanel(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Version Control", parent)
        self.setObjectName("VersionControlDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable)

        self._git = _Git()
        self._recent_commits: list[dict] = []
        self._branches_text: str = ""
        self._branches_list: list[str] = []
        self._status_entries: list[dict] = []
        self._last_status_output: str = ""
        self._watch_timer = QTimer(self)
        self._watch_timer.timeout.connect(self._refresh_status)
        self._watch_interval: int = 3000
        self._diff_cache: dict[str, str] = {}

        self._setup_ui()
        self._watch_timer.start(self._watch_interval)
        self._try_detect()
        self._refresh_all()

    def _setup_ui(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QToolBar()
        toolbar.setIconSize(QSize(scale(16), scale(16)))
        toolbar.setMovable(False)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh_all)
        toolbar.addWidget(self._refresh_btn)

        self._repo_label = QLabel(" No repo ")
        self._repo_label.setStyleSheet("padding: 2px 6px;")
        toolbar.addWidget(self._repo_label)

        self._branch_label = QLabel("")
        self._branch_label.setStyleSheet("padding: 2px 6px; font-weight: bold;")
        toolbar.addWidget(self._branch_label)

        self._ahead_behind_label = QLabel("")
        toolbar.addWidget(self._ahead_behind_label)

        toolbar.addSeparator()

        self._stash_btn = QPushButton("Stash")
        self._stash_btn.clicked.connect(self._stash_push)
        toolbar.addWidget(self._stash_btn)

        self._progress = QProgressBar()
        self._progress.setMaximumWidth(scale(150))
        self._progress.setMaximumHeight(scale(14))
        self._progress.hide()
        toolbar.addWidget(self._progress)

        layout.addWidget(toolbar)

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._status_tree = _StatusTree()
        self._status_tree.file_selected.connect(self._on_file_selected)
        self._status_tree.file_action_requested.connect(self._on_file_action)
        top_splitter.addWidget(self._status_tree)

        tabs = QTabWidget()
        self._diff_view = _DiffView()
        tabs.addTab(self._diff_view, "Diff")

        self._log_tree = _LogTree()
        self._log_tree.commit_selected.connect(self._on_commit_selected)
        tabs.addTab(self._log_tree, "Log")

        self._branch_list = _BranchList()
        self._branch_list.branch_selected.connect(self._switch_branch)
        tabs.addTab(self._branch_list, "Branches")

        self._stash_list = _StashList()
        self._stash_list.stash_action.connect(self._on_stash_action)
        tabs.addTab(self._stash_list, "Stashes")

        self._remote_widget = _RemoteWidget()
        self._remote_widget.push_requested.connect(self._on_push)
        self._remote_widget.pull_requested.connect(self._on_pull)
        self._remote_widget.fetch_requested.connect(self._on_fetch)
        tabs.addTab(self._remote_widget, "Remotes")

        tabs.currentChanged.connect(self._on_tab_changed)
        top_splitter.addWidget(tabs)
        top_splitter.setSizes([scale(300), scale(500)])
        main_splitter.addWidget(top_splitter)

        self._commit_panel = _CommitPanel()
        self._commit_panel.commit_requested.connect(self._on_commit)
        main_splitter.addWidget(self._commit_panel)
        main_splitter.setSizes([scale(400), scale(120)])

        layout.addWidget(main_splitter)

        self.setWidget(w)

    def _try_detect(self):
        eng = None
        try:
            from core.engine import Engine
            eng = Engine.instance()
        except Exception:
            pass
        project_path = ""
        if eng:
            project_path = getattr(eng, "_project_path", "") or ""
        if not project_path:
            project_path = os.getcwd()
        if self._git.detect(project_path):
            self._repo_label.setText(f" {os.path.basename(self._git.repo_root)} ")
            self._repo_label.setToolTip(self._git.repo_root)
            return True
        else:
            self._repo_label.setText(" No repository ")
            self._branch_label.setText("")
            self._ahead_behind_label.setText("")
            return False

    def _refresh_all(self):
        if not self._git.available:
            return
        if not self._git.repo_root:
            if not self._try_detect():
                return
        self._refresh_branch_info()
        self._refresh_status()
        self._refresh_log()
        self._refresh_branches()
        self._refresh_stashes()
        self._refresh_remotes()
        self._refresh_tags()

    def _refresh_branch_info(self):
        branch = self._git.current_branch()
        if branch:
            self._branch_label.setText(f" [{branch}] ")
            self._branch_label.setToolTip(f"Current branch: {branch}")
        ab = self._git.ahead_behind(branch) if branch else ""
        self._ahead_behind_label.setText(ab)

    def _refresh_status(self):
        if not self._git.repo_root:
            return
        rc, out, _ = self._git.status()
        if rc != 0:
            return
        if out == self._last_status_output:
            return
        self._last_status_output = out
        self._status_entries = _parse_porcelain(out)
        self._status_tree.set_entries(self._status_entries)
        staged = sum(1 for e in self._status_entries if e["staged"])
        unstaged = sum(1 for e in self._status_entries
                       if not e["staged"] and e["status"] != "untracked")
        untracked = sum(1 for e in self._status_entries if e["status"] == "untracked")
        conflicts = sum(1 for e in self._status_entries if e["status"] == "conflict")
        parts = []
        if conflicts:
            parts.append(f"Conflicts: {conflicts}")
        if staged:
            parts.append(f"Staged: {staged}")
        if unstaged:
            parts.append(f"Modified: {unstaged}")
        if untracked:
            parts.append(f"Untracked: {untracked}")
        self._commit_panel.set_status(" | ".join(parts) if parts else " Working tree clean")
        self._diff_cache.clear()

    def _refresh_log(self):
        commits = self._git.log(100)
        if commits != self._recent_commits:
            self._recent_commits = commits
            graph_text = self._git.log_graph(100)
            self._log_tree.set_log(graph_text, commits)

    def _refresh_branches(self):
        rc, out, _ = self._git.branch_list()
        if rc == 0:
            self._branches_text = out
            current = self._git.current_branch()
            self._branch_list.set_branches(out, current)
            self._branches_list = []
            for line in out.strip().split("\n"):
                line = line.strip()
                if line:
                    name = line[2:] if line.startswith("* ") else line
                    self._branches_list.append(name.strip())
            self._remote_widget.set_remotes(self._git.remotes(),
                                            self._branches_list, current)

    def _refresh_stashes(self):
        stashes = self._git.stash_list()
        self._stash_list.set_stashes(stashes)

    def _refresh_remotes(self):
        remotes = self._git.remotes()
        self._remote_widget.set_remotes(remotes, self._branches_list,
                                        self._git.current_branch())

    def _refresh_tags(self):
        pass

    def _on_file_selected(self, path: str, staged: bool):
        cache_key = f"{'staged' if staged else 'unstaged'}:{path}"
        if cache_key in self._diff_cache:
            self._diff_view.show_diff(self._diff_cache[cache_key])
            return
        diff = self._git.file_diff(path, staged)
        self._diff_cache[cache_key] = diff
        self._diff_view.show_diff(diff)

    def _on_file_action(self, path: str, action: str):
        if action == "stage":
            rc, _, err = self._git.add([path])
            if rc == 0:
                self._refresh_status()
                self._commit_panel.set_status(f"Staged: {path}")
            else:
                self._commit_panel.set_status(f"Error: {err}")
        elif action == "unstage":
            rc, _, err = self._git.unstage([path])
            if rc == 0:
                self._refresh_status()
                self._commit_panel.set_status(f"Unstaged: {path}")
            else:
                self._commit_panel.set_status(f"Error: {err}")
        elif action == "discard":
            reply = QMessageBox.question(
                self, "Discard Changes",
                f"Discard changes in {path}?\nThis cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                rc, _, err = self._git.restore([path])
                if rc == 0:
                    self._refresh_status()
                    self._commit_panel.set_status(f"Restored: {path}")
                else:
                    self._commit_panel.set_status(f"Error: {err}")
        elif action == "open":
            full = os.path.join(self._git.repo_root, path)
            try:
                os.startfile(full)
            except Exception:
                pass
        elif action == "gitignore":
            gi_path = os.path.join(self._git.repo_root, ".gitignore")
            try:
                with open(gi_path, "a", encoding="utf-8") as f:
                    f.write(f"\n/{path}\n")
                self._refresh_status()
                self._commit_panel.set_status(f"Added {path} to .gitignore")
            except Exception as e:
                self._commit_panel.set_status(f"Error: {e}")

    def _on_commit(self, message: str, amend: bool, signoff: bool):
        rc, out, err = self._git.commit(message, amend, signoff)
        if rc == 0:
            self._commit_panel.clear_message()
            self._commit_panel.set_status(f"Committed: {message}")
            self._refresh_all()
        else:
            err_str = err or out
            self._commit_panel.set_status(f"Commit failed: {err_str[:200]}")

    def _on_commit_selected(self, rev: str):
        detail = self._git.show_commit(rev)
        idx = self._log_tree.parent().indexOf(self._log_tree)
        if idx >= 0:
            parent_tabs = self._log_tree.parent()
            for i in range(parent_tabs.count()):
                if parent_tabs.widget(i) is self._diff_view:
                    parent_tabs.setCurrentIndex(i)
                    break
        self._diff_view.show_diff(detail)

    def _switch_branch(self, name: str):
        current = self._git.current_branch()
        if name == current:
            return
        reply = QMessageBox.question(
            self, "Switch Branch",
            f"Switch to branch '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        rc, out, err = self._git.switch_branch(name)
        if rc == 0:
            self._commit_panel.set_status(f"Switched to branch '{name}'")
            self._refresh_all()
        else:
            self._commit_panel.set_status(f"Error: {err or out}")

    def _delete_branch(self, name: str):
        current = self._git.current_branch()
        if name == current:
            self._commit_panel.set_status("Cannot delete current branch")
            return
        reply = QMessageBox.question(
            self, "Delete Branch",
            f"Delete branch '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        rc, out, err = self._git.delete_branch(name)
        if rc == 0:
            self._commit_panel.set_status(f"Deleted branch '{name}'")
            self._refresh_branches()
        else:
            self._commit_panel.set_status(f"Error: {err or out}")

    def _merge_branch(self, name: str):
        reply = QMessageBox.question(
            self, "Merge Branch",
            f"Merge '{name}' into current branch?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        rc, out, err = self._git.merge_branch(name)
        if rc == 0:
            self._commit_panel.set_status(f"Merged '{name}'")
            self._refresh_all()
        else:
            self._commit_panel.set_status(f"Merge error: {err or out}")

    def _stash_push(self):
        msg, ok = QInputDialog.getText(self, "Stash Changes",
                                        "Stash message (optional):")
        if not ok:
            return
        rc, out, err = self._git.stash_push(msg)
        if rc == 0:
            self._commit_panel.set_status("Changes stashed")
            self._refresh_all()
        else:
            self._commit_panel.set_status(f"Stash error: {err or out}")

    def _on_stash_action(self, ref: str, action: str):
        if action == "pop":
            rc, out, err = self._git.stash_pop(ref)
            if rc == 0:
                self._commit_panel.set_status(f"Popped {ref}")
                self._refresh_all()
            else:
                self._commit_panel.set_status(f"Error: {err or out}")
        elif action == "drop":
            reply = QMessageBox.question(
                self, "Drop Stash",
                f"Drop {ref}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                rc, out, err = self._git.stash_drop(ref)
                if rc == 0:
                    self._commit_panel.set_status(f"Dropped {ref}")
                    self._refresh_stashes()
                else:
                    self._commit_panel.set_status(f"Error: {err or out}")

    def _on_push(self, remote: str, branch: str):
        self._progress.show()
        self._progress.setValue(0)
        self._progress.setRange(0, 0)
        proc = self._git.push(remote, branch)
        proc.finished.connect(lambda rc, out, err: self._on_async_done(
            rc, out, err, "Push"))
        self._commit_panel.set_status(f"Pushing to {remote}/{branch or '(current)'}...")

    def _on_pull(self, remote: str, branch: str):
        self._progress.show()
        self._progress.setRange(0, 0)
        proc = self._git.pull(remote, branch)
        proc.finished.connect(lambda rc, out, err: self._on_async_done(
            rc, out, err, "Pull"))
        self._commit_panel.set_status(f"Pulling from {remote}/{branch or '(current)'}...")

    def _on_fetch(self, remote: str):
        self._progress.show()
        self._progress.setRange(0, 0)
        proc = self._git.fetch(remote)
        proc.finished.connect(lambda rc, out, err: self._on_async_done(
            rc, out, err, "Fetch"))
        self._commit_panel.set_status(f"Fetching from {remote}...")

    def _on_async_done(self, rc: int, out: str, err: str, label: str):
        self._progress.hide()
        if rc == 0:
            self._commit_panel.set_status(f"{label} completed")
            QTimer.singleShot(500, self._refresh_all)
        else:
            err_msg = err or out
            self._commit_panel.set_status(f"{label} failed: {err_msg[:200]}")
            self._diff_view.show_diff(f"--- {label} Output ---\n{out}\n{err}")

    def _on_tab_changed(self, index: int):
        tab = self.sender()
        if tab and tab.widget(index) is self._log_tree:
            self._refresh_log()

    def load_config(self, config):
        pass
