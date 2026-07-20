# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
import traceback
import urllib.request
import urllib.error
import urllib.parse
import webbrowser
from datetime import datetime, timezone

from PyQt6.QtWidgets import QMessageBox, QApplication
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal

from core.config.constants import APP_VERSION_DISPLAY

_GITHUB_REPO = "EgorKonstrukt/ZarinEngine"
_GITHUB_API = f"https://api.github.com/repos/{_GITHUB_REPO}/issues"
_GITHUB_NEW_ISSUE = f"https://github.com/{_GITHUB_REPO}/issues/new"
_MIN_INTERVAL = 60
_last_submit_ts = 0.0


def _get_github_token() -> str:
    return os.environ.get("GITHUB_TOKEN", "")


def _get_gpu_name() -> str:
    app = QApplication.instance()
    if app is None:
        return "N/A"
    for w in app.topLevelWidgets():
        vp = getattr(w, '_viewport', None)
        if vp is not None:
            cache = getattr(vp, '_gl_info_cache', None)
            if cache:
                name = cache.get("GL_RENDERER", "")
                if name:
                    return name
            break
    return "N/A"


def _collect_context() -> str:
    parts = []
    parts.append(f"**Version:** {APP_VERSION_DISPLAY}")
    parts.append(f"**OS:** {platform.system()} {platform.release()} ({platform.version()})")
    parts.append(f"**Machine:** {platform.machine()}")
    parts.append(f"**Python:** {sys.version.split()[0]}")
    parts.append(f"**Processor:** {platform.processor() or 'N/A'}")
    parts.append(f"**GPU:** {_get_gpu_name()}")
    gil_enabled = getattr(sys, '_is_gil_enabled', lambda: True)()
    parts.append(f"**GIL:** {'Enabled' if gil_enabled else 'Disabled (nogil)'}")
    return "\n".join(parts)


def _format_issue(title: str, body: str, context: str = "") -> dict:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    md = f"## Bug Report\n\n"
    md += f"**Timestamp:** {now}\n\n"
    if context:
        md += f"### Environment\n{context}\n\n"
    md += f"### Description\n{body}\n"
    return {"title": title, "body": md, "labels": ["bug", "auto-report"]}


def _build_browser_url(title: str, body: str, context: str = "") -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    full_body = f"## Bug Report\n\n**Timestamp:** {now}\n\n"
    if context:
        full_body += f"### Environment\n{context}\n\n"
    full_body += f"### Description\n{body}\n"
    params = urllib.parse.urlencode({"title": title, "body": full_body, "labels": "bug,auto-report"})
    return f"{_GITHUB_NEW_ISSUE}?{params}"


def _submit_issue(title: str, body: str, context: str = "") -> dict:
    global _last_submit_ts
    now = time.time()
    if now - _last_submit_ts < _MIN_INTERVAL:
        return {"error": "rate_limited", "retry_after": _MIN_INTERVAL - (now - _last_submit_ts)}
    _last_submit_ts = now

    payload = _format_issue(title, body, context)
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        _GITHUB_API,
        data=data,
        headers={
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": f"ZarinEngine/{APP_VERSION_DISPLAY}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )

    token = _get_github_token()
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return {"url": result.get("html_url", ""), "number": result.get("number", 0)}
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return {"error": f"HTTP {e.code}", "detail": err_body}
    except Exception as e:
        return {"error": str(e)}


class _SubmitWorker(QObject):
    finished = pyqtSignal(dict)

    def __init__(self, title: str, body: str, context: str):
        super().__init__()
        self._title = title
        self._body = body
        self._context = context

    def run(self):
        result = _submit_issue(self._title, self._body, self._context)
        self.finished.emit(result)


def _submit_async(parent, title: str, body: str, context: str = "", on_done=None):
    thread = QThread(parent)
    worker = _SubmitWorker(title, body, context)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(lambda result: _on_submit_done(result, on_done, thread, worker))
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread


def _on_submit_done(result, on_done, thread, worker):
    if "url" in result:
        QMessageBox.information(
            None,
            "Bug Report Submitted",
            f"Thank you!\n\nIssue created:\n{result['url']}",
        )
    elif result.get("error") == "rate_limited":
        QMessageBox.warning(
            None,
            "Rate Limited",
            f"Too many requests. Please wait {int(result.get('retry_after', 60))} seconds.",
        )
    else:
        error_msg = result.get("error", "Unknown error")
        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Submission Failed")
        box.setText(f"Could not submit via API:\n{error_msg}")
        box.setInformativeText("Open GitHub in your browser to submit manually?")
        box.setDetailedText(result.get("detail", ""))
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.Yes)
        box.button(QMessageBox.StandardButton.Yes).setText("Open in Browser")
        box.button(QMessageBox.StandardButton.No).setText("Cancel")
        if box.exec() == QMessageBox.StandardButton.Yes:
            url = _build_browser_url(_last_title, _last_body, _last_context)
            webbrowser.open(url)
    if on_done:
        on_done(result)


_last_title = ""
_last_body = ""
_last_context = ""


def show_bug_report_dialog(parent=None, exc_info: tuple = None):
    global _last_title, _last_body, _last_context

    if exc_info is not None:
        exc_type, exc_value, exc_tb = exc_info
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        default_title = f"{exc_type.__name__}: {exc_value}"
        default_body = f"```\n{tb_str}\n```"
    else:
        default_title = ""
        default_body = ""

    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QLineEdit, QPushButton

    dlg = QDialog(parent)
    dlg.setWindowTitle("Report Bug")
    dlg.setFixedSize(600, 520)
    dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

    root = QVBoxLayout(dlg)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(8)

    root.addWidget(QLabel("Title:"))
    title_edit = QLineEdit(default_title)
    title_edit.setPlaceholderText("Brief description of the issue...")
    root.addWidget(title_edit)

    root.addWidget(QLabel("Description:"))
    body_edit = QTextEdit()
    body_edit.setPlaceholderText("Steps to reproduce, expected behavior, actual behavior...")
    body_edit.setPlainText(default_body)
    root.addWidget(body_edit)

    token = _get_github_token()
    status_parts = [f""]
    if token:
        status_parts.append("Authenticated (GITHUB_TOKEN set)")
    else:
        status_parts.append("No GITHUB_TOKEN — will open GitHub in browser")
    auth_lbl = QLabel(" | ".join(status_parts))
    auth_lbl.setStyleSheet("color: #888; font-size: 10px;")
    root.addWidget(auth_lbl)

    btn_row = QHBoxLayout()
    btn_row.addStretch()
    cancel_btn = QPushButton("Cancel")
    cancel_btn.setFixedWidth(90)
    cancel_btn.clicked.connect(dlg.reject)
    btn_row.addWidget(cancel_btn)
    submit_btn = QPushButton("Submit to GitHub")
    submit_btn.setFixedWidth(140)
    btn_row.addWidget(submit_btn)
    root.addLayout(btn_row)

    def _do_submit():
        global _last_title, _last_body, _last_context
        title = title_edit.text().strip()
        body = body_edit.toPlainText().strip()
        if not title:
            QMessageBox.warning(dlg, "Error", "Title is required.")
            return
        _last_title = title
        _last_body = body
        _last_context = _collect_context()
        submit_btn.setEnabled(False)
        submit_btn.setText("Submitting...")
        _submit_async(
            dlg, title, body, _last_context,
            on_done=lambda _: dlg.accept(),
        )

    submit_btn.clicked.connect(_do_submit)
    dlg.exec()


def install_hooks():
    _prev_excepthook = sys.excepthook

    def _editor_excepthook(exc_type, exc_value, exc_tb):
        _prev_excepthook(exc_type, exc_value, exc_tb)
        _auto_report(exc_type, exc_value, exc_tb)

    sys.excepthook = _editor_excepthook

    _prev_thread_hook = threading.excepthook if hasattr(threading, 'excepthook') else None

    def _thread_excepthook(args):
        if _prev_thread_hook:
            _prev_thread_hook(args)
        _auto_report(args.exc_type, args.exc_value, args.exc_traceback)

    if hasattr(threading, 'excepthook'):
        threading.excepthook = _thread_excepthook


def _auto_report(exc_type, exc_value, exc_tb):
    if exc_type is KeyboardInterrupt or exc_type is SystemExit:
        return
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    _log_crash_file(exc_type, exc_value, tb_str)

    app = QApplication.instance()
    if app is None:
        return
    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(200, lambda: _show_auto_report_dialog(exc_type, exc_value, tb_str))
    except Exception:
        pass


def _log_crash_file(exc_type, exc_value, tb_str):
    try:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "crash_logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"crash_{ts}_{os.getpid()}.txt"
        fpath = os.path.join(log_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(f"Zarin Engine Crash Log\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Version: {APP_VERSION_DISPLAY}\n")
            f.write(f"GPU: {_get_gpu_name()}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"{exc_type.__name__}: {exc_value}\n\n")
            f.write(tb_str)
    except Exception:
        pass


def _show_auto_report_dialog(exc_type, exc_value, tb_str):
    app = QApplication.instance()
    if app is None:
        return
    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle("Unexpected Error")
    box.setText(f"{exc_type.__name__}: {exc_value}")
    box.setInformativeText("An unexpected error occurred. Would you like to report it?")
    box.setDetailedText(tb_str[-4000:] if len(tb_str) > 4000 else tb_str)
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    box.setDefaultButton(QMessageBox.StandardButton.Yes)
    box.button(QMessageBox.StandardButton.Yes).setText("Report to GitHub")
    box.button(QMessageBox.StandardButton.No).setText("Dismiss")
    result = box.exec()
    if result == QMessageBox.StandardButton.Yes:
        show_bug_report_dialog(
            exc_info=(exc_type, exc_value, None),
        )
