# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from PyQt6.QtCore import QCoreApplication, QObject, Qt, pyqtSignal


class _MainThreadDispatcher(QObject):
    request = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.request.connect(self._execute, Qt.ConnectionType.QueuedConnection)

    def _execute(self, fn):
        try:
            fn()
        except BaseException:
            pass


_dispatcher: Optional[_MainThreadDispatcher] = None


def ensure_dispatcher() -> _MainThreadDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = _MainThreadDispatcher()
    return _dispatcher


def run_on_main_thread(fn: Callable[[], Any], timeout_ms: int = 60000) -> Any:
    app = QCoreApplication.instance()
    if app is None or threading.current_thread() is app.thread():
        return fn()
    dispatcher = ensure_dispatcher()
    done = threading.Event()
    result: dict = {}

    def wrapper():
        try:
            result["value"] = fn()
        except BaseException as e:
            result["error"] = e
        finally:
            done.set()

    dispatcher.request.emit(wrapper)
    if not done.wait(timeout_ms / 1000.0):
        raise TimeoutError(f"Main thread did not respond within {timeout_ms} ms")
    if "error" in result:
        raise result["error"]
    return result.get("value")
