# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtCore import QObject, pyqtSignal


class _MainThreadDispatcher(QObject):
    invoke = pyqtSignal(object)


_dispatcher = _MainThreadDispatcher()
_dispatcher.invoke.connect(lambda fn: fn())


def call_on_main(fn: Callable[[], Any]) -> None:
    """Queue ``fn`` for execution on the Qt main/GUI thread.

    Safe to call from any thread. The signal lives on a QObject created in
    the main thread, so emission is delivered there via a queued connection.
    """
    _dispatcher.invoke.emit(fn)