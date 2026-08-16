# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import threading
import time

_tasks: dict[str, dict] = {}
_notifications: list[dict] = []
_lock = threading.Lock()


def task_start(task_id: str, title: str, fraction: float | None = None,
               total: float | None = None, units: str | None = None) -> None:
    with _lock:
        _tasks[task_id] = {
            "title": title,
            "fraction": fraction,
            "detail": None,
            "total": total,
            "units": units,
            "started": time.monotonic(),
        }


def task_update(task_id: str, fraction: float | None = None, detail: str | None = None,
                total: float | None = None, units: str | None = None) -> None:
    with _lock:
        task = _tasks.get(task_id)
        if task is not None:
            if fraction is not None:
                task["fraction"] = fraction
            if detail is not None:
                task["detail"] = detail
            if total is not None:
                task["total"] = total
            if units is not None:
                task["units"] = units


def task_set_title(task_id: str, title: str) -> None:
    with _lock:
        task = _tasks.get(task_id)
        if task is not None:
            task["title"] = title


def task_set_detail(task_id: str, detail: str | None) -> None:
    with _lock:
        task = _tasks.get(task_id)
        if task is not None:
            task["detail"] = detail


def task_complete(task_id: str) -> None:
    with _lock:
        _tasks.pop(task_id, None)


def clear() -> None:
    with _lock:
        _tasks.clear()
        _notifications.clear()


def notify_error(message: str) -> None:
    with _lock:
        _notifications.append({"message": message, "time": time.monotonic()})
        del _notifications[:-8]


def snapshot_notifications() -> list[dict]:
    with _lock:
        return list(_notifications)


def clear_notifications() -> None:
    with _lock:
        _notifications.clear()


def snapshot() -> list[dict]:
    with _lock:
        items = sorted(_tasks.items(), key=lambda kv: kv[1]["started"])
        return [
            {
                "id": tid,
                "title": t["title"],
                "fraction": t["fraction"],
                "detail": t["detail"],
                "total": t["total"],
                "units": t["units"],
                "started": t["started"],
            }
            for tid, t in items
        ]
