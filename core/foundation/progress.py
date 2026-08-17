# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import multiprocessing
import threading
import time

_tasks: dict[str, dict] = {}
_notifications: list[dict] = []
_completed: list[dict] = []
_lock = threading.Lock()

_PROGRESS_QUEUE: multiprocessing.managers.SyncManager.Queue | None = None
_reader_thread: threading.Thread | None = None


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
        task = _tasks.pop(task_id, None)
        if task is not None:
            _completed.append({
                "title": task["title"],
                "duration": time.monotonic() - task["started"],
                "time": time.monotonic(),
            })
            del _completed[:-8]


def clear() -> None:
    with _lock:
        _tasks.clear()
        _notifications.clear()
        _completed.clear()


def snapshot_completed() -> list[dict]:
    with _lock:
        return list(_completed)


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


def _reader_loop(queue: multiprocessing.Queue) -> None:
    _FN = {
        "start": task_start,
        "update": task_update,
        "set_title": task_set_title,
        "set_detail": task_set_detail,
        "complete": task_complete,
    }
    while True:
        try:
            msg = queue.get()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        fn = _FN.get(msg[0])
        if fn is not None:
            try:
                fn(*msg[1], **msg[2])
            except Exception:
                pass


def get_progress_queue():
    global _PROGRESS_QUEUE, _reader_thread
    if _PROGRESS_QUEUE is None:
        mgr = multiprocessing.Manager()
        _PROGRESS_QUEUE = mgr.Queue()
        _reader_thread = threading.Thread(target=_reader_loop, args=(_PROGRESS_QUEUE,),
                                          daemon=True, name="bvh-progress-reader")
        _reader_thread.start()
    return _PROGRESS_QUEUE
