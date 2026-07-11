# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
import traceback
from enum import IntEnum
from typing import Callable, Optional
from core.config.constants import LOGGER_MAX_ENTRIES
class LogLevel(IntEnum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
class LogEntry:
    __slots__ = ("level", "message", "timestamp", "traceback_str")
    def __init__(self, level: LogLevel, message: str, tb: str = ""):
        self.level = level
        self.message = message
        self.timestamp = time.time()
        self.traceback_str = tb
    def formatted_time(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))
    def level_str(self) -> str:
        return LogLevel(self.level).name
class Logger:
    _entries: list[LogEntry] = []
    _listeners: list[Callable[[LogEntry], None]] = []
    _max_entries: int = LOGGER_MAX_ENTRIES
    _min_level: LogLevel = LogLevel.DEBUG
    @classmethod
    def add_listener(cls, fn: Callable[[LogEntry], None]):
        cls._listeners.append(fn)
    @classmethod
    def remove_listener(cls, fn: Callable[[LogEntry], None]):
        cls._listeners.remove(fn)
    @classmethod
    def _emit(cls, entry: LogEntry):
        cls._entries.append(entry)
        if len(cls._entries) > cls._max_entries:
            cls._entries.pop(0)
        for fn in cls._listeners:
            try: fn(entry)
            except: pass
    @classmethod
    def debug(cls, msg: str):
        if cls._min_level <= LogLevel.DEBUG:
            cls._emit(LogEntry(LogLevel.DEBUG, str(msg)))
    @classmethod
    def info(cls, msg: str):
        if cls._min_level <= LogLevel.INFO:
            cls._emit(LogEntry(LogLevel.INFO, str(msg)))
    @classmethod
    def warning(cls, msg: str):
        if cls._min_level <= LogLevel.WARNING:
            cls._emit(LogEntry(LogLevel.WARNING, str(msg)))
    @classmethod
    def error(cls, msg: str, exc: Optional[Exception] = None):
        tb = traceback.format_exc() if exc else ""
        cls._emit(LogEntry(LogLevel.ERROR, str(msg), tb))
    @classmethod
    def get_entries(cls) -> list[LogEntry]:
        return list(cls._entries)
    @classmethod
    def clear(cls):
        cls._entries.clear()
