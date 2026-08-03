"""
SQLite connection management for Pratik AI.

A single module-level connection (with WAL mode + thread-safety settings)
is shared across the memory and conversation stores so we don't open a new
file handle per request.
"""

from __future__ import annotations

import sqlite3
import threading

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_connection: sqlite3.Connection | None = None


def get_connection() -> sqlite3.Connection:
    """Return a process-wide SQLite connection, creating it on first use."""
    global _connection
    if _connection is None:
        with _lock:
            if _connection is None:
                logger.info("Opening SQLite database at %s", config.SQLITE_DB_PATH)
                _connection = sqlite3.connect(
                    str(config.SQLITE_DB_PATH),
                    check_same_thread=False,
                )
                _connection.row_factory = sqlite3.Row
                _connection.execute("PRAGMA journal_mode=WAL;")
                _connection.execute("PRAGMA foreign_keys=ON;")
    return _connection


def close_connection() -> None:
    """Close the shared connection (used on app shutdown / in tests)."""
    global _connection
    with _lock:
        if _connection is not None:
            _connection.close()
            _connection = None
