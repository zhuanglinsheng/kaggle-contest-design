"""Disposable solver memoization, separate from scientific result records.

JK9_CACHE_MODE=sqlite (default), memory, off, or files (legacy diagnostics).
SQLite stores many evaluations in one file per cache directory. Failed solves
are cached verbatim too; storage never changes their scientific interpretation.
Old per-point JSON files remain readable without migration or deletion.
"""
from collections import OrderedDict
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import zlib

NAMES = {'coarse_cache', 'fine_cache', 'point_cache', 'fine_recovery',
         'cross_coordinate_recovery', 'evaluations'}
_memory = OrderedDict()
_lock = threading.RLock()
_limit = 2048


def mode():
    value = os.environ.get('JK9_CACHE_MODE', 'sqlite')
    if value not in {'sqlite', 'memory', 'off', 'files'}:
        raise ValueError(f'Unknown JK9_CACHE_MODE: {value}')
    return value


def is_cache(path):
    return Path(path).parent.name in NAMES


def get(path):
    path = Path(path)
    selected = mode()
    if selected == 'off':
        return None
    if selected == 'memory':
        with _lock:
            payload = _memory.get(str(path.resolve()))
        return None if payload is None else json.loads(payload)
    # Never silently override a retained historical point evaluation.
    if path.is_file():
        return json.loads(path.read_text())
    if selected == 'files':
        return None
    database = path.parent / 'cache.sqlite3'
    if not database.exists():
        return None
    with closing(sqlite3.connect(f'{database.resolve().as_uri()}?mode=ro', uri=True, timeout=60)) as connection:
        # Another process may have created the DB but not its first table yet.
        table = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='evaluations'").fetchone()
        if table is None:
            return None
        row = connection.execute('SELECT payload, sha256 FROM evaluations WHERE key=?',
                                 (path.name,)).fetchone()
    if row is None:
        return None
    payload = zlib.decompress(row[0])
    if hashlib.sha256(payload).hexdigest() != row[1]:
        raise ValueError(f'Corrupt solver cache entry: {path}')
    return json.loads(payload)


def put(path, value):
    path = Path(path)
    selected = mode()
    if selected == 'off':
        return
    payload = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
    if selected == 'memory':
        with _lock:
            key = str(path.resolve())
            _memory[key] = payload
            _memory.move_to_end(key)
            while len(_memory) > _limit:
                _memory.popitem(last=False)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if selected == 'files':
        temporary = path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
        temporary.write_bytes(payload)
        temporary.replace(path)
        return
    with closing(sqlite3.connect(path.parent / 'cache.sqlite3', timeout=60)) as connection:
        connection.execute('CREATE TABLE IF NOT EXISTS evaluations '
                           '(key TEXT PRIMARY KEY, payload BLOB NOT NULL, sha256 TEXT NOT NULL)')
        connection.execute('INSERT OR REPLACE INTO evaluations VALUES (?,?,?)',
                           (path.name, zlib.compress(payload, 1), hashlib.sha256(payload).hexdigest()))
        connection.commit()


def clear_memory():
    with _lock:
        _memory.clear()
