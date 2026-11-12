"""SQLite connection helper for the TSA runtime.

Every runtime script that needs the database opens it through
get_connection() here, instead of opening its own Postgres
connection from a database URL. The path comes from the SQLITE_PATH
environment variable, defaulting to data/tsa.db under the repo
root (the directory above this file's directory).

Rows are plain tuples, the same as the old driver's default: no
row_factory is set, because every caller in the runtime path
reads rows positionally (row[0], tuple unpacking). Do not add
one without checking all callers first.
"""

import os
import sqlite3
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "tsa.db"


def get_sqlite_path():
    """Return the SQLite database path: SQLITE_PATH, or data/tsa.db."""
    raw = os.environ.get("SQLITE_PATH")
    if raw:
        return Path(raw)
    return DEFAULT_PATH


def get_connection():
    """Open a sqlite3 connection to the TSA database.

    Raises sqlite3.OperationalError if the file cannot be opened
    (for example, the data/ directory does not exist). Callers
    that must not crash on a missing database catch that.
    """
    return sqlite3.connect(get_sqlite_path())
