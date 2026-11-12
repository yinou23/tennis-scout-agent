"""TSA Step 2 doctor check.

Prints a checklist, never prints secrets. The database is a local
SQLite file, SQLITE_PATH or data/tsa.db under the repo root.
Run it from the project root: python scripts/check_env.py
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def mark(ok):
    return "[OK] " if ok else "[MISSING] "


def check_command(name):
    path = shutil.which(name)
    version = ""
    if path:
        try:
            out = subprocess.run(
                [name, "--version"], capture_output=True, text=True, timeout=10
            )
            version = (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr).strip() else ""
        except Exception:
            version = ""
    print(f"{mark(bool(path))}{name}: {path or 'not found'} {version}")
    return bool(path)


def check_import(name):
    try:
        __import__(name)
        print(f"[OK] python import: {name}")
        return True
    except Exception as exc:
        print(f"[MISSING] python import: {name} ({type(exc).__name__})")
        return False


def main():
    print("=== TSA Step 2 doctor ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    print("--- commands ---")
    has_python312 = check_command("python3.12")
    check_command("node")
    check_command("docker")
    check_command("git")
    print("--- python packages ---")
    check_import("fastapi")
    check_import("langgraph")
    check_import("sqlite3")
    print("--- env and database ---")
    env_exists = os.path.exists(".env")
    print(f"{mark(env_exists)}.env file: {'found' if env_exists else 'not found, copy .env.example to .env'}")
    from db import get_connection, get_sqlite_path

    db_path = get_sqlite_path()
    if db_path.exists():
        print(f"[OK] SQLite database file: {db_path}")
    else:
        print(f"[MISSING] SQLite database file not found: {db_path} (set SQLITE_PATH to the tsa.db location)")
    db_ok = False
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT sqlite_version();")
            row = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM atp_players;")
            players = cur.fetchone()[0]
        print(f"[OK] database connection: SQLite {row[0] if row else 'connected'}, {players} players in atp_players")
        db_ok = True
    except Exception as exc:
        print(f"[MISSING] database connection: {type(exc).__name__} (check SQLITE_PATH, is data/tsa.db present?)")
    print("--- summary ---")
    if has_python312 and db_ok:
        print("Core Step 2 checks look good. Next: run the FastAPI smoke test in README-STEP2.md.")
    else:
        print("Some checks are missing. Paste this whole output to Muse for one complete fix list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
