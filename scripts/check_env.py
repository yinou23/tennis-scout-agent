"""TSA Step 2 doctor check.

Prints a checklist, never prints secrets. If DATABASE_URL is set,
only host, port, user and database are shown, the password is masked.
Run it from the project root: python scripts/check_env.py
"""

import os
import platform
import shutil
import subprocess
import sys
from urllib.parse import urlparse


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
    check_import("psycopg")
    print("--- env and database ---")
    env_exists = os.path.exists(".env")
    print(f"{mark(env_exists)}.env file: {'found' if env_exists else 'not found, copy .env.example to .env'}")
    url = os.environ.get("DATABASE_URL", "")
    if url:
        parsed = urlparse(url)
        print(f"[OK] DATABASE_URL is set: host={parsed.hostname} port={parsed.port} user={parsed.username} db={parsed.path.lstrip('/')}")
    else:
        print("[MISSING] DATABASE_URL is not set in this shell. If you use .env, load it first or check the FastAPI /db endpoint after starting uvicorn.")
        print("  Hint: the backend reads DATABASE_URL from the environment, not automatically from .env in Step 2.")
    db_ok = False
    if url:
        try:
            import psycopg

            with psycopg.connect(url, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    row = cur.fetchone()
            print(f"[OK] database connection: {row[0].split(',')[0] if row else 'connected'}")
            db_ok = True
        except Exception as exc:
            print(f"[MISSING] database connection: {type(exc).__name__} (check: docker compose ps, is tsa-db running?)")
    print("--- summary ---")
    if has_python312 and db_ok:
        print("Core Step 2 checks look good. Next: run the FastAPI smoke test in README-STEP2.md.")
    else:
        print("Some checks are missing. Paste this whole output to Muse for one complete fix list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
