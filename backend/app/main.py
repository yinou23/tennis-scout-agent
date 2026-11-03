"""TSA Step 2 smoke test only.

Two endpoints, no business logic yet:
- GET /health returns ok
- GET /db tries one SELECT version() against DATABASE_URL

The /db endpoint never returns the database URL or password,
only ok plus the Postgres version, or an error type plus a hint.
"""

import os

from fastapi import FastAPI

app = FastAPI(title="TSA Step 2 smoke")


@app.get("/health")
def health():
    return {"status": "ok", "step": "step2"}


@app.get("/db")
def db():
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        return {
            "status": "error",
            "detail": "DATABASE_URL is not set. Copy .env.example to .env and export it in this shell.",
        }
    try:
        import psycopg
    except Exception as exc:
        return {
            "status": "error",
            "detail": f"psycopg is not installed: {type(exc).__name__}",
        }
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                row = cur.fetchone()
        return {"status": "ok", "postgres": row[0] if row else None}
    except Exception as exc:
        return {
            "status": "error",
            "detail": f"{type(exc).__name__}. Check: docker compose ps, is tsa-db running?",
        }
