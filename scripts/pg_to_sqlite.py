#!/usr/bin/env python3
"""Dump the TSA Postgres database to a SQLite file.

Reads the 10 live tables from Postgres (psycopg) and writes them to
SQLite (sqlite3), recreating each table's schema (same column names,
compatible types), its PRIMARY KEY, and its main lookup indexes,
then copying every row in batches.

Tables copied (atp_rankings is intentionally NOT copied: it was
dropped from Postgres on 2026-11-11 and replaced by
atp_year_end_rankings + atp_player_ranking_summary):
    atp_players, atp_matches, atp_elo_history, atp_elo_current,
    atp_surface_elo_history, atp_form_history, atp_games_form_history,
    atp_match_games, atp_year_end_rankings, atp_player_ranking_summary

Type handling:
    Postgres DATE    -> SQLite DATE column, stored as ISO text 'YYYY-MM-DD'
    Postgres NUMERIC -> SQLite REAL (float)
    Postgres BOOLEAN -> SQLite INTEGER (0/1)
    integer/text     -> INTEGER/TEXT unchanged

Connection: standard PG env vars. PGPASSWORD is required; on the VM
load it from ~/workspace/tsa-vm/.env (key TSA_PASSWORD) first:
    set -a; . ~/workspace/tsa-vm/.env; set +a
    export PGPASSWORD="$TSA_PASSWORD"

Usage (run from the repo root, with the repo venv):
    .venv/bin/python scripts/pg_to_sqlite.py [--out data/tsa.db]

Defaults: host 127.0.0.1, port 5432, database tsa, user tsa,
output <repo root>/data/tsa.db. Safe to re-run: existing tables are
dropped and rebuilt, and per-table row counts are verified against
Postgres before the script exits successfully.
"""

import argparse
import datetime
import decimal
import os
import sqlite3
import sys
from pathlib import Path

import psycopg

BATCH_SIZE = 5000

# (column name, SQLite type). Column order matches Postgres.
# PRIMARY KEYs and indexes below mirror Postgres exactly.
TABLES = {
    "atp_players": {
        "columns": [
            ("player_id", "INTEGER"),
            ("first_name", "TEXT"),
            ("last_name", "TEXT"),
            ("hand", "TEXT"),
            ("dob", "DATE"),
            ("ioc", "TEXT"),
            ("height_cm", "INTEGER"),
            ("wikidata_id", "TEXT"),
        ],
        "pk": ["player_id"],
        "indexes": [],
    },
    "atp_matches": {
        "columns": [
            ("tourney_id", "TEXT"),
            ("tourney_name", "TEXT"),
            ("surface", "TEXT"),
            ("draw_size", "INTEGER"),
            ("tourney_level", "TEXT"),
            ("tourney_date", "DATE"),
            ("match_num", "INTEGER"),
            ("winner_id", "INTEGER"),
            ("winner_seed", "INTEGER"),
            ("winner_entry", "TEXT"),
            ("winner_name", "TEXT"),
            ("winner_hand", "TEXT"),
            ("winner_ht", "INTEGER"),
            ("winner_ioc", "TEXT"),
            ("winner_age", "REAL"),
            ("loser_id", "INTEGER"),
            ("loser_seed", "INTEGER"),
            ("loser_entry", "TEXT"),
            ("loser_name", "TEXT"),
            ("loser_hand", "TEXT"),
            ("loser_ht", "INTEGER"),
            ("loser_ioc", "TEXT"),
            ("loser_age", "REAL"),
            ("score", "TEXT"),
            ("best_of", "INTEGER"),
            ("round", "TEXT"),
            ("minutes", "INTEGER"),
            ("w_ace", "INTEGER"),
            ("w_df", "INTEGER"),
            ("w_svpt", "INTEGER"),
            ("w_1st_in", "INTEGER"),
            ("w_1st_won", "INTEGER"),
            ("w_2nd_won", "INTEGER"),
            ("w_sv_gms", "INTEGER"),
            ("w_bp_saved", "INTEGER"),
            ("w_bp_faced", "INTEGER"),
            ("l_ace", "INTEGER"),
            ("l_df", "INTEGER"),
            ("l_svpt", "INTEGER"),
            ("l_1st_in", "INTEGER"),
            ("l_1st_won", "INTEGER"),
            ("l_2nd_won", "INTEGER"),
            ("l_sv_gms", "INTEGER"),
            ("l_bp_saved", "INTEGER"),
            ("l_bp_faced", "INTEGER"),
            ("winner_rank", "INTEGER"),
            ("winner_rank_points", "INTEGER"),
            ("loser_rank", "INTEGER"),
            ("loser_rank_points", "INTEGER"),
        ],
        "pk": ["tourney_id", "match_num"],
        "indexes": [
            ("idx_matches_date", ["tourney_date"]),
            ("idx_matches_winner", ["winner_id"]),
            ("idx_matches_loser", ["loser_id"]),
            ("idx_matches_surface", ["surface"]),
        ],
    },
    "atp_elo_history": {
        "columns": [
            ("tourney_id", "TEXT"),
            ("match_num", "INTEGER"),
            ("tourney_date", "DATE"),
            ("winner_id", "INTEGER"),
            ("loser_id", "INTEGER"),
            ("winner_elo_before", "REAL"),
            ("loser_elo_before", "REAL"),
            ("winner_elo_after", "REAL"),
            ("loser_elo_after", "REAL"),
            ("expected_winner_prob", "REAL"),
        ],
        "pk": ["tourney_id", "match_num"],
        "indexes": [
            ("idx_elo_history_date", ["tourney_date"]),
            ("idx_elo_history_winner", ["winner_id"]),
            ("idx_elo_history_loser", ["loser_id"]),
        ],
    },
    "atp_elo_current": {
        "columns": [
            ("player_id", "INTEGER"),
            ("elo", "REAL"),
            ("matches_played", "INTEGER"),
            ("last_match_date", "DATE"),
        ],
        "pk": ["player_id"],
        "indexes": [],
    },
    "atp_surface_elo_history": {
        "columns": [
            ("tourney_id", "TEXT"),
            ("match_num", "INTEGER"),
            ("tourney_date", "DATE"),
            ("winner_id", "INTEGER"),
            ("loser_id", "INTEGER"),
            ("surface", "TEXT"),
            ("winner_overall_before", "REAL"),
            ("loser_overall_before", "REAL"),
            ("winner_surface_before", "REAL"),
            ("loser_surface_before", "REAL"),
            ("winner_surface_n_before", "INTEGER"),
            ("loser_surface_n_before", "INTEGER"),
            ("winner_surface_after", "REAL"),
            ("loser_surface_after", "REAL"),
        ],
        "pk": ["tourney_id", "match_num"],
        "indexes": [],
    },
    "atp_form_history": {
        "columns": [
            ("tourney_id", "TEXT"),
            ("match_num", "INTEGER"),
            ("tourney_date", "DATE"),
            ("winner_id", "INTEGER"),
            ("loser_id", "INTEGER"),
            ("winner_last10_pct", "REAL"),
            ("winner_last10_n", "INTEGER"),
            ("winner_last52_pct", "REAL"),
            ("winner_last52_n", "INTEGER"),
            ("loser_last10_pct", "REAL"),
            ("loser_last10_n", "INTEGER"),
            ("loser_last52_pct", "REAL"),
            ("loser_last52_n", "INTEGER"),
        ],
        "pk": ["tourney_id", "match_num"],
        "indexes": [
            ("idx_form_winner", ["winner_id"]),
            ("idx_form_loser", ["loser_id"]),
        ],
    },
    "atp_games_form_history": {
        "columns": [
            ("tourney_id", "TEXT"),
            ("match_num", "INTEGER"),
            ("tourney_date", "DATE"),
            ("winner_id", "INTEGER"),
            ("loser_id", "INTEGER"),
            ("winner_last10_games_pct", "REAL"),
            ("winner_last10_matches", "INTEGER"),
            ("winner_last10_games", "INTEGER"),
            ("winner_last52_games_pct", "REAL"),
            ("winner_last52_matches", "INTEGER"),
            ("winner_last52_games", "INTEGER"),
            ("loser_last10_games_pct", "REAL"),
            ("loser_last10_matches", "INTEGER"),
            ("loser_last10_games", "INTEGER"),
            ("loser_last52_games_pct", "REAL"),
            ("loser_last52_matches", "INTEGER"),
            ("loser_last52_games", "INTEGER"),
        ],
        "pk": ["tourney_id", "match_num"],
        "indexes": [],
    },
    "atp_match_games": {
        "columns": [
            ("tourney_id", "TEXT"),
            ("match_num", "INTEGER"),
            ("winner_games", "INTEGER"),
            ("loser_games", "INTEGER"),
            ("winner_games_pct", "REAL"),
            ("is_complete", "INTEGER"),
        ],
        "pk": ["tourney_id", "match_num"],
        "indexes": [],
    },
    "atp_year_end_rankings": {
        "columns": [
            ("player_id", "INTEGER"),
            ("year", "INTEGER"),
            ("ranking", "INTEGER"),
            ("points", "INTEGER"),
            ("ranking_date", "DATE"),
        ],
        "pk": ["player_id", "year"],
        "indexes": [
            ("idx_year_end_rank", ["year", "ranking"]),
        ],
    },
    "atp_player_ranking_summary": {
        "columns": [
            ("player_id", "INTEGER"),
            ("best_ranking", "INTEGER"),
            ("best_points", "INTEGER"),
            ("best_date", "DATE"),
            ("latest_ranking", "INTEGER"),
            ("latest_points", "INTEGER"),
            ("latest_date", "DATE"),
        ],
        "pk": ["player_id"],
        "indexes": [
            ("idx_summary_best", ["best_ranking"]),
        ],
    },
}


def convert_value(value):
    """Convert one psycopg value to a SQLite-storable value."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()[:10]
    if isinstance(value, decimal.Decimal):
        return float(value)
    return value


def pg_columns(conn, table):
    """Column names of a Postgres table, in ordinal order."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s "
            "ORDER BY ordinal_position",
            (table,),
        )
        return [row[0] for row in cur.fetchall()]


def create_sqlite_table(sqlite_conn, table, spec):
    cols_sql = ", ".join(f'"{name}" {ctype}' for name, ctype in spec["columns"])
    pk_sql = ", ".join(f'"{c}"' for c in spec["pk"])
    sqlite_conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    sqlite_conn.execute(
        f'CREATE TABLE "{table}" ({cols_sql}, PRIMARY KEY ({pk_sql}))'
    )
    for index_name, index_cols in spec["indexes"]:
        cols = ", ".join(f'"{c}"' for c in index_cols)
        sqlite_conn.execute(
            f'CREATE INDEX "{index_name}" ON "{table}" ({cols})'
        )


def copy_table(pg_conn, sqlite_conn, table, spec):
    expected = [name for name, _ in spec["columns"]]
    actual = pg_columns(pg_conn, table)
    if actual != expected:
        raise SystemExit(
            f"SCHEMA MISMATCH on {table}: Postgres has {actual}, "
            f"script expects {expected}. Refusing to copy."
        )
    col_list = ", ".join(f'"{c}"' for c in expected)
    placeholders = ", ".join("?" for _ in expected)
    insert_sql = (
        f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})'
    )
    # Server-side cursor streams rows instead of loading the table.
    with pg_conn.cursor(name=f"dump_{table}") as cur:
        cur.execute(f'SELECT {col_list} FROM "{table}"')
        copied = 0
        while True:
            rows = cur.fetchmany(BATCH_SIZE)
            if not rows:
                break
            sqlite_conn.executemany(
                insert_sql, [[convert_value(v) for v in row] for row in rows]
            )
            copied += len(rows)
    sqlite_conn.commit()
    with pg_conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{table}"')
        pg_count = cur.fetchone()[0]
    sqlite_count = sqlite_conn.execute(
        f'SELECT count(*) FROM "{table}"'
    ).fetchone()[0]
    status = "OK" if pg_count == sqlite_count == copied else "MISMATCH"
    print(f"{table:32s} pg={pg_count:>7d} sqlite={sqlite_count:>7d} {status}")
    if status != "OK":
        raise SystemExit(f"Row-count mismatch on {table}; aborting.")
    return copied


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(repo_root / "data" / "tsa.db"),
        help="SQLite output path (default: <repo root>/data/tsa.db)",
    )
    parser.add_argument("--host", default=os.environ.get("PGHOST", "127.0.0.1"))
    parser.add_argument("--port", default=os.environ.get("PGPORT", "5432"))
    parser.add_argument("--dbname", default=os.environ.get("PGDATABASE", "tsa"))
    parser.add_argument("--user", default=os.environ.get("PGUSER", "tsa"))
    args = parser.parse_args()

    if not os.environ.get("PGPASSWORD"):
        raise SystemExit(
            "PGPASSWORD is not set. Load it first: "
            "set -a; . ~/workspace/tsa-vm/.env; set +a; "
            'export PGPASSWORD="$TSA_PASSWORD"'
        )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=os.environ["PGPASSWORD"],
    ) as pg_conn:
        # atp_rankings must be gone (dropped 2026-11-11); fail loudly
        # if it ever comes back, so nobody silently copies a stale table.
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' "
                "AND table_name = 'atp_rankings'"
            )
            if cur.fetchone():
                raise SystemExit(
                    "atp_rankings exists in Postgres again; "
                    "decide explicitly whether to copy it."
                )
        if out_path.exists():
            out_path.unlink()
        sqlite_conn = sqlite3.connect(out_path)
        try:
            sqlite_conn.execute("PRAGMA journal_mode = DELETE")
            sqlite_conn.execute("PRAGMA synchronous = NORMAL")
            total = 0
            for table, spec in TABLES.items():
                create_sqlite_table(sqlite_conn, table, spec)
                total += copy_table(pg_conn, sqlite_conn, table, spec)
            sqlite_conn.execute("ANALYZE")
            sqlite_conn.commit()
        finally:
            sqlite_conn.close()
    print(f"Total rows copied: {total}")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
