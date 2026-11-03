"""Import Sackmann ATP data (players + main tour matches) into PostgreSQL.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/import_atp.py

Scope for v1:
- players: atp_players.csv
- matches: atp_matches_19*.csv and atp_matches_20*.csv only (main tour).
  Futures and qual_chall files stay on disk but are NOT imported in v1,
  so Elo and form windows are tour-level only.
- rankings: NOT imported by this script yet. Its header is checked separately.

The script drops and recreates atp_players and atp_matches, so rerunning
it gives a clean import. It reads DATABASE_URL from the environment
(loaded from ~/TSA/.env by the shell commands above).
"""

import csv
import os
import sys
from datetime import date
from pathlib import Path

import psycopg


def to_int(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return int(float(value))


def to_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return float(value)


def to_date(value):
    """Convert YYYYMMDD (int or string) to a date, or None."""
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or len(value) != 8:
        return None
    try:
        return date(int(value[0:4]), int(value[4:6]), int(value[6:8]))
    except ValueError:
        return None


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = Path(os.environ.get("TSA_DATA_DIR", project_root / "data" / "raw" / "tennis_atp"))
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a", file=sys.stderr)
        return 1
    if not data_dir.is_dir():
        print(f"ERROR: data dir not found: {data_dir}", file=sys.stderr)
        return 1

    players_file = data_dir / "atp_players.csv"
    match_files = sorted(data_dir.glob("atp_matches_19*.csv")) + sorted(data_dir.glob("atp_matches_20*.csv"))
    if not players_file.exists():
        print(f"ERROR: players file not found: {players_file}", file=sys.stderr)
        return 1
    if not match_files:
        print(f"ERROR: no main tour match files found in {data_dir}", file=sys.stderr)
        return 1

    print(f"Data dir: {data_dir}")
    print(f"Match files: {len(match_files)} (first {match_files[0].name}, last {match_files[-1].name})")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS atp_matches CASCADE")
            cur.execute("DROP TABLE IF EXISTS atp_players CASCADE")
            cur.execute(
                """
                CREATE TABLE atp_players (
                    player_id   INTEGER PRIMARY KEY,
                    first_name  TEXT,
                    last_name   TEXT,
                    hand        TEXT,
                    dob         DATE,
                    ioc         TEXT,
                    height_cm   INTEGER,
                    wikidata_id TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE atp_matches (
                    tourney_id        TEXT NOT NULL,
                    tourney_name      TEXT,
                    surface           TEXT,
                    draw_size         INTEGER,
                    tourney_level     TEXT,
                    tourney_date      DATE,
                    match_num         INTEGER NOT NULL,
                    winner_id         INTEGER,
                    winner_seed       INTEGER,
                    winner_entry      TEXT,
                    winner_name       TEXT,
                    winner_hand       TEXT,
                    winner_ht         INTEGER,
                    winner_ioc        TEXT,
                    winner_age        NUMERIC(4, 1),
                    loser_id          INTEGER,
                    loser_seed        INTEGER,
                    loser_entry       TEXT,
                    loser_name        TEXT,
                    loser_hand        TEXT,
                    loser_ht          INTEGER,
                    loser_ioc         TEXT,
                    loser_age         NUMERIC(4, 1),
                    score             TEXT,
                    best_of           INTEGER,
                    round             TEXT,
                    minutes           INTEGER,
                    w_ace             INTEGER,
                    w_df              INTEGER,
                    w_svpt            INTEGER,
                    w_1st_in          INTEGER,
                    w_1st_won         INTEGER,
                    w_2nd_won         INTEGER,
                    w_sv_gms          INTEGER,
                    w_bp_saved        INTEGER,
                    w_bp_faced        INTEGER,
                    l_ace             INTEGER,
                    l_df              INTEGER,
                    l_svpt            INTEGER,
                    l_1st_in          INTEGER,
                    l_1st_won         INTEGER,
                    l_2nd_won         INTEGER,
                    l_sv_gms          INTEGER,
                    l_bp_saved        INTEGER,
                    l_bp_faced        INTEGER,
                    winner_rank       INTEGER,
                    winner_rank_points INTEGER,
                    loser_rank        INTEGER,
                    loser_rank_points INTEGER,
                    PRIMARY KEY (tourney_id, match_num)
                )
                """
            )
            cur.execute("CREATE INDEX idx_matches_winner ON atp_matches (winner_id)")
            cur.execute("CREATE INDEX idx_matches_loser ON atp_matches (loser_id)")
            cur.execute("CREATE INDEX idx_matches_date ON atp_matches (tourney_date)")
            cur.execute("CREATE INDEX idx_matches_surface ON atp_matches (surface)")

            # Players
            with open(players_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [
                    (
                        to_int(r["player_id"]),
                        r.get("name_first") or None,
                        r.get("name_last") or None,
                        r.get("hand") or None,
                        to_date(r.get("dob")),
                        r.get("ioc") or None,
                        to_int(r.get("height")),
                        r.get("wikidata_id") or None,
                    )
                    for r in reader
                ]
            cur.executemany(
                "INSERT INTO atp_players VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                rows,
            )
            print(f"Players inserted: {len(rows)}")

            # Matches, file by file, batched
            total = 0
            # 49 columns, matching the atp_matches_2025.csv header exactly.
            match_sql = (
                "INSERT INTO atp_matches VALUES ("
                + ", ".join(["%s"] * 49)
                + ") ON CONFLICT (tourney_id, match_num) DO NOTHING"
            )
            for path in match_files:
                with open(path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    batch = []
                    for r in reader:
                        batch.append(
                            (
                                r.get("tourney_id"),
                                r.get("tourney_name") or None,
                                r.get("surface") or None,
                                to_int(r.get("draw_size")),
                                r.get("tourney_level") or None,
                                to_date(r.get("tourney_date")),
                                to_int(r.get("match_num")),
                                to_int(r.get("winner_id")),
                                to_int(r.get("winner_seed")),
                                r.get("winner_entry") or None,
                                r.get("winner_name") or None,
                                r.get("winner_hand") or None,
                                to_int(r.get("winner_ht")),
                                r.get("winner_ioc") or None,
                                to_float(r.get("winner_age")),
                                to_int(r.get("loser_id")),
                                to_int(r.get("loser_seed")),
                                r.get("loser_entry") or None,
                                r.get("loser_name") or None,
                                r.get("loser_hand") or None,
                                to_int(r.get("loser_ht")),
                                r.get("loser_ioc") or None,
                                to_float(r.get("loser_age")),
                                r.get("score") or None,
                                to_int(r.get("best_of")),
                                r.get("round") or None,
                                to_int(r.get("minutes")),
                                to_int(r.get("w_ace")),
                                to_int(r.get("w_df")),
                                to_int(r.get("w_svpt")),
                                to_int(r.get("w_1stIn")),
                                to_int(r.get("w_1stWon")),
                                to_int(r.get("w_2ndWon")),
                                to_int(r.get("w_SvGms")),
                                to_int(r.get("w_bpSaved")),
                                to_int(r.get("w_bpFaced")),
                                to_int(r.get("l_ace")),
                                to_int(r.get("l_df")),
                                to_int(r.get("l_svpt")),
                                to_int(r.get("l_1stIn")),
                                to_int(r.get("l_1stWon")),
                                to_int(r.get("l_2ndWon")),
                                to_int(r.get("l_SvGms")),
                                to_int(r.get("l_bpSaved")),
                                to_int(r.get("l_bpFaced")),
                                to_int(r.get("winner_rank")),
                                to_int(r.get("winner_rank_points")),
                                to_int(r.get("loser_rank")),
                                to_int(r.get("loser_rank_points")),
                            )
                        )
                        if len(batch) >= 1000:
                            cur.executemany(match_sql, batch)
                            total += len(batch)
                            batch = []
                    if batch:
                        cur.executemany(match_sql, batch)
                        total += len(batch)
                print(f"Imported {path.name}")
            print(f"Matches inserted (attempted): {total}")
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM atp_players")
            print("DB players:", cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM atp_matches")
            print("DB matches:", cur.fetchone()[0])
            cur.execute("SELECT MIN(tourney_date), MAX(tourney_date) FROM atp_matches")
            print("Match date range (data-as-of check):", cur.fetchone())
            cur.execute("SELECT surface, COUNT(*) FROM atp_matches GROUP BY surface ORDER BY COUNT(*) DESC")
            print("Matches by surface:", cur.fetchall())
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
