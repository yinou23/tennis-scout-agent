"""Compute per-match games won from atp_matches.score.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/compute_games.py

Writes atp_match_games (dropped and recreated):
- winner_games, loser_games: total games per player in the match.
- winner_games_pct: winner_games / (winner_games + loser_games),
  only for complete matches, else NULL.
- is_complete: False for retirements, walkovers, missing scores,
  and unparsable scores.

This is only the per-match parsing layer. Games-% form windows
(last10 games %, last52 games %) are built on top of this table
in the next step, not in this script, so parsing errors can be
checked here first, before any window is built on them.
"""

import os
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_parser import parse_score  # noqa: E402


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a", file=sys.stderr)
        return 1

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tourney_id, match_num, score FROM atp_matches ORDER BY tourney_date, tourney_id, match_num")
            matches = cur.fetchall()
        print(f"Matches to process: {len(matches)}")

        rows = []
        complete = 0
        incomplete = 0
        failed = 0
        for tourney_id, match_num, score in matches:
            w, l, is_complete = parse_score(score)
            if w is None:
                failed += 1
                pct = None
            else:
                pct = (w / (w + l)) if is_complete and (w + l) > 0 else None
                if is_complete:
                    complete += 1
                else:
                    incomplete += 1
            rows.append((tourney_id, match_num, w, l, pct, is_complete))

        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS atp_match_games CASCADE")
            cur.execute(
                """
                CREATE TABLE atp_match_games (
                    tourney_id       TEXT NOT NULL,
                    match_num        INTEGER NOT NULL,
                    winner_games     INTEGER,
                    loser_games      INTEGER,
                    winner_games_pct NUMERIC,
                    is_complete      BOOLEAN,
                    PRIMARY KEY (tourney_id, match_num)
                )
                """
            )
            for i in range(0, len(rows), 5000):
                cur.executemany(
                    "INSERT INTO atp_match_games VALUES (%s, %s, %s, %s, %s, %s)",
                    rows[i : i + 5000],
                )
        conn.commit()

        print(f"Complete parsed matches: {complete}")
        print(f"Incomplete matches (retirement etc., games counted but pct NULL): {incomplete}")
        print(f"Missing or unparsable scores: {failed}")
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM atp_match_games")
            print("DB match games rows:", cur.fetchone()[0])
            cur.execute("SELECT ROUND(AVG(winner_games_pct), 4) FROM atp_match_games WHERE is_complete")
            print("Average winner games pct (complete matches only):", cur.fetchone()[0])
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
