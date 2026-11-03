"""Compute surface Elo with K=24 (not K=32), for the K24 re-test.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    bash scripts/run_checked.sh tests/test_surface_elo.py scripts/compute_surface_elo_k24.py

Why this script exists:
- The K sweep found K=24 beats K=32 for overall Elo alone.
  That result alone does not prove K=24 plus surface blending
  beats K=32 plus surface blending, because surface Elo values
  themselves change when K changes.
- This script recomputes overall and surface Elo with K=24,
  writing atp_surface_elo_history_k24, leaving the original
  K=32 surface table untouched for comparison.
- The backtest script for this table then tests, with the
  same prior=100 blend already adopted at K=32, whether the
  K=24 blend still helps, and whether it beats the K=32 blend.
"""

import os
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elo import INITIAL_ELO, update_elo  # noqa: E402

K = 24


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a", file=sys.stderr)
        return 1

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tourney_id, match_num, tourney_date, winner_id, loser_id, surface "
                "FROM atp_matches WHERE winner_id IS NOT NULL AND loser_id IS NOT NULL "
                "ORDER BY tourney_date, tourney_id, match_num"
            )
            matches = cur.fetchall()
        print(f"Matches to process: {len(matches)}")

        overall = {}
        surface_rating = {}
        surface_n = {}
        rows = []
        missing_surface = 0
        for tourney_id, match_num, tourney_date, winner_id, loser_id, surface in matches:
            w_over = overall.get(winner_id, INITIAL_ELO)
            l_over = overall.get(loser_id, INITIAL_ELO)
            w_over_new, l_over_new, _ = update_elo(w_over, l_over, k=K)
            overall[winner_id] = w_over_new
            overall[loser_id] = l_over_new

            if surface is None:
                missing_surface += 1
                rows.append((tourney_id, match_num, tourney_date, winner_id, loser_id, None,
                             w_over, l_over, None, None, 0, 0, None, None))
                continue
            w_key = (winner_id, surface)
            l_key = (loser_id, surface)
            w_surf = surface_rating.get(w_key, INITIAL_ELO)
            l_surf = surface_rating.get(l_key, INITIAL_ELO)
            w_n = surface_n.get(w_key, 0)
            l_n = surface_n.get(l_key, 0)
            w_surf_new, l_surf_new, _ = update_elo(w_surf, l_surf, k=K)
            surface_rating[w_key] = w_surf_new
            surface_rating[l_key] = l_surf_new
            surface_n[w_key] = w_n + 1
            surface_n[l_key] = l_n + 1
            rows.append((tourney_id, match_num, tourney_date, winner_id, loser_id, surface,
                         w_over, l_over, w_surf, l_surf, w_n, l_n, w_surf_new, l_surf_new))

        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS atp_surface_elo_history_k24 CASCADE")
            cur.execute(
                """
                CREATE TABLE atp_surface_elo_history_k24 (
                    tourney_id               TEXT NOT NULL,
                    match_num                INTEGER NOT NULL,
                    tourney_date             DATE,
                    winner_id                INTEGER,
                    loser_id                 INTEGER,
                    surface                  TEXT,
                    winner_overall_before    NUMERIC,
                    loser_overall_before     NUMERIC,
                    winner_surface_before    NUMERIC,
                    loser_surface_before     NUMERIC,
                    winner_surface_n_before  INTEGER,
                    loser_surface_n_before   INTEGER,
                    winner_surface_after     NUMERIC,
                    loser_surface_after      NUMERIC,
                    PRIMARY KEY (tourney_id, match_num)
                )
                """
            )
            for i in range(0, len(rows), 5000):
                cur.executemany(
                    "INSERT INTO atp_surface_elo_history_k24 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    rows[i : i + 5000],
                )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM atp_surface_elo_history_k24")
            print("DB surface elo history k24 rows:", cur.fetchone()[0])
            print("Matches with missing surface (overall only):", missing_surface)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
