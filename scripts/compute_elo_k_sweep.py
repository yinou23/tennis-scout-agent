"""Compute classic overall Elo for several K values, in one run.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    bash scripts/run_checked.sh tests/test_elo.py scripts/compute_elo_k_sweep.py

Why this script, fixed before seeing its results:
- Week 2 used K=32 without testing it. K is how much Elo moves
  per match, and it should be tested on held-out data, not
  chosen by habit. This script tests K=16, 24, 32, 40, the four
  values parked in Week 2.
- This is overall Elo only, no surface blending. Surface
  blending is tested separately, with prior=100 already adopted
  as the current best. If a K other than 32 wins here, the
  next step is to recompute surface Elo with that K and
  re-test the blend, not to assume the old surface result
  still holds at the new K.
- Each K gets its own table, atp_elo_history_k16, k24, k32, k40,
  so the backtest script can read each one without recomputing
  Elo, and so the original atp_elo_history (K=32) is left
  untouched for comparison.

Each table has one row per match, with pre-match Elo for both
players, so the backtest can compute the winner probability
from Elo alone, with no leakage.
"""

import os
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elo import INITIAL_ELO, update_elo  # noqa: E402

K_VALUES = (16, 24, 32, 40)


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a", file=sys.stderr)
        return 1

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tourney_id, match_num, tourney_date, winner_id, loser_id "
                "FROM atp_matches WHERE winner_id IS NOT NULL AND loser_id IS NOT NULL "
                "ORDER BY tourney_date, tourney_id, match_num"
            )
            matches = cur.fetchall()
        print(f"Matches to process: {len(matches)}")

        for k in K_VALUES:
            ratings = {}
            history = []
            for tourney_id, match_num, tourney_date, winner_id, loser_id in matches:
                w = ratings.get(winner_id, INITIAL_ELO)
                l = ratings.get(loser_id, INITIAL_ELO)
                w_new, l_new, expected = update_elo(w, l, k=k)
                ratings[winner_id] = w_new
                ratings[loser_id] = l_new
                history.append((tourney_id, match_num, tourney_date, winner_id, loser_id, w, l, expected))
            table = f"atp_elo_history_k{k}"
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                cur.execute(
                    f"""
                    CREATE TABLE {table} (
                        tourney_id           TEXT NOT NULL,
                        match_num            INTEGER NOT NULL,
                        tourney_date         DATE,
                        winner_id            INTEGER,
                        loser_id             INTEGER,
                        winner_elo_before    NUMERIC,
                        loser_elo_before     NUMERIC,
                        expected_winner_prob NUMERIC,
                        PRIMARY KEY (tourney_id, match_num)
                    )
                    """
                )
                for i in range(0, len(history), 5000):
                    cur.executemany(
                        f"INSERT INTO {table} VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        history[i : i + 5000],
                    )
            conn.commit()
            print(f"K={k}: DB rows in {table}: {len(history)}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
