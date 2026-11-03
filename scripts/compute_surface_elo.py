"""Compute surface Elo from atp_matches.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    bash scripts/run_checked.sh tests/test_surface_elo.py scripts/compute_surface_elo.py

Reads atp_matches in chronological order (tourney_date, tourney_id,
match_num). Maintains overall Elo (K=32, same as Week 2) and, per
player per surface, a surface Elo (K=32, updated only on that
surface). The overall Elo recomputed here should match
atp_elo_history; the script prints a check count for that.

Writes atp_surface_elo_history (dropped and recreated):
- surface, winner/loser overall Elo before, winner/loser surface
  Elo before (1500 if no prior match on that surface),
  winner/loser surface matches before, winner/loser surface Elo
  after. The backtest script then blends surface and overall
  for any prior value without recomputing Elo.
- Matches with a missing surface have NULL surface Elo fields
  and do not update any surface rating.
"""

import os
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elo import INITIAL_ELO, DEFAULT_K, update_elo  # noqa: E402


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
            w_over_new, l_over_new, _ = update_elo(w_over, l_over, k=DEFAULT_K)
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
            w_surf_new, l_surf_new, _ = update_elo(w_surf, l_surf, k=DEFAULT_K)
            surface_rating[w_key] = w_surf_new
            surface_rating[l_key] = l_surf_new
            surface_n[w_key] = w_n + 1
            surface_n[l_key] = l_n + 1
            rows.append((tourney_id, match_num, tourney_date, winner_id, loser_id, surface,
                         w_over, l_over, w_surf, l_surf, w_n, l_n, w_surf_new, l_surf_new))

        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS atp_surface_elo_history CASCADE")
            cur.execute(
                """
                CREATE TABLE atp_surface_elo_history (
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
                    "INSERT INTO atp_surface_elo_history VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    rows[i : i + 5000],
                )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM atp_surface_elo_history")
            print("DB surface elo history rows:", cur.fetchone()[0])
            print("Matches with missing surface (overall only):", missing_surface)
            cur.execute("SELECT surface, COUNT(*) FROM atp_surface_elo_history WHERE surface IS NOT NULL GROUP BY surface ORDER BY COUNT(*) DESC")
            for row in cur.fetchall():
                print(" ", row)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
