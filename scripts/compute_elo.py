"""Compute classic Elo from atp_matches and store it in PostgreSQL.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/compute_elo.py

Reads atp_matches in chronological order (tourney_date, tourney_id,
match_num ascending; match_num ascending is early rounds first
inside a tournament). Every player starts at 1500.0, K = 32.

Writes two tables (dropped and recreated, so rerunning is clean):
- atp_elo_history: one row per match, with both players' Elo
  before and after, and the pre-match expected winner probability.
  This is what the Week 2 backtest reads, so backtest accuracy
  can be computed without recomputing Elo.
- atp_elo_current: one row per player, final Elo, matches played,
  and last match date.

Note on the top list: because classic Elo never decays, retired
greats keep their end-of-career Elo. The script therefore prints
two top lists: all-time, and active players (last match in 2026).
The active list is the sanity check for current scouting; the
all-time list is expected to include retired greats.
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
                "SELECT tourney_id, match_num, tourney_date, winner_id, loser_id "
                "FROM atp_matches "
                "WHERE winner_id IS NOT NULL AND loser_id IS NOT NULL "
                "ORDER BY tourney_date, tourney_id, match_num"
            )
            matches = cur.fetchall()
        print(f"Matches to process: {len(matches)}")

        ratings = {}
        matches_played = {}
        last_date = {}
        history = []
        for tourney_id, match_num, tourney_date, winner_id, loser_id in matches:
            w = ratings.get(winner_id, INITIAL_ELO)
            l = ratings.get(loser_id, INITIAL_ELO)
            w_new, l_new, expected = update_elo(w, l, k=DEFAULT_K)
            ratings[winner_id] = w_new
            ratings[loser_id] = l_new
            matches_played[winner_id] = matches_played.get(winner_id, 0) + 1
            matches_played[loser_id] = matches_played.get(loser_id, 0) + 1
            last_date[winner_id] = tourney_date
            last_date[loser_id] = tourney_date
            history.append(
                (tourney_id, match_num, tourney_date, winner_id, loser_id, w, l, w_new, l_new, expected)
            )

        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS atp_elo_history CASCADE")
            cur.execute("DROP TABLE IF EXISTS atp_elo_current CASCADE")
            cur.execute(
                """
                CREATE TABLE atp_elo_history (
                    tourney_id           TEXT NOT NULL,
                    match_num            INTEGER NOT NULL,
                    tourney_date         DATE,
                    winner_id            INTEGER,
                    loser_id             INTEGER,
                    winner_elo_before    NUMERIC,
                    loser_elo_before     NUMERIC,
                    winner_elo_after     NUMERIC,
                    loser_elo_after      NUMERIC,
                    expected_winner_prob NUMERIC,
                    PRIMARY KEY (tourney_id, match_num)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE atp_elo_current (
                    player_id      INTEGER PRIMARY KEY,
                    elo            NUMERIC,
                    matches_played INTEGER,
                    last_match_date DATE
                )
                """
            )
            cur.execute("CREATE INDEX idx_elo_history_winner ON atp_elo_history (winner_id)")
            cur.execute("CREATE INDEX idx_elo_history_loser ON atp_elo_history (loser_id)")
            cur.execute("CREATE INDEX idx_elo_history_date ON atp_elo_history (tourney_date)")
            for i in range(0, len(history), 5000):
                cur.executemany(
                    "INSERT INTO atp_elo_history VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    history[i : i + 5000],
                )
            cur.executemany(
                "INSERT INTO atp_elo_current VALUES (%s, %s, %s, %s)",
                [(pid, ratings[pid], matches_played[pid], last_date[pid]) for pid in ratings],
            )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM atp_elo_history")
            print("DB elo history rows:", cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM atp_elo_current")
            print("DB players rated:", cur.fetchone()[0])
            print("Top 10 all-time Elo:")
            cur.execute(
                "SELECT c.player_id, p.first_name, p.last_name, ROUND(c.elo, 1), c.matches_played, c.last_match_date "
                "FROM atp_elo_current c LEFT JOIN atp_players p ON p.player_id = c.player_id "
                "ORDER BY c.elo DESC LIMIT 10"
            )
            for row in cur.fetchall():
                print(" ", row)
            print("Top 10 active Elo (last match in 2026):")
            cur.execute(
                "SELECT c.player_id, p.first_name, p.last_name, ROUND(c.elo, 1), c.matches_played, c.last_match_date "
                "FROM atp_elo_current c LEFT JOIN atp_players p ON p.player_id = c.player_id "
                "WHERE c.last_match_date >= DATE '2026-01-01' "
                "ORDER BY c.elo DESC LIMIT 10"
            )
            for row in cur.fetchall():
                print(" ", row)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
