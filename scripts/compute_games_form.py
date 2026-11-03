"""Compute pre-match games-% form windows.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/compute_games_form.py

Reads atp_matches joined to atp_match_games, in chronological order
(tourney_date, tourney_id, match_num). For every match, it snapshots
both players' games-% form BEFORE the match, then records the match
only if it is complete.

Writes atp_games_form_history (dropped and recreated). This is a
feature table, like atp_form_history, not a new Elo. The next test
joins it to atp_elo_history the same way the win-% form test did:
does games-% add signal beyond Elo, and beyond win-% form?
"""

import os
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from games_form import PlayerGamesForm  # noqa: E402


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a", file=sys.stderr)
        return 1

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT m.tourney_id, m.match_num, m.tourney_date, m.winner_id, m.loser_id, "
                "g.winner_games, g.loser_games, g.is_complete "
                "FROM atp_matches m LEFT JOIN atp_match_games g "
                "ON g.tourney_id = m.tourney_id AND g.match_num = m.match_num "
                "WHERE m.winner_id IS NOT NULL AND m.loser_id IS NOT NULL "
                "ORDER BY m.tourney_date, m.tourney_id, m.match_num"
            )
            matches = cur.fetchall()
        print(f"Matches to process: {len(matches)}")

        forms = {}
        rows = []
        for tourney_id, match_num, tourney_date, winner_id, loser_id, w_games, l_games, is_complete in matches:
            wf = forms.setdefault(winner_id, PlayerGamesForm())
            lf = forms.setdefault(loser_id, PlayerGamesForm())
            w10, w10m, w10g, w52, w52m, w52g = wf.snapshot(tourney_date)
            l10, l10m, l10g, l52, l52m, l52g = lf.snapshot(tourney_date)
            rows.append(
                (tourney_id, match_num, tourney_date, winner_id, loser_id,
                 w10, w10m, w10g, w52, w52m, w52g,
                 l10, l10m, l10g, l52, l52m, l52g)
            )
            if is_complete and w_games is not None and l_games is not None:
                total = w_games + l_games
                wf.record(tourney_date, w_games, total, True)
                lf.record(tourney_date, l_games, total, True)

        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS atp_games_form_history CASCADE")
            cur.execute(
                """
                CREATE TABLE atp_games_form_history (
                    tourney_id           TEXT NOT NULL,
                    match_num            INTEGER NOT NULL,
                    tourney_date         DATE,
                    winner_id            INTEGER,
                    loser_id             INTEGER,
                    winner_last10_games_pct NUMERIC,
                    winner_last10_matches INTEGER,
                    winner_last10_games  INTEGER,
                    winner_last52_games_pct NUMERIC,
                    winner_last52_matches INTEGER,
                    winner_last52_games  INTEGER,
                    loser_last10_games_pct NUMERIC,
                    loser_last10_matches INTEGER,
                    loser_last10_games   INTEGER,
                    loser_last52_games_pct NUMERIC,
                    loser_last52_matches INTEGER,
                    loser_last52_games   INTEGER,
                    PRIMARY KEY (tourney_id, match_num)
                )
                """
            )
            for i in range(0, len(rows), 5000):
                cur.executemany(
                    "INSERT INTO atp_games_form_history VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    rows[i : i + 5000],
                )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM atp_games_form_history")
            print("DB games form history rows:", cur.fetchone()[0])
            cur.execute(
                "SELECT COUNT(*) FROM atp_games_form_history "
                "WHERE winner_last10_matches >= 10 AND loser_last10_matches >= 10"
            )
            print("Matches where both players had 10+ complete prior matches:", cur.fetchone()[0])
            cur.execute(
                "SELECT COUNT(*) FROM atp_games_form_history "
                "WHERE winner_last52_matches > 0 AND loser_last52_matches > 0"
            )
            print("Matches where both players had 52-week games form:", cur.fetchone()[0])
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
