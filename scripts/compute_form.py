"""Compute pre-match form windows and store them in PostgreSQL.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/compute_form.py

Reads atp_matches in the same chronological order as compute_elo.py
(tourney_date, tourney_id, match_num). For every match, it snapshots
both players' form BEFORE the match, then records the result.

Writes atp_form_history (dropped and recreated):
- winner/loser last10 win rate and sample size
- winner/loser last52 win rate and sample size
A win rate is NULL when the player has no previous match in that
window. This table is a feature table, not a new Elo: later steps
join it to atp_elo_history to test whether form improves prediction
over Elo alone.
"""

import os
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from form import PlayerForm  # noqa: E402


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

        forms = {}
        rows = []
        for tourney_id, match_num, tourney_date, winner_id, loser_id in matches:
            wf = forms.setdefault(winner_id, PlayerForm())
            lf = forms.setdefault(loser_id, PlayerForm())
            w10, w10n, w52, w52n = wf.snapshot(tourney_date)
            l10, l10n, l52, l52n = lf.snapshot(tourney_date)
            rows.append(
                (tourney_id, match_num, tourney_date, winner_id, loser_id,
                 w10, w10n, w52, w52n, l10, l10n, l52, l52n)
            )
            wf.record(tourney_date, True)
            lf.record(tourney_date, False)

        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS atp_form_history CASCADE")
            cur.execute(
                """
                CREATE TABLE atp_form_history (
                    tourney_id        TEXT NOT NULL,
                    match_num         INTEGER NOT NULL,
                    tourney_date      DATE,
                    winner_id         INTEGER,
                    loser_id          INTEGER,
                    winner_last10_pct NUMERIC,
                    winner_last10_n   INTEGER,
                    winner_last52_pct NUMERIC,
                    winner_last52_n   INTEGER,
                    loser_last10_pct  NUMERIC,
                    loser_last10_n    INTEGER,
                    loser_last52_pct  NUMERIC,
                    loser_last52_n    INTEGER,
                    PRIMARY KEY (tourney_id, match_num)
                )
                """
            )
            cur.execute("CREATE INDEX idx_form_winner ON atp_form_history (winner_id)")
            cur.execute("CREATE INDEX idx_form_loser ON atp_form_history (loser_id)")
            for i in range(0, len(rows), 5000):
                cur.executemany(
                    "INSERT INTO atp_form_history VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    rows[i : i + 5000],
                )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM atp_form_history")
            print("DB form history rows:", cur.fetchone()[0])
            cur.execute(
                "SELECT COUNT(*) FROM atp_form_history "
                "WHERE winner_last10_n >= 10 AND loser_last10_n >= 10"
            )
            print("Matches where both players had 10+ prior matches:", cur.fetchone()[0])
            cur.execute(
                "SELECT COUNT(*) FROM atp_form_history "
                "WHERE winner_last52_n > 0 AND loser_last52_n > 0"
            )
            print("Matches where both players had 52-week form:", cur.fetchone()[0])
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
