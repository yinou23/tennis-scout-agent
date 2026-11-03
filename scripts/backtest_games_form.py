"""Test whether games-% form adds signal beyond Elo alone.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/backtest_games_form.py

Joins atp_elo_history to atp_games_form_history on
(tourney_id, match_num). All games-% values are pre-match and
from complete matches only, so there is no leakage and no
partial-match bias.

For each year, for last10 games % and last52 games % separately:
- games accuracy: share where the player with the higher pre-match
  games % actually won, counting only matches where both players
  have a games-% value and the values are not tied.
- agree: Elo favorite and games favorite are the same player.
- disagree: Elo favorite and games favorite differ. If Elo still
  wins more than 50% in disagree matches, Elo is stronger; if Elo
  wins less than 50%, games favorite wins more often, so games %
  adds independent signal beyond Elo.

This mirrors backtest_form.py exactly, so games accuracy and
disagree Elo win rate can be compared directly to the win-%
form results (2024/2025 win form accuracy about 61%, disagree
Elo win rate about 57-59%).
"""

import os
import sys

import psycopg


def _sign(a, b):
    if a > b:
        return 1
    if a < b:
        return -1
    return 0


def analyse(rows):
    rows = [r for r in rows if r[2] is not None and r[3] is not None]
    rows = [r for r in rows if _sign(r[2], r[3]) != 0]
    if not rows:
        return None
    games_wins = sum(1 for r in rows if r[2] > r[3])
    agree = []
    disagree = []
    for w_elo, l_elo, w_games, l_games in rows:
        elo_sign = _sign(w_elo, l_elo)
        games_sign = _sign(w_games, l_games)
        if elo_sign == games_sign:
            agree.append(w_elo > l_elo)
        elif elo_sign != 0:
            disagree.append(w_elo > l_elo)
    return {
        "games_n": len(rows),
        "games_accuracy": games_wins / len(rows),
        "agree_n": len(agree),
        "agree_elo_win_rate": (sum(agree) / len(agree)) if agree else None,
        "disagree_n": len(disagree),
        "disagree_elo_win_rate": (sum(disagree) / len(disagree)) if disagree else None,
    }


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a", file=sys.stderr)
        return 1
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for year in (2024, 2025):
                cur.execute(
                    "SELECT e.winner_elo_before, e.loser_elo_before, "
                    "g.winner_last10_games_pct, g.loser_last10_games_pct, "
                    "g.winner_last52_games_pct, g.loser_last52_games_pct "
                    "FROM atp_elo_history e "
                    "JOIN atp_games_form_history g ON g.tourney_id = e.tourney_id AND g.match_num = e.match_num "
                    "WHERE EXTRACT(YEAR FROM e.tourney_date) = %s",
                    (year,),
                )
                rows = cur.fetchall()
                print(f"Year {year}: total matches {len(rows)}")
                for label, wi, li in (("last10 games", 2, 3), ("last52 games", 4, 5)):
                    sub = [(r[0], r[1], r[wi], r[li]) for r in rows]
                    m = analyse(sub)
                    if m is None:
                        print(f"  {label}: no games form data")
                        continue
                    print(
                        f"  {label}: games_n={m['games_n']}, games_accuracy={m['games_accuracy']:.3f}, "
                        f"agree_n={m['agree_n']}, agree_elo_win_rate={m['agree_elo_win_rate']:.3f}, "
                        f"disagree_n={m['disagree_n']}, disagree_elo_win_rate={m['disagree_elo_win_rate']:.3f}"
                    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
