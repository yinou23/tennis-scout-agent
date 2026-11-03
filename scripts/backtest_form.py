"""Test whether form windows add signal beyond Elo alone.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/backtest_form.py

Joins atp_elo_history to atp_form_history on (tourney_id, match_num).
All form values are pre-match, so there is no leakage from the
match itself.

For each year, it prints, for last10 and for last52 separately:
- form accuracy: share where the player with the higher pre-match
  form win rate actually won, counting only matches where both
  players have a form value in that window.
- agree: matches where Elo favorite and form favorite are the
  same player. In those matches, Elo favorite win rate should be
  higher than the overall Elo baseline if form confirms Elo.
- disagree: matches where Elo favorite and form favorite differ.
  In those matches, if Elo favorite still wins more than 50%,
  Elo is stronger than form; if Elo favorite wins less than 50%,
  form favorite wins more often, so form adds independent signal
  beyond Elo. This disagree subset is the key test, not the
  overall form accuracy alone.
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
    """rows: (w_elo_before, l_elo_before, w_form, l_form).

    Ties are excluded from form accuracy and from agree/disagree:
    last10 win rates are discrete (0.0, 0.1, ..., 1.0), so ties are
    common, and counting a tie as a correct form pick would inflate
    form accuracy. Elo ties are almost never exact, because Elo is
    continuous.
    """
    form_rows = [r for r in rows if r[2] is not None and r[3] is not None]
    form_rows = [r for r in form_rows if _sign(r[2], r[3]) != 0]
    if not form_rows:
        return None
    form_wins = sum(1 for r in form_rows if r[2] > r[3])
    agree = []
    disagree = []
    for w_elo, l_elo, w_form, l_form in form_rows:
        elo_sign = _sign(w_elo, l_elo)
        form_sign = _sign(w_form, l_form)
        if elo_sign == form_sign:
            agree.append(w_elo > l_elo)
        elif elo_sign != 0:
            disagree.append(w_elo > l_elo)
    return {
        "form_n": len(form_rows),
        "form_accuracy": form_wins / len(form_rows),
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
                    "f.winner_last10_pct, f.loser_last10_pct, "
                    "f.winner_last52_pct, f.loser_last52_pct "
                    "FROM atp_elo_history e "
                    "JOIN atp_form_history f ON f.tourney_id = e.tourney_id AND f.match_num = e.match_num "
                    "WHERE EXTRACT(YEAR FROM e.tourney_date) = %s",
                    (year,),
                )
                rows = cur.fetchall()
                print(f"Year {year}: total matches {len(rows)}")
                for label, wi, li in (("last10", 2, 3), ("last52", 4, 5)):
                    sub = [(r[0], r[1], r[wi], r[li]) for r in rows]
                    sub = [r for r in sub if r[2] is not None and r[3] is not None]
                    m = analyse(sub)
                    if m is None:
                        print(f"  {label}: no form data")
                        continue
                    print(
                        f"  {label}: form_n={m['form_n']}, form_accuracy={m['form_accuracy']:.3f}, "
                        f"agree_n={m['agree_n']}, agree_elo_win_rate={m['agree_elo_win_rate']:.3f}, "
                        f"disagree_n={m['disagree_n']}, disagree_elo_win_rate={m['disagree_elo_win_rate']:.3f}"
                    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
