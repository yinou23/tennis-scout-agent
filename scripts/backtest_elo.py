"""Baseline backtest for classic Elo, using atp_elo_history.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/backtest_elo.py

This is a walk-forward backtest, not a random train/test split:
atp_elo_history stores the pre-match Elo for every match, computed
only from matches before it in chronological order. So a 2025 match
is scored with Elo that never saw any 2025 match after it, and never
saw its own result. That makes 2024 and 2025 accuracy honest
out-of-sample numbers for the K=32 baseline, as long as K was not
tuned on those same years yet.

Metrics per year:
- n: matches
- favorite_win_rate: share where the pre-match higher-Elo player won.
  50% is the coin-flip baseline.
- log_loss: -mean(log(prob assigned to the actual winner)).
  Lower is better; a perfect model is 0, coin flip is 0.693.
- brier: mean((1 - prob assigned to the actual winner) ** 2).
  Lower is better; coin flip is 0.25.
"""

import math
import os
import sys

import psycopg


def metrics(rows):
    n = len(rows)
    if n == 0:
        return None
    fav_wins = 0
    log_loss = 0.0
    brier = 0.0
    for w_before, l_before, expected in rows:
        # expected is the pre-match prob that the eventual winner wins.
        fav_wins += 1 if w_before >= l_before else 0
        p = float(expected)
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        log_loss += -math.log(p)
        brier += (1.0 - p) ** 2
    return {
        "n": n,
        "favorite_win_rate": fav_wins / n,
        "log_loss": log_loss / n,
        "brier": brier / n,
    }


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a", file=sys.stderr)
        return 1
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for year in (2024, 2025, 2026):
                cur.execute(
                    "SELECT winner_elo_before, loser_elo_before, expected_winner_prob "
                    "FROM atp_elo_history "
                    "WHERE EXTRACT(YEAR FROM tourney_date) = %s",
                    (year,),
                )
                m = metrics(cur.fetchall())
                if m is None:
                    print(f"{year}: no matches")
                else:
                    print(
                        f"{year}: n={m['n']}, favorite_win_rate={m['favorite_win_rate']:.3f}, "
                        f"log_loss={m['log_loss']:.4f}, brier={m['brier']:.4f}"
                    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
