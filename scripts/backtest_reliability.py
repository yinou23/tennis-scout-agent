"""Test reliability shrinkage: does shrinking low-sample Elo toward 1500
improve held-out Log Loss and Brier?

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/backtest_reliability.py

Method, agreed 2026-10-28:
- Reliability is continuous shrinkage, not a binary reliable/unreliable
  switch. For a player with n pre-match main-tour matches, the shrunk
  rating is: 1500 + (elo - 1500) * n / (n + K), where K is the
  shrinkage strength. n = 5 with K = 20 shrinks a 1700 rating to
  1540; n = 100 with K = 20 shrinks it only to 1667.
- The win probability is then computed from the two shrunk ratings
  with the same Elo formula: 1 / (1 + 10^(-(elo_w - elo_l) / 400)).
- This script reports baseline (no shrinkage) and K = 10, 20, 50
  for 2024, 2025, and 2026 partial, so a K that improves only one
  year is visible as tuning noise, not a real gain.
- It also splits out low-sample matches (either player with fewer
  than 10 pre-match matches), because shrinkage should help there
  most and barely change high-sample matches.

Baseline to beat, from backtest_elo.py:
  2024: LL 0.6271, Brier 0.2194
  2025: LL 0.6343, Brier 0.2220
  2026 partial: LL 0.6343, Brier 0.2210
"""

import math
import os
import sys
from collections import defaultdict

import psycopg

BASE_ELO = 1500.0


def win_prob(elo_w, elo_l):
    # Elo values come back from Postgres as Decimal, not float, so
    # convert here. Without this, Decimal / float raises TypeError.
    elo_w = float(elo_w)
    elo_l = float(elo_l)
    return 1.0 / (1.0 + 10.0 ** (-(elo_w - elo_l) / 400.0))


def shrunk(elo, n, k):
    if k is None:
        return float(elo)
    return BASE_ELO + (float(elo) - BASE_ELO) * n / (n + k)


def metrics(probs):
    """probs: list of probabilities assigned to the actual winner."""
    n = len(probs)
    if n == 0:
        return None, None, 0
    ll = -sum(math.log(max(p, 1e-12)) for p in probs) / n
    brier = sum((1.0 - p) ** 2 for p in probs) / n
    return ll, brier, n


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a", file=sys.stderr)
        return 1
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tourney_date, winner_id, loser_id, winner_elo_before, loser_elo_before "
                "FROM atp_elo_history ORDER BY tourney_date, tourney_id, match_num"
            )
            matches = cur.fetchall()

    played = defaultdict(int)
    # year -> method -> list of winner probabilities, plus low-sample subset
    results = defaultdict(lambda: defaultdict(list))
    low_results = defaultdict(lambda: defaultdict(list))

    for tourney_date, winner_id, loser_id, w_elo, l_elo in matches:
        year = tourney_date.year
        if year not in (2024, 2025, 2026):
            played[winner_id] += 1
            played[loser_id] += 1
            continue
        nw = played[winner_id]
        nl = played[loser_id]
        for k in (None, 10, 20, 50):
            label = "baseline" if k is None else f"K={k}"
            p = win_prob(shrunk(w_elo, nw, k), shrunk(l_elo, nl, k))
            results[year][label].append(p)
            if min(nw, nl) < 10:
                low_results[year][label].append(p)
        played[winner_id] += 1
        played[loser_id] += 1

    for year in (2024, 2025, 2026):
        print(f"Year {year}:")
        for label in ("baseline", "K=10", "K=20", "K=50"):
            ll, brier, n = metrics(results[year][label])
            print(f"  {label}: n={n}, log_loss={ll:.4f}, brier={brier:.4f}")
        print("  Low-sample matches only (either player < 10 prior matches):")
        for label in ("baseline", "K=10", "K=20", "K=50"):
            ll, brier, n = metrics(low_results[year][label])
            if n:
                print(f"    {label}: n={n}, log_loss={ll:.4f}, brier={brier:.4f}")
            else:
                print(f"    {label}: n=0")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
