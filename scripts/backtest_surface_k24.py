"""Test K=24 plus surface blend, versus K=32 plus surface blend.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/backtest_surface_k24.py

Prerequisite: compute_surface_elo_k24.py must have run first.
This script reads atp_surface_elo_history_k24 only. The K=32
blend numbers to beat are already known from the earlier test
and are printed here for reference, they are not recomputed
from the K=32 table in this script, to keep this script focused
on the K=24 candidate:
  K=32, surface-known, prior=100: 2018-2025 LL=0.6261 Brier=0.2186,
  2024-2025 LL=0.6287 Brier=0.2201 (from backtest_surface_elo3.py,
  all prior=100 model, n=21,768 / 6,020).

Decision rule, fixed before seeing results:
- K=24 plus surface prior=100 is adopted as the new best model
  only if it beats BOTH of these on BOTH averages (2018-2025
  and 2024-2025): (a) K=24 overall only, and (b) K=32 plus
  surface prior=100. If it beats (a) but not (b), keep K=32
  plus surface prior=100. If it beats neither, keep K=32.
- Also printed, for completeness: K=24 overall only, and
  K=24 plus surface prior=50, so the prior choice at K=24
  is visible and not hidden.
"""

import math
import os
import sys

import psycopg


def blended_elo(surface_elo, overall_elo, n_surface, prior):
    surface_elo = float(surface_elo)
    overall_elo = float(overall_elo)
    if prior is None:
        return overall_elo
    if n_surface is None or n_surface <= 0:
        return overall_elo
    w = n_surface / (n_surface + prior)
    return w * surface_elo + (1.0 - w) * overall_elo


def win_prob(elo_a, elo_b):
    return 1.0 / (1.0 + 10.0 ** (-(float(elo_a) - float(elo_b)) / 400.0))


def metrics(probs):
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

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT tourney_date, surface, winner_overall_before, loser_overall_before, "
                    "winner_surface_before, loser_surface_before, "
                    "winner_surface_n_before, loser_surface_n_before "
                    "FROM atp_surface_elo_history_k24 "
                    "WHERE surface IS NOT NULL AND EXTRACT(YEAR FROM tourney_date) BETWEEN 2018 AND 2025"
                )
                rows = cur.fetchall()
    except Exception as e:
        print("ERROR: could not read atp_surface_elo_history_k24. Run compute_surface_elo_k24.py first.")
        print(f"Database error was: {e}")
        return 1

    if not rows:
        print("ERROR: no rows found. Run compute_surface_elo_k24.py first.")
        return 1

    print("Reference to beat (K=32, surface-known):")
    print("  K=32 prior=100: 2018-2025 LL=0.6261 Brier=0.2186 | 2024-2025 LL=0.6287 Brier=0.2201")
    print("K=24 candidates, surface-known matches only:")
    for label, prior in (("K=24 overall only", None), ("K=24 prior=50", 50), ("K=24 prior=100", 100)):
        all_probs = [win_prob(blended_elo(r[4], r[2], r[6], prior), blended_elo(r[5], r[3], r[7], prior)) for r in rows]
        recent_probs = [win_prob(blended_elo(r[4], r[2], r[6], prior), blended_elo(r[5], r[3], r[7], prior)) for r in rows if r[0].year in (2024, 2025)]
        ll, brier, n = metrics(all_probs)
        llr, brierr, nr = metrics(recent_probs)
        print(f"  {label}: 2018-2025 n={n} LL={ll:.4f} Brier={brier:.4f} | 2024-2025 n={nr} LL={llr:.4f} Brier={brierr:.4f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
