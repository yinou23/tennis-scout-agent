"""Test overall Elo K values against each other, multi-year.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/backtest_k.py

Prerequisite: compute_elo_k_sweep.py must have run first, because
this script reads atp_elo_history_k16, k24, k32, k40.

Decision rule, fixed before seeing results, same as surface test:
- Primary window is 2018-2025 full years, 2024-2025 printed
  separately for continuity.
- A K is adopted only if it lowers BOTH Log Loss and Brier
  versus K=32 on average over 2018-2025 AND on average over
  2024-2025. If no K clears both averages, keep K=32.
- This test is overall Elo only. If a new K wins here, surface
  Elo must be recomputed with that K and the prior=100 blend
  re-tested before the combined model is adopted.
"""

import argparse
import math
import os
import sys

import psycopg


def metrics(probs):
    n = len(probs)
    if n == 0:
        return None, None, 0
    ll = -sum(math.log(max(p, 1e-12)) for p in probs) / n
    brier = sum((1.0 - p) ** 2 for p in probs) / n
    return ll, brier, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a", file=sys.stderr)
        return 1

    try:
        with psycopg.connect(database_url) as conn:
            data = {}
            for k in (16, 24, 32, 40):
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT tourney_date, expected_winner_prob FROM atp_elo_history_k{k} "
                        "WHERE EXTRACT(YEAR FROM tourney_date) BETWEEN %s AND %s",
                        (args.start, args.end),
                    )
                    data[k] = cur.fetchall()
    except Exception as e:
        print("ERROR: could not read K tables. Run compute_elo_k_sweep.py first.")
        print(f"Database error was: {e}")
        return 1

    for k in (16, 24, 32, 40):
        rows = data[k]
        all_probs = [float(p) for _, p in rows]
        recent_probs = [float(p) for d, p in rows if d.year in (2024, 2025)]
        ll, brier, n = metrics(all_probs)
        llr, brierr, nr = metrics(recent_probs)
        print(f"K={k}: {args.start}-{args.end} n={n} LL={ll:.4f} Brier={brier:.4f} | 2024-2025 n={nr} LL={llr:.4f} Brier={brierr:.4f}")

    print("Per-year, for reference:")
    for year in range(args.start, args.end + 1):
        line = f"  {year}: "
        for k in (16, 24, 32, 40):
            probs = [float(p) for d, p in data[k] if d.year == year]
            ll, brier, n = metrics(probs)
            line += f"K={k} LL={ll:.4f} "
        print(line)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
