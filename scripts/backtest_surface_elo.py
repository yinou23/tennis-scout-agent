"""Test surface Elo blending against overall Elo alone, multi-year.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/backtest_surface_elo.py
    python scripts/backtest_surface_elo.py --start 2018 --end 2025

Prerequisite, in order: compute_surface_elo.py must have run first,
because this script reads atp_surface_elo_history. If that table
does not exist yet, run the compute step first, do not run this
script alone.

This version is self-contained: it does NOT import surface_elo.py,
so it cannot fail with ModuleNotFoundError just because a helper
file is in a different folder. The blend formula is copied here,
and it is the same formula as in surface_elo.py:
    blended = w * surface_elo + (1 - w) * overall_elo
    w = n_surface / (n_surface + prior)
If n_surface = 0, blended = overall Elo. If prior is None,
blended = overall Elo. If prior is 0, blended = surface Elo.

Decision rule, fixed before seeing results:
- Primary window is 2018-2025 full years (2026 partial excluded,
  surface mix biased). 2024 and 2025 are printed separately for
  continuity with earlier Week 2 tests.
- A prior is adopted only if it lowers BOTH Log Loss and Brier
  on average over 2018-2025 AND on average over 2024-2025.
"""

import argparse
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
    if prior == 0:
        return surface_elo
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
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT tourney_date, surface, winner_overall_before, loser_overall_before, "
                    "winner_surface_before, loser_surface_before, "
                    "winner_surface_n_before, loser_surface_n_before "
                    "FROM atp_surface_elo_history "
                    "WHERE surface IS NOT NULL AND EXTRACT(YEAR FROM tourney_date) BETWEEN %s AND %s",
                    (args.start, args.end),
                )
                rows = cur.fetchall()
    except Exception as e:
        print("ERROR: could not read atp_surface_elo_history. This usually means the compute step has not run yet.")
        print("Run first: bash scripts/run_checked.sh tests/test_surface_elo.py scripts/compute_surface_elo.py")
        print(f"Database error was: {e}")
        return 1

    if not rows:
        print("ERROR: no rows found in atp_surface_elo_history for the requested years.")
        print("Run the compute step first: bash scripts/run_checked.sh tests/test_surface_elo.py scripts/compute_surface_elo.py")
        return 1

    priors = (None, 10, 20, 50, 0)
    labels = {None: "overall only", 10: "prior=10", 20: "prior=20", 50: "prior=50", 0: "surface alone"}
    all_probs = {p: [] for p in priors}
    recent_probs = {p: [] for p in priors}

    for year in range(args.start, args.end + 1):
        year_rows = [r for r in rows if r[0].year == year]
        if not year_rows:
            continue
        print(f"Year {year}: surface-known matches n={len(year_rows)}")
        for prior in priors:
            probs = []
            for _, _, w_over, l_over, w_surf, l_surf, w_n, l_n in year_rows:
                w_blend = blended_elo(w_surf, w_over, w_n, prior)
                l_blend = blended_elo(l_surf, l_over, l_n, prior)
                probs.append(win_prob(w_blend, l_blend))
            ll, brier, n = metrics(probs)
            print(f"  {labels[prior]}: log_loss={ll:.4f}, brier={brier:.4f}")
            all_probs[prior].extend(probs)
            if year in (2024, 2025):
                recent_probs[prior].extend(probs)

    print(f"Average over {args.start}-{args.end} (weighted by matches):")
    for prior in priors:
        ll, brier, n = metrics(all_probs[prior])
        print(f"  {labels[prior]}: n={n}, log_loss={ll:.4f}, brier={brier:.4f}")
    print("Average over 2024-2025 only:")
    for prior in priors:
        ll, brier, n = metrics(recent_probs[prior])
        print(f"  {labels[prior]}: n={n}, log_loss={ll:.4f}, brier={brier:.4f}")

    print(f"Per-surface, prior=20, averaged over {args.start}-{args.end}:")
    for surface in sorted({r[1] for r in rows}):
        sub = [r for r in rows if r[1] == surface]
        base = [win_prob(r[2], r[3]) for r in sub]
        blend = [win_prob(blended_elo(r[4], r[2], r[6], 20), blended_elo(r[5], r[3], r[7], 20)) for r in sub]
        ll0, b0, _ = metrics(base)
        ll1, b1, _ = metrics(blend)
        print(f"  {surface}: n={len(sub)}, baseline LL={ll0:.4f} Brier={b0:.4f}, prior20 LL={ll1:.4f} Brier={b1:.4f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
