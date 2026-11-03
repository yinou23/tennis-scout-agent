"""Surface Elo test 3: larger priors and per-surface priors.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/backtest_surface_elo3.py

Prerequisite: atp_surface_elo_history must already exist. No recompute.

Why this test, fixed before seeing its results:
- Previous test found prior=50 best among 10, 20, 50, with a
  long-average gain but a tiny recent gain. If larger priors
  (100, 200, 500) beat 50, the optimal shrinkage is heavier
  than 50. If they lose to 50, 50 is near the optimum and
  further increasing prior is not useful.
- Previous per-surface test (prior=20) found Clay gains a lot,
  Hard is neutral, Grass loses. Using one prior for all
  surfaces therefore forces Grass to use a prior that hurts it,
  and forces Clay to use a prior that may be too heavy for it.
  This script therefore tests per-surface priors separately:
  first, for each surface alone, which prior is best on that
  surface's matches; second, a few pre-specified combined
  models that use different priors per surface, not a full
  grid search over all combinations (which would be tuning
  noise on the same data).
- Pre-specified combined models, chosen from the previous
  per-surface result, not from this script's output:
  A) All surfaces prior=50 (previous best, the baseline to beat)
  B) All surfaces prior=100
  C) All surfaces prior=200
  D) Clay 50, Hard 100, Grass overall only (Grass prior=None)
  E) Clay 20, Hard 100, Grass overall only
  F) Clay 50 only, Hard and Grass overall only (Clay-only)
- Decision rule, same as before: a model is adopted only if
  it lowers BOTH Log Loss and Brier versus overall Elo, AND
  versus all-surface prior=50, on average over 2018-2025 AND
  on average over 2024-2025. If no model clears both averages,
  keep overall Elo only for v1, with surface Elo as a report
  descriptor, especially on Clay.
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
        print("ERROR: could not read atp_surface_elo_history. Run compute_surface_elo.py first.")
        print(f"Database error was: {e}")
        return 1

    if not rows:
        print("ERROR: no rows found. Run compute_surface_elo.py first.")
        return 1

    priors = (None, 20, 50, 100, 200, 500, 0)
    labels = {None: "overall only", 20: "prior=20", 50: "prior=50", 100: "prior=100", 200: "prior=200", 500: "prior=500", 0: "surface alone"}

    print(f"Part 1: per-surface prior sweep, {args.start}-{args.end} (each surface on its own matches only)")
    for surface in sorted({r[1] for r in rows}):
        sub = [r for r in rows if r[1] == surface]
        print(f"  {surface}: n={len(sub)}")
        for prior in priors:
            probs = [win_prob(blended_elo(r[4], r[2], r[6], prior), blended_elo(r[5], r[3], r[7], prior)) for r in sub]
            ll, brier, n = metrics(probs)
            print(f"    {labels[prior]}: LL={ll:.4f} Brier={brier:.4f}")

    print("Part 2: combined models, all surfaces together")
    # Each model is a dict: surface -> prior, missing surface means overall only (None).
    models = [
        ("overall only", {}),
        ("all prior=50", {"Clay": 50, "Grass": 50, "Hard": 50}),
        ("all prior=100", {"Clay": 100, "Grass": 100, "Hard": 100}),
        ("all prior=200", {"Clay": 200, "Grass": 200, "Hard": 200}),
        ("Clay 50, Hard 100, Grass overall", {"Clay": 50, "Hard": 100}),
        ("Clay 20, Hard 100, Grass overall", {"Clay": 20, "Hard": 100}),
        ("Clay 50 only", {"Clay": 50}),
        ("Clay 100 only", {"Clay": 100}),
    ]
    for label, prior_map in models:
        def prob_for(row, prior_map=prior_map):
            _, surface, w_over, l_over, w_surf, l_surf, w_n, l_n = row
            prior = prior_map.get(surface, None)
            return win_prob(blended_elo(w_surf, w_over, w_n, prior),
                            blended_elo(l_surf, l_over, l_n, prior))
        all_probs = [prob_for(r) for r in rows]
        recent_probs = [prob_for(r) for r in rows if r[0].year in (2024, 2025)]
        ll, brier, n = metrics(all_probs)
        llr, brierr, nr = metrics(recent_probs)
        print(f"  {label}: {args.start}-{args.end} n={n} LL={ll:.4f} Brier={brier:.4f} | 2024-2025 n={nr} LL={llr:.4f} Brier={brierr:.4f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
