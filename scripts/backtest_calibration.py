"""Calibration check and Platt-style scaling test for the current best model.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/backtest_calibration.py

Current best model, fixed before this test:
- K=32 overall Elo, surface Elo blended with prior=100.
  This is the model adopted after the surface test, not K=24,
  because K=24 plus surface did not beat K=32 plus surface
  on both averages.

Part 1, calibration table (diagnostic, no fitting):
- For each match, compute the blended probability for the actual
  winner, then derive the favorite probability (the larger of
  the winner probability and 1 minus that probability) and
  whether the favorite actually won.
- Bin favorite probability into 0.5-0.6, 0.6-0.7, 0.7-0.8,
  0.8-0.9, 0.9-1.0, and compare average predicted probability
  in each bin to the actual favorite win rate in that bin.
  If predicted is higher than actual in high bins, the model
  is overconfident; if lower, it is underconfident.

Part 2, scaling test (fitting on past, testing on future only):
- Scaling formula: logit(p_cal) = a * logit(p) + b.
  a < 1 shrinks probabilities toward 50% (less confident),
  a > 1 pushes them away from 50% (more confident).
- Grid search a in 0.5 to 1.2 step 0.1, b in -0.2 to 0.2 step
  0.1, on training years 2018-2023 only, picking the pair
  with the lowest training Log Loss. No scipy is needed.
- That pair is then evaluated, once, on test years 2024-2025.
  It is adopted only if it lowers BOTH Log Loss and Brier
  on 2024-2025 versus the unscaled model. Fitting on the
  test years themselves would be cheating, so it is not done.
"""

import math
import os
import sys

import psycopg


def blended_elo(surface_elo, overall_elo, n_surface, prior=100):
    surface_elo = float(surface_elo)
    overall_elo = float(overall_elo)
    if n_surface is None or n_surface <= 0:
        return overall_elo
    w = n_surface / (n_surface + prior)
    return w * surface_elo + (1.0 - w) * overall_elo


def win_prob(elo_a, elo_b):
    return 1.0 / (1.0 + 10.0 ** (-(float(elo_a) - float(elo_b)) / 400.0))


def logit(p):
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


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
                    "SELECT tourney_date, winner_overall_before, loser_overall_before, "
                    "winner_surface_before, loser_surface_before, "
                    "winner_surface_n_before, loser_surface_n_before "
                    "FROM atp_surface_elo_history "
                    "WHERE surface IS NOT NULL AND EXTRACT(YEAR FROM tourney_date) BETWEEN 2018 AND 2025"
                )
                rows = cur.fetchall()
    except Exception as e:
        print("ERROR: could not read atp_surface_elo_history. Run compute_surface_elo.py first.")
        print(f"Database error was: {e}")
        return 1

    # p_winner: blended probability assigned to the actual winner.
    data = []
    for tourney_date, w_over, l_over, w_surf, l_surf, w_n, l_n in rows:
        w_blend = blended_elo(w_surf, w_over, w_n, 100)
        l_blend = blended_elo(l_surf, l_over, l_n, 100)
        p_winner = win_prob(w_blend, l_blend)
        data.append((tourney_date.year, p_winner))

    print("Part 1: calibration table, 2018-2025, favorite probability bins")
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        subset = []
        for _, p_winner in data:
            fav_prob = p_winner if p_winner >= 0.5 else 1.0 - p_winner
            fav_won = 1 if p_winner >= 0.5 else 0
            if lo <= fav_prob < hi:
                subset.append((fav_prob, fav_won))
        if not subset:
            print(f"  {lo:.1f}-{hi:.1f}: n=0")
            continue
        avg_pred = sum(p for p, _ in subset) / len(subset)
        actual = sum(w for _, w in subset) / len(subset)
        print(f"  {lo:.1f}-{hi:.1f}: n={len(subset)}, avg predicted={avg_pred:.3f}, actual favorite win rate={actual:.3f}, difference={avg_pred - actual:+.3f}")

    train = [p for y, p in data if 2018 <= y <= 2023]
    test = [p for y, p in data if y in (2024, 2025)]
    print(f"Part 2: scaling, train 2018-2023 n={len(train)}, test 2024-2025 n={len(test)}")
    ll0, b0, _ = metrics(test)
    print(f"  Unscaled test: LL={ll0:.4f} Brier={b0:.4f}")

    best = None
    for a in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        for b in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            scaled_train = [sigmoid(a * logit(p) + b) for p in train]
            ll_train, _, _ = metrics(scaled_train)
            if best is None or ll_train < best[0]:
                best = (ll_train, a, b)
    _, best_a, best_b = best
    print(f"  Best on train (lowest train LL): a={best_a}, b={best_b}")
    scaled_test = [sigmoid(best_a * logit(p) + best_b) for p in test]
    ll1, b1, _ = metrics(scaled_test)
    print(f"  Scaled test with a={best_a}, b={best_b}: LL={ll1:.4f} Brier={b1:.4f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
