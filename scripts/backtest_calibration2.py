"""Calibration scaling test, CORRECTED version.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/backtest_calibration2.py

Why this version exists: the first calibration script had a
leakage bug in Part 2, and its scaled numbers must NOT be used.
The bug: it applied scaling to p_winner, the probability
assigned to the actual winner, which is only known after the
match. Scaling that quantity with a positive b systematically
inflates p_winner, especially for underdog wins, and therefore
produces a fake, huge improvement (LL 0.6287 to 0.5432 in the
first run). That improvement cannot be achieved in a real
prediction, because in a real prediction you do not know who
will win yet.

Correct method, used here:
- Favorite is defined BEFORE the match, by blended Elo:
  the player with the higher blended Elo is the favorite,
  p_fav is the model's probability for that favorite (always
  >= 0.5, except for exact ties), and fav_won is 1 if the
  favorite actually won, 0 otherwise.
- Scaling is applied to p_fav only:
      p_fav_cal = sigmoid(a * logit(p_fav) + b)
  and the loser's calibrated probability is 1 - p_fav_cal.
  This preserves complementarity and uses only pre-match
  information.
- Grid search a in 0.5 to 1.2 step 0.1, b in -0.2 to 0.2 step
  0.1, on training years 2018-2023 only, picking the lowest
  training Log Loss. That pair is then evaluated, once, on
  test years 2024-2025.
- Adoption rule, same as before: adopt scaling only if it
  lowers BOTH Log Loss and Brier on 2024-2025 versus the
  unscaled model. A legitimate calibration gain should be
  small (order 0.001 to 0.005 in LL), not 0.08. If the corrected
  test shows a huge gain again, that is a sign of another
  bug, not a real gain, and it must be investigated, not
  adopted.
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


def metrics_fav(p_fav_list, fav_won_list):
    n = len(p_fav_list)
    if n == 0:
        return None, None, 0
    ll = 0.0
    brier = 0.0
    for p, won in zip(p_fav_list, fav_won_list):
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        ll += -math.log(p) if won else -math.log(1.0 - p)
        brier += (won - p) ** 2
    return ll / n, brier / n, n


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

    # Favorite defined by blended Elo, before the match.
    data = []
    for tourney_date, w_over, l_over, w_surf, l_surf, w_n, l_n in rows:
        w_blend = blended_elo(w_surf, w_over, w_n, 100)
        l_blend = blended_elo(l_surf, l_over, l_n, 100)
        p_winner = win_prob(w_blend, l_blend)
        if w_blend >= l_blend:
            p_fav = p_winner
            fav_won = 1
        else:
            p_fav = 1.0 - p_winner
            fav_won = 0
        data.append((tourney_date.year, p_fav, fav_won))

    print("Calibration table, 2018-2025, favorite probability bins (same as before, this part was correct)")
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        subset = [(p, w) for _, p, w in data if lo <= p < hi]
        if not subset:
            print(f"  {lo:.1f}-{hi:.1f}: n=0")
            continue
        avg_pred = sum(p for p, _ in subset) / len(subset)
        actual = sum(w for _, w in subset) / len(subset)
        print(f"  {lo:.1f}-{hi:.1f}: n={len(subset)}, avg predicted={avg_pred:.3f}, actual favorite win rate={actual:.3f}, difference={avg_pred - actual:+.3f}")

    train = [(p, w) for y, p, w in data if 2018 <= y <= 2023]
    test = [(p, w) for y, p, w in data if y in (2024, 2025)]
    print(f"Corrected scaling test, train 2018-2023 n={len(train)}, test 2024-2025 n={len(test)}")
    ll0, b0, _ = metrics_fav([p for p, _ in test], [w for _, w in test])
    print(f"  Unscaled test: LL={ll0:.4f} Brier={b0:.4f}")

    best = None
    for a in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        for b in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            scaled_train = [sigmoid(a * logit(p) + b) for p, _ in train]
            ll_train, _, _ = metrics_fav(scaled_train, [w for _, w in train])
            if best is None or ll_train < best[0]:
                best = (ll_train, a, b)
    _, best_a, best_b = best
    print(f"  Best on train (lowest train LL): a={best_a}, b={best_b}")
    scaled_test = [sigmoid(best_a * logit(p) + best_b) for p, _ in test]
    ll1, b1, _ = metrics_fav(scaled_test, [w for _, w in test])
    print(f"  Scaled test with a={best_a}, b={best_b}: LL={ll1:.4f} Brier={b1:.4f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
