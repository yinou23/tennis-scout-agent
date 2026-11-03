"""Test a small Elo + win-form blend: does it lower held-out LL/Brier?

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/backtest_blend.py

Method, fixed before seeing the results:
- Only win-% form is blended, not games-% form, because the two
  form signals were shown to be the same signal (games backtest
  2026-10-28). Adding both would double-count recent matches.
- Only last10 win-% is blended, for the same reason.
- Blend is on the logit scale, so a form difference moves the
  probability, but never flips it on its own at small alpha:
      logit(p_final) = logit(p_elo) + alpha * (winner_form - loser_form)
  where winner_form / loser_form are the actual winner's and
  loser's pre-match last10 win rates.
- Alpha values tested: 0 (baseline), 0.25, 0.5, 1.0.
- Two subsets, both with baseline recomputed on the same subset
  so the comparison is fair:
  A) both players have a last10 win rate (any number of prior matches)
  B) both players have 10+ prior matches in the last10 window
    (this removes the noisiest 0/0.5/1.0 rates from 1-2 matches)
- Keep the blend only if an alpha lowers BOTH Log Loss and Brier
  in BOTH 2024 and 2025. A gain in one year only is tuning noise.
"""

import math
import os
import sys

import psycopg


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
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT e.tourney_date, e.expected_winner_prob, "
                "f.winner_last10_pct, f.loser_last10_pct, "
                "f.winner_last10_n, f.loser_last10_n "
                "FROM atp_elo_history e "
                "JOIN atp_form_history f ON f.tourney_id = e.tourney_id AND f.match_num = e.match_num "
                "WHERE EXTRACT(YEAR FROM e.tourney_date) IN (2024, 2025)"
            )
            rows = cur.fetchall()

    for year in (2024, 2025):
        year_rows = [r for r in rows if r[0].year == year]
        for label, subset in (
            ("A: both have last10 form", [r for r in year_rows if r[2] is not None and r[3] is not None]),
            ("B: both have 10+ prior matches", [r for r in year_rows if r[4] is not None and r[5] is not None and r[4] >= 10 and r[5] >= 10]),
        ):
            print(f"Year {year}, {label}: n={len(subset)}")
            for alpha in (0.0, 0.25, 0.5, 1.0):
                probs = []
                for _, p_elo, w_form, l_form, _, _ in subset:
                    if alpha == 0.0:
                        probs.append(float(p_elo))
                    else:
                        form_diff = float(w_form) - float(l_form)
                        probs.append(sigmoid(logit(p_elo) + alpha * form_diff))
                ll, brier, n = metrics(probs)
                print(f"  alpha={alpha}: log_loss={ll:.4f}, brier={brier:.4f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
