"""Hyperparameter grid search on validation only (2023-2024).

Stage 1 grid: K in {16, 24, 32, 48} x surface prior in {50, 100, 200}
(12 Elo walks). Calibration slope a in {0.7, 0.8, 0.9, 1.0} is
scored OFFLINE from the recorded blended probabilities:
calibration only affects predict, never update, so one walk per
(K, prior) scores all 4 calibration slopes (48 combos total
from 12 walks).

Calibration intercept b is deliberately NOT searched (fixed 0.0).
Two reasons, both verified, not assumed:
1. Leakage: this backtest scores every match from the winner's
   side (predict is called with winner_id first), so a positive b
   always inflates the eventual winner's probability. The first
   run of this script (with b in {-0.1, 0.0, +0.1}) "found"
   b=+0.1 winning by 0.0196 Brier and beating Pinnacle
   (D=-0.0031) -- an impossible result for a pure-Elo model,
   and the top 20 rows were all b=+0.1 while the best b=-0.1
   row was 0.2383. A parameter whose sign flips the result by
   0.04 Brier cannot be real; it is the winner-side labeling.
2. Symmetry: production returns prob_b = 1 - prob_a
   (scripts/tennis_tools.py), so the model must satisfy
   p(A beats B) + p(B beats A) = 1. sigmoid(a*logit(p) + b)
   only satisfies that for b = 0; any b != 0 makes "A vs B"
   and "B vs A" disagree on the same match. b = 0 is forced
   by the model's own symmetry, not a tunable choice.
Slope a is symmetric-safe (b=0 preserves p <-> 1-p), so a is
the only calibration parameter searched.

Scoring set: main-tour non-team matches (level A/F/G/M, excluding
align._is_team_exhibition), same as ablate_team.py. Selection is
on validation 2023-2024 only. The test set (2025-2026) is never
touched, and this script never changes the production model or
default-report.json: it only measures.

Selection rule: adopt the best validation config only if it beats
the locked baseline (K32 / prior100 / a=0.8 / b=0.0) by >= 2 SE
on the paired per-match Brier difference; otherwise keep the
locked baseline. (Caveat: the paired z is computed after picking
the best of 144 combos on the same validation set, so it is
optimistic; treat a marginal pass as weak evidence.)

Usage (repo root, project venv):
    .venv/bin/python -m scripts.backtest.grid_search

Writes scripts/backtest/outputs/grid_search.json (git-ignored).
"""

import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablate_team import _is_scored, _load_matches, _paired_variant_diff  # noqa: E402
from align import align_matches  # noqa: E402
from db import get_connection  # noqa: E402
from elo_state import EloState  # noqa: E402
from metrics import devig, paired_diff, score_outcome  # noqa: E402
from names import name_key  # noqa: E402
from odds import read_tennis_data_xlsx  # noqa: E402
from tennis_tools import calibrated_prob  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent / "outputs"

K_VALUES = (16, 24, 32, 48)
PRIOR_VALUES = (50, 100, 200)
CAL_A_VALUES = (0.7, 0.8, 0.9, 1.0)
# b is fixed at 0.0: see module docstring (winner-side leakage +
# A/B symmetry). Do not re-add b values without fixing the scoring
# side first (predict must be called with an outcome-independent
# A/B ordering, and even then b should be ~0 by symmetry).
CAL_B_VALUES = (0.0,)
BASELINE = {"k": 32, "prior": 100, "a": 0.8, "b": 0.0}
VALIDATION_YEARS = (2023, 2024)


def _pinnacle_matched_keys(matches):
    """(tourney_id, match_num) -> de-vigged Pinnacle prob of winner,
    validation years only. Alignment does not depend on Elo, so all
    grid configs share this matched set (same as ablate_team.py)."""
    odds_by_year = {}
    for year in VALIDATION_YEARS:
        rows = []
        for fy in (year - 1, year, year + 1):
            p = REPO_ROOT / "scripts" / "backtest" / "data" / f"{fy}.xlsx"
            if p.exists():
                rows += read_tennis_data_xlsx(str(p), fy)
        odds_by_year[year] = rows
    matched_keys = {}
    for year in VALIDATION_YEARS:
        year_matches = [m for m in matches if m["tourney_date"].year == year]
        matched, _ = align_matches(year_matches, odds_by_year[year])
        for m, o in matched:
            if "pinnacle" not in o.odds:
                continue
            oa, ob = o.odds["pinnacle"]
            try:
                pa, pb = devig(oa, ob)
            except ValueError:
                continue
            if name_key(o.player_a) == name_key(m["winner_name"]):
                matched_keys[(m["tourney_id"], m["match_num"])] = pa
            elif name_key(o.player_b) == name_key(m["winner_name"]):
                matched_keys[(m["tourney_id"], m["match_num"])] = pb
    return matched_keys


def _walk(matches, k, prior):
    """One walk-forward pass. Returns (val, train) where val is a
    list of (key, p_blended) for validation scored matches in walk
    order, and train is [p_blended] for 2001-2022 scored matches."""
    state = EloState(k=k, prior=prior)
    val = []
    train = []
    for m in matches:
        year = m["tourney_date"].year
        if year >= 2001 and _is_scored(m):
            _p_cal, p_blended, _ba, _bb = state.predict(
                m["winner_id"], m["loser_id"], m["surface"])
            if year in VALIDATION_YEARS:
                val.append(((m["tourney_id"], m["match_num"]), p_blended))
            else:
                train.append(p_blended)
        state.update(m["winner_id"], m["loser_id"], m["surface"])
    return val, train


def _calibrated(blended_list, a, b):
    return [calibrated_prob(p, a=a, b=b) for p in blended_list]


def main():
    conn = get_connection()
    matches = _load_matches(conn)
    print(f"Matches loaded (to 2024): {len(matches)}")
    matched_keys = _pinnacle_matched_keys(matches)
    print(f"Pinnacle-matched validation matches: {len(matched_keys)}")

    # 12 walks, one per (K, prior); calibration scored offline.
    walks = {}
    for k in K_VALUES:
        for prior in PRIOR_VALUES:
            val, train = _walk(matches, k, prior)
            walks[(k, prior)] = (val, train)
            print(f"  walk K={k} prior={prior}: "
                  f"val n={len(val)} train n={len(train)}")

    results = []
    calibrated_val = {}  # (k, prior, a, b) -> [p_winner] validation order
    for (k, prior), (val, _train) in walks.items():
        keys = [key for key, _ in val]
        blended = [p for _, p in val]
        for a in CAL_A_VALUES:
            for b in CAL_B_VALUES:
                p_list = _calibrated(blended, a, b)
                calibrated_val[(k, prior, a, b)] = p_list
                metrics = score_outcome(p_list)
                results.append({
                    "k": k, "prior": prior, "a": a, "b": b,
                    "val_n": metrics["n"],
                    "val_brier": metrics["brier"],
                    "val_log_loss": metrics["log_loss"],
                    "val_accuracy": metrics["accuracy"],
                })
    results.sort(key=lambda r: r["val_brier"])

    baseline_key = (BASELINE["k"], BASELINE["prior"], BASELINE["a"], BASELINE["b"])
    baseline = next(r for r in results
                    if (r["k"], r["prior"], r["a"], r["b"]) == baseline_key)
    best = results[0]

    # Paired difference best vs baseline, same validation matches.
    diff = _paired_variant_diff(
        calibrated_val[(best["k"], best["prior"], best["a"], best["b"])],
        calibrated_val[baseline_key])
    # Note: _paired_variant_diff returns D = first minus second.
    paired = {"n": diff["n"], "D_best_minus_baseline": diff["D"],
              "se": diff["se"], "z": diff["z"]}

    # Best config vs Pinnacle, same matched set as the baseline run.
    best_val, _ = walks[(best["k"], best["prior"])]
    best_blended = dict(best_val)
    model_p, market_p = [], []
    for key in [key for key, _ in best_val]:
        if key in matched_keys:
            model_p.append(calibrated_prob(best_blended[key], a=best["a"], b=best["b"]))
            market_p.append(matched_keys[key])
    vs_pinnacle = paired_diff(model_p, market_p)

    # Best config's train-period Brier (sanity: no wild overfit).
    _val, best_train = walks[(best["k"], best["prior"])]
    best_train_metrics = score_outcome(_calibrated(best_train, best["a"], best["b"]))

    adopted = bool(paired["D_best_minus_baseline"] < 0
                   and abs(paired["D_best_minus_baseline"]) >= 2 * paired["se"])
    verdict = ("ADOPT" if adopted else "KEEP BASELINE")
    print()
    print("Top 10 configs by validation Brier (main-tour non-team, "
          f"n={best['val_n']}):")
    for i, r in enumerate(results[:10], 1):
        print(f"  {i:2d}. K={r['k']:2d} prior={r['prior']:3d} a={r['a']:.1f} "
              f"b={r['b']:+.1f}  Brier={r['val_brier']:.4f} "
              f"logloss={r['val_log_loss']:.4f} acc={r['val_accuracy']:.4f}")
    print(f"Baseline K=32 prior=100 a=0.8 b=0.0: "
          f"Brier={baseline['val_brier']:.4f}")
    print(f"Best vs baseline (paired): D={paired['D_best_minus_baseline']:.4f} "
          f"se={paired['se']:.4f} z={paired['z']:.2f}")
    print(f"Best vs Pinnacle: D={vs_pinnacle['D']:.4f} "
          f"se={vs_pinnacle['se']:.4f} z={vs_pinnacle['z']:.2f} "
          f"(n={vs_pinnacle['n']})")
    print(f"Best train Brier (2001-2022): {best_train_metrics['brier']:.4f}")
    print(f"Verdict: {verdict} "
          f"(adopt only if best beats baseline by >= 2 SE)")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "grid_search.json"
    with open(out, "w") as f:
        json.dump({
            "run_date": _dt.date.today().isoformat(),
            "grid": {"k": list(K_VALUES), "prior": list(PRIOR_VALUES),
                     "cal_a": list(CAL_A_VALUES), "cal_b": list(CAL_B_VALUES),
                     "n_combos": len(results), "n_walks": len(walks)},
            "scoring_set": "main-tour non-team matches, validation 2023-2024",
            "selection_rule": "adopt best only if paired Brier difference "
                              "vs locked baseline <= -2 SE on validation; "
                              "otherwise keep locked baseline. Test set "
                              "(2025-2026) never touched. Calibration "
                              "intercept b fixed at 0.0 (see script "
                              "docstring: b != 0 leaks through winner-side "
                              "scoring and breaks A/B symmetry).",
            "baseline": baseline,
            "best": best,
            "best_train": {"n": best_train_metrics["n"],
                           "brier": best_train_metrics["brier"],
                           "log_loss": best_train_metrics["log_loss"],
                           "accuracy": best_train_metrics["accuracy"]},
            "paired_best_vs_baseline": paired,
            "best_vs_pinnacle": vs_pinnacle,
            "verdict": verdict,
            "ranked": results,
        }, f, indent=2)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
