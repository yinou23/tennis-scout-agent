#!/usr/bin/env python3
"""Locked test: frozen 20-column model vs locked baseline, test 2025-2026.

One-shot protocol (agreed 2026-11-19, data merged 2026-11-24):
- The 20 columns, C=0.01, l1_ratio=0.0 and coefficients are FROZEN from
  logreg_round4.json 'final' (validation Brier 0.2121, paired vs
  baseline D=-0.0064, z=-6.65). Nothing is tuned on the test set.
- The runner first REFITS the 20 columns on train (2001-2022) with the
  frozen hyperparameters and asserts it reproduces the validation
  Brier (0.2121 +- 0.0005) and the frozen coefficients. That refit is
  the frozen model (same data, same procedure); the test predictions
  come from it. If the refit does not reproduce, the run aborts
  before touching the test set.
- Test set: main-tour non-team matches, 2025-2026, walk-forward
  features (same build_rows as run_logreg_round4, split extended to
  'test'). Scored once, never re-run for tuning.
- Market gap: Pinnacle closing odds from tennis-data 2025.xlsx
  (neighbouring-year pool 2024+2025), aligned by align_matches.
  No 2026 odds file exists, so the market gap is 2025-only.
- Near-line candidate re-checks (protocol: adopt only if paired z
  vs the 20-col model on the test set is <= -2.0): ret_self,
  event_games, serve-joint. Each is fit on TRAIN with the 20 cols +
  candidate (frozen hyperparameters), then scored on TEST. Home-x-Elo
  and the Glicko gate need columns outside the round-4 build and are
  NOT re-checked here; they stay candidates for a follow-up.

Writes scripts/backtest/outputs/locked_test.json (git-ignored).

Usage (repo root, project venv):
    .venv/bin/python -m scripts.backtest.run_locked_test
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from align import align_matches  # noqa: E402
from db import get_connection  # noqa: E402
from logreg import (  # noqa: E402
    COL,
    Standardizer,
    fit_logreg,
    load_dob,
    load_matches_with_serve,
    load_year_end_rankings,
    predict_proba,
    tour_mean_minutes,
)
from metrics import devig, paired_diff, score_outcome  # noqa: E402
from names import name_key  # noqa: E402
from odds import read_tennis_data_xlsx  # noqa: E402
from run_logreg_round4 import build_rows  # noqa: E402
from run_logreg_serve import _paired_vs_baseline, _p_winner  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent / "outputs"
DATA_DIR = Path(__file__).resolve().parent / "data"

FROZEN = json.loads(
    (OUT_DIR / "logreg_round4.json").read_text()
)["final"]
FINAL_COLS = [COL[n] for n in FROZEN["adopted_columns"]]
FINAL_NAMES = list(FROZEN["adopted_columns"])
FROZEN_C = FROZEN["C"]
FROZEN_L1 = FROZEN["l1_ratio"]
FROZEN_COEF = FROZEN["coef"]
FROZEN_VAL_BRIER = FROZEN["val_brier"]
BASELINE_VAL_BRIER = 0.21846441507745834


def _fit_frozen(train_rows, cols):
    """Fit the frozen hyperparameters on train. Returns (clf, scaler)."""
    Xtr = np.array([r["x"] for r in train_rows])[:, cols]
    ytr = np.array([r["y"] for r in train_rows])
    scaler = Standardizer().fit(Xtr)
    clf = fit_logreg(scaler.transform(Xtr), ytr, FROZEN_C, FROZEN_L1)
    return clf, scaler


def _p_winner_list(clf, scaler, rows, cols):
    X = np.array([r["x"] for r in rows])[:, cols]
    p_a = predict_proba(clf, scaler.transform(X))
    return [_p_winner(p, r["y"]) for p, r in zip(p_a, rows)]


def _matched_pinnacle_2025(matches):
    """(tourney_id, match_num) -> Pinnacle prob of the winner, 2025.

    Same alignment as run_logreg_serve._matched_pinnacle_keys but
    for the test year: odds pool is the neighbouring-year files
    2024 + 2025 (no 2026 file exists).
    """
    rows = []
    for fy in (2024, 2025):
        p = DATA_DIR / f"{fy}.xlsx"
        if p.exists():
            rows += read_tennis_data_xlsx(str(p), fy)
    year_matches = [m for m in matches if m["tourney_date"].year == 2025]
    matched, unmatched = align_matches(year_matches, rows)
    matched_keys = {}
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
    from collections import Counter
    reasons = Counter(reason for _, reason in unmatched)
    return matched_keys, len(year_matches), dict(reasons)


def main():
    conn = get_connection()
    matches = load_matches_with_serve(conn, end_year=2026)
    dob = load_dob(conn)
    rankings = load_year_end_rankings(conn)
    tour_mean_min = tour_mean_minutes(matches)
    print(f"Matches loaded (to 2026): {len(matches)}; "
          f"tour mean minutes (train) = {tour_mean_min:.1f}; "
          f"year-end ranking rows = {len(rankings)}")

    rows, cov = build_rows(matches, dob, tour_mean_min, rankings)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validate"]
    test_rows = [r for r in rows if r["split"] == "test"]
    test25 = [r for r in test_rows if r["year"] == 2025]
    test26 = [r for r in test_rows if r["year"] == 2026]
    print(f"Rows: train={len(train_rows)} validate={len(val_rows)} "
          f"test={len(test_rows)} (2025={len(test25)}, 2026={len(test26)})")
    print(f"Test coverage: point-serve present "
          f"{cov['point_serve_ok']}/{cov['point_serve_n']}; "
          f"clutch nonzero {cov['clutch_tb_any']}/{cov['clutch_tb_n']}; "
          f"retirement flag nonzero "
          f"{cov['ret_flag_nonzero']}/{cov['ret_flag_n']}")
    assert test_rows, "no test rows: is the DB merged through 2026?"

    # --- sanity: refit must reproduce the frozen validation result ---
    clf, scaler = _fit_frozen(train_rows, FINAL_COLS)
    p_win_val = _p_winner_list(clf, scaler, val_rows, FINAL_COLS)
    val_brier = score_outcome(p_win_val)["brier"]
    coef = [float(c) for c in clf.coef_[0]]
    max_coef_diff = max(abs(a - b) for a, b in zip(coef, FROZEN_COEF))
    print(f"Refit check: val Brier={val_brier:.4f} "
          f"(frozen {FROZEN_VAL_BRIER:.4f}), "
          f"max |coef diff|={max_coef_diff:.2e}")
    assert abs(val_brier - FROZEN_VAL_BRIER) < 0.0005, (
        f"refit val Brier {val_brier:.4f} does not reproduce "
        f"frozen {FROZEN_VAL_BRIER:.4f}; aborting before the test set")
    assert max_coef_diff < 1e-6, (
        f"refit coefficients differ from frozen by {max_coef_diff:.2e}")

    # --- primary: frozen 20-col vs locked baseline on the test set ---
    base_p_win = [r["p_base_winner"] for r in test_rows]
    model_p_win = _p_winner_list(clf, scaler, test_rows, FINAL_COLS)
    base = score_outcome(base_p_win)
    model = score_outcome(model_p_win)
    paired = _paired_vs_baseline(model_p_win, base_p_win)
    print(f"TEST 2025-2026: n={len(test_rows)}")
    print(f"  baseline Brier={base['brier']:.4f} "
          f"logloss={base['log_loss']:.4f} acc={base['accuracy']:.4f}")
    print(f"  20-col    Brier={model['brier']:.4f} "
          f"logloss={model['log_loss']:.4f} acc={model['accuracy']:.4f}")
    print(f"  paired vs baseline: D={paired['D']:.4f} "
          f"se={paired['se']:.4f} z={paired['z']:.2f}")
    by_year = {}
    for label, subset in (("2025", test25), ("2026", test26)):
        idx = [i for i, r in enumerate(test_rows) if r["year"] == int(label)]
        b = score_outcome([base_p_win[i] for i in idx])
        m = score_outcome([model_p_win[i] for i in idx])
        p = _paired_vs_baseline([model_p_win[i] for i in idx],
                                [base_p_win[i] for i in idx])
        by_year[label] = {"n": len(idx), "baseline_brier": b["brier"],
                          "model_brier": m["brier"],
                          "paired": p}
        print(f"  {label}: n={len(idx)} baseline={b['brier']:.4f} "
              f"model={m['brier']:.4f} D={p['D']:.4f} z={p['z']:.2f}")

    # --- market gap (2025 only; no 2026 odds file) ---
    matched_keys, n2025_scored, unmatched_reasons = _matched_pinnacle_2025(
        matches)
    test_pos = {r["key"]: i for i, r in enumerate(test_rows)}
    keys = [k for k in matched_keys if k in test_pos]
    market = {"n_scored_2025": n2025_scored,
              "n_matched": len(keys),
              "coverage": len(keys) / n2025_scored if n2025_scored else 0.0,
              "unmatched_reasons": unmatched_reasons}
    if keys:
        p_win_m = [model_p_win[test_pos[k]] for k in keys]
        p_win_b = [base_p_win[test_pos[k]] for k in keys]
        mkt = [matched_keys[k] for k in keys]
        market["model_vs_pinnacle"] = paired_diff(p_win_m, mkt)
        market["baseline_vs_pinnacle"] = paired_diff(p_win_b, mkt)
        print(f"  market (2025): matched {len(keys)}/{n2025_scored} "
              f"({market['coverage']:.1%}); model vs Pinnacle "
              f"D={market['model_vs_pinnacle']['D']:.4f}; baseline vs "
              f"Pinnacle D={market['baseline_vs_pinnacle']['D']:.4f}")
    else:
        print("  market (2025): no matched Pinnacle rows")

    # --- near-line candidate re-checks (fit on train, score on test) ---
    candidates = {
        "ret_self": [COL["ret_self_diff"]],
        "event_games": [COL["event_games_diff"]],
        "serve_joint": [COL["first_win_diff"], COL["second_win_diff"],
                        COL["ace_rate_diff"], COL["df_rate_diff"],
                        COL["bp_conv_diff"], COL["bp_save_diff"]],
    }
    cand_results = {}
    for label, extra in candidates.items():
        cols = FINAL_COLS + extra
        c_clf, c_scaler = _fit_frozen(train_rows, cols)
        c_p_win = _p_winner_list(c_clf, c_scaler, test_rows, cols)
        c_score = score_outcome(c_p_win)
        c_paired = _paired_vs_baseline(c_p_win, model_p_win)
        cand_results[label] = {
            "cols": extra, "test_brier": c_score["brier"],
            "paired_vs_20col": c_paired,
            "adopt": bool(c_paired["z"] <= -2.0),
        }
        print(f"  candidate {label}: test Brier={c_score['brier']:.4f} "
              f"vs 20-col D={c_paired['D']:.4f} z={c_paired['z']:.2f} "
              f"-> {'ADOPT' if c_paired['z'] <= -2.0 else 'DROP'}")

    out = {
        "model": "frozen 20-col logreg (logreg_round4.json final), "
                 "C=0.01 l1=0.0, no intercept",
        "frozen_val_brier": FROZEN_VAL_BRIER,
        "refit_val_brier": val_brier,
        "refit_max_coef_diff": max_coef_diff,
        "test": {"n": len(test_rows), "n_2025": len(test25),
                 "n_2026": len(test26),
                 "baseline": base, "model": model, "paired": paired,
                 "by_year": by_year},
        "market_2025": market,
        "candidates": cand_results,
        "coverage": cov,
    }
    (OUT_DIR / "locked_test.json").write_text(json.dumps(out, indent=1))
    print("wrote scripts/backtest/outputs/locked_test.json")


if __name__ == "__main__":
    main()
