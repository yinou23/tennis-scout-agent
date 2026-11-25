#!/usr/bin/env python3
"""Re-check two near-line candidates on the locked test set (2025-2026).

Candidates (both were near the z <= -2 adoption line on validation,
but needed columns outside the round-4 50-col build, so the locked
test run did not cover them):

1. home_x_elo: home_diff * elo_diff, the signed interaction from
   run_logreg_home_x.py (validation z = -1.57 vs the 20-col base).
   home_diff is the A-minus-B home flag from
   run_logreg_benchmark._home_diff (player IOC vs tournament
   country); elo_diff is the row's blended Elo diff.
2. glicko-gate (elo_x_highrd): elo_diff when max(rd_a, rd_b) >= the
   train-median max-RD threshold, else 0, from
   run_logreg_alternatives.py / features.elo_high_rd_interaction
   (validation z = -1.55 vs the all-batches adopted set, val Brier
   0.2166 -- a DIFFERENT base than the 20-col model; the re-check
   here is vs the frozen 20-col model, same as the other three
   candidates in run_locked_test.py). RDs from
   glicko.compute_glicko_features; threshold = median max-RD over
   train scored matches only. The original threshold was
   84.52718750458531; this run asserts it reproduces (train data is
   unchanged by the 2026 merge).

Protocol (identical to run_locked_test.py): for each candidate, fit
the 20 columns + candidate on TRAIN (2001-2022) with the frozen
hyperparameters (C=0.01, l1_ratio=0.0, no intercept, Standardizer
fit on train only), score on TEST (2025-2026), and compute the paired
difference vs the frozen 20-col model on the test set. Adopt only if
paired z <= -2.0. Nothing is tuned on the test set; the 20-col refit
must first reproduce the frozen validation Brier (0.2121) before any
test row is scored.

This script is read-only analysis: it writes only
scripts/backtest/outputs/candidate_recheck_home_glicko.json. It does
not modify production code, does not commit/push, and does not
re-run the locked test.

Usage (repo root, project venv):
    .venv/bin/python -m scripts.backtest.run_candidate_recheck_home_glicko
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_connection  # noqa: E402
from features import elo_high_rd_interaction  # noqa: E402
from glicko import compute_glicko_features  # noqa: E402
from logreg import (  # noqa: E402
    COL,
    load_dob,
    load_ioc,
    load_matches_with_serve,
    load_year_end_rankings,
    tour_mean_minutes,
)
from metrics import score_outcome  # noqa: E402
from run_locked_test import (  # noqa: E402
    FINAL_COLS,
    FROZEN_VAL_BRIER,
    _fit_frozen,
    _p_winner_list,
)
from run_logreg_home_x import _add_home_columns  # noqa: E402
from run_logreg_round4 import _is_scored, build_rows  # noqa: E402
from run_logreg_alternatives import rd_median_threshold  # noqa: E402
from run_logreg_serve import _paired_vs_baseline  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent / "outputs"

ORIG_RD_THRESHOLD = 84.52718750458531  # from logreg_alternatives.json


def _add_glicko_gate(rows, matches, glicko, threshold):
    """Set COL['elo_x_highrd'] on each row (index 27, inside the
    50-col build but left at 0.0 by round-4 build_rows).

    Alignment is the same walk as _add_home_columns: scored matches
    (year >= 2001, main-tour non-team) in walk order, zipped with
    rows. Returns a coverage dict for test rows.
    """
    cov = {"test_n": 0, "test_gate_on": 0, "test_gate_on_2025": 0,
           "test_gate_on_2026": 0}
    scored = [m for m in matches
              if m["tourney_date"].year >= 2001 and _is_scored(m)]
    assert len(scored) == len(rows), (len(scored), len(rows))
    for r, m in zip(rows, scored):
        rd_a, rd_b = glicko.get((m["tourney_id"], m["match_num"]),
                                (350.0, 350.0))
        elo_d = r["x"][COL["elo_diff"]]
        r["x"][COL["elo_x_highrd"]] = elo_high_rd_interaction(
            elo_d, rd_a, rd_b, threshold)
        if r["split"] == "test":
            cov["test_n"] += 1
            if r["x"][COL["elo_x_highrd"]] != 0.0:
                cov["test_gate_on"] += 1
                if r["year"] == 2025:
                    cov["test_gate_on_2025"] += 1
                elif r["year"] == 2026:
                    cov["test_gate_on_2026"] += 1
    return cov


def main():
    conn = get_connection()
    matches = load_matches_with_serve(conn, end_year=2026)
    dob = load_dob(conn)
    rankings = load_year_end_rankings(conn)
    ioc = load_ioc(conn)
    tour_mean_min = tour_mean_minutes(matches)
    print(f"Matches loaded (to 2026): {len(matches)}; "
          f"tour mean minutes (train) = {tour_mean_min:.1f}")

    rows, cov = build_rows(matches, dob, tour_mean_min, rankings)
    home_cov = _add_home_columns(rows, matches, ioc)
    # _add_home_columns counts coverage over year>=2023
    # (validation + test); compute the test-only home coverage here.
    test_home_nonzero = sum(
        1 for r in rows
        if r["split"] == "test"
        and r["x"][COL["home_diff"]] != 0.0)

    glicko = compute_glicko_features(matches)
    print(f"Glicko features computed for {len(glicko)} matches")
    rd_threshold = rd_median_threshold(matches, glicko)
    print(f"RD gate threshold (median max-RD, train scored) = "
          f"{rd_threshold:.4f} (original {ORIG_RD_THRESHOLD:.4f})")
    assert abs(rd_threshold - ORIG_RD_THRESHOLD) < 0.01, (
        f"RD threshold {rd_threshold:.4f} does not reproduce the "
        f"original {ORIG_RD_THRESHOLD:.4f}; train RDs changed, aborting")
    gate_cov = _add_glicko_gate(rows, matches, glicko, rd_threshold)

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validate"]
    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"Rows: train={len(train_rows)} validate={len(val_rows)} "
          f"test={len(test_rows)}")
    print(f"Home coverage on test: flag nonzero "
          f"{test_home_nonzero}/{len(test_rows)}")
    print(f"Glicko gate on test: {gate_cov['test_gate_on']}/"
          f"{gate_cov['test_n']} matches gated "
          f"(2025: {gate_cov['test_gate_on_2025']}, "
          f"2026: {gate_cov['test_gate_on_2026']})")
    assert test_rows, "no test rows: is the DB merged through 2026?"

    # --- sanity: refit must reproduce the frozen validation result ---
    clf, scaler = _fit_frozen(train_rows, FINAL_COLS)
    p_win_val = _p_winner_list(clf, scaler, val_rows, FINAL_COLS)
    val_brier = score_outcome(p_win_val)["brier"]
    print(f"Refit check: val Brier={val_brier:.4f} "
          f"(frozen {FROZEN_VAL_BRIER:.4f})")
    assert abs(val_brier - FROZEN_VAL_BRIER) < 0.0005, (
        f"refit val Brier {val_brier:.4f} does not reproduce "
        f"frozen {FROZEN_VAL_BRIER:.4f}; aborting before the test set")

    # --- frozen 20-col model on the test set (the reference) ---
    model_p_win = _p_winner_list(clf, scaler, test_rows, FINAL_COLS)
    model = score_outcome(model_p_win)
    print(f"20-col on test: n={len(test_rows)} "
          f"Brier={model['brier']:.4f}")

    # --- candidate re-checks (fit on train, score on test) ---
    candidates = {
        "home_x_elo": {
            "cols": [COL["home_x_elo"]],
            "definition": "home_diff * elo_diff; home_diff is the "
                          "A-minus-B home flag (-1/0/1) from player "
                          "IOC vs tournament country "
                          "(run_logreg_benchmark._home_diff, country "
                          "from features.tournament_country); "
                          "elo_diff is the row's blended Elo diff. "
                          "Validation z=-1.57 vs the 20-col base.",
        },
        "glicko_gate": {
            "cols": [COL["elo_x_highrd"]],
            "definition": "elo_diff when max(rd_a, rd_b) >= "
                          f"{rd_threshold:.4f} (median max-RD over "
                          "train scored matches), else 0 "
                          "(features.elo_high_rd_interaction; RDs "
                          "from glicko.compute_glicko_features, "
                          "pre-week state). Validation z=-1.55 vs "
                          "the all-batches adopted set (0.2166), a "
                          "different base than the 20-col model.",
        },
    }
    cand_results = {}
    for label, spec in candidates.items():
        cols = FINAL_COLS + spec["cols"]
        c_clf, c_scaler = _fit_frozen(train_rows, cols)
        c_p_win = _p_winner_list(c_clf, c_scaler, test_rows, cols)
        c_score = score_outcome(c_p_win)
        c_paired = _paired_vs_baseline(c_p_win, model_p_win)
        # By-year split, for context (same paired test per year).
        by_year = {}
        for yr in (2025, 2026):
            idx = [i for i, r in enumerate(test_rows) if r["year"] == yr]
            p = _paired_vs_baseline([c_p_win[i] for i in idx],
                                    [model_p_win[i] for i in idx])
            by_year[str(yr)] = {"n": len(idx), "paired": p}
        cand_results[label] = {
            "cols": [int(c) for c in spec["cols"]],
            "definition": spec["definition"],
            "test_brier": c_score["brier"],
            "test_log_loss": c_score["log_loss"],
            "test_accuracy": c_score["accuracy"],
            "paired_vs_20col": c_paired,
            "by_year": by_year,
            "adopt": bool(c_paired["z"] <= -2.0),
        }
        print(f"  candidate {label}: test Brier={c_score['brier']:.4f} "
              f"vs 20-col D={c_paired['D']:.4f} se={c_paired['se']:.4f} "
              f"z={c_paired['z']:.2f} "
              f"-> {'ADOPT' if c_paired['z'] <= -2.0 else 'DROP'}")

    out = {
        "model": "frozen 20-col logreg (logreg_round4.json final), "
                 "C=0.01 l1=0.0, no intercept",
        "frozen_val_brier": FROZEN_VAL_BRIER,
        "refit_val_brier": val_brier,
        "rd_threshold": rd_threshold,
        "rd_threshold_original": ORIG_RD_THRESHOLD,
        "test": {"n": len(test_rows),
                 "n_2025": sum(1 for r in test_rows if r["year"] == 2025),
                 "n_2026": sum(1 for r in test_rows if r["year"] == 2026),
                 "model_20col": model},
        "coverage": {"home_year_ge_2023": home_cov,
                     "home_test_nonzero": test_home_nonzero,
                     "glicko_gate": gate_cov},
        "candidates": cand_results,
    }
    (OUT_DIR / "candidate_recheck_home_glicko.json").write_text(
        json.dumps(out, indent=1))
    print("wrote scripts/backtest/outputs/"
          "candidate_recheck_home_glicko.json")


if __name__ == "__main__":
    main()
