#!/usr/bin/env python3
"""Refit the frozen 20-column model on 2001-2024 (train + validation).

Purpose (user decision 2026-11-24): the locked test confirmed the
20-column feature set (test Brier 0.2157 vs baseline 0.2192, z=-3.59),
so refit the SAME 20 columns with the SAME frozen hyperparameters
(C=0.01, l1_ratio=0.0, no intercept) on all pre-2025 data, and ARCHIVE
the new coefficients. This does NOT change the production model
(the deployed backend still uses the Elo model); it only saves a
refit coefficient set for a future deployment decision.

Output: scripts/backtest/outputs/logreg_refit_2001_2024.json with
columns, C, l1_ratio, coef, and the train-fitted Standardizer
(mean/scale) needed to apply it, plus a test-set sanity check
(2025-2026 Brier and paired comparison vs the frozen model and vs
the baseline). The test set is used only as a check, never for
tuning: the hyperparameters and columns are frozen.

Usage (repo root, project venv):
    .venv/bin/python -m scripts.backtest.run_refit_2001_2024
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from metrics import score_outcome  # noqa: E402
from run_locked_test import _p_winner_list  # noqa: E402
from run_logreg_round4 import build_rows  # noqa: E402
from run_logreg_serve import _paired_vs_baseline  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"

FROZEN = json.loads((OUT_DIR / "logreg_round4.json").read_text())["final"]
FINAL_COLS = [COL[n] for n in FROZEN["adopted_columns"]]
FINAL_NAMES = list(FROZEN["adopted_columns"])
LOCKED_TEST = json.loads((OUT_DIR / "locked_test.json").read_text())


def main():
    conn = get_connection()
    matches = load_matches_with_serve(conn, end_year=2026)
    dob = load_dob(conn)
    rankings = load_year_end_rankings(conn)
    tour_mean_min = tour_mean_minutes(matches)

    rows, _cov = build_rows(matches, dob, tour_mean_min, rankings)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validate"]
    test_rows = [r for r in rows if r["split"] == "test"]
    fit_rows = train_rows + val_rows
    print(f"Refit rows: 2001-2024 n={len(fit_rows)} "
          f"(train={len(train_rows)}, validate={len(val_rows)}); "
          f"test n={len(test_rows)} (check only)")

    X = np.array([r["x"] for r in fit_rows])[:, FINAL_COLS]
    y = np.array([r["y"] for r in fit_rows])
    scaler = Standardizer().fit(X)
    clf = fit_logreg(scaler.transform(X), y, FROZEN["C"], FROZEN["l1_ratio"])
    coef = [float(c) for c in clf.coef_[0]]

    # Sanity check on the test set (out-of-sample for the refit too,
    # since test is 2025-2026). Not used for tuning. The frozen model's
    # per-match test predictions are recomputed by refitting it on
    # train only; the locked test verified that refit reproduces the
    # frozen coefficients exactly (max diff 0.00e+00).
    from run_locked_test import _fit_frozen
    frozen_clf, frozen_scaler = _fit_frozen(train_rows, FINAL_COLS)
    frozen_p_win = _p_winner_list(frozen_clf, frozen_scaler, test_rows,
                                  FINAL_COLS)
    refit_p_win = _p_winner_list(clf, scaler, test_rows, FINAL_COLS)
    base_p_win = [r["p_base_winner"] for r in test_rows]
    refit = score_outcome(refit_p_win)
    frozen_test_brier = LOCKED_TEST["test"]["model"]["brier"]
    paired_vs_frozen = _paired_vs_baseline(refit_p_win, frozen_p_win)
    paired_vs_baseline = _paired_vs_baseline(refit_p_win, base_p_win)
    print(f"Refit test Brier={refit['brier']:.4f} "
          f"(frozen model test Brier={frozen_test_brier:.4f}); "
          f"paired vs frozen: D={paired_vs_frozen['D']:.4f} "
          f"z={paired_vs_frozen['z']:.2f}")
    print(f"Refit test vs baseline: D={paired_vs_baseline['D']:.4f} "
          f"z={paired_vs_baseline['z']:.2f}")

    out = {
        "model": "20-col logreg, refit on 2001-2024 (train+validate), "
                 "frozen hyperparameters C=0.01 l1=0.0, no intercept",
        "columns": FINAL_NAMES,
        "C": FROZEN["C"],
        "l1_ratio": FROZEN["l1_ratio"],
        "coef": coef,
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "n_fit": len(fit_rows),
        "frozen_val_brier": FROZEN["val_brier"],
        "frozen_test_brier": frozen_test_brier,
        "refit_test_brier": refit["brier"],
        "refit_test_log_loss": refit["log_loss"],
        "refit_test_accuracy": refit["accuracy"],
        "refit_test_paired_vs_frozen": paired_vs_frozen,
        "note": "Archive only. Production still uses the Elo model; "
                "deploying these coefficients is a separate decision.",
    }
    (OUT_DIR / "logreg_refit_2001_2024.json").write_text(
        json.dumps(out, indent=1))
    print("wrote scripts/backtest/outputs/logreg_refit_2001_2024.json")


if __name__ == "__main__":
    main()
