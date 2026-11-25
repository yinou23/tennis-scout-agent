"""Run the Elastic Net container, first feature batch: serve/return.

Walk 1968-2024 once (locked Elo + walk-forward serve state), build
one row per scored main-tour non-team match (year >= 2001), fit on
train 2001-2022, select C / l1_ratio on validation 2023-2024 only.

Ablation reported:
  (a) elo-only container (sanity: should land near the locked
      baseline validation Brier 0.2185)
  (b) elo + serve/return container

For each: validation Brier / log loss / accuracy, paired Brier
difference vs the locked baseline on the same 5,464 matches (D, SE,
z), and vs-Pinnacle D on the same tennis-data matched subset
(n = 5,202; alignment procedure copied from ablate_team.py).

Adoption rule: a feature batch is kept only if it beats the baseline
by >= 2 SE on validation; otherwise it is reported as no evidence and
stays out of the model. Nothing here changes the production model.

Usage (repo root, project venv):
    .venv/bin/python -m scripts.backtest.run_logreg_serve

Writes scripts/backtest/outputs/logreg_serve.json (git-ignored).
"""

import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from align import MAIN_TOUR_LEVELS, _is_team_exhibition, align_matches  # noqa: E402
from db import get_connection  # noqa: E402
from elo_state import EloState  # noqa: E402
from features import ServeState  # noqa: E402
from logreg import (  # noqa: E402
    ELO_ONLY_NAMES,
    FEATURE_NAMES,
    Standardizer,
    fit_logreg,
    load_matches_with_serve,
    predict_proba,
)
from metrics import devig, paired_diff, score_outcome  # noqa: E402
from names import name_key  # noqa: E402
from odds import read_tennis_data_xlsx  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent / "outputs"

C_GRID = [0.01, 0.1, 1.0]
L1_GRID = [0.0, 0.5, 1.0]


def _is_scored(m):
    return m["tourney_level"] in MAIN_TOUR_LEVELS and not _is_team_exhibition(m["tourney_name"])


def _split_for(year):
    return "train" if year <= 2022 else "validate"


def build_rows(matches):
    """One walk. Returns rows for scored matches, year >= 2001.

    Row: {"year", "split", "x": [elo_diff, hold_diff, break_diff],
    "y": 1 if lower-id player (A) won, "p_base_winner": locked-model
    probability of the eventual winner, "key": (tourney_id, match_num)}.
    """
    elo = EloState()
    serve = ServeState()
    rows = []
    for m in matches:
        year = m["tourney_date"].year
        if year >= 2001 and _is_scored(m):
            w_id, l_id = m["winner_id"], m["loser_id"]
            a_id, b_id = (w_id, l_id) if w_id < l_id else (l_id, w_id)
            y = 1 if a_id == w_id else 0
            blended_a, _, _, _ = elo.blended(a_id, m["surface"])
            blended_b, _, _, _ = elo.blended(b_id, m["surface"])
            hold_d, brk_d = serve.diffs(a_id, b_id, m["surface"])
            p_base_winner, _, _, _ = elo.predict(w_id, l_id, m["surface"])
            rows.append({
                "year": year,
                "split": _split_for(year),
                "x": [blended_a - blended_b, hold_d, brk_d],
                "y": y,
                "p_base_winner": p_base_winner,
                "key": (m["tourney_id"], m["match_num"]),
            })
        # Update AFTER features were read: no leakage by construction.
        elo.update(m["winner_id"], m["loser_id"], m["surface"])
        serve.update(
            m["winner_id"], m["loser_id"], m["surface"],
            m["w_sv_gms"], m["w_bp_faced"], m["w_bp_saved"],
            m["l_sv_gms"], m["l_bp_faced"], m["l_bp_saved"],
        )
    return rows


def _p_winner(p_a, y):
    return p_a if y == 1 else 1.0 - p_a


def _paired_vs_baseline(p_winner_list, p_base_list):
    n = len(p_winner_list)
    ds = [(1.0 - p) ** 2 - (1.0 - pb) ** 2
          for p, pb in zip(p_winner_list, p_base_list)]
    mean_d = sum(ds) / n
    var = sum((d - mean_d) ** 2 for d in ds) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else 0.0
    return {"n": n, "D": mean_d, "se": se, "z": mean_d / se if se > 0 else 0.0}


def _matched_pinnacle_keys(matches):
    """(tourney_id, match_num) -> Pinnacle prob of the winner, 2023-2024.

    Same alignment procedure as ablate_team.py: tennis-data xlsx rows
    from the neighbouring-year pool, devigged Pinnacle odds.
    """
    odds_by_year = {}
    for year in (2022, 2023, 2024):
        rows = []
        for fy in (year - 1, year, year + 1):
            p = REPO_ROOT / "scripts" / "backtest" / "data" / f"{fy}.xlsx"
            if p.exists():
                rows += read_tennis_data_xlsx(str(p), fy)
        odds_by_year[year] = rows
    matched_keys = {}
    for year in (2023, 2024):
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


def _fit_and_select(train_rows, val_rows, cols, label):
    """Grid-search C/l1_ratio on validation Brier. Returns best result."""
    Xtr = np.array([r["x"] for r in train_rows])[:, cols]
    ytr = np.array([r["y"] for r in train_rows])
    Xva = np.array([r["x"] for r in val_rows])[:, cols]
    yva = np.array([r["y"] for r in val_rows])
    scaler = Standardizer().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
    p_base_va = [r["p_base_winner"] for r in val_rows]
    best = None
    grid = []
    for C in C_GRID:
        for l1 in L1_GRID:
            clf = fit_logreg(Xtr_s, ytr, C, l1)
            p_a = predict_proba(clf, Xva_s)
            p_win = [_p_winner(p, y) for p, y in zip(p_a, yva)]
            m = score_outcome(p_win)
            paired = _paired_vs_baseline(p_win, p_base_va)
            grid.append({
                "C": C, "l1_ratio": l1, "val_brier": m["brier"],
                "val_log_loss": m["log_loss"], "val_accuracy": m["accuracy"],
                "paired_vs_baseline": paired,
            })
            if best is None or m["brier"] < best["val_brier"]:
                best = dict(grid[-1])
                best["coef"] = [float(c) for c in clf.coef_[0]]
                best["p_winner_val"] = p_win
    # Train Brier for the selected config (sanity only).
    clf = fit_logreg(Xtr_s, ytr, best["C"], best["l1_ratio"])
    p_a_tr = predict_proba(clf, Xtr_s)
    best["train_brier"] = score_outcome(
        [_p_winner(p, y) for p, y in zip(p_a_tr, ytr)])["brier"]
    best["feature_names"] = [FEATURE_NAMES[c] for c in cols]
    best["n_train"] = len(train_rows)
    best["n_val"] = len(val_rows)
    print(f"  {label}: best C={best['C']} l1={best['l1_ratio']} "
          f"val Brier={best['val_brier']:.4f} train Brier={best['train_brier']:.4f} "
          f"paired D={best['paired_vs_baseline']['D']:.4f} "
          f"se={best['paired_vs_baseline']['se']:.4f} "
          f"z={best['paired_vs_baseline']['z']:.2f}")
    return best, grid


def main():
    conn = get_connection()
    matches = load_matches_with_serve(conn)
    print(f"Matches loaded (to 2024): {len(matches)}")
    rows = build_rows(matches)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validate"]
    print(f"Rows: train={len(train_rows)} validate={len(val_rows)}")

    elo_only, grid_a = _fit_and_select(train_rows, val_rows, [0], "elo-only")
    full, grid_b = _fit_and_select(train_rows, val_rows, [0, 1, 2], "elo+serve")

    # Vs-Pinnacle D for both, on the same matched validation subset.
    matched_keys = _matched_pinnacle_keys(matches)
    val_pos = {r["key"]: i for i, r in enumerate(val_rows)}
    keys = [k for k in matched_keys if k in val_pos]
    for result, tag in ((elo_only, "elo-only"), (full, "elo+serve")):
        p_win = [result["p_winner_val"][val_pos[k]] for k in keys]
        market = [matched_keys[k] for k in keys]
        result["vs_pinnacle"] = paired_diff(p_win, market)
        del result["p_winner_val"]
        print(f"  {tag} vs Pinnacle: D={result['vs_pinnacle']['D']:.4f} "
              f"se={result['vs_pinnacle']['se']:.4f} "
              f"z={result['vs_pinnacle']['z']:.2f} "
              f"(n={result['vs_pinnacle']['n']})")

    for result in (elo_only, full):
        d = result["paired_vs_baseline"]
        result["verdict"] = ("KEEP" if d["z"] <= -2.0 else "NO EVIDENCE")
    print(f"Verdict: elo-only {elo_only['verdict']}, "
          f"elo+serve {full['verdict']} "
          f"(keep only if paired z <= -2)")

    out = {
        "model": "logistic regression, no intercept, features as A-minus-B diffs",
        "run_date": _dt.date.today().isoformat(),
        "scoring_set": "main-tour non-team matches, validation 2023-2024",
        "baseline": {"val_brier": 0.21846441507745834, "vs_pinnacle_D": 0.01626537711788756},
        "grids": {"C": C_GRID, "l1_ratio": L1_GRID},
        "elo_only": {k: v for k, v in elo_only.items()},
        "elo_serve": {k: v for k, v in full.items()},
        "grid_elo_only": grid_a,
        "grid_elo_serve": grid_b,
    }
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "logreg_serve.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
