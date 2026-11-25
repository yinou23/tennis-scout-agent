"""Run ALL remaining Elastic Net feature batches, sequential forward
selection, and report once.

One walk (1968-2024) builds every feature column; batches are then
tested in the agreed order, each on top of the currently ADOPTED set
only. A batch is adopted only if it beats the current best by >= 2 SE
on validation (paired Brier difference on the same 5,464 main-tour
non-team matches, 2023-2024). Otherwise it is dropped and the walk
moves on. The production model is unchanged whatever the outcome.

Batches, in order:
  1. form-raw    recent-2-month + last-10 win% diffs
  2. form-resid  residual form (won - Elo-expected), only if raw fails
  3. fatigue     rest / games7 / games14 / prev-minutes diffs
  4. welo        games-weighted Elo diff (alongside Elo, never replacing)
  5. glicko-A    Glicko-2 RD diff
  6. glicko-B    RD diff + Elo diff shrunk by mean RD
  7. age         age diff + age x best-of-5 / Elo-closeness / load7
  8. h2h         shrunk head-to-head win% diff (k picked on validation)
  9. joint-serve adopted set + serve/return (separate row; serve was
                 NOT adopted on its own, z = -1.24, so it is only
                 tested jointly here, never silently re-added)

NO INTERCEPT anywhere (fit_intercept=False): production requires
p(A) + p(B) = 1, and winner-side scoring would leak through any
intercept (see logreg.py). Calibration intercept b is likewise never
searched. Fit on train 2001-2022; C / l1_ratio selected on validation
only; test set 2025-2026 is never touched.

Caveat (multiple comparisons): ~9 adoption tests run on the same
validation set at a 2-SE threshold each, so the family-wise chance
of at least one false adoption is higher than 2.5%. An adopted batch
is a candidate, not proof; the locked test set (2025-2026, after the
TML merge) is the confirmation.

Usage (repo root, project venv):
    .venv/bin/python -m scripts.backtest.run_logreg_all

Writes scripts/backtest/outputs/logreg_all_batches.json (git-ignored).
"""

import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from align import MAIN_TOUR_LEVELS, _is_team_exhibition  # noqa: E402
from db import get_connection  # noqa: E402
from elo_state import EloState  # noqa: E402
from features import (  # noqa: E402
    FatigueState,
    FormState,
    H2HState,
    ServeState,
    WEloState,
    age_years,
    parse_score,
)
from glicko import compute_glicko_features  # noqa: E402
from logreg import (  # noqa: E402
    COL,
    FEATURE_NAMES,
    load_dob,
    load_matches_with_serve,
    tour_mean_minutes,
)
from run_logreg_serve import (  # noqa: E402
    _fit_and_select,
    _matched_pinnacle_keys,
    _paired_vs_baseline,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent / "outputs"
BASELINE = {"val_brier": 0.21846441507745834, "vs_pinnacle_D": 0.01626537711788756}
# This runner builds the original 21 columns only. FEATURE_NAMES has
# grown since (alternatives appended in logreg.py), so the row width
# is pinned here instead of len(FEATURE_NAMES).
N_COLS = 21


def _is_scored(m):
    return m["tourney_level"] in MAIN_TOUR_LEVELS and not _is_team_exhibition(m["tourney_name"])


def _split_for(year):
    return "train" if year <= 2022 else "validate"


def build_rows(matches, dob, tour_mean_min):
    """One walk. Returns rows for scored matches, year >= 2001.

    Row: {"year", "split", "x": 21 features in FEATURE_NAMES order,
    "y": 1 if lower-id player (A) won, "p_base_winner": locked-model
    probability of the eventual winner, "key": (tourney_id, match_num)}.
    """
    glicko = compute_glicko_features(matches)
    print(f"Glicko features computed for {len(glicko)} matches")
    elo = EloState()
    serve = ServeState()
    form = FormState()
    fatigue = FatigueState(tour_mean_minutes=tour_mean_min)
    welo = WEloState()
    h2h = H2HState()
    rows = []
    for m in matches:
        date = m["tourney_date"]
        year = date.year
        w_id, l_id = m["winner_id"], m["loser_id"]
        a_id, b_id = (w_id, l_id) if w_id < l_id else (l_id, w_id)
        w_games, l_games, _sets, retired, walkover = parse_score(m.get("score"))
        total_games = w_games + l_games
        # Locked-model probability of the winner, for every match:
        # the form state needs it for burn-in matches too (a 2001
        # match's form history starts in 1968).
        p_base_winner, _, _, _ = elo.predict(w_id, l_id, m["surface"])
        if year >= 2001 and _is_scored(m):
            y = 1 if a_id == w_id else 0
            p_cal_a, p_blended_a, blended_a, blended_b = elo.predict(
                a_id, b_id, m["surface"])
            elo_diff = blended_a - blended_b
            hold_d, brk_d = serve.diffs(a_id, b_id, m["surface"])
            f2m, f10, r2m, r10 = form.diffs(a_id, b_id, date)
            rest_d, g7_d, g14_d, pmin_d = fatigue.diffs(a_id, b_id, date)
            welo_d = welo.diff(a_id, b_id, m["surface"])
            rd_a, rd_b = glicko.get(
                (m["tourney_id"], m["match_num"]), (350.0, 350.0))
            rd_d = rd_a - rd_b
            mean_rd = 0.5 * (rd_a + rd_b)
            elo_x_rd = elo_diff / (1.0 + mean_rd / 350.0)
            age_a = age_years(dob.get(a_id), date)
            age_b = age_years(dob.get(b_id), date)
            if age_a is None or age_b is None:
                age_d = 0.0
            else:
                age_d = age_a - age_b
            bo5 = 1.0 if m.get("best_of") == 5 else 0.0
            close = 1.0 - abs(2.0 * p_blended_a - 1.0)
            h_k3, h_k5, h_k10 = h2h.diffs(a_id, b_id)
            x = [
                elo_diff, hold_d, brk_d,
                f2m, f10, r2m, r10,
                rest_d, g7_d, g14_d, pmin_d,
                welo_d, rd_d, elo_x_rd,
                age_d, age_d * bo5, age_d * close, age_d * g7_d,
                h_k3, h_k5, h_k10,
            ]
            assert len(x) == N_COLS, (len(x), N_COLS)
            rows.append({
                "year": year,
                "split": _split_for(year),
                "x": x,
                "y": y,
                "p_base_winner": p_base_winner,
                "key": (m["tourney_id"], m["match_num"]),
            })
        # Update AFTER features were read: no leakage by construction.
        elo.update(w_id, l_id, m["surface"])
        serve.update(
            w_id, l_id, m["surface"],
            m["w_sv_gms"], m["w_bp_faced"], m["w_bp_saved"],
            m["l_sv_gms"], m["l_bp_faced"], m["l_bp_saved"],
        )
        form.update(w_id, l_id, date, p_base_winner)
        fatigue.update(w_id, l_id, date, total_games, m.get("minutes"))
        welo.update(w_id, l_id, m["surface"], w_games, total_games,
                    retired=retired, walkover=walkover)
        h2h.update(w_id, l_id)
    return rows


def _vs_pinnacle(result, val_rows, matched_keys):
    """Attach vs-Pinnacle D to a result dict (same matched subset)."""
    val_pos = {r["key"]: i for i, r in enumerate(val_rows)}
    keys = [k for k in matched_keys if k in val_pos]
    p_win = [result["p_winner_val"][val_pos[k]] for k in keys]
    market = [matched_keys[k] for k in keys]
    from metrics import paired_diff  # noqa: E402

    result["vs_pinnacle"] = paired_diff(p_win, market)
    return result


def _summarize(result, label, base_p_win, cur_p_win, val_rows, matched_keys):
    """Common reporting for one fitted variant."""
    p_win = result["p_winner_val"]
    result["paired_vs_baseline"] = _paired_vs_baseline(p_win, base_p_win)
    result["paired_vs_current_best"] = _paired_vs_baseline(p_win, cur_p_win)
    # Cast to plain Python types: paired diffs are built from numpy
    # probabilities, and np.bool_ / np.float64 do not all serialize
    # (np.float64 is a float subclass and passes, np.bool_ does not).
    for key in ("paired_vs_baseline", "paired_vs_current_best"):
        result[key] = {k: (float(v) if isinstance(v, float) else v)
                       for k, v in result[key].items()}
    _vs_pinnacle(result, val_rows, matched_keys)
    del result["p_winner_val"]
    d = result["paired_vs_current_best"]
    print(f"  {label}: C={result['C']} l1={result['l1_ratio']} "
          f"val Brier={result['val_brier']:.4f} "
          f"vs baseline D={result['paired_vs_baseline']['D']:.4f} "
          f"z={result['paired_vs_baseline']['z']:.2f} "
          f"vs current-best D={d['D']:.4f} z={d['z']:.2f} "
          f"vs Pinnacle D={result['vs_pinnacle']['D']:.4f}")
    return result


def main():
    conn = get_connection()
    matches = load_matches_with_serve(conn)
    dob = load_dob(conn)
    tour_mean_min = tour_mean_minutes(matches)
    print(f"Matches loaded (to 2024): {len(matches)}; "
          f"tour mean minutes (train) = {tour_mean_min:.1f}")
    rows = build_rows(matches, dob, tour_mean_min)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validate"]
    print(f"Rows: train={len(train_rows)} validate={len(val_rows)}")
    matched_keys = _matched_pinnacle_keys(matches)
    base_p_win = [r["p_base_winner"] for r in val_rows]

    # Current best: columns, fitted result, validation p_winner list.
    cur_cols = [COL["elo_diff"]]
    cur_best, _ = _fit_and_select(train_rows, val_rows, cur_cols, "elo-only")
    cur_p_win = cur_best["p_winner_val"]
    _summarize(cur_best, "elo-only (base)", base_p_win, cur_p_win,
               val_rows, matched_keys)

    results = {"elo-only": cur_best}
    adopted_cols = list(cur_cols)

    def run_batch(label, extra_cols, note=None):
        """Test one batch on top of the adopted set. Returns True if
        adopted (paired z vs current best <= -2)."""
        nonlocal adopted_cols, cur_cols, cur_best, cur_p_win
        cols = adopted_cols + extra_cols
        best, _grid = _fit_and_select(train_rows, val_rows, cols, label)
        _summarize(best, label, base_p_win, cur_p_win, val_rows, matched_keys)
        d = best["paired_vs_current_best"]
        adopted = d["z"] <= -2.0
        best["adopted"] = adopted
        best["note"] = note
        results[label] = best
        print(f"    -> {'ADOPT' if adopted else 'DROP'} "
              f"(paired z vs current best = {d['z']:.2f})")
        if adopted:
            adopted_cols = list(cols)
            cur_cols = list(cols)
            # Refit to recover the adopted model's validation probs
            # (summarize deleted p_winner_val).
            refit, _ = _fit_and_select(train_rows, val_rows, cols,
                                       label + " (adopted refit)")
            cur_p_win = refit["p_winner_val"]
            cur_best = best
        return adopted

    # 1. Form, raw.
    form_raw_adopted = run_batch("form-raw", [COL["form2m_diff"], COL["form10_diff"]])
    # 2. Form, residual: only if raw failed (raw overlaps Elo; the
    # residual is the cleaner signal and is only needed then).
    if not form_raw_adopted:
        run_batch("form-resid", [COL["resid2m_diff"], COL["resid10_diff"]])
    else:
        results["form-resid"] = {"adopted": False,
                                 "note": "not tried: raw form adopted"}
        print("  form-resid: not tried (raw form adopted)")
    # 3. Fatigue.
    run_batch("fatigue", [COL["rest_diff"], COL["games7_diff"],
                          COL["games14_diff"], COL["prevmin_diff"]])
    # 4. WElo.
    run_batch("welo", [COL["welo_diff"]])
    # 5. Glicko-A: RD diff.
    glicko_a_adopted = run_batch("glicko-A", [COL["rd_diff"]])
    # 6. Glicko-B: RD diff + Elo diff shrunk by mean RD.
    run_batch("glicko-B", [COL["rd_diff"], COL["elo_x_rdshrink"]],
              note="superset of glicko-A: tests whether RD shrinkage adds")
    # 7. Age + interactions.
    run_batch("age", [COL["age_diff"], COL["age_x_bo5"],
                      COL["age_x_close"], COL["age_x_load7"]])
    # 8. Head-to-head: try k = 3, 5, 10, keep the best on validation.
    h2h_candidates = {}
    for k_name in ("h2h_k3", "h2h_k5", "h2h_k10"):
        best, _grid = _fit_and_select(train_rows, val_rows,
                                      adopted_cols + [COL[k_name]],
                                      f"h2h-{k_name}")
        _summarize(best, f"h2h-{k_name}", base_p_win, cur_p_win,
                   val_rows, matched_keys)
        h2h_candidates[k_name] = best
    h2h_best_name = min(h2h_candidates,
                        key=lambda n: h2h_candidates[n]["val_brier"])
    h2h_best = h2h_candidates[h2h_best_name]
    d = h2h_best["paired_vs_current_best"]
    h2h_adopted = d["z"] <= -2.0
    h2h_best["adopted"] = h2h_adopted
    h2h_best["note"] = (f"k picked on validation: {h2h_best_name}; "
                        f"other k: " + ", ".join(
                            f"{n}={h2h_candidates[n]['val_brier']:.4f}"
                            for n in h2h_candidates))
    results["h2h"] = h2h_best
    print(f"    -> {'ADOPT' if h2h_adopted else 'DROP'} "
          f"(h2h best k = {h2h_best_name}, "
          f"paired z vs current best = {d['z']:.2f})")
    if h2h_adopted:
        adopted_cols = adopted_cols + [COL[h2h_best_name]]
        refit, _ = _fit_and_select(train_rows, val_rows, adopted_cols,
                                   "h2h (adopted refit)")
        cur_p_win = refit["p_winner_val"]
        cur_best = h2h_best
    # 9. Joint serve: adopted set + serve/return, separate row only.
    run_batch("joint-serve", [COL["hold_diff"], COL["break_diff"]],
              note="serve was NOT adopted alone (z=-1.24); joint test only")

    final = {
        "adopted_columns": [FEATURE_NAMES[c] for c in adopted_cols],
        "val_brier": cur_best["val_brier"],
        "paired_vs_baseline": cur_best["paired_vs_baseline"],
        "vs_pinnacle": cur_best["vs_pinnacle"],
        "C": cur_best["C"],
        "l1_ratio": cur_best["l1_ratio"],
        "coef": cur_best["coef"],
    }
    print(f"Final adopted set: {final['adopted_columns']}")
    print(f"Final validation Brier={final['val_brier']:.4f} "
          f"vs Pinnacle D={final['vs_pinnacle']['D']:.4f}")

    out = {
        "model": "logistic regression, no intercept, features as A-minus-B diffs",
        "run_date": _dt.date.today().isoformat(),
        "scoring_set": "main-tour non-team matches, validation 2023-2024",
        "baseline": BASELINE,
        "selection_rule": "adopt iff paired Brier z vs current best <= -2.0",
        "batches": results,
        "final": final,
    }
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "logreg_all_batches.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
