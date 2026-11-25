"""Follow-up ablations: alternative formulations for the dropped batches.

Six alternatives, each tested on top of the ADOPTED set from
logreg_all_batches.json (elo_diff + 4 fatigue + age_diff + 3 age
interactions; validation Brier 0.2166), except #6, which replaces
elo_diff with the RD-shrunk variant head-to-head. Adopt iff paired
Brier z vs the adopted set <= -2.0 on validation (same 5,464
main-tour non-team matches, 2023-2024). The production model is
unchanged whatever the outcome.

  1. serve-recent  recent-60-day hold/break diffs (current serving
                    form, not career skill)
  2. serve-resid   hold/break minus the Elo-implied rate: the
                    Elo-implied mapping is an OLS fit of hold/break
                    on overall Elo over train-period player
                    snapshots (end of 2022, >= 50 service games),
                    frozen before validation is scored
  3. age-x-serve   one column: age_diff * career hold_diff (serve
                    matters more for older players)
  4. dominance     games-share in the last 10 wins (how convincingly
                    they win), NOT a second WElo rating
  5. glicko-gate   one column: elo_diff * high-RD flag, where
                    high-RD is max(rd_a, rd_b) >= the train median
                    of max-RD over train scored matches (trust Elo
                    less when either player is low-information)
  6. glicko-shrink adopted set with elo_diff REPLACED by the
                    RD-shrunk elo_diff (elo_diff / (1 + meanRD/350)),
                    head-to-head against the plain elo_diff

NO INTERCEPT anywhere (fit_intercept=False): production requires
p(A) + p(B) = 1, and winner-side scoring would leak through any
intercept (see logreg.py). Calibration intercept b is likewise never
searched. Fit on train 2001-2022; C / l1_ratio selected on validation
only; test set 2025-2026 is never touched.

Caveat (multiple comparisons): these are 6 more adoption tests on
the same validation set, on top of the ~9 in run_logreg_all, so the
family-wise false-adoption risk keeps growing. An adopted
alternative is a candidate, not proof; the locked test set
(2025-2026, after the TML merge) is the confirmation.

Usage (repo root, project venv):
    .venv/bin/python -m scripts.backtest.run_logreg_alternatives

Writes scripts/backtest/outputs/logreg_alternatives.json (git-ignored).
"""

import datetime as _dt
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from align import MAIN_TOUR_LEVELS, _is_team_exhibition  # noqa: E402
from db import get_connection  # noqa: E402
from elo_state import EloState  # noqa: E402
from features import (  # noqa: E402
    DominanceState,
    FatigueState,
    H2HState,
    RecentServeState,
    ServeState,
    WEloState,
    age_years,
    elo_high_rd_interaction,
    fit_rate_vs_elo,
    parse_score,
    serve_residual,
)
from glicko import compute_glicko_features  # noqa: E402
from logreg import (  # noqa: E402
    COL,
    FEATURE_NAMES,
    load_dob,
    load_matches_with_serve,
    tour_mean_minutes,
)
from metrics import paired_diff  # noqa: E402
from run_logreg_serve import (  # noqa: E402
    _fit_and_select,
    _matched_pinnacle_keys,
    _paired_vs_baseline,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent / "outputs"
BASELINE = {"val_brier": 0.21846441507745834, "vs_pinnacle_D": 0.01626537711788756}
# This runner builds the original 28 columns in FEATURE_NAMES order.
# FEATURE_NAMES has grown since (benchmark columns appended in
# logreg.py), so the row width is pinned here instead of
# len(FEATURE_NAMES).
N_COLS = 28
ADOPTED_NAMES = ["elo_diff", "rest_diff", "games7_diff", "games14_diff",
                 "prevmin_diff", "age_diff", "age_x_bo5", "age_x_close",
                 "age_x_load7"]
# Minimum career service games for a player snapshot to enter the
# Elo-implied hold mapping fit.
RATE_FIT_MIN_GAMES = 50


def _is_scored(m):
    return m["tourney_level"] in MAIN_TOUR_LEVELS and not _is_team_exhibition(m["tourney_name"])


def _split_for(year):
    return "train" if year <= 2022 else "validate"


def fit_elo_implied_rates(matches):
    """OLS mapping of hold/break on overall Elo, train data only.

    Walks matches to 2022-12-31 with a locked Elo + ServeState,
    snapshots every player with >= RATE_FIT_MIN_GAMES career service
    games (overall Elo, shrunk overall hold, shrunk overall break),
    and fits fit_rate_vs_elo on the snapshots. Returns
    (coefs, n_players). The coefs are frozen before the main walk
    scores any validation match.
    """
    elo = EloState()
    serve = ServeState()
    for m in matches:
        if m["tourney_date"].year > 2022:
            break
        elo.update(m["winner_id"], m["loser_id"], m["surface"])
        serve.update(
            m["winner_id"], m["loser_id"], m["surface"],
            m["w_sv_gms"], m["w_bp_faced"], m["w_bp_saved"],
            m["l_sv_gms"], m["l_bp_faced"], m["l_bp_saved"],
        )
    snapshots = []
    for pid, entry in serve._players.items():
        svc_played = entry["overall"][1]
        if svc_played < RATE_FIT_MIN_GAMES:
            continue
        hold, brk = serve.rates(pid, None)
        snapshots.append((elo.blended(pid, None)[0], hold, brk))
    coefs = fit_rate_vs_elo(snapshots)
    return coefs, len(snapshots)


def rd_median_threshold(matches, glicko):
    """Median max-RD over train scored matches (train only)."""
    maxs = []
    for m in matches:
        if m["tourney_date"].year > 2022:
            continue
        if not _is_scored(m):
            continue
        rd_a, rd_b = glicko.get((m["tourney_id"], m["match_num"]),
                                (350.0, 350.0))
        maxs.append(max(rd_a, rd_b))
    return statistics.median(maxs)


def build_rows(matches, dob, tour_mean_min, rd_threshold, rate_coefs):
    """One walk. Returns rows for scored matches, year >= 2001.

    Row x: the 21 all-batches columns (identical computation to
    run_logreg_all.build_rows, so the adopted set refit must
    reproduce its 0.2166) plus 7 alternative columns:
    hold_rec_diff, break_rec_diff, hold_resid_diff, break_resid_diff,
    age_x_hold_diff, dominance_diff, elo_x_highrd.
    """
    glicko = compute_glicko_features(matches)
    print(f"Glicko features computed for {len(glicko)} matches")
    elo = EloState()
    serve = ServeState()
    recent_serve = RecentServeState()
    fatigue = FatigueState(tour_mean_minutes=tour_mean_min)
    welo = WEloState()
    h2h = H2HState()
    dominance = DominanceState()
    rows = []
    for m in matches:
        date = m["tourney_date"]
        year = date.year
        w_id, l_id = m["winner_id"], m["loser_id"]
        a_id, b_id = (w_id, l_id) if w_id < l_id else (l_id, w_id)
        w_games, l_games, _sets, retired, walkover = parse_score(m.get("score"))
        total_games = w_games + l_games
        p_base_winner, _, _, _ = elo.predict(w_id, l_id, m["surface"])
        if year >= 2001 and _is_scored(m):
            y = 1 if a_id == w_id else 0
            p_cal_a, p_blended_a, blended_a, blended_b = elo.predict(
                a_id, b_id, m["surface"])
            elo_diff = blended_a - blended_b
            hold_d, brk_d = serve.diffs(a_id, b_id, m["surface"])
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
            # --- alternative columns ---
            hold_rec_d, brk_rec_d = recent_serve.diffs(
                a_id, b_id, m["surface"], date)
            # Serve residual vs the Elo-implied rate (overall rates).
            elo_ov_a = elo.blended(a_id, None)[0]
            elo_ov_b = elo.blended(b_id, None)[0]
            hold_ov_a, brk_ov_a = serve.rates(a_id, None)
            hold_ov_b, brk_ov_b = serve.rates(b_id, None)
            hold_resid_d = (
                serve_residual(hold_ov_a, elo_ov_a, rate_coefs["hold"])
                - serve_residual(hold_ov_b, elo_ov_b, rate_coefs["hold"]))
            brk_resid_d = (
                serve_residual(brk_ov_a, elo_ov_a, rate_coefs["break"])
                - serve_residual(brk_ov_b, elo_ov_b, rate_coefs["break"]))
            age_x_hold_d = age_d * hold_d
            dominance_d = dominance.diff(a_id, b_id)
            elo_x_highrd = elo_high_rd_interaction(
                elo_diff, rd_a, rd_b, rd_threshold)
            x = [
                elo_diff, hold_d, brk_d,
                0.0, 0.0, 0.0, 0.0,  # form cols not rebuilt here
                rest_d, g7_d, g14_d, pmin_d,
                welo_d, rd_d, elo_x_rd,
                age_d, age_d * bo5, age_d * close, age_d * g7_d,
                h_k3, h_k5, h_k10,
                hold_rec_d, brk_rec_d,
                hold_resid_d, brk_resid_d,
                age_x_hold_d, dominance_d, elo_x_highrd,
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
        recent_serve.update(
            w_id, l_id, m["surface"], date,
            m["w_sv_gms"], m["w_bp_faced"], m["w_bp_saved"],
            m["l_sv_gms"], m["l_bp_faced"], m["l_bp_saved"],
        )
        fatigue.update(w_id, l_id, date, total_games, m.get("minutes"))
        welo.update(w_id, l_id, m["surface"], w_games, total_games,
                    retired=retired, walkover=walkover)
        h2h.update(w_id, l_id)
        dominance.update(w_id, l_id, w_games, total_games,
                         retired=retired, walkover=walkover)
    return rows


def _summarize(result, label, base_p_win, adopted_p_win, val_rows, matched_keys):
    """Attach paired diffs + vs-Pinnacle D; strip the prob list."""
    p_win = result["p_winner_val"]
    result["paired_vs_baseline"] = _paired_vs_baseline(p_win, base_p_win)
    result["paired_vs_adopted"] = _paired_vs_baseline(p_win, adopted_p_win)
    for key in ("paired_vs_baseline", "paired_vs_adopted"):
        result[key] = {k: (float(v) if isinstance(v, float) else v)
                       for k, v in result[key].items()}
    val_pos = {r["key"]: i for i, r in enumerate(val_rows)}
    keys = [k for k in matched_keys if k in val_pos]
    result["vs_pinnacle"] = paired_diff(
        [p_win[val_pos[k]] for k in keys],
        [matched_keys[k] for k in keys])
    del result["p_winner_val"]
    d = result["paired_vs_adopted"]
    print(f"  {label}: C={result['C']} l1={result['l1_ratio']} "
          f"val Brier={result['val_brier']:.4f} "
          f"vs adopted D={d['D']:.4f} z={d['z']:.2f} "
          f"vs baseline D={result['paired_vs_baseline']['D']:.4f} "
          f"z={result['paired_vs_baseline']['z']:.2f} "
          f"vs Pinnacle D={result['vs_pinnacle']['D']:.4f}")
    return result


def main():
    conn = get_connection()
    matches = load_matches_with_serve(conn)
    dob = load_dob(conn)
    tour_mean_min = tour_mean_minutes(matches)
    print(f"Matches loaded (to 2024): {len(matches)}; "
          f"tour mean minutes (train) = {tour_mean_min:.1f}")

    # Pre-pass 1: Glicko RDs, then the train-median max-RD threshold.
    glicko = compute_glicko_features(matches)
    rd_threshold = rd_median_threshold(matches, glicko)
    print(f"RD gate threshold (median max-RD, train scored) = {rd_threshold:.1f}")

    # Pre-pass 2: Elo-implied hold/break mapping, train data only.
    rate_coefs, n_snap = fit_elo_implied_rates(matches)
    print(f"Elo-implied rates fit on {n_snap} train player snapshots: "
          f"hold = {rate_coefs['hold'][0]:.4f} + {rate_coefs['hold'][1]:.6f}*Elo, "
          f"break = {rate_coefs['break'][0]:.4f} + {rate_coefs['break'][1]:.6f}*Elo")

    rows = build_rows(matches, dob, tour_mean_min, rd_threshold, rate_coefs)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validate"]
    print(f"Rows: train={len(train_rows)} validate={len(val_rows)}")
    matched_keys = _matched_pinnacle_keys(matches)
    base_p_win = [r["p_base_winner"] for r in val_rows]

    adopted_cols = [COL[n] for n in ADOPTED_NAMES]
    adopted, _ = _fit_and_select(train_rows, val_rows, adopted_cols, "adopted")
    adopted_p_win = adopted["p_winner_val"]
    _summarize(adopted, "adopted (refit)", base_p_win, adopted_p_win,
               val_rows, matched_keys)
    # Sanity: the refit must reproduce the all-batches adopted Brier.
    assert abs(adopted["val_brier"] - 0.2165820615864405) < 0.0005, (
        f"adopted refit {adopted['val_brier']:.4f} does not reproduce 0.2166")
    print("  adopted refit reproduces logreg_all_batches 0.2166: OK")

    results = {"adopted": adopted}
    # Alternatives are each tested independently on top of the
    # adopted set; the final set is the adopted set PLUS every
    # alternative that passed, accumulated here (not overwritten by
    # the last passing one).
    adopted_extra = []
    final_cols = list(adopted_cols)

    def run_alt(label, extra_cols, note=None):
        """Test one alternative on top of the adopted set."""
        nonlocal final_cols
        cols = adopted_cols + extra_cols
        best, _grid = _fit_and_select(train_rows, val_rows, cols, label)
        _summarize(best, label, base_p_win, adopted_p_win,
                   val_rows, matched_keys)
        d = best["paired_vs_adopted"]
        keep = d["z"] <= -2.0
        best["adopted"] = keep
        best["note"] = note
        results[label] = best
        print(f"    -> {'ADOPT' if keep else 'DROP'} "
              f"(paired z vs adopted = {d['z']:.2f})")
        if keep:
            adopted_extra.extend(c for c in extra_cols
                                 if c not in adopted_extra)
            final_cols = adopted_cols + adopted_extra
        return keep

    # 1. Recent-window serve.
    run_alt("serve-recent", [COL["hold_rec_diff"], COL["break_rec_diff"]])
    # 2. Serve residual vs Elo-implied rate.
    run_alt("serve-resid", [COL["hold_resid_diff"], COL["break_resid_diff"]])
    # 3. Age x serve interaction.
    run_alt("age-x-serve", [COL["age_x_hold_diff"]])
    # 4. Dominance ratio (last-10-wins games-share).
    run_alt("dominance", [COL["dominance_diff"]])
    # 5. Glicko gate (Elo diff trusted less when either RD is high).
    run_alt("glicko-gate", [COL["elo_x_highrd"]])

    # 6. Glicko single-column shrinkage: replace elo_diff with the
    # RD-shrunk variant, head-to-head against the plain elo_diff.
    shrink_cols = [COL["elo_x_rdshrink"] if c == COL["elo_diff"] else c
                   for c in adopted_cols]
    best6, _ = _fit_and_select(train_rows, val_rows, shrink_cols,
                               "glicko-shrink")
    _summarize(best6, "glicko-shrink (elo_diff replaced)", base_p_win,
               adopted_p_win, val_rows, matched_keys)
    d6 = best6["paired_vs_adopted"]
    keep6 = d6["z"] <= -2.0
    best6["adopted"] = keep6
    best6["note"] = ("replaces elo_diff with elo_x_rdshrink; "
                     "head-to-head, not an extra column")
    results["glicko-shrink"] = best6
    print(f"    -> {'ADOPT (replace elo_diff)' if keep6 else 'DROP'} "
          f"(paired z vs adopted = {d6['z']:.2f})")
    if keep6:
        # Apply the replacement to the accumulated final set, not
        # to the base adopted set alone.
        final_cols = [COL["elo_x_rdshrink"] if c == COL["elo_diff"] else c
                      for c in final_cols]

    final = {
        "adopted_columns": [FEATURE_NAMES[c] for c in final_cols],
        "val_brier": results["adopted"]["val_brier"] if final_cols == adopted_cols
                     else None,
        "rd_threshold": rd_threshold,
        "rate_coefs": {k: [float(a), float(b)] for k, (a, b) in rate_coefs.items()},
        "rate_fit_n_players": n_snap,
    }
    # Final Brier / vs-Pinnacle: refit the final column set if it changed.
    if final_cols != adopted_cols:
        refit, _ = _fit_and_select(train_rows, val_rows, final_cols,
                                   "final refit")
        _summarize(refit, "final", base_p_win, adopted_p_win,
                   val_rows, matched_keys)
        final["val_brier"] = refit["val_brier"]
        final["paired_vs_baseline"] = refit["paired_vs_baseline"]
        final["vs_pinnacle"] = refit["vs_pinnacle"]
        final["C"] = refit["C"]
        final["l1_ratio"] = refit["l1_ratio"]
        final["coef"] = refit["coef"]
    else:
        final["val_brier"] = adopted["val_brier"]
        final["paired_vs_baseline"] = adopted["paired_vs_baseline"]
        final["vs_pinnacle"] = adopted["vs_pinnacle"]
        final["C"] = adopted["C"]
        final["l1_ratio"] = adopted["l1_ratio"]
        final["coef"] = adopted["coef"]
    print(f"Final adopted set: {final['adopted_columns']}")
    print(f"Final validation Brier={final['val_brier']:.4f} "
          f"vs Pinnacle D={final['vs_pinnacle']['D']:.4f}")

    out = {
        "model": "logistic regression, no intercept, features as A-minus-B diffs",
        "run_date": _dt.date.today().isoformat(),
        "scoring_set": "main-tour non-team matches, validation 2023-2024",
        "baseline": BASELINE,
        "selection_rule": "adopt iff paired Brier z vs adopted set <= -2.0",
        "alternatives": results,
        "final": final,
    }
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "logreg_alternatives.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
