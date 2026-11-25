"""Round-4 features: cross-model benchmark items 1-6, tested on top
of the 16-column adopted set from the benchmark round.

Base for this round (logreg_benchmark.json final, validation Brier
0.2128, vs Pinnacle D = 0.0099): elo_diff + 4 fatigue + age_diff +
3 age interactions + recent serve (hold_rec/break_rec) + age_x_hold
+ experience (exp/surf_exp) + ranking (logrank/logpoints).
Sequential forward selection: each item is tested on top of the
current best and adopted only if paired Brier z vs current best
<= -2.0 on validation (same 5,464 main-tour non-team matches,
2023-2024).

  1. point-serve  1st/2nd-serve win%, ace rate, DF rate, BP
                    conversion% and BP save%, per player,
                    walk-forward, surface-aware shrunk rates
                    (features.PointServeState), as A-minus-B diffs.
                    Tour priors from train 2001-2022 only. NULL
                    serve columns -> match does not update the
                    state (same rule as ServeState).
  2. momentum     form slope: last-5 win% minus last-20 win%,
                    per player, differenced. Trajectory, not
                    level (form level/residual both dropped).
  3. clutch       tiebreak win% + deciding-set record + comeback
                    after losing the first set (features.ClutchState,
                    parse_clutch). Shrunk toward 0.5 / 0.5 / 0.19.
                    Literature says clutch persistence is weak;
                    reported honestly either way.
  4. event-games  games already played at this tournament before
                    this match (features.EventState), as a diff.
                    In-tournament progression, distinct from
                    games7/games14.
  5. retirement   flags for self-retired / opponent-retired in
                    the last 60 days (features.RetirementState),
                    as diffs. RET marks the loser as the retiree.
  6. elo-x-bo5    elo_diff * best-of-5 indicator, one column.
                    (age_x_bo5 exists; elo_diff x bo5 never tried.)

NO INTERCEPT anywhere (fit_intercept=False): production requires
p(A) + p(B) = 1, and winner-side scoring would leak through any
intercept (see logreg.py). Calibration intercept b is likewise never
searched. Fit on train 2001-2022; C / l1_ratio selected on validation
only; test set 2025-2026 is never touched.

Caveat (multiple comparisons): this is the fourth round of adoption
tests on the same validation set, so the family-wise false-adoption
risk keeps growing. An adopted feature is a candidate, not proof;
the locked test set (2025-2026, after the TML merge) is the
confirmation.

Usage (repo root, project venv):
    .venv/bin/python -m scripts.backtest.run_logreg_round4

Writes scripts/backtest/outputs/logreg_round4.json (git-ignored).
"""

import datetime as _dt
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from align import MAIN_TOUR_LEVELS, _is_team_exhibition  # noqa: E402
from db import get_connection  # noqa: E402
from elo_state import EloState  # noqa: E402
from features import (  # noqa: E402
    ClutchState,
    EventState,
    FatigueState,
    FormState,
    PointServeState,
    RecentServeState,
    RetirementState,
    ServeState,
    WinRateState,
    age_years,
    parse_score,
)
from logreg import (  # noqa: E402
    COL,
    FEATURE_NAMES,
    load_dob,
    load_matches_with_serve,
    load_year_end_rankings,
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
# Adopted set after the benchmark round (logreg_benchmark.json).
BASE_NAMES = ["elo_diff", "rest_diff", "games7_diff", "games14_diff",
              "prevmin_diff", "age_diff", "age_x_bo5", "age_x_close",
              "age_x_load7", "hold_rec_diff", "break_rec_diff",
              "age_x_hold_diff", "exp_diff", "surf_exp_diff",
              "logrank_diff", "logpoints_diff"]
BASE_VAL_BRIER = 0.21282268218640088  # refit must reproduce this
# This runner builds 50 columns: 36 pre-round-4 + 14 new ones.
N_COLS = 50


def _is_scored(m):
    return m["tourney_level"] in MAIN_TOUR_LEVELS and not _is_team_exhibition(m["tourney_name"])


def _split_for(year):
    # Test split added 2026-11-24 for the locked test (2025-2026).
    # Round-4 only ever reads train/validate, so this is a no-op for it.
    if year <= 2022:
        return "train"
    if year <= 2024:
        return "validate"
    return "test"


def _rank_feats(rankings, a_id, b_id, year):
    """(logrank_diff, logpoints_diff) from the previous year's
    year-end ranking. Positive means A is better ranked / has more
    points. Either side missing -> 0 for that component."""
    ra = rankings.get((a_id, year - 1))
    rb = rankings.get((b_id, year - 1))
    if ra is None or rb is None or ra[0] is None or rb[0] is None:
        logrank_d = 0.0
    else:
        logrank_d = math.log(rb[0]) - math.log(ra[0])
    if ra is None or rb is None or ra[1] is None or rb[1] is None:
        logpoints_d = 0.0
    else:
        logpoints_d = math.log(ra[1] + 1) - math.log(rb[1] + 1)
    return logrank_d, logpoints_d


def build_rows(matches, dob, tour_mean_min, rankings):
    """One walk. Returns rows for scored matches, year >= 2001, plus
    a coverage dict for the new features.

    Row x is indexed by the global COL registry (N_COLS entries,
    zeros elsewhere). The 16 base columns use the identical
    computation as run_logreg_benchmark.build_rows, so the base
    refit must reproduce 0.2128.
    """
    elo = EloState()
    serve = ServeState()
    recent_serve = RecentServeState()
    point_serve = PointServeState()
    form = FormState()
    fatigue = FatigueState(tour_mean_minutes=tour_mean_min)
    clutch = ClutchState()
    event = EventState()
    retire = RetirementState()
    winrate = WinRateState()
    rows = []
    cov = {"point_serve_n": 0, "point_serve_ok": 0,
           "clutch_tb_n": 0, "clutch_tb_any": 0,
           "ret_flag_n": 0, "ret_flag_nonzero": 0}
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
            hold_rec_d, brk_rec_d = recent_serve.diffs(
                a_id, b_id, m["surface"], date)
            rest_d, g7_d, g14_d, pmin_d = fatigue.diffs(a_id, b_id, date)
            age_a = age_years(dob.get(a_id), date)
            age_b = age_years(dob.get(b_id), date)
            age_d = 0.0 if age_a is None or age_b is None else age_a - age_b
            bo5 = 1.0 if m.get("best_of") == 5 else 0.0
            close = 1.0 - abs(2.0 * p_blended_a - 1.0)
            # --- round-4 columns ---
            (first_d, second_d, ace_d, df_d,
             bpc_d, bps_d) = point_serve.diffs(a_id, b_id, m["surface"])
            mom_d = form.momentum_diff(a_id, b_id, date)
            tb_d, dec_d, come_d = clutch.diffs(a_id, b_id)
            ev_d = event.diff(a_id, b_id, m["tourney_id"])
            ret_s_d, ret_o_d = retire.diffs(a_id, b_id, date)
            _wp_d, _swp_d, exp_d, sexp_d = winrate.diffs(
                a_id, b_id, m["surface"])
            logrank_d, logpoints_d = _rank_feats(rankings, a_id, b_id, year)
            x = [0.0] * N_COLS
            x[COL["elo_diff"]] = elo_diff
            x[COL["rest_diff"]] = rest_d
            x[COL["games7_diff"]] = g7_d
            x[COL["games14_diff"]] = g14_d
            x[COL["prevmin_diff"]] = pmin_d
            x[COL["age_diff"]] = age_d
            x[COL["age_x_bo5"]] = age_d * bo5
            x[COL["age_x_close"]] = age_d * close
            x[COL["age_x_load7"]] = age_d * g7_d
            x[COL["hold_rec_diff"]] = hold_rec_d
            x[COL["break_rec_diff"]] = brk_rec_d
            x[COL["age_x_hold_diff"]] = age_d * hold_d
            x[COL["exp_diff"]] = exp_d
            x[COL["surf_exp_diff"]] = sexp_d
            x[COL["logrank_diff"]] = logrank_d
            x[COL["logpoints_diff"]] = logpoints_d
            x[COL["first_win_diff"]] = first_d
            x[COL["second_win_diff"]] = second_d
            x[COL["ace_rate_diff"]] = ace_d
            x[COL["df_rate_diff"]] = df_d
            x[COL["bp_conv_diff"]] = bpc_d
            x[COL["bp_save_diff"]] = bps_d
            x[COL["momentum_diff"]] = mom_d
            x[COL["tb_winpct_diff"]] = tb_d
            x[COL["decset_winpct_diff"]] = dec_d
            x[COL["comeback_diff"]] = come_d
            x[COL["event_games_diff"]] = ev_d
            x[COL["ret_self_diff"]] = ret_s_d
            x[COL["ret_opp_diff"]] = ret_o_d
            x[COL["elo_x_bo5"]] = elo_diff * bo5
            assert len(x) == N_COLS, (len(x), N_COLS)
            rows.append({
                "year": year,
                "split": _split_for(year),
                "x": x,
                "y": y,
                "p_base_winner": p_base_winner,
                "key": (m["tourney_id"], m["match_num"]),
            })
            if year >= 2023:
                cov["point_serve_n"] += 1
                if (m["w_svpt"] is not None and m["l_svpt"] is not None
                        and m["w_1st_in"] is not None
                        and m["l_1st_in"] is not None):
                    cov["point_serve_ok"] += 1
                cov["clutch_tb_n"] += 1
                if tb_d != 0.0 or dec_d != 0.0 or come_d != 0.0:
                    cov["clutch_tb_any"] += 1
                cov["ret_flag_n"] += 1
                if ret_s_d != 0.0 or ret_o_d != 0.0:
                    cov["ret_flag_nonzero"] += 1
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
        point_serve.update(
            w_id, l_id, m["surface"],
            m["w_svpt"], m["w_1st_in"], m["w_1st_won"], m["w_2nd_won"],
            m["w_ace"], m["w_df"], m["w_bp_faced"], m["w_bp_saved"],
            m["l_svpt"], m["l_1st_in"], m["l_1st_won"], m["l_2nd_won"],
            m["l_ace"], m["l_df"], m["l_bp_faced"], m["l_bp_saved"],
        )
        form.update(w_id, l_id, date, p_base_winner)
        fatigue.update(w_id, l_id, date, total_games, m.get("minutes"))
        clutch.update(w_id, l_id, m.get("score"), m.get("best_of"))
        event.update(w_id, l_id, m["tourney_id"], total_games)
        retire.update(w_id, l_id, date, retired)
        winrate.update(w_id, l_id, m["surface"])
    return rows, cov


def _summarize(result, label, base_p_win, cur_p_win, val_rows, matched_keys):
    """Attach paired diffs + vs-Pinnacle D; strip the prob list."""
    p_win = result["p_winner_val"]
    result["paired_vs_baseline"] = _paired_vs_baseline(p_win, base_p_win)
    result["paired_vs_current_best"] = _paired_vs_baseline(p_win, cur_p_win)
    for key in ("paired_vs_baseline", "paired_vs_current_best"):
        result[key] = {k: (float(v) if isinstance(v, float) else v)
                       for k, v in result[key].items()}
    val_pos = {r["key"]: i for i, r in enumerate(val_rows)}
    keys = [k for k in matched_keys if k in val_pos]
    result["vs_pinnacle"] = paired_diff(
        [p_win[val_pos[k]] for k in keys],
        [matched_keys[k] for k in keys])
    del result["p_winner_val"]
    d = result["paired_vs_current_best"]
    print(f"  {label}: C={result['C']} l1={result['l1_ratio']} "
          f"val Brier={result['val_brier']:.4f} "
          f"vs current-best D={d['D']:.4f} z={d['z']:.2f} "
          f"vs baseline D={result['paired_vs_baseline']['D']:.4f} "
          f"z={result['paired_vs_baseline']['z']:.2f} "
          f"vs Pinnacle D={result['vs_pinnacle']['D']:.4f}")
    return result


def main():
    conn = get_connection()
    matches = load_matches_with_serve(conn)
    dob = load_dob(conn)
    rankings = load_year_end_rankings(conn)
    tour_mean_min = tour_mean_minutes(matches)
    print(f"Matches loaded (to 2024): {len(matches)}; "
          f"tour mean minutes (train) = {tour_mean_min:.1f}; "
          f"year-end ranking rows = {len(rankings)}")

    rows, cov = build_rows(matches, dob, tour_mean_min, rankings)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validate"]
    print(f"Rows: train={len(train_rows)} validate={len(val_rows)}")
    print(f"Validation coverage: point-serve data present "
          f"{cov['point_serve_ok']}/{cov['point_serve_n']} "
          f"({cov['point_serve_ok'] / cov['point_serve_n']:.1%}); "
          f"any clutch column nonzero "
          f"{cov['clutch_tb_any']}/{cov['clutch_tb_n']} "
          f"({cov['clutch_tb_any'] / cov['clutch_tb_n']:.1%}); "
          f"any retirement flag nonzero "
          f"{cov['ret_flag_nonzero']}/{cov['ret_flag_n']} "
          f"({cov['ret_flag_nonzero'] / cov['ret_flag_n']:.1%})")
    matched_keys = _matched_pinnacle_keys(matches)
    base_p_win = [r["p_base_winner"] for r in val_rows]

    base_cols = [COL[n] for n in BASE_NAMES]
    cur_best, _ = _fit_and_select(train_rows, val_rows, base_cols, "base")
    cur_p_win = cur_best["p_winner_val"]
    _summarize(cur_best, "base (benchmark final)", base_p_win, cur_p_win,
               val_rows, matched_keys)
    assert abs(cur_best["val_brier"] - BASE_VAL_BRIER) < 0.0005, (
        f"base refit {cur_best['val_brier']:.4f} does not reproduce 0.2128")
    print("  base refit reproduces logreg_benchmark 0.2128: OK")

    results = {"base": cur_best}
    adopted_cols = list(base_cols)

    def run_batch(label, extra_cols, note=None):
        """Test one batch on top of the current best. Adopt iff
        paired z vs current best <= -2."""
        nonlocal adopted_cols, cur_best, cur_p_win
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
            refit, _ = _fit_and_select(train_rows, val_rows, cols,
                                       label + " (adopted refit)")
            cur_p_win = refit["p_winner_val"]
            cur_best = best
        return adopted

    # 1. Point-level serve stats (6 columns).
    run_batch("point-serve", [
        COL["first_win_diff"], COL["second_win_diff"],
        COL["ace_rate_diff"], COL["df_rate_diff"],
        COL["bp_conv_diff"], COL["bp_save_diff"],
    ], note="tour priors from train 2001-2022; NULL serve columns "
            "do not update the state")
    # 2. Form momentum (form_5 - form_20).
    run_batch("momentum", [COL["momentum_diff"]],
              note="trajectory, not level; form level/residual dropped")
    # 3. Clutch stats (tiebreak, deciding set, comeback).
    run_batch("clutch", [
        COL["tb_winpct_diff"], COL["decset_winpct_diff"],
        COL["comeback_diff"],
    ], note="shrunk toward 0.5 / 0.5 / 0.19; literature says clutch "
            "persistence is weak")
    # 4. Games already played at this event.
    run_batch("event-games", [COL["event_games_diff"]],
              note="in-tournament progression, distinct from games7/14")
    # 5. Recent-retirement flags.
    run_batch("retirement", [
        COL["ret_self_diff"], COL["ret_opp_diff"],
    ], note="60-day window; RET marks the loser as the retiree")
    # 6. Elo x best-of-5 interaction.
    run_batch("elo-x-bo5", [COL["elo_x_bo5"]],
              note="one column; age_x_bo5 exists but elo x bo5 never tried")

    final = {
        "adopted_columns": [FEATURE_NAMES[c] for c in adopted_cols],
        "val_brier": cur_best["val_brier"],
        "paired_vs_baseline": cur_best["paired_vs_baseline"],
        "vs_pinnacle": cur_best["vs_pinnacle"],
        "C": cur_best["C"],
        "l1_ratio": cur_best["l1_ratio"],
        "coef": cur_best["coef"],
        "coverage": cov,
    }
    print(f"Final adopted set: {final['adopted_columns']}")
    print(f"Final validation Brier={final['val_brier']:.4f} "
          f"vs Pinnacle D={final['vs_pinnacle']['D']:.4f}")

    out = {
        "model": "logistic regression, no intercept, features as A-minus-B diffs",
        "run_date": _dt.date.today().isoformat(),
        "scoring_set": "main-tour non-team matches, validation 2023-2024",
        "baseline": BASELINE,
        "base": {"columns": BASE_NAMES, "val_brier": BASE_VAL_BRIER},
        "selection_rule": "adopt iff paired Brier z vs current best <= -2.0",
        "batches": results,
        "final": final,
    }
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "logreg_round4.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
