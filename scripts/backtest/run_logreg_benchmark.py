"""Benchmark-features round: features the published models use that
TSA has not tried yet, tested in the same container.

Base for this round is the adopted set AFTER the alternatives round
(12 columns, validation Brier 0.2154, vs Pinnacle D = 0.0130):
elo_diff + 4 fatigue + age_diff + 3 age interactions + recent
serve (hold_rec/break_rec) + age_x_hold. Sequential forward
selection: each feature is tested on top of the current best and
adopted only if paired Brier z vs current best <= -2.0 on validation
(same 5,464 main-tour non-team matches, 2023-2024).

  1. winpct      career win% diff + surface win% diff (walk-forward,
                    shrunk toward 0.5 / the player's overall rate)
  2. experience  log1p career matches + log1p surface matches diffs
  3. ranking     log-ranking diff + log-points diff, from the
                    PREVIOUS YEAR's year-end ranking
                    (atp_year_end_rankings). The summary table
                    (atp_player_ranking_summary) holds only the
                    latest ranking and would leak, so it is NOT
                    used; the cost is up to ~12 months staleness.
                    Missing either side -> 0 (neutral). Coverage:
                    99.8% train, 99.7% validation (both present).
  4. welo-swap   WElo REPLACES elo_diff in the adopted set (same
                    other columns), head-to-head. Tests the
                    Angelini/Pickledger form (WElo as the rating)
                    rather than the stacked form already dropped
                    (z=-0.82). Not an extra column.
  5. venue       win-rate excess at this tournament (tourney_id),
                    walk-forward, shrunk toward the player's own
                    overall rate; 0 with no prior match there.
                    Sparse by construction; coverage reported.
  6. home        1 if the player's IOC matches the tournament's
                    country, as an A-minus-B diff (-1/0/1).
                    Tournament country from features.tournament_country
                    (hand-built map, verified against the distinct
                    2001-2024 tourney names; Davis Cup host = first
                    team; Olympics by city; movers by year).
                    Unknown country -> 0 (neutral); coverage reported.

NO INTERCEPT anywhere (fit_intercept=False): production requires
p(A) + p(B) = 1, and winner-side scoring would leak through any
intercept (see logreg.py). Calibration intercept b is likewise never
searched. Fit on train 2001-2022; C / l1_ratio selected on validation
only; test set 2025-2026 is never touched.

Caveat (multiple comparisons): this is the third round of adoption
tests on the same validation set (alternatives round included), so
the family-wise false-adoption risk keeps growing. An adopted
feature is a candidate, not proof; the locked test set (2025-2026,
after the TML merge) is the confirmation.

Usage (repo root, project venv):
    .venv/bin/python -m scripts.backtest.run_logreg_benchmark

Writes scripts/backtest/outputs/logreg_benchmark.json (git-ignored).
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
    FatigueState,
    RecentServeState,
    ServeState,
    VenueState,
    WEloState,
    WinRateState,
    age_years,
    parse_score,
    tournament_country,
)
from logreg import (  # noqa: E402
    COL,
    FEATURE_NAMES,
    load_dob,
    load_ioc,
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
# This runner builds the original 36 columns in FEATURE_NAMES order.
# FEATURE_NAMES has grown since (round-4 columns appended in
# logreg.py), so the width is pinned here instead of
# len(FEATURE_NAMES).
N_COLS = 36
BASELINE = {"val_brier": 0.21846441507745834, "vs_pinnacle_D": 0.01626537711788756}
# Adopted set after the alternatives round (logreg_alternatives.json).
BASE_NAMES = ["elo_diff", "rest_diff", "games7_diff", "games14_diff",
              "prevmin_diff", "age_diff", "age_x_bo5", "age_x_close",
              "age_x_load7", "hold_rec_diff", "break_rec_diff",
              "age_x_hold_diff"]
BASE_VAL_BRIER = 0.21544016072848954  # refit must reproduce this


def _is_scored(m):
    return m["tourney_level"] in MAIN_TOUR_LEVELS and not _is_team_exhibition(m["tourney_name"])


def _split_for(year):
    return "train" if year <= 2022 else "validate"


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


def _home_diff(ioc, a_id, b_id, country):
    """A-minus-B home flag (-1/0/1); unknown country -> 0."""
    if country is None:
        return 0.0
    ha = 1.0 if ioc.get(a_id) == country else 0.0
    hb = 1.0 if ioc.get(b_id) == country else 0.0
    return ha - hb


def build_rows(matches, dob, tour_mean_min, rankings, ioc):
    """One walk. Returns rows for scored matches, year >= 2001, plus
    a coverage dict for the ranking / venue / home features.

    Row x is indexed by the global COL registry (N_COLS entries,
    zeros elsewhere): the 12 base columns (identical
    computation to run_logreg_alternatives.build_rows, so the base
    refit must reproduce 0.2154), welo_diff (for the swap test), and
    the 8 benchmark columns.
    """
    elo = EloState()
    serve = ServeState()
    recent_serve = RecentServeState()
    fatigue = FatigueState(tour_mean_minutes=tour_mean_min)
    welo = WEloState()
    winrate = WinRateState()
    venue = VenueState()
    rows = []
    cov = {"rank_both": 0, "rank_n": 0, "venue_prior3": 0, "venue_n": 0,
           "home_known": 0, "home_n": 0, "home_nonzero": 0}
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
            welo_d = welo.diff(a_id, b_id, m["surface"])
            age_a = age_years(dob.get(a_id), date)
            age_b = age_years(dob.get(b_id), date)
            age_d = 0.0 if age_a is None or age_b is None else age_a - age_b
            bo5 = 1.0 if m.get("best_of") == 5 else 0.0
            close = 1.0 - abs(2.0 * p_blended_a - 1.0)
            # --- benchmark columns ---
            wp_d, swp_d, exp_d, sexp_d = winrate.diffs(
                a_id, b_id, m["surface"])
            logrank_d, logpoints_d = _rank_feats(rankings, a_id, b_id, year)
            venue_d = venue.diff(a_id, b_id, m["tourney_id"])
            country = tournament_country(m["tourney_name"], date)
            home_d = _home_diff(ioc, a_id, b_id, country)
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
            x[COL["welo_diff"]] = welo_d
            x[COL["winpct_diff"]] = wp_d
            x[COL["surf_winpct_diff"]] = swp_d
            x[COL["exp_diff"]] = exp_d
            x[COL["surf_exp_diff"]] = sexp_d
            x[COL["logrank_diff"]] = logrank_d
            x[COL["logpoints_diff"]] = logpoints_d
            x[COL["venue_winpct_diff"]] = venue_d
            x[COL["home_diff"]] = home_d
            assert len(x) == N_COLS
            rows.append({
                "year": year,
                "split": _split_for(year),
                "x": x,
                "y": y,
                "p_base_winner": p_base_winner,
                "key": (m["tourney_id"], m["match_num"]),
            })
            if year >= 2023:
                cov["rank_n"] += 1
                if (rankings.get((a_id, year - 1)) is not None
                        and rankings.get((b_id, year - 1)) is not None):
                    cov["rank_both"] += 1
                cov["venue_n"] += 1
                if (venue.n_prior(a_id, m["tourney_id"]) >= 3
                        or venue.n_prior(b_id, m["tourney_id"]) >= 3):
                    cov["venue_prior3"] += 1
                cov["home_n"] += 1
                if country is not None:
                    cov["home_known"] += 1
                if home_d != 0.0:
                    cov["home_nonzero"] += 1
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
        winrate.update(w_id, l_id, m["surface"])
        venue.update(w_id, l_id, m["tourney_id"])
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
    ioc = load_ioc(conn)
    tour_mean_min = tour_mean_minutes(matches)
    print(f"Matches loaded (to 2024): {len(matches)}; "
          f"tour mean minutes (train) = {tour_mean_min:.1f}; "
          f"year-end ranking rows = {len(rankings)}; "
          f"players with IOC = {len(ioc)}")

    rows, cov = build_rows(matches, dob, tour_mean_min, rankings, ioc)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validate"]
    print(f"Rows: train={len(train_rows)} validate={len(val_rows)}")
    print(f"Validation coverage: ranking both present "
          f"{cov['rank_both']}/{cov['rank_n']} "
          f"({cov['rank_both'] / cov['rank_n']:.1%}); "
          f"venue >=3 prior matches for either player "
          f"{cov['venue_prior3']}/{cov['venue_n']} "
          f"({cov['venue_prior3'] / cov['venue_n']:.1%}); "
          f"tournament country known {cov['home_known']}/{cov['home_n']} "
          f"({cov['home_known'] / cov['home_n']:.1%}), "
          f"home flag nonzero {cov['home_nonzero']}")
    matched_keys = _matched_pinnacle_keys(matches)
    base_p_win = [r["p_base_winner"] for r in val_rows]

    base_cols = [COL[n] for n in BASE_NAMES]
    cur_best, _ = _fit_and_select(train_rows, val_rows, base_cols, "base")
    cur_p_win = cur_best["p_winner_val"]
    _summarize(cur_best, "base (alternatives final)", base_p_win, cur_p_win,
               val_rows, matched_keys)
    assert abs(cur_best["val_brier"] - BASE_VAL_BRIER) < 0.0005, (
        f"base refit {cur_best['val_brier']:.4f} does not reproduce 0.2154")
    print("  base refit reproduces logreg_alternatives 0.2154: OK")

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

    # 1. Win% (career + surface).
    run_batch("winpct", [COL["winpct_diff"], COL["surf_winpct_diff"]])
    # 2. Experience (log1p matches, career + surface).
    run_batch("experience", [COL["exp_diff"], COL["surf_exp_diff"]])
    # 3. Ranking (previous-year year-end log-rank + log-points).
    run_batch("ranking", [COL["logrank_diff"], COL["logpoints_diff"]],
              note="previous-year year-end ranking; summary table NOT used (leaks)")

    # 4. WElo swap: replace elo_diff with welo_diff in the current
    # adopted set, head-to-head. welo_diff is a column of this
    # runner's rows, so the swap is a column substitution; if it
    # wins, later batches build on the swapped set.
    swap_cols = [COL["welo_diff"] if c == COL["elo_diff"] else c
                 for c in adopted_cols]
    swap_best, _ = _fit_and_select(train_rows, val_rows, swap_cols,
                                   "welo-swap")
    swap_p_win = swap_best["p_winner_val"]
    _summarize(swap_best, "welo-swap (elo_diff replaced by welo_diff)",
               base_p_win, cur_p_win, val_rows, matched_keys)
    d_swap = swap_best["paired_vs_current_best"]
    swap_adopted = d_swap["z"] <= -2.0
    swap_best["adopted"] = swap_adopted
    swap_best["note"] = ("WElo as the rating itself (Angelini/Pickledger "
                         "form), not stacked; head-to-head swap, not an "
                         "extra column")
    results["welo-swap"] = swap_best
    print(f"    -> {'ADOPT (swap)' if swap_adopted else 'DROP'} "
          f"(paired z vs current best = {d_swap['z']:.2f})")
    if swap_adopted:
        adopted_cols = swap_cols
        cur_p_win = swap_p_win
        cur_best = swap_best

    # 5. Venue win-rate excess.
    run_batch("venue", [COL["venue_winpct_diff"]])
    # 6. Home flag.
    run_batch("home", [COL["home_diff"]],
              note=f"tournament country known for "
                   f"{cov['home_known']}/{cov['home_n']} validation matches")

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
    out_path = OUT_DIR / "logreg_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
