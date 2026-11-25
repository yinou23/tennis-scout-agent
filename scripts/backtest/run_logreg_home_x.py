"""Home x Elo interaction: does crowd support matter most when the
home player is the Elo underdog?

Base: the 20-column adopted set after round 4 (validation Brier
0.2121, vs Pinnacle D = 0.0093). Same selection rule as every round:
adopt iff paired Brier z vs current best <= -2.0 on validation
(main-tour non-team matches, 2023-2024, n = 5,464).

Background: the plain home flag tested at z = +0.15 (round 3) --
nothing. Two reasons it may still hide an effect:
  1. the flag is binary by passport, but crowd support is continuous
     and star-driven (Federer is "home" everywhere, even at the US
     Open against an American);
  2. crowd support should matter most exactly when the home player
     is the underdog -- when the favourite is home, the crowd adds
     little to what Elo already says.
So the candidates are interactions of the home flag with the Elo
diff, not the flag alone:

  home_diff        A-minus-B home flag (-1/0/1), replication control
  home_x_elo       home_diff * elo_diff (signed: negative when the
                   home player is the underdog)
  home_x_underdog  max(0, -elo_diff * home_diff): how much of an Elo
                   underdog the home player is, 0 otherwise
  home_x_fav       max(0, elo_diff * home_diff): the favourite side

Batches, each on top of the current best:
  1. home           home_diff alone (replication)
  2. home-x-elo     the signed interaction alone
  3. home-x-split   underdog + favourite pieces (asymmetric slopes)
  4. home-plus-ud   home_diff + home_x_underdog (main effect plus
                    the underdog piece -- the natural crowd model)

Home flag construction is unchanged from the benchmark round:
tournament country from features.tournament_country (100% coverage
of scored events), player IOC from load_ioc. Note the known
limitation: IOC is the player's current listed nationality, not
necessarily the one held at match time.

NO INTERCEPT anywhere (fit_intercept=False); calibration intercept b
is never searched. Fit on train 2001-2022; C / l1_ratio selected on
validation only; test set 2025-2026 is never touched.

Caveat (multiple comparisons): this is a fifth round of adoption
tests on the same validation set. An adopted feature is a candidate,
not proof; the locked test set is the confirmation.

Usage (repo root, project venv):
    .venv/bin/python -m scripts.backtest.run_logreg_home_x

Writes scripts/backtest/outputs/logreg_home_x.json (git-ignored).
"""

import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_connection  # noqa: E402
from features import tournament_country  # noqa: E402
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
from run_logreg_benchmark import _home_diff  # noqa: E402
from run_logreg_round4 import (  # noqa: E402
    BASE_NAMES,
    _is_scored,
    build_rows,
)
from run_logreg_serve import (  # noqa: E402
    _fit_and_select,
    _matched_pinnacle_keys,
    _paired_vs_baseline,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent / "outputs"
BASELINE = {"val_brier": 0.21846441507745834, "vs_pinnacle_D": 0.01626537711788756}
# Adopted set after round 4 (logreg_round4.json): 16 benchmark cols
# + clutch (3) + elo_x_bo5.
ADOPTED20 = BASE_NAMES + ["tb_winpct_diff", "decset_winpct_diff",
                          "comeback_diff", "elo_x_bo5"]
BASE_VAL_BRIER = 0.2120761841362994  # refit must reproduce this
N_COLS = len(FEATURE_NAMES)  # 53: round-4 50 + 3 home-x columns


def _add_home_columns(rows, matches, ioc):
    """Extend each row's x to N_COLS with the home columns.

    Rows were built by run_logreg_round4.build_rows in walk order
    over scored matches (year >= 2001, main-tour non-team), so
    zipping the same filtered walk reproduces the alignment.
    Returns a coverage dict for validation matches.
    """
    cov = {"home_n": 0, "home_known": 0, "home_nonzero": 0,
           "home_ud_n": 0, "home_ud_nonzero": 0}
    scored = [m for m in matches
              if m["tourney_date"].year >= 2001 and _is_scored(m)]
    assert len(scored) == len(rows), (len(scored), len(rows))
    for r, m in zip(rows, scored):
        w_id, l_id = m["winner_id"], m["loser_id"]
        a_id, b_id = (w_id, l_id) if w_id < l_id else (l_id, w_id)
        country = tournament_country(m["tourney_name"], m["tourney_date"])
        home_d = _home_diff(ioc, a_id, b_id, country)
        elo_d = r["x"][COL["elo_diff"]]
        x = list(r["x"]) + [0.0] * (N_COLS - len(r["x"]))
        x[COL["home_diff"]] = home_d
        x[COL["home_x_elo"]] = home_d * elo_d
        x[COL["home_x_underdog"]] = max(0.0, -elo_d * home_d)
        x[COL["home_x_fav"]] = max(0.0, elo_d * home_d)
        assert len(x) == N_COLS, (len(x), N_COLS)
        r["x"] = x
        if r["year"] >= 2023:
            cov["home_n"] += 1
            if country is not None:
                cov["home_known"] += 1
            if home_d != 0.0:
                cov["home_nonzero"] += 1
                cov["home_ud_n"] += 1
                if max(0.0, -elo_d * home_d) > 0.0:
                    cov["home_ud_nonzero"] += 1
    return cov


def _summarize(result, label, base_p_win, cur_p_win, val_rows, matched_keys):
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
          f"tour mean minutes (train) = {tour_mean_min:.1f}")

    rows, cov = build_rows(matches, dob, tour_mean_min, rankings)
    home_cov = _add_home_columns(rows, matches, ioc)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validate"]
    print(f"Rows: train={len(train_rows)} validate={len(val_rows)}")
    print(f"Validation home coverage: country known "
          f"{home_cov['home_known']}/{home_cov['home_n']} "
          f"({home_cov['home_known'] / home_cov['home_n']:.1%}), "
          f"home flag nonzero {home_cov['home_nonzero']}, "
          f"of which home player is the underdog "
          f"{home_cov['home_ud_nonzero']}/{home_cov['home_ud_n']}")
    matched_keys = _matched_pinnacle_keys(matches)
    base_p_win = [r["p_base_winner"] for r in val_rows]

    base_cols = [COL[n] for n in ADOPTED20]
    cur_best, _ = _fit_and_select(train_rows, val_rows, base_cols, "base")
    cur_p_win = cur_best["p_winner_val"]
    _summarize(cur_best, "base (round-4 final)", base_p_win, cur_p_win,
               val_rows, matched_keys)
    assert abs(cur_best["val_brier"] - BASE_VAL_BRIER) < 0.0005, (
        f"base refit {cur_best['val_brier']:.4f} does not reproduce 0.2121")
    print("  base refit reproduces logreg_round4 0.2121: OK")

    results = {"base": cur_best}
    adopted_cols = list(base_cols)

    def run_batch(label, extra_cols, note=None):
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

    # 1. Plain home flag alone (replication of the round-3 result).
    run_batch("home", [COL["home_diff"]],
              note="replication control; round 3 gave z = +0.15")
    # 2. Signed interaction: crowd should matter most for the underdog.
    run_batch("home-x-elo", [COL["home_x_elo"]],
              note="home_diff * elo_diff; negative when the home "
                   "player is the underdog")
    # 3. Split pieces: separate slopes for underdog-home and
    # favourite-home (asymmetric crowd effect).
    run_batch("home-x-split", [COL["home_x_underdog"], COL["home_x_fav"]],
              note="two-piece interaction; underdog piece is the "
                   "crowd-lifts-the-underdog hypothesis")
    # 4. Main effect + underdog piece.
    run_batch("home-plus-ud", [COL["home_diff"], COL["home_x_underdog"]],
              note="plain flag plus the underdog piece")

    final = {
        "adopted_columns": [FEATURE_NAMES[c] for c in adopted_cols],
        "val_brier": cur_best["val_brier"],
        "paired_vs_baseline": cur_best["paired_vs_baseline"],
        "vs_pinnacle": cur_best["vs_pinnacle"],
        "C": cur_best["C"],
        "l1_ratio": cur_best["l1_ratio"],
        "coef": cur_best["coef"],
        "coverage": {**cov, **home_cov},
    }
    print(f"Final adopted set: {final['adopted_columns']}")
    print(f"Final validation Brier={final['val_brier']:.4f} "
          f"vs Pinnacle D={final['vs_pinnacle']['D']:.4f}")

    out = {
        "model": "logistic regression, no intercept, features as A-minus-B diffs",
        "run_date": _dt.date.today().isoformat(),
        "scoring_set": "main-tour non-team matches, validation 2023-2024",
        "baseline": BASELINE,
        "base": {"columns": ADOPTED20, "val_brier": BASE_VAL_BRIER},
        "selection_rule": "adopt iff paired Brier z vs current best <= -2.0",
        "batches": results,
        "final": final,
    }
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "logreg_home_x.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
