"""Retirement / walkover retest with subgroup evaluation (task 3).

Round 4 tested the retirement flags (ret_self_diff + ret_opp_diff,
60-day window, RetirementState) on top of the 16-column base and
dropped them at overall z=+0.06. That verdict used the overall
average over all 5,464 validation matches plus the z<=-2 rule --
unfair to a rare flag, because ~98% of matches carry no flag and
dilute the average.

This runner retests with subgroup evaluation:
  - overall D/z (all 5,464) for comparison with round 4, AND
  - subgroup D/z on the matches where the flag actually fires
    (ret_self != 0, ret_opp != 0, walkover self/opp, and the
    combined self-injury flag), plus trigger rates and fitted
    coefficients (direction check: a self-injury flag must get a
    NEGATIVE coefficient -- the flagged player should win less).

Retirement definition (verified in features.RetirementState, reused
here via run_logreg_round4.build_rows): 1 if the player retired in a
match in the last 60 days (RET marks the loser as the retiree), and
1 if an opponent retired against them. Both are pre-match known:
the state updates only AFTER features are read, so there is no
leakage. The round-4 batch tested the two columns JOINTLY, which
dilutes the self signal with the opp column (winning by retirement
says little about your own fitness); here self and opp are also
tested SEPARATELY.

Walkover definition (new here; no WalkoverState existed): a match
whose score string is exactly "W/O" or "Walkover". The loser is the
withdrawer (verified on 2023 samples: Cilic withdrew, listed as
loser; Fucsovics withdrew, listed as loser), so the self flag goes
to the loser and the opp flag to the winner, same 60-day window as
retirement. Bare "RET" (1 match in 2001-2024) and NULL scores
(0 in 2001-2024) are NOT counted as walkovers. Walkovers are rare:
371 in 2001-2024, 38 in validation main-tour matches.

Fitting: base (16 col) and the joint-retirement batch use the full
_fit_and_select grid (reproduction of round 4). The remaining
variants use the base's selected config (C=0.01, l1_ratio=0.0) so
the comparison isolates the feature, not re-tuning.

Scoring set: main-tour non-team, validation 2023-2024 (n=5,464).
Test set 2025-2026 is never touched. No calibration intercept b.

Usage (repo root, project venv):
    .venv/bin/python -m scripts.backtest.run_retirement_retest

Writes scripts/backtest/outputs/retirement_retest.json (git-ignored).
"""

import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from align import MAIN_TOUR_LEVELS, _is_team_exhibition  # noqa: E402
from db import get_connection  # noqa: E402
from logreg import (  # noqa: E402
    COL,
    FEATURE_NAMES,
    Standardizer,
    fit_logreg,
    load_dob,
    load_matches_with_serve,
    load_year_end_rankings,
    predict_proba,
    tour_mean_minutes,
)
from metrics import score_outcome  # noqa: E402
from run_logreg_round4 import BASE_NAMES, BASE_VAL_BRIER, build_rows  # noqa: E402
from run_logreg_serve import _fit_and_select, _paired_vs_baseline  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"
WO_WINDOW_DAYS = 60
FIXED_C = 0.01
FIXED_L1 = 0.0


def _is_scored(m):
    return m["tourney_level"] in MAIN_TOUR_LEVELS and not _is_team_exhibition(m["tourney_name"])


def build_walkover_flags(matches):
    """(tourney_id, match_num) -> (wo_self_diff, wo_opp_diff).

    Mirrors RetirementState: 60-day window, pre-match only (the
    event is recorded AFTER the current match is read). A W/O
    match's loser withdrew (self flag for the loser); the winner
    advanced (opp flag for the winner).
    """
    self_ev, opp_ev = {}, {}
    out = {}
    n_events = 0
    for m in matches:
        date = m["tourney_date"]
        year = date.year
        w_id, l_id = m["winner_id"], m["loser_id"]
        score = (m.get("score") or "").strip()
        if year >= 2001 and _is_scored(m):
            a_id, b_id = (w_id, l_id) if w_id < l_id else (l_id, w_id)

            def flag(dates, d):
                if not dates:
                    return 0.0
                cutoff = d - _dt.timedelta(days=WO_WINDOW_DAYS)
                return 1.0 if any(x > cutoff for x in dates) else 0.0

            wo_s = flag(self_ev.get(a_id, []), date) - flag(self_ev.get(b_id, []), date)
            wo_o = flag(opp_ev.get(a_id, []), date) - flag(opp_ev.get(b_id, []), date)
            out[(m["tourney_id"], m["match_num"])] = (wo_s, wo_o)
        if score in ("W/O", "Walkover"):
            self_ev.setdefault(l_id, []).append(date)
            opp_ev.setdefault(w_id, []).append(date)
            n_events += 1
    return out, n_events


def _p_winner(p_a, y):
    return p_a if y == 1 else 1.0 - p_a


def _fit_fixed(train_rows, val_rows, cols, C=FIXED_C, l1_ratio=FIXED_L1):
    """One fit at a fixed (C, l1_ratio); returns probs + metrics."""
    Xtr = np.array([r["x"] for r in train_rows])[:, cols]
    ytr = np.array([r["y"] for r in train_rows])
    Xva = np.array([r["x"] for r in val_rows])[:, cols]
    yva = np.array([r["y"] for r in val_rows])
    scaler = Standardizer().fit(Xtr)
    clf = fit_logreg(scaler.transform(Xtr), ytr, C, l1_ratio)
    p_a = predict_proba(clf, scaler.transform(Xva))
    p_win = [_p_winner(p, y) for p, y in zip(p_a, yva)]
    m = score_outcome(p_win)
    p_a_tr = predict_proba(clf, scaler.transform(Xtr))
    train_brier = score_outcome(
        [_p_winner(p, y) for p, y in zip(p_a_tr, ytr)])["brier"]
    return {
        "C": C, "l1_ratio": l1_ratio, "val_brier": m["brier"],
        "val_log_loss": m["log_loss"], "val_accuracy": m["accuracy"],
        "train_brier": train_brier,
        "coef": [float(c) for c in clf.coef_[0]],
        "feature_names": [FEATURE_NAMES[c] if c < len(FEATURE_NAMES) else f"col{c}"
                          for c in cols],
        "p_winner_val": p_win,
    }


def _subgroup(p_var, p_base, mask):
    """Paired Brier difference on a subset (mask = list of indices)."""
    if not mask:
        return None
    ds = [(1.0 - p_var[i]) ** 2 - (1.0 - p_base[i]) ** 2 for i in mask]
    n = len(ds)
    mean_d = sum(ds) / n
    var = sum((d - mean_d) ** 2 for d in ds) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else 0.0
    brier_var = sum((1.0 - p_var[i]) ** 2 for i in mask) / n
    brier_base = sum((1.0 - p_base[i]) ** 2 for i in mask) / n
    return {"n": n, "brier_variant": brier_var, "brier_base": brier_base,
            "D": mean_d, "se": se, "z": mean_d / se if se > 0 else 0.0}


def main():
    conn = get_connection()
    matches = load_matches_with_serve(conn)
    dob = load_dob(conn)
    rankings = load_year_end_rankings(conn)
    tour_mean_min = tour_mean_minutes(matches)
    print(f"Matches loaded (to 2024): {len(matches)}")

    rows, cov = build_rows(matches, dob, tour_mean_min, rankings)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validate"]
    print(f"Rows: train={len(train_rows)} validate={len(val_rows)}; "
          f"retirement flag nonzero "
          f"{cov['ret_flag_nonzero']}/{cov['ret_flag_n']} "
          f"({cov['ret_flag_nonzero'] / cov['ret_flag_n']:.1%})")

    # Attach walkover flags (columns 50, 51) to the same rows.
    wo_flags, n_wo_events = build_walkover_flags(matches)
    print(f"Walkover events in walk (2001-2024, all levels): {n_wo_events}")
    missing = 0
    for r in rows:
        wo_s, wo_o = wo_flags.get(r["key"], (0.0, 0.0))
        if r["key"] not in wo_flags:
            missing += 1
        r["x"] = list(r["x"]) + [wo_s, wo_o]
    print(f"Rows missing a walkover flag entry: {missing} (should be 0)")
    assert missing == 0
    WO_SELF, WO_OPP = 50, 51

    # Flag masks on validation rows (index into val_rows).
    def _mask(col_idx):
        return [i for i, r in enumerate(val_rows) if r["x"][col_idx] != 0.0]

    masks = {
        "ret_self": _mask(COL["ret_self_diff"]),
        "ret_opp": _mask(COL["ret_opp_diff"]),
        "ret_any": [i for i, r in enumerate(val_rows)
                    if r["x"][COL["ret_self_diff"]] != 0.0
                    or r["x"][COL["ret_opp_diff"]] != 0.0],
        "wo_self": _mask(WO_SELF),
        "wo_opp": _mask(WO_OPP),
        "wo_any": [i for i, r in enumerate(val_rows)
                   if r["x"][WO_SELF] != 0.0 or r["x"][WO_OPP] != 0.0],
        "injury_self": [i for i, r in enumerate(val_rows)
                        if r["x"][COL["ret_self_diff"]] != 0.0
                        or r["x"][WO_SELF] != 0.0],
    }
    print("Trigger rates (validation, n=5,464):")
    for name, idxs in masks.items():
        print(f"  {name}: {len(idxs)} ({len(idxs) / len(val_rows):.2%})")

    base_cols = [COL[n] for n in BASE_NAMES]
    results = {}

    def evaluate(label, result, base_p_win, subgroup_names):
        """Attach overall + subgroup paired diffs vs base16; strip prob list.

        Overall D vs the LOCKED baseline (0.218464) equals
        val_brier - 0.218464 exactly (paired D is the mean of per-match
        differences), so it is not stored separately.
        """
        p_win = result["p_winner_val"]
        result["paired_vs_base16"] = _paired_vs_baseline(p_win, base_p_win)
        result["subgroups"] = {}
        for name in subgroup_names:
            result["subgroups"][name] = _subgroup(p_win, base16_p_win, masks[name])
        result["paired_vs_base16"] = {
            kk: (float(vv) if isinstance(vv, float) else vv)
            for kk, vv in result["paired_vs_base16"].items()}
        d = result["paired_vs_base16"]
        print(f"  {label}: C={result['C']} l1={result['l1_ratio']} "
              f"val Brier={result['val_brier']:.6f} "
              f"overall vs base16 D={d['D']:.6f} z={d['z']:.2f}")
        for name in subgroup_names:
            s = result["subgroups"][name]
            print(f"    subgroup {name}: n={s['n']} "
                  f"Brier var={s['brier_variant']:.4f} base={s['brier_base']:.4f} "
                  f"D={s['D']:.6f} se={s['se']:.6f} z={s['z']:.2f}")
        del result["p_winner_val"]
        return result

    # --- Base 16-col model (reproduction of logreg_benchmark 0.2128). ---
    base16, _ = _fit_and_select(train_rows, val_rows, base_cols, "base16")
    base16_p_win = base16["p_winner_val"]
    print(f"Base16 refit: val Brier={base16['val_brier']:.6f} "
          f"(benchmark 0.212823), train Brier={base16['train_brier']:.4f}")
    assert abs(base16["val_brier"] - BASE_VAL_BRIER) < 0.0005
    base16_out = {k: v for k, v in base16.items() if k != "p_winner_val"}
    results["base16"] = base16_out

    # --- Joint retirement (reproduction of round-4 batch, z=+0.06). ---
    ret_joint, _ = _fit_and_select(
        train_rows, val_rows,
        base_cols + [COL["ret_self_diff"], COL["ret_opp_diff"]],
        "retirement-joint")
    results["retirement_joint"] = evaluate(
        "retirement joint (ret_self + ret_opp)",
        ret_joint, base16_p_win, ["ret_self", "ret_opp", "ret_any"])

    # --- Fixed-config variants (C=0.01, l1=0.0 = base's config). ---
    fixed = [
        ("ret_self_only", base_cols + [COL["ret_self_diff"]],
         ["ret_self", "ret_any"]),
        ("ret_opp_only", base_cols + [COL["ret_opp_diff"]],
         ["ret_opp", "ret_any"]),
        ("walkover_joint", base_cols + [WO_SELF, WO_OPP],
         ["wo_self", "wo_opp", "wo_any"]),
        ("walkover_self_only", base_cols + [WO_SELF],
         ["wo_self", "wo_any"]),
        ("walkover_opp_only", base_cols + [WO_OPP],
         ["wo_opp", "wo_any"]),
        ("injury_self_combined", base_cols + [COL["ret_self_diff"], WO_SELF],
         ["ret_self", "wo_self", "injury_self"]),
    ]
    for label, cols, subs in fixed:
        res = _fit_fixed(train_rows, val_rows, cols)
        # Name the two appended walkover columns for readability.
        names = list(res["feature_names"])
        for j, c in enumerate(cols):
            if c == WO_SELF:
                names[j] = "wo_self_diff"
            elif c == WO_OPP:
                names[j] = "wo_opp_diff"
        res["feature_names"] = names
        results[label] = evaluate(label, res, base16_p_win, subs)

    # --- Adoption test on top of the CURRENT best (20-col final model). ---
    # The variants above sit on base16 (the round-4 setup). The adoption
    # decision must be relative to the 20-column final set, so refit it
    # (must reproduce 0.212076) and test ret_self on top of it.
    FINAL20 = BASE_NAMES + ["tb_winpct_diff", "decset_winpct_diff",
                            "comeback_diff", "elo_x_bo5"]
    final20_cols = [COL[n] for n in FINAL20]
    final20, _ = _fit_and_select(train_rows, val_rows, final20_cols, "final20")
    final20_p_win = final20["p_winner_val"]
    print(f"Final20 refit: val Brier={final20['val_brier']:.6f} "
          f"(round-4 final 0.212076)")
    assert abs(final20["val_brier"] - 0.2120761841362994) < 0.0005
    final20_out = {k: v for k, v in final20.items() if k != "p_winner_val"}
    results["final20"] = final20_out
    ret_on_final, _ = _fit_and_select(
        train_rows, val_rows, final20_cols + [COL["ret_self_diff"]],
        "ret_self_on_final20")
    # Subgroup evaluation vs the final-20 model (not base16).
    p_win = ret_on_final["p_winner_val"]
    ret_on_final["paired_vs_final20"] = _paired_vs_baseline(p_win, final20_p_win)
    ret_on_final["subgroups"] = {}
    for name in ["ret_self", "ret_any", "injury_self"]:
        ret_on_final["subgroups"][name] = _subgroup(p_win, final20_p_win, masks[name])
    ret_on_final["paired_vs_final20"] = {
        kk: (float(vv) if isinstance(vv, float) else vv)
        for kk, vv in ret_on_final["paired_vs_final20"].items()}
    d = ret_on_final["paired_vs_final20"]
    print(f"  ret_self on final20: C={ret_on_final['C']} l1={ret_on_final['l1_ratio']} "
          f"val Brier={ret_on_final['val_brier']:.6f} "
          f"overall vs final20 D={d['D']:.6f} z={d['z']:.2f}")
    for name in ["ret_self", "ret_any", "injury_self"]:
        s = ret_on_final["subgroups"][name]
        print(f"    subgroup {name}: n={s['n']} "
              f"Brier var={s['brier_variant']:.4f} base={s['brier_base']:.4f} "
              f"D={s['D']:.6f} se={s['se']:.6f} z={s['z']:.2f}")
    del ret_on_final["p_winner_val"]
    results["ret_self_on_final20"] = ret_on_final

    # Coefficient direction check for the self-injury columns.
    print("Coefficient check (standardized; self-injury flags must be NEGATIVE):")
    for label in ["retirement_joint", "ret_self_only", "walkover_joint",
                  "walkover_self_only", "injury_self_combined",
                  "ret_self_on_final20"]:
        r = results[label]
        coefs = dict(zip(r["feature_names"], r["coef"]))
        interesting = {k: round(v, 4) for k, v in coefs.items()
                       if k in ("ret_self_diff", "ret_opp_diff",
                                "wo_self_diff", "wo_opp_diff")}
        print(f"  {label}: {interesting}")

    out = {
        "model": "logistic regression, no intercept, A-minus-B diffs; subgroup retest of retirement/walkover flags",
        "run_date": _dt.date.today().isoformat(),
        "scoring_set": "main-tour non-team matches, validation 2023-2024 (n=5,464)",
        "base": {"columns": BASE_NAMES, "val_brier": BASE_VAL_BRIER,
                 "C": FIXED_C, "l1_ratio": FIXED_L1},
        "retirement_definition": "RetirementState, 60-day window, pre-match known (update after features read); RET marks the loser as retiree",
        "walkover_definition": "score string exactly 'W/O' or 'Walkover'; loser withdrew (self flag), winner advanced (opp flag); 60-day window; bare RET (1 match) and NULL scores (0) not counted",
        "walkover_events_2001_2024": n_wo_events,
        "trigger_rates": {name: {"n": len(idxs), "rate": len(idxs) / len(val_rows)}
                          for name, idxs in masks.items()},
        "results": results,
    }
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "retirement_retest.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
