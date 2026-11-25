"""Pinnacle OPENING vs CLOSING Brier on the validation set (2023-2024).

Question (user, 2026-11-19): the closing line (0.2010) is the end of a
correction process, so the opening line should be less efficient and
the edge bigger early. Is the opening Pinnacle Brier actually worse
than our final 20-column model (0.2121), and by how much? If yes,
there is a crossing point where the market goes from worse than us
to better than us.

Method: Valuebetennis CSVs carry both Pinnacle opening odds
(cote1/2_ouverture) and closing odds (cote1/2_cloture) for the same
matches. Both are aligned to the same validation matches (main tour,
non-team, 2023-2024) with the standard align_matches procedure, then
devigged. All three Briers (opening / closing / final-20 model) are
computed on the SAME intersection subset (matches with both opening
and closing odds), with paired D/SE/z.

Caveats, stated up front:
- Valuebetennis 2024 coverage is only ~73% (16 tournaments missing
  site-wide), so the matched subset is smaller than the tennis-data
  closing subset (5,202) and skews toward bigger events.
- Opening odds are sparser than closing odds in this source; the
  intersection is the matches where both exist.
- Pinnacle opening lines are set by Pinnacle's own sharp traders, so
  the opening line may already be efficient; the "early = dumb money"
  intuition applies more cleanly to exchange markets (Polymarket /
  Kalshi) than to Pinnacle. This run measures Pinnacle only.

Usage (repo root): .venv/bin/python -m scripts.backtest.run_opening_odds
Writes scripts/backtest/outputs/opening_odds.json.
Does not touch the test set (2025-2026).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from align import align_matches  # noqa: E402
from db import get_connection  # noqa: E402
from logreg import COL, load_dob, load_matches_with_serve, load_year_end_rankings, tour_mean_minutes  # noqa: E402
from metrics import devig, paired_diff  # noqa: E402
from names import name_key  # noqa: E402
from odds import read_valuebetennis_csv  # noqa: E402
from run_logreg_round4 import BASE_NAMES, build_rows  # noqa: E402
from run_logreg_serve import _paired_vs_baseline  # noqa: E402
from run_retirement_retest import _fit_fixed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "scripts" / "backtest" / "data"
OUT_DIR = REPO_ROOT / "scripts" / "backtest" / "outputs"

FINAL20 = BASE_NAMES + [
    "tb_winpct_diff", "decset_winpct_diff", "comeback_diff", "elo_x_bo5",
]


def _matched_vb_keys(matches, which):
    """(tourney_id, match_num) -> Pinnacle prob of the winner, 2023-2024.

    Same alignment procedure as run_logreg_serve._matched_pinnacle_keys,
    but against the Valuebetennis CSVs and the requested odds column
    (which="opening" reads cote*_ouverture, "closing" reads cote*_cloture).
    """
    odds_by_year = {}
    for year in (2023, 2024):
        rows = []
        for fy in (year - 1, year, year + 1):
            p = DATA_DIR / f"valuebetennis-matchs-{fy}.csv"
            if p.exists():
                rows += read_valuebetennis_csv(str(p), fy, which=which)
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


def _brier(p_winner_list):
    return sum((1.0 - p) ** 2 for p in p_winner_list) / len(p_winner_list)


def main():
    conn = get_connection()
    matches = load_matches_with_serve(conn)
    dob = load_dob(conn)
    rankings = load_year_end_rankings(conn)
    tour_mean_min = tour_mean_minutes(matches)
    print(f"Matches loaded (to 2024): {len(matches)}")

    rows, _cov = build_rows(matches, dob, tour_mean_min, rankings)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validate"]
    print(f"Rows: train={len(train_rows)} validate={len(val_rows)}")

    final20_cols = [COL[n] for n in FINAL20]
    final20 = _fit_fixed(train_rows, val_rows, final20_cols)
    print(f"Final20 refit: val Brier={final20['val_brier']:.6f} "
          f"(expect 0.212076)")
    assert abs(final20["val_brier"] - 0.2120761841362994) < 0.0005
    p_model_win = final20["p_winner_val"]

    opening = _matched_vb_keys(matches, "opening")
    closing = _matched_vb_keys(matches, "closing")
    print(f"Valuebetennis matched: opening={len(opening)} closing={len(closing)}")

    # Intersection subset: same matches for all three Briers.
    idx_both = [i for i, r in enumerate(val_rows)
                if r["key"] in opening and r["key"] in closing]
    idx_open_only = [i for i, r in enumerate(val_rows) if r["key"] in opening]
    idx_close_only = [i for i, r in enumerate(val_rows) if r["key"] in closing]
    print(f"Validation matches with opening odds: {len(idx_open_only)}; "
          f"with closing odds: {len(idx_close_only)}; with both: {len(idx_both)}")

    p_open = [opening[val_rows[i]["key"]] for i in idx_both]
    p_close = [closing[val_rows[i]["key"]] for i in idx_both]
    p_model = [p_model_win[i] for i in idx_both]

    b_open = _brier(p_open)
    b_close = _brier(p_close)
    b_model = _brier(p_model)
    print(f"On the {len(idx_both)}-match intersection:")
    print(f"  Pinnacle opening Brier = {b_open:.6f}")
    print(f"  Pinnacle closing Brier = {b_close:.6f}")
    print(f"  Final20 model  Brier   = {b_model:.6f}")
    print(f"  opening - closing      = {b_open - b_close:+.6f}")
    print(f"  opening - model        = {b_open - b_model:+.6f}")
    print(f"  model   - closing      = {b_model - b_close:+.6f}")

    out = {
        "n_val": len(val_rows),
        "n_opening": len(idx_open_only),
        "n_closing": len(idx_close_only),
        "n_both": len(idx_both),
        "brier_opening": b_open,
        "brier_closing": b_close,
        "brier_model": b_model,
        "paired_opening_vs_closing": _paired_vs_baseline(p_open, p_close),
        "paired_model_vs_opening": _paired_vs_baseline(p_model, p_open),
        "paired_model_vs_closing": _paired_vs_baseline(p_model, p_close),
        # Sanity: model Brier on the closing-only subset, for comparison
        # with the known 0.2103 on the tennis-data closing subset.
        "n_close_only": len(idx_close_only),
        "brier_model_on_close_only": _brier([p_model_win[i] for i in idx_close_only]),
        "brier_closing_on_close_only": _brier([closing[val_rows[i]["key"]] for i in idx_close_only]),
    }
    # _paired_vs_baseline returns (D, se, z) with model first: D is
    # (first - second), so paired_model_vs_opening D<0 means model better.
    # paired_diff from metrics has the same orientation; keep both keys.
    out["paired_opening_vs_closing_metrics"] = paired_diff(p_open, p_close)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "opening_odds.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote scripts/backtest/outputs/opening_odds.json")
    print(json.dumps({k: v for k, v in out.items()
                      if k.startswith("paired")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
