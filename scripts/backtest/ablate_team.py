"""Ablation: does feeding team/exhibition matches into Elo help or hurt?

Two variants of the locked model, run over the same matches:
  all      -- Elo updates on every match (current baseline behaviour)
  no-team  -- Elo updates skip United Cup / Laver Cup / NextGen Finals
              (align._is_team_exhibition); Davis Cup and Olympics are
              still fed in, in both variants.

Scoring set is identical in both variants: main-tour matches
(level A/F/G/M, excluding team/exhibition names), 2001-2024.
So any Brier difference is the effect of the excluded matches on
the Elo state, not a difference in what is scored. Market D is
computed on the same tennis-data Pinnacle-matched validation
subset for both variants.

Usage (repo root, project venv):
    .venv/bin/python -m scripts.backtest.ablate_team

Writes scripts/backtest/outputs/ablation_team.json (git-ignored).
"""

import datetime as _dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from align import MAIN_TOUR_LEVELS, _is_team_exhibition, align_matches  # noqa: E402
from db import get_connection  # noqa: E402
from elo_state import EloState  # noqa: E402
from metrics import devig, paired_diff, score_outcome  # noqa: E402
from names import name_key  # noqa: E402
from odds import read_tennis_data_xlsx  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _load_matches(conn, end_year=2024):
    cur = conn.cursor()
    cur.execute(
        "SELECT tourney_id, match_num, tourney_date, tourney_name, surface, "
        "round, tourney_level, winner_id, loser_id, winner_name, loser_name "
        "FROM atp_matches WHERE winner_id IS NOT NULL AND loser_id IS NOT NULL "
        "AND tourney_date <= ? ORDER BY tourney_date, tourney_id, match_num",
        (f"{end_year}-12-31",),
    )
    cols = [d[0] for d in cur.description]
    matches = []
    for row in cur.fetchall():
        m = dict(zip(cols, row))
        m["tourney_date"] = _dt.date.fromisoformat(str(m["tourney_date"]))
        matches.append(m)
    return matches


def _is_scored(m):
    return m["tourney_level"] in MAIN_TOUR_LEVELS and not _is_team_exhibition(m["tourney_name"])


def _run_variant(matches, skip_team):
    """Walk 1968-2024 once. Returns {year: [p_winner]} for scored matches."""
    state = EloState()
    scored = defaultdict(list)
    n_skipped_updates = 0
    for m in matches:
        year = m["tourney_date"].year
        if year >= 2001 and _is_scored(m):
            p_a, _, _, _ = state.predict(m["winner_id"], m["loser_id"], m["surface"])
            scored[year].append(p_a)
        if skip_team and _is_team_exhibition(m["tourney_name"]):
            n_skipped_updates += 1
            continue
        state.update(m["winner_id"], m["loser_id"], m["surface"])
    return scored, n_skipped_updates


def _paired_variant_diff(a_list, b_list):
    """Per-match Brier difference (variant A minus variant B), with SE/z."""
    n = len(a_list)
    ds = [(1.0 - pa) ** 2 - (1.0 - pb) ** 2 for pa, pb in zip(a_list, b_list)]
    mean_d = sum(ds) / n
    var = sum((d - mean_d) ** 2 for d in ds) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else 0.0
    return {"n": n, "D": mean_d, "se": se, "z": mean_d / se if se > 0 else 0.0}


def main():
    conn = get_connection()
    matches = _load_matches(conn)
    print(f"Matches loaded (to 2024): {len(matches)}")

    # Tennis-data odds, for the market-D comparison (alignment does not
    # depend on Elo, so both variants share the same matched set).
    odds_by_year = {}
    for year in (2022, 2023, 2024):
        rows = []
        for fy in (year - 1, year, year + 1):
            p = REPO_ROOT / "scripts" / "backtest" / "data" / f"{fy}.xlsx"
            if p.exists():
                rows += read_tennis_data_xlsx(str(p), fy)
        odds_by_year[year] = rows
    matched_keys = {}  # (tourney_id, match_num) -> Pinnacle prob of winner
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

    results = {}
    per_match = {}
    for variant, skip_team in (("all", False), ("no-team", True)):
        scored, n_skip = _run_variant(matches, skip_team)
        val = [p for y in (2023, 2024) for p in scored[y]]
        train = [p for y in range(2001, 2023) for p in scored[y]]
        per_match[variant] = scored
        results[variant] = {
            "skipped_elo_updates": n_skip,
            "train_main_tour": {"n": len(train), **score_outcome(train)},
            "validate_main_tour": {"n": len(val), **score_outcome(val)},
        }
        # Market D on the validation matched subset, per variant.
        val_keys = [(m["tourney_id"], m["match_num"]) for m in matches
                    if m["tourney_date"].year in (2023, 2024) and _is_scored(m)
                    and (m["tourney_id"], m["match_num"]) in matched_keys]
        # Rebuild the variant's p_winner in match order for those keys.
        state_p = {}
        # scored lists are in walk order, so zip them back onto scored matches.
        scored_matches = [m for m in matches if m["tourney_date"].year >= 2001 and _is_scored(m)]
        # scored[year] lists are per-year in walk order; flatten in the same order.
        flat = [p for y in sorted(scored) for p in scored[y]]
        for m, p in zip(scored_matches, flat):
            state_p[(m["tourney_id"], m["match_num"])] = p
        model_p = [state_p[k] for k in val_keys]
        market_p = [matched_keys[k] for k in val_keys]
        results[variant]["validate_vs_pinnacle"] = paired_diff(model_p, market_p)
        print(f"{variant}: skipped_updates={n_skip} "
              f"train Brier={results[variant]['train_main_tour']['brier']:.4f} "
              f"validate Brier={results[variant]['validate_main_tour']['brier']:.4f} "
              f"vs Pinnacle D={results[variant]['validate_vs_pinnacle']['D']:.4f} "
              f"(n={results[variant]['validate_vs_pinnacle']['n']})")

    val_all = [p for y in (2023, 2024) for p in per_match["all"][y]]
    val_no = [p for y in (2023, 2024) for p in per_match["no-team"][y]]
    diff = _paired_variant_diff(val_all, val_no)
    results["paired_difference_all_minus_no_team_validate"] = diff
    print(f"Paired Brier difference (all - no-team) on validation: "
          f"D={diff['D']:.4f} se={diff['se']:.4f} z={diff['z']:.2f} n={diff['n']}")
    print("Rule: |z| < 2 means no evidence either way; keep the simpler "
          "variant (all) unless no-team wins by >= 2 SE.")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "ablation_team.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
