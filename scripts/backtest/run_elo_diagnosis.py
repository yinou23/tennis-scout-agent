"""Elo diagnosis and variants (tasks 1-2).

Question: the locked blended Elo baseline scores validation Brier
0.2185 (2023-2024, main-tour non-team, n=5,464) while Poudel et al.
report a tuned Elo at 0.2120. Is our Elo implementation the problem,
or is the gap a dataset/test-set difference? And can any Elo
adjustment close it?

Diagnosis (task 1): the simplest possible standard Elo -- one overall
rating, win/loss only, K=32, no surface split, calibration a=0.8
(plus the uncalibrated a=1.0 line from the same walk) -- scored on
the same 5,464 validation matches. If the simple Elo lands near
0.2185, the blended Elo is fine and the gap to Poudel is a dataset
difference; if it lands near 0.2120, our blended Elo is the problem.

Variants (task 2): each adjustment applied to the BLENDED Elo (the
production model), so every variant is directly comparable to the
locked baseline 0.2185. Calibration intercept b is never touched
(b = 0 forced, see grid_search.py). Calibration slope a stays 0.8
for every variant, so the only difference is the rating update.

  - dynK:      dynamic K, K(n) = 32*100/(100+n), n = player's career
               match count in the walk (before this match). Newcomers
               adapt fast (K=32), veterans barely move (K->0).
  - dynK_floor: dynamic K with a floor, K(n) = 8 + 24*100/(100+n).
               K(0)=32, K(100)=20, K(500)=12, K(1000)=10, floor 8.
  - decay365:  time decay, rating regresses toward 1500 with a
               365-day half-life: r_eff = 1500 + (r-1500)*0.5^(dt/365),
               dt = days since the player's last match. Applied lazily
               to overall and surface ratings (per-surface last date).
  - decay180:  same, 180-day half-life.
  - games:     games-weighted update (Angelini WElo rule, as in
               features.WEloState): step scaled by f = winner games /
               total games; retirements and walkovers skipped (no
               reliable games margin), exactly as WEloState does.

All walks: 1968-2024 in (tourney_date, tourney_id, match_num) order;
Elo updates on EVERY match (production behaviour, all levels);
scored set is main-tour non-team 2023-2024 only (n=5,464).
Test set 2025-2026 is never touched.

Usage (repo root, project venv):
    .venv/bin/python -m scripts.backtest.run_elo_diagnosis

Writes scripts/backtest/outputs/elo_diagnosis.json (git-ignored).
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
from elo import INITIAL_ELO, expected_win_prob  # noqa: E402
from features import parse_score  # noqa: E402
from logreg import load_matches_with_serve  # noqa: E402
from metrics import score_outcome  # noqa: E402
from run_logreg_serve import _paired_vs_baseline  # noqa: E402
from surface_elo import blended_elo, win_prob  # noqa: E402
from tennis_tools import calibrated_prob  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"
BASELINE_VAL_BRIER = 0.21846441507745834  # locked blended Elo, same 5,464


def _is_scored(m):
    return m["tourney_level"] in MAIN_TOUR_LEVELS and not _is_team_exhibition(m["tourney_name"])


class FlexElo:
    """Overall (+optional surface) Elo with switchable update rules.

    Mirrors EloState exactly when surface=True, k_mode="fixed",
    decay_halflife=None, games_weighted=False (same formulas,
    same update order). The walk asserts that configuration
    reproduces the locked baseline Brier.
    """

    def __init__(self, k=32.0, prior=100, cal_a=0.8, surface=True,
                 k_mode="fixed", decay_halflife=None, games_weighted=False):
        self.k = k
        self.prior = prior
        self.cal_a = cal_a
        self.surface_on = surface
        self.k_mode = k_mode
        self.decay_halflife = decay_halflife
        self.games_weighted = games_weighted
        self.overall = {}
        self.surface_rating = {}
        self.surface_n = {}
        self.n_matches = {}
        self.last_date = {}
        self.last_surf_date = {}

    def _k_for(self, pid):
        if self.k_mode == "fixed":
            return self.k
        n = self.n_matches.get(pid, 0)
        if self.k_mode == "dyn":
            return self.k * 100.0 / (100.0 + n)
        if self.k_mode == "dyn_floor":
            return 8.0 + 24.0 * 100.0 / (100.0 + n)
        raise ValueError(f"unknown k_mode {self.k_mode}")

    def _decay(self, rating, last_date, date):
        if self.decay_halflife is None or last_date is None:
            return rating
        dt = (date - last_date).days
        if dt <= 0:
            return rating
        return INITIAL_ELO + (rating - INITIAL_ELO) * (0.5 ** (dt / self.decay_halflife))

    def _overall_eff(self, pid, date):
        return self._decay(self.overall.get(pid, INITIAL_ELO),
                           self.last_date.get(pid), date)

    def _surface_eff(self, pid, surface, date):
        if surface is None or not self.surface_on:
            return None, 0
        key = (pid, surface)
        r = self._decay(self.surface_rating.get(key, INITIAL_ELO),
                        self.last_surf_date.get(key), date)
        return r, self.surface_n.get(key, 0)

    def blended(self, pid, surface, date):
        overall = self._overall_eff(pid, date)
        if surface is None or not self.surface_on:
            return overall
        surf, n = self._surface_eff(pid, surface, date)
        return blended_elo(surf, overall, n, self.prior)

    def predict(self, player_a, player_b, surface, date):
        """(p_calibrated a=0.8, p_blended = uncalibrated a=1.0)."""
        ba = self.blended(player_a, surface, date)
        bb = self.blended(player_b, surface, date)
        p_bl = win_prob(ba, bb)
        p_cal = calibrated_prob(p_bl, a=self.cal_a, b=0.0)
        return p_cal, p_bl

    def update(self, winner_id, loser_id, surface, date,
               winner_games=0, total_games=0, retired=False, walkover=False):
        """Apply one match result. Call only AFTER predict()."""
        if self.games_weighted and (walkover or retired or total_games <= 0):
            return  # no reliable games margin; rating unchanged, no count
        f = (winner_games / total_games) if self.games_weighted else 1.0
        self._update_pair(winner_id, loser_id, None, date, f)
        if surface is not None and self.surface_on:
            self._update_pair(winner_id, loser_id, surface, date, f)
        self.n_matches[winner_id] = self.n_matches.get(winner_id, 0) + 1
        self.n_matches[loser_id] = self.n_matches.get(loser_id, 0) + 1

    def _update_pair(self, winner_id, loser_id, surface, date, f):
        if surface is None:
            w_r = self._overall_eff(winner_id, date)
            l_r = self._overall_eff(loser_id, date)
        else:
            w_r, _ = self._surface_eff(winner_id, surface, date)
            l_r, _ = self._surface_eff(loser_id, surface, date)
        exp = expected_win_prob(w_r, l_r)
        kw = self._k_for(winner_id)
        kl = self._k_for(loser_id)
        dw = kw * f * (1.0 - exp)
        dl = kl * f * (1.0 - exp)
        if surface is None:
            self.overall[winner_id] = w_r + dw
            self.overall[loser_id] = l_r - dl
            self.last_date[winner_id] = date
            self.last_date[loser_id] = date
        else:
            self.surface_rating[(winner_id, surface)] = w_r + dw
            self.surface_rating[(loser_id, surface)] = l_r - dl
            self.surface_n[(winner_id, surface)] = \
                self.surface_n.get((winner_id, surface), 0) + 1
            self.surface_n[(loser_id, surface)] = \
                self.surface_n.get((loser_id, surface), 0) + 1
            self.last_surf_date[(winner_id, surface)] = date
            self.last_surf_date[(loser_id, surface)] = date


def run_walk(matches, state):
    """One full walk. Returns winner-side probs for scored matches.

    val_p: validation 2023-2024 (main-tour non-team) calibrated probs.
    val_p_raw: same matches, uncalibrated (a=1.0) probs.
    train_p: train 2001-2022 scored matches, calibrated (sanity).
    """
    val_p, val_p_raw, train_p = [], [], []
    n_val = 0
    for m in matches:
        date = m["tourney_date"]
        year = date.year
        w_id, l_id = m["winner_id"], m["loser_id"]
        if year >= 2001 and _is_scored(m):
            p_cal, p_bl = state.predict(w_id, l_id, m["surface"], date)
            if year >= 2023:
                val_p.append(p_cal)
                val_p_raw.append(p_bl)
                n_val += 1
            else:
                train_p.append(p_cal)
        w_games, l_games, _sets, retired, walkover = parse_score(m.get("score"))
        state.update(w_id, l_id, m["surface"], date,
                     w_games, w_games + l_games, retired, walkover)
    assert n_val == 5464, f"validation scored n={n_val}, expected 5464"
    return val_p, val_p_raw, train_p


def summarize(label, val_p, base_p_win, extra=None):
    m = score_outcome(val_p)
    out = {
        "label": label,
        "n_val": m["n"],
        "val_brier": m["brier"],
        "val_log_loss": m["log_loss"],
        "val_accuracy": m["accuracy"],
        "paired_vs_baseline": _paired_vs_baseline(val_p, base_p_win),
    }
    if extra:
        out.update(extra)
    d = out["paired_vs_baseline"]
    print(f"  {label}: val Brier={m['brier']:.6f} logloss={m['log_loss']:.4f} "
          f"acc={m['accuracy']:.4f} vs baseline D={d['D']:.6f} "
          f"se={d['se']:.6f} z={d['z']:.2f}")
    return out


def main():
    conn = get_connection()
    matches = load_matches_with_serve(conn)
    print(f"Matches loaded (to 2024): {len(matches)}")

    results = {}
    probs = {}

    # --- Control: blended Elo must reproduce the locked baseline. ---
    base_state = FlexElo()
    base_p, _base_raw, base_train = run_walk(matches, base_state)
    probs["blended_baseline"] = base_p
    base_m = score_outcome(base_p)
    print(f"Blended baseline reproduction: val Brier={base_m['brier']:.6f} "
          f"(locked 0.218464), train Brier={score_outcome(base_train)['brier']:.4f}")
    assert abs(base_m["brier"] - BASELINE_VAL_BRIER) < 0.0005, (
        f"FlexElo default does not reproduce locked baseline: {base_m['brier']}")
    print("  reproduction OK")
    results["blended_baseline"] = summarize(
        "blended baseline (locked)", base_p, base_p,
        extra={"note": "FlexElo default; must equal locked 0.218464"})

    # --- Task 1: simplest standard Elo (single rating, K32, no surface). ---
    simple_state = FlexElo(surface=False)
    s_p, s_raw, s_train = run_walk(matches, simple_state)
    probs["simple_elo_a08"] = s_p
    probs["simple_elo_a10"] = s_raw
    results["simple_elo_a08"] = summarize(
        "simple Elo, single rating, K32, a=0.8", s_p, base_p,
        extra={"train_brier": score_outcome(s_train)["brier"],
               "note": "diagnosis control: no surface split"})
    results["simple_elo_a10"] = summarize(
        "simple Elo, single rating, K32, a=1.0 (uncalibrated)", s_raw, base_p,
        extra={"note": "same walk as a=0.8; calibration does not affect updates"})

    # --- Task 2: variants on the blended Elo. ---
    variants = [
        ("blended_dynK",
         FlexElo(k_mode="dyn"),
         "dynamic K, K(n)=32*100/(100+n), n=career matches"),
        ("blended_dynK_floor",
         FlexElo(k_mode="dyn_floor"),
         "dynamic K with floor, K(n)=8+24*100/(100+n)"),
        ("blended_decay365",
         FlexElo(decay_halflife=365),
         "time decay, rating -> 1500, half-life 365 days"),
        ("blended_decay180",
         FlexElo(decay_halflife=180),
         "time decay, rating -> 1500, half-life 180 days"),
        ("blended_games",
         FlexElo(games_weighted=True),
         "games-weighted update f=winner games/total games; RET/WO skipped"),
    ]
    for key, state, note in variants:
        v_p, _v_raw, v_train = run_walk(matches, state)
        probs[key] = v_p
        results[key] = summarize(
            f"{key} ({note})", v_p, base_p,
            extra={"train_brier": score_outcome(v_train)["brier"], "note": note})

    # Paired D/z of each blended variant vs the SIMPLE Elo too
    # (isolates the adjustment from the surface blend).
    for key in ["blended_dynK", "blended_dynK_floor", "blended_decay365",
                "blended_decay180", "blended_games"]:
        results[key]["paired_vs_simple_elo"] = _paired_vs_baseline(
            probs[key], probs["simple_elo_a08"])
        d = results[key]["paired_vs_simple_elo"]
        print(f"  {key} vs simple Elo: D={d['D']:.6f} se={d['se']:.6f} z={d['z']:.2f}")

    out = {
        "model": "Elo variants, walk-forward, scored on validation 2023-2024 main-tour non-team (n=5,464)",
        "run_date": _dt.date.today().isoformat(),
        "baseline_val_brier": BASELINE_VAL_BRIER,
        "selection_note": "no test set (2025-2026) touched; no calibration intercept b",
        "variants": results,
    }
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "elo_diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
