# TSA walk-forward backtest framework (baseline only)

Scores the locked TSA model against historical results and, when
odds files are present, against de-vigged bookmaker closing prices.
Framework + baseline only: no tuning, no test-set runs, no new
features in this package.

## Locked model (the baseline)

`K32 overall Elo + surface blend prior100 + calibration a=0.8, b=0.0`
(see `scripts/tennis_tools.py`):

- Overall Elo: start 1500, K=32.
- Surface Elo: per surface, start 1500, K=32, updated only on
  that surface.
- Blend: `blended = w*surface + (1-w)*overall`,
  `w = n_surface/(n_surface+100)`; no matches on the surface
  means overall Elo only; NULL surface means overall only.
- Calibration: `p_final = sigmoid(0.8 * logit(p_blended))`.

## Elo reuse (no second Elo implementation)

The walk-forward state (`elo_state.py`) imports the production
formulas verbatim: `scripts/elo.py` (`update_elo`, K=32) and
`scripts/surface_elo.py` (`blended_elo`, `win_prob`), plus
`calibrated_prob` from `scripts/tennis_tools.py`. The only new
code is the sequential state container and the pre-match
predict-then-update loop.

Verification: every run recomputes Elo from `atp_matches` (1968 on)
and compares it against the stored `atp_elo_history` /
`atp_surface_elo_history` tables, which were built by the same
sequential process. Last run: 195,078 matches checked to 2024,
max overall difference 5e-12, max surface difference 5e-12,
0 mismatches. The run stops if either exceeds 1e-6.

Known quirk found by that check: player id 199999 ("Unknown",
a placeholder for unnamed players in early Sackmann data)
plays himself in a few pre-1970 matches (winner_id ==
loser_id). The history tables store the pairwise computed
after-values for such a match; the dict state collapses them.
The verifier compares pairwise values, so it passes. Player
199999 has 0 matches from 2001 on, so the scored period is
unaffected.

Ordering: `(tourney_date, tourney_id, match_num)`, the same
order `compute_elo.py` / `compute_surface_elo.py` use.
`tourney_date` is the tournament start (Monday), not the exact
match day; within a tournament, `match_num` orders early rounds
first. That is the repo's existing convention, documented here,
not silently changed.

## Splits (locked)

| Split | Years | Use |
|---|---|---|
| burn-in | 1968-2000 | Elo updates only, no scoring (123,468 matches) |
| train | 2001-2022 | Scoring; tuning happens here (later step, not this package) |
| validate | 2023-2024 | Scoring; model / parameter selection happens here |
| test | 2025-2026 | LOCKED. `run_baseline.py` refuses `--end-year >= 2025` |

## How to run

From the repo root, with the project venv (stdlib only, no new
dependencies):

```bash
# Outcome-only baseline (no odds needed)
.venv/bin/python -m scripts.backtest.run_baseline

# With odds files in scripts/backtest/data/ (see below)
.venv/bin/python -m scripts.backtest.run_baseline \
    --odds-dir scripts/backtest/data --odds-source valuebetennis
.venv/bin/python -m scripts.backtest.run_baseline \
    --odds-dir scripts/backtest/data --odds-source tennis-data
```

Tests:

```bash
.venv/bin/python -m pytest tests/test_backtest_framework.py -v
```

## Outputs (scripts/backtest/outputs/, git-ignored)

- `baseline_by_year.csv` – one row per year (outcome metrics,
  plus one row per bookmaker when odds are matched).
- `baseline_summary.json` – pooled train/validate metrics, run
  manifest (model string, date, DB path, git HEAD, odds files
  with SHA-256).
- `baseline_predictions.csv` – one row per scored match
  (model prob of the winner, market prob where matched).
- `alignment_report.json` – per-year, per-level match rates
  and unmatched reasons (only when odds are given).

## Metrics (single-sided binary Brier)

- Brier = mean((1 - p_winner)^2), range 0-1, coin flip 0.25.
  Hand-checked: p=0.7 for the winner gives 0.09.
- Log loss = mean(-log(p_winner)), coin flip 0.6931.
- Accuracy = share where the model gave the eventual winner
  p >= 0.5. ECE bins the favourite probability
  (0.5-0.6 ... 0.9-1.0).
- Market comparison (matched matches only): D =
  mean(Brier_model - Brier_market) per match, SE = sd(d)/sqrt(n),
  z = D/SE. Positive D means the model is worse than the market.
- De-vigging: each side's implied probability (1/odds) divided
  by the sum of the two, so the two de-vigged probs sum to 1.

## Odds sources

### tennis-data.co.uk (primary, BLOCKED at download)

One Excel file per season:
`https://www.tennis-data.co.uk/{year}/{year}.xlsx` (ATP),
columns Winner/Loser/Tournament/Round plus B365W/B365L
(Bet365) and PSW/PSL (Pinnacle). Place the files as
`scripts/backtest/data/{year}.xlsx`.

Status 2026-11-17 (updated 2026-11-18): the site returns Cloudflare
403 to every automated fetch from this VM and to the user's own
US Wi-Fi and mobile data (Ray ID a5c23d0c0e4a8cc3); a friend in
Europe opens it fine, so the block is IP/region-based, not the
site being down. Real 2022-2025 files arrived via that friend and
are in `scripts/backtest/data/` (2022-2025.xlsx). First run
against real files exposed a real bug in `names.name_key`
(multi-word surnames with an initial, e.g. "O Connell C.",
"De Minaur A.", "Kwon S.W."): the 2022-2024 match rate sat at
~72% until it was fixed; after the fix it is 92.8-93.5% on main
tour (95.1-95.3% excluding team/exhibition). See names.py for
the cases. The 2025 file has two extra columns (BFEW/BFEL,
Betfair) which the reader ignores; its rows are loaded as the
adjacent-year pool for 2024 matches, but 2025 itself is NOT
scored (test set locked).

Because of this block, the 98-99.5% alignment target for
tennis-data could NOT be measured in this run.

### Valuebetennis (fallback, Pinnacle closing only, 2021+)

`https://www.valuebetennis.com/datasets/valuebetennis-matchs-{year}.csv`,
semicolon CSV, CC BY 4.0, downloaded 2026-11-17 (files in
`scripts/backtest/data/`, git-ignored). Columns: date (actual
match date), tournoi, categorie, genre, surface, joueur1/2,
cote1/2_ouverture, cote1/2_cloture. Only Pinnacle closing odds;
no Bet365; nothing before 2021. Closing-odds coverage of ATP
main-tour rows: 2021: 0/3977 (none), 2022: 3034/4194 (72%),
2023: 4079/4268 (96%), 2024: 3258/3494 (93%). Numbers from
this source are labelled `valuebetennis` / `pinnacle` in the
outputs and are NOT the tennis-data baseline.

## Alignment (align.py)

A Sackmann match is matched only when exactly one odds row
survives: same unordered player pair ((surname, first-initial),
accents stripped, apostrophes deleted so O'Connell ==
Oconnell), odds date in [tourney_date - 4 days,
tourney_date + 13 days] (Monday start, Sunday/Friday starts,
year-boundary season openers), round equality when the odds
file has a round, tournament-name token overlap as tie-breaker.
Anything else is reported unmatched with a reason (`no_pair`,
`no_date_window`, `ambiguous`), never forced. Adjacent years'
odds files are loaded into each year's pool (a January match
can sit in the previous year's file). The headline rate is over
Sackmann main-tour matches (levels A/F/G/M); Davis Cup (D)
and Olympics (O) are reported separately, as are United Cup /
Laver Cup / Next Gen Finals (team/exhibition events odds
providers do not publish).

## Baseline results (this run, 2026-11-17)

Outcome-only, locked model:

| Split | n | Brier | Log loss | Accuracy | ECE |
|---|---|---|---|---|---|
| train 2001-2022 | 65,466 | 0.2093 | 0.6051 | 0.6676 | 0.0206 |
| validate 2023-2024 | 6,144 | 0.2188 | 0.6260 | 0.6405 | 0.0063 |

Per-year Brier is in `baseline_by_year.csv` (train range
0.1963-0.2187; 2023: 0.2199, 2024: 0.2178). For reference,
the README's earlier 2024-2025 backtest quoted log loss
0.6268 / Brier 0.2193, consistent with the 2023-2024 numbers
here (different years, same model).

Vs Pinnacle closing (Valuebetennis fallback, de-vigged):

| Split | n matched | Market Brier | D (model - market) | SE | z |
|---|---|---|---|---|---|
| train 2022 only | 1,985 | 0.1976 | 0.0151 | 0.0023 | 6.64 |
| validate 2023-2024 | 4,856 | 0.2005 | 0.0173 | 0.0013 | 12.95 |

Per year: 2023 D=0.0167 (n=2,687), 2024 D=0.0182 (n=2,169).
The gap 0.015-0.017 is in the expected range from the Brier
report (TSA estimated 0.015-0.025; Nick's pure-Elo baseline
0.019). These are Valuebetennis-Pinnacle numbers, not
tennis-data Bet365/Pinnacle numbers; the training-period
market gap (2001-2021) is still unmeasured pending the
tennis-data download.

## Grid search (validation only, run 2026-11-18)

Script: `scripts/backtest/grid_search.py`. Stage 1 grid, 12 Elo walks
x 4 calibration slopes = 48 combos, scored on validation 2023-2024
main-tour non-team matches (n=5,464):

| Parameter | Values searched |
|---|---|
| K | 16, 24, 32, 48 |
| surface prior | 50, 100, 200 |
| calibration slope a | 0.7, 0.8, 0.9, 1.0 |
| calibration intercept b | 0.0 only (see below) |

Top 5 by validation Brier:

| Rank | K | prior | a | b | Brier | Log loss | Accuracy |
|---|---|---|---|---|---|---|---|
| 1 | 32 | 100 | 0.8 | 0.0 | 0.218464 | 0.6253 | 0.6409 |
| 2 | 32 | 200 | 0.8 | 0.0 | 0.218535 | 0.6255 | 0.6407 |
| 3 | 48 | 100 | 0.7 | 0.0 | 0.218613 | 0.6257 | 0.6429 |
| 4 | 32 | 50 | 0.8 | 0.0 | 0.218678 | 0.6259 | 0.6407 |
| 5 | 48 | 200 | 0.7 | 0.0 | 0.218715 | 0.6259 | 0.6396 |

Worst combo (K=16, prior=50, a=0.7): 0.222247. Whole-grid range
is 0.0038, and the top 5 sit within 0.0003 of each other: the grid
is flat. The best config IS the locked baseline (K32 / prior100 /
a=0.8 / b=0.0), so the paired difference vs baseline is D=0.0000
(z=0.00), vs Pinnacle D=0.0163 (SE=0.0013, z=12.60, n=5,202),
train Brier 0.2091. **Verdict: KEEP BASELINE.** No parameter
change, no production change. Full ranking in
`outputs/grid_search.json`.

Why b is not searched (a real finding, not a shortcut): the first
run of the script searched b in {-0.1, 0.0, +0.1} and "found"
b=+0.1 winning by 0.0196 Brier and beating Pinnacle (D=-0.0031).
That is leakage, not signal. This backtest scores every match
from the winner's side (predict is called with winner_id first),
so a positive b always inflates the eventual winner's
probability; the top 20 rows were all b=+0.1 while the best
b=-0.1 row was 0.2383. Separately, production returns
prob_b = 1 - prob_a, so the model must satisfy p(A beats B) +
p(B beats A) = 1, which sigmoid(a*logit(p) + b) only satisfies
for b = 0. Any b != 0 makes "A vs B" and "B vs A" disagree on
the same match. b = 0 is forced by the model's own symmetry.
Slope a is symmetric-safe, so a is the only calibration
parameter searched. Do not re-add b values without fixing the
scoring side first.

## Elastic Net container, first feature batch: serve/return (run 2026-11-18)

Scripts: `scripts/backtest/features.py` (walk-forward serve state),
`scripts/backtest/logreg.py` (container), `scripts/backtest/run_logreg_serve.py`.
Run: `.venv/bin/python -m scripts.backtest.run_logreg_serve`.
Results in `outputs/logreg_serve.json`. scikit-learn 1.9.1 is required
and was installed into the project venv on 2026-11-18
(`.venv/bin/python -m pip install scikit-learn`, pulls numpy/scipy);
the logreg tests skip nothing and need it.

### Features (all A-minus-B differences, walk-forward only)

Per player, from matches already processed (never the current match):
hold rate = service games won / played, break rate = return games
won / played. Service games won = sv_gms - (bp_faced - bp_saved);
return games won = opponent's bp_faced - bp_saved. Shrinkage with
K = 50 games: overall rate shrunk toward the tour average, surface
rate shrunk toward the player's own overall rate. Tour hold priors,
computed once from train-period data only (2001-2022, both players):
Hard 0.7909, Clay 0.7464, Grass 0.8251, Carpet 0.8141, overall 0.7819
(break prior = 1 - hold prior). Matches with NULL serve columns
(pre-2001 coverage ~26%, 2001+ ~91%) do not update the state; a
player with no history falls back to the tour prior, so the diff is 0.

Columns: `elo_diff` (locked blended Elo, A minus B), `hold_diff`,
`break_diff`. A is the lower player id (outcome-independent order);
label is 1 if A won. Features are standardized with train-only
mean/std. The container is logistic regression with NO INTERCEPT
(`fit_intercept=False`): production requires p(A beats B) +
p(B beats A) = 1, which a nonzero intercept breaks, and winner-side
scoring would leak through it (same leak as the calibration
intercept b in the grid search). Swapping A/B negates the features
and returns exactly 1 - p; this is tested.

### Hyperparameters (selected on validation only)

C in {0.01, 0.1, 1.0} x l1_ratio in {0.0, 0.5, 1.0} (9 combos).
Selection metric: validation Brier. The grid is flat: all 9
elo+serve combos land at Brier 0.2180-0.2181 (z -0.94 to -1.24),
so the edge pick (C=0.01, l1_ratio=1.0) beats C=1.0 by only 0.0001.

### Ablation (validation 2023-2024, main-tour non-team, n=5,464)

| Model | C | l1_ratio | Val Brier | Log loss | Acc | Paired D vs baseline | z | vs Pinnacle D |
|---|---|---|---|---|---|---|---|---|
| elo-only (sanity) | 0.01 | 1.0 | 0.2184 | 0.6251 | 0.6400 | -0.0001 (SE 0.0002) | -0.32 | 0.0161 (n=5,202) |
| elo + serve/return | 0.01 | 1.0 | 0.2180 | 0.6242 | 0.6426 | -0.0005 (SE 0.0004) | -1.24 | 0.0158 (n=5,202) |

Locked baseline for reference: Brier 0.2185, vs Pinnacle D=0.0163.
Selected elo+serve coefficients (standardized): elo_diff 0.85,
hold_diff 0.16, break_diff 0.13; train Brier 0.2076 (no overfit
signal). Serve/return helps in the right direction but only by
0.0005 Brier, 1.24 standard errors: **verdict NO EVIDENCE, do not
keep**. The production model is unchanged. The next feature batches
(form, fatigue, WElo diff, Glicko RD, age, head-to-head) go through
the same container and the same >= 2 SE adoption rule.

## Elastic Net container, all remaining feature batches (run 2026-11-18)

Scripts: `scripts/backtest/features.py` (FormState, FatigueState,
WEloState, H2HState, parse_score, age_years),
`scripts/backtest/glicko.py` (Glicko-2, weekly rating periods),
`scripts/backtest/logreg.py` (container, full 21-column list),
`scripts/backtest/run_logreg_all.py` (one walk, sequential forward
selection). Run: `.venv/bin/python -m scripts.backtest.run_logreg_all`.
Results in `outputs/logreg_all_batches.json`.

### Method

One walk (1968-2024) builds all 21 feature columns; batches are
tested in order, each on top of the currently ADOPTED set only.
Adopt iff paired Brier z vs current best <= -2.0 on validation
(2023-2024, main-tour non-team, n=5,464). Same C/l1 grid as the
serve batch, selected on validation only. No intercept, same as
before. Test set 2025-2026 never touched.

Caveat: ~9 adoption tests on one validation set at 2 SE each, so
the family-wise false-adoption risk is above 2.5% (roughly 1 in 5
for at least one false positive across 9 tests). An adopted batch
is a candidate until the locked test set (2025-2026) confirms it;
fatigue at z=-2.54 is the closest to the line and the one to be
most skeptical about, age at z=-3.46 is more solid.

### Feature definitions (all A-minus-B diffs, walk-forward only)

- Form (`FormState`): per-player (date, won, p_expected) history,
  p_expected = locked model's calibrated pre-match P(player wins).
  form2m = win% in last 60 days, form10 = last 10 matches, both
  shrunk toward 0.5 (k=5 / k=2) and centred at 0. resid2m/resid10
  = shrunk mean of (won - p_expected), i.e. how much the player
  beat Elo's expectation. Raw form overlaps Elo by construction;
  the residual is the cleaner signal and was only tested because
  raw failed.
- Fatigue (`FatigueState`): rest = days since last match with
  games > 0 (walkovers do not count), capped at 60, 7 if no
  history; games7/games14 = total games played in matches dated
  within the last 7/14 days before the current date; prevmin =
  minutes of the most recent timed match, else the player's
  running mean, else the train-period tour mean (106.7 min).
- WElo (`WEloState`): parallel Elo, K=32, same surface-blend
  prior-100; update weighted by f = winner games / total games
  (Angelini et al. 2022). Retirements skipped (neutral handling,
  as in Pickledger); walkovers and unparseable scores skipped.
  Plain Elo still updates on retirements.
- Glicko-2 (`glicko.py`): direct port of Glickman 2022, steps
  1-8, tau=0.5, initial 1500/RD350/sigma0.06. Rating periods are
  ISO calendar weeks; all games in a week are simultaneous, so
  predictions use pre-week ratings and updates apply at week end.
  Idle RD growth is exact and lazy (phi^2 += sigma^2 per idle
  week). Regression test reproduces the paper's worked example
  (1500/200/0.06 -> 1464.06/151.52/0.05999, rating within 0.05:
  the paper's arithmetic used 4-decimal intermediates). Features:
  rd_diff (pre-week RDs) and elo_x_rdshrink = elo_diff /
  (1 + meanRD/350). One real bug found while building this: the
  volatility bracket search in the paper's step 5 must stop at the
  first a-k*tau with f > 0 (f(A) < 0 in that branch, so the
  bracket needs the opposite sign); the inverted condition hung
  forever. Fixed with iteration caps.
- Age: dob from atp_players (missing in only 7 matches of
  2001-2024, feature 0 there). age_diff plus age_diff x best_of-5
  (best_of column), age_diff x Elo closeness
  (1 - |2*p_blended - 1|), age_diff x games7_diff.
- Head-to-head (`H2HState`): prior meetings per pair, shrunk win%
  (wins + k*0.5)/(total + k) - 0.5, doubled to a win%-diff scale;
  0 when fewer than 3 prior meetings. k tried on validation.
- Score parsing (`parse_score`): winner games / loser games /
  sets / retired / walkover from the Sackmann score string.
  Bracketed match tiebreaks "[10-7]" (58 matches, 2001-2024)
  count as a 7-6 set: points are not games, and 7-6 is the least
  distorting one-set proxy.

### Results (validation 2023-2024, main-tour non-team, n=5,464)

| Batch | C | l1 | Val Brier | Paired D vs baseline (z) | Paired D vs current best (z) | vs Pinnacle D | Verdict |
|---|---|---|---|---|---|---|---|
| elo-only (base) | 0.01 | 1.0 | 0.2184 | -0.0001 (-0.32) | 0.0000 (0.00) | 0.0161 | base |
| form-raw | 0.01 | 0.5 | 0.2184 | -0.0001 (-0.38) | -0.0000 (-0.33) | 0.0161 | DROP |
| form-resid | 0.01 | 0.5 | 0.2183 | -0.0001 (-0.59) | -0.0001 (-0.89) | 0.0160 | DROP |
| fatigue | 0.01 | 0.0 | 0.2175 | -0.0009 (-2.27) | -0.0009 (-2.54) | 0.0152 | ADOPT |
| welo | 0.01 | 0.0 | 0.2173 | -0.0011 (-2.40) | -0.0002 (-0.82) | 0.0150 | DROP |
| glicko-A (rd_diff) | 0.01 | 0.0 | 0.2174 | -0.0011 (-2.25) | -0.0001 (-0.66) | 0.0149 | DROP |
| glicko-B (rd + shrink) | 1.0 | 1.0 | 0.2172 | -0.0012 (-2.35) | -0.0003 (-1.07) | 0.0147 | DROP |
| age | 0.01 | 0.0 | 0.2166 | -0.0019 (-3.80) | -0.0010 (-3.46) | 0.0142 | ADOPT |
| h2h (best k=10) | 0.01 | 0.0 | 0.2166 | -0.0019 (-3.69) | +0.0000 (+0.03) | 0.0143 | DROP |
| joint-serve | 0.01 | 0.0 | 0.2160 | -0.0024 (-3.96) | -0.0005 (-1.41) | 0.0138 | DROP |

Final adopted set (9 features): elo_diff, rest_diff, games7_diff,
games14_diff, prevmin_diff, age_diff, age_x_bo5, age_x_close,
age_x_load7. Validation Brier 0.2166 (baseline 0.2185), paired D vs
baseline -0.0019 (SE 0.0005, z=-3.80), vs Pinnacle D=0.0142
(n=5,202), down from 0.0163: about 13% of the gap to market closed.
Standardized coefficients: elo 0.97, rest -0.08, games7 -0.02,
games14 +0.10, prevmin -0.02, age -0.09, age_x_bo5 -0.00,
age_x_close +0.01, age_x_load7 -0.02. Train Brier 0.2074, no
overfit signal. Full suite: 282 passed, 1 skipped.

Reading the coefficients honestly: the fatigue batch's gain does
not look like pure tiredness. games14 is positive (+0.10: more
games in the last 14 days predicts winning) while games7 is
slightly negative (-0.02) and rest is negative (-0.08: longer rest
than the opponent predicts losing). That pattern is rhythm/rust,
not load: players who played a lot in the last two weeks are
usually players who kept winning, and long rest often means a
layoff or an early loss last week. The age batch's gain is mostly
the plain age_diff (-0.09); the three interactions are near zero
(-0.00 / +0.01 / -0.02), so the stamina-interaction story did not
show up in the numbers. WElo splits its weight evenly with Elo
(0.49 / 0.49) and adds nothing incremental: the two ratings are
nearly the same signal. Glicko-B's coefficients swing wildly
(elo -1.23, shrink +2.12): collinearity cancelling out, not a
real effect. Head-to-head is exactly zero (coef 0.06, z=+0.03).
Joint serve helps a bit more with the other features in (z=-1.41
vs -1.24 alone) but still misses the bar.

## Elastic Net container, alternative formulations (run 2026-11-18)

Scripts: `scripts/backtest/features.py` (RecentServeState,
fit_rate_vs_elo, serve_residual, DominanceState,
elo_high_rd_interaction), `scripts/backtest/logreg.py` (28 columns),
`scripts/backtest/run_logreg_alternatives.py` (one walk, each
alternative tested independently on top of the adopted set).
Results in `outputs/logreg_alternatives.json`.

### Method

Base: the adopted set from the all-batches round (9 columns,
validation Brier 0.2166, vs Pinnacle D = 0.0142). Each alternative
is added on top of that base only and adopted iff paired Brier z
vs the base <= -2.0. Two alternatives were adopted, so the final
set is the base PLUS both (12 columns, refit: Brier 0.2154).

- serve-recent: hold/break over the last 60 days only (K = 20
  games shrinkage), surface rate toward the player's recent overall
  rate. Hypothesis: current serving form, not career skill.
- serve-resid: actual hold/break minus the Elo-implied rate. The
  mapping is an OLS fit of hold/break on overall Elo over 1,367
  train player snapshots (end of 2022, >= 50 service games):
  hold = 0.3100 + 0.000279*Elo, break = 0.0041 + 0.000132*Elo.
  Frozen before validation is scored.
- age-x-serve: one column, age_diff * career hold_diff.
- dominance: mean games-share in the last 10 wins, shrunk toward
  0.5 (k = 2 wins); retirements and walkovers excluded.
- glicko-gate: one column, elo_diff when max(rd_a, rd_b) >= 84.5
  (median max-RD over train scored matches), else 0.
- glicko-shrink: elo_diff replaced by elo_diff / (1 + meanRD/350),
  head-to-head, not an extra column.

A real bug was caught while building this round: the first version
of the final-set logic overwrote the accumulated set with the last
passing alternative, so the refit (0.2162) was worse than
serve-recent alone (0.2156). Fixed by accumulating all passing
alternatives; the corrected final refit is 0.2154.

### Results (validation 2023-2024, main-tour non-team, n=5,464)

| Alternative | Kept? | Val Brier | Paired D vs base (z) | vs Pinnacle D |
|---|---|---|---|---|
| serve-recent | ADOPT | 0.2156 | -0.0010 (-3.35) | 0.0132 |
| serve-resid | DROP | 0.2166 | -0.0000 (-0.20) | 0.0143 |
| age-x-serve | ADOPT | 0.2162 | -0.0003 (-2.16) | 0.0139 |
| dominance | DROP | 0.2164 | -0.0002 (-0.61) | 0.0139 |
| glicko-gate | DROP | 0.2165 | -0.0001 (-1.55) | 0.0142 |
| glicko-shrink | DROP | 0.2167 | +0.0001 (+1.26) | 0.0144 |
| **final (base + 2 adopted)** | | **0.2154** | **-0.0011 (-3.61)** | **0.0130** |

Final adopted set (12 columns): elo_diff, rest_diff, games7_diff,
games14_diff, prevmin_diff, age_diff, age_x_bo5, age_x_close,
age_x_load7, hold_rec_diff, break_rec_diff, age_x_hold_diff.
Vs baseline (0.2185): D = -0.0030, z = -5.20. Age-x-serve at
z=-2.16 is the closest to the line in this round.

## Elastic Net container, benchmark features (run 2026-11-18)

Scripts: `scripts/backtest/features.py` (WinRateState, VenueState,
tournament_country + TOURNAMENT_COUNTRY map, OLYMPIC_CITY_COUNTRY,
year-based mover tables), `scripts/backtest/logreg.py` (36 columns
at the time; 50 after the round-4 append -- see below),
`scripts/backtest/run_logreg_benchmark.py` (one walk, sequential
forward selection). Results in `outputs/logreg_benchmark.json`.

### Method

Base: the alternatives-round final set (12 columns, validation
Brier 0.2154, vs Pinnacle D = 0.0130). Batches in order, each on
top of the current best; adopt iff paired Brier z vs current best
<= -2.0.

- winpct: career win% (shrunk toward 0.5, K = 10 matches) and
  surface win% (shrunk toward the player's overall rate).
- experience: log1p career matches and log1p surface matches.
- ranking: log-ranking diff (log rank_B - log rank_A, positive
  means A is better ranked) + log(points+1) diff, from the
  PREVIOUS YEAR's year-end ranking (atp_year_end_rankings).
  The summary table (atp_player_ranking_summary) holds only the
  latest ranking and would leak, so it is NOT used; the cost is
  up to ~12 months staleness. Missing either side -> 0.
  Coverage: both present for 5,450/5,464 validation matches (99.7%),
  59,157/59,284 train (99.8%).
- welo-swap: WElo REPLACES elo_diff in the current set,
  head-to-head (Angelini/Pickledger form, not the stacked form
  dropped earlier at z=-0.82).
- venue: win-rate excess at this tournament (tourney_id),
  shrunk toward the player's own overall rate (K = 5); 0 with no
  prior match there. Coverage: >= 3 prior matches at the venue
  for either player in only 583/5,464 validation matches (10.7%).
- home: 1 if the player's IOC matches the tournament country
  (A-minus-B, -1/0/1). Map covers all 149 distinct 2001-2024
  tourney names that could be placed from the name; Davis Cup
  host = first-named team; Olympics by city; Tour Finals / Laver
  Cup / NextGen by year; United Cup / Atp Cup = AUS. Unknown
  country -> 0. Coverage: country known for 5,464/5,464
  validation matches (100%), flag nonzero in 1,216. Limitation:
  IOC is the player's listed (current) nationality, not
  necessarily the one held at match time.

### Results (validation 2023-2024, main-tour non-team, n=5,464)

| Batch | Kept? | Val Brier | Paired D vs best (z) | vs Pinnacle D |
|---|---|---|---|---|
| winpct | DROP | 0.2153 | -0.0001 (-0.38) | 0.0129 |
| experience | ADOPT | 0.2138 | -0.0016 (-2.76) | 0.0111 |
| ranking | ADOPT | 0.2128 | -0.0010 (-2.79) | 0.0099 |
| welo-swap | DROP | 0.2126 | -0.0002 (-0.80) | 0.0098 |
| venue | DROP | 0.2129 | +0.0001 (+0.98) | 0.0100 |
| home | DROP | 0.2128 | +0.0000 (+0.15) | 0.0100 |
| **final** | | **0.2128** | | **0.0099** |

Final adopted set (16 columns): the 12 base columns plus
exp_diff, surf_exp_diff, logrank_diff, logpoints_diff.
Vs baseline (0.2185): D = -0.0056, z = -6.03; vs Pinnacle the gap
is 0.0163 -> 0.0099, about 39% closed.

Coefficient reading (standardized, final fit): the two experience
columns split -- surface experience is strongly positive (+0.49)
while career experience is negative (-0.28) once surface is
controlled, i.e. a player with a huge career count but little
surface experience is typically an older journeyman. Ranking
points carry the signal (+0.20); log-rank itself is ~0 (-0.02),
so the position adds nothing beyond the points. Age grew to
-0.17 once experience is controlled. Venue failed for lack of
data (10.7% coverage); home had full coverage and still nothing
(z=+0.15).

Caveat (all three rounds): ~21 adoption tests on the same
validation set at 2 SE each, so the family-wise false-adoption
risk is now substantial (well above the 1-in-5 of the first
round). Experience (z=-2.76) and ranking (z=-2.79) are both near
the line; the cumulative 0.2154 -> 0.2128 step is the more solid
story. Everything adopted is a candidate until the locked test
set (2025-2026, after the TML merge) confirms it.

## Elastic Net container, round 4: cross-model benchmark items 1-6 (run 2026-11-18)

Scripts: `scripts/backtest/features.py` (PointServeState,
FormState.form_5/form_20/momentum_diff, parse_clutch +
ClutchState, EventState, RetirementState),
`scripts/backtest/logreg.py` (50 columns; point-serve columns
added to load_matches_with_serve),
`scripts/backtest/run_logreg_round4.py` (one walk, sequential
forward selection). Results in `outputs/logreg_round4.json`.
The benchmark runner pins N_COLS = 36 (FEATURE_NAMES has grown
since), following the same precedent as run_logreg_all (21) and
run_logreg_alternatives (28).

### Method

Base: the benchmark-round final set (16 columns, validation
Brier 0.2128, vs Pinnacle D = 0.0099). Items in order, each on
top of the current best; adopt iff paired Brier z vs current best
<= -2.0.

- point-serve: 1st-serve win%, 2nd-serve win%, ace rate, DF rate,
  BP conversion% and BP save%, per player, walk-forward,
  surface-aware shrunk rates (same pattern as ServeState:
  surface rate toward the player's own overall rate, overall
  toward the tour prior). Tour priors from train 2001-2022:
  1st 0.7116, 2nd 0.5059, ace 0.0758, DF 0.0376, BP conv 0.3970,
  BP save 0.6030; per-surface priors for 1st/2nd/ace
  (Clay 1st 0.6823 vs Grass 0.7393). Shrinkage K: 100 points
  (1st/2nd), 200 (ace/DF), 50 chances (BP conv/save). NULL
  serve columns -> match does not update the state.
  Coverage: point-serve data present for 5,426/5,464 validation
  matches (99.3%).
- momentum: form slope, last-5 win% minus last-20 win% per
  player, differenced (shrink K = 2, same as form_10).
  Trajectory, not level; form level/residual were both dropped.
- clutch: tiebreak win% (K = 10, toward 0.5), deciding-set
  record (K = 5, toward 0.5; deciding set = last set when sets
  == best_of; best_of None skips it), comeback after losing the
  first set (K = 5, toward the 0.1923 train tour rate). Set-level
  parsing in parse_clutch: a set counts as a tiebreak set when
  its token carries "(...)" points or is a bracketed match
  tiebreak; a bare "7-6" carries no tiebreak info. Coverage:
  any clutch column nonzero in 5,462/5,464 validation matches.
- event-games: games already played at this tournament before
  this match, as a diff. The walk visits matches in match_num
  order within a tournament, so the count is exact.
- retirement: 1 if the player retired in a match in the last 60
  days, and 1 if an opponent retired against them, as diffs.
  RET marks the loser as the retiree. Coverage: any flag nonzero
  in 1,553/5,464 validation matches (28.4%).
- elo-x-bo5: elo_diff * best-of-5 indicator, one column
  (best_of is 3 or 5 throughout 2001+, no NULLs).

### Results (validation 2023-2024, main-tour non-team, n=5,464)

| Item | Kept? | Val Brier | Paired D vs best (z) | vs Pinnacle D |
|---|---|---|---|---|
| point-serve | DROP | 0.2129 | +0.0001 (+0.51) | 0.0100 |
| momentum | DROP | 0.2128 | +0.0000 (+0.31) | 0.0099 |
| clutch | ADOPT | 0.2125 | -0.0003 (-2.18) | 0.0096 |
| event-games | DROP | 0.2124 | -0.0001 (-1.45) | 0.0095 |
| retirement | DROP | 0.2125 | +0.0000 (+0.06) | 0.0096 |
| elo-x-bo5 | ADOPT | 0.2121 | -0.0004 (-2.14) | 0.0093 |
| **final** | | **0.2121** | | **0.0093** |

Final adopted set (20 columns): the 16 base columns plus
tb_winpct_diff, decset_winpct_diff, comeback_diff, elo_x_bo5.
Vs baseline (0.2185): D = -0.0064, z = -6.65; vs Pinnacle the gap
is 0.0163 -> 0.0093, about 43% closed.

Coefficient reading (standardized, final fit): the clutch gain
does NOT come from tiebreaks (tb coefficient -0.005, ~0) but
from deciding sets (-0.068) and comebacks (+0.059) -- the
opposite of what the "clutch" label suggests, and the
literature's warning that clutch persistence is weak still
stands. Elo_x_bo5 takes +0.168 while the plain elo_diff slope
falls from 0.97 (round 1) to 0.60: the Elo slope splits, with
rating gaps mattering more over five sets, exactly as Pickledger
claims. Point-serve decomposition adds nothing beyond the
game-level hold/break the model already has (z=+0.51): the serve
story was already told by hold_rec/break_rec.

Caveat (all four rounds): ~27 adoption tests on the same
validation set at 2 SE each, so the family-wise false-adoption
risk is now high. Both round-4 adoptions sit just over the line
(clutch z=-2.18, elo-x-bo5 z=-2.14) and are the weakest adoptions
so far; treat them as the most likely to fail on the locked test
set (2025-2026, after the TML merge).

## Alignment results (tennis-data, real files 2022-2025, run 2026-11-18)

Main-tour match rate (excl. team/exhibition in brackets):

| Year | Rate | Unmatched reasons |
|---|---|---|
| 2022 | 93.5% (93.8%) | no_pair 290, no_date_window 68, ambiguous 26 |
| 2023 | 93.5% (95.1%) | no_pair 322, no_date_window 51, ambiguous 28 |
| 2024 | 92.8% (95.3%) | no_pair 404, no_date_window 107, ambiguous 14 |

The 16 tournaments missing from Valuebetennis 2024 (448 matches)
are 94.4% matched here (423/448). What is still unmatched is
mostly structural, not a bug: Davis Cup 259/260 and Olympics
63/64 are not in tennis-data at all; 2023 level-F is only the
NextGen Finals (15/15 unmatched, tennis-data does not cover it;
the real Tour Finals is "Tour Finals", level A in 2023, and its
15 matches sit in the A bucket); United Cup / Laver Cup team
matches are not in tennis-data either. The rest is spread thin
across regular events.

Vs tennis-data closing (de-vigged), validation 2023-2024 pooled:
Bet365 n=5,214, market Brier 0.2014, D=0.0160, SE=0.0013, z=12.48;
Pinnacle n=5,204, market Brier 0.2010, D=0.0163, SE=0.0013,
z=12.62. Per year, Pinnacle: 2023 D=0.0170 (n=2,361),
2024 D=0.0159 (n=2,503). This agrees with the Valuebetennis
Pinnacle gap (D=0.0173, n=4,856) within noise: two independent
market sources put the baseline gap at ~0.016-0.017.

## Alignment results (Valuebetennis fallback, earlier run)

Main-tour match rate (excl. team/exhibition in brackets):

| Year | Rate | Unmatched reasons |
|---|---|---|
| 2022 | 70.4% (70.3%) | no_pair 712, no_date_window 215, ambiguous 5 |
| 2023 | 94.4% (94.5%) | no_pair 254, no_date_window 30, ambiguous 15 |
| 2024 | 73.2% (72.9%) | no_pair 723, no_date_window 257, ambiguous 9 |

Why 2024 is low: 16 whole tournaments (448 matches) are
entirely absent from the Valuebetennis 2024 file (Montpellier,
Dallas, Marseille, Cordoba, Rotterdam, Delray Beach, Buenos
Aires, Doha, Rio, Los Cabos, Dubai, Acapulco, Santiago in
Feb 2024, plus Stockholm, Antwerp, Almaty in Oct 2024).
2023 has zero entirely-missing tournaments. This is a hole in
the fallback source, not an alignment bug. Davis Cup match
rate is low in all years (2024: 34/260) and Olympics 2024 is
60/64; both are reported separately and excluded from the
headline rate.

## Data gaps (for later feature work)

- Serve statistics (`w_svpt` present) by year, 2001-2024:
  87-90% in 2001-2015, 94-99% from 2016 on (2016: 99.1%,
  2024: 98.1%). A serve/return feature needs shrinkage and a
  missing-data fallback for the 2000s, not a full-history
  assumption.
- Pre-2001: no closing odds anywhere (tennis-data odds start
  2001), so the training window cannot start earlier for the
  market comparison. 1968-2000 is used only as Elo burn-in.
- 2021 Valuebetennis has no closing odds at all; 2022 only 72%.
- Test set 2025-2026: not run. The 2025 tennis-data file IS
  downloaded (2,637 rows, 2024-12-29 to 2025-11-16, Bet365
  2,631/2,637, Pinnacle 2,533/2,637) but not scored: the runner
  refuses --end-year >= 2025. The 2026 second half (TML merge)
  is still pending after the 2026-11-22 Finals, per the standing
  decision.

## Extension points (NOT implemented here)

- `elo_state.EloState` exposes `predict(player_a, player_b,
  surface)` and `update(winner, loser, surface)`. The Elastic Net
  container (`logreg.py`) plus the walk-forward states in
  `features.py` / `glicko.py` are now the model under test: grid
  search (2026-11-18, baseline kept), serve/return, form, fatigue,
  WElo, Glicko-2 RD, age, head-to-head, the six alternative
  formulations, and the six benchmark features have all been run
  through it (see the sections above). Adopted on validation, in
  order: fatigue, age, recent serve, age x serve, experience,
  ranking. Final validation Brier 0.2128 vs the locked baseline
  0.2185 (gap to Pinnacle 0.0163 -> 0.0099). The scorer in
  `run_baseline.py` calls only those two methods.
- Grid search and feature selection must be evaluated on the
  validation split (2023-2024), never on the test split.
  `run_baseline.py` refuses to score 2025-2026.
- The tennis-data reader was re-verified against real 2022-2025
  files on 2026-11-18 (see the tennis-data section above): it
  found and fixed a real name-matching bug. Any future odds
  source still gets the same treatment: run with
  `--odds-source <source>` and check the alignment report
  before trusting any market number from it.
