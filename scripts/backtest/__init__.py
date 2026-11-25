"""Walk-forward backtest framework for TSA (baseline only).

This package scores the locked TSA model against historical results
and, when odds files are present, against bookmaker closing prices.

Scope of this package (2026-11-17): framework + baseline only.
No hyperparameter tuning, no test-set runs, no new features here.
Tuning and feature work (grid search, Elastic Net, WElo, Glicko-2,
form / serve / H2H / fatigue / age) plug in later through the
``Model`` protocol in ``elo.py`` / ``run_baseline.py``; they are
NOT implemented in this package.

Layout:
    elo.py          Walk-forward Elo state. Reuses scripts/elo.py
                    (update_elo, K=32) and scripts/surface_elo.py
                    (blended_elo, win_prob) verbatim, so the rating
                    here is the same rating the production tools use.
                    (File is elo_state.py; scripts/elo.py is the
                    production module of the same name, so the
                    backtest one is named differently on purpose.)
    metrics.py      Single-sided Brier, log loss, accuracy, ECE,
                    paired model-vs-market difference, SE, z.
    names.py        Player-name normalisation for odds alignment.
    odds.py         Odds readers (tennis-data.co.uk xlsx, stdlib
                    only; Valuebetennis CSV) and de-vigging.
    align.py        Sackmann match <-> odds row alignment.
    run_baseline.py CLI entry point. Writes per-year CSV + JSON
                    under outputs/ (git-ignored).
    data/           Downloaded odds files (git-ignored, not committed).
    outputs/        Run outputs (git-ignored, not committed).

Splits (locked, see the Brier report 2026-11-16):
    burn-in   1968-2000  Elo only, no scoring.
    train     2001-2022  Scoring + tuning (tuning is a later step).
    validate  2023-2024  Scoring + model/parameter selection.
    test      2025-2026  LOCKED. This package refuses to score it.
"""
