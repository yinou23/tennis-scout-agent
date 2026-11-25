"""Scoring metrics for the backtest. Stdlib only.

Conventions (single-sided binary Brier, as in the Brier report):
- Every record is one match, scored from the winner's side:
  p is the probability the model assigned to the eventual winner.
  Brier = mean((1 - p) ** 2), range 0-1, coin flip 0.25.
  Log loss = mean(-log(p)), coin flip 0.6931.
  Accuracy = share where the favourite (p >= 0.5 for the winner's
  side, i.e. the model picked the winner) won. Here the winner
  is known, so accuracy = share of matches where the model gave
  the eventual winner p >= 0.5.
- ECE: bin by the model's favourite probability
  p_fav = max(p_a, 1 - p_a), actual = 1 if the favourite won.
  Bins are 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0.
  ECE = sum over bins of (n_bin / n) * |actual_rate - mean_p|.
- Paired model-vs-market difference: for each matched match,
  d = brier_model - brier_market (both single-sided, winner side).
  D = mean(d), SE = sd(d) / sqrt(n), z = D / SE.
  A positive D means the model is worse than the market.
"""

import math


def _clip(p):
    return min(max(float(p), 1e-9), 1.0 - 1e-9)


def score_outcome(p_winner_list):
    """Metrics vs the actual outcome. p_winner_list: model prob of winner."""
    n = len(p_winner_list)
    if n == 0:
        return None
    ps = [_clip(p) for p in p_winner_list]
    brier = sum((1.0 - p) ** 2 for p in ps) / n
    log_loss = sum(-math.log(p) for p in ps) / n
    accuracy = sum(1 for p in ps if p >= 0.5) / n
    return {"n": n, "brier": brier, "log_loss": log_loss, "accuracy": accuracy}


def ece(p_fav_list, fav_won_list):
    """Expected calibration error over favourite-probability bins."""
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    n = len(p_fav_list)
    if n == 0:
        return None
    total = 0.0
    detail = []
    for lo, hi in bins:
        idx = [i for i, p in enumerate(p_fav_list) if lo <= p < hi or (hi == 1.0 and p == 1.0)]
        if not idx:
            detail.append({"bin": f"{lo}-{hi}", "n": 0, "mean_p": None, "actual": None})
            continue
        mean_p = sum(p_fav_list[i] for i in idx) / len(idx)
        actual = sum(fav_won_list[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(actual - mean_p)
        detail.append({"bin": f"{lo}-{hi}", "n": len(idx), "mean_p": mean_p, "actual": actual})
    return {"ece": total, "bins": detail}


def paired_diff(model_p_winner, market_p_winner):
    """Paired Brier difference model minus market, with SE and z."""
    if len(model_p_winner) != len(market_p_winner):
        raise ValueError("model and market lists must have the same length")
    n = len(model_p_winner)
    if n == 0:
        return None
    ds = []
    for pm, pk in zip(model_p_winner, market_p_winner):
        pm, pk = _clip(pm), _clip(pk)
        ds.append((1.0 - pm) ** 2 - (1.0 - pk) ** 2)
    mean_d = sum(ds) / n
    var = sum((d - mean_d) ** 2 for d in ds) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else 0.0
    z = mean_d / se if se > 0 else 0.0
    return {"n": n, "D": mean_d, "se": se, "z": z}


def devig(odds_a, odds_b):
    """De-vig two decimal odds: implied probs each divided by their sum.

    Returns (p_a, p_b) with p_a + p_b = 1. Raises ValueError on
    non-positive or missing odds.
    """
    if not odds_a or not odds_b or float(odds_a) <= 1.0 or float(odds_b) <= 1.0:
        raise ValueError(f"bad odds for de-vigging: {odds_a}, {odds_b}")
    ia = 1.0 / float(odds_a)
    ib = 1.0 / float(odds_b)
    s = ia + ib
    return ia / s, ib / s
