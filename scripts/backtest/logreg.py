"""Elastic Net logistic-regression container for the backtest.

The container takes per-match A-minus-B difference features and fits
p(A wins) = sigmoid(w . x). Features are differences only, so swapping
A and B negates x and the model returns 1 - p exactly.

NO INTERCEPT. This is a hard constraint, not a tuning choice. The
production model requires p(A beats B) + p(B beats A) = 1
(scripts/tennis_tools.py returns prob_b = 1 - prob_a), which
sigmoid(w . x + b) satisfies only for b = 0. A nonzero intercept also
leaks in this backtest: rows are scored from the winner's side, so a
positive intercept inflates the eventual winner's probability (the
same leak that invalidated the calibration-intercept grid search in
grid_search.py). Every fit here uses fit_intercept=False, and the
symmetry is tested in tests/test_backtest_framework.py.

Player order (A vs B) is fixed independently of the outcome: A is the
player with the LOWER player id. winner_id/loser_id are
outcome-dependent and must never define the order. Label is 1 if A
won, 0 otherwise.

Feature columns (in order), A-minus-B diffs throughout, A = lower id:
  elo_diff        blended Elo (locked model) of A minus B, Elo points
  hold_diff       shrunk hold rate of A minus B (features.ServeState)
  break_diff      shrunk break rate of A minus B
  form2m_diff     recent-2-month win% (shrunk, centred) A minus B
  form10_diff     last-10 win% (shrunk, centred) A minus B
  resid2m_diff    recent-2-month mean (won - Elo-expected) A minus B
  resid10_diff    last-10 mean (won - Elo-expected) A minus B
  rest_diff       days since last played match A minus B (capped 60)
  games7_diff     games played in the last 7 days A minus B
  games14_diff    games played in the last 14 days A minus B
  prevmin_diff    minutes of previous match A minus B
  welo_diff       blended WElo (games-weighted Elo) A minus B
  rd_diff         Glicko-2 RD (pre-week) A minus B, rating points
  elo_x_rdshrink  elo_diff shrunk by mean RD: elo_diff / (1 + meanRD/350)
  age_diff        age in years at match date A minus B
  age_x_bo5       age_diff * best-of-5 indicator (best_of == 5)
  age_x_close     age_diff * Elo closeness 1 - |2*p_blended - 1|
  age_x_load7     age_diff * games7_diff
  h2h_k3          shrunk head-to-head win% diff, k = 3 (< 3 meetings: 0)
  h2h_k5          same, k = 5
  h2h_k10         same, k = 10
  hold_rec_diff   recent-60-day shrunk hold rate A minus B
  break_rec_diff  recent-60-day shrunk break rate A minus B
  hold_resid_diff actual hold minus Elo-implied hold, A minus B
  break_resid_diff actual break minus Elo-implied break, A minus B
  age_x_hold_diff age_diff * career hold_diff (serve matters more older)
  dominance_diff  games-share in last 10 wins, shrunk, A minus B
  elo_x_highrd    elo_diff when max RD >= train-median threshold, else 0
  winpct_diff     career win% (shrunk, centred) A minus B
  surf_winpct_diff surface win% (shrunk) A minus B
  exp_diff        log1p career matches played A minus B
  surf_exp_diff   log1p surface matches played A minus B
  logrank_diff    log ranking of B minus log ranking of A (positive
                  means A is better ranked); previous-year year-end
                  ranking, missing -> 0
  logpoints_diff  log(points+1) A minus B, previous-year year-end
  venue_winpct_diff win-rate excess at this tournament A minus B
  home_diff       1 if A's IOC matches the tournament country minus
                  the same for B (-1/0/1); unknown country -> 0
  first_win_diff  1st-serve points won% A minus B (point serve state)
  second_win_diff 2nd-serve points won% A minus B
  ace_rate_diff   aces per service point A minus B
  df_rate_diff    double faults per service point A minus B
  bp_conv_diff    break points converted% A minus B
  bp_save_diff    break points saved% A minus B
  momentum_diff   (last-5 win% minus last-20 win%) of A minus the
                  same for B: form trajectory, not level
  tb_winpct_diff  tiebreak win% A minus B (shrunk toward 0.5)
  decset_winpct_diff deciding-set win% A minus B (shrunk toward 0.5)
  comeback_diff   win% of matches after losing the first set,
                  A minus B (shrunk toward the 0.19 tour rate)
  event_games_diff games already played at this tournament before
                  this match, A minus B
  ret_self_diff   1 if A retired in a match in the last 60 days
                  minus the same for B
  ret_opp_diff    1 if an opponent retired against A in the last
                  60 days minus the same for B
  elo_x_bo5       elo_diff * best-of-5 indicator (best_of == 5)
  home_x_elo      home_diff * elo_diff (signed interaction: negative
                  when the home player is the Elo underdog -- crowd
                  support should matter most there)
  home_x_underdog max(0, -elo_diff * home_diff): how much of an Elo
                  underdog the home player is, 0 otherwise
  home_x_fav      max(0, elo_diff * home_diff): how much of an Elo
                  favourite the home player is, 0 otherwise

Columns appended after the first 21 keep the original indices
stable; the all-batches results reference them by name.

Standardization: each column is standardized with the TRAIN mean/std
only (2001-2022); validation uses the same frozen scaler. No
validation information enters the scaler or the fit.

Fit: sklearn LogisticRegression, solver saga, fit_intercept=False.
sklearn 1.9 deprecated the ``penalty`` argument, so the penalty is
selected with ``l1_ratio`` alone: 0.0 is pure L2, 1.0 pure L1, in
between Elastic Net. C and l1_ratio are selected on validation Brier
only (see run_logreg_serve.py).
"""

import datetime as _dt
import math

import numpy as np

FEATURE_NAMES = [
    "elo_diff", "hold_diff", "break_diff",
    "form2m_diff", "form10_diff", "resid2m_diff", "resid10_diff",
    "rest_diff", "games7_diff", "games14_diff", "prevmin_diff",
    "welo_diff", "rd_diff", "elo_x_rdshrink",
    "age_diff", "age_x_bo5", "age_x_close", "age_x_load7",
    "h2h_k3", "h2h_k5", "h2h_k10",
    "hold_rec_diff", "break_rec_diff",
    "hold_resid_diff", "break_resid_diff",
    "age_x_hold_diff", "dominance_diff", "elo_x_highrd",
    "winpct_diff", "surf_winpct_diff",
    "exp_diff", "surf_exp_diff",
    "logrank_diff", "logpoints_diff",
    "venue_winpct_diff", "home_diff",
    # Round 4 (2026-11-18): point-level serve, momentum, clutch,
    # in-event games, retirement flags, Elo x best-of-5. Appended so
    # earlier column indices stay stable; older runners pin their
    # own N_COLS (see run_logreg_all.py).
    "first_win_diff", "second_win_diff", "ace_rate_diff",
    "df_rate_diff", "bp_conv_diff", "bp_save_diff",
    "momentum_diff",
    "tb_winpct_diff", "decset_winpct_diff", "comeback_diff",
    "event_games_diff",
    "ret_self_diff", "ret_opp_diff",
    "elo_x_bo5",
    # Home x Elo interactions (2026-11-18): crowd support should
    # matter most when the home player is the underdog. Appended so
    # earlier column indices stay stable.
    "home_x_elo", "home_x_underdog", "home_x_fav",
]
ELO_ONLY_NAMES = ["elo_diff"]
COL = {name: i for i, name in enumerate(FEATURE_NAMES)}


def load_matches_with_serve(conn, end_year=2024):
    """All matches to end_year, in walk order, with serve columns,
    score, minutes and best_of (needed by the fatigue / WElo /
    Glicko / age states)."""
    cur = conn.cursor()
    cur.execute(
        "SELECT tourney_id, match_num, tourney_date, tourney_name, surface, "
        "round, tourney_level, winner_id, loser_id, winner_name, loser_name, "
        "w_sv_gms, w_bp_faced, w_bp_saved, l_sv_gms, l_bp_faced, l_bp_saved, "
        "w_svpt, w_1st_in, w_1st_won, w_2nd_won, w_ace, w_df, "
        "l_svpt, l_1st_in, l_1st_won, l_2nd_won, l_ace, l_df, "
        "score, minutes, best_of "
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


def load_dob(conn):
    """player_id -> dob date (None where unknown)."""
    cur = conn.cursor()
    cur.execute("SELECT player_id, dob FROM atp_players")
    out = {}
    for pid, dob in cur.fetchall():
        out[pid] = _dt.date.fromisoformat(str(dob)) if dob else None
    return out


def load_year_end_rankings(conn):
    """(player_id, year) -> (ranking, points) from atp_year_end_rankings.

    This is the ONLY dated ranking data in the database. The
    atp_player_ranking_summary table has one row per player with the
    latest ranking only, which would leak future information and is
    deliberately NOT used. A match in year Y uses the year-end
    ranking of year Y-1 (ranking_date is late December of that year,
    so it is strictly pre-match); the cost is staleness of up to
    ~12 months for late-season matches. Points are present for all
    rows from 2020 on; older years have NULL points.
    """
    cur = conn.cursor()
    cur.execute("SELECT player_id, year, ranking, points "
                "FROM atp_year_end_rankings")
    return {(pid, y): (r, p) for pid, y, r, p in cur.fetchall()}


def load_ioc(conn):
    """player_id -> IOC nationality code (None where unknown).

    One listed nationality per player in atp_players (the player's
    current/listed nationality, not necessarily the one held at
    match time -- a known limitation of the home flag).
    """
    cur = conn.cursor()
    cur.execute("SELECT player_id, ioc FROM atp_players")
    return {pid: ioc for pid, ioc in cur.fetchall() if ioc}


def tour_mean_minutes(matches, start_year=2001, end_year=2022):
    """Mean recorded minutes over train-period matches (train only,
    so it carries no validation information). Used as the fallback
    for players with no previous timed match."""
    vals = [m["minutes"] for m in matches
            if start_year <= m["tourney_date"].year <= end_year
            and m["minutes"] is not None and m["minutes"] > 0]
    return sum(vals) / len(vals) if vals else 100.0


class Standardizer:
    """Mean/std from train only; frozen for validation."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def fit_logreg(X_train, y_train, C, l1_ratio):
    """Fit the container. No intercept, by construction."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        C=C, l1_ratio=l1_ratio, solver="saga",
        fit_intercept=False, max_iter=5000, tol=1e-6, random_state=0,
    )
    clf.fit(X_train, y_train)
    if clf.intercept_.size and float(np.abs(clf.intercept_).max()) != 0.0:
        raise AssertionError("container must have zero intercept")
    return clf


def predict_proba(clf, X):
    """P(A wins) for standardized feature rows. No intercept anywhere."""
    return clf.predict_proba(X)[:, 1]
