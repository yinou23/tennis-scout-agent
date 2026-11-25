"""Walk-forward serve/return state for the backtest.

Definitions (per player, from matches already processed only -- a
match's features never use that match or any later match):
  hold rate  = service games won / service games played
  break rate = return games won / return games played

Service games won are derived from the Sackmann serve columns, not
stored directly:
  server games won = sv_gms - (bp_faced - bp_saved)   (breaks conceded)
  return games won = opponent's bp_faced - bp_saved   (breaks converted)
  return games played = opponent's sv_gms

So for a match row, the winner's hold uses the w_* columns and the
winner's break uses the l_* columns, and vice versa for the loser.

Shrinkage (same idea as the surface-Elo prior-100 blend):
  overall rate = (won + K * tour_prior) / (played + K)
  surface rate = (won_surface + K * overall_rate) / (played_surface + K)
with K = 50 games (SHRINK_K_GAMES). The surface rate is shrunk toward
the player's own overall rate, which is itself shrunk toward the tour
average; a player with no data at all gets the tour prior for the
surface, so the A-minus-B difference for that component is 0.

Tour priors are per-surface hold rates computed once from train-period
data only (2001-2022, both players' service games, unbiased):
  Hard 0.7909, Clay 0.7464, Grass 0.8251, Carpet 0.8141, overall 0.7819.
Break prior for a surface is 1 - hold prior. These are constants, not
fitted per match, so they carry no lookahead into validation.

Missing data: matches with NULL serve columns (pre-2001 coverage is
~26%, 2001+ ~91%) simply do not update the state; a player with no
serve history on the surface falls back to the tour prior as above.
Surface None (rare) falls back to the player's overall rate.

Features enter the regression as A-minus-B differences only
(see logreg.py), so the model is symmetric under swapping A and B.
"""

import datetime as _dt
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elo import INITIAL_ELO  # noqa: E402
from surface_elo import blended_elo, win_prob  # noqa: E402

# Tour-average hold rates, train period 2001-2022 (see module docstring).
TOUR_HOLD_PRIOR = {
    "Hard": 0.7909,
    "Clay": 0.7464,
    "Grass": 0.8251,
    "Carpet": 0.8141,
}
TOUR_HOLD_OVERALL = 0.7819

# Games of shrinkage. Tunable on validation; 50 is the starting value.
SHRINK_K_GAMES = 50


def _clamp_won(won, played):
    if won < 0:
        return 0
    if won > played:
        return played
    return won


class ServeState:
    """Per-player walk-forward serve/return game counts.

    Internal layout per player: {"overall": [svc_won, svc_played,
    ret_won, ret_played], "by_surface": {surface: [same four]}}.
    Only matches with complete serve columns update the state.
    """

    def __init__(self, shrink_k=SHRINK_K_GAMES):
        self.shrink_k = shrink_k
        self._players = {}

    def _entry(self, player_id):
        entry = self._players.get(player_id)
        if entry is None:
            entry = {"overall": [0, 0, 0, 0], "by_surface": {}}
            self._players[player_id] = entry
        return entry

    @staticmethod
    def _add(counts, svc_won, svc_played, ret_won, ret_played):
        counts[0] += svc_won
        counts[1] += svc_played
        counts[2] += ret_won
        counts[3] += ret_played

    def update(self, winner_id, loser_id, surface,
               w_sv_gms, w_bp_faced, w_bp_saved,
               l_sv_gms, l_bp_faced, l_bp_saved):
        """Apply one match result. Call only AFTER features were read.

        Any None stat means the match carries no serve data: the state
        is left untouched. All eight stats must be present.
        """
        stats = (w_sv_gms, w_bp_faced, w_bp_saved,
                 l_sv_gms, l_bp_faced, l_bp_saved)
        if any(s is None for s in stats):
            return
        if w_sv_gms <= 0 or l_sv_gms <= 0:
            return
        # Winner: holds from w_* columns, breaks from l_* columns.
        w_svc_won = _clamp_won(w_sv_gms - (w_bp_faced - w_bp_saved), w_sv_gms)
        w_ret_won = _clamp_won(l_bp_faced - l_bp_saved, l_sv_gms)
        # Loser: mirror image.
        l_svc_won = _clamp_won(l_sv_gms - (l_bp_faced - l_bp_saved), l_sv_gms)
        l_ret_won = _clamp_won(w_bp_faced - w_bp_saved, w_sv_gms)
        for pid, svc_won, svc_played, ret_won, ret_played in (
            (winner_id, w_svc_won, w_sv_gms, w_ret_won, l_sv_gms),
            (loser_id, l_svc_won, l_sv_gms, l_ret_won, w_sv_gms),
        ):
            entry = self._entry(pid)
            self._add(entry["overall"], svc_won, svc_played, ret_won, ret_played)
            if surface is not None:
                surf = entry["by_surface"].setdefault(surface, [0, 0, 0, 0])
                self._add(surf, svc_won, svc_played, ret_won, ret_played)

    def _shrunk(self, won, played, prior):
        return (won + self.shrink_k * prior) / (played + self.shrink_k)

    def _overall_rates(self, player_id):
        """(hold, break) shrunk toward the tour overall prior."""
        entry = self._players.get(player_id)
        if entry is None:
            return TOUR_HOLD_OVERALL, 1.0 - TOUR_HOLD_OVERALL
        svc_won, svc_played, ret_won, ret_played = entry["overall"]
        hold = self._shrunk(svc_won, svc_played, TOUR_HOLD_OVERALL)
        brk = self._shrunk(ret_won, ret_played, 1.0 - TOUR_HOLD_OVERALL)
        return hold, brk

    def rates(self, player_id, surface):
        """(hold, break) for one player on a surface, pre-match state.

        Surface rate shrunk toward the player's own overall rate; a
        player with no history gets the tour prior for the surface.
        Surface None falls back to the overall rate.
        """
        hold_overall, brk_overall = self._overall_rates(player_id)
        if surface is None:
            return hold_overall, brk_overall
        entry = self._players.get(player_id)
        if entry is None:
            prior = TOUR_HOLD_PRIOR.get(surface, TOUR_HOLD_OVERALL)
            return prior, 1.0 - prior
        surf = entry["by_surface"].get(surface)
        if surf is None:
            return hold_overall, brk_overall
        svc_won, svc_played, ret_won, ret_played = surf
        hold = self._shrunk(svc_won, svc_played, hold_overall)
        brk = self._shrunk(ret_won, ret_played, brk_overall)
        return hold, brk

    def diffs(self, player_a, player_b, surface):
        """(hold_a - hold_b, break_a - break_b) from pre-match state."""
        ha, ba = self.rates(player_a, surface)
        hb, bb = self.rates(player_b, surface)
        return ha - hb, ba - bb

    def n_players(self):
        return len(self._players)


# ---------------------------------------------------------------------------
# Score parsing (shared by fatigue, WElo and Glicko states)
# ---------------------------------------------------------------------------

def parse_score(score):
    """Parse a Sackmann score string.

    Returns (winner_games, loser_games, sets, retired, walkover).
    The score is from the winner's perspective: the first number in
    each set is the winner's games. Tiebreak points in parentheses
    are ignored (7-6 counts 13 games). A bracketed set "[10-7]" is a
    match tiebreak (58 matches in 2001-2024); points are not games,
    so it counts as a 7-6 tiebreak set (7 winner games, 6 loser
    games), the least distorting one-set proxy. "W/O", "Walkover" and
    bare "RET" mean no games were played. A retirement mid-match
    ("6-2 0-0 RET") keeps the games actually played.
    """
    if score is None:
        return 0, 0, 0, False, True
    s = str(score).strip()
    if s in ("", "W/O", "Walkover", "RET"):
        return 0, 0, 0, False, True
    retired = "RET" in s
    w_games = 0
    l_games = 0
    sets = 0
    for token in s.split():
        if token == "RET":
            continue
        if token.startswith("[") and token.endswith("]"):
            sets += 1
            w_games += 7
            l_games += 6
            continue
        core = token.split("(")[0]
        if "-" not in core:
            continue
        try:
            a, b = core.split("-", 1)
            ga, gb = int(a), int(b)
        except ValueError:
            continue
        w_games += ga
        l_games += gb
        sets += 1
    if sets == 0:
        return 0, 0, 0, retired, True
    return w_games, l_games, sets, retired, False


# ---------------------------------------------------------------------------
# Form: recent-2-month and last-10 win%, raw and residual-vs-Elo
# ---------------------------------------------------------------------------

FORM_2M_DAYS = 60
FORM_2M_SHRINK_K = 5.0   # matches; shrink toward 0.5 (raw) or 0 (residual)
FORM_10_SHRINK_K = 2.0


class FormState:
    """Walk-forward per-player form from matches already processed.

    Each entry is (date, won, p_expected) where p_expected is the
    locked model's calibrated probability that this player would win
    that match, computed from the pre-match Elo state. Raw form is
    the shrunk win%; residual form is the shrunk mean of
    (won - p_expected), i.e. how much the player beat Elo's
    expectation lately. Residual form is the cleaner signal because
    raw form overlaps Elo (a strong player wins a lot lately simply
    by being strong); both are computed so the runner can test them
    as separate batches.
    """

    def __init__(self):
        self._players = {}

    def _entries(self, player_id):
        return self._players.setdefault(player_id, [])

    def update(self, winner_id, loser_id, date, p_winner):
        """Apply one match result. Call only AFTER features were read.

        p_winner is the locked model's pre-match probability that the
        eventual winner wins (calibrated). The loser entries get
        p_expected = 1 - p_winner.
        """
        self._entries(winner_id).append((date, 1, p_winner))
        self._entries(loser_id).append((date, 0, 1.0 - p_winner))

    @staticmethod
    def _window(entries, date, days):
        if days is None:
            return entries
        cutoff = date - _dt.timedelta(days=days)
        return [e for e in entries if e[0] > cutoff]

    def _raw(self, entries, shrink_k):
        n = len(entries)
        if n == 0:
            return 0.0
        w = sum(e[1] for e in entries)
        return (w + shrink_k * 0.5) / (n + shrink_k) - 0.5

    def _resid(self, entries, shrink_k):
        n = len(entries)
        if n == 0:
            return 0.0
        mean_resid = sum(e[1] - e[2] for e in entries) / n
        return mean_resid * n / (n + shrink_k)

    def form_2m(self, player_id, date):
        return self._raw(
            self._window(self._players.get(player_id, []), date, FORM_2M_DAYS),
            FORM_2M_SHRINK_K)

    def form_10(self, player_id, date):
        entries = self._players.get(player_id, [])[-10:]
        return self._raw(entries, FORM_10_SHRINK_K)

    def resid_2m(self, player_id, date):
        return self._resid(
            self._window(self._players.get(player_id, []), date, FORM_2M_DAYS),
            FORM_2M_SHRINK_K)

    def resid_10(self, player_id, date):
        entries = self._players.get(player_id, [])[-10:]
        return self._resid(entries, FORM_10_SHRINK_K)

    def form_5(self, player_id, date):
        entries = self._players.get(player_id, [])[-5:]
        return self._raw(entries, FORM_10_SHRINK_K)

    def form_20(self, player_id, date):
        entries = self._players.get(player_id, [])[-20:]
        return self._raw(entries, FORM_10_SHRINK_K)

    def momentum_diff(self, player_a, player_b, date):
        """(form5 - form20) of A minus the same for B.

        Trajectory, not level: a player at 60% over the last 5
        after 45% over the last 20 is rising; the level versions
        (form raw / residual) were both dropped (z=-0.33 / -0.89),
        so only the slope is tested here. Raw win% is used
        (centred at 0.5, which cancels in the slope); no Elo
        expectation enters.
        """
        mom_a = self.form_5(player_a, date) - self.form_20(player_a, date)
        mom_b = self.form_5(player_b, date) - self.form_20(player_b, date)
        return mom_a - mom_b

    def diffs(self, player_a, player_b, date):
        """(form2m, form10, resid2m, resid10) A-minus-B diffs."""
        return (
            self.form_2m(player_a, date) - self.form_2m(player_b, date),
            self.form_10(player_a, date) - self.form_10(player_b, date),
            self.resid_2m(player_a, date) - self.resid_2m(player_b, date),
            self.resid_10(player_a, date) - self.resid_10(player_b, date),
        )


# ---------------------------------------------------------------------------
# Fatigue: rest, recent load, previous-match minutes
# ---------------------------------------------------------------------------

FATIGUE_REST_CAP_DAYS = 60
FATIGUE_REST_NEUTRAL_DAYS = 7


class FatigueState:
    """Walk-forward per-player workload from matches already processed.

    rest_days: days since the player's last match with games > 0
    (walkovers do not count as playing), capped at 60, 7 if no
    history (a typical in-season gap). games7/games14: total games
    played in matches dated within the last 7/14 days before the
    current date. prev_minutes: minutes of the most recent match with
    a recorded duration, falling back to the player's running mean
    and then to the train-period tour mean supplied at construction.
    All four enter the container as A-minus-B diffs.
    """

    def __init__(self, tour_mean_minutes=100.0):
        self.tour_mean_minutes = tour_mean_minutes
        self._players = {}

    def _entry(self, player_id):
        return self._players.setdefault(player_id, {
            "last_date": None, "games": [], "last_minutes": None,
            "minutes_sum": 0.0, "minutes_n": 0,
        })

    def update(self, winner_id, loser_id, date, total_games, minutes):
        for pid in (winner_id, loser_id):
            entry = self._entry(pid)
            if total_games > 0:
                entry["last_date"] = date
                entry["games"].append((date, total_games))
            if minutes is not None and minutes > 0:
                entry["last_minutes"] = minutes
                entry["minutes_sum"] += minutes
                entry["minutes_n"] += 1

    def _rest(self, player_id, date):
        entry = self._players.get(player_id)
        if entry is None or entry["last_date"] is None:
            return float(FATIGUE_REST_NEUTRAL_DAYS)
        days = (date - entry["last_date"]).days
        return float(min(max(days, 0), FATIGUE_REST_CAP_DAYS))

    def _games_in(self, player_id, date, days):
        entry = self._players.get(player_id)
        if entry is None:
            return 0.0
        cutoff = date - _dt.timedelta(days=days)
        return float(sum(g for d, g in entry["games"]
                         if cutoff <= d < date))

    def _prev_minutes(self, player_id):
        entry = self._players.get(player_id)
        if entry is None:
            return self.tour_mean_minutes
        if entry["last_minutes"] is not None:
            return float(entry["last_minutes"])
        if entry["minutes_n"] > 0:
            return entry["minutes_sum"] / entry["minutes_n"]
        return self.tour_mean_minutes

    def diffs(self, player_a, player_b, date):
        """(rest, games7, games14, prev_minutes) A-minus-B diffs."""
        return (
            self._rest(player_a, date) - self._rest(player_b, date),
            self._games_in(player_a, date, 7) - self._games_in(player_b, date, 7),
            self._games_in(player_a, date, 14) - self._games_in(player_b, date, 14),
            self._prev_minutes(player_a) - self._prev_minutes(player_b),
        )

    def games7_diff(self, player_a, player_b, date):
        return (self._games_in(player_a, date, 7)
                - self._games_in(player_b, date, 7))


# ---------------------------------------------------------------------------
# WElo: games-weighted Elo, run in parallel with the plain Elo state
# ---------------------------------------------------------------------------

class WEloState:
    """Parallel Elo whose update is weighted by the winner's share of
    games, f = winner_games / total_games (Angelini, Candila &
    De Angelis, EJOR 297(1), 2022). Same K=32 and same surface-blend
    prior-100 formula as EloState, so the WElo diff is directly
    comparable with the Elo diff; it is a feature alongside Elo, never
    a replacement. Retirements are skipped (neutral handling, as in
    Pickledger: a retired match carries no reliable games margin),
    as are walkovers and matches with no parseable games. The plain
    Elo state still updates on retirements, exactly as production
    does; only this WElo state skips them.
    """

    def __init__(self, k=32.0, prior=100):
        self.k = k
        self.prior = prior
        self.overall = {}
        self.surface_rating = {}
        self.surface_n = {}

    def _overall(self, player_id):
        return self.overall.get(player_id, INITIAL_ELO)

    def _surface(self, player_id, surface):
        if surface is None:
            return None, 0
        key = (player_id, surface)
        return self.surface_rating.get(key, INITIAL_ELO), self.surface_n.get(key, 0)

    def blended(self, player_id, surface):
        overall = self._overall(player_id)
        if surface is None:
            return overall
        surf, n = self._surface(player_id, surface)
        return blended_elo(surf, overall, n, self.prior)

    def diff(self, player_a, player_b, surface):
        return self.blended(player_a, surface) - self.blended(player_b, surface)

    def update(self, winner_id, loser_id, surface, winner_games, total_games,
               retired=False, walkover=False):
        """Apply one match result. Call only AFTER features were read."""
        if walkover or retired or total_games <= 0:
            return
        f = winner_games / total_games
        self._update_pair(winner_id, loser_id, None, f)
        if surface is not None:
            self._update_pair(winner_id, loser_id, surface, f)

    def _update_pair(self, winner_id, loser_id, surface, f):
        if surface is None:
            w_rating = self._overall(winner_id)
            l_rating = self._overall(loser_id)
        else:
            w_rating, _ = self._surface(winner_id, surface)
            l_rating, _ = self._surface(loser_id, surface)
        expected = win_prob(w_rating, l_rating)
        delta = self.k * f * (1.0 - expected)
        if surface is None:
            self.overall[winner_id] = w_rating + delta
            self.overall[loser_id] = l_rating - delta
        else:
            self.surface_rating[(winner_id, surface)] = w_rating + delta
            self.surface_rating[(loser_id, surface)] = l_rating - delta
            self.surface_n[(winner_id, surface)] = \
                self.surface_n.get((winner_id, surface), 0) + 1
            self.surface_n[(loser_id, surface)] = \
                self.surface_n.get((loser_id, surface), 0) + 1


# ---------------------------------------------------------------------------
# Head-to-head: shrunk win% with a minimum-meetings floor
# ---------------------------------------------------------------------------

class H2HState:
    """Prior meetings between a pair, walk-forward.

    Keyed by (lower_id, higher_id) -> [wins by lower-id player,
    total meetings]. Feature for A (the lower-id player) is the
    shrunk win% minus 0.5, (wins + k*0.5) / (total + k) - 0.5, and
    0 when total < 3 meetings: with fewer meetings the number is
    noise, and the floor keeps the feature exactly neutral there.
    The same value negated is B's feature, so the diff is symmetric.
    """

    MIN_MEETINGS = 3

    def __init__(self):
        self._pairs = {}

    def update(self, winner_id, loser_id):
        lo, hi = (winner_id, loser_id) if winner_id < loser_id else (loser_id, winner_id)
        entry = self._pairs.setdefault((lo, hi), [0, 0])
        if winner_id == lo:
            entry[0] += 1
        entry[1] += 1

    def _feat(self, player_a, player_b, k):
        lo, hi = (player_a, player_b) if player_a < player_b else (player_b, player_a)
        entry = self._pairs.get((lo, hi))
        if entry is None or entry[1] < self.MIN_MEETINGS:
            return 0.0
        wins_lo, total = entry
        wins_a = wins_lo if player_a == lo else total - wins_lo
        return (wins_a + k * 0.5) / (total + k) - 0.5

    def diffs(self, player_a, player_b):
        """(h2h_k3, h2h_k5, h2h_k10) A-minus-B diffs.

        Each is 2x the shrunk win% deviation so the scale is roughly
        comparable with a win% difference: for A the feature is
        2 * (shrunk win% - 0.5), for B its negation.
        """
        return tuple(2.0 * self._feat(player_a, player_b, k)
                     for k in (3, 5, 10))


# ---------------------------------------------------------------------------
# Age: years at match date, from atp_players dob
# ---------------------------------------------------------------------------

def age_years(dob, date):
    """Age in years at `date`, or None if dob is missing."""
    if dob is None:
        return None
    if isinstance(dob, str):
        dob = _dt.date.fromisoformat(dob)
    return (date - dob).days / 365.25


# ---------------------------------------------------------------------------
# Recent-window serve: hold/break over the last 60 days only
# ---------------------------------------------------------------------------

RECENT_SERVE_DAYS = 60
RECENT_SERVE_SHRINK_K = 20  # games; fewer than the career state's 50,
# because a 60-day window holds far fewer games per player


class RecentServeState:
    """Per-player hold/break from matches in the last 60 days only.

    Same serve-column derivation as ServeState, but entries expire:
    only matches dated within RECENT_SERVE_DAYS before the current
    match count. The hypothesis is current serving form, not career
    skill: a player whose serve has recently improved (or declined)
    shows it here before the career rate, and the Elo rating, catch
    up. Shrinkage mirrors ServeState (surface rate toward the
    player's own recent overall rate, overall toward the tour prior)
    with a smaller K because the window is thin. A player with no
    match in the window gets the tour prior, so the diff is 0.
    """

    def __init__(self, shrink_k=RECENT_SERVE_SHRINK_K,
                 window_days=RECENT_SERVE_DAYS):
        self.shrink_k = shrink_k
        self.window_days = window_days
        self._players = {}

    def _entries(self, player_id):
        return self._players.setdefault(player_id, [])

    def update(self, winner_id, loser_id, surface, date,
               w_sv_gms, w_bp_faced, w_bp_saved,
               l_sv_gms, l_bp_faced, l_bp_saved):
        """Apply one match result. Call only AFTER features were read."""
        stats = (w_sv_gms, w_bp_faced, w_bp_saved,
                 l_sv_gms, l_bp_faced, l_bp_saved)
        if any(s is None for s in stats):
            return
        if w_sv_gms <= 0 or l_sv_gms <= 0:
            return
        w_svc_won = _clamp_won(w_sv_gms - (w_bp_faced - w_bp_saved), w_sv_gms)
        w_ret_won = _clamp_won(l_bp_faced - l_bp_saved, l_sv_gms)
        l_svc_won = _clamp_won(l_sv_gms - (l_bp_faced - l_bp_saved), l_sv_gms)
        l_ret_won = _clamp_won(w_bp_faced - w_bp_saved, w_sv_gms)
        for pid, svc_won, svc_played, ret_won, ret_played in (
            (winner_id, w_svc_won, w_sv_gms, w_ret_won, l_sv_gms),
            (loser_id, l_svc_won, l_sv_gms, l_ret_won, w_sv_gms),
        ):
            self._entries(pid).append(
                (date, surface, svc_won, svc_played, ret_won, ret_played))

    def _window(self, player_id, date):
        cutoff = date - _dt.timedelta(days=self.window_days)
        return [e for e in self._players.get(player_id, [])
                if e[0] > cutoff]

    def _shrunk(self, won, played, prior):
        return (won + self.shrink_k * prior) / (played + self.shrink_k)

    def rates(self, player_id, surface, date):
        """(hold, break) over the last 60 days, pre-match state."""
        entries = self._window(player_id, date)
        if not entries:
            prior = TOUR_HOLD_PRIOR.get(surface, TOUR_HOLD_OVERALL) \
                if surface is not None else TOUR_HOLD_OVERALL
            return prior, 1.0 - prior
        svc_won = sum(e[2] for e in entries)
        svc_played = sum(e[3] for e in entries)
        ret_won = sum(e[4] for e in entries)
        ret_played = sum(e[5] for e in entries)
        hold_overall = self._shrunk(svc_won, svc_played, TOUR_HOLD_OVERALL)
        brk_overall = self._shrunk(ret_won, ret_played, 1.0 - TOUR_HOLD_OVERALL)
        if surface is None:
            return hold_overall, brk_overall
        surf = [e for e in entries if e[1] == surface]
        if not surf:
            return hold_overall, brk_overall
        hold = self._shrunk(sum(e[2] for e in surf),
                            sum(e[3] for e in surf), hold_overall)
        brk = self._shrunk(sum(e[4] for e in surf),
                           sum(e[5] for e in surf), brk_overall)
        return hold, brk

    def diffs(self, player_a, player_b, surface, date):
        """(hold_a - hold_b, break_a - break_b) from pre-match state."""
        ha, ba = self.rates(player_a, surface, date)
        hb, bb = self.rates(player_b, surface, date)
        return ha - hb, ba - bb


# ---------------------------------------------------------------------------
# Serve residual: actual hold/break minus what the Elo implies
# ---------------------------------------------------------------------------

def fit_rate_vs_elo(snapshots):
    """OLS fit of a per-player rate on overall Elo, train data only.

    snapshots: list of (elo, hold, break) per player, one row each,
    taken at the end of the train period. Returns
    {"hold": (intercept, slope), "break": (intercept, slope)}.
    The mapping is frozen before validation is ever scored, so it
    carries no lookahead. A player's residual is then
    actual_shrunk_rate - expected_rate(elo): serving skill above or
    below what a player of that rating usually serves. By
    construction the residual is orthogonal (in the linear sense) to
    the Elo level, which is exactly the overlap the career hold%
    suffered from.
    """
    n = len(snapshots)
    if n < 10:
        raise ValueError("need at least 10 player snapshots to fit")
    mean_elo = sum(s[0] for s in snapshots) / n
    out = {}
    for idx, name in ((1, "hold"), (2, "break")):
        mean_rate = sum(s[idx] for s in snapshots) / n
        num = sum((s[0] - mean_elo) * (s[idx] - mean_rate) for s in snapshots)
        den = sum((s[0] - mean_elo) ** 2 for s in snapshots)
        slope = num / den if den > 0 else 0.0
        out[name] = (mean_rate - slope * mean_elo, slope)
    return out


def serve_residual(actual_rate, elo, coef):
    """actual rate minus the Elo-implied rate, clamped to [-0.5, 0.5].

    coef is (intercept, slope) from fit_rate_vs_elo. The implied rate
    is clamped to [0.05, 0.95] so extrapolation to extreme Elos
    cannot produce an absurd expectation.
    """
    intercept, slope = coef
    implied = intercept + slope * elo
    implied = min(max(implied, 0.05), 0.95)
    return min(max(actual_rate - implied, -0.5), 0.5)


# ---------------------------------------------------------------------------
# Dominance: how convincingly a player wins, not just that they win
# ---------------------------------------------------------------------------

DOMINANCE_LAST_WINS = 10
DOMINANCE_SHRINK_K = 2.0  # wins; shrink toward a 0.5 games-share


class DominanceState:
    """Average games-share in a player's recent wins.

    Each completed win contributes winner_games / total_games
    (6-0 6-0 is 1.0, 7-6 7-6 is ~0.54). The feature is the mean over
    the player's last 10 wins, shrunk toward 0.5 and centred, so a
    player with no recent wins scores 0. Retirements and walkovers
    are excluded: their games-share is not a dominance signal. This
    tests a different hypothesis from WElo (dropped, z=-0.82): not a
    second rating, but whether the manner of recent wins carries
    information beyond the win/loss Elo already absorbed.
    """

    def __init__(self, last_wins=DOMINANCE_LAST_WINS,
                 shrink_k=DOMINANCE_SHRINK_K):
        self.last_wins = last_wins
        self.shrink_k = shrink_k
        self._players = {}

    def update(self, winner_id, loser_id, winner_games, total_games,
               retired=False, walkover=False):
        """Apply one match result. Call only AFTER features were read."""
        if walkover or retired or total_games <= 0:
            return
        self._players.setdefault(winner_id, []).append(
            winner_games / total_games)

    def _feat(self, player_id):
        entries = self._players.get(player_id, [])[-self.last_wins:]
        n = len(entries)
        if n == 0:
            return 0.0
        return (sum(entries) + self.shrink_k * 0.5) / (n + self.shrink_k) - 0.5

    def diff(self, player_a, player_b):
        """dominance_a - dominance_b, symmetric by construction."""
        return self._feat(player_a) - self._feat(player_b)


# ---------------------------------------------------------------------------
# Glicko gate: trust Elo less when either player is low-information
# ---------------------------------------------------------------------------

def elo_high_rd_interaction(elo_diff, rd_a, rd_b, threshold):
    """elo_diff when either player is low-information, else 0.

    The gate is max(rd_a, rd_b) >= threshold (threshold = median
    max-RD over train scored matches). The model already has the
    plain elo_diff column, so this one extra column gives it a
    separate Elo slope for low-information matches: if uncertainty
    should dampen Elo, the fitted weight on this column is negative
    and partly cancels the plain Elo slope there. Either player
    being uncertain makes the diff unreliable, hence max, not min.
    A linear RD diff (dropped, z=-0.66) cannot express this: RD
    matters only when it is large, i.e. as a gate, not as a slope.
    """
    if max(rd_a, rd_b) >= threshold:
        return elo_diff
    return 0.0


# ---------------------------------------------------------------------------
# Win rate and experience: career / surface win%, match counts
# ---------------------------------------------------------------------------

WINRATE_SHRINK_K = 10  # matches; shrink toward 0.5 (overall) or the
# player's own overall rate (surface)


class WinRateState:
    """Walk-forward per-player win rates and match counts.

    Overall win% is shrunk toward 0.5, surface win% toward the
    player's own overall rate, both with K = 10 matches. Experience
    is log1p of matches played (career and surface): counts are
    large and skewed, so the log keeps a 500-match veteran from
    dominating the scale. Every entry is a difference, so a player
    with no history scores the prior (win% diff 0, experience diff
    from 0 matches vs the opponent's count, which is the honest
    signal: an unknown player really is inexperienced).
    """

    def __init__(self, shrink_k=WINRATE_SHRINK_K):
        self.shrink_k = shrink_k
        self._players = {}

    def _entry(self, player_id):
        return self._players.setdefault(
            player_id, {"overall": [0, 0], "by_surface": {}})

    def update(self, winner_id, loser_id, surface):
        """Apply one match result. Call only AFTER features were read."""
        for pid, won in ((winner_id, 1), (loser_id, 0)):
            entry = self._entry(pid)
            entry["overall"][0] += won
            entry["overall"][1] += 1
            if surface is not None:
                surf = entry["by_surface"].setdefault(surface, [0, 0])
                surf[0] += won
                surf[1] += 1

    def _overall_rate(self, player_id):
        entry = self._players.get(player_id)
        if entry is None:
            return 0.0  # centred: no history -> 0
        w, n = entry["overall"]
        return (w + self.shrink_k * 0.5) / (n + self.shrink_k) - 0.5

    def _surface_rate(self, player_id, surface):
        overall = self._overall_rate(player_id)
        if surface is None:
            return overall
        entry = self._players.get(player_id)
        if entry is None:
            return 0.0
        surf = entry["by_surface"].get(surface)
        if surf is None:
            return overall
        w, n = surf
        # Shrink toward the player's own overall (uncentred) rate.
        overall_uncentred = overall + 0.5
        return (w + self.shrink_k * overall_uncentred) / (n + self.shrink_k) - 0.5

    def _counts(self, player_id, surface):
        entry = self._players.get(player_id)
        if entry is None:
            return 0, 0
        overall_n = entry["overall"][1]
        surf_n = 0
        if surface is not None:
            surf = entry["by_surface"].get(surface)
            surf_n = surf[1] if surf is not None else 0
        return overall_n, surf_n

    def diffs(self, player_a, player_b, surface):
        """(winpct, surface winpct, log1p experience, log1p surface
        experience) A-minus-B diffs, all centred/symmetric."""
        return (
            self._overall_rate(player_a) - self._overall_rate(player_b),
            self._surface_rate(player_a, surface)
            - self._surface_rate(player_b, surface),
            math.log1p(self._counts(player_a, surface)[0])
            - math.log1p(self._counts(player_b, surface)[0]),
            math.log1p(self._counts(player_a, surface)[1])
            - math.log1p(self._counts(player_b, surface)[1]),
        )


# ---------------------------------------------------------------------------
# Venue win rate: how a player does at THIS tournament
# ---------------------------------------------------------------------------

VENUE_SHRINK_K = 5  # matches; shrink toward the player's overall rate


class VenueState:
    """Walk-forward win rate at a specific tournament (tourney_id).

    The feature is the venue-specific EXCESS: the player's win rate
    at this tournament, shrunk toward their own overall win rate,
    minus that overall rate. A player with no prior match at the
    venue scores exactly 0, so the column is silent where there is
    no information instead of restating the career win% (which the
    win-rate batch already measures). Sparse by construction: most
    (player, venue) pairs have 0-2 prior meetings.
    """

    def __init__(self, shrink_k=VENUE_SHRINK_K):
        self.shrink_k = shrink_k
        self._venues = {}  # (player_id, tourney_id) -> [wins, total]
        self._overall = {}  # player_id -> [wins, total]

    def update(self, winner_id, loser_id, tourney_id):
        """Apply one match result. Call only AFTER features were read."""
        for pid, won in ((winner_id, 1), (loser_id, 0)):
            entry = self._venues.setdefault((pid, tourney_id), [0, 0])
            entry[0] += won
            entry[1] += 1
            overall = self._overall.setdefault(pid, [0, 0])
            overall[0] += won
            overall[1] += 1

    def _overall_rate(self, player_id):
        entry = self._overall.get(player_id)
        if entry is None or entry[1] == 0:
            return 0.5
        return entry[0] / entry[1]

    def _excess(self, player_id, tourney_id):
        entry = self._venues.get((player_id, tourney_id))
        if entry is None or entry[1] == 0:
            return 0.0
        overall = self._overall_rate(player_id)
        w, n = entry
        venue_rate = (w + self.shrink_k * overall) / (n + self.shrink_k)
        return venue_rate - overall

    def diff(self, player_a, player_b, tourney_id):
        """venue excess of A minus venue excess of B."""
        return (self._excess(player_a, tourney_id)
                - self._excess(player_b, tourney_id))

    def n_prior(self, player_id, tourney_id):
        """Prior matches at the venue (for coverage reporting)."""
        entry = self._venues.get((player_id, tourney_id))
        return entry[1] if entry is not None else 0


# ---------------------------------------------------------------------------
# Tournament country (for the home flag)
# ---------------------------------------------------------------------------

# IOC country codes, keyed by Sackmann tourney_name, for the 149
# distinct non-Davis-Cup, non-Olympic tournament names that appear in
# 2001-2024 (the scored window). Every entry is a city the author
# could place from the name itself; the five moving events (Tour
# Finals / Masters Cup, Laver Cup, NextGen Finals, United Cup,
# Atp Cup) are handled by year/rules in tournament_country() below,
# not here. Anything not listed returns None (neutral). Obscure
# pre-2001 one-off events are deliberately left out rather than
# guessed.
TOURNAMENT_COUNTRY = {
    "ATP Rio de Janeiro": "BRA", "Acapulco": "MEX", "Adelaide": "AUS",
    "Adelaide 1": "AUS", "Adelaide 2": "AUS", "Almaty": "KAZ",
    "Amersfoort": "NED", "Amsterdam": "NED", "Antalya": "TUR",
    "Antwerp": "BEL", "Astana": "KAZ", "Athens": "GRE",
    "Atlanta": "USA", "Auckland": "NZL", "Australian Open": "AUS",
    "Bangkok": "THA", "Banja Luka": "BIH", "Barcelona": "ESP",
    "Basel": "SUI", "Bastad": "SWE", "Beijing": "CHN",
    "Belgrade": "SRB", "Belgrade ": "SRB", "Belgrade 2": "SRB",
    "Bogota": "COL", "Brisbane": "AUS", "Brussels": "BEL",
    "Bucharest": "ROU", "Budapest": "HUN", "Buenos Aires": "ARG",
    "Cagliari": "ITA", "Canada Masters": "CAN", "Casablanca": "MAR",
    "Chengdu": "CHN", "Chennai": "IND", "Cincinnati Masters": "USA",
    "Cologne 1": "GER", "Cologne 2": "GER", "Copenhagen": "DEN",
    "Cordoba": "ARG", "Costa Do Sauipe": "BRA", "Dallas": "USA",
    "Delray Beach": "USA", "Doha": "QAT", "Dubai": "UAE",
    "Dusseldorf": "GER", "Eastbourne": "GBR", "Estoril": "POR",
    "Florence": "ITA", "Geneva": "SUI", "Gijon": "ESP",
    "Great Ocean Road Open": "AUS", "Gstaad": "SUI", "Halle": "GER",
    "Hamburg": "GER", "Hamburg Masters": "GER", "Hangzhou": "CHN",
    "Ho Chi Minh City": "VIE", "Hong Kong": "HKG", "Houston": "USA",
    "Indian Wells Masters": "USA", "Indianapolis": "USA",
    "Istanbul": "TUR", "Johannesburg": "RSA", "Kitzbuhel": "AUT",
    "Kuala Lumpur": "MAS", "Las Vegas": "USA", "Long Island": "USA",
    "Los Angeles": "USA", "Los Cabos": "MEX", "Lyon": "FRA",
    "Madrid Masters": "ESP", "Mallorca": "ESP", "Marbella": "ESP",
    "Marrakech": "MAR", "Marseille": "FRA", "Melbourne": "AUS",
    "Memphis": "USA", "Metz": "FRA", "Miami Masters": "USA",
    "Milan": "ITA", "Monte Carlo Masters": "MCO",
    "Montpellier": "FRA", "Moscow": "RUS", "Mumbai": "IND",
    "Munich": "GER", "Murray River Open": "AUS", "Naples": "ITA",
    "New Haven": "USA", "New York": "USA", "Newport": "USA",
    "Nice": "FRA", "Nottingham": "GBR", "Nur-Sultan": "KAZ",
    "Palermo": "ITA", "Paris Masters": "FRA", "Parma": "ITA",
    "Poertschach": "AUT", "Pune": "IND", "Queen's Club": "GBR",
    "Quito": "ECU", "Rio De Janeiro": "BRA", "Rio de Janeiro": "BRA",
    "Roland Garros": "FRA", "Rome Masters": "ITA",
    "Rotterdam": "NED", "San Diego": "USA", "San Jose": "USA",
    "Santiago": "CHI", "Sao Paulo": "BRA", "Sardinia": "ITA",
    "Scottsdale": "USA", "Seoul": "KOR", "Shanghai": "CHN",
    "Shanghai Masters": "CHN", "Shenzhen": "CHN", "Singapore": "SGP",
    "Sofia": "BUL", "Sopot": "POL", "St Petersburg": "RUS",
    "St. Petersburg": "RUS", "St. Poelten": "AUT",
    "Stockholm": "SWE", "Stuttgart": "GER",
    "Stuttgart Masters": "GER", "Sydney": "AUS", "Tashkent": "UZB",
    "Tel Aviv": "ISR", "Tokyo": "JPN", "US Open": "USA",
    "Us Open": "USA", "Umag": "CRO", "Valencia": "ESP",
    "Vienna": "AUT", "Vina del Mar": "CHI", "Warsaw": "POL",
    "Washington": "USA", "Wimbledon": "GBR", "Winston-Salem": "USA",
    "Zagreb": "CRO", "Zhuhai": "CHN", "s Hertogenbosch": "NED",
}

# City -> country for "<City> Olympics" names.
OLYMPIC_CITY_COUNTRY = {
    "Paris": "FRA", "Tokyo": "JPN", "London": "GBR", "Rio": "BRA",
    "Beijing": "CHN", "Athens": "GRE", "Sydney": "AUS",
    "Atlanta": "USA", "Barcelona": "ESP", "Seoul": "KOR",
    "Los Angeles": "USA",
}

# Year-based locations for events that moved. (start_year, country)
# in chronological order; the last entry whose start_year <= the
# match year wins. Tour Finals / Masters Cup: Turin from 2021,
# London 2009-2020, Shanghai 2005-2008 and 2002, Houston 2003-2004,
# Sydney 2001, Lisbon 2000; before 2000 the location is not mapped
# (None).
TOUR_FINALS_LOCATIONS = [
    (2000, "POR"), (2001, "AUS"), (2002, "CHN"), (2003, "USA"),
    (2005, "CHN"), (2009, "GBR"), (2021, "ITA"),
]
LAVER_CUP_LOCATIONS = [
    (2017, "CZE"), (2018, "USA"), (2019, "SUI"), (2021, "USA"),
    (2022, "GBR"), (2023, "CAN"), (2024, "GER"),
]
NEXTGEN_LOCATIONS = [(2017, "ITA"), (2023, "KSA")]


def _year_location(locations, year):
    country = None
    for start_year, code in locations:
        if year >= start_year:
            country = code
        else:
            break
    return country


def tournament_country(tourney_name, date):
    """IOC country code for a tournament, or None if unverifiable.

    Rules, in order:
    - Davis Cup ties ("Davis Cup ...: XXX vs YYY"): the host is the
      first-named team (Davis Cup lists the home team first).
    - "<City> Olympics": the city map above.
    - Tour Finals / Laver Cup / NextGen Finals: year-based tables.
    - United Cup / Atp Cup: Australia (both are played in Australia).
    - Otherwise the static TOURNAMENT_COUNTRY map by tourney name.
    `date` is a datetime.date (tourney_date). Unknown -> None.
    """
    name = (tourney_name or "").strip()
    if name.startswith("Davis Cup"):
        # "...: XXX vs YYY" -- host first. Ties without a colon
        # (e.g. "Davis Cup WG2 PO: ...") still carry the pair.
        if ":" in name:
            pair = name.rsplit(":", 1)[1].strip()
            if " vs " in pair:
                host = pair.split(" vs ", 1)[0].strip()
                if len(host) == 3 and host.isalpha():
                    return host.upper()
        return None
    if name.endswith("Olympics"):
        city = name[:-len("Olympics")].strip()
        return OLYMPIC_CITY_COUNTRY.get(city)
    year = date.year if date is not None else None
    if name in ("Tour Finals", "Masters Cup") and year is not None:
        return _year_location(TOUR_FINALS_LOCATIONS, year)
    if name == "Laver Cup" and year is not None:
        return _year_location(LAVER_CUP_LOCATIONS, year)
    if name in ("NextGen Finals", "Next Gen Finals") and year is not None:
        return _year_location(NEXTGEN_LOCATIONS, year)
    if name in ("United Cup", "Atp Cup"):
        return "AUS"
    return TOURNAMENT_COUNTRY.get(name)


# ---------------------------------------------------------------------------
# Point-level serve stats: 1st/2nd win%, ace/DF rates, BP conv/save
# ---------------------------------------------------------------------------

# Tour priors, train period 2001-2022 (both players' serve points).
# 1st win% 0.7116, 2nd win% 0.5059, ace 0.0758, DF 0.0376,
# BP conversion 0.3970, BP save 0.6030. Per-surface priors for the
# three stats that move most with surface; DF/BP use the overall
# prior on every surface (their surface variation is small).
POINT_SERVE_PRIOR = {
    "first": 0.7116, "second": 0.5059, "ace": 0.0758,
    "df": 0.0376, "bp_conv": 0.3970, "bp_save": 0.6030,
}
POINT_SERVE_SURFACE_PRIOR = {
    "Hard": {"first": 0.7212, "second": 0.5071, "ace": 0.0842},
    "Clay": {"first": 0.6823, "second": 0.4991, "ace": 0.0523},
    "Grass": {"first": 0.7393, "second": 0.5168, "ace": 0.0946},
    "Carpet": {"first": 0.7406, "second": 0.5122, "ace": 0.0961},
}
# Shrinkage in points/chances. First/second serves are plentiful
# (K=100); aces/DFs are rarer per match (K=200); break points are
# the scarcest (K=50 chances).
POINT_SERVE_SHRINK_K = {
    "first": 100, "second": 100, "ace": 200,
    "df": 200, "bp_conv": 50, "bp_save": 50,
}
_POINT_STATS = ("first", "second", "ace", "df", "bp_conv", "bp_save")


class PointServeState:
    """Walk-forward per-player point-level serve stats.

    Same shape as ServeState (surface rate shrunk toward the
    player's own overall rate, overall toward the tour prior), but
    on points rather than games: a decomposition of the serve the
    game-level hold% cannot see (a big first serve vs a steady
    second serve vs break-point nerve). The hypothesis is that the
    mix matters beyond the hold rate itself; the career hold%
    batch (dropped twice, z=-1.24 / -1.41) says the level does not.

    Per match, the winner's serving numbers come from the w_*
    columns and the loser's from the l_* columns; break-point
    conversion is a return stat, so the winner's conversion uses
    the l_* columns (opponent's serve) and vice versa. Any NULL
    serve column means the match carries no point data: the state
    is left untouched, the same all-or-nothing rule as ServeState.
    """

    def __init__(self):
        self._players = {}

    def _entry(self, player_id):
        entry = self._players.get(player_id)
        if entry is None:
            entry = {"overall": {s: [0, 0] for s in _POINT_STATS},
                     "by_surface": {}}
            self._players[player_id] = entry
        return entry

    @staticmethod
    def _add(counts, svc):
        for s in _POINT_STATS:
            counts[s][0] += svc[s][0]
            counts[s][1] += svc[s][1]

    @staticmethod
    def _side_stats(svpt, first_in, first_won, second_won, ace, df,
                    bp_faced, bp_saved, opp_bp_faced, opp_bp_saved):
        """Serving numbers for one player in one match.

        second_won is the Sackmann w_2nd_won count; second-serve
        attempts are svpt - first_in. bp_conv uses the opponent's
        serve columns (breaks the player converted).
        """
        return {
            "first": [first_won, first_in],
            "second": [second_won, max(svpt - first_in, 0)],
            "ace": [ace, svpt],
            "df": [df, svpt],
            "bp_conv": [opp_bp_faced - opp_bp_saved, opp_bp_faced],
            "bp_save": [bp_saved, bp_faced],
        }

    def update(self, winner_id, loser_id, surface,
               w_svpt, w_1st_in, w_1st_won, w_2nd_won, w_ace, w_df,
               w_bp_faced, w_bp_saved,
               l_svpt, l_1st_in, l_1st_won, l_2nd_won, l_ace, l_df,
               l_bp_faced, l_bp_saved):
        """Apply one match result. Call only AFTER features were read."""
        stats = (w_svpt, w_1st_in, w_1st_won, w_2nd_won, w_ace, w_df,
                 w_bp_faced, w_bp_saved,
                 l_svpt, l_1st_in, l_1st_won, l_2nd_won, l_ace, l_df,
                 l_bp_faced, l_bp_saved)
        if any(s is None for s in stats):
            return
        if w_svpt <= 0 or l_svpt <= 0:
            return
        w_svc = self._side_stats(
            w_svpt, w_1st_in, w_1st_won, w_2nd_won, w_ace, w_df,
            w_bp_faced, w_bp_saved, l_bp_faced, l_bp_saved)
        l_svc = self._side_stats(
            l_svpt, l_1st_in, l_1st_won, l_2nd_won, l_ace, l_df,
            l_bp_faced, l_bp_saved, w_bp_faced, w_bp_saved)
        for pid, svc in ((winner_id, w_svc), (loser_id, l_svc)):
            entry = self._entry(pid)
            self._add(entry["overall"], svc)
            if surface is not None:
                surf = entry["by_surface"].setdefault(
                    surface, {s: [0, 0] for s in _POINT_STATS})
                self._add(surf, svc)

    def _shrunk(self, counts, stat, prior):
        won, played = counts[stat]
        k = POINT_SERVE_SHRINK_K[stat]
        return (won + k * prior) / (played + k)

    def rates(self, player_id, surface):
        """{stat: shrunk rate} for one player on a surface."""
        entry = self._players.get(player_id)
        out = {}
        for stat in _POINT_STATS:
            if entry is None:
                prior = POINT_SERVE_PRIOR[stat]
                if surface is not None:
                    prior = POINT_SERVE_SURFACE_PRIOR.get(
                        surface, {}).get(stat, prior)
                out[stat] = prior
                continue
            overall = self._shrunk(entry["overall"], stat,
                                   POINT_SERVE_PRIOR[stat])
            if surface is None:
                out[stat] = overall
                continue
            surf = entry["by_surface"].get(surface)
            if surf is None:
                out[stat] = overall
                continue
            prior = overall
            out[stat] = self._shrunk(surf, stat, prior)
        return out

    def diffs(self, player_a, player_b, surface):
        """Six A-minus-B diffs, in _POINT_STATS order."""
        ra = self.rates(player_a, surface)
        rb = self.rates(player_b, surface)
        return tuple(ra[s] - rb[s] for s in _POINT_STATS)


# Momentum (form slope) is implemented as FormState.form_5,
# FormState.form_20 and FormState.momentum_diff methods on the
# FormState class above: (last-5 win% minus last-20 win%) of A
# minus the same for B. Trajectory, not level: the level versions
# (form raw / residual) were both dropped (z=-0.33 / -0.89), so
# only the slope is tested here.


# ---------------------------------------------------------------------------
# Clutch: tiebreak win%, deciding-set record, comeback after losing
# the first set
# ---------------------------------------------------------------------------

# Tour rate for winning a match after losing the first set,
# train period 2001-2022 (65,131 parsed matches): 0.1923.
COMEBACK_PRIOR = 0.1923
CLUTCH_TB_SHRINK_K = 10.0    # tiebreaks; shrink toward 0.5
CLUTCH_DECSET_SHRINK_K = 5.0  # deciding sets; shrink toward 0.5
CLUTCH_COMEBACK_SHRINK_K = 5.0  # opportunities; shrink toward 0.19


def parse_clutch(score):
    """Set-level detail from a Sackmann score string.

    Returns (tb_won_by_winner, tb_played, set_winners) where
    set_winners is a list of booleans, one per completed set, True
    if the match winner won that set. A set counts as a tiebreak
    set when its token carries "(...)" tiebreak points, or is a
    bracketed match tiebreak "[10-7]". A bare "7-6" with no
    parentheses carries no tiebreak information and is not counted.
    Walkovers / bare RET return empty lists.
    """
    if score is None:
        return 0, 0, []
    s = str(score).strip()
    if s in ("", "W/O", "Walkover", "RET"):
        return 0, 0, []
    tb_won = 0
    tb_played = 0
    set_winners = []
    for token in s.split():
        if token == "RET":
            continue
        if token.startswith("[") and token.endswith("]"):
            tb_played += 1
            tb_won += 1  # bracketed set is won by the match winner
            set_winners.append(True)
            continue
        core = token.split("(")[0]
        if "-" not in core:
            continue
        try:
            ga, gb = core.split("-", 1)
            ga, gb = int(ga), int(gb)
        except ValueError:
            continue
        won_set = ga > gb
        set_winners.append(won_set)
        if "(" in token:
            tb_played += 1
            if won_set:
                tb_won += 1
    return tb_won, tb_played, set_winners


class ClutchState:
    """Walk-forward per-player clutch records.

    Tiebreaks: (won, played) per player; the loser of a tiebreak
    set is recorded as a tiebreak loss. Deciding sets: the last set
    of a match that went the distance (sets == best_of); best_of
    None means the deciding set cannot be identified and is
    skipped. Comebacks: a match whose first set the player lost but
    the match they won (opportunities = matches where they lost the
    first set). All three are win% shrunk toward their prior and
    centred, so a player with no history scores 0. The literature
    says clutch persistence is weak, so this batch is expected to
    be small; it is reported honestly either way.
    """

    def __init__(self):
        self._players = {}

    def _entry(self, player_id):
        return self._players.setdefault(player_id, {
            "tb": [0, 0], "dec": [0, 0], "come": [0, 0]})

    def update(self, winner_id, loser_id, score, best_of):
        """Apply one match result. Call only AFTER features were read."""
        tb_won, tb_played, set_winners = parse_clutch(score)
        if not set_winners:
            return
        w_entry = self._entry(winner_id)
        l_entry = self._entry(loser_id)
        w_entry["tb"][0] += tb_won
        w_entry["tb"][1] += tb_played
        l_entry["tb"][0] += tb_played - tb_won
        l_entry["tb"][1] += tb_played
        if best_of is not None and len(set_winners) == best_of:
            dec_won = set_winners[-1]
            w_entry["dec"][0] += 1 if dec_won else 0
            w_entry["dec"][1] += 1
            l_entry["dec"][0] += 0 if dec_won else 1
            l_entry["dec"][1] += 1
        if not set_winners[0]:
            # Winner lost the first set and still won: a comeback.
            w_entry["come"][0] += 1
            w_entry["come"][1] += 1
        else:
            # Loser won the first set and lost: a failed comeback.
            l_entry["come"][0] += 0
            l_entry["come"][1] += 1

    def _rate(self, player_id, key, shrink_k, prior):
        entry = self._players.get(player_id)
        if entry is None:
            return 0.0
        won, played = entry[key]
        if played == 0:
            return 0.0
        return (won + shrink_k * prior) / (played + shrink_k) - prior

    def diffs(self, player_a, player_b):
        """(tiebreak, deciding-set, comeback) A-minus-B diffs."""
        return (
            self._rate(player_a, "tb", CLUTCH_TB_SHRINK_K, 0.5)
            - self._rate(player_b, "tb", CLUTCH_TB_SHRINK_K, 0.5),
            self._rate(player_a, "dec", CLUTCH_DECSET_SHRINK_K, 0.5)
            - self._rate(player_b, "dec", CLUTCH_DECSET_SHRINK_K, 0.5),
            self._rate(player_a, "come", CLUTCH_COMEBACK_SHRINK_K,
                       COMEBACK_PRIOR)
            - self._rate(player_b, "come", CLUTCH_COMEBACK_SHRINK_K,
                         COMEBACK_PRIOR),
        )


# ---------------------------------------------------------------------------
# In-event games: how much a player has already played at THIS event
# ---------------------------------------------------------------------------

class EventState:
    """Games already played at the current tournament, per player.

    Keyed by (player_id, tourney_id). The walk visits matches in
    (tourney_date, tourney_id, match_num) order, so earlier rounds
    of the same event are processed before later ones, and the
    count at feature-read time is exactly the games played so far
    at this event. Walkovers add 0 games. This is in-tournament
    progression, distinct from the games7/games14 fatigue windows
    (which span events).
    """

    def __init__(self):
        self._games = {}

    def update(self, winner_id, loser_id, tourney_id, total_games):
        for pid in (winner_id, loser_id):
            key = (pid, tourney_id)
            self._games[key] = self._games.get(key, 0) + total_games

    def diff(self, player_a, player_b, tourney_id):
        """Games already played at this event, A minus B."""
        return float(self._games.get((player_a, tourney_id), 0)
                     - self._games.get((player_b, tourney_id), 0))


# ---------------------------------------------------------------------------
# Recent retirements: injury proxy from the last 60 days
# ---------------------------------------------------------------------------

RETIRE_WINDOW_DAYS = 60


class RetirementState:
    """Whether a player (or their opponent) retired recently.

    In Sackmann scores RET marks a match the loser retired from,
    so the loser is always the retiree. Two flags per player, each
    1 if such an event happened within the last 60 days, else 0:
    self-retired (the player retired: an injury signal about them)
    and opponent-retired (an opponent retired against them: the
    player won by retirement, which says little about the player's
    own fitness and may instead mark an easy win). Both enter as
    diffs so the regression can weight them separately.
    """

    def __init__(self, window_days=RETIRE_WINDOW_DAYS):
        self.window_days = window_days
        self._self = {}
        self._opp = {}

    def update(self, winner_id, loser_id, date, retired):
        """Apply one match result. Call only AFTER features were read."""
        if not retired:
            return
        self._self.setdefault(loser_id, []).append(date)
        self._opp.setdefault(winner_id, []).append(date)

    def _flag(self, dates, date):
        if not dates:
            return 0.0
        cutoff = date - _dt.timedelta(days=self.window_days)
        return 1.0 if any(d > cutoff for d in dates) else 0.0

    def diffs(self, player_a, player_b, date):
        """(self-retired, opponent-retired) A-minus-B diffs."""
        return (
            self._flag(self._self.get(player_a, []), date)
            - self._flag(self._self.get(player_b, []), date),
            self._flag(self._opp.get(player_a, []), date)
            - self._flag(self._opp.get(player_b, []), date),
        )
