"""Walk-forward Elo state for the backtest.

This module does NOT define its own Elo. It reuses the production
formulas, imported from the scripts/ directory exactly as the
runtime tools use them:

- ``scripts/elo.py``: ``INITIAL_ELO`` (1500), ``DEFAULT_K`` (32),
  ``update_elo`` (win = 1, loss = 0, zero-sum update).
- ``scripts/surface_elo.py``: ``blended_elo`` (prior = 100),
  ``win_prob`` (1 / (1 + 10**(-diff/400))).

The locked model (see scripts/tennis_tools.py, model string
"K32 overall Elo + surface blend prior100 + calibration a=0.8,
b=0.0") is then:

    blended = w * surface_elo + (1 - w) * overall_elo
    w = n_surface / (n_surface + 100);  n_surface = 0 -> overall only
    p_blended = win_prob(blended_a, blended_b)
    p_final = sigmoid(0.8 * logit(p_blended))

Matches with a NULL surface update overall Elo only and are
predicted from overall Elo only, exactly as compute_surface_elo.py
does.

Ordering: matches are processed in (tourney_date, tourney_id,
match_num) order, the same order compute_elo.py and
compute_surface_elo.py use. tourney_date is the tournament start,
not the exact match day, so within-tournament order is by
match_num (early rounds first). That is the repo's existing
convention and is documented, not silently changed.

Leakage rule: for every match, ``predict`` is called BEFORE
``update``. ``verify_against_history`` additionally recomputes
the whole history from 1968 and checks it against the stored
atp_elo_history / atp_surface_elo_history tables, which were
built by the same sequential process; any mismatch means the
ordering or the update formula drifted from production.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elo import DEFAULT_K, INITIAL_ELO, update_elo  # noqa: E402
from surface_elo import blended_elo, win_prob  # noqa: E402
from tennis_tools import calibrated_prob  # noqa: E402

PRIOR = 100
CALIBRATION_A = 0.8
CALIBRATION_B = 0.0
BURN_IN_END = "2000-12-31"


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class EloState:
    """Overall + per-surface Elo, updated strictly sequentially.

    Optional constructor args parameterize the locked model without
    changing its defaults: ``k`` (Elo update step, default 32),
    ``prior`` (surface-blend prior, default 100), ``cal_a`` /
    ``cal_b`` (calibration, default 0.8 / 0.0). With no args the
    state is exactly the locked model, so verify_against_history
    and every existing caller behave identically. Calibration only
    affects ``predict``, never ``update``, so one Elo walk per
    (k, prior) can score every (cal_a, cal_b) combo offline from
    the recorded blended probabilities.
    """

    def __init__(self, k=DEFAULT_K, prior=PRIOR, cal_a=CALIBRATION_A, cal_b=CALIBRATION_B):
        self.k = k
        self.prior = prior
        self.cal_a = cal_a
        self.cal_b = cal_b
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
        """Blended Elo for one player on a surface (pre-match)."""
        overall = self._overall(player_id)
        if surface is None:
            return overall, overall, None, 0
        surf, n = self._surface(player_id, surface)
        return blended_elo(surf, overall, n, self.prior), overall, surf, n

    def predict(self, player_a, player_b, surface):
        """Locked-model calibrated probability that A beats B.

        Uses only ratings built from matches before this one.
        Returns (p_calibrated, p_blended, blended_a, blended_b).
        """
        blended_a, _, _, _ = self.blended(player_a, surface)
        blended_b, _, _, _ = self.blended(player_b, surface)
        p_blended = win_prob(blended_a, blended_b)
        p_cal = calibrated_prob(p_blended, a=self.cal_a, b=self.cal_b)
        return p_cal, p_blended, blended_a, blended_b

    def update(self, winner_id, loser_id, surface):
        """Apply one match result. Call only AFTER predict()."""
        w_over = self._overall(winner_id)
        l_over = self._overall(loser_id)
        w_new, l_new, _ = update_elo(w_over, l_over, k=self.k)
        self.overall[winner_id] = w_new
        self.overall[loser_id] = l_new
        if surface is None:
            return
        w_key = (winner_id, surface)
        l_key = (loser_id, surface)
        w_surf = self.surface_rating.get(w_key, INITIAL_ELO)
        l_surf = self.surface_rating.get(l_key, INITIAL_ELO)
        w_s_new, l_s_new, _ = update_elo(w_surf, l_surf, k=self.k)
        self.surface_rating[w_key] = w_s_new
        self.surface_rating[l_key] = l_s_new
        self.surface_n[w_key] = self.surface_n.get(w_key, 0) + 1
        self.surface_n[l_key] = self.surface_n.get(l_key, 0) + 1


def verify_against_history(conn, upto_date="2024-12-31"):
    """Recompute Elo from atp_matches and compare to history tables.

    Returns a dict with rows checked and max absolute differences
    for overall and surface Elo. A non-zero max difference means
    this module's update order/formula differs from the tables
    the production tools read, and the run must stop.
    """
    state = EloState()
    cur = conn.cursor()
    cur.execute(
        "SELECT tourney_id, match_num, tourney_date, winner_id, loser_id, surface "
        "FROM atp_matches WHERE winner_id IS NOT NULL AND loser_id IS NOT NULL "
        "AND tourney_date <= ? ORDER BY tourney_date, tourney_id, match_num",
        (upto_date,),
    )
    rows = cur.fetchall()
    cur.execute(
        "SELECT winner_elo_after, loser_elo_after FROM atp_elo_history "
        "WHERE tourney_date <= ? ORDER BY tourney_date, tourney_id, match_num",
        (upto_date,),
    )
    elo_after = cur.fetchall()
    cur.execute(
        "SELECT winner_surface_after, loser_surface_after FROM atp_surface_elo_history "
        "WHERE tourney_date <= ? ORDER BY tourney_date, tourney_id, match_num",
        (upto_date,),
    )
    surf_after = cur.fetchall()
    if not (len(rows) == len(elo_after) == len(surf_after)):
        return {
            "checked": 0,
            "error": (
                f"row count mismatch: matches={len(rows)} "
                f"elo_history={len(elo_after)} surface_history={len(surf_after)}"
            ),
        }
    max_over = 0.0
    max_surf = 0.0
    mismatches = 0
    for (tid, mnum, _d, w, l, surf), (w_after, l_after), (ws_after, ls_after) in zip(
        rows, elo_after, surf_after
    ):
        # Compare the PAIRWISE computed after-values to the table.
        # The tables store update_elo's (winner_new, loser_new) for
        # the match; the dict state collapses them when
        # winner_id == loser_id (placeholder player 199999
        # "Unknown" plays himself in a few early matches), so the
        # state itself must not be compared there.
        w_before = state._overall(w)
        l_before = state._overall(l)
        w_new, l_new, _ = update_elo(w_before, l_before, k=DEFAULT_K)
        max_over = max(max_over, abs(w_new - w_after), abs(l_new - l_after))
        state.update(w, l, surf)
        if surf is not None:
            if ws_after is None or ls_after is None:
                mismatches += 1
                continue
            # Surface after-values are likewise pairwise; recompute
            # from the pre-match surface ratings captured before update.
            # (state was already updated above, so surface check uses
            # the stored table only for the winner/loser pair via the
            # same pairwise logic on a fresh read is not possible
            # here; instead compare the state values, which agree
            # with the table whenever winner != loser.)
            if w != l:
                max_surf = max(
                    max_surf,
                    abs(state.surface_rating[(w, surf)] - ws_after),
                    abs(state.surface_rating[(l, surf)] - ls_after),
                )
    return {
        "checked": len(rows),
        "max_overall_diff": max_over,
        "max_surface_diff": max_surf,
        "mismatches": mismatches,
        "error": None,
    }
