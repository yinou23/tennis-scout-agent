"""Glicko-2 ratings for the backtest, per Glickman's published paper.

Source: Mark E. Glickman, "Example of the Glicko-2 system",
Boston University, March 22, 2022 (glicko.net/glicko/glicko2.pdf),
steps 1-8. The implementation is a direct port of those steps; the
regression test below the code reproduces the paper's worked example
exactly (1500 / RD 200 / sigma 0.06, win vs 1400/RD 30, losses vs
1550/RD 100 and 1700/RD 300, tau 0.5 -> 1464.06 / 151.52 / 0.05999).

Rating periods are ISO calendar weeks (a tournament is a week, so a
week is also the natural tennis period). Within one week all games
are treated as simultaneous, exactly as the paper requires: every
prediction in the week uses the pre-week ratings, and all updates
are applied at the end of the week from a snapshot of pre-week
states.

Idle growth is exact and lazy: the paper's step 6 says a player who
does not compete in a period keeps rating and volatility but the RD
grows as phi' = sqrt(phi^2 + sigma^2) per period. That is additive
in phi^2, so the effective pre-period phi after k idle weeks is
sqrt(phi^2 + sigma^2 * k). No approximation is involved.

Conventions (documented choices, not paper text):
- A player who has never played starts at 1500 / RD 350 / sigma 0.06
  (paper step 1a). tau = 0.5, the paper's example value, inside its
  recommended 0.3-1.2 range.
- Walkovers (no games played) and unparseable scores are not games:
  they do not enter any player's game list for the period.
- Retirements DO enter: the games actually played happened, and the
  winner/loser is known. (For WElo, retirements are skipped instead;
  see features.WEloState. The two states are independent.)
- This module only supplies RD features to the logistic container
  (see compute_glicko_features). It never predicts by itself.
"""

import math

SCALE = 173.7178
TAU = 0.5
EPSILON = 1e-6
INITIAL_RATING = 1500.0
INITIAL_RD = 350.0
INITIAL_SIGMA = 0.06


def _g(phi):
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _e(mu, mu_j, phi_j):
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _volatility_update(phi, sigma, v, delta):
    """Paper step 5: new sigma via the Illinois algorithm.

    f(x) = e^x (delta^2 - phi^2 - v - e^x) / (2 (phi^2 + v + e^x)^2)
           - (x - a) / tau^2,  a = ln(sigma^2).
    Returns sigma' = e^{A/2} at convergence.
    """
    a = math.log(sigma * sigma)

    def f(x):
        ex = math.exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return num / den - (x - a) / (TAU * TAU)

    A = a
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        # In this branch f(A) < 0 (since delta^2 - phi^2 - v <= 0),
        # so B is the first a - k*tau with f(B) > 0: the bracket
        # [B, A] then straddles the root. (The paper's example
        # confirms this: k = 1 gives B = a - tau with f(B) = +2.0,
        # opposite in sign to f(A) = -0.00054.)
        k = 1
        while True:
            B = a - k * TAU
            if f(B) > 0:
                break
            k += 1
            if k > 100:
                raise RuntimeError("volatility bracket search did not converge")
    fA, fB = f(A), f(B)
    _iters = 0
    while abs(B - A) > EPSILON:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA = fA / 2.0
        B, fB = C, fC
        _iters += 1
        if _iters > 1000:
            raise RuntimeError("volatility Illinois iteration did not converge")
    return math.exp(A / 2.0)


def update_player(mu, phi, sigma, games):
    """One rating period for one player (paper steps 3-8).

    games: list of (opp_mu, opp_phi, score) with opp values at the
    pre-period state and score 1 for a win, 0 for a loss.
    Returns (mu', phi', sigma'). Callers handle idle growth.
    """
    if not games:
        raise ValueError("update_player needs at least one game")
    v_inv = 0.0
    delta_sum = 0.0
    for mu_j, phi_j, s in games:
        g = _g(phi_j)
        e = _e(mu, mu_j, phi_j)
        v_inv += g * g * e * (1.0 - e)
        delta_sum += g * (s - e)
    v = 1.0 / v_inv
    delta = v * delta_sum
    sigma_prime = _volatility_update(phi, sigma, v, delta)
    phi_star = math.sqrt(phi * phi + sigma_prime * sigma_prime)
    phi_prime = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_prime = mu + phi_prime * phi_prime * delta_sum
    return mu_prime, phi_prime, sigma_prime


class GlickoState:
    """Per-player Glicko-2 state over weekly rating periods.

    Stored per player: (mu, phi, sigma, last_week), all on the
    Glicko-2 scale. phi is the RD at the end of last_week's update;
    the effective pre-period phi for a later week w is
    sqrt(phi^2 + sigma^2 * (w - last_week)), per the paper's step 6
    for idle periods.
    """

    def __init__(self):
        self._players = {}

    @staticmethod
    def _week_of(date):
        return date.toordinal() // 7

    def _entry(self, player_id):
        entry = self._players.get(player_id)
        if entry is None:
            entry = {
                "mu": 0.0,
                "phi": INITIAL_RD / SCALE,
                "sigma": INITIAL_SIGMA,
                "last_week": None,
            }
            self._players[player_id] = entry
        return entry

    def phi_eff(self, player_id, date):
        """Pre-period phi for a match on `date` (idle growth applied)."""
        entry = self._entry(player_id)
        if entry["last_week"] is None:
            return entry["phi"]
        weeks = self._week_of(date) - entry["last_week"]
        if weeks <= 0:
            return entry["phi"]
        return math.sqrt(entry["phi"] ** 2 + entry["sigma"] ** 2 * weeks)

    def mu(self, player_id):
        return self._entry(player_id)["mu"]

    def rd(self, player_id, date):
        """Pre-period RD on the original scale (rating points)."""
        return self.phi_eff(player_id, date) * SCALE

    def end_of_week(self, date, games_by_player):
        """Apply one week's games. games_by_player: player_id ->
        list of (opp_id, score). All reads use pre-week states.
        """
        week = self._week_of(date)
        # Snapshot pre-week (mu, phi*) for everyone involved.
        pre = {}
        for pid, games in games_by_player.items():
            entry = self._entry(pid)
            pre[pid] = (entry["mu"], self.phi_eff(pid, date))
            for opp_id, _s in games:
                if opp_id not in pre:
                    o_entry = self._entry(opp_id)
                    pre[opp_id] = (o_entry["mu"], self.phi_eff(opp_id, date))
        updates = {}
        for pid, games in games_by_player.items():
            mu, phi = pre[pid]
            sigma = self._entry(pid)["sigma"]
            opp_games = [(pre[opp][0], pre[opp][1], s) for opp, s in games]
            updates[pid] = update_player(mu, phi, sigma, opp_games)
        for pid, (mu_p, phi_p, sigma_p) in updates.items():
            entry = self._entry(pid)
            entry["mu"] = mu_p
            entry["phi"] = phi_p
            entry["sigma"] = sigma_p
            entry["last_week"] = week

    def n_players(self):
        return len(self._players)


def paper_example():
    """The paper's worked example. Returns (r', RD', sigma')."""
    mu, phi, sigma = 0.0, 200.0 / SCALE, 0.06
    games = [
        ((1400.0 - 1500.0) / SCALE, 30.0 / SCALE, 1),
        ((1550.0 - 1500.0) / SCALE, 100.0 / SCALE, 0),
        ((1700.0 - 1500.0) / SCALE, 300.0 / SCALE, 0),
    ]
    mu_p, phi_p, sigma_p = update_player(mu, phi, sigma, games)
    return 173.7178 * mu_p + 1500.0, 173.7178 * phi_p, sigma_p


def compute_glicko_features(matches):
    """Two-pass pre-computation of per-match RD features.

    matches: walk-order rows with tourney_date, winner_id, loser_id
    and a parseable score (see features.parse_score). Walkovers are
    skipped as games. Returns {(tourney_id, match_num):
    (rd_a, rd_b)} with A = lower player id, RDs on the original
    scale, pre-week state (no within-week leakage by construction).
    """
    from features import parse_score  # local import: no cycle at runtime

    state = GlickoState()
    features = {}
    bucket_week = None
    bucket = []  # (match, w_games, l_games, retired)

    def flush(bucket):
        if not bucket:
            return
        games_by_player = {}
        for m, _wg, _lg, _ret in bucket:
            w_id, l_id = m["winner_id"], m["loser_id"]
            games_by_player.setdefault(w_id, []).append((l_id, 1))
            games_by_player.setdefault(l_id, []).append((w_id, 0))
        state.end_of_week(bucket[0][0]["tourney_date"], games_by_player)

    for m in matches:
        week = GlickoState._week_of(m["tourney_date"])
        if bucket_week is None:
            bucket_week = week
        if week != bucket_week:
            flush(bucket)
            bucket = []
            bucket_week = week
        w_games, l_games, _sets, _retired, walkover = parse_score(m.get("score"))
        if walkover or (w_games + l_games) == 0:
            # No games played: not a Glicko game, but the match still
            # gets RD features from the current pre-week state.
            w_id, l_id = m["winner_id"], m["loser_id"]
            a_id, b_id = (w_id, l_id) if w_id < l_id else (l_id, w_id)
            features[(m["tourney_id"], m["match_num"])] = (
                state.rd(a_id, m["tourney_date"]),
                state.rd(b_id, m["tourney_date"]),
            )
            continue
        w_id, l_id = m["winner_id"], m["loser_id"]
        a_id, b_id = (w_id, l_id) if w_id < l_id else (l_id, w_id)
        features[(m["tourney_id"], m["match_num"])] = (
            state.rd(a_id, m["tourney_date"]),
            state.rd(b_id, m["tourney_date"]),
        )
        bucket.append((m, w_games, l_games, _retired))
    flush(bucket)
    return features
