"""Week 3 tools, deterministic, no LLM, no guessing.

Tool 1: player Elo lookup.
Tool 2: player recent form (win form and games form, current, not pre-match snapshot).
Tool 3: head-to-head record for two players (raw counts only, no shrinkage).
Tool 4: player surface record (career win-loss per surface, not Elo).
Tool 5: match probability (blended Elo, then calibrated, final locked model).

What this tool does:
- Given a player name, return that player's current overall Elo
  (from atp_elo_current, Week 2 classic Elo, K=32, initial 1500),
  matches played, and last match date.
- For each surface (Hard, Clay, Grass, Carpet), return the current
  surface Elo and surface matches, derived from the last row for
  that player on that surface in atp_surface_elo_history, plus
  the blended Elo for that surface using the locked Week 2 formula:
      blended = w * surface_elo + (1 - w) * overall_elo
      w = n_surface / (n_surface + 100)
  If the player has never played that surface, surface_elo is
  None, surface_matches is 0, and blended is the overall Elo.

What this tool does NOT do:
- It does not predict a match (that is Tool 5, later).
- It does not use live data. Elo tables are as of the Week 1
  import: matches to 2026-05-25. The returned last_match_date
  is the player's own last match, not a live update.
- It does not guess a player. If a name matches no player, it
  returns None. If it matches more than one player, it returns
  the candidates and asks for a full name. Never pick one silently.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/tennis_tools.py "Sinner"
    python scripts/tennis_tools.py "Jannik Sinner" "Hard"
    python scripts/tennis_tools.py form "Jannik Sinner"
    python scripts/tennis_tools.py h2h "Jannik Sinner" "Carlos Alcaraz"
    python scripts/tennis_tools.py surface "Jannik Sinner"
    python scripts/tennis_tools.py match "Jannik Sinner" "Carlos Alcaraz" "Hard"
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from surface_elo import blended_elo
except ImportError:  # pragma: no cover - only when run outside scripts/
    blended_elo = None

SURFACES = ("Hard", "Clay", "Grass", "Carpet")
PRIOR = 100


def _surface_current(conn, player_id, surface):
    """Return (surface_elo, surface_matches) as of the last match.

    Reads the last history row for this player on this surface.
    If the player was the winner, use winner_surface_after and
    winner_surface_n_before + 1; if the loser, use the loser fields.
    Returns (None, 0) if the player never played that surface.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT winner_id, winner_surface_after, winner_surface_n_before,
                   loser_surface_after, loser_surface_n_before
            FROM atp_surface_elo_history
            WHERE surface = %s AND (winner_id = %s OR loser_id = %s)
            ORDER BY tourney_date DESC, tourney_id DESC, match_num DESC
            LIMIT 1
            """,
            (surface, player_id, player_id),
        )
        row = cur.fetchone()
    if row is None:
        return None, 0
    winner_id, w_after, w_n, l_after, l_n = row
    if winner_id == player_id:
        return float(w_after), int(w_n) + 1
    return float(l_after), int(l_n) + 1


def find_players(conn, name):
    """Find players by full name or last name, case-insensitive.

    Returns a list of (player_id, first_name, last_name).
    Empty list means no match. More than one means ambiguous.
    """
    name = name.strip()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT player_id, first_name, last_name
            FROM atp_players
            WHERE LOWER(first_name || ' ' || last_name) = LOWER(%s)
               OR LOWER(last_name) = LOWER(%s)
            ORDER BY last_name, first_name
            """,
            (name, name),
        )
        return cur.fetchall()


def get_player_elo(conn, name):
    """Return a dict for one player, or None, or an ambiguous dict.

    Returns None if no player matches.
    Returns {"ambiguous": True, "candidates": [...]} if more than one matches.
    Otherwise returns a dict with overall Elo, matches, last date,
    and a per-surface breakdown with blended Elo (prior=100).
    """
    candidates = find_players(conn, name)
    if not candidates:
        return None
    if len(candidates) > 1:
        return {
            "ambiguous": True,
            "candidates": [
                {"player_id": pid, "name": f"{first} {last}".strip()}
                for pid, first, last in candidates
            ],
        }
    player_id, first_name, last_name = candidates[0]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT elo, matches_played, last_match_date "
            "FROM atp_elo_current WHERE player_id = %s",
            (player_id,),
        )
        row = cur.fetchone()
    if row is None:
        return {
            "player_id": player_id,
            "name": f"{first_name} {last_name}".strip(),
            "overall_elo": None,
            "matches_played": 0,
            "last_match_date": None,
            "surfaces": {},
            "note": "Player exists in atp_players but has no main-tour Elo row.",
        }
    overall_elo, matches_played, last_match_date = row
    overall_elo = float(overall_elo)
    surfaces = {}
    for surface in SURFACES:
        surf_elo, surf_n = _surface_current(conn, player_id, surface)
        if surf_elo is None:
            blended = overall_elo
        elif blended_elo is None:
            blended = overall_elo
        else:
            blended = blended_elo(surf_elo, overall_elo, surf_n, PRIOR)
        surfaces[surface] = {
            "surface_elo": surf_elo,
            "surface_matches": surf_n,
            "blended_elo": round(float(blended), 1),
        }
    return {
        "player_id": player_id,
        "name": f"{first_name} {last_name}".strip(),
        "overall_elo": round(overall_elo, 1),
        "matches_played": int(matches_played),
        "last_match_date": str(last_match_date) if last_match_date else None,
        "surfaces": surfaces,
        "data_as_of": "Matches imported to 2026-05-25; Elo is not live.",
    }



# ---------------------------------------------------------------------------
# Tool 2: current form
# ---------------------------------------------------------------------------
# Definitions, same as Week 2, but anchored at the data-as-of date
# (the maximum tourney_date in atp_matches, 2026-05-25), not at a
# hypothetical next match:
# - last10 win form: win rate in the player's last 10 main-tour
#   matches, regardless of time gap. If fewer than 10 matches exist,
#   n is that total, pct is wins / n. If no matches, pct is None.
# - last52 win form: win rate in matches with tourney_date in
#   [anchor - 365 days, anchor]. If none, pct is None.
# - retirement: RET losses only (score contains a RET token, loser
#   retired mid-match). W/O is a pre-match walkover, not counted
#   here. Retirement is reported as an observable event, never as
#   a diagnosis of injury, and it does not change any probability.
# - last10 / last52 games form: total games won / total games
#   played in complete matches only (atp_match_games.is_complete),
#   incomplete matches skipped. Same windows as win form.
# Limitation, same as Week 2: tourney_date is tournament start,
# not exact match day, so the 52-week window is accurate to within
# a tournament length. Form is a descriptor, not a probability
# input (Week 2 verdict: form did not improve held-out LL/Brier).


def _win_rate(wins, n):
    if n == 0:
        return None
    return round(wins / n, 4)


def _games_pct(games_won, games_total):
    if games_total == 0:
        return None
    return round(games_won / games_total, 4)


def get_player_form(conn, name):
    """Return current win form and games form for one player.

    Returns None if no player matches.
    Returns {"ambiguous": True, "candidates": [...]} if more than one matches.
    Otherwise returns a dict with last10 and last52 for both win %
    and games %, plus the anchor date used for last52.
    """
    candidates = find_players(conn, name)
    if not candidates:
        return None
    if len(candidates) > 1:
        return {
            "ambiguous": True,
            "candidates": [
                {"player_id": pid, "name": f"{first} {last}".strip()}
                for pid, first, last in candidates
            ],
        }
    player_id, first_name, last_name = candidates[0]
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(tourney_date) FROM atp_matches")
        anchor = cur.fetchone()[0]
        # Last 10 matches, most recent first.
        cur.execute(
            """
            SELECT winner_id
            FROM atp_matches
            WHERE winner_id = %s OR loser_id = %s
            ORDER BY tourney_date DESC, tourney_id DESC, match_num DESC
            LIMIT 10
            """,
            (player_id, player_id),
        )
        last10_winners = [row[0] for row in cur.fetchall()]
        # Last 52 weeks: matches in [anchor - 365 days, anchor].
        cur.execute(
            """
            SELECT winner_id
            FROM atp_matches
            WHERE (winner_id = %s OR loser_id = %s)
              AND tourney_date >= (%s::date - INTERVAL '365 days')
              AND tourney_date <= %s
            """,
            (player_id, player_id, anchor, anchor),
        )
        last52_winners = [row[0] for row in cur.fetchall()]
        # Games form: complete matches only, joined to atp_match_games.
        cur.execute(
            """
            SELECT m.winner_id, g.winner_games, g.loser_games
            FROM atp_matches m JOIN atp_match_games g
              ON g.tourney_id = m.tourney_id AND g.match_num = m.match_num
            WHERE (m.winner_id = %s OR m.loser_id = %s) AND g.is_complete
            ORDER BY m.tourney_date DESC, m.tourney_id DESC, m.match_num DESC
            LIMIT 10
            """,
            (player_id, player_id),
        )
        last10_games_rows = cur.fetchall()
        cur.execute(
            """
            SELECT m.winner_id, g.winner_games, g.loser_games
            FROM atp_matches m JOIN atp_match_games g
              ON g.tourney_id = m.tourney_id AND g.match_num = m.match_num
            WHERE (m.winner_id = %s OR m.loser_id = %s) AND g.is_complete
              AND m.tourney_date >= (%s::date - INTERVAL '365 days')
              AND m.tourney_date <= %s
            """,
            (player_id, player_id, anchor, anchor),
        )
        last52_games_rows = cur.fetchall()

    def _games_summary(rows):
        games_won = 0
        games_total = 0
        for winner_id, w_games, l_games in rows:
            if w_games is None or l_games is None:
                continue
            total = w_games + l_games
            if total == 0:
                continue
            games_total += total
            games_won += w_games if winner_id == player_id else l_games
        return games_won, games_total

    # Retirement descriptor: RET losses only, loser retired mid-match.
    # Fetch last 10 matches with scores, the most recent RET loss,
    # RET count in the last52 window, and total RET losses.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT winner_id, score
            FROM atp_matches
            WHERE winner_id = %s OR loser_id = %s
            ORDER BY tourney_date DESC, tourney_id DESC, match_num DESC
            LIMIT 10
            """,
            (player_id, player_id),
        )
        last10_score_rows = cur.fetchall()
        cur.execute(
            """
            SELECT tourney_date, tourney_id, match_num
            FROM atp_matches
            WHERE loser_id = %s AND score LIKE %s
            ORDER BY tourney_date DESC, tourney_id DESC, match_num DESC
            LIMIT 1
            """,
            (player_id, '%RET%'),
        )
        last_ret_row = cur.fetchone()
        cur.execute(
            """
            SELECT COUNT(*)
            FROM atp_matches
            WHERE loser_id = %s AND score LIKE %s
              AND tourney_date >= (%s::date - INTERVAL '365 days')
              AND tourney_date <= %s
            """,
            (player_id, '%RET%', anchor, anchor),
        )
        ret_last52_n = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM atp_matches WHERE loser_id = %s AND score LIKE %s",
            (player_id, '%RET%'),
        )
        ret_total_n = cur.fetchone()[0]
        matches_since_ret = None
        if last_ret_row is not None:
            ret_date, ret_tourney_id, ret_match_num = last_ret_row
            cur.execute(
                """
                SELECT COUNT(*)
                FROM atp_matches
                WHERE (winner_id = %s OR loser_id = %s)
                  AND (tourney_date, tourney_id, match_num) > (%s, %s, %s)
                """,
                (player_id, player_id, ret_date, ret_tourney_id, ret_match_num),
            )
            matches_since_ret = cur.fetchone()[0]

    def _is_ret_loss(winner_id, score):
        if winner_id == player_id:
            return False
        if score is None:
            return False
        return "RET" in str(score).upper().split()

    ret_last10_n = sum(1 for w, s in last10_score_rows if _is_ret_loss(w, s))
    last_match_was_ret = bool(last10_score_rows) and _is_ret_loss(last10_score_rows[0][0], last10_score_rows[0][1])
    if last_ret_row is not None and anchor is not None:
        days_since_ret = (anchor - last_ret_row[0]).days
        last_ret_date_str = str(last_ret_row[0])
    else:
        days_since_ret = None
        last_ret_date_str = None
    retirement = {
        "last_retirement_date": last_ret_date_str,
        "days_since_last_retirement": days_since_ret,
        "matches_since_last_retirement": matches_since_ret,
        "last_match_was_retirement": last_match_was_ret,
        "retirements_last10": ret_last10_n,
        "retirements_last52": int(ret_last52_n),
        "retirements_total": int(ret_total_n),
    }

    w10_n = len(last10_winners)
    w10_wins = sum(1 for w in last10_winners if w == player_id)
    w52_n = len(last52_winners)
    w52_wins = sum(1 for w in last52_winners if w == player_id)
    g10_won, g10_total = _games_summary(last10_games_rows)
    g52_won, g52_total = _games_summary(last52_games_rows)
    return {
        "player_id": player_id,
        "name": f"{first_name} {last_name}".strip(),
        "anchor_date": str(anchor) if anchor else None,
        "win_form": {
            "last10_pct": _win_rate(w10_wins, w10_n),
            "last10_wins": w10_wins,
            "last10_n": w10_n,
            "last52_pct": _win_rate(w52_wins, w52_n),
            "last52_wins": w52_wins,
            "last52_n": w52_n,
        },
        "retirement": retirement,
        "games_form": {
            "last10_games_pct": _games_pct(g10_won, g10_total),
            "last10_matches": len(last10_games_rows),
            "last10_games_won": g10_won,
            "last10_games_total": g10_total,
            "last52_games_pct": _games_pct(g52_won, g52_total),
            "last52_matches": len(last52_games_rows),
            "last52_games_won": g52_won,
            "last52_games_total": g52_total,
        },
        "data_as_of": "Matches imported to 2026-05-25; form is not live.",
    }



# ---------------------------------------------------------------------------
# Tool 3: head-to-head
# ---------------------------------------------------------------------------
# Raw counts only. No shrinkage, no probability adjustment.
# Week 2 discussed H2H with shrinkage as a later candidate, not an
# adopted layer, so this tool must not be read as a prediction input.
# It returns total meetings, wins per player, a per-surface
# breakdown, and the last 5 meetings in reverse chronological order
# (date, tournament, surface, winner, score) for a report to cite.


def _h2h_summary(rows, id_a, id_b):
    """Pure summary from rows of (winner_id, surface).

    rows are in any order; surface may be None (missing in data).
    Returns total, wins per player, and per-surface wins.
    """
    total = len(rows)
    wins_a = sum(1 for winner_id, _surface in rows if winner_id == id_a)
    wins_b = total - wins_a
    by_surface = {}
    for winner_id, surface in rows:
        key = surface if surface is not None else "Unknown"
        entry = by_surface.setdefault(key, {"total": 0, "wins_a": 0, "wins_b": 0})
        entry["total"] += 1
        if winner_id == id_a:
            entry["wins_a"] += 1
        else:
            entry["wins_b"] += 1
    return {
        "total": total,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "by_surface": by_surface,
    }


def get_head_to_head(conn, name_a, name_b):
    """Return head-to-head record for two players.

    Returns None if either name matches no player.
    Returns {"ambiguous": True, "side": "a" or "b", "candidates": [...]}
    if either side matches more than one player.
    Otherwise returns total, wins, per-surface breakdown, and the
    last 5 meetings (most recent first).
    """
    for side, name in (("a", name_a), ("b", name_b)):
        candidates = find_players(conn, name)
        if not candidates:
            return None
        if len(candidates) > 1:
            return {
                "ambiguous": True,
                "side": side,
                "candidates": [
                    {"player_id": pid, "name": f"{first} {last}".strip()}
                    for pid, first, last in candidates
                ],
            }
        if side == "a":
            id_a, first_a, last_a = candidates[0]
        else:
            id_b, first_b, last_b = candidates[0]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tourney_date, tourney_name, surface, winner_id, score
            FROM atp_matches
            WHERE (winner_id = %s AND loser_id = %s)
               OR (winner_id = %s AND loser_id = %s)
            ORDER BY tourney_date, tourney_id, match_num
            """,
            (id_a, id_b, id_b, id_a),
        )
        meetings = cur.fetchall()
    rows = [(winner_id, surface) for _date, _name, surface, winner_id, _score in meetings]
    summary = _h2h_summary(rows, id_a, id_b)
    recent = [
        {
            "date": str(date) if date else None,
            "tournament": tourney_name,
            "surface": surface,
            "winner": f"{first_a} {last_a}".strip() if winner_id == id_a else f"{first_b} {last_b}".strip(),
            "score": score,
        }
        for date, tourney_name, surface, winner_id, score in reversed(meetings[-5:])
    ]
    return {
        "player_a": {"player_id": id_a, "name": f"{first_a} {last_a}".strip()},
        "player_b": {"player_id": id_b, "name": f"{first_b} {last_b}".strip()},
        "total": summary["total"],
        "wins_a": summary["wins_a"],
        "wins_b": summary["wins_b"],
        "by_surface": summary["by_surface"],
        "last_5_meetings": recent,
        "data_as_of": "Matches imported to 2026-05-25; head-to-head is not live.",
    }



# ---------------------------------------------------------------------------
# Tool 4: player surface record
# ---------------------------------------------------------------------------
# Career win-loss per surface, from atp_matches directly.
# This is different from Tool 1: Tool 1 gives surface Elo (a strength
# rating), Tool 4 gives the plain record (how many won, how many lost).
# A player can have a good surface record but a modest surface Elo
# if those wins came against weaker opponents, and vice versa.
# Surface NULL in the data is reported as "Unknown", not dropped,
# so the per-surface totals sum to the player's total matches.


def _win_pct(wins, total):
    if total == 0:
        return None
    return round(wins / total, 4)


def get_player_surface_record(conn, name):
    """Return career win-loss per surface for one player.

    Returns None if no player matches.
    Returns {"ambiguous": True, "candidates": [...]} if more than one matches.
    Otherwise returns per-surface wins, losses, total, win pct, and
    the overall total, which should equal atp_elo_current.matches_played.
    """
    candidates = find_players(conn, name)
    if not candidates:
        return None
    if len(candidates) > 1:
        return {
            "ambiguous": True,
            "candidates": [
                {"player_id": pid, "name": f"{first} {last}".strip()}
                for pid, first, last in candidates
            ],
        }
    player_id, first_name, last_name = candidates[0]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT surface,
                   SUM(CASE WHEN winner_id = %s THEN 1 ELSE 0 END) AS wins,
                   COUNT(*) AS total
            FROM atp_matches
            WHERE winner_id = %s OR loser_id = %s
            GROUP BY surface
            ORDER BY total DESC
            """,
            (player_id, player_id, player_id),
        )
        rows = cur.fetchall()
    by_surface = {}
    total_matches = 0
    total_wins = 0
    for surface, wins, total in rows:
        key = surface if surface is not None else "Unknown"
        wins = int(wins)
        total = int(total)
        by_surface[key] = {
            "wins": wins,
            "losses": total - wins,
            "total": total,
            "win_pct": _win_pct(wins, total),
        }
        total_matches += total
        total_wins += wins
    return {
        "player_id": player_id,
        "name": f"{first_name} {last_name}".strip(),
        "by_surface": by_surface,
        "total_matches": total_matches,
        "total_wins": total_wins,
        "total_losses": total_matches - total_wins,
        "overall_win_pct": _win_pct(total_wins, total_matches),
        "data_as_of": "Matches imported to 2026-05-25; surface record is not live.",
    }



# ---------------------------------------------------------------------------
# Tool 5: match probability, final locked model
# ---------------------------------------------------------------------------
# Final locked model, 2026-10-28, tuning stops here:
# - Overall Elo: initial 1500, K=32 (atp_elo_current)
# - Surface Elo: per-surface Elo, K=32, blended with prior=100:
#       blended = w * surface_elo + (1 - w) * overall_elo
#       w = n_surface / (n_surface + 100)
#   Missing surface, or no matches on that surface, uses overall only.
# - Calibrated probability: p_cal = sigmoid(0.8 * logit(p_blended))
#   with b=0.0, so calibration is symmetric and does NOT change
#   who is the favorite, only how confident the number is.
# Form, games %, retirement, and head-to-head are descriptors only
# and are NOT inputs to this probability (Week 2 verdict).


def _logit(prob):
    prob = min(max(prob, 1e-9), 1.0 - 1e-9)
    return math.log(prob / (1.0 - prob))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def calibrated_prob(blended_prob, a=0.8, b=0.0):
    """Apply the locked calibration scaling to a blended probability."""
    return _sigmoid(a * _logit(blended_prob) + b)


def get_match_probability(conn, name_a, name_b, surface=None):
    """Return blended and calibrated win probabilities for player A.

    surface is one of Hard, Clay, Grass, Carpet, or None.
    None, or an unknown surface, means overall Elo only.
    Returns None if either name matches no player.
    Returns {"ambiguous": True, "side": "a" or "b", "candidates": [...]}
    if either side matches more than one player.
    """
    players = {}
    for side, name in (("a", name_a), ("b", name_b)):
        candidates = find_players(conn, name)
        if not candidates:
            return None
        if len(candidates) > 1:
            return {
                "ambiguous": True,
                "side": side,
                "candidates": [
                    {"player_id": pid, "name": f"{first} {last}".strip()}
                    for pid, first, last in candidates
                ],
            }
        players[side] = candidates[0]
    if surface is not None and surface not in SURFACES:
        surface = None
    result_players = {}
    blended_elos = {}
    for side in ("a", "b"):
        player_id, first_name, last_name = players[side]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT elo FROM atp_elo_current WHERE player_id = %s",
                (player_id,),
            )
            row = cur.fetchone()
        if row is None:
            return {
                "player_id": player_id,
                "name": f"{first_name} {last_name}".strip(),
                "overall_elo": None,
                "note": "Player exists in atp_players but has no main-tour Elo row.",
            }
        overall_elo = float(row[0])
        if surface is None:
            surf_elo, surf_n = None, 0
            blended = overall_elo
        else:
            surf_elo, surf_n = _surface_current(conn, player_id, surface)
            if surf_elo is None:
                blended = overall_elo
            elif blended_elo is None:
                blended = overall_elo
            else:
                blended = blended_elo(surf_elo, overall_elo, surf_n, PRIOR)
        blended_elos[side] = float(blended)
        result_players[side] = {
            "player_id": player_id,
            "name": f"{first_name} {last_name}".strip(),
            "overall_elo": round(overall_elo, 1),
            "surface": surface,
            "surface_elo": round(float(surf_elo), 1) if surf_elo is not None else None,
            "surface_matches": int(surf_n),
            "blended_elo": round(float(blended), 1),
        }
    # win_prob is defined in surface_elo.py; fall back to the same formula.
    try:
        from surface_elo import win_prob
    except ImportError:
        def win_prob(elo_a, elo_b):
            return 1.0 / (1.0 + 10.0 ** (-(float(elo_a) - float(elo_b)) / 400.0))
    p_blended = win_prob(blended_elos["a"], blended_elos["b"])
    p_cal = calibrated_prob(p_blended, a=0.8, b=0.0)
    favorite = "a" if p_cal >= 0.5 else "b"
    return {
        "player_a": result_players["a"],
        "player_b": result_players["b"],
        "surface": surface,
        "model": "K32 overall Elo + surface blend prior100 + calibration a=0.8, b=0.0",
        "prob_a_blended": round(float(p_blended), 4),
        "prob_b_blended": round(float(1.0 - p_blended), 4),
        "prob_a_calibrated": round(float(p_cal), 4),
        "prob_b_calibrated": round(float(1.0 - p_cal), 4),
        "favorite": result_players[favorite]["name"],
        "data_as_of": "Matches imported to 2026-05-25; probability is not live.",
    }


def main(argv):
    if len(argv) < 2:
        print('Usage: python scripts/tennis_tools.py "Jannik Sinner"')
        print('Usage: python scripts/tennis_tools.py form "Jannik Sinner"')
        print('Usage: python scripts/tennis_tools.py h2h "Jannik Sinner" "Carlos Alcaraz"')
        return 1
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a")
        return 1
    import psycopg

    if argv[1] == "form":
        if len(argv) < 3:
            print('Usage: python scripts/tennis_tools.py form "Jannik Sinner"')
            return 1
        name = argv[2]
        with psycopg.connect(database_url) as conn:
            result = get_player_form(conn, name)
        if result is None:
            print(f"No player found for: {name}")
            return 1
        if result.get("ambiguous"):
            print("Ambiguous name, please use a full name. Candidates:")
            for cand in result["candidates"]:
                print(f"  {cand['player_id']}: {cand['name']}")
            return 1
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if argv[1] == "h2h":
        if len(argv) < 4:
            print('Usage: python scripts/tennis_tools.py h2h "Jannik Sinner" "Carlos Alcaraz"')
            return 1
        with psycopg.connect(database_url) as conn:
            result = get_head_to_head(conn, argv[2], argv[3])
        if result is None:
            print(f"No player found for: {argv[2]} or {argv[3]}")
            return 1
        if result.get("ambiguous"):
            print(f"Ambiguous name on side {result['side']}, please use a full name. Candidates:")
            for cand in result["candidates"]:
                print(f"  {cand['player_id']}: {cand['name']}")
            return 1
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if argv[1] == "surface":
        if len(argv) < 3:
            print('Usage: python scripts/tennis_tools.py surface "Jannik Sinner"')
            return 1
        with psycopg.connect(database_url) as conn:
            result = get_player_surface_record(conn, argv[2])
        if result is None:
            print(f"No player found for: {argv[2]}")
            return 1
        if result.get("ambiguous"):
            print("Ambiguous name, please use a full name. Candidates:")
            for cand in result["candidates"]:
                print(f"  {cand['player_id']}: {cand['name']}")
            return 1
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if argv[1] == "match":
        if len(argv) < 4:
            print('Usage: python scripts/tennis_tools.py match "Jannik Sinner" "Carlos Alcaraz" "Hard"')
            return 1
        match_surface = argv[4] if len(argv) >= 5 else None
        with psycopg.connect(database_url) as conn:
            result = get_match_probability(conn, argv[2], argv[3], match_surface)
        if result is None:
            print(f"No player found for: {argv[2]} or {argv[3]}")
            return 1
        if result.get("ambiguous"):
            print(f"Ambiguous name on side {result['side']}, please use a full name. Candidates:")
            for cand in result["candidates"]:
                print(f"  {cand['player_id']}: {cand['name']}")
            return 1
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    name = argv[1]
    with psycopg.connect(database_url) as conn:
        result = get_player_elo(conn, name)
    if result is None:
        print(f"No player found for: {name}")
        return 1
    if result.get("ambiguous"):
        print("Ambiguous name, please use a full name. Candidates:")
        for cand in result["candidates"]:
            print(f"  {cand['player_id']}: {cand['name']}")
        return 1
    # If a surface was given as a second argument, show only that surface.
    if len(argv) >= 3:
        surface = argv[2]
        if surface in result["surfaces"]:
            result = {**result, "surfaces": {surface: result["surfaces"][surface]}}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
