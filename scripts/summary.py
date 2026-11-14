"""Week 9 Step 3: structured summary for the web UI.

Purpose: the web UI wants a gauge and a side list, not a long
paragraph it has to parse. The API already has every number in
the templated report dict, but only as prose in final_text and
as nested tool outputs in report["tools"]. This module copies
those numbers, unchanged, into one flat JSON-safe dict the
frontend can draw from, without parsing text and without
computing any tennis number.

What this file does, and only this:
  build_summary(report) takes the dict returned by
  build_scout_report in scripts/scout_report.py and returns a
  summary dict, or None if the report is missing, not a dict,
  or an error dict. It never calls the database, never calls
  an LLM, and never invents a number. Every value is copied
  with .get(), and is None when the source field is missing.
  No rounding, no percent conversion, no derived value, except
  where a field is already a computed value in the report
  (probabilities, Elo, win pct), which is copied as is.

Shape returned, for Sinner vs Alcaraz Hard:
  {
    "player_a": "Jannik Sinner",
    "player_b": "Carlos Alcaraz",
    "surface": "Hard",
    "surface_applied": True,
    "favorite": "Jannik Sinner",
    "prob_a_calibrated": 0.6378,
    "prob_b_calibrated": 0.3622,
    "prob_a_blended": 0.6698,
    "prob_b_blended": 0.3302,
    "data_as_of": "2026-05-25",
    "a": {
      "name": "Jannik Sinner",
      "elo": {"overall": 2283.1, "surface_elo": 2236.0,
              "blended": 2247.6, "surface_matches": 308,
              "matches_played": 457},
      "record": {"surface": {"wins": 251, "losses": 57,
                             "total": 308, "win_pct": 0.8149},
                 "total_matches": 457, "total_wins": 363,
                 "total_losses": 94},
      "form": {"win10": 0.9, "win10_pct": 0.9,
               "win10_wins": 9, "win10_n": 10,
               "win52": 0.9121, "win52_pct": 0.9121,
               "win52_wins": 83, "win52_n": 91,
               "games10": 0.6273, "games10_pct": 0.6273,
               "games10_won": 138, "games10_total": 220,
               "games52": 0.635, "games52_pct": 0.635,
               "games52_won": 1298, "games52_total": 2044,
               "last_retirement_date": "2025-10-01",
               "days_since_last_retirement": 236,
               "matches_since_last_retirement": 55,
               "retirements_last10": 0,
               "retirements_last52": 2,
               "retirements_total": 7},
    },
    "b": { ... same shape, for Alcaraz ... },
    "h2h": {"total": 17, "wins_a": 7, "wins_b": 10,
            "surface": {"total": 10, "wins_a": 3, "wins_b": 7}},
  }
  "a" and "b" are the per-player blocks. "win10" and
  "win10_pct" are the same value, kept under both names so a
  caller that expects either name finds it. "surface" inside
  "record" and "h2h" is None when no surface was specified,
  or when the player has no record on that surface.
  Elo surface fields come from tools["match"] first, because
  that is the surface-specific Elo for the requested surface,
  with tools["elo_a"] / tools["elo_b"] as the fallback for
  matches_played and overall Elo.

What this file does NOT do:
  It does not change the locked probability model, any tool,
  any verifier rule, or any writer rule. It does not turn a
  probability into a percent, and it does not compute a gauge
  angle, a bar width, or any other display value. Those are
  frontend presentation transforms, not part of this summary.
"""

from typing import Any, Dict, Optional


def _get(d, key, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def _player_elo(match_player, elo_tool):
    """Copy Elo fields for one player, defensively."""
    if not isinstance(match_player, dict):
        match_player = {}
    if not isinstance(elo_tool, dict):
        elo_tool = {}
    return {
        "overall": _get(match_player, "overall_elo", _get(elo_tool, "overall_elo")),
        "surface_elo": _get(match_player, "surface_elo"),
        "blended": _get(match_player, "blended_elo"),
        "surface_matches": _get(match_player, "surface_matches"),
        "matches_played": _get(elo_tool, "matches_played"),
    }


def _player_record(record_tool, surface):
    """Copy surface record and career totals for one player."""
    if not isinstance(record_tool, dict):
        record_tool = {}
    by_surface = _get(record_tool, "by_surface") or {}
    entry = by_surface.get(surface) if surface is not None else None
    if isinstance(entry, dict):
        surface_record = {
            "wins": _get(entry, "wins"),
            "losses": _get(entry, "losses"),
            "total": _get(entry, "total"),
            "win_pct": _get(entry, "win_pct"),
        }
    else:
        surface_record = None
    return {
        "surface": surface_record,
        "total_matches": _get(record_tool, "total_matches"),
        "total_wins": _get(record_tool, "total_wins"),
        "total_losses": _get(record_tool, "total_losses"),
    }


def _player_form(form_tool):
    """Copy form, games form, and retirement for one player."""
    if not isinstance(form_tool, dict):
        form_tool = {}
    win = _get(form_tool, "win_form") or {}
    games = _get(form_tool, "games_form") or {}
    ret = _get(form_tool, "retirement") or {}
    return {
        "win10": _get(win, "last10_pct"),
        "win10_pct": _get(win, "last10_pct"),
        "win10_wins": _get(win, "last10_wins"),
        "win10_n": _get(win, "last10_n"),
        "win2m": _get(win, "recent2m_pct"),
        "win2m_pct": _get(win, "recent2m_pct"),
        "win2m_wins": _get(win, "recent2m_wins"),
        "win2m_n": _get(win, "recent2m_n"),
        "win52": _get(win, "last52_pct"),
        "win52_pct": _get(win, "last52_pct"),
        "win52_wins": _get(win, "last52_wins"),
        "win52_n": _get(win, "last52_n"),
        "games10": _get(games, "last10_games_pct"),
        "games10_pct": _get(games, "last10_games_pct"),
        "games10_won": _get(games, "last10_games_won"),
        "games10_total": _get(games, "last10_games_total"),
        "games2m": _get(games, "recent2m_games_pct"),
        "games2m_pct": _get(games, "recent2m_games_pct"),
        "games2m_won": _get(games, "recent2m_games_won"),
        "games2m_total": _get(games, "recent2m_games_total"),
        "games52": _get(games, "last52_games_pct"),
        "games52_pct": _get(games, "last52_games_pct"),
        "games52_won": _get(games, "last52_games_won"),
        "games52_total": _get(games, "last52_games_total"),
        "last_retirement_date": _get(ret, "last_retirement_date"),
        "days_since_last_retirement": _get(ret, "days_since_last_retirement"),
        "matches_since_last_retirement": _get(ret, "matches_since_last_retirement"),
        "retirements_last10": _get(ret, "retirements_last10"),
        "retirements_last52": _get(ret, "retirements_last52"),
        "retirements_total": _get(ret, "retirements_total"),
    }


def build_summary(report):
    """Build a JSON-safe summary from a templated report dict.

    Returns None if report is not a dict, or if it is an error
    dict. Otherwise returns the summary dict described in the
    module docstring. All values are copied, never computed.
    """
    if not isinstance(report, dict):
        return None
    if "error" in report:
        return None
    tools = _get(report, "tools") or {}
    if not isinstance(tools, dict):
        tools = {}
    match = _get(tools, "match") or {}
    if not isinstance(match, dict):
        match = {}
    match_a = _get(match, "player_a") or {}
    match_b = _get(match, "player_b") or {}
    name_a = _get(match_a, "name", _get(_get(tools, "elo_a") or {}, "name"))
    name_b = _get(match_b, "name", _get(_get(tools, "elo_b") or {}, "name"))
    surface = _get(report, "surface", _get(match, "surface"))
    h2h = _get(tools, "h2h") or {}
    if not isinstance(h2h, dict):
        h2h = {}
    h2h_by_surface = _get(h2h, "by_surface") or {}
    h2h_surface = h2h_by_surface.get(surface) if surface is not None else None
    if isinstance(h2h_surface, dict):
        h2h_surface_out = {
            "total": _get(h2h_surface, "total"),
            "wins_a": _get(h2h_surface, "wins_a"),
            "wins_b": _get(h2h_surface, "wins_b"),
        }
    else:
        h2h_surface_out = None
    return {
        "player_a": name_a,
        "player_b": name_b,
        "surface": surface,
        "surface_applied": _get(report, "surface_applied"),
        "favorite": _get(report, "favorite", _get(match, "favorite")),
        "prob_a_calibrated": _get(report, "prob_a_calibrated", _get(match, "prob_a_calibrated")),
        "prob_b_calibrated": _get(report, "prob_b_calibrated", _get(match, "prob_b_calibrated")),
        "prob_a_blended": _get(report, "prob_a_blended", _get(match, "prob_a_blended")),
        "prob_b_blended": _get(report, "prob_b_blended", _get(match, "prob_b_blended")),
        "data_as_of": _get(report, "data_as_of"),
        "a": {
            "name": name_a,
            "elo": _player_elo(match_a, _get(tools, "elo_a")),
            "record": _player_record(_get(tools, "record_a"), surface),
            "form": _player_form(_get(tools, "form_a")),
        },
        "b": {
            "name": name_b,
            "elo": _player_elo(match_b, _get(tools, "elo_b")),
            "record": _player_record(_get(tools, "record_b"), surface),
            "form": _player_form(_get(tools, "form_b")),
        },
        "h2h": {
            "total": _get(h2h, "total"),
            "wins_a": _get(h2h, "wins_a"),
            "wins_b": _get(h2h, "wins_b"),
            "surface": h2h_surface_out,
        },
    }
