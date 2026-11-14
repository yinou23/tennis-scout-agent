"""Retrieve database facts for an open-ended tennis question.

Given a question and optional form context (player_a, player_b,
surface), find the players mentioned and fetch their Elo, form,
surface record, and (for two players) head-to-head, match
probability and market odds. Returns a facts dict and a text
context block for the LLM. Every number here comes from a tool
(database, model, or market), never from the LLM.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from market_odds import get_market_odds  # noqa: E402
from tennis_tools import (  # noqa: E402
    find_players,
    get_head_to_head,
    get_match_probability,
    get_player_elo,
    get_player_form,
    get_player_surface_record,
)


def _resolve(conn, name):
    """Return the full name for a player name/surname, or None."""
    if not name or not str(name).strip():
        return None
    found = find_players(conn, str(name).strip())
    if len(found) != 1:
        return None
    _pid, first, last = found[0]
    return f"{first} {last}".strip()


def extract_players(conn, question, context=None):
    """Return a list of resolved full player names mentioned.

    Form context players come first (they are the matchup the
    user filled in). Then names found in the question text, by
    trying single words and consecutive word pairs against the
    player table. Duplicates removed, order kept.
    """
    seen = []
    ctx = context if isinstance(context, dict) else {}
    for key in ("player_a", "player_b"):
        name = _resolve(conn, ctx.get(key))
        if name and name not in seen:
            seen.append(name)
    text = (question or "").strip()
    if not text:
        return seen
    # Tokens: split on anything that is not a letter. Chinese
    # characters split the text, leaving English names intact.
    tokens = [t for t in re.split(r"[^A-Za-z]+", text) if t]
    candidates = []
    for i, tok in enumerate(tokens):
        candidates.append(tok)
        if i + 1 < len(tokens):
            candidates.append(f"{tok} {tokens[i + 1]}")
    for cand in candidates:
        name = _resolve(conn, cand)
        if name and name not in seen:
            seen.append(name)
    return seen


def _pct(x):
    return round(float(x) * 100, 1) if x is not None else None


def retrieve_facts(conn, question, context=None):
    """Return facts for the question: players, matchup, market.

    Returns {"players": [player fact dicts], "match": {...} or
    None, "market": {...} or None, "context_text": str}. The
    context text is the block handed to the LLM; it contains
    only tool numbers, each labelled with its source.
    """
    ctx = context if isinstance(context, dict) else {}
    surface = (ctx.get("surface") or "").strip() or None
    # Surface named in the question wins over the form surface.
    q = (question or "").lower()
    for surf, kws in (("Hard", ["硬地", "hard"]), ("Clay", ["红土", "clay"]),
                      ("Grass", ["草地", "grass"]), ("Carpet", ["地毯", "carpet"])):
        if any(k in q for k in kws):
            surface = surf
            break
    names = extract_players(conn, question, context)
    players = []
    lines = []
    for name in names:
        elo = get_player_elo(conn, name)
        form = get_player_form(conn, name)
        rec = get_player_surface_record(conn, name)
        if elo is None or form is None or rec is None:
            continue
        if isinstance(elo, dict) and elo.get("ambiguous"):
            continue
        if isinstance(form, dict) and form.get("ambiguous"):
            continue
        if isinstance(rec, dict) and rec.get("ambiguous"):
            continue
        wf = form.get("win_form") or {}
        by_surface = rec.get("by_surface") or {}
        surf_rec = by_surface.get(surface) if surface else None
        surf_elo = (elo.get("surfaces") or {}).get(surface) if surface else None
        fact = {
            "name": elo.get("name") or name,
            "overall_elo": elo.get("overall_elo"),
            "matches_played": elo.get("matches_played"),
            "last_match_date": elo.get("last_match_date"),
            "recent2m_wins": wf.get("recent2m_wins"),
            "recent2m_n": wf.get("recent2m_n"),
            "recent2m_pct": _pct(wf.get("recent2m_pct")),
            "last10_wins": wf.get("last10_wins"),
            "last10_n": wf.get("last10_n"),
            "last10_pct": _pct(wf.get("last10_pct")),
            "surface": surface,
            "surface_wins": surf_rec.get("wins") if surf_rec else None,
            "surface_total": surf_rec.get("total") if surf_rec else None,
            "surface_win_pct": _pct(surf_rec.get("win_pct")) if surf_rec else None,
            "surface_elo": round(float(surf_elo["surface_elo"]), 1) if surf_elo and surf_elo.get("surface_elo") is not None else None,
        }
        players.append(fact)
        lines.append(
            f"- {fact['name']}: overall Elo {fact['overall_elo']}, "
            f"matches {fact['matches_played']}, last match {fact['last_match_date']}."
        )
        if fact["recent2m_n"]:
            lines.append(
                f"  Recent 2 months: {fact['recent2m_wins']}/{fact['recent2m_n']} wins "
                f"({fact['recent2m_pct']}%)."
            )
        if fact["last10_n"]:
            lines.append(
                f"  Last 10 matches: {fact['last10_wins']}/{fact['last10_n']} wins "
                f"({fact['last10_pct']}%)."
            )
        if surface and surf_rec is not None:
            lines.append(
                f"  {surface} record: {fact['surface_wins']}/{fact['surface_total']} "
                f"wins ({fact['surface_win_pct']}%), surface Elo {fact['surface_elo']}."
            )
    match = None
    market = None
    if len(names) >= 2:
        a, b = names[0], names[1]
        prob = get_match_probability(conn, a, b, surface)
        if isinstance(prob, dict) and not prob.get("ambiguous"):
            name_a = prob["player_a"]["name"]
            name_b = prob["player_b"]["name"]
            match = {
                "player_a": name_a,
                "player_b": name_b,
                "surface": surface,
                "prob_a_calibrated": _pct(prob.get("prob_a_calibrated")),
                "prob_b_calibrated": _pct(prob.get("prob_b_calibrated")),
                "favorite": prob.get("favorite"),
            }
            lines.append(
                f"- Match {name_a} vs {name_b}"
                + (f" on {surface}" if surface else "")
                + f": model gives {name_a} {match['prob_a_calibrated']}%, "
                f"{name_b} {match['prob_b_calibrated']}%. "
                f"Favorite: {match['favorite']}."
            )
            h2h = get_head_to_head(conn, a, b)
            if isinstance(h2h, dict) and not h2h.get("ambiguous"):
                ha = (h2h.get("player_a") or {}).get("name") or name_a
                hb = (h2h.get("player_b") or {}).get("name") or name_b
                match["h2h"] = {"player_a": ha, "player_b": hb,
                                "wins_a": h2h.get("wins_a"), "wins_b": h2h.get("wins_b"),
                                "total": h2h.get("total")}
                lines.append(
                    f"  Head-to-head: {ha} {h2h.get('wins_a')}, "
                    f"{hb} {h2h.get('wins_b')} "
                    f"({h2h.get('total')} meetings)."
                )
            odds = get_market_odds(a, b)
            if isinstance(odds, dict):
                poly = odds.get("polymarket")
                kal = odds.get("kalshi")
                market = {
                    "polymarket": {"prob_a": _pct(poly.get("prob_a")), "prob_b": _pct(poly.get("prob_b"))} if poly else None,
                    "kalshi": {"prob_a": _pct(kal.get("prob_a")), "prob_b": _pct(kal.get("prob_b"))} if kal else None,
                }
                if poly or kal:
                    parts = []
                    if poly:
                        parts.append(f"Polymarket {name_a} {_pct(poly.get('prob_a'))}%, {name_b} {_pct(poly.get('prob_b'))}%")
                    if kal:
                        parts.append(f"Kalshi {name_a} {_pct(kal.get('prob_a'))}%, {name_b} {_pct(kal.get('prob_b'))}%")
                    lines.append("- Market: " + "; ".join(parts) + ".")
                    for side, key, pname, model_pct in (
                        ("a", "prob_a", name_a, match["prob_a_calibrated"]),
                        ("b", "prob_b", name_b, match["prob_b_calibrated"]),
                    ):
                        vals = []
                        if poly and poly.get(key) is not None:
                            vals.append(float(poly[key]))
                        if kal and kal.get(key) is not None:
                            vals.append(float(kal[key]))
                        if vals:
                            avg = sum(vals) / len(vals)
                            diff = round((avg - model_pct / 100.0) * 100, 1)
                            lines.append(
                                f"  {pname}: market average {_pct(avg)}% vs model "
                                f"{model_pct}%, diff {diff:+.1f} points "
                                f"(market minus model)."
                            )
    context_text = "\n".join(lines) if lines else "No database facts found for the players named."
    return {
        "players": players,
        "match": match,
        "market": market,
        "context_text": context_text,
    }
