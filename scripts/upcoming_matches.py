"""Week 9: live upcoming tennis matches from Polymarket.

Purpose: feed the "upcoming matches" strip on the web page
with real, soon-to-start matches, so a tap loads a matchup
that actually has a market. Read-only, no key, no trading.

Source: Polymarket Gamma events feed, ordered by startDate
so the soonest upcoming matches come first:
  https://gamma-api.polymarket.com/events
  ?limit=20&active=true&closed=false&tag_slug=tennis
  &order=startDate&ascending=false
The unordered tag list is dominated by old ITF matches, so
the ordering matters. The returned page is still sorted
client-side by start time, so the soonest matches win even
if the feed order drifts.

Rules, non-negotiable:
  - Read-only. No API key, no wallet, no trading.
  - No caching to disk, no storing in a database.
  - Timeout 10 seconds per HTTP call.
  - Only match-winner markets: the market's outcomes must
    be exactly the two player names, in either order, and
    the title must carry no prop marker (same guard as
    market_odds.py). Doubles events are skipped: a doubles
    pair is not a singles matchup the report can analyse.
  - Closed/resolved/settled markets are filtered out.
  - Never invent a number. A match with no prices returns
    polymarket None, and the UI must show "no quote".
  - Any HTTP error, timeout, or unrecognised shape returns
    an empty list. These functions never raise.
"""

import urllib.parse
from datetime import datetime, timezone

from market_odds import (
    TIMEOUT_SECONDS,
    _as_float,
    _default_http_get,
    _is_closed,
    _is_match_winner,
    _match_winner_probs,
)

GAMMA_EVENTS_URL = (
    "https://gamma-api.polymarket.com/events"
    "?limit=20&active=true&closed=false&tag_slug=tennis"
    "&order=startDate&ascending=false"
)
MAX_MATCHES = 4


def _parse_start(value):
    """Return an aware datetime for an ISO start time, or None."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_match_title(title):
    """Return (tournament, player_a, player_b) or None.

    Titles look like "Knoxville: Nicolas Mejia vs Max Hans
    Rehberg". The tournament is the part before the first
    ": ". The players are the two sides of " vs ". A title
    with no ": " is still accepted, with tournament None.
    Doubles titles ("(Doubles)" or a "/" in a name) return
    None, they are not singles matchups.
    """
    if not isinstance(title, str) or not title.strip():
        return None
    if "doubles" in title.lower():
        return None
    head, sep, rest = title.partition(": ")
    matchup = rest if sep else title
    lowered = matchup.lower()
    idx = lowered.find(" vs ")
    if idx < 0:
        return None
    player_a = matchup[:idx].strip()
    player_b = matchup[idx + 4:].strip()
    if not player_a or not player_b:
        return None
    if "/" in player_a or "/" in player_b:
        return None
    tournament = head.strip() if sep else None
    return (tournament, player_a, player_b)


def get_upcoming_matches(http_get=None, limit=MAX_MATCHES):
    """Return the soonest upcoming match-winner matches.

    Returns a list of at most `limit` dicts, soonest first:
      {
        "tournament": "Knoxville" or None,
        "player_a": "Nicolas Mejia",
        "player_b": "Max Hans Rehberg",
        "start_time": "2026-11-10T00:06:26Z",
        "polymarket": {"prob_a": 0.485, "prob_b": 0.515,
                       "volume": 1200.0} or None,
      }
    A match whose market has no usable price is still
    returned, with polymarket None. Returns [] on any HTTP
    error, timeout, or unrecognised response shape. Never
    raises.
    """
    if http_get is None:
        http_get = _default_http_get
    try:
        payload = http_get(GAMMA_EVENTS_URL)
    except Exception:
        return []
    try:
        if isinstance(payload, list):
            events = payload
        elif isinstance(payload, dict):
            events = payload.get("events") or []
        else:
            return []
        if not isinstance(events, list):
            return []
        found = []
        for event in events:
            if not isinstance(event, dict):
                continue
            if _is_closed(event):
                continue
            start = _parse_start(event.get("startDate"))
            if start is None:
                continue
            parsed = _parse_match_title(event.get("title") or "")
            if parsed is None:
                continue
            tournament, player_a, player_b = parsed
            match = None
            for market in event.get("markets") or []:
                if not isinstance(market, dict):
                    continue
                if _is_closed(market):
                    continue
                question = market.get("question") or market.get("title") or ""
                if not _is_match_winner(market, question, player_a, player_b):
                    continue
                match = market
                break
            if match is None:
                continue
            probs = _match_winner_probs(match, player_a, player_b, http_get)
            if probs is None:
                poly = None
            else:
                prob_a, prob_b = probs
                poly = {
                    "prob_a": round(prob_a, 4),
                    "prob_b": round(prob_b, 4),
                    "volume": _as_float(match.get("volume")),
                }
            found.append((
                start,
                {
                    "tournament": tournament,
                    "player_a": player_a,
                    "player_b": player_b,
                    "start_time": event.get("startDate"),
                    "polymarket": poly,
                },
            ))
        found.sort(key=lambda item: item[0])
        return [item for _start, item in found[:max(0, int(limit))]]
    except Exception:
        return []
