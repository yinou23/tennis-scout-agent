"""Week 9: live upcoming tennis matches from Polymarket.

Purpose: feed the "upcoming matches" strip on the web page
with real, soon-to-start matches, so a tap loads a matchup
that actually has a market. Read-only, no key, no trading.

Source: Polymarket Gamma events feed. Two facts about the
feed drive the design, both verified live 2026-11-15:

  1. The feed's `startDate` is the market creation time, not
     the match time. A big-tournament market is created days
     before the match: the ATP Finals singles markets were
     created 2026-11-13 for matches played 2026-11-15 to
     2026-11-22. The actual match time lives in the event's
     `startTime` (and the market's `gameStartTime`). So the
     match time is read from `startTime` first, falling back
     to `gameStartTime`, then to `startDate`.
  2. The feed is ordered by creation (`startDate`), newest
     first, and a busy ITF day floods it: on 2026-11-15 the
     newest 100 events were all created that day, and the
     ATP Finals events (created 2026-11-13) sat about 700
     events deep. A single page of 100 can never reach them.
     So the feed is paged (offset 0, 100, 200, ...) in
     parallel, up to _MAX_GAMMA_PAGES, until enough ATP Tour
     events are collected. Pages are fetched concurrently;
     the strip is fetched once per page load, not polled.

The unordered tag list is dominated by old ITF matches, so
the ordering matters. The returned page is still sorted
client-side by start time, so the soonest matches win even
if the feed order drifts. The limit is 100, not 20: the
soonest 20 events are usually Challengers and ITF, so a
narrow window would never reach an ATP Tour event.

Tour level (user direction 2026-11-10): the strip shows
ATP Tour events, 250 and above, only. The feed does not
label the level: every tennis event sits under the same
"ATP" series, Challengers and ITF included, and the slug
starts with "atp-" for all of them. So the level is read
from the tournament name, the only signal the feed
carries. Tour events carry a proper event name (Nordic
Open, Paris Masters, Moselle Open, ...); Challengers and
ITF events are bare city names (Orleans, Knoxville, Lima)
or ITF codes (W15, M15, W35). _is_atp_tour_level encodes
that as a keyword rule. It is a heuristic, documented as
such: a Tour event whose name carries none of the keywords
would be missed and the strip would simply show fewer
matches, and a Challenger listed as "City Open" would slip
through. The strip would rather show fewer matches than a
Challenger, so the rule errs toward exclusion.

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

import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from market_odds import (
    TIMEOUT_SECONDS,
    _as_float,
    _default_http_get,
    _is_closed,
    _is_match_winner,
    _match_winner_probs,
)

_GAMMA_EVENTS_BASE = (
    "https://gamma-api.polymarket.com/events"
    "?limit=100&active=true&closed=false&tag_slug=tennis"
    "&order=startDate&ascending=false"
)
GAMMA_EVENTS_URL = _GAMMA_EVENTS_URL = _GAMMA_EVENTS_BASE + "&offset=0"
# Pages of 100 to scan for ATP Tour events. The ATP Finals
# sat about 700 events deep on 2026-11-15 (created Nov 13,
# buried under ITF markets created Nov 14-15), so 10 pages
# (1000 events) gives headroom. Pages fetch concurrently.
_MAX_GAMMA_PAGES = 10
_GAMMA_PAGE_SIZE = 100
MAX_MATCHES = 4


def _gamma_page_url(offset):
    """Gamma events feed URL for one page (offset in events)."""
    return _GAMMA_EVENTS_BASE + f"&offset={offset}"


# ATP Tour level, 250 and above only. See the module docstring
# for why the level is read from the tournament name.
_ATP_LEVEL_KEYWORDS = (
    "open", "masters", "finals", "championship", "championships",
    "cup", "classic", "trophy", "international", "grand prix",
    "slam", "500", "1000", "250",
)
# The two Slams whose names carry no keyword above.
_ATP_SLAMS_BY_NAME = ("roland garros", "wimbledon")
# Team and exhibition events that carry a keyword but are not
# ATP Tour singles events. "wta" is here too: the WTA Finals
# carries "finals" but is a WTA event, and the TSA model only
# covers ATP players, so a WTA matchup would fail the report.
_ATP_TEAM_EXCLUSIONS = (
    "davis cup", "laver cup", "united cup", "hopman cup",
    "billie jean", "wta",
)
_ITF_CODE_RE = re.compile(r"^[WM]\d+", re.IGNORECASE)


def _is_atp_tour_level(tournament):
    """True when the tournament name is an ATP Tour event, 250+.

    Excludes ITF codes (W15, M15, W35, ...) and team events
    (Davis Cup, Laver Cup, United Cup, ...). Includes a name
    that carries an event keyword (Open, Masters, Finals, ...),
    or one of the two Slams named without a keyword (Roland
    Garros, Wimbledon). A bare city name (Orleans, Knoxville,
    Lima, Monastir) or anything unrecognised returns False.
    A missing or blank name returns False.
    """
    if not isinstance(tournament, str) or not tournament.strip():
        return False
    name = tournament.strip().lower()
    if _ITF_CODE_RE.match(name):
        return False
    if any(ex in name for ex in _ATP_TEAM_EXCLUSIONS):
        return False
    if name in _ATP_SLAMS_BY_NAME:
        return True
    return any(kw in name for kw in _ATP_LEVEL_KEYWORDS)


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


def _match_start_time(event, market):
    """Return (datetime, iso_string) for the actual match time.

    Prefers the event's `startTime` (the scheduled match
    start), then the market's `gameStartTime`, then the
    event's `startDate` (market creation, the old behaviour).
    Returns (None, None) when none of them parses.
    """
    market = market or {}
    for value in (
        event.get("startTime"),
        market.get("gameStartTime"),
        event.get("startDate"),
    ):
        start = _parse_start(value)
        if start is not None:
            return start, value
    return None, None


def _fetch_gamma_pages(http_get, pages):
    """Fetch Gamma feed pages concurrently, return event list.

    Each page is one offset. A page that errors or has an
    unrecognised shape contributes nothing; the rest still
    count. Never raises.
    """
    urls = [_gamma_page_url(i * _GAMMA_PAGE_SIZE) for i in range(pages)]
    events = []
    try:
        with ThreadPoolExecutor(max_workers=min(pages, 8)) as pool:
            results = list(pool.map(_safe_fetch_page, [(http_get, u) for u in urls]))
    except Exception:
        return []
    for payload in results:
        if isinstance(payload, list):
            events.extend(payload)
        elif isinstance(payload, dict):
            events.extend(payload.get("events") or [])
    return events


def _safe_fetch_page(args):
    """Fetch one Gamma page, return the payload or None."""
    http_get, url = args
    try:
        return http_get(url)
    except Exception:
        return None


def get_upcoming_matches(http_get=None, limit=MAX_MATCHES):
    """Return the soonest upcoming match-winner matches.

    Returns a list of at most `limit` dicts, soonest first:
      {
        "tournament": "Knoxville" or None,
        "player_a": "Nicolas Mejia",
        "player_b": "Max Hans Rehberg",
        "start_time": "2026-11-15T19:35:00Z",
        "polymarket": {"prob_a": 0.485, "prob_b": 0.515,
                       "volume": 1200.0} or None,
      }
    `start_time` is the actual scheduled match time (event
    `startTime`, falling back to market `gameStartTime`, then
    to `startDate`). A match whose market has no usable price
    is still returned, with polymarket None. Returns [] on
    any HTTP error, timeout, or unrecognised response shape.
    Never raises.
    """
    if http_get is None:
        http_get = _default_http_get
    try:
        events = _fetch_gamma_pages(http_get, _MAX_GAMMA_PAGES)
        if not isinstance(events, list):
            return []
        found = []
        seen = set()
        for event in events:
            if not isinstance(event, dict):
                continue
            if _is_closed(event):
                continue
            parsed = _parse_match_title(event.get("title") or "")
            if parsed is None:
                continue
            tournament, player_a, player_b = parsed
            if not _is_atp_tour_level(tournament):
                continue
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
            start, start_iso = _match_start_time(event, match)
            if start is None:
                continue
            # Dedupe: the same event can appear on several
            # pages (and a test stub returns one payload for
            # every URL). Key on the event id when present,
            # else on the title.
            key = event.get("id") or event.get("title") or ""
            if key in seen:
                continue
            seen.add(key)
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
                    "start_time": start_iso,
                    "polymarket": poly,
                },
            ))
        found.sort(key=lambda item: item[0])
        return [item for _start, item in found[:max(0, int(limit))]]
    except Exception:
        return []
