"""Week 9 Step 3: read-only market odds, Polymarket and Kalshi.

Purpose: show the crowd's priced probability next to, never
mixed with, the TSA model probability. The model probability
is Elo, historical, data as of 2026-05-25, not live. A market
probability is crowd money, live, and includes injuries, the
draw, fees, and liquidity. A thin market is noisy. If the two
disagree, that is a divergence to discuss, not an error in
either number. Show each market with its source and timestamp,
separately from the model number.

Rules, non-negotiable:
  - Read-only. No API key, no wallet, no trading, no orders.
  - No caching to disk, no storing market data in a database.
    Each call fetches live and returns it. Do not build a
    market history from these calls.
  - Timeout 10 seconds per HTTP call.
  - Closed markets are never served as live. A market with
    closed true, or a status of closed / resolved / settled,
    is filtered out before matching, however high its volume.
  - Matching: both player surnames (case-insensitive) must
    appear in the market question or title, otherwise return
    None. Never invent a market, never guess a slug or ticker.
  - The subject must be clear. A binary market asks
    "Will X beat Y?" Yes is a win for X; a market titled
    "<Name> wins" is also a Yes for that player. If the
    question does not say which player Yes is about, return
    None, do not guess.
  - A match-winner market is different from a Yes/No prop:
    its outcomes are the two player names (full names or
    surnames, either order), as in "Rolex Paris Masters:
    Taylor Fritz vs Carlos Alcaraz". Its prices map to the
    players in outcome order. Set-winner, handicap, spread,
    and other prop markets may also list the player names
    as outcomes, so a title carrying a prop marker
    ("Set Winner", "Handicap", "Spread", "Completed Match",
    "Total Sets", "O/U", "win 3:") is never a match-winner.
  - Ranking: an open match-winner market is always preferred
    over a prop market, whatever the volumes. Within one
    type, highest volume wins. Do not pick by raw volume
    alone across types: a resolved 3:0 set-score prop at
    high volume is not the matchup.
  - Any HTTP error, timeout, empty result, or unrecognised
    response shape returns None. These functions never raise
    to the API caller.

Polymarket, read endpoints, no key:
  Gamma API https://gamma-api.polymarket.com, search via
  /public-search, then CLOB midpoint if needed at
  https://clob.polymarket.com/midpoint?token_id=...
  Price 0 to 1 is the implied probability. Do not guess slugs
  and do not browse /markets blindly, it returns thousands of
  markets, most of them not tennis.

Kalshi, read endpoints, no key for market data:
  https://api.elections.kalshi.com/trade-api/v2/markets
  Price in cents, 1 to 99, is the implied probability of Yes.
  Trading needs a key, a signed request, and KYC, and is out
  of scope here. Kalshi Developer Agreement limits: the API is
  for a member's own trading, and it bars collecting, caching,
  aggregating, or storing API data for any other use, and bars
  sharing that data with third parties without Kalshi's written
  authorization. This module only fetches live for display,
  it does not store or aggregate, and it must not be used to
  build a stored market dataset. Kalshi sports contracts are
  blocked or in court in several US states, so a market that
  exists in the API may not be tradeable everywhere.

What this file does NOT do:
  It does not trade, does not place orders, and does not
  recommend a bet. It does not change the TSA model, and it
  does not feed a market price back into the model probability.
  A market that does not exist for a hypothetical matchup
  (for example Sinner vs Alcaraz on Hard, with no date and no
  tournament) returns None, and the UI must show "no market",
  not a made-up number.
"""

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

TIMEOUT_SECONDS = 10
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB = "https://clob.polymarket.com"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _surname(name):
    """Return the last token of a player name, lowercased."""
    if not isinstance(name, str):
        return ""
    parts = name.strip().split()
    return parts[-1].lower() if parts else ""


def _both_surnames_in(text, surname_a, surname_b):
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return bool(surname_a) and bool(surname_b) and surname_a in lowered and surname_b in lowered


def _yes_subject(text, player_a, player_b):
    """Return "a" or "b" if Yes is clearly a win for that player.

    Accepts "Will <name> beat <name>?", "Will <name> win?",
    "Will <name> defeat <name>?", and a bare "<name> wins"
    title (the Kalshi shape, for example "Taylor Fritz wins").
    Only the player named immediately after "Will", or the
    player named by a "<name> wins" title, is the Yes subject.
    Anything else returns None, so a "vs" title with Yes/No
    outcomes never guesses.
    """
    if not isinstance(text, str):
        return None
    lowered = text.lower()
    for side, name in (("a", player_a), ("b", player_b)):
        if not isinstance(name, str) or not name.strip():
            continue
        first = name.strip().lower()
        if lowered.startswith("will " + first) or ("will " + first + " ") in lowered:
            return side
        stripped = lowered.strip()
        if stripped == first + " wins" or stripped.startswith(first + " wins "):
            return side
    return None


def _is_closed(market):
    """Return True if a market is resolved and must not be served.

    Closed means closed true, an archived flag, or a status of
    closed / resolved / settled / expired. A market with no
    closed flag and no such status is treated as open. Note
    that Polymarket's Gamma payloads carry closed true even
    when active is still true, so active alone is not enough.
    """
    if not isinstance(market, dict):
        return True
    if market.get("closed") is True:
        return True
    if market.get("archived") is True:
        return True
    status = market.get("status")
    if isinstance(status, str) and status.strip().lower() in {
        "closed", "resolved", "settled", "expired",
    }:
        return True
    return False


def _parse_outcomes(market):
    """Return the outcomes list for a Polymarket market, or [].

    Gamma returns outcomes as a list, or as a JSON string like
    '["Taylor Fritz", "Carlos Alcaraz"]'. Anything else is [].
    """
    outcomes = market.get("outcomes")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except ValueError:
            return []
    if isinstance(outcomes, list):
        return [str(o).strip() for o in outcomes if isinstance(o, (str, int, float))]
    return []


def _polymarket_outcome_prices(market):
    """Return the Gamma outcomePrices list as floats, or [].

    outcomePrices is a list, or a JSON string like
    '["0.275", "0.725"]', in outcome order.
    """
    prices = market.get("outcomePrices")
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except ValueError:
            return []
    if isinstance(prices, list):
        return [_as_float(p) for p in prices]
    return []


def _polymarket_clob_midpoint(market, http_get):
    """Return the CLOB midpoint for the first token id, or None.

    The first token corresponds to the first outcome, so this
    is the price of outcomes[0]. Any failure returns None.
    """
    tokens = market.get("clobTokenIds")
    if isinstance(tokens, str):
        try:
            tokens = json.loads(tokens)
        except ValueError:
            return None
    if not isinstance(tokens, list) or not tokens:
        return None
    try:
        data = http_get(f"{POLYMARKET_CLOB}/midpoint?token_id={urllib.parse.quote(str(tokens[0]))}")
        if isinstance(data, dict):
            price = _as_float(data.get("mid") if data.get("mid") is not None else data.get("midpoint"))
            if price is not None and 0.0 <= price <= 1.0:
                return price
    except Exception:
        pass
    return None


# Words in a title that mark a prop market, never the match
# winner, even when its outcomes are the two player names.
# Examples: "Set 1 Winner: ...", "Set Handicap: ...",
# "Game Spread: ...", "Completed Match: ...".
_PROP_MARKERS = (
    "set winner", "set 1", "set 2", "set 3", "handicap", "spread",
    "completed match", "total sets", "o/u", "win 3:", "win 3-",
)


def _is_match_winner(market, question, player_a, player_b):
    """Return True if this is the match-winner market.

    Its outcomes must be exactly the two players, as full
    names or surnames, in either order, and its title must
    carry no prop marker. A "vs" title is not a rejection
    here: the outcomes, not the title, resolve the subject.
    A "vs" title with Yes/No outcomes, or with no outcomes
    at all, is not a match-winner.
    """
    outcomes = _parse_outcomes(market)
    if len(outcomes) != 2:
        return False
    if not isinstance(question, str):
        return False
    lowered_q = question.lower()
    if any(marker in lowered_q for marker in _PROP_MARKERS):
        return False
    names = {player_a.strip().lower(), player_b.strip().lower()}
    surnames = {_surname(player_a), _surname(player_b)}
    covered = set()
    for outcome in outcomes:
        outcome_lower = outcome.lower()
        if outcome_lower in names or outcome_lower in surnames:
            covered.add(outcome_lower)
        else:
            return False
    return len(covered) == 2


def _price_for_player(price_by_outcome, player_name):
    """Return the price of one player's outcome, or None.

    Matches by full name first, then by surname, so both
    ["Taylor Fritz", "Carlos Alcaraz"] and ["Alcaraz", "Fritz"]
    resolve.
    """
    full = player_name.strip().lower()
    if full in price_by_outcome:
        return price_by_outcome[full]
    surname = _surname(player_name)
    if surname and surname in price_by_outcome:
        return price_by_outcome[surname]
    return None


def _match_winner_probs(market, player_a, player_b, http_get):
    """Return (prob_a, prob_b) for a match-winner market, or None.

    Prefers Gamma outcomePrices, which give both sides at
    once in outcome order. Falls back to the CLOB midpoint
    of the first token, which is the price of outcomes[0],
    with the other side as 1 minus that.
    """
    outcomes = _parse_outcomes(market)
    if len(outcomes) != 2:
        return None
    prices = _polymarket_outcome_prices(market)
    if len(prices) == 2 and all(p is not None and 0.0 <= p <= 1.0 for p in prices):
        price_by_outcome = {
            outcomes[0].lower(): prices[0],
            outcomes[1].lower(): prices[1],
        }
    else:
        mid = _polymarket_clob_midpoint(market, http_get)
        if mid is None:
            return None
        price_by_outcome = {
            outcomes[0].lower(): mid,
            outcomes[1].lower(): 1.0 - mid,
        }
    prob_a = _price_for_player(price_by_outcome, player_a)
    prob_b = _price_for_player(price_by_outcome, player_b)
    if prob_a is None or prob_b is None:
        return None
    if not (0.0 <= prob_a <= 1.0 and 0.0 <= prob_b <= 1.0):
        return None
    return (prob_a, prob_b)


def _kalshi_event_matches(event_ticker, player_a, player_b):
    """Return True if a Kalshi event ticker covers both players.

    Kalshi tennis match markets are one per player, titled
    "<Name> wins", so the title alone never names both
    players. The event ticker carries both as 3-letter
    surname codes, for example KXATPMATCH-26NOV06FRIALC
    for Taylor Fritz vs Carlos Alcaraz. Both codes must
    appear in the ticker, and the title must still name
    one of the two players (checked separately).
    """
    if not isinstance(event_ticker, str):
        return False
    ticker = event_ticker.upper()
    code_a = _surname(player_a)[:3].upper()
    code_b = _surname(player_b)[:3].upper()
    return bool(code_a) and bool(code_b) and code_a in ticker and code_b in ticker


def _default_http_get(url, timeout=TIMEOUT_SECONDS):
    """Fetch a URL and return parsed JSON. Raises on HTTP error."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "TSA-market-odds/1.0 (read-only)"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise RuntimeError(f"http_error:{status}")
        return json.loads(response.read().decode("utf-8"))


def _as_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _polymarket_markets(payload):
    """Extract a flat market list from a Gamma search payload.

    Gamma /public-search may return a dict with "markets", or
    with "events" each holding "markets", or a plain list.
    Any other shape returns an empty list, not an error.
    """
    if isinstance(payload, list):
        return [m for m in payload if isinstance(m, dict)]
    if not isinstance(payload, dict):
        return []
    markets = payload.get("markets")
    if isinstance(markets, list):
        return [m for m in markets if isinstance(m, dict)]
    out = []
    events = payload.get("events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                for market in event.get("markets") or []:
                    if isinstance(market, dict):
                        out.append(market)
    return out


def _polymarket_price(market, http_get):
    """Return the Yes price 0 to 1 for a Gamma market, or None.

    Prefers the CLOB midpoint for the first token id, falls
    back to Gamma outcomePrices. outcomePrices is a JSON
    string like '["0.62", "0.38"]', first entry is Yes.
    """
    price = _polymarket_clob_midpoint(market, http_get)
    if price is not None:
        return price
    prices = _polymarket_outcome_prices(market)
    if prices:
        price = prices[0]
        if price is not None and 0.0 <= price <= 1.0:
            return price
    price = _as_float(market.get("lastTradePrice"))
    if price is not None and 0.0 <= price <= 1.0:
        return price
    return None


def get_polymarket_odds(player_a, player_b, http_get=None):
    """Return Polymarket odds for a matchup, or None.

    Searches Gamma /public-search for both surnames. Closed
    markets are filtered out before matching. An open
    match-winner market, whose outcomes are the two player
    names, is always preferred over a Yes/No prop market,
    whatever the volumes; within one type, highest volume
    wins. A match-winner returns its outcome prices mapped
    to the players in order; a prop returns its Yes price
    as the implied probability for player A, with
    prob_b = 1 - prob_a. Returns None when there is no
    matching open market, no clear subject, no price, or
    any HTTP error. Never raises.
    """
    if http_get is None:
        http_get = _default_http_get
    try:
        surname_a = _surname(player_a)
        surname_b = _surname(player_b)
        if not surname_a or not surname_b:
            return None
        query = urllib.parse.quote(f"{player_a} {player_b}")
        payload = http_get(f"{POLYMARKET_GAMMA}/public-search?q={query}")
        match_winners = []
        props = []
        for market in _polymarket_markets(payload):
            if _is_closed(market):
                continue
            question = market.get("question") or market.get("title") or ""
            if not _both_surnames_in(question, surname_a, surname_b):
                continue
            volume = _as_float(market.get("volume")) or 0.0
            if _is_match_winner(market, question, player_a, player_b):
                match_winners.append((volume, market))
                continue
            subject = _yes_subject(question, player_a, player_b)
            if subject is None:
                continue
            props.append((volume, market, subject))
        match_winners.sort(key=lambda item: item[0], reverse=True)
        for _volume, market in match_winners:
            probs = _match_winner_probs(market, player_a, player_b, http_get)
            if probs is None:
                continue
            prob_a, prob_b = probs
            slug = market.get("slug")
            return {
                "source": "polymarket",
                "market_type": "match_winner",
                "question": market.get("question") or market.get("title"),
                "slug": slug,
                "url": f"https://polymarket.com/event/{slug}" if slug else None,
                "prob_a": round(prob_a, 4),
                "prob_b": round(prob_b, 4),
                "volume": _as_float(market.get("volume")),
                "fetched_at": _now_iso(),
            }
        props.sort(key=lambda item: item[0], reverse=True)
        for _volume, market, subject in props:
            yes_price = _polymarket_price(market, http_get)
            if yes_price is None:
                continue
            if subject == "a":
                prob_a = yes_price
            else:
                prob_a = 1.0 - yes_price
            slug = market.get("slug")
            return {
                "source": "polymarket",
                "market_type": "prop",
                "question": market.get("question") or market.get("title"),
                "slug": slug,
                "url": f"https://polymarket.com/event/{slug}" if slug else None,
                "prob_a": round(prob_a, 4),
                "prob_b": round(1.0 - prob_a, 4),
                "volume": _as_float(market.get("volume")),
                "fetched_at": _now_iso(),
            }
        return None
    except Exception:
        return None


def _kalshi_price_cents(market):
    """Return the Yes price in cents 1 to 99 for a Kalshi market.

    Prefers the midpoint of yes_bid and yes_ask, then
    last_price. Fields may be in cents (last_price) or in
    dollars (yes_bid_dollars). A value over 1 is cents, a
    value 0 to 1 is dollars, converted to cents.
    """
    def _to_cents(value):
        price = _as_float(value)
        if price is None:
            return None
        if price > 1.0:
            return price
        return price * 100.0

    bid = _to_cents(market.get("yes_bid") if market.get("yes_bid") is not None else market.get("yes_bid_dollars"))
    ask = _to_cents(market.get("yes_ask") if market.get("yes_ask") is not None else market.get("yes_ask_dollars"))
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    last = _to_cents(market.get("last_price") if market.get("last_price") is not None else market.get("last_price_dollars"))
    if last is not None:
        return last
    if bid is not None:
        return bid
    if ask is not None:
        return ask
    return None


def get_kalshi_odds(player_a, player_b, http_get=None):
    """Return Kalshi odds for a matchup, or None.

    Reads the public /markets list. Closed markets are
    filtered out before matching, even though the query
    already asks for status open. A market titled
    "<Name> wins" is a Yes for that player; since that
    title names only one player, the event ticker must
    carry both players as 3-letter surname codes, for
    example KXATPMATCH-26NOV06FRIALC. When both a
    player-A market and a player-B market are present, the
    player-A market is preferred, so its price is used
    directly instead of inverting the other side; within
    one subject, highest volume wins. Returns the Yes price
    in cents divided by 100 as the implied probability for
    player A, with prob_b = 1 - prob_a. Returns None when
    there is no matching market, no clear Yes subject, no
    price, or any HTTP error. Never raises. Only the first
    page of markets is scanned, so a market beyond it is
    reported as no market, not searched further.
    """
    if http_get is None:
        http_get = _default_http_get
    try:
        surname_a = _surname(player_a)
        surname_b = _surname(player_b)
        if not surname_a or not surname_b:
            return None
        payload = http_get(f"{KALSHI_API}/markets?limit=100&status=open")
        markets = []
        if isinstance(payload, dict):
            markets = payload.get("markets") or []
        elif isinstance(payload, list):
            markets = payload
        candidates = []
        for market in markets:
            if not isinstance(market, dict):
                continue
            if _is_closed(market):
                continue
            title = market.get("title") or market.get("event_title") or ""
            subject = _yes_subject(title, player_a, player_b)
            if subject is None:
                continue
            if not _both_surnames_in(title, surname_a, surname_b) and not _kalshi_event_matches(
                market.get("event_ticker"), player_a, player_b
            ):
                continue
            volume = (
                _as_float(market.get("volume"))
                if market.get("volume") is not None
                else _as_float(market.get("volume_fp"))
            ) or 0.0
            candidates.append((volume, market, subject))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (0 if item[2] == "a" else 1, -item[0]))
        for _volume, market, subject in candidates:
            cents = _kalshi_price_cents(market)
            if cents is None or not 1.0 <= cents <= 99.0:
                continue
            yes_prob = cents / 100.0
            if subject == "a":
                prob_a = yes_prob
            else:
                prob_a = 1.0 - yes_prob
            return {
                "source": "kalshi",
                "question": market.get("title") or market.get("event_title"),
                "ticker": market.get("ticker"),
                "prob_a": round(prob_a, 4),
                "prob_b": round(1.0 - prob_a, 4),
                "price_cents": round(cents, 2),
                "volume": (
                    _as_float(market.get("volume"))
                    if market.get("volume") is not None
                    else _as_float(market.get("volume_fp"))
                ),
                "fetched_at": _now_iso(),
            }
        return None
    except Exception:
        return None


def get_market_odds(player_a, player_b, http_get=None):
    """Return market odds from both venues, or None per venue.

    Returns {"polymarket": {...} or None, "kalshi": {...} or
    None}. Each venue is tried independently, so one venue
    failing does not hide the other. Never raises.
    """
    if http_get is None:
        http_get = _default_http_get
    return {
        "polymarket": get_polymarket_odds(player_a, player_b, http_get=http_get),
        "kalshi": get_kalshi_odds(player_a, player_b, http_get=http_get),
    }
