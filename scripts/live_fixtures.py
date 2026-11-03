"""Week 7 Step 3: fetch a one-time live or upcoming snapshot from Live Tennis API.

Scope, agreed in the Week 7 Step 2 research note and the user questions
that followed it:
- This script does NOT update the Sackmann database, and it does NOT
  recompute Elo, form, games, or head-to-head. The historical TSA data
  remains as of 2026-05-25. This file is a separate, point-in-time
  snapshot, with its own fetched_at timestamp, and it must never be
  presented as a live feed or as updated historical data.
- FREE tier only. Only status=live and status=upcoming are fetched.
  status=completed and status=cancelled are refused before any HTTP
  call, because on a FREE key they belong to the paid History product
  and would spend the 20 free history calls per calendar month.
- Fetch once and stop. No retry loop, no polling, no auto pagination.
  If meta.has_more is true, the snapshot records that fact and stops.
  The user runs again with --offset if a second page is wanted.
- On HTTP 429 or 403, stop with a clear message and no retry. On
  HTTP 401, stop and ask the user to check the key and email
  verification. On any other non-200, stop and print the status and
  a short body snippet. The API key is never printed and never
  written to the output file.
- Auth, verified 2026-11-02 from the official Live Tennis API docs:
  base URL https://api.livetennisapi.com/api/public/v1, key in the
  X-API-Key header, GET /matches?status=live&limit=200, response
  {data, meta}, each request at most 200 matches, empty data can
  mean no matches are live at fetch time, not a failure.
- Match fields, verified from the same docs: id, tournament,
  tournament_id, tour (atp, wta, challenger, itf, juniors, or null),
  surface (hard, clay, grass, or null, provider lowercase, kept
  verbatim, never converted to the Sackmann capitalised form),
  players.p1.name, players.p2.name, status, round, round_code,
  scheduled_time, event_status, and an embedded score object with
  sets, games, points, server, is_tiebreak. Score games is null
  when withheld, and null never means 0-0.
- Score endpoint, verified from the same docs: GET
  /matches/{matchId}/score returns the current score object directly,
  FREE, and is a point-in-time snapshot. A 404 there means no score
  yet for an upcoming match, a cancelled match never played, or no
  recorded final, and it is not treated as a fatal error when the
  score is fetched as an optional extra.
- Rate limit headers, verified from the same docs: every response
  carries X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset.
  They are printed on every run and stored in the snapshot.

Usage, from the project root ~/TSA, after the user has created a
free key himself at livetennisapi.com/subscribe/free and added it
to ~/TSA/.env himself (never in chat, never as a CLI argument):
    set -a; source .env; set +a
    python scripts/live_fixtures.py --status live
    python scripts/live_fixtures.py --status upcoming --output live/fixtures.json
    python scripts/live_fixtures.py --status live --score-for-first 3

The --score-for-first option costs one extra call per match, so it
defaults to 0. The /matches listing already embeds the latest score,
so the extra score call is only for checking the score endpoint
shape, not for getting a score the listing lacks.
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://api.livetennisapi.com/api/public/v1"
ALLOWED_STATUS = ("live", "upcoming")
HISTORICAL_DATA_AS_OF = "2026-05-25"
SNAPSHOT_NOTE = (
    "Point-in-time snapshot, not a live feed. "
    "Historical TSA data remains as of 2026-05-25 and is not updated by this file. "
    "Provider player and match ids do not match Sackmann ids, so no join to "
    "Sackmann players, Elo, form, or head-to-head is made here."
)


class LiveTennisError(Exception):
    """A Live Tennis API call failed. The caller stops, it never retries."""


def rate_limit_from_headers(headers):
    """Return the three rate limit headers, or None for each if absent.

    headers is any mapping with a case-insensitive get, such as the
    headers object returned by urllib. Missing headers are None,
    never guessed.
    """
    return {
        "limit": headers.get("X-RateLimit-Limit"),
        "remaining": headers.get("X-RateLimit-Remaining"),
        "reset": headers.get("X-RateLimit-Reset"),
    }


def _request(url, api_key):
    """Make one HTTP GET, no retry, and return (body_dict, rate_limit).

    Raises LiveTennisError on any HTTP or JSON failure, with the
    status and a short body snippet. The API key is never included
    in the exception message.
    """
    request = urllib.request.Request(
        url, headers={"X-API-Key": api_key, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            rate_limit = rate_limit_from_headers(response.headers)
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            snippet = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            snippet = ""
        if exc.code == 429:
            raise LiveTennisError(
                "Stopped on HTTP 429 rate limit, no retry. "
                "Check the X-RateLimit-Reset header and the daily quota "
                "(FREE: 100/day, 30/minute) before running again. "
                "Body snippet: %s" % snippet
            ) from exc
        if exc.code == 403:
            raise LiveTennisError(
                "Stopped on HTTP 403, no retry. This usually means the "
                "endpoint or tier is not unlocked for this key, or the "
                "free history taste is used. Body snippet: %s" % snippet
            ) from exc
        if exc.code == 401:
            raise LiveTennisError(
                "Stopped on HTTP 401. Check that LIVE_TENNIS_API_KEY is "
                "set, that the key is correct, and that the email "
                "verification for the free key is complete. "
                "Body snippet: %s" % snippet
            ) from exc
        raise LiveTennisError(
            "Stopped on HTTP %s, no retry. Body snippet: %s" % (exc.code, snippet)
        ) from exc
    except urllib.error.URLError as exc:
        raise LiveTennisError("Stopped on network error, no retry: %s" % exc) from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise LiveTennisError("Stopped, response was not valid JSON: %s" % exc) from exc
    if not isinstance(body, dict):
        raise LiveTennisError("Stopped, response JSON was not an object.")
    return body, rate_limit


def fetch_matches(api_key, status, limit=200, offset=0):
    """Fetch one page of matches for a FREE status, once, no retry.

    Returns (data_list, meta_dict, rate_limit_dict). Refuses
    completed and cancelled before any HTTP call. An empty data
    list is a valid result, not a failure.
    """
    if status not in ALLOWED_STATUS:
        raise ValueError(
            "status %r is not a FREE status for this script. "
            "Only live and upcoming are fetched. completed and "
            "cancelled belong to the paid History product." % status
        )
    if not isinstance(limit, int) or not 1 <= limit <= 200:
        raise ValueError("limit must be an integer from 1 to 200.")
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be an integer of 0 or more.")
    query = urllib.parse.urlencode(
        {"status": status, "limit": limit, "offset": offset}
    )
    url = "%s/matches?%s" % (BASE_URL, query)
    body, rate_limit = _request(url, api_key)
    data = body.get("data")
    if not isinstance(data, list):
        raise LiveTennisError("Stopped, response data was not a list.")
    meta = body.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    return data, meta, rate_limit


def fetch_score(api_key, match_id):
    """Fetch the current score for one match, or None on a 404.

    A 404 is the documented answer for an upcoming match with no
    score yet, and it is not a fatal error here. Other HTTP errors
    raise LiveTennisError and stop, with no retry.
    """
    url = "%s/matches/%s/score" % (BASE_URL, match_id)
    request = urllib.request.Request(
        url, headers={"X-API-Key": api_key, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise LiveTennisError(
            "Stopped on HTTP %s from the score endpoint, no retry." % exc.code
        ) from exc
    except urllib.error.URLError as exc:
        raise LiveTennisError("Stopped on network error, no retry: %s" % exc) from exc


def _player_name(players, key):
    """Return a player name, or None if absent. Never invent a name."""
    if not isinstance(players, dict):
        return None
    player = players.get(key)
    if not isinstance(player, dict):
        return None
    name = player.get("name")
    return name if isinstance(name, str) and name else None


def _score_fields(score):
    """Return the five score fields, or None for each if no score.

    games is null when withheld, and that null is preserved, never
    converted to 0-0 or to an empty list.
    """
    if not isinstance(score, dict):
        return {
            "sets": None,
            "games": None,
            "points": None,
            "server": None,
            "is_tiebreak": None,
        }
    return {
        "sets": score.get("sets"),
        "games": score.get("games"),
        "points": score.get("points"),
        "server": score.get("server"),
        "is_tiebreak": score.get("is_tiebreak"),
    }


def normalize_match(raw):
    """Convert one provider match object to a TSA fixture dict.

    Provider ids, names, tournament, tour, and surface are kept
    verbatim. No Sackmann join, no Elo, no form, no head-to-head,
    and no surface conversion is made here.
    """
    if not isinstance(raw, dict):
        raise ValueError("Match object must be a dict.")
    players = raw.get("players")
    return {
        "provider_match_id": raw.get("id"),
        "player_p1": _player_name(players, "p1"),
        "player_p2": _player_name(players, "p2"),
        "tournament": raw.get("tournament"),
        "tournament_id": raw.get("tournament_id"),
        "tour": raw.get("tour"),
        "surface": raw.get("surface"),
        "status": raw.get("status"),
        "round": raw.get("round"),
        "round_code": raw.get("round_code"),
        "scheduled_time": raw.get("scheduled_time"),
        "event_status": raw.get("event_status"),
        "score": _score_fields(raw.get("score")),
    }


def build_snapshot(status, matches_raw, meta, rate_limit, endpoint):
    """Build the snapshot dict. Never includes the API key."""
    fixtures = [normalize_match(raw) for raw in matches_raw]
    fetched_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "fetched_at": fetched_at,
        "source": "Live Tennis API",
        "tier": "FREE",
        "endpoint": endpoint,
        "status_requested": status,
        "historical_data_as_of": HISTORICAL_DATA_AS_OF,
        "snapshot_note": SNAPSHOT_NOTE,
        "count": len(fixtures),
        "meta": meta,
        "has_more": meta.get("has_more") if isinstance(meta, dict) else None,
        "rate_limit": rate_limit,
        "fixtures": fixtures,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch one Live Tennis API snapshot, FREE tier only."
    )
    parser.add_argument("--status", default="live", choices=ALLOWED_STATUS)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", default="live/fixtures.json")
    parser.add_argument(
        "--score-for-first",
        type=int,
        default=0,
        help="Extra score calls for the first N matches, default 0.",
    )
    args = parser.parse_args(argv)

    import os

    api_key = os.environ.get("LIVE_TENNIS_API_KEY")
    if not api_key:
        print(
            "LIVE_TENNIS_API_KEY is not set. Create a free key yourself "
            "at livetennisapi.com/subscribe/free, add it to ~/TSA/.env "
            "yourself, then run: set -a; source .env; set +a",
            file=sys.stderr,
        )
        return 1

    endpoint = "/matches?status=%s&limit=%s&offset=%s" % (
        args.status,
        args.limit,
        args.offset,
    )
    try:
        matches_raw, meta, rate_limit = fetch_matches(
            api_key, args.status, limit=args.limit, offset=args.offset
        )
    except (ValueError, LiveTennisError) as exc:
        print("Stopped: %s" % exc, file=sys.stderr)
        return 1

    print("Endpoint: %s" % endpoint)
    print("Count: %s" % len(matches_raw))
    print("Rate limit: %s" % rate_limit)
    print("Meta: %s" % meta)
    if matches_raw:
        first = matches_raw[0]
        if isinstance(first, dict):
            print("First match keys: %s" % sorted(first.keys()))
            fixture = normalize_match(first)
            print(
                "First match: %s vs %s, tournament %s, status %s"
                % (
                    fixture["player_p1"],
                    fixture["player_p2"],
                    fixture["tournament"],
                    fixture["status"],
                )
            )
    else:
        print(
            "Empty data list. This can mean no matches are %s right now, "
            "not a failure. Try --status upcoming to inspect fixtures."
            % args.status
        )

    snapshot = build_snapshot(
        args.status, matches_raw, meta, rate_limit, endpoint
    )

    if args.score_for_first > 0:
        for fixture in snapshot["fixtures"][: args.score_for_first]:
            match_id = fixture["provider_match_id"]
            if match_id is None:
                continue
            try:
                score = fetch_score(api_key, match_id)
            except LiveTennisError as exc:
                print("Stopped: %s" % exc, file=sys.stderr)
                return 1
            if score is not None:
                fixture["score"] = _score_fields(score)
                fixture["score_source"] = "matches/{id}/score"
            else:
                fixture["score_source"] = "matches listing, score endpoint 404"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Wrote %s" % output_path)
    print("Snapshot note: %s" % SNAPSHOT_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
