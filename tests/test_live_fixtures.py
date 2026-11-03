"""Unit tests for Week 7 Step 3 live fixtures snapshot.

Run from the project root ~/TSA, after moving the staging files to
their canonical names:
    mv ~/Downloads/live_fixtures_w7step3_2026110204.py scripts/live_fixtures.py
    mv ~/Downloads/test_live_fixtures_w7step3_2026110204.py tests/test_live_fixtures.py
    python -m pytest tests/test_live_fixtures.py -v

No network, no API key, no database. All HTTP calls are mocked.
The only live behaviour a real run adds is the first real response
shape, the rate limit headers, and the count, which the script
prints before writing the snapshot.
"""

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import live_fixtures


def _live_match():
    return {
        "id": 21131,
        "tournament": "Example Open",
        "tournament_id": "1217",
        "tour": "atp",
        "surface": "hard",
        "status": "live",
        "round": "Example Open - Quarter-finals",
        "round_code": "QF",
        "scheduled_time": "2026-11-02T15:00:00Z",
        "event_status": None,
        "players": {
            "p1": {"name": "Jannik Sinner"},
            "p2": {"name": "Carlos Alcaraz"},
        },
        "score": {
            "sets": [1, 0],
            "games": [[6, 3], [4, 4]],
            "points": ["0", "0"],
            "server": 1,
            "is_tiebreak": False,
        },
    }


def test_normalize_match_live_example():
    fixture = live_fixtures.normalize_match(_live_match())
    assert fixture["provider_match_id"] == 21131
    assert fixture["player_p1"] == "Jannik Sinner"
    assert fixture["player_p2"] == "Carlos Alcaraz"
    assert fixture["tournament"] == "Example Open"
    assert fixture["tour"] == "atp"
    assert fixture["surface"] == "hard"
    assert fixture["status"] == "live"
    assert fixture["score"]["sets"] == [1, 0]
    assert fixture["score"]["games"] == [[6, 3], [4, 4]]
    assert fixture["score"]["points"] == ["0", "0"]
    assert fixture["score"]["server"] == 1
    assert fixture["score"]["is_tiebreak"] is False


def test_normalize_match_upcoming_no_score():
    raw = _live_match()
    raw["status"] = "upcoming"
    raw["score"] = None
    fixture = live_fixtures.normalize_match(raw)
    assert fixture["status"] == "upcoming"
    assert fixture["score"]["sets"] is None
    assert fixture["score"]["games"] is None
    assert fixture["score"]["points"] is None
    assert fixture["score"]["server"] is None
    assert fixture["score"]["is_tiebreak"] is None


def test_normalize_match_missing_player_name_is_none_not_invented():
    raw = _live_match()
    raw["players"] = {"p1": {}, "p2": {"name": "Carlos Alcaraz"}}
    fixture = live_fixtures.normalize_match(raw)
    assert fixture["player_p1"] is None
    assert fixture["player_p2"] == "Carlos Alcaraz"


def test_score_games_null_is_preserved_not_zero_zero():
    fixture = live_fixtures.normalize_match(
        {**_live_match(), "score": {"sets": None, "games": None, "points": None, "server": None, "is_tiebreak": False}}
    )
    assert fixture["score"]["games"] is None


def test_rate_limit_headers_extraction():
    headers = {
        "X-RateLimit-Limit": "100",
        "X-RateLimit-Remaining": "99",
        "X-RateLimit-Reset": "2026-11-03T00:00:00Z",
    }
    assert live_fixtures.rate_limit_from_headers(headers) == {
        "limit": "100",
        "remaining": "99",
        "reset": "2026-11-03T00:00:00Z",
    }
    assert live_fixtures.rate_limit_from_headers({}) == {
        "limit": None,
        "remaining": None,
        "reset": None,
    }


def test_build_snapshot_metadata_and_note():
    snapshot = live_fixtures.build_snapshot(
        "live",
        [_live_match()],
        {"count": 1, "has_more": False},
        {"limit": "100", "remaining": "99", "reset": None},
        "/matches?status=live&limit=200&offset=0",
    )
    assert snapshot["source"] == "Live Tennis API"
    assert snapshot["tier"] == "FREE"
    assert snapshot["status_requested"] == "live"
    assert snapshot["count"] == 1
    assert snapshot["has_more"] is False
    assert snapshot["fetched_at"].endswith("Z")
    assert "not a live feed" in snapshot["snapshot_note"]
    assert "2026-05-25" in snapshot["snapshot_note"]
    assert "Sackmann" in snapshot["snapshot_note"]
    assert json.dumps(snapshot).find("test-key-123") == -1


def test_empty_data_is_snapshot_not_failure():
    snapshot = live_fixtures.build_snapshot(
        "live", [], {"count": 0, "has_more": False}, {}, "/matches?status=live"
    )
    assert snapshot["count"] == 0
    assert snapshot["fixtures"] == []


def test_completed_refused_before_http(monkeypatch):
    calls = []

    def _fake_urlopen(request, timeout=None):
        calls.append(request)
        raise AssertionError("HTTP must not be called for completed")

    monkeypatch.setattr(live_fixtures.urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(ValueError):
        live_fixtures.fetch_matches("test-key-123", "completed")
    with pytest.raises(ValueError):
        live_fixtures.fetch_matches("test-key-123", "cancelled")
    assert calls == []


def _mock_http_error_response(monkeypatch, status_code):
    class _FakeResponse:
        def __init__(self):
            self.headers = {}

        def read(self):
            return json.dumps({"data": [], "meta": {}}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    calls = []

    def _fake_urlopen(request, timeout=None):
        calls.append(request)
        raise urllib.error.HTTPError(
            url="https://example.invalid",
            code=status_code,
            msg="error",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"rate_limited"}'),
        )

    monkeypatch.setattr(live_fixtures.urllib.request, "urlopen", _fake_urlopen)
    return calls


def test_429_stops_with_exactly_one_call_no_retry(monkeypatch):
    calls = _mock_http_error_response(monkeypatch, 429)
    with pytest.raises(live_fixtures.LiveTennisError, match="429"):
        live_fixtures.fetch_matches("test-key-123", "live")
    assert len(calls) == 1


def test_403_stops_with_exactly_one_call_no_retry(monkeypatch):
    calls = _mock_http_error_response(monkeypatch, 403)
    with pytest.raises(live_fixtures.LiveTennisError, match="403"):
        live_fixtures.fetch_matches("test-key-123", "live")
    assert len(calls) == 1


def test_fetch_score_404_returns_none_not_error(monkeypatch):
    def _fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            url="https://example.invalid",
            code=404,
            msg="not found",
            hdrs=None,
            fp=io.BytesIO(b"{}"),
        )

    monkeypatch.setattr(live_fixtures.urllib.request, "urlopen", _fake_urlopen)
    assert live_fixtures.fetch_score("test-key-123", 21131) is None


def test_fetch_matches_uses_x_api_key_header(monkeypatch):
    seen = {}

    class _FakeResponse:
        def __init__(self):
            self.headers = {"X-RateLimit-Remaining": "99"}

        def read(self):
            return json.dumps(
                {"data": [_live_match()], "meta": {"count": 1, "has_more": False}}
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fake_urlopen(request, timeout=None):
        seen["header"] = request.get_header("X-api-key")
        seen["url"] = request.full_url
        return _FakeResponse()

    monkeypatch.setattr(live_fixtures.urllib.request, "urlopen", _fake_urlopen)
    data, meta, rate_limit = live_fixtures.fetch_matches("test-key-123", "live")
    assert seen["header"] == "test-key-123"
    assert "status=live" in seen["url"]
    assert len(data) == 1
    assert meta["count"] == 1
    assert rate_limit["remaining"] == "99"
