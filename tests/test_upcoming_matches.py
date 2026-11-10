"""Unit tests for Week 9 live upcoming matches.

Run from the project root ~/TSA:
    python -m pytest tests/test_upcoming_matches.py -v

No network, no database, no API key. Tests inject a fake
http_get that returns a fixed Gamma events payload, so they
run everywhere and never touch Polymarket.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import api
from fastapi.testclient import TestClient
from upcoming_matches import _is_atp_tour_level, get_upcoming_matches


def _event(title, start, markets, closed=False):
    return {
        "title": title,
        "startDate": start,
        "closed": closed,
        "active": not closed,
        "markets": markets,
    }


def _winner_market(question, outcomes, prices, volume=1200):
    return {
        "question": question,
        "slug": "some-event",
        "volume": volume,
        "closed": False,
        "outcomes": outcomes,
        "outcomePrices": prices,
        "clobTokenIds": "[]",
    }


def _payload(events):
    def http_get(url, timeout=10):
        assert "gamma-api.polymarket.com/events" in url
        assert "tag_slug=tennis" in url
        return {"events": events}

    return http_get


def test_returns_soonest_match_winner_first():
    late = _event(
        "Moselle Open: Nicolas Mejia vs Max Hans Rehberg",
        "2026-11-10T00:06:26Z",
        [_winner_market(
            "Moselle Open: Nicolas Mejia vs Max Hans Rehberg",
            ["Nicolas Mejia", "Max Hans Rehberg"],
            '["0.485", "0.515"]',
        )],
    )
    early = _event(
        "BNP Paribas Nordic Open: Benjamin Bonzi vs Tom Gentzsch",
        "2026-11-10T00:05:00Z",
        [_winner_market(
            "BNP Paribas Nordic Open: Benjamin Bonzi vs Tom Gentzsch",
            ["Benjamin Bonzi", "Tom Gentzsch"],
            '["0.5", "0.5"]',
        )],
    )
    result = get_upcoming_matches(http_get=_payload([late, early]))
    assert len(result) == 2
    assert result[0]["player_a"] == "Benjamin Bonzi"
    assert result[0]["player_b"] == "Tom Gentzsch"
    assert result[0]["tournament"] == "BNP Paribas Nordic Open"
    assert result[0]["start_time"] == "2026-11-10T00:05:00Z"
    assert result[0]["polymarket"]["prob_a"] == 0.5
    assert result[0]["polymarket"]["prob_b"] == 0.5
    assert result[0]["polymarket"]["volume"] == 1200.0
    assert result[1]["player_a"] == "Nicolas Mejia"
    assert result[1]["polymarket"]["prob_a"] == 0.485
    assert result[1]["polymarket"]["prob_b"] == 0.515


def test_limit_four():
    events = [
        _event(
            f"Moselle Open: Player A{i} vs Player B{i}",
            f"2026-11-10T00:0{i}:00Z",
            [_winner_market(
                f"Moselle Open: Player A{i} vs Player B{i}",
                [f"Player A{i}", f"Player B{i}"],
                '["0.5", "0.5"]',
            )],
        )
        for i in range(5)
    ]
    result = get_upcoming_matches(http_get=_payload(events))
    assert len(result) == 4
    assert result[0]["player_a"] == "Player A0"
    assert result[3]["player_a"] == "Player A3"


def test_prop_market_skipped():
    prop = _event(
        "Moselle Open: Nicolas Mejia vs Max Hans Rehberg",
        "2026-11-10T00:06:26Z",
        [_winner_market(
            "Set 1 Winner: Nicolas Mejia vs Max Hans Rehberg",
            ["Nicolas Mejia", "Max Hans Rehberg"],
            '["0.6", "0.4"]',
        )],
    )
    result = get_upcoming_matches(http_get=_payload([prop]))
    assert result == []


def test_closed_event_skipped():
    closed = _event(
        "Moselle Open: Nicolas Mejia vs Max Hans Rehberg",
        "2026-11-10T00:06:26Z",
        [_winner_market(
            "Moselle Open: Nicolas Mejia vs Max Hans Rehberg",
            ["Nicolas Mejia", "Max Hans Rehberg"],
            '["0.485", "0.515"]',
        )],
        closed=True,
    )
    result = get_upcoming_matches(http_get=_payload([closed]))
    assert result == []


def test_doubles_event_skipped():
    doubles = _event(
        "BNP Paribas Nordic Open (Doubles): Frantzen/Stevens vs Nouza/Venus",
        "2026-11-08T05:10:29Z",
        [_winner_market(
            "BNP Paribas Nordic Open (Doubles): Frantzen/Stevens vs Nouza/Venus",
            ["Frantzen/Stevens", "Nouza/Venus"],
            '["0.5", "0.5"]',
        )],
    )
    result = get_upcoming_matches(http_get=_payload([doubles]))
    assert result == []


def test_non_match_event_skipped():
    outright = _event(
        "Will Novak Djokovic announce his retirement by...?",
        "2026-07-23T17:15:29Z",
        [_winner_market(
            "Will Novak Djokovic announce his retirement by...?",
            ["Yes", "No"],
            '["0.1", "0.9"]',
        )],
    )
    result = get_upcoming_matches(http_get=_payload([outright]))
    assert result == []


def test_no_prices_returns_polymarket_none():
    no_price = _event(
        "Moselle Open: Nicolas Mejia vs Max Hans Rehberg",
        "2026-11-10T00:06:26Z",
        [{
            "question": "Moselle Open: Nicolas Mejia vs Max Hans Rehberg",
            "slug": "some-event",
            "volume": None,
            "closed": False,
            "outcomes": ["Nicolas Mejia", "Max Hans Rehberg"],
            "outcomePrices": None,
            "clobTokenIds": "[]",
        }],
    )
    result = get_upcoming_matches(http_get=_payload([no_price]))
    assert len(result) == 1
    assert result[0]["player_a"] == "Nicolas Mejia"
    assert result[0]["polymarket"] is None


def test_http_error_returns_empty_list():
    def http_get(url, timeout=10):
        raise RuntimeError("network down")

    assert get_upcoming_matches(http_get=http_get) == []


def test_unrecognised_payload_returns_empty_list():
    def http_get(url, timeout=10):
        return {"weird": True}

    assert get_upcoming_matches(http_get=http_get) == []


def test_atp_tour_level_names():
    included = [
        "BNP Paribas Nordic Open",
        "Rolex Paris Masters",
        "Moselle Open",
        "Belgrade Open",
        "Nitto ATP Finals",
        "Next Gen ATP Finals",
        "Australian Open",
        "US Open",
        "Roland Garros",
        "Wimbledon",
        "Adelaide International",
        "Grand Prix Hassan II",
        "U.S. Men's Clay Court Championship",
        "Mallorca Championships",
        "ASB Classic",
    ]
    excluded = [
        "Orleans",
        "Knoxville",
        "Lima",
        "Monastir",
        "Hersonissos 3",
        "Sopo",
        "W15 Sharm ElSheikh",
        "W15 Antalya",
        "M15 Heraklion",
        "M15 Sharm ElSheikh",
        "W35 Heraklion",
        "Davis Cup",
        "Laver Cup",
        "United Cup",
        "Hopman Cup",
        "Billie Jean King Cup",
        "",
        None,
    ]
    for name in included:
        assert _is_atp_tour_level(name) is True, name
    for name in excluded:
        assert _is_atp_tour_level(name) is False, name


def test_challenger_and_itf_events_excluded():
    events = [
        _event(
            f"{tourn}: Player A{i} vs Player B{i}",
            f"2026-11-10T00:0{i}:00Z",
            [_winner_market(
                f"{tourn}: Player A{i} vs Player B{i}",
                [f"Player A{i}", f"Player B{i}"],
                '["0.5", "0.5"]',
            )],
        )
        for i, tourn in enumerate([
            "Orleans", "Knoxville", "W15 Sharm ElSheikh",
            "M15 Heraklion", "Hersonissos 3",
        ])
    ]
    assert get_upcoming_matches(http_get=_payload(events)) == []


def test_level_filter_applies_before_limit():
    challengers = [
        _event(
            f"Orleans: Player C{i} vs Player D{i}",
            f"2026-11-10T00:0{i}:00Z",
            [_winner_market(
                f"Orleans: Player C{i} vs Player D{i}",
                [f"Player C{i}", f"Player D{i}"],
                '["0.5", "0.5"]',
            )],
        )
        for i in range(5)
    ]
    tour = [
        _event(
            "BNP Paribas Nordic Open: Lorenzo Sonego vs Alejandro Tabilo",
            "2026-11-10T00:06:00Z",
            [_winner_market(
                "BNP Paribas Nordic Open: Lorenzo Sonego vs Alejandro Tabilo",
                ["Lorenzo Sonego", "Alejandro Tabilo"],
                '["0.52", "0.48"]',
            )],
        ),
        _event(
            "Moselle Open: Alexander Shevchenko vs Casper Ruud",
            "2026-11-10T00:07:00Z",
            [_winner_market(
                "Moselle Open: Alexander Shevchenko vs Casper Ruud",
                ["Alexander Shevchenko", "Casper Ruud"],
                '["0.4", "0.6"]',
            )],
        ),
    ]
    result = get_upcoming_matches(http_get=_payload(challengers + tour))
    assert len(result) == 2
    assert result[0]["tournament"] == "BNP Paribas Nordic Open"
    assert result[1]["tournament"] == "Moselle Open"


def test_matches_endpoint_returns_list(monkeypatch):
    fixed = [{
        "tournament": "Knoxville",
        "player_a": "Nicolas Mejia",
        "player_b": "Max Hans Rehberg",
        "start_time": "2026-11-10T00:06:26Z",
        "polymarket": {"prob_a": 0.485, "prob_b": 0.515, "volume": 1200.0},
    }]
    monkeypatch.setattr(api, "_get_upcoming_matches", lambda: fixed)
    client = TestClient(api.create_app())
    res = client.get("/matches")
    assert res.status_code == 200
    data = res.json()
    assert data["matches"] == fixed
    assert data["fetched_at"] is not None
