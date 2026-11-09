"""Tests for the /market 60 second cache in scripts/api.py.

Run from the project root ~/TSA:
    python -m pytest tests/test_api_market_cache.py -v

No network, no database, no API key. The cache helper is tested
directly with a fake _get_market_odds that counts calls, and
with an injectable clock, so the tests never touch Polymarket
or Kalshi and never wait 60 real seconds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import api


def _fake_odds_factory(calls):
    def fake(player_a, player_b):
        calls.append((player_a, player_b))
        return {"polymarket": {"prob_a": 0.275}, "kalshi": None}

    return fake


def test_cache_hit_within_ttl(monkeypatch):
    api._market_cache.clear()
    calls = []
    monkeypatch.setattr(api, "_get_market_odds", _fake_odds_factory(calls))
    first = api._get_market_odds_cached("Taylor Fritz", "Carlos Alcaraz", now=1000.0)
    second = api._get_market_odds_cached("Taylor Fritz", "Carlos Alcaraz", now=1050.0)
    assert first is second
    assert len(calls) == 1


def test_cache_expires_after_ttl(monkeypatch):
    api._market_cache.clear()
    calls = []
    monkeypatch.setattr(api, "_get_market_odds", _fake_odds_factory(calls))
    api._get_market_odds_cached("Taylor Fritz", "Carlos Alcaraz", now=1000.0)
    api._get_market_odds_cached("Taylor Fritz", "Carlos Alcaraz", now=1060.0)
    assert len(calls) == 2


def test_cache_key_is_case_insensitive(monkeypatch):
    api._market_cache.clear()
    calls = []
    monkeypatch.setattr(api, "_get_market_odds", _fake_odds_factory(calls))
    api._get_market_odds_cached("Taylor Fritz", "Carlos Alcaraz", now=1000.0)
    api._get_market_odds_cached("taylor fritz", "carlos alcaraz", now=1010.0)
    assert len(calls) == 1


def test_cache_is_per_pair(monkeypatch):
    api._market_cache.clear()
    calls = []
    monkeypatch.setattr(api, "_get_market_odds", _fake_odds_factory(calls))
    api._get_market_odds_cached("Taylor Fritz", "Carlos Alcaraz", now=1000.0)
    api._get_market_odds_cached("Jannik Sinner", "Carlos Alcaraz", now=1010.0)
    assert len(calls) == 2
