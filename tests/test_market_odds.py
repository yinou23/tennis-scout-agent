"""Unit tests for Week 9 Step 3 read-only market odds.

Run from the project root ~/TSA:
    python -m pytest tests/test_market_odds.py -v

No network, no database, no API key. Tests inject a fake
http_get that returns fixed JSON dicts, or raises, so they
run everywhere and never touch Polymarket or Kalshi.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from market_odds import get_kalshi_odds, get_market_odds, get_polymarket_odds


def _poly_market(question="Will Jannik Sinner beat Carlos Alcaraz?", volume=1000):
    return {
        "question": question,
        "slug": "will-sinner-beat-alcaraz",
        "volume": volume,
        "outcomePrices": '["0.62", "0.38"]',
        "clobTokenIds": '["111"]',
    }


def _fake_http_get_poly_midpoint(mid="0.60"):
    def http_get(url, timeout=10):
        if "clob.polymarket.com/midpoint" in url:
            return {"mid": mid}
        if "gamma-api.polymarket.com/public-search" in url:
            return {"markets": [_poly_market()]}
        raise RuntimeError(f"unexpected url: {url}")

    return http_get


def test_polymarket_prefers_clob_midpoint_over_gamma_prices():
    result = get_polymarket_odds("Jannik Sinner", "Carlos Alcaraz", http_get=_fake_http_get_poly_midpoint("0.60"))
    assert result is not None
    assert result["source"] == "polymarket"
    assert result["prob_a"] == 0.60
    assert result["prob_b"] == 0.40
    assert result["slug"] == "will-sinner-beat-alcaraz"
    assert result["question"] == "Will Jannik Sinner beat Carlos Alcaraz?"
    assert result["fetched_at"] is not None


def test_polymarket_falls_back_to_gamma_outcome_prices():
    def http_get(url, timeout=10):
        if "clob.polymarket.com/midpoint" in url:
            raise RuntimeError("clob failed")
        return {"markets": [_poly_market()]}

    result = get_polymarket_odds("Jannik Sinner", "Carlos Alcaraz", http_get=http_get)
    assert result is not None
    assert result["prob_a"] == 0.62
    assert result["prob_b"] == 0.38


def test_polymarket_reversed_yes_subject_inverts_probability():
    def http_get(url, timeout=10):
        if "clob.polymarket.com" in url:
            return {"mid": "0.40"}
        return {"markets": [_poly_market(question="Will Carlos Alcaraz beat Jannik Sinner?")]}

    result = get_polymarket_odds("Jannik Sinner", "Carlos Alcaraz", http_get=http_get)
    assert result is not None
    assert result["prob_a"] == 0.60
    assert result["prob_b"] == 0.40


def test_polymarket_one_surname_missing_returns_none():
    def http_get(url, timeout=10):
        return {"markets": [_poly_market(question="Will Jannik Sinner beat Novak Djokovic?")]}

    assert get_polymarket_odds("Jannik Sinner", "Carlos Alcaraz", http_get=http_get) is None


def test_polymarket_ambiguous_vs_title_returns_none():
    def http_get(url, timeout=10):
        return {"markets": [_poly_market(question="Jannik Sinner vs Carlos Alcaraz")]}

    assert get_polymarket_odds("Jannik Sinner", "Carlos Alcaraz", http_get=http_get) is None


def test_polymarket_http_error_returns_none_not_raise():
    def http_get(url, timeout=10):
        raise RuntimeError("http_error:503")

    assert get_polymarket_odds("Jannik Sinner", "Carlos Alcaraz", http_get=http_get) is None


def test_polymarket_picks_highest_volume_match():
    def http_get(url, timeout=10):
        if "clob.polymarket.com" in url:
            return {"mid": "0.55"}
        return {"markets": [
            _poly_market(volume=10),
            _poly_market(question="Will Jannik Sinner beat Carlos Alcaraz?", volume=9999),
        ]}

    result = get_polymarket_odds("Jannik Sinner", "Carlos Alcaraz", http_get=http_get)
    assert result is not None
    assert result["volume"] == 9999


def _kalshi_market(title="Will Jannik Sinner beat Carlos Alcaraz?", last_price=58, volume=500):
    return {"title": title, "ticker": "KXTEST-1", "last_price": last_price, "volume": volume}


def test_kalshi_cents_to_probability():
    def http_get(url, timeout=10):
        assert "api.elections.kalshi.com/trade-api/v2/markets" in url
        return {"markets": [_kalshi_market(last_price=58)]}

    result = get_kalshi_odds("Jannik Sinner", "Carlos Alcaraz", http_get=http_get)
    assert result is not None
    assert result["source"] == "kalshi"
    assert result["prob_a"] == 0.58
    assert result["prob_b"] == 0.42
    assert result["price_cents"] == 58
    assert result["ticker"] == "KXTEST-1"


def test_kalshi_bid_ask_midpoint_preferred():
    def http_get(url, timeout=10):
        market = _kalshi_market(last_price=10)
        market["yes_bid"] = 60
        market["yes_ask"] = 64
        return {"markets": [market]}

    result = get_kalshi_odds("Jannik Sinner", "Carlos Alcaraz", http_get=http_get)
    assert result is not None
    assert result["prob_a"] == 0.62
    assert result["price_cents"] == 62


def test_kalshi_reversed_yes_subject_inverts_probability():
    def http_get(url, timeout=10):
        return {"markets": [_kalshi_market(title="Will Carlos Alcaraz beat Jannik Sinner?", last_price=40)]}

    result = get_kalshi_odds("Jannik Sinner", "Carlos Alcaraz", http_get=http_get)
    assert result is not None
    assert result["prob_a"] == 0.60


def test_kalshi_one_surname_missing_returns_none():
    def http_get(url, timeout=10):
        return {"markets": [_kalshi_market(title="Will Jannik Sinner beat Novak Djokovic?")]}

    assert get_kalshi_odds("Jannik Sinner", "Carlos Alcaraz", http_get=http_get) is None


def test_kalshi_http_error_returns_none_not_raise():
    def http_get(url, timeout=10):
        raise RuntimeError("http_error:503")

    assert get_kalshi_odds("Jannik Sinner", "Carlos Alcaraz", http_get=http_get) is None


def test_get_market_odds_returns_both_venues():
    def http_get(url, timeout=10):
        if "gamma-api.polymarket.com" in url:
            return {"markets": [_poly_market()]}
        if "clob.polymarket.com" in url:
            return {"mid": "0.60"}
        if "kalshi.com" in url:
            return {"markets": [_kalshi_market(last_price=58)]}
        raise RuntimeError(f"unexpected url: {url}")

    result = get_market_odds("Jannik Sinner", "Carlos Alcaraz", http_get=http_get)
    assert result["polymarket"]["prob_a"] == 0.60
    assert result["kalshi"]["prob_a"] == 0.58


def test_get_market_odds_one_venue_failing_does_not_hide_other():
    def http_get(url, timeout=10):
        if "polymarket.com" in url:
            raise RuntimeError("polymarket down")
        if "kalshi.com" in url:
            return {"markets": [_kalshi_market(last_price=58)]}
        raise RuntimeError(f"unexpected url: {url}")

    result = get_market_odds("Jannik Sinner", "Carlos Alcaraz", http_get=http_get)
    assert result["polymarket"] is None
    assert result["kalshi"]["prob_a"] == 0.58


def _poly_match_winner(question, outcomes, prices, volume, closed=False):
    """A vs-title market whose outcomes are the player names."""
    return {
        "question": question,
        "slug": "paris-masters-fritz-alcaraz",
        "volume": volume,
        "closed": closed,
        "active": True,
        "outcomes": outcomes,
        "outcomePrices": prices,
        "clobTokenIds": '["111"]',
    }


def _fake_http_get_match_winner(markets):
    def http_get(url, timeout=10):
        if "clob.polymarket.com/midpoint" in url:
            raise RuntimeError("clob should not be called when outcomePrices exist")
        if "gamma-api.polymarket.com/public-search" in url:
            return {"markets": markets}
        raise RuntimeError(f"unexpected url: {url}")

    return http_get


def test_polymarket_vs_title_match_winner_outcomes_in_order():
    """Live defect: Paris Masters Fritz vs Alcaraz, 0.275 / 0.725."""
    markets = [
        _poly_match_winner(
            "Rolex Paris Masters: Taylor Fritz vs Carlos Alcaraz",
            '["Taylor Fritz", "Carlos Alcaraz"]',
            '["0.275", "0.725"]',
            11577,
        )
    ]
    result = get_polymarket_odds(
        "Taylor Fritz", "Carlos Alcaraz",
        http_get=_fake_http_get_match_winner(markets),
    )
    assert result is not None
    assert result["market_type"] == "match_winner"
    assert result["prob_a"] == 0.275
    assert result["prob_b"] == 0.725
    assert result["question"] == "Rolex Paris Masters: Taylor Fritz vs Carlos Alcaraz"


def test_polymarket_vs_title_match_winner_outcomes_reversed():
    markets = [
        _poly_match_winner(
            "Rolex Paris Masters: Taylor Fritz vs Carlos Alcaraz",
            '["Carlos Alcaraz", "Taylor Fritz"]',
            '["0.725", "0.275"]',
            11577,
        )
    ]
    result = get_polymarket_odds(
        "Taylor Fritz", "Carlos Alcaraz",
        http_get=_fake_http_get_match_winner(markets),
    )
    assert result is not None
    assert result["prob_a"] == 0.275
    assert result["prob_b"] == 0.725


def test_polymarket_vs_title_match_winner_surname_outcomes():
    """Gamma sometimes lists outcomes as surnames only."""
    markets = [
        _poly_match_winner(
            "ATP World Tour Finals: Carlos Alcaraz vs Taylor Fritz",
            '["Alcaraz", "Fritz"]',
            '["0.725", "0.275"]',
            11577,
        )
    ]
    result = get_polymarket_odds(
        "Taylor Fritz", "Carlos Alcaraz",
        http_get=_fake_http_get_match_winner(markets),
    )
    assert result is not None
    assert result["prob_a"] == 0.275
    assert result["prob_b"] == 0.725


def test_polymarket_closed_market_filtered_even_at_higher_volume():
    """The closed ATP Finals market at volume 1753310 and the
    closed 3:0 prop at 26999 must not beat the open Paris
    market at 11577."""
    markets = [
        _poly_match_winner(
            "Rolex Paris Masters: Taylor Fritz vs Carlos Alcaraz",
            '["Taylor Fritz", "Carlos Alcaraz"]',
            '["0.275", "0.725"]',
            11577,
        ),
        _poly_match_winner(
            "ATP World Tour Finals, Jimmy Connors Group: Carlos Alcaraz vs Taylor Fritz",
            '["Alcaraz", "Fritz"]',
            '["1", "0"]',
            1753310,
            closed=True,
        ),
        _poly_market(
            question="Will Carlos Alcaraz win 3:0 against Taylor Fritz?",
            volume=26999,
        ),
    ]
    markets[2]["closed"] = True
    result = get_polymarket_odds(
        "Taylor Fritz", "Carlos Alcaraz",
        http_get=_fake_http_get_match_winner(markets),
    )
    assert result is not None
    assert result["prob_a"] == 0.275
    assert result["prob_b"] == 0.725
    assert "Paris Masters" in result["question"]


def test_polymarket_prop_not_preferred_over_open_match_winner():
    """An open 3:0 prop at much higher volume must not beat
    an open match-winner market."""
    def http_get(url, timeout=10):
        if "clob.polymarket.com" in url:
            return {"mid": "0.10"}
        return {"markets": [
            _poly_match_winner(
                "Rolex Paris Masters: Taylor Fritz vs Carlos Alcaraz",
                '["Taylor Fritz", "Carlos Alcaraz"]',
                '["0.275", "0.725"]',
                11577,
            ),
            _poly_market(
                question="Will Carlos Alcaraz win 3:0 against Taylor Fritz?",
                volume=99999,
            ),
        ]}

    result = get_polymarket_odds("Taylor Fritz", "Carlos Alcaraz", http_get=http_get)
    assert result is not None
    assert result["market_type"] == "match_winner"
    assert result["prob_a"] == 0.275
    assert result["prob_b"] == 0.725


def test_polymarket_set_winner_market_not_a_match_winner():
    """Set Winner markets list the player names as outcomes
    too, but they price one set, not the match, so they must
    never be returned as the matchup probability."""
    markets = [
        _poly_match_winner(
            "Set 1 Winner: Taylor Fritz vs Carlos Alcaraz",
            '["Taylor Fritz", "Carlos Alcaraz"]',
            '["0.34", "0.66"]',
            5000,
        )
    ]
    result = get_polymarket_odds(
        "Taylor Fritz", "Carlos Alcaraz",
        http_get=_fake_http_get_match_winner(markets),
    )
    assert result is None


def test_polymarket_match_winner_falls_back_to_clob_midpoint():
    def http_get(url, timeout=10):
        if "clob.polymarket.com/midpoint" in url:
            return {"mid": "0.30"}
        return {"markets": [
            {
                "question": "Rolex Paris Masters: Taylor Fritz vs Carlos Alcaraz",
                "slug": "paris-masters-fritz-alcaraz",
                "volume": 11577,
                "closed": False,
                "outcomes": ["Taylor Fritz", "Carlos Alcaraz"],
                "clobTokenIds": '["999"]',
            }
        ]}

    result = get_polymarket_odds("Taylor Fritz", "Carlos Alcaraz", http_get=http_get)
    assert result is not None
    assert result["prob_a"] == 0.30
    assert result["prob_b"] == 0.70


def test_kalshi_name_wins_title_matches():
    """Live Kalshi shape: title is 'Taylor Fritz wins', and
    the event ticker KXATPMATCH-26NOV06FRIALC carries both
    players as 3-letter codes."""
    def http_get(url, timeout=10):
        market = _kalshi_market(title="Taylor Fritz wins", last_price=27, volume=1813)
        market["event_ticker"] = "KXATPMATCH-26NOV06FRIALC"
        return {"markets": [market]}

    result = get_kalshi_odds("Taylor Fritz", "Carlos Alcaraz", http_get=http_get)
    assert result is not None
    assert result["prob_a"] == 0.27
    assert result["prob_b"] == 0.73


def test_kalshi_closed_market_filtered():
    def http_get(url, timeout=10):
        market = _kalshi_market(title="Taylor Fritz wins", last_price=27, volume=1813)
        market["event_ticker"] = "KXATPMATCH-26NOV06FRIALC"
        market["status"] = "closed"
        return {"markets": [market]}

    assert get_kalshi_odds("Taylor Fritz", "Carlos Alcaraz", http_get=http_get) is None


def test_kalshi_prefers_player_a_market_over_higher_volume_player_b():
    """Kalshi lists one market per player. The player-A market
    is used directly instead of inverting the higher-volume
    player-B market, and volume is read from volume_fp."""
    def http_get(url, timeout=10):
        market_b = _kalshi_market(title="Carlos Alcaraz wins", last_price=74, volume=0)
        market_b["event_ticker"] = "KXATPMATCH-26NOV06FRIALC"
        market_b["volume_fp"] = 19649
        market_b["volume"] = None
        market_a = _kalshi_market(title="Taylor Fritz wins", last_price=27, volume=0)
        market_a["event_ticker"] = "KXATPMATCH-26NOV06FRIALC"
        market_a["volume_fp"] = 1813
        market_a["volume"] = None
        return {"markets": [market_b, market_a]}

    result = get_kalshi_odds("Taylor Fritz", "Carlos Alcaraz", http_get=http_get)
    assert result is not None
    assert result["prob_a"] == 0.27
    assert result["prob_b"] == 0.73
    assert result["volume"] == 1813
