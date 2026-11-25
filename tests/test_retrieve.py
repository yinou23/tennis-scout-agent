"""Unit tests for scripts/retrieve.py (database facts for open questions).

The market-vs-model verdict direction is the key behavior: a
negative diff (market below the model) means the market
underrates the player, a positive diff means it overrates
them. This locks in the direction so the LLM does not have
to infer it (an earlier live answer read the sign backwards
and called Fils overvalued when the numbers said underrated).
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

from db import get_connection, get_sqlite_path
from retrieve import retrieve_facts

# Market prices are live and move (Shelton-Fils went 38.5% to
# 38.0% once the match started on 2026-11-15), so freeze them
# here. The test locks the verdict direction for a fixed
# market-vs-model gap, not the live price.
_FIXED_MARKET_ODDS = {
    "polymarket": {"prob_a": 0.615, "prob_b": 0.385},
    "kalshi": {"prob_a": 0.615, "prob_b": 0.385},
}

# The database file (data/tsa.db) is a Release asset, not in
# git, so CI does not have it. Skip these tests there instead
# of failing on a missing table. Same pattern as test_eval.py.
pytestmark = pytest.mark.skipif(
    not get_sqlite_path().exists(),
    reason="SQLite database file not found; set SQLITE_PATH or build it with scripts/pg_to_sqlite.py",
)


def test_fils_verdict_is_underrated():
    with patch("retrieve.get_market_odds", return_value=_FIXED_MARKET_ODDS), get_connection() as conn:
        facts = retrieve_facts(
            conn,
            "fils是否被低估",
            {"player_a": "Ben Shelton", "player_b": "Arthur Fils", "surface": "Hard"},
        )
    text = facts["context_text"]
    assert "Arthur Fils: market average 38.5% vs model 45.2%" in text
    assert "diff -6.7 points (market minus model). Verdict: underrated by the market." in text
    assert "Ben Shelton: market average 61.5% vs model 54.8%" in text
    assert "diff +6.7 points (market minus model). Verdict: overrated by the market." in text


def test_h2h_is_two_one_not_zero_zero():
    with patch("retrieve.get_market_odds", return_value=_FIXED_MARKET_ODDS), get_connection() as conn:
        facts = retrieve_facts(
            conn,
            "fils是否被低估",
            {"player_a": "Ben Shelton", "player_b": "Arthur Fils", "surface": "Hard"},
        )
    assert "Head-to-head: Ben Shelton 2, Arthur Fils 1 (3 meetings)." in facts["context_text"]
