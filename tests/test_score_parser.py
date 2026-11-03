"""Unit tests for score parsing (scripts/score_parser.py).

Run from the project root ~/TSA:
    python -m pytest tests/test_score_parser.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from score_parser import parse_score, winner_games_pct


def test_straight_sets():
    assert parse_score("6-4 6-3") == (12, 7, True)


def test_tiebreak_sets():
    # Tiebreak points in parentheses do not add games.
    assert parse_score("6-7(4) 7-6(5) 6-3") == (19, 16, True)


def test_lost_middle_set():
    assert parse_score("6-4 3-6 7-5") == (16, 15, True)


def test_retirement_is_incomplete_but_counts_games_so_far():
    w, l, complete = parse_score("6-4 2-1 RET")
    assert (w, l) == (8, 5)
    assert complete is False
    assert winner_games_pct("6-4 2-1 RET") is None


def test_missing_and_walkover():
    assert parse_score("") == (None, None, False)
    assert parse_score(None) == (None, None, False)
    assert parse_score("W/O") == (None, None, False)


def test_winner_games_pct():
    assert abs(winner_games_pct("6-4 6-3") - 12 / 19) < 1e-9
