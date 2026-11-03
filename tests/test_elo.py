"""Unit tests for classic Elo (scripts/elo.py, no database needed).

Run from the project root ~/TSA:
    python -m pytest tests/test_elo.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from elo import INITIAL_ELO, expected_win_prob, update_elo


def test_new_player_starts_at_1500():
    assert INITIAL_ELO == 1500.0


def test_equal_ratings_expected_prob_is_half():
    assert expected_win_prob(1500.0, 1500.0) == 0.5


def test_equal_match_winner_gains_16_loser_loses_16_with_k32():
    w_new, l_new, expected = update_elo(1500.0, 1500.0, k=32.0)
    assert expected == 0.5
    assert w_new == 1516.0
    assert l_new == 1484.0


def test_strong_vs_weak_expected_prob_about_76_percent():
    # 1700 vs 1500, 200-point gap.
    assert abs(expected_win_prob(1700.0, 1500.0) - 0.76) < 0.01


def test_upset_moves_more_than_expected_win():
    # Strong player wins: small gain.
    w_new, l_new, _ = update_elo(1700.0, 1500.0, k=32.0)
    strong_win_gain = w_new - 1700.0
    # Weak player upsets: large gain.
    w_new2, l_new2, _ = update_elo(1500.0, 1700.0, k=32.0)
    upset_gain = w_new2 - 1500.0
    assert upset_gain > strong_win_gain
    assert abs(strong_win_gain - 7.7) < 0.2
    assert abs(upset_gain - 24.3) < 0.2


def test_total_rating_is_conserved():
    w_new, l_new, _ = update_elo(1700.0, 1500.0, k=32.0)
    assert abs((w_new + l_new) - (1700.0 + 1500.0)) < 1e-9
