"""Unit tests for games-% form windows (scripts/games_form.py).

Run from the project root ~/TSA:
    python -m pytest tests/test_games_form.py -v
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from games_form import PlayerGamesForm


def test_new_player_has_no_games_form():
    f = PlayerGamesForm()
    pct10, m10, g10, pct52, m52, g52 = f.snapshot(date(2025, 1, 1))
    assert m10 == 0 and pct10 is None
    assert m52 == 0 and pct52 is None


def test_last10_games_pct_weights_by_games_not_by_match():
    f = PlayerGamesForm()
    # Match 1: won 12 of 19 games. Match 2: won 6 of 12 games.
    # Total games pct = 18 / 31, not the average of 12/19 and 6/12.
    f.record(date(2025, 1, 1), 12, 19, True)
    f.record(date(2025, 1, 2), 6, 12, True)
    pct10, m10, g10, _, _, _ = f.snapshot(date(2025, 1, 3))
    assert m10 == 2 and g10 == 31
    assert abs(pct10 - 18 / 31) < 1e-9


def test_incomplete_matches_are_skipped():
    f = PlayerGamesForm()
    f.record(date(2025, 1, 1), 8, 13, False)
    pct10, m10, _, _, _, _ = f.snapshot(date(2025, 1, 2))
    assert m10 == 0 and pct10 is None


def test_last52_excludes_old_matches():
    f = PlayerGamesForm()
    f.record(date(2024, 1, 1), 12, 19, True)
    f.record(date(2024, 6, 1), 6, 12, True)
    _, _, _, pct52, m52, g52 = f.snapshot(date(2025, 1, 2))
    assert m52 == 1 and g52 == 12
    assert pct52 == 0.5
