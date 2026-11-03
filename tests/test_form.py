"""Unit tests for form windows (scripts/form.py, no database needed).

Run from the project root ~/TSA:
    python -m pytest tests/test_form.py -v
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from form import PlayerForm


def test_new_player_has_no_form():
    f = PlayerForm()
    pct10, n10, pct52, n52 = f.snapshot(date(2025, 1, 1))
    assert n10 == 0 and pct10 is None
    assert n52 == 0 and pct52 is None


def test_last10_win_rate():
    f = PlayerForm()
    for won in (True, True, True, False):
        f.record(date(2025, 1, 1), won)
    pct10, n10, _, _ = f.snapshot(date(2025, 1, 2))
    assert n10 == 4
    assert pct10 == 0.75


def test_last10_drops_oldest_after_10():
    f = PlayerForm()
    # 10 wins, then 1 loss: last10 should be 9 wins + 1 loss.
    for _ in range(10):
        f.record(date(2025, 1, 1), True)
    f.record(date(2025, 1, 2), False)
    pct10, n10, _, _ = f.snapshot(date(2025, 1, 3))
    assert n10 == 10
    assert pct10 == 0.9


def test_last52_excludes_matches_over_365_days_old():
    f = PlayerForm()
    f.record(date(2024, 1, 1), True)   # 366+ days before 2025-01-02, excluded
    f.record(date(2024, 6, 1), False)  # within 365 days, included
    _, _, pct52, n52 = f.snapshot(date(2025, 1, 2))
    assert n52 == 1
    assert pct52 == 0.0
