"""Unit tests for surface Elo blending (scripts/surface_elo.py).

Run from the project root ~/TSA:
    python -m pytest tests/test_surface_elo.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from surface_elo import blended_elo, win_prob


def test_no_surface_history_uses_overall():
    assert blended_elo(1500, 1800, 0, 20) == 1800
    assert blended_elo(1500, 1800, None, 20) == 1800


def test_prior_none_uses_overall():
    assert blended_elo(1900, 1800, 100, None) == 1800


def test_prior_zero_uses_surface():
    assert blended_elo(1900, 1800, 5, 0) == 1900


def test_blend_weights_by_surface_matches():
    # n=20, prior=20 gives w=0.5, halfway between surface and overall.
    assert blended_elo(1900, 1700, 20, 20) == 1800
    # n=5, prior=20 gives w=0.2.
    assert abs(blended_elo(1900, 1700, 5, 20) - 1740) < 1e-9


def test_win_prob():
    assert abs(win_prob(1500, 1500) - 0.5) < 1e-9
    assert win_prob(1700, 1500) > 0.5
