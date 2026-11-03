"""Unit tests for Week 3 Tools 1 to 5 (scripts/tennis_tools.py).

Run from the project root ~/TSA:
    python -m pytest tests/test_tennis_tools.py -v

Pure logic tests run everywhere, no database needed.
The database smoke test runs only when DATABASE_URL is set,
otherwise it is skipped. On the Mac, with .env loaded, it checks
Sinner against the known Week 2 top list: overall Elo about
2283.1, 457 matches. If that check fails, the Elo table is not
the one Week 2 produced, and the tool must not be trusted yet.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from surface_elo import blended_elo
from tennis_tools import SURFACES


def test_surfaces_are_the_four_main_surfaces():
    assert SURFACES == ("Hard", "Clay", "Grass", "Carpet")


def test_blended_prior100_example_from_week2():
    # Week 2 example: overall 1800, surface 1900, n=20, prior=100.
    # w = 20 / 120 = 1/6, blended = 1800 + 100/6 = 1816.7.
    assert abs(blended_elo(1900, 1800, 20, 100) - 1816.7) < 0.1


def test_blended_no_surface_history_is_overall():
    assert blended_elo(1900, 1800, 0, 100) == 1800


def test_blended_many_surface_matches_approaches_surface():
    # n=1000, prior=100 gives w=0.909, blended close to surface.
    result = blended_elo(1900, 1800, 1000, 100)
    assert 1890 < result < 1900


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_sinner_overall_elo_matches_week2_top_list():
    import psycopg

    from tennis_tools import get_player_elo

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        result = get_player_elo(conn, "Jannik Sinner")
    assert result is not None
    assert not result.get("ambiguous")
    assert result["name"] == "Jannik Sinner"
    assert abs(result["overall_elo"] - 2283.1) < 1.0
    assert result["matches_played"] == 457
    # Surface breakdown must be present for all four surfaces.
    assert set(result["surfaces"].keys()) == set(SURFACES)
    for surface, info in result["surfaces"].items():
        if info["surface_matches"] == 0:
            assert info["surface_elo"] is None
            assert info["blended_elo"] == result["overall_elo"]
        else:
            assert info["surface_elo"] is not None
            # Blended must lie between overall and surface.
            low = min(result["overall_elo"], info["surface_elo"])
            high = max(result["overall_elo"], info["surface_elo"])
            assert low - 0.1 <= info["blended_elo"] <= high + 0.1


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_last_name_alone_is_ambiguous():
    # Real data has two Sinners: Jannik (206173) and Martin (101441).
    # Last name alone must NOT silently pick one.
    import psycopg

    from tennis_tools import get_player_elo

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        result = get_player_elo(conn, "Sinner")
    assert result.get("ambiguous") is True
    names = [c["name"] for c in result["candidates"]]
    assert "Jannik Sinner" in names
    assert "Martin Sinner" in names


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_unknown_player_returns_none():
    import psycopg

    from tennis_tools import get_player_elo

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        assert get_player_elo(conn, "Not A Real Player XYZ") is None


# ---------------------------------------------------------------------------
# Tool 2 tests
# ---------------------------------------------------------------------------


def test_win_rate_helper():
    from tennis_tools import _games_pct, _win_rate

    assert _win_rate(7, 10) == 0.7
    assert _win_rate(0, 0) is None
    assert _games_pct(60, 100) == 0.6
    assert _games_pct(0, 0) is None


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_sinner_form_is_sane():
    # No exact values hardcoded: Sinner's current form is not known
    # in advance on this machine. Check structure and ranges only.
    import psycopg

    from tennis_tools import get_player_form

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        result = get_player_form(conn, "Jannik Sinner")
    assert result is not None
    assert not result.get("ambiguous")
    assert result["anchor_date"] == "2026-05-25"
    win = result["win_form"]
    assert 1 <= win["last10_n"] <= 10
    assert 0.0 <= win["last10_pct"] <= 1.0
    assert win["last10_wins"] == round(win["last10_pct"] * win["last10_n"])
    assert win["last52_n"] >= 0
    if win["last52_n"] > 0:
        assert 0.0 <= win["last52_pct"] <= 1.0
    else:
        assert win["last52_pct"] is None
    games = result["games_form"]
    assert 1 <= games["last10_matches"] <= 10
    assert 0.0 <= games["last10_games_pct"] <= 1.0
    assert games["last10_games_won"] <= games["last10_games_total"]
    if games["last52_matches"] > 0:
        assert 0.0 <= games["last52_games_pct"] <= 1.0
    else:
        assert games["last52_games_pct"] is None


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_form_last_name_alone_is_ambiguous():
    import psycopg

    from tennis_tools import get_player_form

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        result = get_player_form(conn, "Sinner")
    assert result.get("ambiguous") is True


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_form_unknown_player_returns_none():
    import psycopg

    from tennis_tools import get_player_form

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        assert get_player_form(conn, "Not A Real Player XYZ") is None


# ---------------------------------------------------------------------------
# Tool 3 tests
# ---------------------------------------------------------------------------


def test_h2h_summary_helper():
    from tennis_tools import _h2h_summary

    rows = [(1, "Hard"), (2, "Hard"), (1, "Clay"), (1, None)]
    result = _h2h_summary(rows, 1, 2)
    assert result["total"] == 4
    assert result["wins_a"] == 3
    assert result["wins_b"] == 1
    assert result["by_surface"]["Hard"] == {"total": 2, "wins_a": 1, "wins_b": 1}
    assert result["by_surface"]["Unknown"] == {"total": 1, "wins_a": 1, "wins_b": 0}


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_sinner_alcaraz_h2h_is_sane():
    # No exact count hardcoded: the Sinner-Alcaraz count is not known
    # in advance on this machine. Check structure and totals only.
    import psycopg

    from tennis_tools import get_head_to_head

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        result = get_head_to_head(conn, "Jannik Sinner", "Carlos Alcaraz")
    assert result is not None
    assert not result.get("ambiguous")
    assert result["total"] >= 1
    assert result["wins_a"] + result["wins_b"] == result["total"]
    assert sum(v["total"] for v in result["by_surface"].values()) == result["total"]
    assert len(result["last_5_meetings"]) == min(5, result["total"])
    for meeting in result["last_5_meetings"]:
        assert meeting["winner"] in (result["player_a"]["name"], result["player_b"]["name"])


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_h2h_ambiguous_side():
    import psycopg

    from tennis_tools import get_head_to_head

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        result = get_head_to_head(conn, "Sinner", "Carlos Alcaraz")
    assert result.get("ambiguous") is True
    assert result["side"] == "a"


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_h2h_unknown_returns_none():
    import psycopg

    from tennis_tools import get_head_to_head

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        assert get_head_to_head(conn, "Jannik Sinner", "Not A Real Player XYZ") is None


# ---------------------------------------------------------------------------
# Tool 2 retirement descriptor tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_sinner_retirement_descriptor_is_consistent():
    # No exact RET count hardcoded: Sinner's retirement history is
    # not known in advance on this machine. Check consistency only.
    import psycopg

    from tennis_tools import get_player_form

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        result = get_player_form(conn, "Jannik Sinner")
    ret = result["retirement"]
    assert ret["retirements_last10"] <= 10
    assert ret["retirements_last52"] <= result["win_form"]["last52_n"]
    assert ret["retirements_total"] >= ret["retirements_last52"]
    if ret["last_retirement_date"] is None:
        assert ret["days_since_last_retirement"] is None
        assert ret["matches_since_last_retirement"] is None
        assert ret["last_match_was_retirement"] is False
    else:
        assert ret["days_since_last_retirement"] >= 0
        assert ret["matches_since_last_retirement"] >= 0
        # Anchor is 2026-05-25, days since must match the date.
        from datetime import date

        anchor = date(2026, 5, 25)
        last_ret = date.fromisoformat(ret["last_retirement_date"])
        assert (anchor - last_ret).days == ret["days_since_last_retirement"]
        if ret["last_match_was_retirement"]:
            assert ret["retirements_last10"] >= 1


# ---------------------------------------------------------------------------
# Tool 4 tests
# ---------------------------------------------------------------------------


def test_win_pct_helper():
    from tennis_tools import _win_pct

    assert _win_pct(3, 4) == 0.75
    assert _win_pct(0, 0) is None


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_sinner_surface_record_sums_to_total_matches():
    # Sinner's total matches are known from Tool 1: 457.
    # Per-surface totals must sum to that, otherwise a surface
    # was dropped or double-counted.
    import psycopg

    from tennis_tools import get_player_surface_record

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        result = get_player_surface_record(conn, "Jannik Sinner")
    assert result is not None
    assert not result.get("ambiguous")
    assert result["total_matches"] == 457
    assert result["total_wins"] + result["total_losses"] == 457
    assert sum(v["total"] for v in result["by_surface"].values()) == 457
    for surface, info in result["by_surface"].items():
        assert info["wins"] + info["losses"] == info["total"]
        assert 0.0 <= info["win_pct"] <= 1.0


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_surface_ambiguous():
    import psycopg

    from tennis_tools import get_player_surface_record

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        result = get_player_surface_record(conn, "Sinner")
    assert result.get("ambiguous") is True


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_surface_unknown_returns_none():
    import psycopg

    from tennis_tools import get_player_surface_record

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        assert get_player_surface_record(conn, "Not A Real Player XYZ") is None


# ---------------------------------------------------------------------------
# Tool 5 tests
# ---------------------------------------------------------------------------


def test_calibration_helper():
    from tennis_tools import calibrated_prob

    # Week 2 calibration examples: 0.8 shrinks to about 0.752,
    # 0.9 shrinks to about 0.853, 0.5 stays 0.5 (b=0, symmetric).
    assert abs(calibrated_prob(0.8) - 0.7519) < 0.0001
    assert abs(calibrated_prob(0.9) - 0.8529) < 0.0001
    assert calibrated_prob(0.5) == 0.5
    # Calibration never flips the favorite (a=0.8 > 0, b=0).
    assert calibrated_prob(0.6) > 0.5
    assert calibrated_prob(0.4) < 0.5


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_sinner_alcaraz_hard_probability():
    import psycopg

    from tennis_tools import get_match_probability

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        result = get_match_probability(conn, "Jannik Sinner", "Carlos Alcaraz", "Hard")
    assert result is not None
    assert not result.get("ambiguous")
    # Sinner's Hard blended Elo must match Tool 1: 2247.6.
    assert abs(result["player_a"]["blended_elo"] - 2247.6) < 0.1
    assert result["player_a"]["surface_matches"] == 308
    # Probabilities must sum to 1, calibrated must be closer to 0.5.
    assert abs(result["prob_a_blended"] + result["prob_b_blended"] - 1.0) < 0.001
    assert abs(result["prob_a_calibrated"] + result["prob_b_calibrated"] - 1.0) < 0.001
    assert abs(result["prob_a_calibrated"] - 0.5) < abs(result["prob_a_blended"] - 0.5)
    assert result["favorite"] in ("Jannik Sinner", "Carlos Alcaraz")


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_sinner_alcaraz_overall_probability():
    # Overall only: Sinner 2283.1 vs Alcaraz 2176.7, diff 106.4.
    # Blended prob about 0.6485, calibrated about 0.6201.
    import psycopg

    from tennis_tools import get_match_probability

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        result = get_match_probability(conn, "Jannik Sinner", "Carlos Alcaraz", None)
    assert abs(result["prob_a_blended"] - 0.6485) < 0.001
    assert abs(result["prob_a_calibrated"] - 0.6201) < 0.001
    assert result["favorite"] == "Jannik Sinner"


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_match_ambiguous_and_unknown():
    import psycopg

    from tennis_tools import get_match_probability

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        result = get_match_probability(conn, "Sinner", "Carlos Alcaraz", "Hard")
        assert result.get("ambiguous") is True
        assert result["side"] == "a"
        assert get_match_probability(conn, "Jannik Sinner", "Not A Real Player XYZ", "Hard") is None
