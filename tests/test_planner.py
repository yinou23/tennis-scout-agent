"""Unit tests for Week 3 rule-based planner (scripts/planner.py).

Run from the project root ~/TSA:
    python -m pytest tests/test_planner.py -v

Pure parser tests run everywhere, no database needed.
The database end-to-end test runs only when DATABASE_URL is set,
otherwise it is skipped. On the Mac, it asks the Sinner vs Alcaraz
Hard question end-to-end and checks the answer contains the
Tool 5 numbers (favorite Jannik Sinner, calibrated 0.6378),
proving the answer came from the tools, not from a hardcoded string.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from planner import build_answer, execute_plan, parse_question


def test_parse_match_hard():
    assert parse_question("Jannik Sinner vs Carlos Alcaraz Hard") == {
        "tool": "match",
        "player_a": "Jannik Sinner",
        "player_b": "Carlos Alcaraz",
        "surface": "Hard",
    }


def test_parse_match_chinese_clay():
    assert parse_question("Sinner 对 Alcaraz 红土") == {
        "tool": "match",
        "player_a": "Sinner",
        "player_b": "Alcaraz",
        "surface": "Clay",
    }


def test_parse_match_no_surface():
    assert parse_question("Jannik Sinner vs Carlos Alcaraz") == {
        "tool": "match",
        "player_a": "Jannik Sinner",
        "player_b": "Carlos Alcaraz",
        "surface": None,
    }


def test_parse_match_or_question():
    assert parse_question("Who is more likely to win, Sinner or Alcaraz on hard?") == {
        "tool": "match",
        "player_a": "Sinner",
        "player_b": "Alcaraz",
        "surface": "Hard",
    }


def test_parse_h2h():
    assert parse_question("Jannik Sinner vs Carlos Alcaraz 交手") == {
        "tool": "h2h",
        "player_a": "Jannik Sinner",
        "player_b": "Carlos Alcaraz",
    }


def test_parse_elo_form_surface():
    assert parse_question("Jannik Sinner Elo") == {"tool": "elo", "player": "Jannik Sinner"}
    assert parse_question("Jannik Sinner 状态") == {"tool": "form", "player": "Jannik Sinner"}
    assert parse_question("Jannik Sinner 场地战绩") == {"tool": "surface", "player": "Jannik Sinner"}


def test_parse_unknown_returns_none():
    assert parse_question("你好") is None
    assert parse_question("") is None


def test_build_answer_no_template():
    answer = build_answer(None, {"error": "no_template_matched"})
    assert "Jannik Sinner vs Carlos Alcaraz Hard" in answer


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_end_to_end_sinner_alcaraz_hard():
    import psycopg

    from planner import parse_question as pq

    plan = pq("Jannik Sinner vs Carlos Alcaraz Hard")
    assert plan["tool"] == "match"
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        outputs = execute_plan(conn, plan)
    answer = build_answer(plan, outputs)
    # Tool 5 numbers, verified in the previous step: favorite Sinner,
    # calibrated 0.6378, blended 0.6698. If the answer does not contain
    # them, it did not come from the tools.
    assert "Jannik Sinner" in answer
    assert "0.6378" in answer
    assert "0.6698" in answer
    assert "2026-05-25" in answer


# ---------------------------------------------------------------------------
# Language matching tests
# ---------------------------------------------------------------------------


def test_parse_chinese_question_phrase_suffix():
    assert parse_question("Jannik Sinner 对 Carlos Alcaraz 硬地谁是热门") == {
        "tool": "match",
        "player_a": "Jannik Sinner",
        "player_b": "Carlos Alcaraz",
        "surface": "Hard",
    }


def test_detect_language():
    from planner import detect_language

    assert detect_language("Jannik Sinner vs Carlos Alcaraz Hard") == "en"
    assert detect_language("Sinner 对 Alcaraz 硬地") == "zh"
    assert detect_language(None) == "en"
    assert detect_language("") == "en"


def test_build_answer_language_follows_question():
    from planner import build_answer

    plan = {"tool": "match", "player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz", "surface": "Hard"}
    outputs = {
        "match": {
            "player_a": {"name": "Jannik Sinner", "overall_elo": 2283.1, "blended_elo": 2247.6},
            "player_b": {"name": "Carlos Alcaraz", "overall_elo": 2176.7, "blended_elo": 2124.7},
            "favorite": "Jannik Sinner",
            "prob_a_calibrated": 0.6378,
            "prob_b_calibrated": 0.3622,
            "prob_a_blended": 0.6698,
            "prob_b_blended": 0.3302,
            "surface": "Hard",
        },
        "h2h": {"total": 17, "wins_a": 7, "wins_b": 10},
    }
    en = build_answer(plan, outputs, "Jannik Sinner vs Carlos Alcaraz Hard")
    zh = build_answer(plan, outputs, "Sinner 对 Alcaraz 硬地")
    assert "Favorite is Jannik Sinner" in en
    assert "热门是" in zh
    assert "0.6378" in en and "0.6378" in zh
    # Default, no question, is English.
    default = build_answer(None, {"error": "no_template_matched"})
    assert "Please rephrase" in default
