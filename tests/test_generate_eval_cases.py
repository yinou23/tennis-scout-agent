"""Unit tests for Week 6 Step 3 eval expansion generator, v2.

Run from the project root ~/TSA:
    python -m pytest tests/test_generate_eval_cases.py -v

Pure tests run everywhere, no database, no LLM call. They test
the extraction helpers with fake tool outputs, the question
parsing for the 25 new match pairs, and the new case id list
for collisions with the canonical 30 ids. The canonical 30 ids
are hardcoded here, NOT read from eval/cases.jsonl, because
after the expanded file is moved to canonical, cases.jsonl
contains both the old 30 and the new 70, so reading cases.jsonl
as "existing" would falsely report a collision. That was the
Step 2 test bug, fixed in Step 2 fix2, and kept here.
The file check test is count-agnostic: it checks the expanded
file if it exists, otherwise the canonical file, asserts the
first 30 ids are the canonical 30 in order, asserts the new
70 ids are present, asserts no duplicate ids, and asserts
total at least 100, never exactly 100, so a future expansion
does not break it by hardcoding a total.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_eval_cases import (
    NEW_MATCHES,
    NEW_PLAYERS,
    extract_elo_expected,
    extract_form_expected,
    extract_surface_expected,
)
from planner import parse_question

CANONICAL_30_IDS = [
    "match-hard-sinner-alcaraz",
    "match-overall-sinner-alcaraz",
    "elo-sinner",
    "elo-alcaraz",
    "form-sinner",
    "form-alcaraz",
    "h2h-sinner-alcaraz",
    "surface-sinner",
    "surface-alcaraz",
    "ambiguous-sinner",
    "unknown-player",
    "elo-novak-djokovic",
    "form-novak-djokovic",
    "surface-novak-djokovic",
    "elo-roger-federer",
    "form-roger-federer",
    "surface-roger-federer",
    "elo-rafael-nadal",
    "form-rafael-nadal",
    "surface-rafael-nadal",
    "elo-alexander-zverev",
    "form-alexander-zverev",
    "surface-alexander-zverev",
    "elo-daniil-medvedev",
    "form-daniil-medvedev",
    "surface-daniil-medvedev",
    "match-hard-novak-djokovic-roger-federer",
    "match-clay-novak-djokovic-rafael-nadal",
    "match-grass-roger-federer-rafael-nadal",
    "match-hard-alexander-zverev-daniil-medvedev",
]


def _new_ids():
    new_ids = []
    for player in NEW_PLAYERS:
        slug = player.lower().replace(" ", "-")
        new_ids.extend([f"elo-{slug}", f"form-{slug}", f"surface-{slug}"])
    for player_a, player_b, surface in NEW_MATCHES:
        slug_a = player_a.lower().replace(" ", "-")
        slug_b = player_b.lower().replace(" ", "-")
        new_ids.append(f"match-{surface.lower()}-{slug_a}-{slug_b}")
    return new_ids


def test_new_players_and_matches_counts():
    assert len(NEW_PLAYERS) == 15
    assert len(NEW_MATCHES) == 25
    # 15 players x 3 cases plus 25 matches equals 70 new cases.
    assert 15 * 3 + 25 == 70


def test_new_ids_do_not_collide_with_canonical_30():
    assert len(CANONICAL_30_IDS) == 30
    new_ids = _new_ids()
    assert len(new_ids) == 70
    assert len(set(new_ids)) == 70
    assert not (set(new_ids) & set(CANONICAL_30_IDS))


def test_extract_elo_expected_keeps_only_blended():
    tool_output = {
        "overall_elo": 2033.4,
        "matches_played": 100,
        "surfaces": {
            "Hard": {"surface_elo": 2000.0, "surface_matches": 60, "blended_elo": 2010.0},
            "Carpet": {"surface_elo": None, "surface_matches": 0, "blended_elo": 2033.4},
        },
    }
    expected = extract_elo_expected(tool_output)
    assert expected["overall_elo"] == 2033.4
    assert expected["matches_played"] == 100
    assert expected["surfaces"]["Hard"] == {"blended_elo": 2010.0}
    assert expected["surfaces"]["Carpet"] == {"blended_elo": 2033.4}


def test_extract_form_and_surface_expected_pass_through():
    form_output = {
        "win_form": {"last10_pct": 0.8, "last10_wins": 8, "last10_n": 10},
        "games_form": {"last10_games_pct": 0.6, "last10_games_won": 60, "last10_games_total": 100},
        "retirement": {"last_retirement_date": None, "retirements_total": 0},
    }
    assert extract_form_expected(form_output) == form_output
    surface_output = {
        "by_surface": {"Hard": {"wins": 10, "losses": 5, "total": 15, "win_pct": 0.6667}},
        "total_matches": 15,
        "total_wins": 10,
        "total_losses": 5,
    }
    assert extract_surface_expected(surface_output) == surface_output


def test_new_match_questions_parse_to_same_plan():
    for player_a, player_b, surface in NEW_MATCHES:
        question = f"{player_a} vs {player_b} {surface}"
        plan = parse_question(question)
        assert plan is not None, question
        assert plan["tool"] == "match", question
        assert plan["player_a"] == player_a, question
        assert plan["player_b"] == player_b, question
        assert plan["surface"] == surface, question


def test_eval_file_contains_canonical_first_and_new_ids():
    # After the move to canonical, eval/cases_expanded.jsonl no
    # longer exists, it was moved, not copied, so check the
    # expanded file if it exists, otherwise the canonical file.
    import json

    expanded_path = Path(__file__).resolve().parents[1] / "eval" / "cases_expanded.jsonl"
    canonical_path = Path(__file__).resolve().parents[1] / "eval" / "cases.jsonl"
    path = expanded_path if expanded_path.exists() else canonical_path
    if not path.exists():
        pytest.skip("no eval cases file found on this machine")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [c["id"] for c in cases]
    assert ids[:30] == CANONICAL_30_IDS
    assert len(ids) >= 100
    assert len(set(ids)) == len(ids)
    assert set(_new_ids()) <= set(ids)
