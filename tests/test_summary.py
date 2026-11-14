"""Unit tests for Week 9 Step 3 structured summary.

Run from the project root ~/TSA:
    python -m pytest tests/test_summary.py -v

No database, no LLM, no network. Tests use a fake report dict
shaped like build_scout_report returns for Sinner vs Alcaraz
Hard, with the locked values: calibrated 0.6378 / 0.3622,
blended 0.6698 / 0.3302, Elo Sinner 2283.1 / 2236.0 / 2247.6,
Alcaraz 2176.7 / 2099.1 / 2124.7, H2H 17, Sinner 7, Alcaraz 10.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summary import build_summary


def _fake_report():
    return {
        "text": "Scouting report: Jannik Sinner vs Carlos Alcaraz (Hard)",
        "surface": "Hard",
        "surface_applied": True,
        "favorite": "Jannik Sinner",
        "prob_a_calibrated": 0.6378,
        "prob_b_calibrated": 0.3622,
        "prob_a_blended": 0.6698,
        "prob_b_blended": 0.3302,
        "data_as_of": "2026-05-25",
        "tools": {
            "match": {
                "player_a": {"name": "Jannik Sinner", "overall_elo": 2283.1, "surface_elo": 2236.0, "surface_matches": 308, "blended_elo": 2247.6},
                "player_b": {"name": "Carlos Alcaraz", "overall_elo": 2176.7, "surface_elo": 2099.1, "surface_matches": 203, "blended_elo": 2124.7},
                "prob_a_calibrated": 0.6378,
                "prob_b_calibrated": 0.3622,
                "prob_a_blended": 0.6698,
                "prob_b_blended": 0.3302,
                "favorite": "Jannik Sinner",
                "surface": "Hard",
            },
            "elo_a": {"name": "Jannik Sinner", "overall_elo": 2283.1, "matches_played": 457},
            "elo_b": {"name": "Carlos Alcaraz", "overall_elo": 2176.7, "matches_played": 371},
            "form_a": {
                "win_form": {"last10_pct": 0.9, "last10_wins": 9, "last10_n": 10, "recent2m_pct": 0.9, "recent2m_wins": 9, "recent2m_n": 10, "last52_pct": 0.9121, "last52_wins": 83, "last52_n": 91},
                "games_form": {"last10_games_pct": 0.6273, "last10_games_won": 138, "last10_games_total": 220, "recent2m_games_pct": 0.6273, "recent2m_games_won": 138, "recent2m_games_total": 220, "last52_games_pct": 0.635, "last52_games_won": 1298, "last52_games_total": 2044},
                "retirement": {"last_retirement_date": "2025-10-01", "days_since_last_retirement": 236, "matches_since_last_retirement": 55, "retirements_last10": 0, "retirements_last52": 2, "retirements_total": 7},
            },
            "form_b": {
                "win_form": {"last10_pct": 0.6, "last10_wins": 6, "last10_n": 10, "recent2m_pct": 0.7143, "recent2m_wins": 5, "recent2m_n": 7, "last52_pct": 0.8873, "last52_wins": 63, "last52_n": 71},
                "games_form": {"last10_games_pct": 0.5742, "last10_games_won": 120, "last10_games_total": 209, "recent2m_games_pct": 0.6239, "recent2m_games_won": 73, "recent2m_games_total": 117, "last52_games_pct": 0.5957, "last52_games_won": 1099, "last52_games_total": 1845},
                "retirement": {"last_retirement_date": "2024-02-19", "days_since_last_retirement": 826, "matches_since_last_retirement": 164, "retirements_last10": 0, "retirements_last52": 0, "retirements_total": 3},
            },
            "record_a": {
                "by_surface": {"Hard": {"wins": 251, "losses": 57, "total": 308, "win_pct": 0.8149}},
                "total_matches": 457, "total_wins": 363, "total_losses": 94,
            },
            "record_b": {
                "by_surface": {"Hard": {"wins": 159, "losses": 44, "total": 203, "win_pct": 0.7833}},
                "total_matches": 371, "total_wins": 302, "total_losses": 69,
            },
            "h2h": {
                "total": 17, "wins_a": 7, "wins_b": 10,
                "by_surface": {"Hard": {"total": 10, "wins_a": 3, "wins_b": 7}},
            },
        },
    }


def test_summary_top_level_probabilities():
    summary = build_summary(_fake_report())
    assert summary["player_a"] == "Jannik Sinner"
    assert summary["player_b"] == "Carlos Alcaraz"
    assert summary["surface"] == "Hard"
    assert summary["surface_applied"] is True
    assert summary["favorite"] == "Jannik Sinner"
    assert summary["prob_a_calibrated"] == 0.6378
    assert summary["prob_b_calibrated"] == 0.3622
    assert summary["prob_a_blended"] == 0.6698
    assert summary["prob_b_blended"] == 0.3302
    assert summary["data_as_of"] == "2026-05-25"


def test_summary_elo_per_player():
    summary = build_summary(_fake_report())
    assert summary["a"]["elo"] == {"overall": 2283.1, "surface_elo": 2236.0, "blended": 2247.6, "surface_matches": 308, "matches_played": 457}
    assert summary["b"]["elo"] == {"overall": 2176.7, "surface_elo": 2099.1, "blended": 2124.7, "surface_matches": 203, "matches_played": 371}


def test_summary_record_per_player():
    summary = build_summary(_fake_report())
    assert summary["a"]["record"]["surface"] == {"wins": 251, "losses": 57, "total": 308, "win_pct": 0.8149}
    assert summary["a"]["record"]["total_matches"] == 457
    assert summary["a"]["record"]["total_wins"] == 363
    assert summary["a"]["record"]["total_losses"] == 94
    assert summary["b"]["record"]["surface"] == {"wins": 159, "losses": 44, "total": 203, "win_pct": 0.7833}
    assert summary["b"]["record"]["total_matches"] == 371


def test_summary_form_per_player():
    summary = build_summary(_fake_report())
    form_a = summary["a"]["form"]
    assert form_a["win10"] == 0.9
    assert form_a["win2m"] == 0.9
    assert form_a["win52"] == 0.9121
    assert form_a["win10_wins"] == 9
    assert form_a["win2m_wins"] == 9
    assert form_a["win52_wins"] == 83
    assert form_a["games10"] == 0.6273
    assert form_a["games2m"] == 0.6273
    assert form_a["games52"] == 0.635
    assert form_a["games10_won"] == 138
    assert form_a["games2m_won"] == 138
    assert form_a["games52_won"] == 1298
    assert form_a["last_retirement_date"] == "2025-10-01"
    assert form_a["days_since_last_retirement"] == 236
    assert form_a["matches_since_last_retirement"] == 55
    assert form_a["retirements_total"] == 7
    form_b = summary["b"]["form"]
    assert form_b["win10"] == 0.6
    assert form_b["win52"] == 0.8873
    assert form_b["last_retirement_date"] == "2024-02-19"
    assert form_b["days_since_last_retirement"] == 826
    assert form_b["retirements_total"] == 3


def test_summary_h2h():
    summary = build_summary(_fake_report())
    assert summary["h2h"] == {"total": 17, "wins_a": 7, "wins_b": 10, "surface": {"total": 10, "wins_a": 3, "wins_b": 7}}


def test_summary_is_json_safe():
    summary = build_summary(_fake_report())
    text = json.dumps(summary)
    assert "0.6378" in text
    assert "Jannik Sinner" in text
    assert json.loads(text) == summary


def test_summary_none_and_error_report_return_none():
    assert build_summary(None) is None
    assert build_summary("not a dict") is None
    assert build_summary({"error": "player_not_found:elo_a"}) is None
    assert build_summary({}) is not None
    assert build_summary({})["player_a"] is None


def test_summary_missing_fields_are_none_not_invented():
    report = _fake_report()
    del report["tools"]["form_a"]["retirement"]["last_retirement_date"]
    del report["tools"]["record_b"]["by_surface"]["Hard"]
    summary = build_summary(report)
    assert summary["a"]["form"]["last_retirement_date"] is None
    assert summary["b"]["record"]["surface"] is None


def test_summary_no_surface_gives_none_surface_fields():
    report = _fake_report()
    report["surface"] = None
    report["surface_applied"] = False
    summary = build_summary(report)
    assert summary["surface"] is None
    assert summary["a"]["record"]["surface"] is None
    assert summary["h2h"]["surface"] is None


def test_api_response_includes_summary_when_report_present():
    from api import build_report_response

    state = {
        "question": "Jannik Sinner vs Carlos Alcaraz Hard",
        "final_text": "Final probability (calibrated): Jannik Sinner 0.6378, Carlos Alcaraz 0.3622",
        "planner_source": "llm",
        "plan": {"tool": "match", "player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz", "surface": "Hard"},
        "verification": {"ok": True, "unexpected_numbers": [], "missing_required": [], "tag_error": None},
        "llm_model": "fake",
        "_used_langgraph": True,
        "error": None,
        "report": _fake_report(),
    }
    status, payload = build_report_response(state)
    assert status == 200
    assert payload["summary"]["prob_a_calibrated"] == 0.6378
    assert payload["summary"]["a"]["elo"]["overall"] == 2283.1
    assert payload["summary"]["h2h"]["total"] == 17


def test_api_response_without_report_still_200_with_summary_none():
    from api import build_report_response

    state = {
        "question": "Jannik Sinner vs Carlos Alcaraz Hard",
        "final_text": "Final probability (calibrated): Jannik Sinner 0.6378, Carlos Alcaraz 0.3622",
        "planner_source": "llm",
        "plan": {"tool": "match", "player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz", "surface": "Hard"},
        "verification": {"ok": True, "unexpected_numbers": [], "missing_required": [], "tag_error": None},
        "llm_model": "fake",
        "_used_langgraph": True,
        "error": None,
    }
    status, payload = build_report_response(state)
    assert status == 200
    assert payload["summary"] is None
