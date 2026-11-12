"""Unit tests for Week 4 Step 5 LLM planner in the full graph.

Run from the project root ~/TSA:
    python -m pytest tests/test_full_graph.py -v

Pure tests run everywhere, no database, no LangGraph, and no real
LLM API call. They inject fake llm_caller, fake planner_with_source,
and fake writer functions. A fake planner that returns a correct
plan proves the graph plumbing and the planner_source labelling,
not LLM intelligence. The database graph tests run only when
the SQLite database file exists, otherwise they are skipped. On the Mac, with
.env loaded, they run the full graph with a fake planner and a
fake writer (not a real API call) for Sinner vs Alcaraz Hard, and
check the known Step 2 numbers, planner_source, and verifier ok.
A real LLM planner demo is a manual run of scripts/full_graph.py
with a key in .env, and it must show planner_source "llm" and
verification ok, otherwise it is not a clean LLM planner pass.
That manual demo is NOT in this test file, on purpose, so tests
do not burn quota and do not flake on a 503.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from db import get_connection, get_sqlite_path

from full_graph import plan_with_llm_graph, run_full_graph


def _fake_templated_report():
    return {
        "text": (
            "Final probability (calibrated): Jannik Sinner 0.6378, Carlos Alcaraz 0.3622 "
            "[Surface blend: Hard, prior=100] [Calibration: a=0.8]\n"
            "Blended probability: Jannik Sinner 0.6698, Carlos Alcaraz 0.3302. "
            "Elo: Jannik Sinner overall 2283.1, Hard blended 2247.6; "
            "Carlos Alcaraz overall 2176.7, Hard blended 2124.7. "
            "Head-to-head: 17 meetings, Jannik Sinner 7 wins, Carlos Alcaraz 10 wins. "
            "Data as of 2026-05-25, not live."
        ),
        "values": [
            "0.6378", "0.3622", "0.6698", "0.3302",
            "2283.1", "2247.6", "2176.7", "2124.7",
            "17", "7", "10",
        ],
        "tags": ["[Surface blend: Hard, prior=100]", "[Calibration: a=0.8]"],
        "surface": "Hard",
        "surface_applied": True,
        "prob_a_calibrated": 0.6378,
        "prob_b_calibrated": 0.3622,
        "prob_a_blended": 0.6698,
        "prob_b_blended": 0.3302,
        "tools": {
            "match": {
                "player_a": {"overall_elo": 2283.1, "blended_elo": 2247.6},
                "player_b": {"overall_elo": 2176.7, "blended_elo": 2124.7},
            },
            "h2h": {"total": 17, "wins_a": 7, "wins_b": 10},
        },
    }


def _correct_llm_text():
    return (
        "Jannik Sinner is the favorite on Hard. "
        "Final probability (calibrated): Jannik Sinner 0.6378, Carlos Alcaraz 0.3622 "
        "[Surface blend: Hard, prior=100] [Calibration: a=0.8] "
        "Blended probability was Jannik Sinner 0.6698, Carlos Alcaraz 0.3302. "
        "Elo: Jannik Sinner overall 2283.1, Hard blended 2247.6; "
        "Carlos Alcaraz overall 2176.7, Hard blended 2124.7. "
        "Head-to-head: 17 meetings, Jannik Sinner 7 wins, Carlos Alcaraz 10 wins. "
        "Data as of 2026-05-25, not live."
    )


def _match_plan():
    return {"tool": "match", "player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz", "surface": "Hard"}


def test_planner_valid_llm_returns_llm_source():
    plan, source = plan_with_llm_graph(
        "Jannik Sinner vs Carlos Alcaraz Hard",
        llm_caller=lambda q: '{"tool": "match", "player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz", "surface": "Hard"}',
    )
    assert source == "llm"
    assert plan == _match_plan()


def test_planner_invalid_llm_falls_back_to_rule_labelled():
    plan, source = plan_with_llm_graph(
        "Jannik Sinner vs Carlos Alcaraz Hard",
        llm_caller=lambda q: "not json at all",
    )
    assert source == "rule_fallback"
    assert plan == _match_plan()


def test_planner_raising_llm_falls_back_to_rule_labelled():
    def raising_caller(q):
        raise RuntimeError("llm_http_error:503")

    plan, source = plan_with_llm_graph("Jannik Sinner vs Carlos Alcaraz Hard", llm_caller=raising_caller)
    assert source == "rule_fallback"
    assert plan == _match_plan()


def test_planner_null_llm_falls_back_to_rule_labelled():
    plan, source = plan_with_llm_graph(
        "Jannik Sinner vs Carlos Alcaraz Hard",
        llm_caller=lambda q: "null",
    )
    assert source == "rule_fallback"
    assert plan == _match_plan()


def test_planner_no_key_returns_no_key_rule_without_calling_api():
    old_llm = os.environ.pop("LLM_API_KEY", None)
    old_gemini = os.environ.pop("GEMINI_API_KEY", None)
    try:
        plan, source = plan_with_llm_graph("Jannik Sinner vs Carlos Alcaraz Hard", llm_caller=None)
    finally:
        if old_llm is not None:
            os.environ["LLM_API_KEY"] = old_llm
        if old_gemini is not None:
            os.environ["GEMINI_API_KEY"] = old_gemini
    assert source == "no_key_rule"
    assert plan == _match_plan()


def _fake_report_builder(conn, player_a, player_b, surface, question):
    return _fake_templated_report()


def _fake_llm_writer(report, question):
    return {"text": _correct_llm_text(), "model": "fake", "source": "llm"}


def test_full_graph_llm_planner_happy_path_sequential():
    final = run_full_graph(
        None,
        "Jannik Sinner vs Carlos Alcaraz Hard",
        planner_with_source=lambda q: (_match_plan(), "llm"),
        report_builder=_fake_report_builder,
        llm_writer_func=_fake_llm_writer,
        use_langgraph=False,
    )
    assert final.get("error") is None, final
    assert final["planner_source"] == "llm"
    assert final["verification"]["ok"] is True
    assert "0.6378" in final["final_text"]


def test_full_graph_rule_fallback_planner_still_delivers_labelled():
    final = run_full_graph(
        None,
        "Jannik Sinner vs Carlos Alcaraz Hard",
        planner_with_source=lambda q: (_match_plan(), "rule_fallback"),
        report_builder=_fake_report_builder,
        llm_writer_func=_fake_llm_writer,
        use_langgraph=False,
    )
    assert final.get("error") is None, final
    assert final["planner_source"] == "rule_fallback"
    assert final["planner_source"] != "llm"
    assert final["verification"]["ok"] is True


def test_full_graph_planner_none_returns_error_not_report():
    final = run_full_graph(
        None,
        "some question no template matches at all zzz",
        planner_with_source=lambda q: (None, "rule_fallback"),
        report_builder=_fake_report_builder,
        llm_writer_func=_fake_llm_writer,
        use_langgraph=False,
    )
    assert final.get("error") == "no_template_matched"
    assert final["planner_source"] == "rule_fallback"
    assert "0.6378" not in final.get("final_text", "") or "ERROR" in final.get("final_text", "")


def test_full_graph_non_match_planner_returns_error_not_report():
    final = run_full_graph(
        None,
        "Jannik Sinner vs Carlos Alcaraz h2h",
        planner_with_source=lambda q: ({"tool": "h2h", "player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz"}, "llm"),
        report_builder=_fake_report_builder,
        llm_writer_func=_fake_llm_writer,
        use_langgraph=False,
    )
    assert final.get("error") == "unsupported_plan:h2h"
    assert "0.6378" not in final.get("final_text", "") or "ERROR" in final.get("final_text", "")


@pytest.mark.skipif(not get_sqlite_path().exists(), reason="SQLite database file not found; set SQLITE_PATH or build it with scripts/pg_to_sqlite.py")
def test_full_graph_hard_end_to_end_with_fake_planner_and_writer():

    from full_graph import run_full_graph

    with get_connection() as conn:
        final = run_full_graph(
            conn,
            "Jannik Sinner vs Carlos Alcaraz Hard",
            planner_with_source=lambda q: (_match_plan(), "llm"),
            llm_writer_func=lambda report, question: {
                "text": (
                    "Jannik Sinner is the favorite on Hard. "
                    "Final probability (calibrated): Jannik Sinner 0.6378, Carlos Alcaraz 0.3622 "
                    "[Surface blend: Hard, prior=100] [Calibration: a=0.8] "
                    "Blended probability was Jannik Sinner 0.6698, Carlos Alcaraz 0.3302. "
                    "Elo: Jannik Sinner overall 2283.1, Hard blended 2247.6; "
                    "Carlos Alcaraz overall 2176.7, Hard blended 2124.7. "
                    "Head-to-head: 17 meetings, Jannik Sinner 7 wins, Carlos Alcaraz 10 wins. "
                    "Data as of 2026-05-25, not live."
                ),
                "model": "fake",
                "source": "llm",
            },
        )
    assert final.get("error") is None, final
    assert final["planner_source"] == "llm"
    assert final["verification"]["ok"] is True, final["verification"]
    assert "0.6378" in final["final_text"]
    assert final["_used_langgraph"] is True, "expected real LangGraph on Mac, got fallback"
