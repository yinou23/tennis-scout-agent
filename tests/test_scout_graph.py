"""Unit tests for Week 4 Step 3 graph skeleton.

Run from the project root ~/TSA:
    python -m pytest tests/test_scout_graph.py -v

Pure tests run everywhere, no database and no LangGraph needed.
They inject fake planner, report builder, and verifier functions
and force the sequential graph path, so they check node logic,
not the LangGraph install.
The database end-to-end test runs only when DATABASE_URL is set,
otherwise it is skipped. On the Mac, with .env loaded, it runs the
real graph for Sinner vs Alcaraz Hard and checks the known Week 4
Step 2 numbers: favorite Jannik Sinner, calibrated 0.6378, blended
0.6698, verifier ok. If that check fails, the graph must not be
trusted yet. On the Mac the real LangGraph path must be used; the
test prints used_langgraph in the failure message so a fallback
pass is not mistaken for a LangGraph pass.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scout_graph import build_scout_graph, run_scout_graph


def _fake_planner_match(question):
    return {"tool": "match", "player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz", "surface": "Hard"}


def _fake_planner_none(question):
    return None


def _fake_planner_h2h(question):
    return {"tool": "h2h", "player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz"}


def _fake_report(conn, player_a, player_b, surface, question):
    return {
        "text": "Final probability (calibrated): Jannik Sinner 0.6378, Carlos Alcaraz 0.3622",
        "values": ["0.6378", "0.3622"],
        "tags": ["[Surface blend: Hard, prior=100]", "[Calibration: a=0.8]"],
        "surface_applied": True,
        "favorite": "Jannik Sinner",
        "prob_a_calibrated": 0.6378,
        "prob_b_calibrated": 0.3622,
        "prob_a_blended": 0.6698,
        "prob_b_blended": 0.3302,
    }


def _fake_report_error(conn, player_a, player_b, surface, question):
    return {"error": "player_not_found:match"}


def _fake_verifier_ok(report):
    return {"ok": True, "missing_values": [], "unexpected_numbers": [], "tag_error": None}


def _fake_verifier_fail(report):
    return {"ok": False, "missing_values": ["0.9999"], "unexpected_numbers": [], "tag_error": None}


def test_graph_happy_path_sequential():
    final = run_scout_graph(
        None,
        "Jannik Sinner vs Carlos Alcaraz Hard",
        planner_func=_fake_planner_match,
        report_builder=_fake_report,
        verifier_func=_fake_verifier_ok,
        use_langgraph=False,
    )
    assert final.get("error") is None
    assert final["plan"]["tool"] == "match"
    assert final["verification"]["ok"] is True
    assert "0.6378" in final["final_text"]
    assert final["_used_langgraph"] is False


def test_graph_planner_no_template_returns_error_not_report():
    final = run_scout_graph(
        None,
        "hello",
        planner_func=_fake_planner_none,
        report_builder=_fake_report,
        verifier_func=_fake_verifier_ok,
        use_langgraph=False,
    )
    assert final.get("error") == "no_template_matched"
    assert "0.6378" not in final.get("final_text", "")


def test_graph_non_match_plan_returns_error_not_report():
    final = run_scout_graph(
        None,
        "Jannik Sinner vs Carlos Alcaraz h2h",
        planner_func=_fake_planner_h2h,
        report_builder=_fake_report,
        verifier_func=_fake_verifier_ok,
        use_langgraph=False,
    )
    assert final.get("error") == "unsupported_plan:h2h"
    assert "0.6378" not in final.get("final_text", "")


def test_graph_report_error_returns_error_not_report():
    final = run_scout_graph(
        None,
        "Jannik Sinner vs Carlos Alcaraz Hard",
        planner_func=_fake_planner_match,
        report_builder=_fake_report_error,
        verifier_func=_fake_verifier_ok,
        use_langgraph=False,
    )
    assert "player_not_found" in final.get("error", "")
    assert "0.6378" not in final.get("final_text", "")


def test_graph_verifier_fail_blocks_writer():
    final = run_scout_graph(
        None,
        "Jannik Sinner vs Carlos Alcaraz Hard",
        planner_func=_fake_planner_match,
        report_builder=_fake_report,
        verifier_func=_fake_verifier_fail,
        use_langgraph=False,
    )
    assert final.get("error") == "verification_failed"
    assert "0.6378" not in final.get("final_text", "")
    assert "verification failed" in final.get("final_text", "")


def test_build_graph_fallback_when_langgraph_missing():
    # On a machine without langgraph, use_langgraph=True must fall
    # back to the sequential graph, not raise. This test is skipped
    # on the Mac where langgraph is installed, because there the
    # real path is expected and the fallback is not exercised.
    try:
        import langgraph  # noqa: F401
    except ImportError:
        graph, used = build_scout_graph(
            None,
            planner_func=_fake_planner_match,
            report_builder=_fake_report,
            verifier_func=_fake_verifier_ok,
            use_langgraph=True,
        )
        assert used is False
        final = graph.invoke({"question": "Jannik Sinner vs Carlos Alcaraz Hard"})
        assert "0.6378" in final["final_text"]
    else:
        pytest.skip("langgraph installed, fallback not exercised here")


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_graph_sinner_alcaraz_hard_end_to_end():
    import psycopg

    from scout_graph import run_scout_graph

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        final = run_scout_graph(conn, "Jannik Sinner vs Carlos Alcaraz Hard")
    assert final.get("error") is None, final
    assert final["verification"]["ok"] is True, final["verification"]
    assert "0.6378" in final["final_text"]
    assert "0.6698" in final["final_text"]
    assert "Jannik Sinner" in final["final_text"]
    # On the Mac, the real LangGraph path must be the one that
    # passed. A fallback pass here would be a false pass for Step 3.
    assert final["_used_langgraph"] is True, "expected real LangGraph on Mac, got fallback"


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_graph_chinese_hard_end_to_end():
    import psycopg

    from scout_graph import run_scout_graph

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        final = run_scout_graph(conn, "Jannik Sinner 对 Carlos Alcaraz 硬地谁是热门")
    assert final.get("error") is None, final
    assert "场地修正" in final["final_text"]
    assert "0.6378" in final["final_text"]
    assert final["_used_langgraph"] is True, "expected real LangGraph on Mac, got fallback"
