"""Unit tests for Week 5 Step 3 MCP server.

Run from the project root ~/TSA:
    python -m pytest tests/test_mcp_server.py -v

Pure tests run everywhere, no database, no MCP SDK, and no real
LLM call. They test the tool name list, JSON error handling when
DATABASE_URL is missing, and the scouting report implementation
with a fake graph runner. The database tool tests run only when
DATABASE_URL is set, otherwise they are skipped. On the Mac, with
.env loaded, they run the five deterministic tools against the
real database for Sinner vs Alcaraz Hard, and check the known
Step 2 numbers, with no LLM call. A real MCP Inspector test is
a manual run on the Mac, with the SDK installed, and it must
show the six tools and return verification ok for a scouting
report, otherwise it is not a clean Week 5 Step 3 pass. That
manual Inspector test is NOT in this test file, on purpose, so
tests do not depend on the Inspector or burn LLM quota.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mcp_server import (
    TOOL_NAMES,
    _scouting_report_impl,
    get_player_elo_tool,
)


def test_tool_names_are_six_expected_tools():
    assert TOOL_NAMES == (
        "get_player_elo",
        "get_player_form",
        "get_head_to_head",
        "get_player_surface_record",
        "get_match_probability",
        "generate_scouting_report",
    )


def test_elo_tool_without_database_url_returns_error_json_not_crash():
    old = os.environ.pop("DATABASE_URL", None)
    try:
        payload = json.loads(get_player_elo_tool("Jannik Sinner"))
    finally:
        if old is not None:
            os.environ["DATABASE_URL"] = old
    assert "error" in payload
    assert payload["error"].startswith("database_connection_failed")


def _fake_success_state():
    return {
        "final_text": (
            "Jannik Sinner is the favorite on Hard. "
            "Final probability (calibrated): Jannik Sinner 0.6378, Carlos Alcaraz 0.3622 "
            "[Surface blend: Hard, prior=100] [Calibration: a=0.8] "
            "Blended probability was Jannik Sinner 0.6698, Carlos Alcaraz 0.3302."
        ),
        "planner_source": "llm",
        "plan": {"tool": "match", "player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz", "surface": "Hard"},
        "verification": {"ok": True, "unexpected_numbers": [], "missing_required": [], "tag_error": None},
        "llm_model": "fake",
        "_used_langgraph": True,
        "error": None,
    }


def test_scouting_report_impl_success_with_fake_runner():
    payload = json.loads(_scouting_report_impl(None, "Jannik Sinner vs Carlos Alcaraz Hard", graph_runner=lambda conn, q: _fake_success_state()))
    assert "error" not in payload
    assert "0.6378" in payload["final_text"]
    assert payload["planner_source"] == "llm"
    assert payload["verification"]["ok"] is True


def test_scouting_report_impl_verification_fail_returns_error_without_final_text():
    state = _fake_success_state()
    state["verification"] = {"ok": False, "unexpected_numbers": ["0.9999"], "missing_required": [], "tag_error": None}
    payload = json.loads(_scouting_report_impl(None, "Jannik Sinner vs Carlos Alcaraz Hard", graph_runner=lambda conn, q: state))
    assert payload["error"] == "verification_failed"
    assert "final_text" not in payload


def test_scouting_report_impl_error_state_returns_error_without_final_text():
    state = _fake_success_state()
    state["error"] = "no_template_matched"
    payload = json.loads(_scouting_report_impl(None, "zzz", graph_runner=lambda conn, q: state))
    assert payload["error"] == "no_template_matched"
    assert "final_text" not in payload


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_db_elo_tool_sinner():
    import json as _json

    from mcp_server import get_player_elo_tool

    payload = _json.loads(get_player_elo_tool("Jannik Sinner"))
    assert "error" not in payload, payload
    assert payload["overall_elo"] == 2283.1
    assert payload["matches_played"] == 457


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_db_form_tool_sinner():
    import json as _json

    from mcp_server import get_player_form_tool

    payload = _json.loads(get_player_form_tool("Jannik Sinner"))
    assert "error" not in payload, payload
    assert payload["win_form"]["last10_pct"] == 0.9 or str(payload["win_form"]["last10_pct"]) == "0.9000"


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_db_h2h_tool_sinner_alcaraz():
    import json as _json

    from mcp_server import get_head_to_head_tool

    payload = _json.loads(get_head_to_head_tool("Jannik Sinner", "Carlos Alcaraz"))
    assert "error" not in payload, payload
    assert payload["total"] == 17
    assert payload["wins_a"] == 7
    assert payload["wins_b"] == 10


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_db_surface_tool_sinner():
    import json as _json

    from mcp_server import get_player_surface_record_tool

    payload = _json.loads(get_player_surface_record_tool("Jannik Sinner"))
    assert "error" not in payload, payload
    assert payload["by_surface"]["Hard"]["wins"] == 251
    assert payload["by_surface"]["Hard"]["losses"] == 57


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_db_prob_tool_sinner_alcaraz_hard():
    import json as _json

    from mcp_server import get_match_probability_tool

    payload = _json.loads(get_match_probability_tool("Jannik Sinner", "Carlos Alcaraz", "Hard"))
    assert "error" not in payload, payload
    assert payload["prob_a_calibrated"] == 0.6378
    assert payload["prob_a_blended"] == 0.6698
