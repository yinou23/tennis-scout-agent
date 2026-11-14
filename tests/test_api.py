"""Unit tests for Week 5 Step 1 FastAPI interface.

Run from the project root ~/TSA:
    python -m pytest tests/test_api.py -v

Pure response-builder tests run everywhere, no FastAPI, no
database, no real LLM call. They test build_report_response
directly, which is the function that decides whether a graph
result is a deliverable report (200) or an error (422, with no
final_text presented as a report).
The TestClient tests run only when fastapi and httpx are both
installed, otherwise they are skipped, so a missing TestClient
dependency does not fail the pure tests. On the Mac, with the
venv, they run the app with a fake graph runner (not a real
database or LLM call) for Sinner vs Alcaraz Hard, and check
/health, a successful /report, and an error /report.
A real /report call is a manual curl against a running server
on the Mac, with .env loaded, and it must return 200 with
verification ok, otherwise it is not a clean Week 5 Step 1 pass.
That manual curl is NOT in this test file, on purpose, so tests
do not burn LLM quota and do not flake on a 503.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from api import build_report_response

try:
    from fastapi.testclient import TestClient

    from api import create_app

    TESTCLIENT_AVAILABLE = True
except ImportError:
    TestClient = None
    create_app = None
    TESTCLIENT_AVAILABLE = False


def _success_state():
    return {
        "question": "Jannik Sinner vs Carlos Alcaraz Hard",
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


def test_response_success_returns_200_with_report():
    status, payload = build_report_response(_success_state())
    assert status == 200
    assert "0.6378" in payload["final_text"]
    assert payload["planner_source"] == "llm"
    assert payload["verification"]["ok"] is True


def test_response_error_returns_422_without_final_text():
    state = _success_state()
    state["error"] = "no_template_matched"
    state["final_text"] = "ERROR: no_template_matched"
    status, payload = build_report_response(state)
    assert status == 422
    assert payload["error"] == "no_template_matched"
    assert "final_text" not in payload


def test_response_verification_fail_returns_422_without_final_text():
    state = _success_state()
    state["verification"] = {"ok": False, "unexpected_numbers": ["0.9999"], "missing_required": [], "tag_error": None}
    state["error"] = None
    status, payload = build_report_response(state)
    assert status == 422
    assert payload["error"] == "verification_failed"
    assert "final_text" not in payload


def test_response_invalid_state_returns_422():
    status, payload = build_report_response(None)
    assert status == 422
    assert payload["error"] == "invalid_final_state"


@pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="fastapi/httpx not installed, TestClient tests skipped")
def test_health_endpoint():
    app = create_app(graph_runner=lambda q: _success_state())
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "step": "week5-step1"}


@pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="fastapi/httpx not installed, TestClient tests skipped")
def test_report_endpoint_success_with_fake_runner():
    app = create_app(graph_runner=lambda q: _success_state())
    client = TestClient(app)
    response = client.post("/report", json={"question": "Jannik Sinner vs Carlos Alcaraz Hard"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "0.6378" in payload["final_text"]
    assert payload["planner_source"] == "llm"
    assert payload["verification"]["ok"] is True


@pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="fastapi/httpx not installed, TestClient tests skipped")
def test_report_endpoint_error_with_fake_runner():
    def failing_runner(q):
        state = _success_state()
        state["error"] = "no_template_matched"
        return state

    app = create_app(graph_runner=failing_runner)
    client = TestClient(app)
    response = client.post("/report", json={"question": "zzz no template"})
    assert response.status_code == 422
    assert "final_text" not in str(response.json()) or "ERROR" not in response.json().get("detail", {}).get("final_text", "")


@pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="fastapi/httpx not installed, TestClient tests skipped")
def test_report_endpoint_empty_question_returns_422():
    app = create_app(graph_runner=lambda q: _success_state())
    client = TestClient(app)
    response = client.post("/report", json={"question": "   "})
    assert response.status_code == 422


@pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="fastapi/httpx not installed, TestClient tests skipped")
def test_report_out_of_scope_returns_friendly_message():
    app = create_app(graph_runner=lambda q: _success_state())
    client = TestClient(app)
    with patch("planner_llm.classify_question", return_value=("out_of_scope", None)):
        response = client.post("/report", json={"question": "what is the capital of France"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "out_of_scope"
    assert "outside tennis" in payload["message"]
    assert "error" not in payload


@pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="fastapi/httpx not installed, TestClient tests skipped")
def test_report_open_returns_answer():
    app = create_app(graph_runner=lambda q: _success_state())
    client = TestClient(app)
    with patch("planner_llm.classify_question", return_value=("open", None)), patch(
        "ask.answer_open_question",
        return_value={"text": "Fils is underrated.", "facts": {"players": []}},
    ):
        response = client.post(
            "/report",
            json={"question": "fils是否被低估", "player_a": "Ben Shelton", "player_b": "Arthur Fils", "surface": "Hard"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "answer"
    assert payload["text"] == "Fils is underrated."
