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
    # llm=true is explicit here: the default is llm=false (fast
    # template path), which never calls the graph runner.
    response = client.post(
        "/report", json={"question": "Jannik Sinner vs Carlos Alcaraz Hard", "llm": True}
    )
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
    # llm=true is explicit here, same as the success test above.
    response = client.post("/report", json={"question": "zzz no template", "llm": True})
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
    ), patch("db.get_connection"):
        response = client.post(
            "/report",
            json={"question": "fils是否被低估", "player_a": "Ben Shelton", "player_b": "Arthur Fils", "surface": "Hard"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "answer"
    assert payload["text"] == "Fils is underrated."


# --- Fast /report path (llm=false) and POST /report/text ---------------
#
# The fast path returns the deterministic template report with no LLM
# call; /report/text rewrites that same template with the LLM on
# demand. Both are tested with a fake template report whose text
# contains exactly the required numbers, so the real verifier runs
# and must pass. The graph runner must never run on the fast path.

_FAKE_TEMPLATE_TEXT = (
    "Jannik Sinner is the favorite on Hard. "
    "Final probability (calibrated): Jannik Sinner 0.6378, Carlos Alcaraz 0.3622 "
    "[Surface blend: Hard, prior=100] [Calibration: a=0.8] "
    "Blended probability was Jannik Sinner 0.6698, Carlos Alcaraz 0.3302. "
    "Overall Elo: Jannik Sinner 2500.0, Carlos Alcaraz 2400.0. "
    "Blended Elo: Jannik Sinner 2510.0, Carlos Alcaraz 2390.0. "
    "Head to head: 3 matches, Jannik Sinner 2, Carlos Alcaraz 1."
)

_FAKE_LLM_TEXT = (
    "In natural prose, Jannik Sinner is the favorite on Hard. "
    "Final probability (calibrated): Jannik Sinner 0.6378, Carlos Alcaraz 0.3622 "
    "[Surface blend: Hard, prior=100] [Calibration: a=0.8] "
    "Blended probability was Jannik Sinner 0.6698, Carlos Alcaraz 0.3302. "
    "Overall Elo: Jannik Sinner 2500.0, Carlos Alcaraz 2400.0. "
    "Blended Elo: Jannik Sinner 2510.0, Carlos Alcaraz 2390.0. "
    "Head to head: 3 matches, Jannik Sinner 2, Carlos Alcaraz 1."
)


def _fake_template_report():
    return {
        "text": _FAKE_TEMPLATE_TEXT,
        "values": [
            "0.6378", "0.3622", "0.6698", "0.3302",
            "2500.0", "2400.0", "2510.0", "2390.0",
            "3", "2", "1", "100", "0.8",
        ],
        "surface_applied": True,
        "prob_a_calibrated": 0.6378,
        "prob_b_calibrated": 0.3622,
        "prob_a_blended": 0.6698,
        "prob_b_blended": 0.3302,
        "tools": {
            "match": {
                "player_a": {"name": "Jannik Sinner", "overall_elo": 2500.0, "blended_elo": 2510.0},
                "player_b": {"name": "Carlos Alcaraz", "overall_elo": 2400.0, "blended_elo": 2390.0},
            },
            "h2h": {"total": 3, "wins_a": 2, "wins_b": 1},
        },
    }


_MATCH_PLAN = {"tool": "match", "player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz", "surface": "Hard"}


def _never_runner(question):
    raise AssertionError("graph runner must not run on the fast path")


@pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="fastapi/httpx not installed, TestClient tests skipped")
def test_report_fast_path_returns_template_without_llm():
    app = create_app(graph_runner=_never_runner)
    client = TestClient(app)
    with patch("planner_llm.classify_question", return_value=("match", _MATCH_PLAN)), patch(
        "db.get_connection"
    ), patch("scout_report.build_scout_report", return_value=_fake_template_report()):
        response = client.post(
            "/report",
            json={"question": "Jannik Sinner vs Carlos Alcaraz Hard", "llm": False},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["kind"] == "match"
    assert payload["final_text"] == _FAKE_TEMPLATE_TEXT
    assert payload["text_source"] == "template"
    assert payload["verification"]["ok"] is True
    assert payload["llm_model"] is None


@pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="fastapi/httpx not installed, TestClient tests skipped")
def test_report_fast_path_with_players_skips_classifier_llm():
    # Rule-first routing: a plain frontend-format question with
    # the players in the body is a match with no LLM call, so
    # the classifier must never run. classify_question calls
    # Gemini first whenever a key is set, which cost 10-25s on
    # Cloud Run and made live Predict measure 24s twice while
    # the same call took 0.55s locally (no key, router only).
    def _never_classify(question, context=None):
        raise AssertionError("classifier must not run for a frontend-format match")

    app = create_app(graph_runner=_never_runner)
    client = TestClient(app)
    with patch("planner_llm.classify_question", side_effect=_never_classify), patch(
        "db.get_connection"
    ), patch("scout_report.build_scout_report", return_value=_fake_template_report()):
        response = client.post(
            "/report",
            json={
                "question": "Jannik Sinner vs Carlos Alcaraz Hard",
                "player_a": "Jannik Sinner",
                "player_b": "Carlos Alcaraz",
                "surface": "Hard",
                "llm": False,
            },
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["kind"] == "match"
    assert payload["final_text"] == _FAKE_TEMPLATE_TEXT
    assert payload["text_source"] == "template"
    assert payload["verification"]["ok"] is True


@pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="fastapi/httpx not installed, TestClient tests skipped")
def test_report_default_is_fast_path_without_llm_flag():
    # The LLM is off by default: a /report call that omits llm
    # must take the fast template path, never the graph runner
    # and never the classifier. Only an explicit llm=true, the
    # /report/text button, or a manually typed question may
    # spend an LLM call.
    def _never_classify(question, context=None):
        raise AssertionError("classifier must not run for a frontend-format match")

    app = create_app(graph_runner=_never_runner)
    client = TestClient(app)
    with patch("planner_llm.classify_question", side_effect=_never_classify), patch(
        "db.get_connection"
    ), patch("scout_report.build_scout_report", return_value=_fake_template_report()):
        response = client.post(
            "/report",
            json={
                "question": "Jannik Sinner vs Carlos Alcaraz Hard",
                "player_a": "Jannik Sinner",
                "player_b": "Carlos Alcaraz",
                "surface": "Hard",
            },
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["kind"] == "match"
    assert payload["final_text"] == _FAKE_TEMPLATE_TEXT
    assert payload["text_source"] == "template"
    assert payload["verification"]["ok"] is True


@pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="fastapi/httpx not installed, TestClient tests skipped")
def test_report_text_returns_llm_prose():
    app = create_app(graph_runner=_never_runner)
    client = TestClient(app)
    with patch("planner_llm.classify_question", return_value=("match", _MATCH_PLAN)), patch(
        "db.get_connection"
    ), patch("scout_report.build_scout_report", return_value=_fake_template_report()), patch(
        "writer_llm.write_report_with_llm",
        return_value={"text": _FAKE_LLM_TEXT, "model": "fake"},
    ):
        response = client.post(
            "/report/text",
            json={"question": "Jannik Sinner vs Carlos Alcaraz Hard"},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["kind"] == "match_text"
    assert payload["final_text"] == _FAKE_LLM_TEXT
    assert payload["text_source"] == "llm"
    assert payload["verification"]["ok"] is True
    assert payload["llm_model"] == "fake"


@pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="fastapi/httpx not installed, TestClient tests skipped")
def test_report_text_verification_failure_returns_422():
    app = create_app(graph_runner=_never_runner)
    client = TestClient(app)
    broken_text = _FAKE_LLM_TEXT.replace("0.6378, ", "")
    with patch("planner_llm.classify_question", return_value=("match", _MATCH_PLAN)), patch(
        "db.get_connection"
    ), patch("scout_report.build_scout_report", return_value=_fake_template_report()), patch(
        "writer_llm.write_report_with_llm",
        return_value={"text": broken_text, "model": "fake"},
    ):
        response = client.post(
            "/report/text",
            json={"question": "Jannik Sinner vs Carlos Alcaraz Hard"},
        )
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "verification_failed"
    assert "0.6378" in detail["verification"]["missing_required"]
    assert "final_text" not in detail


@pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="fastapi/httpx not installed, TestClient tests skipped")
def test_report_text_not_a_match_returns_422():
    app = create_app(graph_runner=_never_runner)
    client = TestClient(app)
    with patch("planner_llm.classify_question", return_value=("open", None)):
        response = client.post(
            "/report/text",
            json={"question": "fils是否被低估"},
        )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["error"] == "not_a_match"
