"""Unit tests for Week 4 Step 4 LLM writer and upgraded verifier.

Run from the project root ~/TSA:
    python -m pytest tests/test_writer_llm.py -v

Pure tests run everywhere, no database, no LangGraph, and no real
LLM API call. They inject fake api_caller and fake writer
functions. A fake writer that returns a correct rewrite proves
the graph plumbing and the verifier, not LLM intelligence.
The database graph tests run only when the SQLite database file exists,
otherwise they are skipped. On the Mac, with .env loaded, they
run the LLM graph with a fake writer (not a real API call) for
Sinner vs Alcaraz Hard, and check the known Step 2 numbers and
verifier ok. A real LLM demo is a manual run of
scripts/writer_llm.py with a key in .env, and it must pass the
verifier, otherwise it is an error, not a report. That manual
demo is NOT in this test file, on purpose, so tests do not burn
quota and do not flake on a 503.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from db import get_connection, get_sqlite_path

from writer_llm import (
    build_writer_prompt,
    run_llm_graph,
    verify_llm_report,
    write_report_with_llm,
)


def _fake_templated_report():
    # Mimics the real Step 2 report dict for Sinner vs Alcaraz Hard,
    # with the fields verify_llm_report actually reads.
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


def test_verify_passes_for_correct_rewrite():
    check = verify_llm_report(_correct_llm_text(), _fake_templated_report())
    assert check["ok"] is True, check


def test_verify_catches_invented_number():
    text = _correct_llm_text() + " The model also says 0.9999."
    check = verify_llm_report(text, _fake_templated_report())
    assert check["ok"] is False
    assert "0.9999" in check["unexpected_numbers"]


def test_verify_catches_percent_conversion():
    text = _correct_llm_text().replace("0.6378", "63.78%")
    check = verify_llm_report(text, _fake_templated_report())
    assert check["ok"] is False
    assert "63.78" in check["unexpected_numbers"]
    assert "0.6378" in check["missing_required"]


def test_verify_catches_rounded_probability():
    text = _correct_llm_text().replace("0.6378", "0.64")
    check = verify_llm_report(text, _fake_templated_report())
    assert check["ok"] is False
    assert "0.64" in check["unexpected_numbers"]
    assert "0.6378" in check["missing_required"]


def test_verify_catches_missing_blended():
    text = _correct_llm_text().replace(
        "Blended probability was Jannik Sinner 0.6698, Carlos Alcaraz 0.3302. ", ""
    )
    check = verify_llm_report(text, _fake_templated_report())
    assert check["ok"] is False
    assert "0.6698" in check["missing_required"]


def test_verify_catches_missing_surface_tag():
    text = _correct_llm_text().replace("[Surface blend: Hard, prior=100] ", "")
    check = verify_llm_report(text, _fake_templated_report())
    assert check["ok"] is False
    assert check["tag_error"] == "surface tag does not match surface_applied"


def test_verify_allows_omitting_minor_value():
    # A non-required value may be omitted. Here the templated
    # report gains an extra career-wins value 363, the LLM text
    # does not mention 363, and the verifier must still pass,
    # because it only requires the required numbers, not every
    # templated value. This is the deliberate difference from
    # the template verifier in Step 2.
    template = _fake_templated_report()
    template["values"] = template["values"] + ["363"]
    check = verify_llm_report(_correct_llm_text(), template)
    assert check["ok"] is True, check
    assert "363" not in check["missing_required"]


def test_writer_prompt_contains_template_and_rules():
    prompt = build_writer_prompt(_fake_templated_report()["text"], "Jannik Sinner vs Carlos Alcaraz Hard")
    assert "0.6378" in prompt
    assert "Do not add any number" in prompt
    assert "63.78" not in prompt or "not 63.78" in prompt


def test_write_with_fake_api_caller_returns_text_not_error():
    def fake_caller(prompt, model):
        assert "0.6378" in prompt
        assert model == "gemini-3.5-flash-lite"
        return _correct_llm_text()

    result = write_report_with_llm(_fake_templated_report(), "Jannik Sinner vs Carlos Alcaraz Hard", api_caller=fake_caller)
    assert "error" not in result
    assert result["text"] == _correct_llm_text()
    assert result["model"] == "gemini-3.5-flash-lite"


def test_write_with_failing_api_caller_returns_error_not_template():
    def failing_caller(prompt, model):
        raise RuntimeError("llm_http_error:503")

    result = write_report_with_llm(_fake_templated_report(), "Jannik Sinner vs Carlos Alcaraz Hard", api_caller=failing_caller)
    assert "error" in result
    assert "0.6378" not in result.get("text", "")


def _fake_planner_match(question):
    return {"tool": "match", "player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz", "surface": "Hard"}


def _fake_report_builder(conn, player_a, player_b, surface, question):
    return _fake_templated_report()


def _fake_llm_writer(report, question):
    return {"text": _correct_llm_text(), "model": "fake", "source": "llm"}


def _fake_llm_writer_bad(report, question):
    return {"text": _correct_llm_text() + " Invented 0.9999.", "model": "fake", "source": "llm"}


def test_llm_graph_happy_path_sequential():
    final = run_llm_graph(
        None,
        "Jannik Sinner vs Carlos Alcaraz Hard",
        planner_func=_fake_planner_match,
        report_builder=_fake_report_builder,
        llm_writer_func=_fake_llm_writer,
        use_langgraph=False,
    )
    assert final.get("error") is None, final
    assert final["verification"]["ok"] is True
    assert "0.6378" in final["final_text"]
    # Final text must be the LLM text, not the templated text alone.
    # The LLM text starts with a natural sentence, the template starts with Final probability.
    assert final["final_text"].startswith("Jannik Sinner is the favorite")


def test_llm_graph_verifier_fail_blocks_llm_text():
    final = run_llm_graph(
        None,
        "Jannik Sinner vs Carlos Alcaraz Hard",
        planner_func=_fake_planner_match,
        report_builder=_fake_report_builder,
        llm_writer_func=_fake_llm_writer_bad,
        use_langgraph=False,
    )
    assert final.get("error") == "verification_failed"
    assert "0.9999" not in final.get("final_text", "") or "unexpected" in final.get("final_text", "")
    assert "verification failed" in final.get("final_text", "")


@pytest.mark.skipif(not get_sqlite_path().exists(), reason="SQLite database file not found; set SQLITE_PATH or build it with scripts/pg_to_sqlite.py")
def test_llm_graph_hard_end_to_end_with_fake_writer():

    from writer_llm import run_llm_graph

    with get_connection() as conn:
        final = run_llm_graph(
            conn,
            "Jannik Sinner vs Carlos Alcaraz Hard",
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
    assert final["verification"]["ok"] is True, final["verification"]
    assert "0.6378" in final["final_text"]
    assert final["_used_langgraph"] is True, "expected real LangGraph on Mac, got fallback"
