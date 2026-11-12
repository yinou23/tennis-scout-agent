"""Unit tests for Week 7 Step 1 weekly sample report scaffold.

Run from the project root ~/TSA:
    python -m pytest tests/test_weekly_sample.py -v

Pure tests run everywhere, no database, no LLM call. They use
a fake templated report dict and a mocked verifier, because
the real verifier checks the full templated report text, and
these tests are about the sample wrapper, metadata, sources,
tool trace, cost, latency, and coverage, not about the real
verifier, which is already tested in test_scout_report.py.
The database test runs only when the SQLite database file exists,
otherwise it is skipped. On the Mac, with .env loaded, it
generates a real sample for Jannik Sinner vs Carlos Alcaraz
Hard and checks the calibrated and blended probabilities,
the favorite, the data-as-of, and that the sample is
templated with 0 LLM calls. A real manual run is
python scripts/weekly_sample.py on the Mac, and it must
print the full sample and write samples/weekly_sample.md,
otherwise it is not a clean Week 7 Step 1 pass.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from db import get_connection, get_sqlite_path

import weekly_sample


def _fake_report():
    return {
        "text": "Scouting report: Jannik Sinner vs Carlos Alcaraz (Hard)\nFinal probability (calibrated): Jannik Sinner 0.6378, Carlos Alcaraz 0.3622 [Surface blend: Hard, prior=100] [Calibration: a=0.8]",
        "favorite": "Jannik Sinner",
        "prob_a_calibrated": 0.6378,
        "prob_b_calibrated": 0.3622,
        "prob_a_blended": 0.6698,
        "prob_b_blended": 0.3302,
    }


def _mock_verifier_ok(report):
    return {"ok": True, "unexpected_numbers": [], "missing_required": [], "tag_error": None}


def test_build_sample_markdown_includes_metadata_sources_trace_cost_coverage(monkeypatch):
    monkeypatch.setattr(weekly_sample, "verify_scout_report", _mock_verifier_ok)
    markdown = weekly_sample.build_sample_markdown(_fake_report(), "Jannik Sinner", "Carlos Alcaraz", "Hard", 12.3)
    assert "Weekly sample scouting report" in markdown
    assert "Jannik Sinner vs Carlos Alcaraz" in markdown
    assert "Data as of: 2026-05-25, not live" in markdown
    assert "LLM calls: 0" in markdown
    assert "LLM cost: $0.00" in markdown
    assert "12.3 ms" in markdown
    assert "Sources" in markdown
    assert "Tool trace" in markdown
    assert "get_match_probability" in markdown
    assert "get_head_to_head" in markdown
    assert "Coverage note" in markdown
    assert "Historical only" in markdown
    assert _fake_report()["text"] in markdown


def test_build_sample_markdown_does_not_claim_live(monkeypatch):
    monkeypatch.setattr(weekly_sample, "verify_scout_report", _mock_verifier_ok)
    markdown = weekly_sample.build_sample_markdown(_fake_report(), "Jannik Sinner", "Carlos Alcaraz", "Hard", 1.0)
    assert "not live" in markdown
    assert "does not include live scores" in markdown


def test_build_sample_markdown_fails_when_verification_not_ok(monkeypatch):
    def _verifier_fail(report):
        return {"ok": False, "unexpected_numbers": ["0.9999"], "missing_required": [], "tag_error": None}

    monkeypatch.setattr(weekly_sample, "verify_scout_report", _verifier_fail)
    with pytest.raises(ValueError):
        weekly_sample.build_sample_markdown(_fake_report(), "Jannik Sinner", "Carlos Alcaraz", "Hard", 1.0)


@pytest.mark.skipif(not get_sqlite_path().exists(), reason="SQLite database file not found; set SQLITE_PATH or build it with scripts/pg_to_sqlite.py")
def test_generate_sample_sinner_alcaraz_hard_on_real_db():

    with get_connection() as conn:
        markdown, report, latency_ms = weekly_sample.generate_sample(conn, "Jannik Sinner", "Carlos Alcaraz", "Hard")
    assert "Jannik Sinner vs Carlos Alcaraz" in markdown
    assert "0.6378" in markdown
    assert "0.6698" in markdown
    assert "Jannik Sinner" in markdown
    assert "Data as of: 2026-05-25, not live" in markdown
    assert "LLM calls: 0" in markdown
    assert latency_ms > 0
    assert report["favorite"] == "Jannik Sinner"
