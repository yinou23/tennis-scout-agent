"""Unit tests for Week 4 Step 2 templated scouting report.

Run from the project root ~/TSA:
    python -m pytest tests/test_scout_report.py -v

Pure verifier tests run everywhere, no database needed. They use
small hand-built reports, so they check the verifier logic itself,
not the database.
The database end-to-end test runs only when DATABASE_URL is set,
otherwise it is skipped. On the Mac, with .env loaded, it asks for
Sinner vs Alcaraz Hard and checks the known Week 3 numbers:
favorite Jannik Sinner, calibrated 0.6378, blended 0.6698, surface
tag present, calibration tag present, verifier ok. If that check
fails, the report builder must not be trusted yet.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scout_report import detect_language, verify_scout_report


def test_detect_language_english_default():
    assert detect_language(None) == "en"
    assert detect_language("") == "en"
    assert detect_language("Jannik Sinner vs Carlos Alcaraz Hard") == "en"


def test_detect_language_chinese():
    assert detect_language("Jannik Sinner 对 Carlos Alcaraz 硬地谁是热门") == "zh"
    assert detect_language("谁是热门") == "zh"


def _base_report():
    # A minimal report that follows the template rules: every number
    # is either in values or in ALLOWED_PARAMETERS, tags are correct.
    text = (
        "Final probability (calibrated): Jannik Sinner 0.6378, "
        "Carlos Alcaraz 0.3622 [Surface blend: Hard, prior=100] "
        "[Calibration: a=0.8]\n"
        "Data as of 2026-05-25, not live."
    )
    return {
        "text": text,
        "values": ["0.6378", "0.3622"],
        "tags": ["[Surface blend: Hard, prior=100]", "[Calibration: a=0.8]"],
        "surface_applied": True,
    }


def test_verifier_passes_for_correct_report():
    check = verify_scout_report(_base_report())
    assert check["ok"] is True
    assert check["missing_values"] == []
    assert check["unexpected_numbers"] == []
    assert check["tag_error"] is None


def test_verifier_catches_invented_number():
    report = _base_report()
    report["text"] = report["text"] + "\nInvented number 9999 here."
    check = verify_scout_report(report)
    assert check["ok"] is False
    assert "9999" in check["unexpected_numbers"]


def test_verifier_catches_missing_value():
    report = _base_report()
    report["values"] = ["0.6378", "0.3622", "0.1234"]
    check = verify_scout_report(report)
    assert check["ok"] is False
    assert "0.1234" in check["missing_values"]


def test_verifier_catches_missing_calibration_tag():
    report = _base_report()
    report["text"] = report["text"].replace("[Calibration: a=0.8]", "")
    report["tags"] = ["[Surface blend: Hard, prior=100]"]
    check = verify_scout_report(report)
    assert check["ok"] is False
    assert check["tag_error"] == "calibration tag missing"


def test_verifier_catches_surface_tag_mismatch_applied_but_absent():
    report = _base_report()
    report["text"] = report["text"].replace("[Surface blend: Hard, prior=100] ", "")
    report["tags"] = ["[Calibration: a=0.8]"]
    check = verify_scout_report(report)
    assert check["ok"] is False
    assert check["tag_error"] == "surface tag does not match surface_applied"


def test_verifier_catches_surface_tag_mismatch_not_applied_but_present():
    report = _base_report()
    report["surface_applied"] = False
    check = verify_scout_report(report)
    assert check["ok"] is False
    assert check["tag_error"] == "surface tag does not match surface_applied"


def test_verifier_passes_overall_only_report_without_surface_tag():
    report = {
        "text": (
            "Final probability (calibrated): Jannik Sinner 0.6201, "
            "Carlos Alcaraz 0.3799 [Calibration: a=0.8]\n"
            "Data as of 2026-05-25, not live."
        ),
        "values": ["0.6201", "0.3799"],
        "tags": ["[Calibration: a=0.8]"],
        "surface_applied": False,
    }
    check = verify_scout_report(report)
    assert check["ok"] is True


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_sinner_alcaraz_hard_report_end_to_end():
    import psycopg

    from scout_report import build_scout_report

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        report = build_scout_report(conn, "Jannik Sinner", "Carlos Alcaraz", "Hard")
    assert "error" not in report
    assert report["favorite"] == "Jannik Sinner"
    assert report["prob_a_calibrated"] == 0.6378
    assert report["prob_a_blended"] == 0.6698
    assert report["surface_applied"] is True
    assert any("Surface blend" in t for t in report["tags"])
    assert any("Calibration" in t for t in report["tags"])
    assert "0.6378" in report["text"]
    assert "0.6698" in report["text"]
    assert "2283.1" in report["text"]
    assert "2247.6" in report["text"]
    assert "2176.7" in report["text"]
    assert "2124.7" in report["text"]
    # Head-to-head, verified in Week 3.
    assert "17" in report["text"]
    check = verify_scout_report(report)
    assert check["ok"] is True, check


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_sinner_alcaraz_overall_only_report_has_no_surface_tag():
    import psycopg

    from scout_report import build_scout_report, verify_scout_report

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        report = build_scout_report(conn, "Jannik Sinner", "Carlos Alcaraz", None)
    assert "error" not in report
    assert report["surface_applied"] is False
    assert "Surface blend" not in report["text"]
    assert "Calibration" in report["text"]
    check = verify_scout_report(report)
    assert check["ok"] is True, check


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_sinner_alcaraz_hard_chinese_report():
    import psycopg

    from scout_report import build_scout_report, verify_scout_report

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        report = build_scout_report(
            conn,
            "Jannik Sinner",
            "Carlos Alcaraz",
            "Hard",
            "Jannik Sinner 对 Carlos Alcaraz 硬地谁是热门",
        )
    assert "error" not in report
    assert "场地修正" in report["text"]
    assert "校准" in report["text"]
    assert "0.6378" in report["text"]
    check = verify_scout_report(report)
    assert check["ok"] is True, check


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_unknown_player_returns_error_not_report():
    import psycopg

    from scout_report import build_scout_report

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        report = build_scout_report(conn, "Not A Real Player XYZ", "Carlos Alcaraz", "Hard")
    assert "error" in report
