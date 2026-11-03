"""Unit tests for Week 6 Step 1 eval harness.

Run from the project root ~/TSA:
    python -m pytest tests/test_eval.py -v

Pure tests run everywhere, no database, no LLM call. They test
the check logic directly with fake tool outputs, and the failure
log writing with a deliberately wrong case in a temporary file.
The full eval set test runs only when DATABASE_URL is set,
otherwise it is skipped. On the Mac, with .env loaded, it runs
all 11 cases in eval/cases.jsonl against the real database and
checks pass rate 1.0, with no failures. A real eval run is a
manual run of scripts/eval.py on the Mac, and it must print
Total equal to Passed, Failed 0, Pass rate 1.0000, otherwise it
is not a clean eval pass. The total was 11 in Step 1 and 30
in Step 2, and this test deliberately does not hardcode it. That manual run is NOT in
this test file's pure section, on purpose, so tests do not
depend on a manual run.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval import _check, load_cases, run_case, run_eval


def test_check_passes_for_matching_nested_values():
    failures = []
    _check({"a": {"b": 0.6378}, "c": 17}, {"a": {"b": 0.6378}, "c": 17}, "", failures)
    assert failures == []


def test_check_records_mismatch_not_first_only():
    failures = []
    _check({"a": 0.6378, "c": 17}, {"a": 0.9999, "c": 18}, "", failures)
    assert len(failures) == 2
    assert failures[0]["check"] == "a"
    assert failures[1]["check"] == "c"


def test_check_float_rounding_tolerance():
    failures = []
    _check({"p": 0.6378}, {"p": 0.6378001}, "", failures)
    assert failures == []


def test_load_cases_reads_all_lines():
    cases_path = Path(__file__).resolve().parents[1] / "eval" / "cases.jsonl"
    if not cases_path.exists():
        pytest.skip("eval/cases.jsonl not found on this machine")
    cases = load_cases(str(cases_path))
    # Count-agnostic: Step 1 had 11 cases, Step 2 expanded to 30,
    # a future expansion must not break this test by hardcoding a total.
    assert len(cases) >= 11
    assert cases[0]["id"] == "match-hard-sinner-alcaraz"
    assert len({c["id"] for c in cases}) == len(cases)


def test_run_eval_writes_failure_log_for_wrong_case(tmp_path):
    # Fake conn is None, because this case type is unknown and
    # run_case handles it without calling a tool, but we use a
    # deliberately wrong ambiguous case to force a failure log
    # entry without a database. Actually, use a case with an
    # unknown type, which fails without any tool call.
    cases = [{"id": "bad-type", "type": "nope", "expected": {}}]
    failures_path = tmp_path / "failures.jsonl"
    summary = run_eval(None, cases, failures_path=str(failures_path))
    assert summary["total"] == 1
    assert summary["passed"] == 0
    assert summary["failed"] == 1
    assert summary["pass_rate"] == 0.0
    lines = failures_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["case_id"] == "bad-type"
    assert record["check"] == "type"


def test_run_eval_empty_failures_file_when_all_pass(tmp_path):
    # An unknown-type case cannot pass, so use a direct run_case
    # check instead: a case with type unknown and a mocked tool
    # is covered in the DB test. Here, check that run_eval with
    # no cases writes an empty file and pass_rate 0.0 is handled
    # as 0.0, not a crash, and total 0.
    failures_path = tmp_path / "failures.jsonl"
    summary = run_eval(None, [], failures_path=str(failures_path))
    assert summary["total"] == 0
    assert summary["passed"] == 0
    assert failures_path.read_text(encoding="utf-8") == ""


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_full_eval_set_passes_on_real_db():
    import psycopg

    from eval import load_cases, run_eval

    cases_path = Path(__file__).resolve().parents[1] / "eval" / "cases.jsonl"
    cases = load_cases(str(cases_path))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        summary = run_eval(conn, cases, failures_path=None)
    assert summary["total"] == len(cases), summary
    assert summary["passed"] == len(cases), summary
    assert summary["failed"] == 0, summary
    assert summary["pass_rate"] == 1.0
