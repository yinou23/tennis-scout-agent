"""Week 6 Step 1: evaluation harness and failure log, scaffold only.

Purpose: give the project a repeatable, deterministic check that
does not depend on a model writing free text, and a failure log
that a human can read and fix from, instead of a pass rate that
hides which question broke. This is the scaffold for the Week 6
milestone, eval set to 100 questions with a CI gate and a
failure log. This step does NOT deliver 100 questions. It
delivers the harness, the case format, the check rules, the
failure log format, and an initial set of 11 cases in
eval/cases.jsonl, all of whose expected values are already
verified in Weeks 2 to 4, so the harness can be proven on known
answers before it is expanded.

Why deterministic, and why no LLM in eval v1:
  A CI gate must be repeatable. If eval calls the LLM planner
  and the LLM writer, a passing run today can fail tomorrow
  because the model paraphrased a sentence, even though every
  number was right, and a failing run cannot be reproduced.
  Eval v1 therefore uses the rule planner and the templated
  report builder for match cases, and the deterministic tools
  directly for elo, form, h2h, and surface cases. LLM planner
  and LLM writer are already covered in Week 4 tests with fake
  writers and in manual demos with verification ok, and they
  are not re-tested here for flakiness. A later eval step may
  add an LLM report check, but only as a separate, non-gating
  check that verifies numbers, never as a fuzzy text score.

Case format, one JSON object per line in eval/cases.jsonl:
  {"id": "match-hard-sinner-alcaraz",
   "type": "match",
   "question": "Jannik Sinner vs Carlos Alcaraz Hard",
   "expected": {"favorite": "Jannik Sinner",
                "prob_a_calibrated": 0.6378, ...}}
  Types: match, elo, form, h2h, surface, ambiguous, unknown.
  For match, expected may include favorite, prob_a_calibrated,
  prob_b_calibrated, prob_a_blended, prob_b_blended,
  h2h_total, h2h_wins_a, h2h_wins_b, h2h_surface, and
  surface_applied. For other types, expected keys are the
  tool output fields to check, in nested form, for example
  elo surfaces Hard blended_elo, form win_form last10_pct,
  surface by_surface Hard wins. Unknown and ambiguous cases
  have expected result None or ambiguous True with candidate
  names, and they pass only when the tool returns exactly
  that, never an invented player.

Check rules:
  Counts are exact integers. Probabilities, Elo, and win
  rates are compared after rounding to the decimals the tools
  already use: probabilities to 4 decimals, Elo to 1 decimal,
  win and games rates to 4 decimals. A difference larger than
  0.00005 for a 4-decimal value, or 0.05 for a 1-decimal Elo,
  is a failure, not a rounding note. Dates are compared as
  strings, exactly. For match cases, verification ok must be
  True, the calibration tag must be present, and the surface
  tag must be present exactly when surface_applied is True,
  otherwise the case fails even if every number matches.
  A single failed check fails the whole case, and every
  failed check is written to the failure log, not just the
  first one, so one run shows all breakages.

Failure log:
  eval/failures.jsonl, overwritten on every run, one JSON
  object per failed check, with run timestamp, case id,
  question, check name, expected, and actual. An empty file
  means no failures in the last run. The log is a working
  file, not a report to publish, and it is overwritten, not
  appended, so a stale failure from an old run cannot masquerade
  as a current failure. The runner prints a summary to stdout:
  total cases, passed cases, failed cases, pass rate, and the
  failure log path. It exits 0 only when every case passes,
  otherwise it exits 1, so a CI job can gate on the exit code.

Expected values and data-as-of:
  All expected values in cases.jsonl are a snapshot of the
  database at data-as-of 2026-05-25, the same snapshot used
  in Weeks 2 to 4. If the database is re-imported with newer
  data, expected values must be re-verified and updated
  deliberately, with a note in the case file, never silently
  regenerated from the new database, because that would make
  the eval always pass and therefore useless. Expansion to
  100 cases follows the same rule: each new case's expected
  values must be verified from a tool output that a human has
  checked, not copied from an unverified source.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/eval.py
    python scripts/eval.py --only match-hard-sinner-alcaraz
    python scripts/eval.py --cases eval/cases.jsonl --failures eval/failures.jsonl
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from planner import parse_question  # noqa: E402
from scout_report import build_scout_report, verify_scout_report  # noqa: E402
from tennis_tools import (  # noqa: E402
    get_head_to_head,
    get_match_probability,
    get_player_elo,
    get_player_form,
    get_player_surface_record,
)


def _close(actual, expected, decimals):
    """Return True if actual rounds to expected at the given decimals."""
    if actual is None or expected is None:
        return actual == expected
    try:
        return round(float(actual), decimals) == round(float(expected), decimals)
    except (TypeError, ValueError):
        return False


def _check(expected, actual, prefix, failures):
    """Recursively check expected dict against actual dict, recording failures."""
    if not isinstance(expected, dict):
        if actual != expected:
            failures.append({"check": prefix, "expected": expected, "actual": actual})
        return
    if not isinstance(actual, dict):
        failures.append({"check": prefix, "expected": expected, "actual": None})
        return
    for key, exp_value in expected.items():
        act_value = actual.get(key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(exp_value, dict):
            _check(exp_value, act_value, path, failures)
        elif isinstance(exp_value, float):
            if not _close(act_value, exp_value, 4):
                failures.append({"check": path, "expected": exp_value, "actual": act_value})
        elif isinstance(exp_value, int):
            if act_value != exp_value:
                failures.append({"check": path, "expected": exp_value, "actual": act_value})
        else:
            if act_value != exp_value:
                failures.append({"check": path, "expected": exp_value, "actual": act_value})


def _check_match(conn, case, failures):
    question = case.get("question", "")
    plan = parse_question(question)
    if plan is None or plan.get("tool") != "match":
        failures.append({"check": "plan", "expected": "match plan", "actual": plan})
        return None
    report = build_scout_report(conn, plan["player_a"], plan["player_b"], plan.get("surface"), question)
    if not isinstance(report, dict) or "error" in report:
        failures.append({"check": "report", "expected": "no error", "actual": report})
        return report
    verification = verify_scout_report(report)
    if not verification.get("ok"):
        failures.append({"check": "verification.ok", "expected": True, "actual": verification})
    expected = case.get("expected", {})
    # Map friendly expected keys to report fields.
    actual_map = {
        "favorite": report.get("tools", {}).get("match", {}).get("favorite"),
        "prob_a_calibrated": report.get("prob_a_calibrated"),
        "prob_b_calibrated": report.get("prob_b_calibrated"),
        "prob_a_blended": report.get("prob_a_blended"),
        "prob_b_blended": report.get("prob_b_blended"),
        "h2h_total": report.get("tools", {}).get("h2h", {}).get("total"),
        "h2h_wins_a": report.get("tools", {}).get("h2h", {}).get("wins_a"),
        "h2h_wins_b": report.get("tools", {}).get("h2h", {}).get("wins_b"),
        "surface_applied": report.get("surface_applied"),
    }
    h2h_surface = expected.get("h2h_surface")
    if isinstance(h2h_surface, dict):
        by_surface = report.get("tools", {}).get("h2h", {}).get("by_surface", {})
        surface_name = h2h_surface.get("surface")
        entry = by_surface.get(surface_name, {}) if surface_name else {}
        for key in ("total", "wins_a", "wins_b"):
            if key in h2h_surface and entry.get(key) != h2h_surface[key]:
                failures.append({"check": f"h2h_surface.{key}", "expected": h2h_surface[key], "actual": entry.get(key)})
    for key, exp_value in expected.items():
        if key == "h2h_surface":
            continue
        act_value = actual_map.get(key)
        if isinstance(exp_value, float):
            if not _close(act_value, exp_value, 4):
                failures.append({"check": key, "expected": exp_value, "actual": act_value})
        else:
            if act_value != exp_value:
                failures.append({"check": key, "expected": exp_value, "actual": act_value})
    return report


def run_case(conn, case):
    """Run one case, return {"id": ..., "passed": bool, "failures": [...]}."""
    case_id = case.get("id", "?")
    failures = []
    case_type = case.get("type")
    expected = case.get("expected", {})
    try:
        if case_type == "match":
            _check_match(conn, case, failures)
        elif case_type == "elo":
            result = get_player_elo(conn, case["player"])
            if result is None:
                failures.append({"check": "result", "expected": "player found", "actual": None})
            else:
                _check(expected, result, "", failures)
        elif case_type == "form":
            result = get_player_form(conn, case["player"])
            if result is None:
                failures.append({"check": "result", "expected": "player found", "actual": None})
            else:
                _check(expected, result, "", failures)
        elif case_type == "h2h":
            result = get_head_to_head(conn, case["player_a"], case["player_b"])
            if result is None:
                failures.append({"check": "result", "expected": "players found", "actual": None})
            else:
                _check(expected, result, "", failures)
        elif case_type == "surface":
            result = get_player_surface_record(conn, case["player"])
            if result is None:
                failures.append({"check": "result", "expected": "player found", "actual": None})
            else:
                _check(expected, result, "", failures)
        elif case_type == "ambiguous":
            result = get_player_elo(conn, case["player"])
            if not isinstance(result, dict) or not result.get("ambiguous"):
                failures.append({"check": "ambiguous", "expected": True, "actual": result})
            else:
                names = [c.get("name") for c in result.get("candidates", [])]
                for name in expected.get("candidate_names", []):
                    if name not in names:
                        failures.append({"check": "candidate_names", "expected": name, "actual": names})
        elif case_type == "unknown":
            result = get_player_elo(conn, case["player"])
            if result is not None:
                failures.append({"check": "result", "expected": None, "actual": result})
        else:
            failures.append({"check": "type", "expected": "known type", "actual": case_type})
    except Exception as exc:  # noqa: BLE001 - a tool exception is an eval failure, not a crash
        failures.append({"check": "exception", "expected": "no exception", "actual": f"{type(exc).__name__}:{exc}"})
    return {"id": case_id, "passed": not failures, "failures": failures}


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_eval(conn, cases, failures_path=None, only=None):
    """Run cases, write failures log if a path is given, return summary dict."""
    if only:
        cases = [c for c in cases if c.get("id") == only]
    results = [run_case(conn, case) for case in cases]
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    failed = total - passed
    summary = {"total": total, "passed": passed, "failed": failed, "pass_rate": (passed / total if total else 0.0), "results": results}
    if failures_path is not None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(failures_path, "w", encoding="utf-8") as handle:
            for result in results:
                if result["passed"]:
                    continue
                case = next((c for c in cases if c.get("id") == result["id"]), {})
                for failure in result["failures"]:
                    record = {
                        "timestamp": timestamp,
                        "case_id": result["id"],
                        "question": case.get("question") or case.get("player") or "",
                        "check": failure.get("check"),
                        "expected": failure.get("expected"),
                        "actual": failure.get("actual"),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run TSA eval cases, deterministic, no LLM.")
    parser.add_argument("--cases", default="eval/cases.jsonl")
    parser.add_argument("--failures", default="eval/failures.jsonl")
    parser.add_argument("--only", default=None)
    args = parser.parse_args(argv)
    from db import get_connection

    cases = load_cases(args.cases)
    try:
        conn = get_connection()
    except Exception as exc:  # noqa: BLE001 - missing SQLite file is a clean CLI error, not a crash
        print(f"ERROR: could not open SQLite database: {exc}")
        return 1
    with conn:
        summary = run_eval(conn, cases, failures_path=args.failures, only=args.only)
    print(f"Total: {summary['total']}, Passed: {summary['passed']}, Failed: {summary['failed']}, Pass rate: {summary['pass_rate']:.4f}")
    print(f"Failures log: {args.failures}")
    for result in summary["results"]:
        if not result["passed"]:
            print(f"FAILED {result['id']}: {result['failures']}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
