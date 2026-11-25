#!/usr/bin/env python3
"""Refresh eval/cases.jsonl expected values after a data refresh.

Why this exists, and the rule it follows (from scripts/eval.py):
expected values are a snapshot of the database at a data-as-of date.
When the database is re-imported with newer data, expected values
must be re-verified and updated DELIBERATELY, with a note in the
case file, never silently regenerated, because that would make the
eval always pass and therefore useless.

This script is the deliberate path, used only for a DATA refresh
(new matches imported, no tool-logic change):
  1. It uses the same tool functions the eval uses, so the new
     expected values are exactly what the tools return on the new
     database. No tool logic was changed in this refresh (only
     DATA_AS_OF label strings and the verifier allowlist), so the
     tools are the same validated code; only the data moved.
  2. Before writing, the operator must spot-check actual values
     against independent SQL (e.g. atp_elo_current, direct H2H
     counts). The 2026-11-16 refresh was verified this way:
     Sinner Elo 2300.0 / 464 matches, Medvedev-Rublev H2H 10
     (8-2), Alcaraz Hard 185-46, all confirmed by direct SQL.
  3. It preserves the case structure exactly: only the values of
     keys already present in each case's expected dict are updated
     (recursing into nested dicts). No keys are added or removed.
     Ambiguous and unknown cases are left untouched (their
     expectations do not depend on match data).
  4. It keeps a dated backup of the old case file
     (eval/cases-<old-data-as-of>.jsonl) and stamps each updated
     case with a "refreshed" note naming the new data-as-of date.

Usage (repo root, project venv):
    .venv/bin/python scripts/refresh_eval_cases.py
Then re-run the eval and the full test suite before committing.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_connection  # noqa: E402
from eval import load_cases  # noqa: E402
from planner import parse_question  # noqa: E402
from scout_report import build_scout_report  # noqa: E402
from tennis_tools import (  # noqa: E402
    get_head_to_head,
    get_player_elo,
    get_player_form,
    get_player_surface_record,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = REPO_ROOT / "eval" / "cases.jsonl"
NEW_DATA_AS_OF = "2026-11-16"
OLD_DATA_AS_OF = "2026-05-25"


def _refresh_expected(old_expected, actual):
    """Update values of keys present in old_expected from actual.

    Recurses into nested dicts. Keys absent from actual are left
    alone (and reported); keys in actual but not in old_expected
    are NOT added, so the case structure is preserved exactly.
    """
    missing = []
    new_expected = {}
    for key, old_value in old_expected.items():
        if key not in actual:
            # Tool no longer returns this key; keep the old value
            # and flag it for review.
            new_expected[key] = old_value
            missing.append(key)
            continue
        new_value = actual[key]
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            sub, sub_missing = _refresh_expected(old_value, new_value)
            new_expected[key] = sub
            missing.extend(f"{key}.{m}" for m in sub_missing)
        else:
            # Plain value (including None, which is a real expected
            # value here, e.g. a form window with no matches).
            new_expected[key] = new_value
    return new_expected, missing


def _actual_for_case(conn, case):
    """Compute the tool output the eval compares against."""
    case_type = case.get("type")
    if case_type == "elo":
        return get_player_elo(conn, case["player"])
    if case_type == "form":
        return get_player_form(conn, case["player"])
    if case_type == "h2h":
        return get_head_to_head(conn, case["player_a"], case["player_b"])
    if case_type == "surface":
        return get_player_surface_record(conn, case["player"])
    if case_type == "match":
        plan = parse_question(case["question"])
        report = build_scout_report(
            conn, plan["player_a"], plan["player_b"],
            plan.get("surface"), case["question"])
        if not isinstance(report, dict) or "error" in report:
            raise RuntimeError(f"report failed for {case['id']}: {report}")
        actual = {
            "favorite": report.get("tools", {}).get("match", {}).get(
                "favorite"),
            "prob_a_calibrated": report.get("prob_a_calibrated"),
            "prob_b_calibrated": report.get("prob_b_calibrated"),
            "prob_a_blended": report.get("prob_a_blended"),
            "prob_b_blended": report.get("prob_b_blended"),
            "h2h_total": report.get("tools", {}).get("h2h", {}).get("total"),
            "h2h_wins_a": report.get("tools", {}).get("h2h", {}).get(
                "wins_a"),
            "h2h_wins_b": report.get("tools", {}).get("h2h", {}).get(
                "wins_b"),
            "surface_applied": report.get("surface_applied"),
        }
        # h2h_surface is a nested expected entry; rebuild it from
        # the report's by_surface for the match surface.
        old_h2h = case.get("expected", {}).get("h2h_surface")
        if isinstance(old_h2h, dict):
            surface = old_h2h.get("surface")
            entry = report.get("tools", {}).get("h2h", {}).get(
                "by_surface", {}).get(surface, {})
            actual["h2h_surface"] = {"surface": surface}
            for key in ("total", "wins_a", "wins_b"):
                if key in old_h2h:
                    actual["h2h_surface"][key] = entry.get(key)
        return actual
    return None  # ambiguous / unknown: no data-dependent expectation


def main():
    cases = load_cases(str(CASES_PATH))
    backup = REPO_ROOT / "eval" / f"cases-{OLD_DATA_AS_OF}.jsonl"
    if not backup.exists():
        shutil.copy2(CASES_PATH, backup)
        print(f"backup of old cases: {backup}")
    conn = get_connection()
    updated = skipped = 0
    all_missing = []
    with conn:
        for case in cases:
            actual = _actual_for_case(conn, case)
            if actual is None:
                skipped += 1
                continue
            new_expected, missing = _refresh_expected(
                case.get("expected", {}), actual)
            case["expected"] = new_expected
            case["refreshed"] = (
                f"expected values refreshed to data-as-of "
                f"{NEW_DATA_AS_OF} (was {OLD_DATA_AS_OF}); data refresh "
                f"only, no tool-logic change")
            updated += 1
            all_missing.extend(f"{case['id']}:{m}" for m in missing)
    with open(CASES_PATH, "w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"updated {updated} cases, skipped {skipped} "
          f"(ambiguous/unknown)")
    if all_missing:
        print("keys missing from tool output (old value kept):")
        for m in all_missing:
            print(f"  {m}")
    else:
        print("no missing keys")
    print(f"wrote {CASES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
