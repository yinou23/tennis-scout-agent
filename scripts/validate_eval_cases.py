"""Week 6 Step 4: validate eval cases file, no database, no LLM.

Purpose: a cheap, deterministic check that can run in CI,
where the Sackmann database is not available, and also locally
before a commit. It checks the structure of eval/cases.jsonl,
not the values against the database. Value checking is the
job of scripts/eval.py, which needs DATABASE_URL and runs
locally on the Mac as a pre-merge gate. This validator must
never claim to have checked values, only structure.

Checks, all printed, none silent:
  File exists and each line is valid JSON, one object per line.
  Each case has a non-empty string id, a type in the allowed
  set, and an expected field. For type unknown, expected may
  be an empty dict, that is the correct expected value for a
  player that does not exist. For every other type, expected
  must be a non-empty dict, otherwise the case cannot check
  anything and is a silent pass.
  Ids are unique, no duplicates, otherwise a later case would
  silently shadow an earlier one in some runners.
  The first 30 ids are exactly the canonical 30 ids, in order,
  from Step 1 (11 cases) plus Step 2 (19 cases) plus Step 3
  (70 cases, which are appended after the first 30, not
  inserted before them). If the first 30 change, the
  validator fails, because a regression baseline must not
  move without a deliberate, reviewed change.
  Total cases is at least 100. It is at least, not exactly,
  so a future expansion to more than 100 does not break
  this validator by hardcoding a total.
  For match cases, question must be present. For elo, form,
  and surface cases, player must be present. For h2h cases,
  player_a and player_b must be present. For ambiguous
  cases, player must be present and expected must contain
  candidate_names.

Usage, from the project root ~/TSA:
    python scripts/validate_eval_cases.py
    python scripts/validate_eval_cases.py --cases eval/cases.jsonl
Exit 0 only when every check passes, otherwise exit 1, with
each failure printed, so a CI job can gate on the exit code.
"""

import argparse
import json
import sys
from pathlib import Path

ALLOWED_TYPES = {"match", "elo", "form", "h2h", "surface", "ambiguous", "unknown"}

CANONICAL_30_IDS = [
    "match-hard-sinner-alcaraz",
    "match-overall-sinner-alcaraz",
    "elo-sinner",
    "elo-alcaraz",
    "form-sinner",
    "form-alcaraz",
    "h2h-sinner-alcaraz",
    "surface-sinner",
    "surface-alcaraz",
    "ambiguous-sinner",
    "unknown-player",
    "elo-novak-djokovic",
    "form-novak-djokovic",
    "surface-novak-djokovic",
    "elo-roger-federer",
    "form-roger-federer",
    "surface-roger-federer",
    "elo-rafael-nadal",
    "form-rafael-nadal",
    "surface-rafael-nadal",
    "elo-alexander-zverev",
    "form-alexander-zverev",
    "surface-alexander-zverev",
    "elo-daniil-medvedev",
    "form-daniil-medvedev",
    "surface-daniil-medvedev",
    "match-hard-novak-djokovic-roger-federer",
    "match-clay-novak-djokovic-rafael-nadal",
    "match-grass-roger-federer-rafael-nadal",
    "match-hard-alexander-zverev-daniil-medvedev",
]


def validate_cases(path):
    """Validate a cases file, return (ok, errors, summary)."""
    errors = []
    cases = []
    path_obj = Path(path)
    if not path_obj.exists():
        return False, [f"cases file not found: {path}"], {"total": 0}
    with open(path_obj, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(case, dict):
                errors.append(f"line {line_no}: case is not a JSON object")
                continue
            cases.append(case)
    ids = []
    for idx, case in enumerate(cases):
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            errors.append(f"case {idx}: missing or empty id")
            continue
        ids.append(case_id)
        case_type = case.get("type")
        if case_type not in ALLOWED_TYPES:
            errors.append(f"case {case_id}: unknown type {case_type!r}")
        if "expected" not in case:
            errors.append(f"case {case_id}: missing expected")
        elif case_type != "unknown" and not isinstance(case.get("expected"), dict):
            errors.append(f"case {case_id}: expected is not a dict")
        elif case_type != "unknown" and case.get("expected") == {}:
            errors.append(f"case {case_id}: expected is empty, case cannot check anything")
        if case_type == "match" and not case.get("question"):
            errors.append(f"case {case_id}: match case missing question")
        if case_type in ("elo", "form", "surface", "ambiguous") and not case.get("player"):
            errors.append(f"case {case_id}: {case_type} case missing player")
        if case_type == "h2h" and (not case.get("player_a") or not case.get("player_b")):
            errors.append(f"case {case_id}: h2h case missing player_a or player_b")
        if case_type == "ambiguous" and "candidate_names" not in (case.get("expected") or {}):
            errors.append(f"case {case_id}: ambiguous case expected missing candidate_names")
    if len(ids) != len(set(ids)):
        seen = set()
        duplicates = set()
        for cid in ids:
            if cid in seen:
                duplicates.add(cid)
            seen.add(cid)
        errors.append(f"duplicate ids: {sorted(duplicates)}")
    if ids[:30] != CANONICAL_30_IDS:
        errors.append("first 30 ids are not the canonical 30 ids, in order")
    if len(cases) < 100:
        errors.append(f"total cases {len(cases)} is less than 100")
    summary = {"total": len(cases), "unique_ids": len(set(ids)), "first_30_preserved": ids[:30] == CANONICAL_30_IDS}
    return not errors, errors, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate eval cases file structure, no DB, no LLM.")
    parser.add_argument("--cases", default="eval/cases.jsonl")
    args = parser.parse_args(argv)
    ok, errors, summary = validate_cases(args.cases)
    print(f"Cases file: {args.cases}")
    print(f"Total: {summary['total']}, Unique ids: {summary['unique_ids']}, First 30 preserved: {summary['first_30_preserved']}")
    if ok:
        print("Validation: OK")
        return 0
    print("Validation: FAILED")
    for error in errors:
        print(f"  - {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
