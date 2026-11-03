"""Unit tests for Week 6 Step 4 eval cases validator.

Run from the project root ~/TSA:
    python -m pytest tests/test_validate_eval_cases.py -v

Pure tests run everywhere, no database, no LLM call. They use
temporary files, not the real eval/cases.jsonl, except for the
last test, which runs on the real file only if it exists,
otherwise it is skipped. A valid temp file in these tests has
100 cases: the canonical first 30 ids, plus 70 dummy ids,
each with a type and a non-empty expected dict, so the
validator's total check, first-30 check, and uniqueness check
are all exercised without a database.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_eval_cases import CANONICAL_30_IDS, validate_cases


def _write_cases(tmp_path, cases):
    path = tmp_path / "cases.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for case in cases:
            if isinstance(case, str):
                handle.write(case + "\n")
            else:
                handle.write(json.dumps(case) + "\n")
    return str(path)


def _valid_100_cases():
    cases = []
    for cid in CANONICAL_30_IDS:
        if cid.startswith("match-"):
            cases.append({"id": cid, "type": "match", "question": "A vs B Hard", "expected": {"favorite": "A"}})
        elif cid == "ambiguous-sinner":
            cases.append({"id": cid, "type": "ambiguous", "player": "Sinner", "expected": {"candidate_names": ["Jannik Sinner"]}})
        elif cid == "unknown-player":
            cases.append({"id": cid, "type": "unknown", "player": "ZZZ", "expected": {}})
        elif cid.startswith("h2h-"):
            cases.append({"id": cid, "type": "h2h", "player_a": "A", "player_b": "B", "expected": {"total": 1}})
        elif cid.startswith("elo-"):
            cases.append({"id": cid, "type": "elo", "player": "A", "expected": {"overall_elo": 2000.0}})
        elif cid.startswith("form-"):
            cases.append({"id": cid, "type": "form", "player": "A", "expected": {"win_form": {}}})
        elif cid.startswith("surface-"):
            cases.append({"id": cid, "type": "surface", "player": "A", "expected": {"total_matches": 1}})
        else:
            cases.append({"id": cid, "type": "elo", "player": "A", "expected": {"overall_elo": 2000.0}})
    for i in range(70):
        cases.append({"id": f"dummy-{i:03d}", "type": "elo", "player": "A", "expected": {"overall_elo": 2000.0}})
    return cases


def test_validator_passes_for_valid_100_file(tmp_path):
    path = _write_cases(tmp_path, _valid_100_cases())
    ok, errors, summary = validate_cases(path)
    assert ok, errors
    assert summary["total"] == 100
    assert summary["first_30_preserved"] is True


def test_validator_fails_for_duplicate_id(tmp_path):
    cases = _valid_100_cases()
    cases[31] = dict(cases[31])
    cases[31]["id"] = cases[30]["id"]
    path = _write_cases(tmp_path, cases)
    ok, errors, _summary = validate_cases(path)
    assert not ok
    assert any("duplicate" in e for e in errors)


def test_validator_fails_for_invalid_json(tmp_path):
    path = _write_cases(tmp_path, ["{not json"])
    ok, errors, _summary = validate_cases(path)
    assert not ok
    assert any("invalid JSON" in e for e in errors)


def test_validator_fails_for_changed_first_30(tmp_path):
    cases = _valid_100_cases()
    cases[0] = dict(cases[0])
    cases[0]["id"] = "changed-first-id"
    path = _write_cases(tmp_path, cases)
    ok, errors, _summary = validate_cases(path)
    assert not ok
    assert any("first 30" in e for e in errors)


def test_validator_fails_for_total_under_100(tmp_path):
    cases = _valid_100_cases()[:30]
    path = _write_cases(tmp_path, cases)
    ok, errors, _summary = validate_cases(path)
    assert not ok
    assert any("less than 100" in e for e in errors)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "eval" / "cases.jsonl").exists(),
    reason="eval/cases.jsonl not found on this machine",
)
def test_validator_passes_on_real_cases_file():
    path = Path(__file__).resolve().parents[1] / "eval" / "cases.jsonl"
    ok, errors, summary = validate_cases(str(path))
    assert ok, errors
    assert summary["total"] >= 100
    assert summary["first_30_preserved"] is True
