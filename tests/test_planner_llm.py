"""Unit tests for Week 3 LLM planner (scripts/planner_llm.py).

Run from the project root ~/TSA:
    python -m pytest tests/test_planner_llm.py -v

No test here makes a real API call. The LLM is replaced by a fake
caller that returns a fixed string, so these tests run everywhere,
with or without an API key. The database end-to-end path is already
covered by test_planner.py; this file tests only the LLM plan step:
prompt, parsing, validation, and fallback.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from planner_llm import build_prompt, classify_question, parse_llm_response, plan_with_llm, validate_plan


def test_prompt_lists_tools_and_forbids_numbers():
    prompt = build_prompt("Sinner vs Alcaraz?")
    assert '"tool": "match"' in prompt
    assert "Hard" in prompt
    assert "must NOT invent any numbers" in prompt


def test_parse_plain_json():
    assert parse_llm_response('{"tool": "match", "player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz", "surface": "Hard"}') == {
        "tool": "match",
        "player_a": "Jannik Sinner",
        "player_b": "Carlos Alcaraz",
        "surface": "Hard",
    }


def test_parse_markdown_fence():
    text = '```json\n{"tool": "elo", "player": "Jannik Sinner"}\n```'
    assert parse_llm_response(text) == {"tool": "elo", "player": "Jannik Sinner"}


def test_parse_null_and_invalid():
    assert parse_llm_response("null") is None
    assert parse_llm_response("not json") is None
    assert parse_llm_response('{"tool": "unknown", "player": "X"}') is None
    assert parse_llm_response('{"tool": "match", "player_a": "A"}') is None
    assert parse_llm_response('{"tool": "match", "player_a": "A", "player_b": "B", "surface": "Ice"}') is None


def test_validate_plan_strips_whitespace():
    assert validate_plan({"tool": "elo", "player": "  Jannik Sinner  "}) == {
        "tool": "elo",
        "player": "Jannik Sinner",
    }


def test_plan_with_fake_llm_returns_llm_plan():
    def fake_llm(question):
        return '{"tool": "match", "player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz", "surface": "Hard"}'

    plan, source = plan_with_llm("Who wins, Sinner or Alcaraz on hard?", llm_caller=fake_llm)
    assert source == "llm"
    assert plan["tool"] == "match"
    assert plan["surface"] == "Hard"


def test_plan_with_invalid_llm_falls_back_to_rule():
    def fake_llm(question):
        return "not a plan"

    plan, source = plan_with_llm("Jannik Sinner vs Carlos Alcaraz Hard", llm_caller=fake_llm)
    assert source == "rule_fallback"
    assert plan["tool"] == "match"


def test_plan_without_key_falls_back_to_rule(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    plan, source = plan_with_llm("Jannik Sinner vs Carlos Alcaraz Hard")
    assert source == "no_key_rule"
    assert plan["tool"] == "match"


def test_classify_out_of_scope():
    kind, plan = classify_question(
        "what is the capital of France", {}, llm_caller=lambda p: "OUT_OF_SCOPE"
    )
    assert kind == "out_of_scope"
    assert plan is None


def test_classify_open():
    kind, plan = classify_question(
        "fils是否被低估",
        {"player_a": "Ben Shelton", "player_b": "Arthur Fils", "surface": "Hard"},
        llm_caller=lambda p: "OPEN",
    )
    assert kind == "open"
    assert plan is None


def test_classify_match_plan():
    kind, plan = classify_question(
        "Shelton 对 Fils 硬地",
        {},
        llm_caller=lambda p: '{"tool": "match", "player_a": "Ben Shelton", "player_b": "Arthur Fils", "surface": "Hard"}',
    )
    assert kind == "match"
    assert plan["tool"] == "match"
    assert plan["player_a"] == "Ben Shelton"
    assert plan["player_b"] == "Arthur Fils"
    assert plan["surface"] == "Hard"


def test_classify_fallback_match_template(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    kind, plan = classify_question("Jannik Sinner vs Carlos Alcaraz Hard", {})
    assert kind == "match"
    assert plan["tool"] == "match"


def test_classify_fallback_non_match_is_open(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    kind, plan = classify_question("fils是否被低估", {})
    assert kind == "open"
    assert plan is None
