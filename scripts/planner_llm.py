"""Week 3, LLM planner, plug-in replacement for the rule planner.

Interface, identical to scripts/planner.py:
- Input: a question string.
- Output: a plan dict in the same format as parse_question:
      {"tool": "match", "player_a": ..., "player_b": ..., "surface": ...}
      {"tool": "h2h", "player_a": ..., "player_b": ...}
      {"tool": "elo", "player": ...}
      {"tool": "form", "player": ...}
      {"tool": "surface", "player": ...}
- After the plan, scripts/planner.py execute_plan and build_answer
  run unchanged. The LLM never writes numbers directly; every number
  in an answer still comes from a tennis_tools function.

Provider, v1: Gemini Developer API, REST generateContent.
Endpoint, verified from Google API docs examples:
    POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    Header: x-goog-api-key: <key>
    Body: {"contents": [{"parts": [{"text": prompt}]}]}
    Response text: candidates[0].content.parts[0].text
No SDK is required; this module uses only the Python standard library
(urllib), so no new dependency is added to the venv.

API key handling, non-negotiable:
- The key is read ONLY from the environment: LLM_API_KEY first,
  then GEMINI_API_KEY. It is never hardcoded, never printed, and
  never sent in chat. The user creates it themselves in
  Google AI Studio and adds it to ~/TSA/.env as, for example:
      LLM_API_KEY=...
  The .env file is already gitignored, so the key never goes to GitHub.
- If no key is set, this module does NOT fail the demo: it prints
  a clear fallback notice and uses the rule planner instead, so the
  user can see the pipeline work before creating a key.
- If the LLM returns an invalid plan, the same fallback applies.
  An invalid plan is never executed, and never guessed at.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/planner_llm.py "Who is more likely to win, Sinner or Alcaraz on hard?"
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from planner import build_answer, detect_language, execute_plan, parse_question  # noqa: E402
from tennis_tools import SURFACES  # noqa: E402

ALLOWED_TOOLS = ("match", "h2h", "elo", "form", "surface")
# Week 9 Step 1 fix: the old default gemini-2.5-flash returns 404
# for new keys. The verified working model is
# gemini-3.5-flash-lite, same as the writer and the full graph.
DEFAULT_MODEL = "gemini-3.5-flash-lite"


def build_prompt(question):
    """Build the prompt that asks ONLY for a tool plan, never for numbers."""
    return (
        "You are a planner for a tennis scouting system. "
        "Your ONLY job is to translate the user's question into ONE tool plan. "
        "You must NOT answer the question, and you must NOT invent any numbers, "
        "Elo ratings, probabilities, or match results. Those come from tools later.\n\n"
        "Allowed tools and their arguments:\n"
        '- match: {"tool": "match", "player_a": "<full name>", "player_b": "<full name>", "surface": "Hard" or "Clay" or "Grass" or "Carpet" or null}\n'
        '- h2h: {"tool": "h2h", "player_a": "<full name>", "player_b": "<full name>"}\n'
        '- elo: {"tool": "elo", "player": "<full name>"}\n'
        '- form: {"tool": "form", "player": "<full name>"}\n'
        '- surface: {"tool": "surface", "player": "<full name>"}\n\n'
        "Rules:\n"
        "- Use full player names when the question gives them; otherwise use the name as given.\n"
        "- Surface is null when the question does not name a surface.\n"
        "- A question naming two players with vs (for example \"Jannik Sinner vs Carlos Alcaraz\") is a match plan, even when no surface is named. Surface is null in that case.\n"
        "- Use h2h ONLY when the question explicitly asks about head-to-head history (words such as h2h, head-to-head, or 交手). A plain vs matchup is never h2h.\n"
        "- If the question does not match any tool, output exactly: null\n"
        "- Output ONLY the JSON plan, or null. No explanation, no markdown, no extra text.\n\n"
        f"Question: {question}\n"
    )


def validate_plan(plan):
    """Return a cleaned plan, or None if the plan is invalid."""
    if not isinstance(plan, dict):
        return None
    tool = plan.get("tool")
    if tool not in ALLOWED_TOOLS:
        return None
    if tool in ("match", "h2h"):
        a = plan.get("player_a")
        b = plan.get("player_b")
        if not isinstance(a, str) or not a.strip() or not isinstance(b, str) or not b.strip():
            return None
        cleaned = {"tool": tool, "player_a": a.strip(), "player_b": b.strip()}
        if tool == "match":
            surface = plan.get("surface")
            if surface is not None and surface not in SURFACES:
                return None
            cleaned["surface"] = surface
        return cleaned
    player = plan.get("player")
    if not isinstance(player, str) or not player.strip():
        return None
    return {"tool": tool, "player": player.strip()}


def parse_llm_response(text):
    """Extract and validate a plan from raw LLM output text.

    Handles plain JSON, JSON inside a markdown code fence, and
    surrounding whitespace. Returns None for null, invalid JSON,
    or a plan that fails validation.
    """
    if text is None:
        return None
    text = str(text).strip()
    if text == "null":
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    # If there is extra text around the JSON object, take the first object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return validate_plan(parsed)


def call_gemini(question, api_key, model=DEFAULT_MODEL):
    """Call Gemini generateContent and return the raw response text.

    Week 9 Step 1: this now goes through scripts/llm_client.py,
    which retries a transient error once on the same model after
    2 seconds, then falls back to gemini-3.1-flash-lite. On
    failure after all retries and the fallback, it raises a
    RuntimeError with the same llm_http_error:<code> message as
    the writer, instead of the raw urllib HTTPError. Callers
    already catch Exception, so planner fallback to the rule
    planner, labelled rule_fallback, still works unchanged.
    """
    from llm_client import call_gemini_text

    text, _model_used = call_gemini_text(
        build_prompt(question), api_key, model=model
    )
    return text


def plan_with_llm(question, llm_caller=None):
    """Return (plan, source) where source is llm, rule_fallback, or no_key_rule.

    llm_caller, when given, is a function(question) -> raw text, used
    by tests to avoid a real API call. In production it is None and
    call_gemini is used with the key from the environment.
    """
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    if llm_caller is None and not api_key:
        return parse_question(question), "no_key_rule"
    try:
        if llm_caller is not None:
            raw = llm_caller(question)
        else:
            raw = call_gemini(question, api_key, model=model)
        plan = parse_llm_response(raw)
    except Exception:
        plan = None
    if plan is None:
        return parse_question(question), "rule_fallback"
    return plan, "llm"


# ---------------------------------------------------------------------------
# Router classification for /report: match / open / out_of_scope
# ---------------------------------------------------------------------------
# The /report endpoint routes a question three ways:
# - "match": a match prediction (two players, optional surface).
#   Runs the verified full graph (numbers from tools only).
# - "open": about tennis or the tennis betting market, but not a
#   match prediction (head-to-head, Elo, form, surface record,
#   value/underrated, style, history, opinion). Runs the open
#   answer path: database facts plus the LLM's own understanding.
# - "out_of_scope": not about tennis or the tennis market. The
#   API answers with a friendly message, never a raw error code.
# A match plan is only for "predict this match" questions. A
# value question such as "is Fils underrated" is "open", because
# the open path fetches the model probability and the market
# prices and lets the LLM explain the gap.


def build_classify_prompt(question, context=None):
    ctx = context if isinstance(context, dict) else {}
    form = ""
    if ctx.get("player_a") or ctx.get("player_b") or ctx.get("surface"):
        form = (
            f"\nForm context (the matchup filled in on the page): "
            f"player A {ctx.get('player_a') or '-'}, player B {ctx.get('player_b') or '-'}, "
            f"surface {ctx.get('surface') or 'any'}."
        )
    return (
        "You classify a question for a tennis scouting system. "
        "Answer with exactly one of the following, and nothing else.\n\n"
        "Scope: only questions about tennis or the tennis betting market. "
        "If the question is NOT about tennis or the tennis market, output exactly:\n"
        "OUT_OF_SCOPE\n\n"
        "If it IS about tennis or the tennis market AND it asks for a match prediction "
        "(who wins, who is the favorite, or the win probability for a named matchup), "
        "output one JSON object and nothing else:\n"
        '{"tool": "match", "player_a": "<player A>", "player_b": "<player B>", "surface": "<Hard|Clay|Grass|Carpet|null>"}\n'
        "Use the form context for a player or surface the question does not name. "
        "A bare \"A vs B\" is a match prediction.\n\n"
        "If it IS about tennis or the tennis market but does NOT ask for a match prediction "
        "(for example head-to-head, Elo, recent form, a surface record, whether a player is "
        "underrated or overrated, playing style, history, or an opinion), output exactly:\n"
        "OPEN\n\n"
        "Do not explain. Do not add any other text."
        + form
        + "\n\nQuestion: "
        + question
        + "\n"
    )


def classify_question(question, context=None, llm_caller=None):
    """Classify a question for /report routing.

    Returns (kind, plan) where kind is "match", "open", or
    "out_of_scope", and plan is the match plan dict for kind
    "match" (else None). llm_caller, when given, is a function
    (prompt) -> raw text, used by tests to avoid a real call.
    Without a key or on LLM failure, a match template from the
    rule planner is treated as "match"; anything else is "open"
    so the open answer path can still try it.
    """
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    raw = None
    if llm_caller is not None:
        try:
            raw = llm_caller(build_classify_prompt(question, context))
        except Exception:
            raw = None
    elif api_key:
        try:
            raw = call_gemini(build_classify_prompt(question, context), api_key, model=model)
        except Exception:
            raw = None
    if raw is not None:
        text = str(raw).strip()
        if text == "OUT_OF_SCOPE":
            return "out_of_scope", None
        if text == "OPEN":
            return "open", None
        plan = parse_llm_response(text)
        if plan is not None and plan.get("tool") == "match":
            return "match", plan
    # Fallback: rule planner. A match template is a match;
    # anything else goes to the open answer path.
    plan = parse_question(question)
    if plan is not None and plan.get("tool") == "match":
        return "match", plan
    return "open", None


def main(argv):
    if len(argv) < 2:
        print('Usage: python scripts/planner_llm.py "Who is more likely to win, Sinner or Alcaraz on hard?"')
        return 1
    question = argv[1]
    plan, source = plan_with_llm(question)
    if source != "llm":
        if detect_language(question) == "zh":
            print(f"[注意：本次用规则规划器兜底，原因：{source}。大模型没有参与计划，答案数字仍来自工具。]")
        else:
            print(f"[Note: falling back to the rule planner, reason: {source}. The LLM did not plan this question; answer numbers still come from tools.]")
    if plan is None:
        print(build_answer(None, {"error": "no_template_matched"}, question))
        return 1
    from db import get_connection

    try:
        conn = get_connection()
    except Exception as exc:  # noqa: BLE001 - missing SQLite file is a clean CLI error, not a crash
        print(f"ERROR: could not open SQLite database: {exc}")
        return 1
    with conn:
        outputs = execute_plan(conn, plan)
    print(build_answer(plan, outputs, question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
