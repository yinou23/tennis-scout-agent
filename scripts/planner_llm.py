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

Fix, 2026-10-28 23:32 EDT, found from the live-key run:
- The .env template ships with an empty LLM_MODEL= line. The old
  code used os.environ.get("LLM_MODEL", DEFAULT_MODEL), which
  returns "" (not the default) when the variable is set but empty.
  That built the URL .../models/:generateContent, Google returned
  404, the exception was swallowed, and every question fell back
  to the rule planner even though the key was loaded. The fix is
  os.environ.get("LLM_MODEL") or DEFAULT_MODEL, so an empty value
  falls back to gemini-2.5-flash, exactly as leaving the line blank
  was meant to do.
- The LLM failure is no longer silent: the exception type and
  message are printed to stderr (the key is in a header, never in
  the URL or the error body, so it cannot leak through this line).
  A future 401, 403, 404, or 429 will now be visible instead of
  hiding behind a generic rule_fallback notice.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/planner_llm.py "Who is more likely to win, Sinner or Alcaraz on hard?"
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from planner import build_answer, detect_language, execute_plan, parse_question  # noqa: E402
from tennis_tools import SURFACES  # noqa: E402

ALLOWED_TOOLS = ("match", "h2h", "elo", "form", "surface")
DEFAULT_MODEL = "gemini-2.5-flash"


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
    """Call Gemini generateContent and return the raw response text."""
    if not model:
        model = DEFAULT_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = json.dumps({"contents": [{"parts": [{"text": build_prompt(question)}]}]}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
    parts = body.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    return "".join(texts)


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
    except Exception as exc:
        print(f"[LLM call failed: {type(exc).__name__}: {exc}]", file=sys.stderr)
        plan = None
    if plan is None:
        return parse_question(question), "rule_fallback"
    return plan, "llm"


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
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a")
        return 1
    import psycopg

    with psycopg.connect(database_url) as conn:
        outputs = execute_plan(conn, plan)
    print(build_answer(plan, outputs, question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
