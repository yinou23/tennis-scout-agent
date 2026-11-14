"""Open-ended tennis answers: LLM plus database facts.

For a tennis question that does not fit a fixed tool, retrieve
the relevant database facts (Elo, form, surface record, match
probability, market odds) and let the LLM answer with its own
understanding, grounded in those facts. The preamble below is
the rule the user set: the LLM answers any tennis question,
using the database and the probability model for numbers, plus
its own understanding for interpretation. Numbers from the
Database facts block are verified; anything else is the LLM's
own knowledge or opinion, and the answer must say so when it
is unsure. Non-tennis questions never reach this module; the
planner returns OUT_OF_SCOPE for those first.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from planner import detect_language  # noqa: E402
from retrieve import retrieve_facts  # noqa: E402

DEFAULT_MODEL = "gemini-3.5-flash-lite"


def build_ask_prompt(question, context_text):
    """Build the prompt: preamble, database facts, question."""
    zh = detect_language(question) == "zh"
    lang_line = (
        "Answer in Chinese." if zh else "Answer in English."
    )
    return (
        "You are a tennis scouting assistant. Answer the user's tennis question.\n\n"
        "Rules:\n"
        "- Use ONLY the numbers in the Database facts below (Elo, win rates, probabilities, market prices, "
        "diffs, and verdicts). Do not use any other number, percentage, or record, and do not invent one. "
        "If a number the question needs is not in the facts, say it is not available instead of guessing.\n"
        "- The facts already state each player's market-vs-model verdict (underrated, overrated, or fairly "
        "priced). Report that verdict as given; do not recompute or reverse it.\n"
        "- Use your own understanding for interpretation, playing style, history, and opinion. "
        "When something comes from your own knowledge rather than the Database facts, say so plainly "
        "(for example, \"from my own knowledge\" / \"就我所知\").\n"
        "- If the Database facts do not cover the question, say what is missing instead of guessing a number.\n"
        "- If you are unsure about a fact from your own knowledge, say you are unsure.\n"
        "- Write in plain text only. Do not use markdown formatting: no **bold**, no # headings, no bullet "
        "lists with - or *.\n"
        "- Keep the answer focused and concise. " + lang_line + "\n\n"
        "Database facts (verified, from the match database, the Elo model, and the betting markets; "
        "data as of 2026-05-25, market prices are live):\n"
        + context_text
        + "\n\nQuestion: "
        + question
        + "\n"
    )


def answer_open_question(conn, question, context=None, api_key=None, model=None):
    """Answer an open-ended tennis question. Returns a dict.

    Returns {"text": ..., "facts": {...}} on success, or
    {"error": ...} when the LLM call fails or no key is set.
    The facts are the retrieved database numbers, so the caller
    can show what came from the database.
    """
    if api_key is None:
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "no_key"}
    facts = retrieve_facts(conn, question, context)
    prompt = build_ask_prompt(question, facts["context_text"])
    try:
        from llm_client import call_gemini_text

        text, _model_used = call_gemini_text(
            prompt, api_key, model=model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        )
    except Exception as exc:  # noqa: BLE001 - LLM failure is a clean error, not a crash
        return {"error": f"llm_failed:{type(exc).__name__}"}
    text = (text or "").strip()
    if not text:
        return {"error": "llm_empty"}
    return {"text": text, "facts": facts}
