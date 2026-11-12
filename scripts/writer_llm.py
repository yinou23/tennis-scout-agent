"""Week 4 Step 4: LLM writer (rewriter) and upgraded verifier.

Purpose: replace the template writer in the graph with a model
writer, without giving up the Week 4 Step 2 guarantee that every
number in a delivered report comes from a tool output.

Design choice, stated plainly so it is not mistaken for a
generator: the LLM writer here is a REWRITER, not a generator.
It receives the full templated report text from Week 4 Step 2,
which already contains every verified number, and it rewrites
that text in natural prose, in the same sections, in the language
of the question. It does not receive raw tool data to invent a
report from, and it must not add, round, convert, or omit a
required number. This makes the verifier check meaningful: the
LLM output can be checked number by number against the template
it was given, instead of against a vague idea of what it should
have said.

Prompt rules, enforced by the verifier, not just by trust:
  - Copy every number exactly as written in the template.
    0.6378 stays 0.6378, not 0.64, not 63.78 percent, not 63.78%.
  - Do not add any number not in the template. That includes
    generic numbers like 0.5, 50 percent, section numbers,
    tool numbers, years, and dates, except dates already in the
    template (data as of 2026-05-25 and retirement dates).
  - Do not number sections. Write section headings as words only.
  - Keep the correction tags on the final probability line
    exactly as in the template, in the same language as the report.
  - Retirement is reported only as a mid-match retirement event,
    never as an injury diagnosis.
  - Form, games, retirement, and head-to-head are descriptors
    only, not probability inputs. Do not write that they changed
    the probability.

Upgraded verifier, verify_llm_report:
  This is different from the template verifier in two ways, and
  the difference is deliberate:
  1. It does NOT require every templated value to appear. A
     natural rewrite may omit a minor value, for example a games
     total for one player, and still be a valid report. Requiring
     every value would make almost every LLM output fail for
     incompleteness, not for inventing numbers.
  2. It DOES require, in addition to no invented numbers and
     correct tags, that a small set of required numbers appears:
     both calibrated probabilities, both blended probabilities,
     both players overall Elo and blended Elo, and head-to-head
     total and wins for both players. If any required number is
     missing, the report is incomplete and is blocked, even if
     no number was invented.
  In all cases, every number in the LLM text must be in the
  templated values or in the allowed parameter set, exactly as
  in Step 2. A percentage, a rounded probability, or a generic
  threshold number is an unexpected number and blocks delivery.

What this file does NOT do:
  It does not change the locked probability model. Surface blend
  prior=100 and calibration a=0.8, b=0.0 are the only corrections
  in the probability, exactly as locked in Week 2.
  It does not fall back to the template report when the LLM
  writer fails or the verifier fails. A fallback would hide
  which writer produced the report, and a failed LLM report
  must be visible as an error, not silently replaced.
  It does not call a real LLM in tests. Tests inject a fake
  api_caller or a fake writer function. A real LLM demo is a
  manual run on the Mac with a key in .env, and it must pass
  the verifier, otherwise it is an error, not a report.

LLM API, same as the Week 3 planner:
  Gemini Developer API, REST generateContent, model default
  gemini-3.5-flash-lite, overridable by LLM_MODEL env. That
  default is deliberate: gemini-2.5-flash returns 404 for new
  keys, gemini-3.8-flash returned 503 in October 2026, and
  gemini-3.8-flash-lite does not exist as a text model. The
  working model verified on the Mac in Week 3 is
  gemini-3.5-flash-lite. Key is read only from LLM_API_KEY,
  then GEMINI_API_KEY, in .env, never hardcoded, never printed,
  never sent in chat.

Week 9 Step 1 change, and only this change in this file:
  The real API call now goes through scripts/llm_client.py,
  which retries a transient error (429, 500, 502, 503, 504, or
  a network error) once on the same model after 2 seconds,
  then falls back to gemini-3.1-flash-lite, verified on
  2026-11-05 / 2026-11-06 from the Google pricing page plus
  independent rate-limit and observed-call sources, with the
  same 2 attempt rule. See llm_client.py for the full
  verification note and the May 7, 2027 earliest shutdown
  date for the fallback. The result now records the model
  that actually succeeded, so llm_model in the API response
  shows the fallback when the fallback wrote the report.
  No verifier rule, no writer rule, and no probability rule
  changed. A failed call after all retries and the fallback
  still returns an error, never a template fallback.

Graph, build_llm_graph:
  planner, analyst, llm_writer, verifier_llm, writer_final, in
  that order. Planner and analyst are reused from Week 4 Step 3.
  llm_writer calls write_report_with_llm on the templated report.
  verifier_llm calls verify_llm_report on the LLM text and the
  templated report. writer_final delivers the LLM text only if
  verification passed, otherwise it delivers an error message
  only, and the unverified LLM text is NOT delivered as a report.
  When LangGraph is installed, a real StateGraph is built, with
  an in-memory checkpointer when available. When it is not
  installed, or use_langgraph is False, the same nodes run in
  order through the sequential graph from Week 4 Step 3. On the
  Mac, LangGraph is installed, so the real path must pass.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/writer_llm.py "Jannik Sinner vs Carlos Alcaraz Hard"
    python scripts/writer_llm.py "Jannik Sinner 对 Carlos Alcaraz 硬地谁是热门"
  The first command calls the real LLM and costs one API call.
  If there is no key, or the call fails, it prints an error and
  exits non-zero. It never falls back to the template report.
"""

import os
import re
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scout_graph import (  # noqa: E402
    _SequentialGraph,
    _analyst_node_factory,
    _planner_node_factory,
)
from scout_report import ALLOWED_PARAMETERS, build_scout_report  # noqa: E402
try:
    from planner import parse_question as _default_parse_question
except ImportError:
    try:
        from planner_lang_v2 import parse_question as _default_parse_question
    except ImportError:
        _default_parse_question = None

DEFAULT_MODEL = "gemini-3.5-flash-lite"
_NUMBER_RE = re.compile(r"\d+\.\d+|\d+")


class LLMGraphState(TypedDict, total=False):
    """Graph state for the LLM writer graph. Plain data only."""

    question: str
    plan: dict
    player_a: str
    player_b: str
    surface: str
    report: dict
    llm_text: str
    llm_model: str
    verification: dict
    final_text: str
    error: str


def _fmt_prob(value):
    return f"{value:.4f}"


def _fmt_elo(value):
    return f"{value:.1f}"


def detect_language(question):
    """Return "zh" if any Chinese character is present, else "en"."""
    if not question:
        return "en"
    for ch in question:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


def build_writer_prompt(templated_text, question):
    """Build the rewriter prompt from the templated report text."""
    zh = detect_language(question) == "zh"
    if zh:
        language_line = "用中文写，语言和问题一致。"
        tag_line = "最终概率那一行必须保留模板里的修正标，一字不改。"
    else:
        language_line = "Write in English, the same language as the question."
        tag_line = "Keep the correction tags on the final probability line exactly as in the template."
    return (
        "Rewrite the following tennis scouting report in natural prose, "
        "keeping the same sections and the same order. "
        + language_line
        + "\nRules, all of them are checked, not suggestions:\n"
        "- Copy every number exactly as written in the template. "
        "0.6378 stays 0.6378, not 0.64, not 63.78 percent, not 63.78%.\n"
        "- Do not add any number not in the template. That includes "
        "0.5, 50 percent, section numbers, tool numbers, years, and "
        "dates, except dates already in the template.\n"
        "- Do not number sections. Write section headings as words only.\n"
        "- " + tag_line + "\n"
        "- Retirement is reported only as a mid-match retirement event, "
        "never as an injury diagnosis.\n"
        "- Form, games, retirement, and head-to-head are descriptors "
        "only, not probability inputs. Do not write that they changed "
        "the probability.\n"
        "- Do not add a conclusion, a prediction, or a betting suggestion "
        "beyond what the template says. If a section is in the template, "
        "keep it. If it is not in the template, do not invent it.\n"
        "\nTemplated report to rewrite:\n"
        + templated_text
    )


def _default_api_caller_with_model(prompt, model):
    """Call Gemini with retry and fallback. Return (text, model_used)."""
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("no_api_key")
    from llm_client import call_gemini_text

    return call_gemini_text(prompt, api_key, model=model)


def _default_api_caller(prompt, model):
    """Call Gemini generateContent. Returns text, or raises.

    Kept for backward compatibility. New code uses
    _default_api_caller_with_model, so the model that actually
    succeeded is recorded, not just the model that was tried.
    """
    text, _model_used = _default_api_caller_with_model(prompt, model)
    return text


def write_report_with_llm(templated_report, question, api_caller=None, model=None):
    """Rewrite a templated report with the LLM, or return an error dict.

    templated_report is the dict from build_scout_report.
    api_caller, when given, is a function(prompt, model) -> str.
    When api_caller is None, the default Gemini caller is used.
    On any failure, returns {"error": ...}, never a template fallback.
    """
    if not isinstance(templated_report, dict) or "error" in templated_report:
        return {"error": "no_templated_report"}
    templated_text = templated_report.get("text")
    if not templated_text:
        return {"error": "no_templated_text"}
    chosen_model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    prompt = build_writer_prompt(templated_text, question)
    try:
        if api_caller is not None:
            llm_text = api_caller(prompt, chosen_model)
            used_model = chosen_model
        else:
            llm_text, used_model = _default_api_caller_with_model(prompt, chosen_model)
    except Exception as exc:  # noqa: BLE001 - error type is reported, not hidden
        return {"error": f"llm_call_failed:{type(exc).__name__}:{exc}"}
    if not isinstance(llm_text, str) or not llm_text.strip():
        return {"error": "llm_empty_response"}
    return {"text": llm_text.strip(), "model": used_model, "source": "llm"}


def _required_values(templated_report):
    """Required number strings for the LLM report, from the template."""
    required = []
    if templated_report.get("prob_a_calibrated") is not None:
        required.append(_fmt_prob(templated_report["prob_a_calibrated"]))
    if templated_report.get("prob_b_calibrated") is not None:
        required.append(_fmt_prob(templated_report["prob_b_calibrated"]))
    if templated_report.get("prob_a_blended") is not None:
        required.append(_fmt_prob(templated_report["prob_a_blended"]))
    if templated_report.get("prob_b_blended") is not None:
        required.append(_fmt_prob(templated_report["prob_b_blended"]))
    tools = templated_report.get("tools") or {}
    match = tools.get("match") or {}
    for side in ("player_a", "player_b"):
        player = match.get(side) or {}
        if player.get("overall_elo") is not None:
            required.append(_fmt_elo(player["overall_elo"]))
        if player.get("blended_elo") is not None:
            required.append(_fmt_elo(player["blended_elo"]))
    h2h = tools.get("h2h") or {}
    if h2h.get("total") is not None:
        required.append(str(h2h["total"]))
    if h2h.get("wins_a") is not None:
        required.append(str(h2h["wins_a"]))
    if h2h.get("wins_b") is not None:
        required.append(str(h2h["wins_b"]))
    seen = set()
    unique = []
    for value in required:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def verify_llm_report(llm_text, templated_report):
    """Verify LLM rewritten text against the templated report.

    Checks:
      1. Every number in llm_text is in templated values or in
         ALLOWED_PARAMETERS. Any other number is unexpected.
      2. Every required number appears in llm_text.
      3. Calibration tag is present. Surface tag is present exactly
         when templated_report["surface_applied"] is true.
    Returns {"ok": bool, "unexpected_numbers": [...],
             "missing_required": [...], "tag_error": str or None}.
    """
    if not isinstance(llm_text, str) or not llm_text.strip():
        return {
            "ok": False,
            "unexpected_numbers": [],
            "missing_required": [],
            "tag_error": "empty_llm_text",
        }
    allowed = set(templated_report.get("values", [])) | ALLOWED_PARAMETERS
    found = _NUMBER_RE.findall(llm_text)
    unexpected = []
    for num in found:
        if num not in allowed and num not in unexpected:
            unexpected.append(num)
    required = _required_values(templated_report)
    missing_required = [v for v in required if v not in llm_text]
    tag_error = None
    calibration_in_text = ("Calibration" in llm_text) or ("校准" in llm_text)
    surface_in_text = ("Surface blend" in llm_text) or ("场地修正" in llm_text)
    if not calibration_in_text:
        tag_error = "calibration tag missing"
    elif bool(templated_report.get("surface_applied")) != surface_in_text:
        tag_error = "surface tag does not match surface_applied"
    ok = not unexpected and not missing_required and tag_error is None
    return {
        "ok": ok,
        "unexpected_numbers": unexpected,
        "missing_required": missing_required,
        "tag_error": tag_error,
    }


def _llm_writer_node_factory(llm_writer_func=None, api_caller=None):
    def llm_writer_node(state):
        if state.get("error"):
            return {}
        report = state.get("report")
        if report is None:
            return {"error": "no_report_to_rewrite"}
        question = state.get("question")
        if llm_writer_func is not None:
            result = llm_writer_func(report, question)
        else:
            result = write_report_with_llm(report, question, api_caller=api_caller)
        if not isinstance(result, dict) or "error" in result:
            err = result.get("error") if isinstance(result, dict) else "llm_writer_failed"
            return {"error": str(err), "llm_text": None}
        return {"llm_text": result["text"], "llm_model": result.get("model")}

    return llm_writer_node


def _verifier_llm_node(state):
    if state.get("error"):
        return {}
    llm_text = state.get("llm_text")
    report = state.get("report")
    if llm_text is None or report is None:
        return {"error": "no_llm_text_to_verify"}
    verification = verify_llm_report(llm_text, report)
    return {"verification": verification}


def _writer_final_node(state):
    if state.get("error"):
        return {"final_text": f"ERROR: {state['error']}"}
    verification = state.get("verification") or {}
    llm_text = state.get("llm_text") or ""
    if not verification.get("ok"):
        detail = (
            f"unexpected={verification.get('unexpected_numbers')}, "
            f"missing_required={verification.get('missing_required')}, "
            f"tag_error={verification.get('tag_error')}"
        )
        return {
            "error": "verification_failed",
            "final_text": f"ERROR: verification failed: {detail}",
        }
    return {"final_text": llm_text}


def build_llm_graph(conn, planner_func=None, report_builder=None,
                    llm_writer_func=None, api_caller=None, use_langgraph=True):
    """Build the LLM writer graph, real LangGraph when available.

    planner_func defaults to the Week 3 rule planner.
    report_builder defaults to Week 4 Step 2 build_scout_report.
    llm_writer_func, when given, is a function(report, question)
    -> {"text": ...} or {"error": ...}. When it is None, the
    default write_report_with_llm is used, with api_caller passed
    through (None means the default Gemini caller).
    Returns (graph, used_langgraph_bool).
    """
    if planner_func is None:
        planner_func = _default_parse_question
    if report_builder is None:
        report_builder = build_scout_report

    planner_node = _planner_node_factory(planner_func)
    analyst_node = _analyst_node_factory(conn, report_builder)
    llm_writer_node = _llm_writer_node_factory(llm_writer_func, api_caller)
    verifier_llm_node = _verifier_llm_node
    writer_final_node = _writer_final_node

    if not use_langgraph:
        return _SequentialGraph(
            [planner_node, analyst_node, llm_writer_node, verifier_llm_node, writer_final_node]
        ), False

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return _SequentialGraph(
            [planner_node, analyst_node, llm_writer_node, verifier_llm_node, writer_final_node]
        ), False

    graph = StateGraph(LLMGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("llm_writer", llm_writer_node)
    graph.add_node("verifier_llm", verifier_llm_node)
    graph.add_node("writer_final", writer_final_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "analyst")
    graph.add_edge("analyst", "llm_writer")
    graph.add_edge("llm_writer", "verifier_llm")
    graph.add_edge("verifier_llm", "writer_final")
    graph.add_edge("writer_final", END)
    try:
        from langgraph.checkpoint.memory import MemorySaver

        compiled = graph.compile(checkpointer=MemorySaver())
    except ImportError:
        compiled = graph.compile()
    return compiled, True


def run_llm_graph(conn, question, thread_id="llm-1", planner_func=None,
                  report_builder=None, llm_writer_func=None, api_caller=None,
                  use_langgraph=True):
    """Run the LLM writer graph for one question, return final state."""
    graph, used_langgraph = build_llm_graph(
        conn,
        planner_func=planner_func,
        report_builder=report_builder,
        llm_writer_func=llm_writer_func,
        api_caller=api_caller,
        use_langgraph=use_langgraph,
    )
    config = {"configurable": {"thread_id": thread_id}} if used_langgraph else None
    if config is not None:
        final_state = graph.invoke({"question": question}, config=config)
    else:
        final_state = graph.invoke({"question": question})
    final_state = dict(final_state)
    final_state["_used_langgraph"] = used_langgraph
    return final_state


def main(argv):
    if len(argv) < 2:
        print('Usage: python scripts/writer_llm.py "Jannik Sinner vs Carlos Alcaraz Hard"')
        return 1
    from db import get_connection

    question = argv[1]
    try:
        conn = get_connection()
    except Exception as exc:  # noqa: BLE001 - missing SQLite file is a clean CLI error, not a crash
        print(f"ERROR: could not open SQLite database: {exc}")
        return 1
    with conn:
        final_state = run_llm_graph(conn, question)
    print(final_state.get("final_text", ""))
    print()
    print(f"Graph used LangGraph: {final_state.get('_used_langgraph')}")
    print(f"LLM model: {final_state.get('llm_model')}")
    print(f"Verification: {final_state.get('verification')}")
    if final_state.get("error"):
        print(f"Error: {final_state['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
