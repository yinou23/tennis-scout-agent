"""Week 4 Step 5: LLM planner wired into the full graph.

Purpose: make the graph end to end LLM-driven at the two places
where language matters, planner and writer, while keeping every
number deterministic. Planner and writer are the only LLM nodes.
Analyst (tools), verifier (number checks), and the probability
model are unchanged from Week 4 Steps 2 to 4.

What changes in this step, and only this:
  The planner node in the full graph uses the Week 3 LLM planner
  first, instead of the rule planner. The rule planner is still
  there, but only as a labelled fallback, never as a silent
  replacement.

Why the fallback is labelled, and why it is different from the
writer fallback in Step 4:
  Writer fallback to the template report is forbidden, because
  it would hide which writer produced the report. Planner
  fallback to the rule planner is allowed, because the rule
  planner invents no numbers, it only parses the question
  literally, and because the fallback is recorded in state as
  planner_source, and printed by main, so the user can always
  see whether the plan came from the LLM or from the rule
  planner. A fallback plan that is not labelled would be a lie
  about provenance, so labelling is the whole point.

Planner sources, same three values as Week 3 planner_llm:
  "llm"          the LLM returned a valid plan, and it is used.
  "rule_fallback" the LLM call failed, or its plan was invalid
                 or null, so the rule planner plan is used.
  "no_key_rule"  no API key was set, so the rule planner plan
                 is used, and no LLM call was attempted.
  If the rule planner also returns None, the graph errors with
  no_template_matched, and planner_source is still recorded,
  so the failure is visible, not hidden.

Model correction, stated plainly:
  This file does NOT use planner_llm.DEFAULT_MODEL, which is
  gemini-2.5-flash and returns 404 for new keys, and it does
  NOT use os.environ.get("LLM_MODEL", DEFAULT) in a way that
  keeps an empty string when LLM_MODEL is set but empty.
  It uses os.environ.get("LLM_MODEL") or "gemini-3.5-flash-lite",
  the model verified working on the Mac in Week 3 and used by
  the Step 4 writer. It calls planner_llm.call_gemini with that
  model explicitly, and planner_llm.parse_llm_response to parse
  and validate, so the parsing and validation rules are the
  same as Week 3, only the model choice is corrected here.

Graph, build_full_graph:
  planner_llm, analyst, llm_writer, verifier_llm, writer_final,
  in that order. Planner and analyst are reused in spirit from
  Steps 3 and 4, but the planner node is new in this file,
  because the Step 3 planner node does not store planner_source,
  and changing it in place would rewrite a file already verified
  on the Mac. Analyst, llm_writer, verifier_llm, and writer_final
  are imported from scout_graph and writer_llm, not copied.
  When LangGraph is installed, a real StateGraph is built, with
  an in-memory checkpointer when available. When it is not
  installed, or use_langgraph is False, the same nodes run in
  order through the sequential graph from Step 3. On the Mac,
  LangGraph is installed, so the real path must pass.

What this file does NOT do:
  It does not change the locked probability model. Surface blend
  prior=100 and calibration a=0.8, b=0.0 are the only corrections.
  It does not change any tool, any verifier rule, or any writer
  rule. It does not call a real LLM in tests. Tests inject a fake
  llm_caller or a fake planner_with_source function. A real LLM
  planner demo is a manual run on the Mac with a key in .env,
  and it must show planner_source "llm" and verification ok,
  otherwise the demo is not a clean LLM planner pass.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/full_graph.py "Jannik Sinner vs Carlos Alcaraz Hard"
    python scripts/full_graph.py "Jannik Sinner 对 Carlos Alcaraz 硬地谁是热门"
  Each run costs two API calls, one for the planner, one for the
  writer. If there is no key, the planner falls back to the rule
  planner, the writer fails with an error, and no report is
  delivered as an LLM report, because the writer has no fallback.
"""

import os
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from planner import parse_question as _rule_parse_question
except ImportError:
    try:
        from planner_lang_v2 import parse_question as _rule_parse_question
    except ImportError:
        _rule_parse_question = None

try:
    from planner_llm import call_gemini as _call_gemini
    from planner_llm import parse_llm_response as _parse_llm_response
except ImportError:
    _call_gemini = None
    _parse_llm_response = None

from scout_graph import _SequentialGraph, _analyst_node_factory  # noqa: E402
from scout_report import build_scout_report  # noqa: E402
from writer_llm import (  # noqa: E402
    _llm_writer_node_factory,
    _verifier_llm_node,
    _writer_final_node,
    detect_language,
)

DEFAULT_MODEL = "gemini-3.5-flash-lite"


class FullGraphState(TypedDict, total=False):
    """Graph state for the full LLM graph. Plain data only."""

    question: str
    plan: dict
    planner_source: str
    player_a: str
    player_b: str
    surface: str
    report: dict
    llm_text: str
    llm_model: str
    verification: dict
    final_text: str
    error: str


def plan_with_llm_graph(question, llm_caller=None):
    """Return (plan, source) for a question, LLM first, rule fallback labelled.

    llm_caller, when given, is a function(question) -> raw text,
    used by tests to avoid a real API call. When it is None, the
    real Gemini call is used, with the key from LLM_API_KEY then
    GEMINI_API_KEY, and the model from LLM_MODEL or
    gemini-3.5-flash-lite. No key means no call is attempted.
    An invalid, null, or failed LLM plan falls back to the rule
    planner, and the source records that, it does not claim llm.
    """
    if llm_caller is None:
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            plan = _rule_parse_question(question) if _rule_parse_question else None
            return plan, "no_key_rule"
        if _call_gemini is None or _parse_llm_response is None:
            plan = _rule_parse_question(question) if _rule_parse_question else None
            return plan, "rule_fallback"
        model = os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        try:
            raw = _call_gemini(question, api_key, model=model)
            plan = _parse_llm_response(raw)
        except Exception:  # noqa: BLE001 - any LLM failure falls back, labelled
            plan = None
        if plan is not None:
            return plan, "llm"
        plan = _rule_parse_question(question) if _rule_parse_question else None
        return plan, "rule_fallback"
    try:
        raw = llm_caller(question)
        plan = _parse_llm_response(raw) if _parse_llm_response else None
    except Exception:  # noqa: BLE001 - any fake caller failure falls back, labelled
        plan = None
    if plan is not None:
        return plan, "llm"
    plan = _rule_parse_question(question) if _rule_parse_question else None
    return plan, "rule_fallback"


def _planner_llm_node_factory(planner_with_source):
    def planner_llm_node(state):
        if state.get("error"):
            return {}
        question = state.get("question") or ""
        if planner_with_source is None:
            return {"error": "planner_not_available"}
        plan, source = planner_with_source(question)
        if plan is None:
            return {"error": "no_template_matched", "plan": None, "planner_source": source}
        return {
            "plan": plan,
            "planner_source": source,
            "player_a": plan.get("player_a"),
            "player_b": plan.get("player_b"),
            "surface": plan.get("surface"),
        }

    return planner_llm_node


def build_full_graph(conn, planner_with_source=None, planner_llm_caller=None,
                     report_builder=None, llm_writer_func=None, api_caller=None,
                     use_langgraph=True):
    """Build the full LLM graph, real LangGraph when available.

    planner_with_source, when given, is a function(question) ->
    (plan, source). When it is None, plan_with_llm_graph is used,
    with planner_llm_caller passed through (None means the real
    Gemini call). report_builder defaults to Step 2
    build_scout_report. llm_writer_func and api_caller are the
    Step 4 writer injectables. Returns (graph, used_langgraph_bool).
    """
    if planner_with_source is None:
        def _default_planner(question):
            return plan_with_llm_graph(question, llm_caller=planner_llm_caller)

        planner_with_source = _default_planner
    if report_builder is None:
        report_builder = build_scout_report

    planner_node = _planner_llm_node_factory(planner_with_source)
    analyst_node = _analyst_node_factory(conn, report_builder)
    llm_writer_node = _llm_writer_node_factory(llm_writer_func, api_caller)
    verifier_llm_node = _verifier_llm_node
    writer_final_node = _writer_final_node

    nodes = [planner_node, analyst_node, llm_writer_node, verifier_llm_node, writer_final_node]
    if not use_langgraph:
        return _SequentialGraph(nodes), False

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return _SequentialGraph(nodes), False

    graph = StateGraph(FullGraphState)
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


def run_full_graph(conn, question, thread_id="full-1", planner_with_source=None,
                   planner_llm_caller=None, report_builder=None,
                   llm_writer_func=None, api_caller=None, use_langgraph=True):
    """Run the full LLM graph for one question, return final state."""
    graph, used_langgraph = build_full_graph(
        conn,
        planner_with_source=planner_with_source,
        planner_llm_caller=planner_llm_caller,
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
        print('Usage: python scripts/full_graph.py "Jannik Sinner vs Carlos Alcaraz Hard"')
        return 1
    from db import get_connection

    question = argv[1]
    try:
        conn = get_connection()
    except Exception as exc:  # noqa: BLE001 - missing SQLite file is a clean CLI error, not a crash
        print(f"ERROR: could not open SQLite database: {exc}")
        return 1
    with conn:
        final_state = run_full_graph(conn, question)
    print(final_state.get("final_text", ""))
    print()
    print(f"Graph used LangGraph: {final_state.get('_used_langgraph')}")
    print(f"Planner source: {final_state.get('planner_source')}")
    if final_state.get("planner_source") not in (None, "llm"):
        if detect_language(question) == "zh":
            print(f"[注意：本次用规则规划器兜底，原因：{final_state.get('planner_source')}。大模型没有参与计划，答案数字仍来自工具。]")
        else:
            print(f"[Note: falling back to the rule planner, reason: {final_state.get('planner_source')}. The LLM did not plan this question; answer numbers still come from tools.]")
    print(f"Plan: {final_state.get('plan')}")
    print(f"LLM model: {final_state.get('llm_model')}")
    print(f"Verification: {final_state.get('verification')}")
    if final_state.get("error"):
        print(f"Error: {final_state['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
