"""Week 4 Step 3: LangGraph graph skeleton, no LLM writer yet.

Purpose: prove the Week 4 graph flow end to end with the parts
already verified in Week 3 and Week 4 Step 2, before any LLM writer
is added. The graph does not change any number, any tool, any
probability model, or any tag rule. It only runs the existing parts
in order, and it stops the report if the verifier fails.

Graph nodes, in order:
  planner: parse the question into a plan, using the rule planner
    from Week 3. Only a match plan is supported in this skeleton.
    A single-player plan, a head-to-head plan, or a question with
    no matching template returns an error, it does not silently
    build a scouting report for the wrong question.
  analyst: run the templated report builder from Week 4 Step 2.
    That builder already calls all five tools and returns a report
    dict with text, values, tags, and surface_applied.
  verifier: run verify_scout_report from Week 4 Step 2 on that
    report. It checks every value appears in the text, every number
    in the text is a tool value or an allowed parameter, and the
    tags match the actual computation.
  writer: version 1 is a pass-through, not a model. If verification
    passed, final_text is the report text, unchanged. If verification
    failed, or any earlier node errored, final_text is an error
    message only, and the unverified report text is NOT delivered
    as a scouting report. A future LLM writer will replace this
    node only, and it will need a stronger verifier, because LLM
    text can contain numbers in free sentences, not only in
    template slots. That upgrade is NOT part of this step.

State: a plain dict of strings and tool-output dicts, no database
connection in state, so checkpointing can serialize it. The
connection is captured by the node closures when the graph is
built, it is never stored in state and never passed through a node
return value.

LangGraph use, and the fallback:
  When LangGraph 1.x is installed, build_scout_graph builds a real
  StateGraph with START, planner, analyst, verifier, writer, END,
  compiled with an in-memory checkpointer (MemorySaver) when that
  import is available, otherwise compiled without a checkpointer.
  When LangGraph is not installed, or use_langgraph is False, the
  same node functions run in the same order through a small
  sequential graph with the same invoke(state, config=None)
  interface. The fallback exists so the node logic can be tested
  without LangGraph, and so a missing LangGraph install does not
  hide a node bug. On the Mac, LangGraph is installed, so the real
  path is the one that must pass. If the real path raises an import
  or API error on the Mac, stop and paste the error, do not
  silently accept the fallback as a pass for the LangGraph step.

What this file does NOT do:
  It does not use an LLM, an API key, or live data.
  It does not change the locked probability model. Surface blend
  prior=100 and calibration a=0.8, b=0.0 are the only corrections
  in the probability, exactly as locked in Week 2.
  It does not support non-match questions as scouting reports.
  It does not save state to disk. MemorySaver is in-memory only,
  so a restart loses the thread history. Persistent checkpointing
  is a later step, not this skeleton.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/scout_graph.py "Jannik Sinner vs Carlos Alcaraz Hard"
    python scripts/scout_graph.py "Jannik Sinner 对 Carlos Alcaraz 硬地谁是热门"
  The question is the only required argument. The graph language
  follows the report builder: a Chinese question gives a Chinese
  report, otherwise English.
"""

import os
import sys
from pathlib import Path
from typing import Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from planner import parse_question as _default_parse_question
except ImportError:  # pragma: no cover - planner file name on some setups
    try:
        from planner_lang_v2 import parse_question as _default_parse_question
    except ImportError:
        _default_parse_question = None

from scout_report import build_scout_report, verify_scout_report  # noqa: E402


class ScoutState(TypedDict, total=False):
    """Graph state. All fields are plain data, no connection."""

    question: str
    plan: dict
    player_a: str
    player_b: str
    surface: str
    report: dict
    verification: dict
    final_text: str
    error: str


def _planner_node_factory(planner_func):
    def planner_node(state):
        if state.get("error"):
            return {}
        question = state.get("question") or ""
        if planner_func is None:
            return {"error": "planner_not_available"}
        plan = planner_func(question)
        if plan is None:
            return {"error": "no_template_matched", "plan": None}
        return {
            "plan": plan,
            "player_a": plan.get("player_a"),
            "player_b": plan.get("player_b"),
            "surface": plan.get("surface"),
        }

    return planner_node


def _analyst_node_factory(conn, report_builder):
    def analyst_node(state):
        if state.get("error"):
            return {}
        plan = state.get("plan") or {}
        if plan.get("tool") != "match":
            return {"error": f"unsupported_plan:{plan.get('tool')}"}
        player_a = state.get("player_a")
        player_b = state.get("player_b")
        surface = state.get("surface")
        question = state.get("question")
        report = report_builder(conn, player_a, player_b, surface, question)
        if not isinstance(report, dict) or "error" in report:
            err = report.get("error") if isinstance(report, dict) else "report_builder_failed"
            return {"error": str(err), "report": None}
        return {"report": report}

    return analyst_node


def _verifier_node_factory(verifier_func):
    def verifier_node(state):
        if state.get("error"):
            return {}
        report = state.get("report")
        if report is None:
            return {"error": "no_report_to_verify"}
        verification = verifier_func(report)
        return {"verification": verification}

    return verifier_node


def _writer_node(state):
    if state.get("error"):
        return {"final_text": f"ERROR: {state['error']}"}
    verification = state.get("verification") or {}
    report = state.get("report") or {}
    if not verification.get("ok"):
        detail = (
            f"missing={verification.get('missing_values')}, "
            f"unexpected={verification.get('unexpected_numbers')}, "
            f"tag_error={verification.get('tag_error')}"
        )
        return {
            "error": "verification_failed",
            "final_text": f"ERROR: verification failed: {detail}",
        }
    return {"final_text": report.get("text", "")}


class _SequentialGraph:
    """Fallback graph with the same invoke interface as LangGraph.

    Runs the node functions in planner, analyst, verifier, writer
    order, merging each returned dict into state. Used only when
    LangGraph is not installed, or when use_langgraph is False in
    tests. It is not a substitute for a LangGraph pass on the Mac.
    """

    def __init__(self, node_funcs):
        self._node_funcs = node_funcs

    def invoke(self, state, config=None):
        current = dict(state)
        for func in self._node_funcs:
            update = func(current) or {}
            current.update(update)
        return current


def build_scout_graph(conn, planner_func=None, report_builder=None,
                      verifier_func=None, use_langgraph=True):
    """Build the scouting graph, real LangGraph when available.

    planner_func defaults to the Week 3 rule planner parse_question.
    report_builder defaults to Week 4 Step 2 build_scout_report.
    verifier_func defaults to Week 4 Step 2 verify_scout_report.
    Injectable functions exist so pure tests can use fakes without
    a database, and so a future LLM planner can replace only the
    planner function, without rewriting the graph.
    Returns (graph, used_langgraph_bool).
    """
    if planner_func is None:
        planner_func = _default_parse_question
    if report_builder is None:
        report_builder = build_scout_report
    if verifier_func is None:
        verifier_func = verify_scout_report

    planner_node = _planner_node_factory(planner_func)
    analyst_node = _analyst_node_factory(conn, report_builder)
    verifier_node = _verifier_node_factory(verifier_func)
    writer_node = _writer_node

    if not use_langgraph:
        return _SequentialGraph([planner_node, analyst_node, verifier_node, writer_node]), False

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return _SequentialGraph([planner_node, analyst_node, verifier_node, writer_node]), False

    graph = StateGraph(ScoutState)
    graph.add_node("planner", planner_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("writer", writer_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "analyst")
    graph.add_edge("analyst", "verifier")
    graph.add_edge("verifier", "writer")
    graph.add_edge("writer", END)
    try:
        from langgraph.checkpoint.memory import MemorySaver

        compiled = graph.compile(checkpointer=MemorySaver())
    except ImportError:
        compiled = graph.compile()
    return compiled, True


def run_scout_graph(conn, question, thread_id="scout-1", planner_func=None,
                    report_builder=None, verifier_func=None, use_langgraph=True):
    """Run the graph for one question and return the final state."""
    graph, used_langgraph = build_scout_graph(
        conn,
        planner_func=planner_func,
        report_builder=report_builder,
        verifier_func=verifier_func,
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
        print('Usage: python scripts/scout_graph.py "Jannik Sinner vs Carlos Alcaraz Hard"')
        return 1
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a")
        return 1
    import psycopg

    question = argv[1]
    with psycopg.connect(database_url) as conn:
        final_state = run_scout_graph(conn, question)
    print(final_state.get("final_text", ""))
    print()
    print(f"Graph used LangGraph: {final_state.get('_used_langgraph')}")
    print(f"Plan: {final_state.get('plan')}")
    print(f"Verification: {final_state.get('verification')}")
    if final_state.get("error"):
        print(f"Error: {final_state['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
