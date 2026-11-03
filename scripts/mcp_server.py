"""Week 5 Step 3: MCP server for the Tennis Scout Agent.

Purpose: expose the same verified tools and the same full
scouting report to other agents and hosts, through the Model
Context Protocol, instead of only through the FastAPI HTTP API
and the Next.js page. A host that can call MCP tools can call
these tools without custom glue, and it gets the same numbers,
because the tools are the same Week 3 deterministic tools and
the same Week 4 Step 5 full graph, not a second copy.

Tools, six in total, all return a JSON string, never a bare
number and never a free-text number that is not in the tool
output. A client that parses the JSON gets the numbers with
their labels, so it does not have to scrape a report sentence.
  get_player_elo(player)
    Week 3 Tool 1. Overall Elo, per-surface Elo and blended
    Elo, matches played. Returns {"error": "player_not_found"}
    or an ambiguous dict with candidates, never a guessed
    player, when the name does not match exactly one player.
  get_player_form(player)
    Week 3 Tool 2. Last 10 and last 52 weeks win rates, games
    percentages, and the retirement descriptor. Retirement is
    reported only as a mid-match retirement event, never as an
    injury diagnosis. Form is a descriptor only.
  get_head_to_head(player_a, player_b)
    Week 3 Tool 3. Total meetings, wins for each player, and
    per-surface meetings. Raw counts only, no shrinkage and
    no probability.
  get_player_surface_record(player)
    Week 3 Tool 4. Career win and loss counts per surface,
    distinct from surface Elo.
  get_match_probability(player_a, player_b, surface=None)
    Week 3 Tool 5, locked model only. Surface blend prior=100
    and calibration a=0.8, b=0.0 are the only corrections.
    Returns blended and calibrated probabilities, with the
    correction tags in the same form as the report. Surface
    must be Hard, Clay, Grass, Carpet, or None. An unknown
    surface is treated as None, exactly as in Tool 5, and the
    returned surface field says what was actually used.
  generate_scouting_report(question)
    Week 4 Step 5 full graph, planner_llm, analyst,
    llm_writer, verifier_llm, writer_final. Costs two LLM
    calls in the real runner, one for the planner, one for
    the writer. Returns final_text, planner_source, plan,
    verification, llm_model, and graph_used_langgraph, only
    when verification ok is True. When the planner fails, the
    writer fails, or verification fails, it returns an error
    JSON with NO final_text presented as a report, exactly as
    the FastAPI /report endpoint does with HTTP 422.

What this file does NOT do:
  It does not invent numbers. Every number in a tool return
  comes from a database query, or, for the scouting report,
  from a verified LLM rewrite of a templated report, checked
  by verify_llm_report. It does not change any tool, any
  probability, any verifier rule, or any writer rule.
  It does not write to stdout in tool functions. For a stdio
  MCP server, stdout is the JSON-RPC channel, and any print
  to stdout corrupts it. Logs and errors in this file go to
  stderr only, and tool errors are returned as JSON, not
  printed.
  Name shadowing, fixed in the second fix: the MCP tool
  functions must be named get_player_elo and so on, because
  that name becomes the tool name a client sees, so the
  tennis_tools imports are aliased to _tennis_* names.
  Without the alias, the tool definition overwrites the
  imported function at module load, and the implementation
  then calls the tool function with a connection argument
  it does not take, which is exactly the TypeError seen on
  the Mac after the first fix made the adapter work.
  It does not require the MCP SDK to import the pure tool
  implementations. The pure implementations are plain
  functions that take a connection and return a JSON string,
  so they can be tested without the SDK. The FastMCP adapter
  at the bottom is guarded: if neither the official SDK
  (from mcp.server.fastmcp import FastMCP) nor the standalone
  fastmcp package (from fastmcp import FastMCP) is installed,
  the module still imports, the adapter is None, and main()
  prints a clear install instruction to stderr and exits
  non-zero, instead of a confusing import error at import time.

SDK choice, stated plainly, because the ecosystem is confusing
in 2026:
  The official Python SDK v1.x provides FastMCP from
  mcp.server.fastmcp. In SDK 2.x, which pip install
  "mcp[cli]" installs in November 2026, FastMCP was renamed
  to MCPServer, importable from mcp.server, with the same
  @mcp.tool() decorator and mcp.run(transport="stdio")
  signature, verified locally with a stdio client call.
  The standalone fastmcp package, version 3.x, provides
  FastMCP from fastmcp. This file tries, in order: official
  v1 FastMCP, standalone fastmcp FastMCP, official v2
  MCPServer from mcp.server, and MCPServer from
  mcp.server.mcpserver.server, and uses whichever is
  installed, so the user does not have to guess which
  package version their venv already has. The install
  instruction in main() names the official SDK, because
  that is the SDK that ships with the spec: pip install
  "mcp[cli]".

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python -c "import mcp" || python -m pip install "mcp[cli]"
    python scripts/mcp_server.py
  Running it directly starts a stdio MCP server and waits for
  JSON-RPC on stdin. Do not expect readable output on the
  terminal, that is normal for stdio. For a visual test, use
  the MCP Inspector, which the official SDK provides:
    mcp dev scripts/mcp_server.py
  Then call get_player_elo with "Jannik Sinner" and check
  overall 2283.1, and call generate_scouting_report with
  "Jannik Sinner vs Carlos Alcaraz Hard" and check planner
  source llm, verification ok, calibrated 0.6378, blended
  0.6698. Stop the Inspector server with Ctrl-C when done.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tennis_tools import SURFACES  # noqa: E402
from tennis_tools import get_head_to_head as _tennis_get_head_to_head  # noqa: E402
from tennis_tools import get_match_probability as _tennis_get_match_probability  # noqa: E402
from tennis_tools import get_player_elo as _tennis_get_player_elo  # noqa: E402
from tennis_tools import get_player_form as _tennis_get_player_form  # noqa: E402
from tennis_tools import get_player_surface_record as _tennis_get_player_surface_record  # noqa: E402

TOOL_NAMES = (
    "get_player_elo",
    "get_player_form",
    "get_head_to_head",
    "get_player_surface_record",
    "get_match_probability",
    "generate_scouting_report",
)


def _to_json(value):
    """Serialize a tool result to JSON, dates as strings, never crash on a date."""
    return json.dumps(value, default=str, ensure_ascii=False)


def _error_json(message):
    return _to_json({"error": str(message)})


def _open_conn():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("database_url_not_set")
    import psycopg

    return psycopg.connect(database_url)


def _run_with_conn(impl, *args, **kwargs):
    """Open a connection, run impl(conn, ...), close it, return a JSON string."""
    try:
        conn = _open_conn()
    except Exception as exc:  # noqa: BLE001 - MCP tools must return JSON errors, not crash the server
        return _error_json(f"database_connection_failed:{type(exc).__name__}:{exc}")
    try:
        with conn:
            result = impl(conn, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - same reason as above
        return _error_json(f"tool_failed:{type(exc).__name__}:{exc}")
    if result is None:
        return _error_json("player_not_found")
    return _to_json(result)


def _elo_impl(conn, player):
    return _tennis_get_player_elo(conn, player)


def _form_impl(conn, player):
    return _tennis_get_player_form(conn, player)


def _h2h_impl(conn, player_a, player_b):
    return _tennis_get_head_to_head(conn, player_a, player_b)


def _surface_impl(conn, player):
    return _tennis_get_player_surface_record(conn, player)


def _prob_impl(conn, player_a, player_b, surface=None):
    if surface is not None and surface not in SURFACES:
        surface = None
    return _tennis_get_match_probability(conn, player_a, player_b, surface)


def _scouting_report_impl(conn, question, graph_runner=None):
    """Run the full graph for a question, return a JSON string.

    graph_runner, when given, is a function(conn, question) ->
    final_state dict, used by tests to avoid a real LLM call.
    When it is None, the real Week 4 Step 5 run_full_graph is
    used. On any error, or on verification failure, returns an
    error JSON with NO final_text presented as a report.
    """
    if graph_runner is not None:
        final_state = graph_runner(conn, question)
    else:
        try:
            from full_graph import run_full_graph
        except ImportError as exc:
            return _error_json(f"full_graph_not_installed:{exc}")
        final_state = run_full_graph(conn, question)
    if not isinstance(final_state, dict):
        return _error_json("invalid_final_state")
    if final_state.get("error"):
        return _to_json({
            "error": str(final_state["error"]),
            "planner_source": final_state.get("planner_source"),
            "plan": final_state.get("plan"),
            "verification": final_state.get("verification") or {},
        })
    verification = final_state.get("verification") or {}
    if not verification.get("ok"):
        return _to_json({
            "error": "verification_failed",
            "planner_source": final_state.get("planner_source"),
            "plan": final_state.get("plan"),
            "verification": verification,
        })
    return _to_json({
        "question": question,
        "final_text": final_state.get("final_text", ""),
        "planner_source": final_state.get("planner_source"),
        "plan": final_state.get("plan"),
        "verification": verification,
        "llm_model": final_state.get("llm_model"),
        "graph_used_langgraph": final_state.get("_used_langgraph"),
    })


def get_player_elo_tool(player):
    """Get a player's Elo, overall and per-surface blended. Args: player, full name."""
    return _run_with_conn(_elo_impl, player)


def get_player_form_tool(player):
    """Get a player's recent form and retirement descriptor. Args: player, full name."""
    return _run_with_conn(_form_impl, player)


def get_head_to_head_tool(player_a, player_b):
    """Get head-to-head counts for two players. Args: player_a, player_b, full names."""
    return _run_with_conn(_h2h_impl, player_a, player_b)


def get_player_surface_record_tool(player):
    """Get a player's career win and loss counts per surface. Args: player, full name."""
    return _run_with_conn(_surface_impl, player)


def get_match_probability_tool(player_a, player_b, surface=None):
    """Get a match probability, locked model only. Args: player_a, player_b, surface Hard, Clay, Grass, Carpet, or None."""
    return _run_with_conn(_prob_impl, player_a, player_b, surface)


def generate_scouting_report_tool(question, graph_runner=None):
    """Generate a verified scouting report for a match question. Args: question, for example Jannik Sinner vs Carlos Alcaraz Hard."""
    # _run_with_conn passes conn as the first argument to the impl,
    # and the impl signature is (conn, question, graph_runner).
    # Use a small wrapper so the graph_runner injection works.
    def _impl(conn, question_inner):
        return _scouting_report_impl(conn, question_inner, graph_runner=graph_runner)

    # _scouting_report_impl returns a JSON string, not a dict, so
    # it cannot go through _run_with_conn, which would JSON-encode
    # the string again. Handle the connection here instead.
    try:
        conn = _open_conn()
    except Exception as exc:  # noqa: BLE001 - MCP tools must return JSON errors, not crash
        return _error_json(f"database_connection_failed:{type(exc).__name__}:{exc}")
    try:
        with conn:
            return _impl(conn, question)
    except Exception as exc:  # noqa: BLE001 - same reason as above
        return _error_json(f"tool_failed:{type(exc).__name__}:{exc}")


# FastMCP / MCPServer adapter, guarded so pure functions import without the SDK.
mcp = None
_FastMCP = None
for _import_path in (
    "from mcp.server.fastmcp import FastMCP as _FastMCP",
    "from fastmcp import FastMCP as _FastMCP",
    "from mcp.server import MCPServer as _FastMCP",
    "from mcp.server.mcpserver.server import MCPServer as _FastMCP",
):
    try:
        exec(_import_path)
        break
    except ImportError:
        continue
if _FastMCP is not None:
    mcp = _FastMCP("tennis-scout-agent")

    @mcp.tool()
    def get_player_elo(player: str) -> str:
        """Get a player's Elo, overall and per-surface blended."""
        return get_player_elo_tool(player)

    @mcp.tool()
    def get_player_form(player: str) -> str:
        """Get a player's recent form and retirement descriptor."""
        return get_player_form_tool(player)

    @mcp.tool()
    def get_head_to_head(player_a: str, player_b: str) -> str:
        """Get head-to-head counts for two players."""
        return get_head_to_head_tool(player_a, player_b)

    @mcp.tool()
    def get_player_surface_record(player: str) -> str:
        """Get a player's career win and loss counts per surface."""
        return get_player_surface_record_tool(player)

    @mcp.tool()
    def get_match_probability(player_a: str, player_b: str, surface: str = None) -> str:
        """Get a match probability, locked model only."""
        return get_match_probability_tool(player_a, player_b, surface)

    @mcp.tool()
    def generate_scouting_report(question: str) -> str:
        """Generate a verified scouting report for a match question."""
        return generate_scouting_report_tool(question)


def main():
    if mcp is None:
        print(
            "ERROR: MCP SDK is not installed. Run: python -m pip install \"mcp[cli]\"",
            file=sys.stderr,
        )
        return 1
    try:
        mcp.run(transport="stdio")
    except TypeError:
        mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
