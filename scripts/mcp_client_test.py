"""Manual MCP client test for Week 5 Step 3, no Inspector, no uv.

Why this file exists: the MCP Inspector's default server command
is "uv run --with mcp==2.3.0 mcp run scripts/mcp_server.py".
On a Mac where uv is not installed, or not in the Inspector
process PATH, that command fails with spawn uv ENOENT, before
any tool is called. Even if uv were installed, that uv
environment would not have the venv's psycopg, and would not
have DATABASE_URL or LLM_API_KEY from ~/TSA/.env, so the
database tools and the scouting report would fail anyway.
This client avoids both problems: it runs the server with the
venv Python, from ~/TSA, with the current environment inherited,
so DATABASE_URL and LLM_API_KEY from .env are available, and
psycopg is available from the venv.

What it does, in order, and what it costs:
  1. Lists tools, expected 6 tools.
  2. Calls get_player_elo with Jannik Sinner, expected overall
     2283.1, matches 457, no LLM call, no cost.
  3. Calls get_match_probability with Sinner vs Alcaraz Hard,
     expected calibrated 0.6378, blended 0.6698, no LLM call.
  4. Calls generate_scouting_report with the same Hard question,
     expected planner_source llm, verification ok, calibrated
     0.6378, blended 0.6698. This step costs two LLM calls,
     one for the planner, one for the writer, so run it once
     and check the output, do not loop it.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/mcp_client_test.py
If a tool returns an error JSON, that JSON is printed as-is,
not hidden, so a database or verification failure is visible.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


async def main():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["scripts/mcp_server.py"],
        env=dict(os.environ),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Tools:", sorted([t.name for t in tools.tools]))
            elo = await session.call_tool("get_player_elo", {"player": "Jannik Sinner"})
            print("Elo:", elo.content[0].text if elo.content else "")
            prob = await session.call_tool(
                "get_match_probability",
                {"player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz", "surface": "Hard"},
            )
            print("Probability:", prob.content[0].text if prob.content else "")
            report = await session.call_tool(
                "generate_scouting_report",
                {"question": "Jannik Sinner vs Carlos Alcaraz Hard"},
            )
            report_text = report.content[0].text if report.content else ""
            try:
                report_json = json.loads(report_text)
                print("Report planner_source:", report_json.get("planner_source"))
                print("Report verification:", report_json.get("verification"))
                print("Report final_text:")
                print(report_json.get("final_text", report_text))
            except json.JSONDecodeError:
                print("Report raw:", report_text)


if __name__ == "__main__":
    asyncio.run(main())
