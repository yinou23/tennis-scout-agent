"""Week 5 Step 2 CORS update: FastAPI interface for the full graph.

Purpose: turn the command-line full graph from Week 4 Step 5
into an HTTP service, so a web UI in the next step, and later
an MCP server, can call the same verified report pipeline,
instead of each caller re-implementing planner, tools, writer,
and verifier. This step adds an interface only. It does not
change any tool, any probability, any verifier rule, or any
writer rule.

Endpoints:
  GET  /health
    Returns {"status": "ok", "step": "week5-step1"}.
    No database, no LLM, no API key needed. Use it to check
    the server is up before calling /report.
  POST /report
    Body: {"question": "Jannik Sinner vs Carlos Alcaraz Hard"}
    Success, HTTP 200, returns:
      question, final_text, planner_source, plan, verification,
      llm_model, graph_used_langgraph.
    Failure, HTTP 422, returns:
      error, planner_source, plan, verification, with NO
      final_text presented as a report. A failed planner, a
      failed writer, or a failed verification is not a report,
      and the API must not present it as one. The error message
      from the graph already starts with ERROR, but the API
      response on failure does not include it as final_text,
      so a UI cannot accidentally display an unverified report
      as if it had passed.
    Each /report call costs two LLM calls in the real runner,
    one for the planner, one for the writer, plus database
    queries. The test runner in tests is fake and costs nothing.

Runner injection, for tests:
  create_app(graph_runner=None) builds the app. graph_runner,
  when given, is a function(question) -> final_state dict, used
  by tests to avoid a database and a real LLM call. When it is
  None, the default runner opens a psycopg connection from
  DATABASE_URL, runs run_full_graph, and closes the connection.
  If DATABASE_URL is not set, the default runner returns a
  final_state with an error, and /report returns 422, it does
  not crash the server.
  The graph runner is stored on app.state.graph_runner, so a
  test can also replace it after creation.

CORS, added in Week 5 Step 2, and the ONLY change in this version:
  The Next.js dev server runs on port 3000 and the API runs on
  port 8000, so without CORS the browser blocks the fetch.
  This version allows only http://127.0.0.1:3000 and
  http://localhost:3000, only GET and POST. /health and /report
  logic, and the /health step value, are unchanged, so the
  Week 5 Step 1 tests still pass.

What this file does NOT do:
  It does not serve a web page, that is the Next.js step next.
  It does not serve an MCP server, that is a later Week 5 step.
  It does not add authentication, rate limiting, or persistence.
  Those are deployment concerns, not Week 5 Step 1.
  It does not change the locked probability model. Surface blend
  prior=100 and calibration a=0.8, b=0.0 are the only corrections.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python -c "import uvicorn" || python -m pip install uvicorn
    python scripts/api.py
  Then, in a second terminal, with the server still running:
    curl http://127.0.0.1:8000/health
    curl -X POST http://127.0.0.1:8000/report \
      -H "Content-Type: application/json" \
      -d '{"question": "Jannik Sinner vs Carlos Alcaraz Hard"}'
  Stop the server with Ctrl-C in the server terminal when done.
  If fastapi is not installed, importing this file still works
  for the pure response builder, but create_app raises a clear
  error instead of a confusing import error at server start.
"""

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - assistant VM has no fastapi, Mac does
    FastAPI = None
    HTTPException = None
    BaseModel = object
    FASTAPI_AVAILABLE = False


def build_report_response(final_state):
    """Turn a graph final_state into (status_code, payload).

    Pure function, no FastAPI, no database, no LLM, so it can be
    tested everywhere. Success is only when there is no error
    AND verification ok is True. Anything else is 422, with no
    final_text presented as a report.
    """
    if not isinstance(final_state, dict):
        return 422, {"error": "invalid_final_state"}
    error = final_state.get("error")
    verification = final_state.get("verification") or {}
    if error:
        return 422, {
            "error": str(error),
            "planner_source": final_state.get("planner_source"),
            "plan": final_state.get("plan"),
            "verification": verification,
        }
    if not verification.get("ok"):
        return 422, {
            "error": "verification_failed",
            "planner_source": final_state.get("planner_source"),
            "plan": final_state.get("plan"),
            "verification": verification,
        }
    return 200, {
        "question": final_state.get("question"),
        "final_text": final_state.get("final_text", ""),
        "planner_source": final_state.get("planner_source"),
        "plan": final_state.get("plan"),
        "verification": verification,
        "llm_model": final_state.get("llm_model"),
        "graph_used_langgraph": final_state.get("_used_langgraph"),
    }


def default_report_runner(question):
    """Run the real full graph for one question, return final_state."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return {"error": "database_url_not_set", "question": question}
    try:
        import psycopg
    except ImportError as exc:
        return {"error": f"psycopg_not_installed:{exc}", "question": question}
    try:
        from full_graph import run_full_graph
    except ImportError as exc:
        return {"error": f"full_graph_not_installed:{exc}", "question": question}
    try:
        with psycopg.connect(database_url) as conn:
            final_state = run_full_graph(conn, question, thread_id=f"api-{uuid.uuid4().hex}")
    except Exception as exc:  # noqa: BLE001 - API must report, not crash, on a runner failure
        return {"error": f"runner_failed:{type(exc).__name__}:{exc}", "question": question}
    final_state = dict(final_state)
    final_state["question"] = question
    return final_state


def create_app(graph_runner=None):
    """Build the FastAPI app, with an injectable graph runner."""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("fastapi is not installed. Run: python -m pip install fastapi uvicorn")
    app = FastAPI(title="Tennis Scout Agent")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.state.graph_runner = graph_runner or default_report_runner

    class ReportRequest(BaseModel):
        question: str

    @app.get("/health")
    def health():
        return {"status": "ok", "step": "week5-step1"}

    @app.post("/report")
    def report(request: ReportRequest):
        question = (request.question or "").strip()
        if not question:
            raise HTTPException(status_code=422, detail={"error": "empty_question"})
        runner = app.state.graph_runner or default_report_runner
        try:
            final_state = runner(question)
        except Exception as exc:  # noqa: BLE001 - API must report, not crash, on a runner failure
            raise HTTPException(
                status_code=422,
                detail={"error": f"runner_failed:{type(exc).__name__}:{exc}"},
            ) from exc
        status_code, payload = build_report_response(final_state)
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=payload)
        return payload

    return app


app = None
try:
    app = create_app()
except RuntimeError:
    # FastAPI is not installed on this machine (for example, the
    # assistant VM). The pure build_report_response function above
    # is still importable and testable. On the Mac, fastapi is
    # installed, so app is created normally.
    app = None


def main():
    if app is None:
        print("ERROR: fastapi is not installed. Run: python -m pip install fastapi uvicorn")
        return 1
    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is not installed. Run: python -m pip install uvicorn")
        return 1
    uvicorn.run(app, host="127.0.0.1", port=8000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
