"""Week 7 Step 1: weekly sample report scaffold.

Purpose: give the project a repeatable format for a weekly
published sample report, before any live data is added. This
scaffold uses only historical data, with a fixed default
matchup, Jannik Sinner vs Carlos Alcaraz on Hard, so the
first sample can be checked against numbers already verified
in Weeks 2 to 4. Future weeks change only the input, player
names and surface, not the report structure, the metadata
fields, or the verification rule.

What this script does, and what it does not do:
  It builds a templated scouting report with
  build_scout_report, which uses only database tool outputs,
  no LLM call, no LLM cost, no free text number. It then
  wraps that report text in a sample report document with
  metadata, a sources table, a tool trace, cost and latency,
  and a coverage note. It verifies the templated report
  first, and if verification is not ok, it writes NO file
  and exits 1. An unverified report is never published as
  a sample, even if the report text looks plausible.
  It does NOT claim live data. The data-as-of is
  2026-05-25, the same snapshot used in Weeks 2 to 4, and
  the coverage note says so in plain terms. It does NOT
  predict a specific upcoming real match, it does NOT
  include live scores, recent results after 2026-05-25,
  draws, or injury news. Those are Week 7 live-fixture
  work, not this scaffold, and they must not be added to
  a sample by editing this file's coverage note to sound
  live. When live fixtures are added, they get their own
  section, with their own data-as-of, and the historical
  section stays historical.

Sample report structure, in order, fixed for future weeks:
  Title and matchup, with surface.
  Metadata: sample type, data-as-of, report language,
  verification result, LLM calls, LLM cost, latency,
  tool trace count.
  Scouting report text, verbatim from the templated
  builder, with its own tags, calibration tag always,
  surface tag only when surface is applied.
  Sources table: data source, tables used, data-as-of.
  Tool trace: tool names in the order the report builder
  uses them, match probability, elo for both players,
  form for both players, surface record for both players,
  head-to-head. This is a list of tool names, not a dump
  of every tool output, so the sample stays readable.
  Cost and latency: LLM calls 0, LLM cost $0.00, because
  this scaffold is templated, plus measured latency in
  milliseconds for the report build, and the number of
  tool calls. A future LLM sample must replace these
  fields with its real LLM calls, model, and cost, not
  leave them at 0 while using an LLM.
  Coverage note: historical only, data-as-of, form
  anchored at data-as-of, tourney_date is tournament
  start not match day, serve and ranking coverage limits
  from Week 1, and what this sample does not cover.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/weekly_sample.py
    python scripts/weekly_sample.py --player-a "Novak Djokovic" --player-b "Roger Federer" --surface Hard --output samples/weekly_sample.md
  The default output is samples/weekly_sample.md. The script
  creates the samples directory if it does not exist, writes
  the file, and also prints the full sample to stdout, so a
  manual run can be checked without opening the file.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scout_report import build_scout_report, verify_scout_report  # noqa: E402

DATA_AS_OF = "2026-05-25"
TOOL_TRACE = [
    "get_match_probability",
    "get_player_elo (player_a)",
    "get_player_elo (player_b)",
    "get_player_form (player_a)",
    "get_player_form (player_b)",
    "get_player_surface_record (player_a)",
    "get_player_surface_record (player_b)",
    "get_head_to_head",
]


def build_sample_markdown(report, player_a, player_b, surface, latency_ms):
    """Build the sample report markdown from a verified templated report dict."""
    verification = verify_scout_report(report)
    if not verification.get("ok"):
        raise ValueError(f"report verification not ok: {verification}")
    surface_text = surface if surface else "Overall (no surface specified)"
    report_text = report["text"]
    favorite = report["favorite"]
    prob_a_cal = report["prob_a_calibrated"]
    prob_b_cal = report["prob_b_calibrated"]
    prob_a_blended = report["prob_a_blended"]
    prob_b_blended = report["prob_b_blended"]
    tool_count = len(TOOL_TRACE)
    lines = []
    lines.append(f"# Weekly sample scouting report: {player_a} vs {player_b} ({surface_text})")
    lines.append("")
    lines.append("Sample report, historical data only, not live.")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- Sample type: weekly sample, templated, no LLM writer")
    lines.append(f"- Matchup: {player_a} vs {player_b}")
    lines.append(f"- Surface: {surface_text}")
    lines.append(f"- Data as of: {DATA_AS_OF}, not live")
    lines.append(f"- Verification: ok={verification.get('ok')}, unexpected_numbers={verification.get('unexpected_numbers')}, missing_required={verification.get('missing_required')}, tag_error={verification.get('tag_error')}")
    lines.append(f"- Favorite: {favorite}")
    lines.append(f"- Final probability (calibrated): {player_a} {prob_a_cal}, {player_b} {prob_b_cal}")
    lines.append(f"- Blended probability (before calibration): {player_a} {prob_a_blended}, {player_b} {prob_b_blended}")
    lines.append(f"- LLM calls: 0")
    lines.append(f"- LLM cost: $0.00")
    lines.append(f"- Latency (report build): {latency_ms:.1f} ms")
    lines.append(f"- Tool calls: {tool_count}")
    lines.append("")
    lines.append("## Scouting report")
    lines.append("")
    lines.append(report_text)
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append("| Source | Tables used | Data as of |")
    lines.append("| --- | --- | --- |")
    lines.append("| Jeff Sackmann ATP data, via local PostgreSQL import | atp_players, atp_matches, atp_elo_current, atp_surface_elo_history, atp_form_history, atp_match_games | 2026-05-25, not live |")
    lines.append("")
    lines.append("## Tool trace")
    lines.append("")
    for tool_name in TOOL_TRACE:
        lines.append(f"- {tool_name}")
    lines.append("")
    lines.append("## Cost and latency")
    lines.append("")
    lines.append(f"- LLM calls: 0, LLM cost: $0.00, because this sample is templated, no LLM writer was used.")
    lines.append(f"- Report build latency: {latency_ms:.1f} ms, measured in this run, not a target or a guarantee.")
    lines.append(f"- Tool calls: {tool_count}, all database queries, not model text.")
    lines.append("")
    lines.append("## Coverage note")
    lines.append("")
    lines.append("Historical only. This sample uses matches imported to 2026-05-25, and does not include matches, scores, draws, or news after that date. It does not predict a specific upcoming real match, and it does not include live scores or in-progress points. Form is anchored at the data-as-of date, not at a hypothetical next match. Tourney_date in the source data is tournament start, not exact match day, so the 52-week form window is accurate to within a tournament length. Serve statistics are sparse before the 1990s and are not used as a probability input in this report. Rankings in the import go to 2026-06-08, but the report's data-as-of is 2026-05-25, the maximum match date, and that is the date that governs the report.")
    lines.append("")
    return "\n".join(lines)


def generate_sample(conn, player_a, player_b, surface):
    """Generate a sample report, return (markdown, report, latency_ms), or raise on verification failure."""
    question = f"{player_a} vs {player_b} {surface}" if surface else f"{player_a} vs {player_b}"
    start = time.perf_counter()
    report = build_scout_report(conn, player_a, player_b, surface, question)
    latency_ms = (time.perf_counter() - start) * 1000.0
    if not isinstance(report, dict) or "error" in report:
        raise ValueError(f"report build failed: {report}")
    markdown = build_sample_markdown(report, player_a, player_b, surface, latency_ms)
    return markdown, report, latency_ms


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a weekly sample scouting report, historical only, templated, no LLM.")
    parser.add_argument("--player-a", default="Jannik Sinner")
    parser.add_argument("--player-b", default="Carlos Alcaraz")
    parser.add_argument("--surface", default="Hard")
    parser.add_argument("--output", default="samples/weekly_sample.md")
    args = parser.parse_args(argv)
    from db import get_connection

    try:
        conn = get_connection()
    except Exception as exc:  # noqa: BLE001 - missing SQLite file is a clean CLI error, not a crash
        print(f"ERROR: could not open SQLite database: {exc}")
        return 1
    with conn:
        try:
            markdown, _report, latency_ms = generate_sample(conn, args.player_a, args.player_b, args.surface)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            print("No sample file was written, verification failed.")
            return 1
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print("")
    print(f"Sample written to: {args.output}")
    print(f"Latency (report build): {latency_ms:.1f} ms, LLM calls: 0, LLM cost: $0.00")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
