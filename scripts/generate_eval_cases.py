"""Week 6 Step 3: expand eval set from 30 cases to 100 cases.

This script does NOT hand-write expected values, and it does
NOT overwrite the canonical eval/cases.jsonl. It reads the
canonical 30 cases, keeps them verbatim, generates 70 new cases
from the current database tools, and writes the full 100 cases
to a separate file, eval/cases_expanded.jsonl, for review.
Only after the review, with the expanded eval run passing,
is the expanded file moved to eval/cases.jsonl, in a separate
manual step, not by this script. That gate is deliberate:
a generated snapshot must be reviewed before it becomes
canonical, otherwise a bad generation would silently become
the new truth.
This v2 generator supersedes the Step 2 generator. The Step 2
players and matches are already in the canonical 30 cases,
so they are NOT regenerated here, otherwise they would
collide with canonical ids.

New cases, 70 in total, 30 plus 70 equals 100:
  NEW_PLAYERS, 15 players, 3 cases each, 45 cases:
    Andy Murray, Stan Wawrinka, Dominic Thiem,
    Stefanos Tsitsipas, Andrey Rublev, Casper Ruud,
    Taylor Fritz, Ben Shelton, Frances Tiafoe,
    Lorenzo Musetti, Holger Rune, Jack Draper,
    Tommy Paul, Grigor Dimitrov, Matteo Berrettini
    Each player gets one elo case, one form case, and one
    surface case. Players already in the canonical 30 are
    deliberately NOT repeated here.
  NEW_MATCHES, 25 pairs, 1 case each, 25 cases:
    See NEW_MATCHES below. Each pair is a real historical
    matchup in most cases, but a pair with H2H total 0 is
    still a valid regression case, prob is Elo based, H2H
    is 0, and the per-surface H2H entry is omitted when
    the players never met on that surface.
  No new ambiguous or unknown cases in this step, those
  failure modes are already covered in the original 11,
  and adding more of them would not add regression value.

What is taken from each tool output, and what is not:
  Elo: overall_elo, matches_played, and blended_elo for
  every surface the tool returns. The tool always returns
  Hard, Clay, Grass, and Carpet, with blended equal to
  overall when the player has no matches on that surface.
  Form: win_form, games_form, and retirement, in full,
  including None values when a player has no matches in a
  window. None is a real expected value here, not a missing
  value, and the eval runner compares it exactly.
  Surface: by_surface for every surface present in the
  tool output, plus total_matches, total_wins, total_losses.
  A surface the player never played on has no entry, and
  is simply omitted, never invented as 0 wins.
  Match: the templated report builder, not the LLM writer,
  so the case is deterministic. Favorite, calibrated and
  blended probabilities, H2H total and wins, surface_applied,
  and, only when the H2H by_surface entry for the match
  surface exists, the per-surface H2H entry. If the two
  players never met on that surface, the per-surface entry
  is omitted, because the eval runner would otherwise
  compare None against 0 and fail a case that is actually
  correct. If the report verification is not ok, or the
  rule planner does not parse the generated question back
  to the same plan, that match case is skipped with a
  warning printed, never added as an unverified case.

Safety checks in the script, all printed, none silent:
  A new player that is ambiguous or missing is skipped,
  with a warning, not added as a broken case.
  A case id that already exists in the original set is
  never overwritten, and the script aborts if a new id
  collides with an original id.
  The original 11 cases are written first, verbatim, and
  the script prints their count, so a review can check
  that the original set was not changed.
  The script prints, for each new match case, the favorite
  and the calibrated probability, and for each new elo
  case, the overall Elo and matches, so a human can spot
  a nonsense value before moving the file to canonical.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/generate_eval_cases.py
    python scripts/eval.py --cases eval/cases_expanded.jsonl --failures eval/failures_expanded.jsonl
  Review the expanded eval output first. Only after it
  passes 30 out of 30, move it to canonical:
    mv eval/cases_expanded.jsonl eval/cases.jsonl
    python scripts/eval.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from planner import parse_question  # noqa: E402
from scout_report import build_scout_report, verify_scout_report  # noqa: E402
from tennis_tools import (  # noqa: E402
    get_player_elo,
    get_player_form,
    get_player_surface_record,
)

NEW_PLAYERS = [
    "Andy Murray",
    "Stan Wawrinka",
    "Dominic Thiem",
    "Stefanos Tsitsipas",
    "Andrey Rublev",
    "Casper Ruud",
    "Taylor Fritz",
    "Ben Shelton",
    "Frances Tiafoe",
    "Lorenzo Musetti",
    "Holger Rune",
    "Jack Draper",
    "Tommy Paul",
    "Grigor Dimitrov",
    "Matteo Berrettini",
]

NEW_MATCHES = [
    ("Andy Murray", "Stan Wawrinka", "Hard"),
    ("Andy Murray", "Dominic Thiem", "Clay"),
    ("Stan Wawrinka", "Dominic Thiem", "Hard"),
    ("Stefanos Tsitsipas", "Andrey Rublev", "Hard"),
    ("Stefanos Tsitsipas", "Casper Ruud", "Clay"),
    ("Andrey Rublev", "Casper Ruud", "Hard"),
    ("Taylor Fritz", "Ben Shelton", "Hard"),
    ("Taylor Fritz", "Frances Tiafoe", "Hard"),
    ("Ben Shelton", "Frances Tiafoe", "Hard"),
    ("Lorenzo Musetti", "Holger Rune", "Clay"),
    ("Lorenzo Musetti", "Jack Draper", "Grass"),
    ("Holger Rune", "Jack Draper", "Hard"),
    ("Tommy Paul", "Grigor Dimitrov", "Hard"),
    ("Tommy Paul", "Matteo Berrettini", "Grass"),
    ("Grigor Dimitrov", "Matteo Berrettini", "Hard"),
    ("Novak Djokovic", "Andy Murray", "Hard"),
    ("Novak Djokovic", "Stefanos Tsitsipas", "Clay"),
    ("Roger Federer", "Andy Murray", "Grass"),
    ("Rafael Nadal", "Dominic Thiem", "Clay"),
    ("Alexander Zverev", "Stefanos Tsitsipas", "Hard"),
    ("Daniil Medvedev", "Andrey Rublev", "Hard"),
    ("Jannik Sinner", "Novak Djokovic", "Hard"),
    ("Carlos Alcaraz", "Rafael Nadal", "Clay"),
    ("Jannik Sinner", "Taylor Fritz", "Hard"),
    ("Carlos Alcaraz", "Ben Shelton", "Hard"),
]


def _slug(name):
    return name.lower().replace(" ", "-")


def extract_elo_expected(tool_output):
    """Return the eval expected dict for an elo tool output."""
    return {
        "overall_elo": tool_output["overall_elo"],
        "matches_played": tool_output["matches_played"],
        "surfaces": {
            surface: {"blended_elo": entry["blended_elo"]}
            for surface, entry in tool_output["surfaces"].items()
        },
    }


def extract_form_expected(tool_output):
    """Return the eval expected dict for a form tool output."""
    return {
        "win_form": tool_output["win_form"],
        "games_form": tool_output["games_form"],
        "retirement": tool_output["retirement"],
    }


def extract_surface_expected(tool_output):
    """Return the eval expected dict for a surface tool output."""
    return {
        "by_surface": tool_output["by_surface"],
        "total_matches": tool_output["total_matches"],
        "total_wins": tool_output["total_wins"],
        "total_losses": tool_output["total_losses"],
    }


def build_match_case(conn, player_a, player_b, surface):
    """Build one match eval case from the templated report, or None if unverified."""
    question = f"{player_a} vs {player_b} {surface}"
    plan = parse_question(question)
    if plan is None or plan.get("tool") != "match":
        print(f"WARNING: question did not parse to match plan, skipping: {question} -> {plan}")
        return None
    if plan["player_a"] != player_a or plan["player_b"] != player_b or plan.get("surface") != surface:
        print(f"WARNING: parsed plan does not match intended pair, skipping: {question} -> {plan}")
        return None
    report = build_scout_report(conn, player_a, player_b, surface, question)
    if not isinstance(report, dict) or "error" in report:
        print(f"WARNING: report error, skipping match case: {question} -> {report}")
        return None
    if not verify_scout_report(report).get("ok"):
        print(f"WARNING: report verification not ok, skipping match case: {question}")
        return None
    h2h = report["tools"]["h2h"]
    expected = {
        "favorite": report["favorite"],
        "prob_a_calibrated": report["prob_a_calibrated"],
        "prob_b_calibrated": report["prob_b_calibrated"],
        "prob_a_blended": report["prob_a_blended"],
        "prob_b_blended": report["prob_b_blended"],
        "h2h_total": h2h["total"],
        "h2h_wins_a": h2h["wins_a"],
        "h2h_wins_b": h2h["wins_b"],
        "surface_applied": report["surface_applied"],
    }
    surface_entry = h2h.get("by_surface", {}).get(surface)
    if surface_entry is not None:
        expected["h2h_surface"] = {
            "surface": surface,
            "total": surface_entry["total"],
            "wins_a": surface_entry["wins_a"],
            "wins_b": surface_entry["wins_b"],
        }
    case_id = f"match-{surface.lower()}-{_slug(player_a)}-{_slug(player_b)}"
    return {"id": case_id, "type": "match", "question": question, "expected": expected}


def generate_new_cases(conn):
    """Generate the 19 new cases, skipping ambiguous or missing players with a warning."""
    new_cases = []
    for player in NEW_PLAYERS:
        elo = get_player_elo(conn, player)
        if elo is None or (isinstance(elo, dict) and elo.get("ambiguous")):
            print(f"WARNING: skipping player, ambiguous or missing: {player} -> {elo}")
            continue
        new_cases.append({"id": f"elo-{_slug(player)}", "type": "elo", "player": player, "expected": extract_elo_expected(elo)})
        print(f"elo {player}: overall {elo['overall_elo']}, matches {elo['matches_played']}")
        form = get_player_form(conn, player)
        if form is None or (isinstance(form, dict) and form.get("ambiguous")):
            print(f"WARNING: skipping form case, ambiguous or missing: {player}")
            continue
        new_cases.append({"id": f"form-{_slug(player)}", "type": "form", "player": player, "expected": extract_form_expected(form)})
        record = get_player_surface_record(conn, player)
        if record is None or (isinstance(record, dict) and record.get("ambiguous")):
            print(f"WARNING: skipping surface case, ambiguous or missing: {player}")
            continue
        new_cases.append({"id": f"surface-{_slug(player)}", "type": "surface", "player": player, "expected": extract_surface_expected(record)})
    for player_a, player_b, surface in NEW_MATCHES:
        case = build_match_case(conn, player_a, player_b, surface)
        if case is None:
            continue
        new_cases.append(case)
        exp = case["expected"]
        print(f"match {case['question']}: favorite {exp['favorite']}, calibrated {exp['prob_a_calibrated']} vs {exp['prob_b_calibrated']}, h2h total {exp['h2h_total']}")
    return new_cases


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate expanded eval cases, 30 canonical plus 70 new, to a review file, not canonical.")
    parser.add_argument("--cases", default="eval/cases.jsonl")
    parser.add_argument("--output", default="eval/cases_expanded.jsonl")
    args = parser.parse_args(argv)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a")
        return 1
    import psycopg

    canonical_cases = []
    with open(args.cases, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                canonical_cases.append(json.loads(line))
    canonical_ids = {c["id"] for c in canonical_cases}
    with psycopg.connect(database_url) as conn:
        new_cases = generate_new_cases(conn)
    new_ids = [c["id"] for c in new_cases]
    collisions = [cid for cid in new_ids if cid in canonical_ids]
    if collisions:
        print(f"ERROR: new case ids collide with canonical ids, aborting, nothing written: {collisions}")
        return 1
    if len(new_ids) != len(set(new_ids)):
        print("ERROR: duplicate new case ids, aborting, nothing written")
        return 1
    all_cases = canonical_cases + new_cases
    with open(args.output, "w", encoding="utf-8") as handle:
        for case in all_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Canonical cases preserved: {len(canonical_cases)}")
    print(f"New cases added: {len(new_cases)}")
    print(f"Total cases in review file: {len(all_cases)}")
    print(f"Review file: {args.output} (canonical file NOT changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
