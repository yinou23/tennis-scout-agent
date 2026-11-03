"""Week 7 Step 4, minimal: add a separate live section to a weekly sample.

This is the smallest live integration that was agreed after the
Step 3 live call, and after the user question about whether live
scores help the database or the core answers. They do not. So this
script does the minimum, and nothing more:

- It does NOT call the Live Tennis API. It reads a snapshot file
  already written by scripts/live_fixtures.py, usually
  live/fixtures.json. Cost: 0 API calls, 0 quota.
- It does NOT change the historical sample. It appends a new
  section after the historical markdown, verbatim, with no edit
  to the historical text, its probabilities, its Elo, its form,
  or its coverage note.
- It does NOT compute Elo, form, games, head-to-head, or a match
  probability from live data. The live section contains no Elo
  and no probability, calibrated or otherwise.
- It does NOT join provider ids to Sackmann ids. Player and match
  ids in the snapshot are provider ids, and they are not used
  here except as display context, if at all. The section shows
  player names, tournament, tour, status, and score only.
- It does NOT claim a live feed. The section states its own
  fetched_at, source, tier, and count, and states that the
  historical data as of 2026-05-25 is unchanged by this section.
- It does NOT relabel tours. The tour field from the snapshot
  is shown as fetched, so an ITF W35 match reads as ITF, never
  as ATP main tour.
- Score rendering, from the provider score object verified in
  Step 3: sets is per player, for example [1, 0] means p1 leads
  one set to nil. games is player-major, not set-major: for
  example [[6, 3], [4, 4]] means 6-4 in the first set and
  3-4 in the second, because the first list is p1 across
  sets and the second list is p2 across sets.
  points is per player in the current game, server is 1 for p1
  and 2 for p2. A missing
  score reads as No score yet, never as 0-0. A score with
  games null reads as games withheld, never as 0-0.

Usage, from the project root ~/TSA, after a snapshot exists:
    python scripts/live_section.py
    python scripts/live_section.py --sample samples/weekly_sample.md --snapshot live/fixtures.json --output samples/weekly_sample_with_live.md --max-fixtures 5

The default output is a new file, samples/weekly_sample_with_live.md.
The original samples/weekly_sample.md is never overwritten by
this script.
"""

import argparse
import json
import sys
from pathlib import Path

HISTORICAL_DATA_AS_OF = "2026-05-25"
DEFAULT_MAX_FIXTURES = 5


def _player_or_unknown(name):
    """Return a name, or Unknown if absent. Never invent a name."""
    return name if isinstance(name, str) and name else "Unknown"


def _format_games(games):
    """Format player-major games, or None if the shape is not usable.

    Expected shape: [[p1 set1, p1 set2], [p2 set1, p2 set2]].
    Returns a string like 6-4, 3-4, or None.
    """
    if not isinstance(games, list) or len(games) != 2:
        return None
    p1_games, p2_games = games
    if not isinstance(p1_games, list) or not isinstance(p2_games, list):
        return None
    if len(p1_games) != len(p2_games):
        return None
    if not p1_games:
        return None
    try:
        return ", ".join("%s-%s" % (a, b) for a, b in zip(p1_games, p2_games))
    except Exception:
        return None


def format_score(score):
    """Format a provider score object for display, or No score yet.

    A missing score, and a score with no sets, games, or points,
    both read as No score yet. A present score with games null
    reads as games withheld, never as 0-0.
    """
    if not isinstance(score, dict):
        return "No score yet"
    sets = score.get("sets")
    games = score.get("games")
    points = score.get("points")
    server = score.get("server")
    is_tiebreak = score.get("is_tiebreak")
    if sets is None and games is None and points is None:
        return "No score yet"
    parts = []
    if isinstance(sets, list) and len(sets) == 2:
        parts.append("Sets %s-%s" % (sets[0], sets[1]))
    games_text = _format_games(games)
    if games_text is not None:
        parts.append("Games %s" % games_text)
    elif games is None and sets is not None:
        parts.append("Games withheld")
    if isinstance(points, list) and len(points) == 2:
        parts.append("Points %s-%s" % (points[0], points[1]))
    if server == 1:
        parts.append("Server: p1")
    elif server == 2:
        parts.append("Server: p2")
    if is_tiebreak is True:
        parts.append("Tiebreak")
    if not parts:
        return "No score yet"
    return ", ".join(parts)


def format_fixture(fixture):
    """Format one fixture dict as a single markdown bullet line."""
    if not isinstance(fixture, dict):
        return "- Unknown fixture"
    p1 = _player_or_unknown(fixture.get("player_p1"))
    p2 = _player_or_unknown(fixture.get("player_p2"))
    tournament = fixture.get("tournament") or "Unknown tournament"
    tour = fixture.get("tour") or "unknown tour"
    status = fixture.get("status") or "unknown status"
    surface = fixture.get("surface")
    scheduled_time = fixture.get("scheduled_time")
    line = "- %s vs %s | %s | tour: %s | status: %s" % (p1, p2, tournament, tour, status)
    if isinstance(surface, str) and surface:
        line += " | surface: %s" % surface
    if isinstance(scheduled_time, str) and scheduled_time:
        line += " | scheduled: %s" % scheduled_time
    line += " | Score: %s" % format_score(fixture.get("score"))
    return line


def build_live_section(snapshot, max_fixtures=DEFAULT_MAX_FIXTURES):
    """Build the live section markdown from a snapshot dict.

    No Elo, no probability, no calibration, no Sackmann join.
    Missing fetched_at reads as Unknown, never as a guessed date.
    """
    if not isinstance(snapshot, dict):
        raise ValueError("Snapshot must be a dict.")
    if not isinstance(max_fixtures, int) or max_fixtures < 1:
        raise ValueError("max_fixtures must be an integer of 1 or more.")
    fixtures = snapshot.get("fixtures")
    if not isinstance(fixtures, list):
        fixtures = []
    fetched_at = snapshot.get("fetched_at")
    if not isinstance(fetched_at, str) or not fetched_at:
        fetched_at = "Unknown"
    source = snapshot.get("source") or "Live Tennis API"
    tier = snapshot.get("tier") or "FREE"
    status_requested = snapshot.get("status_requested") or "unknown"
    historical_as_of = snapshot.get("historical_data_as_of") or HISTORICAL_DATA_AS_OF
    snapshot_note = snapshot.get("snapshot_note")
    total = len(fixtures)
    shown = fixtures[:max_fixtures]
    lines = []
    lines.append("## Live section (separate from historical report)")
    lines.append("")
    lines.append(
        "This section is a point-in-time snapshot, not a live feed, "
        "and it does not change the historical report above."
    )
    lines.append("")
    lines.append("- Fetched at: %s" % fetched_at)
    lines.append("- Source: %s, tier: %s" % (source, tier))
    lines.append("- Status requested: %s" % status_requested)
    lines.append("- Count in snapshot: %s" % total)
    if total > max_fixtures:
        lines.append("- Showing first %s of %s fixtures in snapshot order." % (max_fixtures, total))
    lines.append("- Historical data as of: %s (unchanged by this section)" % historical_as_of)
    if isinstance(snapshot_note, str) and snapshot_note:
        lines.append("- Snapshot note: %s" % snapshot_note)
    lines.append("")
    if not shown:
        lines.append("No live matches at fetch time.")
    else:
        for fixture in shown:
            lines.append(format_fixture(fixture))
    lines.append("")
    return "\n".join(lines)


def append_live_section(sample_markdown, snapshot, max_fixtures=DEFAULT_MAX_FIXTURES):
    """Return sample markdown plus the live section, historical text verbatim."""
    section = build_live_section(snapshot, max_fixtures=max_fixtures)
    base = sample_markdown if sample_markdown.endswith("\n") else sample_markdown + "\n"
    return base + "\n" + section


def load_snapshot(path):
    """Load a snapshot JSON file, or raise ValueError on a bad file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("Snapshot file not found: %s" % path) from exc
    except (OSError, ValueError) as exc:
        raise ValueError("Snapshot file could not be read as JSON: %s" % exc) from exc
    if not isinstance(data, dict):
        raise ValueError("Snapshot JSON must be an object.")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Append a minimal live section to a weekly sample, no API call."
    )
    parser.add_argument("--sample", default="samples/weekly_sample.md")
    parser.add_argument("--snapshot", default="live/fixtures.json")
    parser.add_argument("--output", default="samples/weekly_sample_with_live.md")
    parser.add_argument("--max-fixtures", type=int, default=DEFAULT_MAX_FIXTURES)
    args = parser.parse_args(argv)
    try:
        sample_markdown = Path(args.sample).read_text(encoding="utf-8")
    except FileNotFoundError:
        print("Sample file not found: %s" % args.sample, file=sys.stderr)
        return 1
    except OSError as exc:
        print("Sample file could not be read: %s" % exc, file=sys.stderr)
        return 1
    try:
        snapshot = load_snapshot(args.snapshot)
        combined = append_live_section(sample_markdown, snapshot, max_fixtures=args.max_fixtures)
    except ValueError as exc:
        print("Stopped: %s" % exc, file=sys.stderr)
        return 1
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(combined, encoding="utf-8")
    print(build_live_section(snapshot, max_fixtures=args.max_fixtures))
    print("Wrote %s" % output_path)
    print("Original sample file was not overwritten: %s" % args.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
