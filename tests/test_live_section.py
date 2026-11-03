"""Unit tests for Week 7 Step 4 minimal live section.

Run from the project root ~/TSA, after moving the staging files to
their canonical names:
    mv ~/Downloads/live_section_w7step4_2026110205.py scripts/live_section.py
    mv ~/Downloads/test_live_section_w7step4_2026110205.py tests/test_live_section.py
    python -m pytest tests/test_live_section.py -v

No network, no API key, no database. The live section is built
from a snapshot dict, never from an API call, and never from
the historical database.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import live_section


def _fixture_live():
    return {
        "provider_match_id": 21131,
        "player_p1": "Ayumi Miyamoto",
        "player_p2": "Kurea Hayasaka",
        "tournament": "W35 Hamamatsu",
        "tournament_id": "999",
        "tour": "itf",
        "surface": "hard",
        "status": "live",
        "round": "Quarter-finals",
        "round_code": "QF",
        "scheduled_time": "2026-11-02T15:00:00Z",
        "event_status": None,
        "score": {
            "sets": [1, 0],
            "games": [[6, 3], [4, 4]],
            "points": ["0", "0"],
            "server": 1,
            "is_tiebreak": False,
        },
    }


def _fixture_upcoming():
    return {
        "provider_match_id": 21132,
        "player_p1": "Jannik Sinner",
        "player_p2": "Carlos Alcaraz",
        "tournament": "Example Open",
        "tour": "atp",
        "surface": "hard",
        "status": "upcoming",
        "scheduled_time": "2026-11-03T15:00:00Z",
        "score": None,
    }


def _snapshot(fixtures):
    return {
        "fetched_at": "2026-11-02T21:28:00Z",
        "source": "Live Tennis API",
        "tier": "FREE",
        "endpoint": "/matches?status=live&limit=200&offset=0",
        "status_requested": "live",
        "historical_data_as_of": "2026-05-25",
        "snapshot_note": "Point-in-time snapshot, not a live feed.",
        "count": len(fixtures),
        "fixtures": fixtures,
    }


def test_format_score_live():
    text = live_section.format_score(_fixture_live()["score"])
    assert text == "Sets 1-0, Games 6-4, 3-4, Points 0-0, Server: p1"


def test_format_score_no_score_is_not_zero_zero():
    assert live_section.format_score(None) == "No score yet"
    assert "0-0" not in live_section.format_score(None)


def test_format_score_games_withheld_is_not_zero_zero():
    text = live_section.format_score(
        {"sets": [0, 0], "games": None, "points": None, "server": None, "is_tiebreak": False}
    )
    assert "Games withheld" in text
    # Sets 0-0 is legitimate here, because sets is [0, 0] in the input.
    # The withheld part is games, so the check must be that games
    # was not rendered as 0-0, not that the string 0-0 never appears.
    assert "Games 0-0" not in text


def test_build_live_section_includes_metadata_and_fixtures():
    section = live_section.build_live_section(_snapshot([_fixture_live(), _fixture_upcoming()]))
    assert "Live section (separate from historical report)" in section
    assert "2026-11-02T21:28:00Z" in section
    assert "Live Tennis API" in section
    assert "FREE" in section
    assert "Count in snapshot: 2" in section
    assert "Ayumi Miyamoto vs Kurea Hayasaka" in section
    assert "W35 Hamamatsu" in section
    assert "Sets 1-0, Games 6-4, 3-4" in section
    assert "Jannik Sinner vs Carlos Alcaraz" in section
    assert "No score yet" in section
    assert "Historical data as of: 2026-05-25 (unchanged by this section)" in section
    assert "not a live feed" in section


def test_build_live_section_does_not_claim_elo_or_probability():
    section = live_section.build_live_section(_snapshot([_fixture_live()]))
    assert "Elo" not in section
    assert "probability" not in section.lower()
    assert "calibrated" not in section
    assert "blended" not in section


def test_build_live_section_shows_tour_so_itf_is_not_mislabeled():
    section = live_section.build_live_section(_snapshot([_fixture_live()]))
    assert "tour: itf" in section
    assert "W35 Hamamatsu" in section


def test_build_live_section_truncates_to_max_and_notes_total():
    fixtures = [_fixture_live() for _ in range(7)]
    section = live_section.build_live_section(_snapshot(fixtures), max_fixtures=5)
    assert "Count in snapshot: 7" in section
    assert "Showing first 5 of 7 fixtures" in section
    assert section.count("Ayumi Miyamoto vs Kurea Hayasaka") == 5


def test_build_live_section_empty_snapshot():
    section = live_section.build_live_section(_snapshot([]))
    assert "Count in snapshot: 0" in section
    assert "No live matches at fetch time." in section


def test_build_live_section_missing_fetched_at_is_unknown():
    snapshot = _snapshot([_fixture_live()])
    del snapshot["fetched_at"]
    section = live_section.build_live_section(snapshot)
    assert "Fetched at: Unknown" in section


def test_build_live_section_missing_player_name_is_unknown():
    fixture = _fixture_live()
    fixture["player_p1"] = None
    section = live_section.build_live_section(_snapshot([fixture]))
    assert "Unknown vs Kurea Hayasaka" in section


def test_append_live_section_preserves_historical_text_verbatim():
    historical = "# Weekly sample scouting report: Jannik Sinner vs Carlos Alcaraz (Hard)\n\nSample report, historical data only, not live.\n"
    combined = live_section.append_live_section(historical, _snapshot([_fixture_live()]))
    assert combined.startswith(historical)
    assert "Live section (separate from historical report)" in combined
    assert historical in combined
