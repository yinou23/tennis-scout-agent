"""Games-won percentage form windows, pre-match only.

This is the games layer, built on atp_match_games, not on win/loss.
For each player, before a match, it computes:
- last10 games pct: total games won / total games played in the
  player's previous 10 complete matches.
- last52 games pct: same ratio, for complete matches in the
  previous 365 days.

Why total games, not the average of per-match percentages?
A 6-0 6-0 match has 12 games, a 7-6 6-7 7-6 match has 39 games.
Averaging their percentages would give both matches equal weight,
even though the long match carries far more game-level evidence.
Summing games first gives each game equal weight.

Incomplete matches (retirements, etc.) are skipped entirely for
games windows: their partial game counts would bias the ratio.
If a player has no complete previous match in a window, that
window's percentage is None, not 0, not 50%.
"""

from collections import deque
from datetime import date


class PlayerGamesForm:
    def __init__(self):
        # Each entry: (tourney_date, games_won, games_played)
        self.last10 = deque(maxlen=10)
        self.recent = []

    def snapshot(self, current_date):
        """Return (last10_pct, last10_matches, last10_games,
        last52_pct, last52_matches, last52_games)."""
        w10 = sum(e[1] for e in self.last10)
        g10 = sum(e[2] for e in self.last10)
        pct10 = (w10 / g10) if g10 else None
        if current_date is None:
            return pct10, len(self.last10), g10, None, 0, 0
        recent = [
            e for e in self.recent
            if e[0] is not None and 0 <= (current_date - e[0]).days <= 365
        ]
        w52 = sum(e[1] for e in recent)
        g52 = sum(e[2] for e in recent)
        pct52 = (w52 / g52) if g52 else None
        return pct10, len(self.last10), g10, pct52, len(recent), g52

    def record(self, tourney_date, games_won, games_played, is_complete):
        if not is_complete or games_played is None or games_played == 0:
            return
        entry = (tourney_date, games_won, games_played)
        self.last10.append(entry)
        self.recent.append(entry)
