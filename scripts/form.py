"""Form windows for tennis, pre-match only.

Two windows, both computed from matches BEFORE the current match:
- last10: win rate in the player's previous 10 main-tour matches,
  regardless of how long ago they were.
- last52: win rate in matches in the previous 365 days.

If a player has no previous match in a window, that window's
win rate is None (not 0, not 50%), so downstream code can tell
"no data" apart from "lost every match".

Limitation, stated openly: atp_matches.tourney_date is the
tournament start date, not the exact match day. The 52-week
window is therefore accurate to within a tournament's length,
not to the day. That is acceptable for a form signal, and it is
why form is reported alongside Elo, not folded into Elo itself.
"""

from collections import deque
from datetime import date


class PlayerForm:
    def __init__(self):
        self.last10 = deque(maxlen=10)
        # List of (tourney_date, won) in chronological order.
        self.recent = []

    def snapshot(self, current_date):
        """Return (last10_pct, last10_n, last52_pct, last52_n)."""
        n10 = len(self.last10)
        pct10 = (sum(self.last10) / n10) if n10 else None
        if current_date is None:
            return pct10, n10, None, 0
        recent = [
            won
            for d, won in self.recent
            if d is not None and 0 <= (current_date - d).days <= 365
        ]
        n52 = len(recent)
        pct52 = (sum(recent) / n52) if n52 else None
        return pct10, n10, pct52, n52

    def record(self, tourney_date, won):
        self.last10.append(1 if won else 0)
        self.recent.append((tourney_date, 1 if won else 0))
