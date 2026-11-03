"""Parse a Sackmann match score into games won.

Score format in atp_matches.score: winner's games first in every set,
sets separated by spaces, tiebreak points in parentheses, e.g.
"6-4 6-3", "6-7(4) 7-6(5) 6-3", "6-4 3-6 7-5".

Returns (winner_games, loser_games, is_complete):
- is_complete is True only if every set token parses and the match
  has no retirement / walkover / default marker.
- If the score is missing or a set token does not parse, both game
  counts are None and is_complete is False.
- If the match ended by retirement, the games counted so far are
  returned, but is_complete is False, so downstream games-% code
  can exclude incomplete matches instead of treating a partial
  score as a full match.
"""

import re

_SET_RE = re.compile(r"^(\d+)-(\d+)(?:\(\d+\))?$")
_INCOMPLETE_MARKERS = {"RET", "W/O", "WO", "DEF", "ABN", "RET."}


def parse_score(score):
    if score is None:
        return None, None, False
    score = str(score).strip()
    if score == "":
        return None, None, False
    winner_games = 0
    loser_games = 0
    for token in score.split():
        if token.upper() in _INCOMPLETE_MARKERS:
            if winner_games == 0 and loser_games == 0:
                return None, None, False
            return winner_games, loser_games, False
        m = _SET_RE.match(token)
        if not m:
            return None, None, False
        winner_games += int(m.group(1))
        loser_games += int(m.group(2))
    if winner_games == 0 and loser_games == 0:
        return None, None, False
    return winner_games, loser_games, True


def winner_games_pct(score):
    """Winner's share of games, or None if not a complete parsed score."""
    w, l, complete = parse_score(score)
    if not complete or w is None:
        return None
    total = w + l
    if total == 0:
        return None
    return w / total
