"""Classic Elo for tennis, win/loss only.

This is v1, deliberately simple:
- Every player starts at 1500.0.
- Only the match result matters (win = 1, loss = 0).
- No surface adjustment, no games-won percentage, no expiry,
  no reliability weighting yet. Those are later layers, added
  only after this classic version runs end to end and is backtested.

Functions here are pure (no database, no I/O) so they can be
unit-tested directly.
"""

INITIAL_ELO = 1500.0
DEFAULT_K = 32.0


def expected_win_prob(rating_a, rating_b):
    """Probability that player A beats player B, from Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(winner_rating, loser_rating, k=DEFAULT_K):
    """Update ratings after one match.

    Returns (new_winner_rating, new_loser_rating, expected_winner_prob).
    The total of the two ratings is unchanged: what the winner gains,
    the loser loses.
    """
    expected = expected_win_prob(winner_rating, loser_rating)
    delta = k * (1.0 - expected)
    return winner_rating + delta, loser_rating - delta, expected
