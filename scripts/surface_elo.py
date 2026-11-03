"""Surface Elo blending, pure functions, no database I/O.

Design, fixed before seeing backtest results:
- Every player has an overall Elo (the Week 2 classic Elo) and a
  separate Elo per surface (Hard, Clay, Grass, Carpet), each
  starting at 1500 and updated with the same K=32, but only on
  matches played on that surface.
- A surface Elo on its own is noisy when the player has played
  few matches on that surface, so the prediction does NOT use
  surface Elo alone. It uses a blend toward the overall Elo:
      blended = w * surface_elo + (1 - w) * overall_elo
      w = n_surface / (n_surface + prior)
  where n_surface is the number of previous matches on that
  surface, and prior is the shrinkage strength to test.
- If n_surface = 0, w = 0, so blended = overall Elo. No harm
  for a surface debut.
- If n_surface is large, w approaches 1, so blended approaches
  surface Elo.
- This is shrinkage toward a good prior (overall Elo), not
  toward 1500. That is the key difference from the reliability
  shrinkage that failed in Week 2, which shrank toward 1500.
- Matches with a missing surface (NULL in the data, 2,990
  matches) do not update any surface Elo, and their prediction
  uses overall Elo only.
"""

INITIAL_ELO = 1500.0


def blended_elo(surface_elo, overall_elo, n_surface, prior):
    """Blend surface Elo toward overall Elo.

    prior = None means no blending, use overall Elo only.
    prior = 0 means use surface Elo alone when n_surface > 0,
    otherwise overall Elo.
    """
    surface_elo = float(surface_elo)
    overall_elo = float(overall_elo)
    if prior is None:
        return overall_elo
    if n_surface is None or n_surface <= 0:
        return overall_elo
    if prior == 0:
        return surface_elo
    w = n_surface / (n_surface + prior)
    return w * surface_elo + (1.0 - w) * overall_elo


def win_prob(elo_a, elo_b):
    return 1.0 / (1.0 + 10.0 ** (-(float(elo_a) - float(elo_b)) / 400.0))
