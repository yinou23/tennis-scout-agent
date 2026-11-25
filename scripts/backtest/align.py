"""Sackmann match <-> odds row alignment.

Rules, in order (a match is only counted as matched when exactly
one odds row survives all filters; anything else is reported
unmatched with a reason, never forced to a nearest row):

1. Player pair: unordered (surname, first-initial) key from
   names.pair_key, Sackmann winner/loser vs odds player_a/player_b.
2. Date window: the odds date must fall in
   [tourney_date - 4 days, tourney_date + 13 days]. Sackmann
   tourney_date is the Monday of the tournament week, but main
   draws can start on the Sunday before (Auckland 2024: matches
   2024-01-07, tourney_date 2024-01-08), and team events can
   start on the Friday before (United Cup 2024: matches
   2023-12-29, tourney_date 2024-01-01). The -4 day start and
   the adjacent-year odds files (see run_baseline._load_odds)
   cover both cases. A wider window raises the chance that the
   same pair meets in two consecutive weeks; those cases land
   in "ambiguous" and are reported unmatched, never forced.
3. Round (tennis-data only, when the odds row has one): the
   Sackmann round must equal it after normalisation ("R32" ==
   "2nd Round" is NOT assumed; only exact normalised equality
   counts, otherwise the row is kept only if it is the unique
   survivor after the date filter).
4. Tournament name: used only as a tie-breaker when more than
   one row survives (same pair twice in one event, e.g. round
   robin then final). Token overlap >= 1 content token, else
   the pair stays ambiguous.

Unmatched reasons recorded per match:
    no_pair            no odds row has this player pair at all
    no_date_window     pair exists, none inside the date window
    ambiguous          more than one row survives the filters
    no_closing_odds    matched row has no odds for this bookmaker
                       (reported per bookmaker in the run output)

The headline match rate is computed over Sackmann main-tour
matches (tourney_level in A/F/G/M) per year. Davis Cup (D) and
Olympics (O) are reported separately: odds providers do not
cover them the same way, and mixing them into the headline rate
would understate the rate on the matches the model is actually
scored on.
"""

import datetime as _dt

from names import normalize_name, pair_key

MAIN_TOUR_LEVELS = {"A", "F", "G", "M"}
OTHER_LEVELS = {"D", "O"}

_STOP_TOKENS = {
    "open", "masters", "championships", "championship", "cup", "classic",
    "tournament", "the", "de", "des", "du", "la", "le", "der", "den",
}


def _tokens(text):
    return {t for t in normalize_name(text).split() if t not in _STOP_TOKENS}


def align_matches(matches, odds_rows):
    """Align Sackmann matches to odds rows.

    matches: iterable of dicts with keys tourney_id, match_num,
        tourney_date (date), tourney_name, round, surface,
        winner_name, loser_name, tourney_level.
    odds_rows: iterable of OddsRow.
    Returns (matched, unmatched) where matched is a list of
    (match, odds_row) and unmatched is a list of
    (match, reason).
    """
    by_pair = {}
    for row in odds_rows:
        by_pair.setdefault(pair_key(row.player_a, row.player_b), []).append(row)

    matched, unmatched = [], []
    for m in matches:
        key = pair_key(m["winner_name"], m["loser_name"])
        cands = by_pair.get(key, [])
        if not cands:
            unmatched.append((m, "no_pair"))
            continue
        start = m["tourney_date"] - _dt.timedelta(days=4)
        end = m["tourney_date"] + _dt.timedelta(days=13)
        in_window = []
        date_missing = []
        for row in cands:
            if row.date is None:
                date_missing.append(row)
            elif start <= row.date <= end:
                in_window.append(row)
        pool = in_window if in_window else date_missing
        if not pool:
            unmatched.append((m, "no_date_window"))
            continue
        # Round filter, only when the odds rows carry a round.
        with_round = [r for r in pool if r.round]
        if with_round and len(pool) > 1:
            same_round = [r for r in with_round
                          if normalize_name(r.round) == normalize_name(m["round"])]
            if same_round:
                pool = same_round
        if len(pool) == 1:
            matched.append((m, pool[0]))
            continue
        # Tournament-name tie-breaker.
        m_tokens = _tokens(m["tourney_name"])
        scored = []
        for row in pool:
            overlap = len(m_tokens & _tokens(row.tournament))
            scored.append((overlap, row))
        scored.sort(key=lambda t: -t[0])
        if scored and scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            matched.append((m, scored[0][1]))
        else:
            unmatched.append((m, "ambiguous"))
    return matched, unmatched


def _is_team_exhibition(tourney_name):
    """United Cup / Laver Cup / Next Gen Finals: team or exhibition
    events that odds providers do not publish (tennis-data.co.uk
    has no rows for them). They are in Sackmann's main tour, so
    they must be reported separately, not counted against the
    headline match rate."""
    norm = normalize_name(tourney_name)
    return ("united cup" in norm or "laver cup" in norm or "next gen" in norm
            or "nextgen" in norm)  # Sackmann spells it both ways:
    # "Next Gen Finals" (2023-2024) and "NextGen Finals" (2017-2022).


def match_rate_report(matches, matched, unmatched):
    """Per-level match rates and unmatched-reason counts."""
    from collections import Counter
    matched_ids = {(m["tourney_id"], m["match_num"]) for m, _ in matched}
    report = {"by_level": {}, "unmatched_reasons": Counter(), "n_matches": len(matches)}
    for level in sorted({m["tourney_level"] for m in matches}):
        subset = [m for m in matches if m["tourney_level"] == level]
        hit = sum(1 for m in subset if (m["tourney_id"], m["match_num"]) in matched_ids)
        report["by_level"][level] = {
            "n": len(subset),
            "matched": hit,
            "rate": hit / len(subset) if subset else None,
        }
    for _m, reason in unmatched:
        report["unmatched_reasons"][reason] += 1
    main = [m for m in matches if m["tourney_level"] in MAIN_TOUR_LEVELS]
    hit = sum(1 for m in main if (m["tourney_id"], m["match_num"]) in matched_ids)
    report["main_tour"] = {
        "n": len(main),
        "matched": hit,
        "rate": hit / len(main) if main else None,
    }
    team = [m for m in main if _is_team_exhibition(m["tourney_name"])]
    team_hit = sum(1 for m in team if (m["tourney_id"], m["match_num"]) in matched_ids)
    rate_matches = [m for m in main if not _is_team_exhibition(m["tourney_name"])]
    rate_hit = sum(1 for m in rate_matches if (m["tourney_id"], m["match_num"]) in matched_ids)
    report["main_tour_excl_team_exhibition"] = {
        "n": len(rate_matches),
        "matched": rate_hit,
        "rate": rate_hit / len(rate_matches) if rate_matches else None,
        "team_exhibition_n": len(team),
        "team_exhibition_matched": team_hit,
    }
    return report
