# Weekly sample scouting report: Jannik Sinner vs Carlos Alcaraz (Hard)

Sample report, historical data only, not live.

## Metadata

- Sample type: weekly sample, templated, no LLM writer
- Matchup: Jannik Sinner vs Carlos Alcaraz
- Surface: Hard
- Data as of: 2026-05-25, not live
- Verification: ok=True, unexpected_numbers=[], missing_required=None, tag_error=None
- Favorite: Jannik Sinner
- Final probability (calibrated): Jannik Sinner 0.6378, Carlos Alcaraz 0.3622
- Blended probability (before calibration): Jannik Sinner 0.6698, Carlos Alcaraz 0.3302
- LLM calls: 0
- LLM cost: $0.00
- Latency (report build): 426.8 ms
- Tool calls: 8

## Scouting report

Scouting report: Jannik Sinner vs Carlos Alcaraz (Hard)
Data as of 2026-05-25, not live.
Final probability (calibrated): Jannik Sinner 0.6378, Carlos Alcaraz 0.3622 [Surface blend: Hard, prior=100] [Calibration: a=0.8]
Blended probability (before calibration): Jannik Sinner 0.6698, Carlos Alcaraz 0.3302. Favorite is Jannik Sinner.
Elo comparison:
Jannik Sinner: overall 2283.1, Hard Elo 2236.0 (308 matches), Hard blended 2247.6, 457 matches total.
Carlos Alcaraz: overall 2176.7, Hard Elo 2099.1 (203 matches), Hard blended 2124.7, 371 matches total.
Surface record:
Jannik Sinner Hard: 251 wins, 57 losses, 308 matches, win rate 0.8149; career 457 matches, 363 wins, 94 losses.
Carlos Alcaraz Hard: 159 wins, 44 losses, 203 matches, win rate 0.7833; career 371 matches, 302 wins, 69 losses.
Recent form (descriptors only, not probability inputs):
Jannik Sinner: last 10 win rate 0.9000 (9 wins in 10 matches), last 52 weeks win rate 0.9121 (83 wins in 91 matches); games last 10 0.6273 (138 games won, 220 total), games last 52 weeks 0.6350 (1298 games won, 2044 total); last mid-match retirement 2025-10-01, 236 days before data as of, 55 matches since; retirements last 10: 0, last 52 weeks: 2, total: 7.
Carlos Alcaraz: last 10 win rate 0.6000 (6 wins in 10 matches), last 52 weeks win rate 0.8873 (63 wins in 71 matches); games last 10 0.5742 (120 games won, 209 total), games last 52 weeks 0.5957 (1099 games won, 1845 total); last mid-match retirement 2024-02-19, 826 days before data as of, 164 matches since; retirements last 10: 0, last 52 weeks: 0, total: 3.
Head-to-head (descriptor only, not a probability input):
17 meetings, Jannik Sinner 7 wins, Carlos Alcaraz 10 wins; Hard 10 meetings, Jannik Sinner 3 wins, Carlos Alcaraz 7 wins.
Sources and tools: player Elo, recent form, head-to-head, surface record, match probability, all from database queries, not model text. Form, games, retirement, and head-to-head are descriptors only.

## Sources

| Source | Tables used | Data as of |
| --- | --- | --- |
| Jeff Sackmann ATP data, via local PostgreSQL import | atp_players, atp_matches, atp_elo_current, atp_surface_elo_history, atp_form_history, atp_match_games | 2026-05-25, not live |

## Tool trace

- get_match_probability
- get_player_elo (player_a)
- get_player_elo (player_b)
- get_player_form (player_a)
- get_player_form (player_b)
- get_player_surface_record (player_a)
- get_player_surface_record (player_b)
- get_head_to_head

## Cost and latency

- LLM calls: 0, LLM cost: $0.00, because this sample is templated, no LLM writer was used.
- Report build latency: 426.8 ms, measured in this run, not a target or a guarantee.
- Tool calls: 8, all database queries, not model text.

## Coverage note

Historical only. This sample uses matches imported to 2026-05-25, and does not include matches, scores, draws, or news after that date. It does not predict a specific upcoming real match, and it does not include live scores or in-progress points. Form is anchored at the data-as-of date, not at a hypothetical next match. Tourney_date in the source data is tournament start, not exact match day, so the 52-week form window is accurate to within a tournament length. Serve statistics are sparse before the 1990s and are not used as a probability input in this report. Rankings in the import go to 2026-06-08, but the report's data-as-of is 2026-05-25, the maximum match date, and that is the date that governs the report.

## Live section (separate from historical report)

This section is a point-in-time snapshot, not a live feed, and it does not change the historical report above.

- Fetched at: 2026-11-03T02:28:32.442167Z
- Source: Live Tennis API, tier: FREE
- Status requested: live
- Count in snapshot: 15
- Showing first 5 of 15 fixtures in snapshot order.
- Historical data as of: 2026-05-25 (unchanged by this section)
- Snapshot note: Point-in-time snapshot, not a live feed. Historical TSA data remains as of 2026-05-25 and is not updated by this file. Provider player and match ids do not match Sackmann ids, so no join to Sackmann players, Elo, form, or head-to-head is made here.

- Ayumi Miyamoto vs Kurea Hayasaka | W35 Hamamatsu | tour: itf | status: live | surface: hard | scheduled: 2026-11-03T01:10:00Z | Score: Sets 0-1, Games 3-6, 1-4, Points 15-40, Server: p2
- Sera Nishimoto vs Kokono Sakaguchi | W35 Hamamatsu | tour: itf | status: live | surface: hard | scheduled: 2026-11-03T01:10:00Z | Score: Sets 1-0, Games 6-3, 3-4, Points 40-15, Server: p2
- Nanari Katsumi vs Michika Ozeki | W35 Hamamatsu | tour: itf | status: live | surface: hard | scheduled: 2026-11-03T01:10:00Z | Score: Sets 0-1, Games 3-6, 4-3, Points 15-30, Server: p2
- Juan Carlos Prado Angelo vs Pedro Boscardin Dias | Costa do Sauipe | tour: challenger | status: live | surface: clay | scheduled: 2026-11-03T01:45:00Z | Score: Sets 1-0, Games 6-1, 0-1, Points 40-15, Server: p1
- Xiaotong Wang vs Ho Ching Wu | ITF W15 Maanshan 8 Women | tour: itf | status: live | surface: hard | scheduled: 2026-11-03T02:00:00Z | Score: Sets 0-0, Games 2-0, Points 30-30, Server: p1
