-- Build compact ranking summary tables from atp_rankings (3.4M weekly rows).
-- Replaces the full weekly table for all ranking questions the product answers.
--   atp_year_end_rankings: one row per player per year (last ranking date of the year).
--   atp_player_ranking_summary: one row per player (career-high + latest ranking).
-- After verifying these tables, atp_rankings can be dropped to reclaim ~300 MB.
-- Run: PGPASSWORD=$TSA_PASSWORD psql -h 127.0.0.1 -U tsa -d tsa -f scripts/build_ranking_summary.sql

DROP TABLE IF EXISTS atp_year_end_rankings;
CREATE TABLE atp_year_end_rankings AS
SELECT DISTINCT ON (player_id, EXTRACT(YEAR FROM ranking_date))
    player_id,
    EXTRACT(YEAR FROM ranking_date)::int AS year,
    ranking,
    points,
    ranking_date
FROM atp_rankings
ORDER BY player_id, EXTRACT(YEAR FROM ranking_date), ranking_date DESC;

ALTER TABLE atp_year_end_rankings ADD PRIMARY KEY (player_id, year);
CREATE INDEX idx_year_end_rank ON atp_year_end_rankings (year, ranking);

DROP TABLE IF EXISTS atp_player_ranking_summary;
CREATE TABLE atp_player_ranking_summary AS
WITH best AS (
    SELECT DISTINCT ON (player_id)
        player_id, ranking AS best_ranking, points AS best_points, ranking_date AS best_date
    FROM atp_rankings
    ORDER BY player_id, ranking ASC, ranking_date ASC
),
latest AS (
    SELECT DISTINCT ON (player_id)
        player_id, ranking AS latest_ranking, points AS latest_points, ranking_date AS latest_date
    FROM atp_rankings
    ORDER BY player_id, ranking_date DESC
)
SELECT b.player_id, b.best_ranking, b.best_points, b.best_date,
       l.latest_ranking, l.latest_points, l.latest_date
FROM best b JOIN latest l USING (player_id);

ALTER TABLE atp_player_ranking_summary ADD PRIMARY KEY (player_id);
CREATE INDEX idx_summary_best ON atp_player_ranking_summary (best_ranking);

SELECT 'year_end_rows' AS what, count(*) FROM atp_year_end_rankings
UNION ALL
SELECT 'summary_rows', count(*) FROM atp_player_ranking_summary;
