-- The purpose of this analysis is to identify common traits among the best NBA players of all time.
-- Specifically, it aims to determine whether physical attributes such as height and weight, the country of origin, 
-- college background, or team affiliation play a significant role in a player's success.
-- To explore this, a sample of the top 15 players across all seasons was selected based on a weighted average score that 
-- incorporates total points, point efficiency (true shooting percentage), and contribution to team success through assists.

-- -- I'm creating a temporary table of top 15 players from all seasons so I can refer to it in multiple queries
CREATE TEMPORARY TABLE top_15_players AS
WITH players_statistics AS (
    SELECT 
        DISTINCT player_name,
        AVG(pts) AS avg_points,
        AVG(ts_pct) AS avg_true_shooting_percentage,
        AVG(ast_pct) AS avg_assist_percentage,
        CASE
            WHEN SUM(CASE WHEN draft_number = 'Undrafted' THEN 0 ELSE 1 END) = 0 THEN 0
            ELSE COALESCE(AVG(pts) / NULLIF(SUM(CASE WHEN draft_number = 'Undrafted' THEN NULL ELSE draft_number::int END), 0), 0)
        END AS avg_points_per_draft_position
    FROM all_seasons
    GROUP BY player_name
), weighted_scores AS (
    SELECT 
        player_name,
        ROUND((avg_points * 0.4 + avg_true_shooting_percentage * 0.2 + avg_assist_percentage * 0.2 + avg_points_per_draft_position * 0.2)::numeric, 2) AS weighted_score
    FROM players_statistics
)
SELECT *
FROM (
    SELECT *, ROW_NUMBER() OVER (ORDER BY weighted_score DESC) AS rank
    FROM weighted_scores
) ranked
WHERE rank <= 15
ORDER BY rank ASC;

SELECT * FROM top_15_players;

-- Analysis of physical characteristics and origin of the best players
SELECT 
	t.rank AS player_id,
    t.player_name AS player,
    t.weighted_score AS weighted_score,
    MAX(a.college) AS college,
    MAX(a.country) AS origin,
    ROUND(AVG(a.player_height)::numeric, 2) AS height,
    ROUND(AVG(AVG(a.player_height)) OVER ()::numeric, 2) AS avg_height,
    ROUND((AVG(a.player_height) - AVG(AVG(a.player_height)) OVER ())::numeric, 2) AS players_height_diff_from_avg,
    ROUND(AVG(a.player_weight)::numeric, 2) AS weight,
    ROUND(AVG(AVG(a.player_weight)) OVER ()::numeric, 2) AS avg_weight,
    ROUND((AVG(a.player_weight) - AVG(AVG(a.player_weight)) OVER ())::numeric, 2) AS players_weight_diff_from_avg,
    COUNT(DISTINCT a.season) AS seasons_played,
    COUNT(DISTINCT a.team_abbreviation) AS nb_of_teams,
    STRING_AGG(DISTINCT a.team_abbreviation, ', ') AS teams,
    COUNT(DISTINCT a.season) AS seasons_played,
    ROUND(AVG(a.pts)::numeric, 2) AS avg_points
FROM top_15_players t
JOIN all_seasons a ON t.player_name = a.player_name
GROUP BY t.rank, t.player_name, t.weighted_score
ORDER BY t.weighted_score DESC;

-- Paolo Banchero has the highest weighted score of 12.14 but has played only one season.
-- LeBron James played the most seasons (20) and for the highest number of teams (3).
-- The average height of these top players is 197.44 cm.
-- The average weight is 101.63 kg.
-- Allen Iverson is significantly shorter and lighter than the average. The wide variety in physical traits suggests there is no single ideal body type for NBA success.
-- Most players are from the USA, with exceptions like Luka Doncic (Slovenia) and Joel Embiid (Cameroon).
-- The dominance of American players confirms the strength of the U.S. basketball system, but the presence of international stars shows the NBA's growing global reach.


-- Check at what age players had their best performances and in which season to see if they played against each other at that time
WITH players_stats AS (
    SELECT 
        player_name,
        age,
        season,
        AVG(pts) AS avg_points,
        AVG(ts_pct) AS avg_true_shooting_percentage,
        AVG(ast_pct) AS avg_assist_percentage,
        CASE
            WHEN SUM(CASE WHEN draft_number = 'Undrafted' THEN 0 ELSE 1 END) = 0 THEN 0
            ELSE COALESCE(AVG(pts) / NULLIF(SUM(CASE WHEN draft_number = 'Undrafted' THEN NULL ELSE draft_number::int END), 0), 0)
        END AS avg_points_per_draft_position
    FROM all_seasons
    GROUP BY player_name, age, season
),
weighted_scores AS (
    SELECT 
        player_name,
        age,
        season,
        ROUND((avg_points * 0.4 + avg_true_shooting_percentage * 0.2 + avg_assist_percentage * 0.2 + avg_points_per_draft_position * 0.2)::numeric, 2) AS weighted_score
    FROM players_stats
),
ranked_scores AS (
    SELECT 
        player_name,
        age,
        season,
        weighted_score,
        LAG(weighted_score) OVER (PARTITION BY player_name ORDER BY season) AS prev_year_score,
        LEAD(weighted_score) OVER (PARTITION BY player_name ORDER BY season) AS next_year_score,
        ROW_NUMBER() OVER (PARTITION BY player_name ORDER BY weighted_score DESC) AS rank
    FROM weighted_scores
)
SELECT 
	t.rank as player_id,
    rs.player_name,
    rs.age AS best_performance_age,
    ROUND(AVG(rs.age) OVER()::int) AS average_best_age,
    CASE
        WHEN rs.age > AVG(rs.age) OVER() THEN 'above'
        ELSE 'below'
    END AS below_above_avg_best_age,
    rs.season AS best_performance_season, 
    rs.weighted_score AS best_score,
    rs.prev_year_score AS score_year_before,
    rs.next_year_score AS score_year_after
FROM ranked_scores rs
JOIN top_15_players t ON rs.player_name = t.player_name
WHERE rs.rank = 1
GROUP BY t.rank, rs.player_name, rs.age, rs.season, rs.weighted_score, rs.prev_year_score, rs.next_year_score
ORDER BY rs.weighted_score DESC;

-- The average age for best performances is 26, but there's a wide range – from 20 (Zion Williamson) to 34 (Michael Jordan).
-- This shows peak performance can occur at various stages of a career.
-- The 2022-23 season was the best for several players, including Joel Embiid, Luka Doncic, and Damian Lillard.
-- LeBron James stands out for career length (20 seasons) and number of teams (3), proving his exceptional durability and value.
-- The presence of players like Paolo Banchero or Anthony Edwards in the top 15 shows young players can quickly reach elite levels in the NBA.


-- Check how many top 15 players attended each college
SELECT 
    college,
    COUNT(DISTINCT player_name) AS nb_of_players
FROM all_seasons
WHERE college IS NOT NULL AND college <> 'None' AND player_name IN (SELECT DISTINCT player_name FROM top_15_players)
GROUP BY college
ORDER BY nb_of_players DESC;

-- 11 different colleges are represented among these top players. Most colleges produced only one top player, showing diversity in player backgrounds.
-- Duke University had the most top players (3) on the list, but no other school had a clear dominance. Some top players like LeBron James or Luka Doncic never attended college before the NBA.


-- Check how many top 15 players belonged to each team
SELECT 
    a.team_abbreviation AS team,
    COUNT(DISTINCT t.player_name) AS nb_of_top_players
FROM all_seasons a
JOIN top_15_players t ON a.player_name = t.player_name
GROUP BY a.team_abbreviation
ORDER BY nb_of_top_players DESC;


-- 21 different NBA teams had at least one of the top 15 players in their roster.
-- Brooklyn Nets, Cleveland Cavaliers, and Philadelphia 76ers each had 3 top players, suggesting effectiveness in acquiring top talent.

-- Summary: The data suggests that NBA success depends on many factors, not just college or physical attributes. 
-- Individual skills, work ethic, and adaptation to high-level play seem more important.
-- The dominance of U.S. players confirms the strength of the American basketball system, but the presence of international stars shows that the NBA is becoming increasingly global.

-- Comment: Interesting presentation approach. Business analysis is visible. I did not notice any logical errors.
-- Tableau presentation is fine. I would change the scale for height and weight (to better highlight the differences).
-- The “Seasonal Performance Analysis” is a bit risky – it took me a moment to understand it, so a live explanation would help (I couldn’t find an alternative view either).