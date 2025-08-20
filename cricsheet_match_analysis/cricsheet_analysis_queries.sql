-- =========================================
-- Cricsheet Match Data Analysis Queries
-- =========================================
USE cricsheet_analysis;

--  Top 10 Batsmen by Total Runs (ODI)
SELECT p.name, SUM(d.runs_batsman) AS total_runs
FROM deliveries d
JOIN players p ON d.batsman_id = p.player_id
JOIN matches m ON d.match_id = m.match_id
WHERE m.format = 'ODI'
GROUP BY p.name
ORDER BY total_runs DESC
LIMIT 10;

--  Leading Wicket-Takers (All Formats)
SELECT p.name, COUNT(*) AS wickets
FROM deliveries d
JOIN players p ON d.player_dismissed_id = p.player_id
WHERE d.dismissal_kind IS NOT NULL
GROUP BY p.name
ORDER BY wickets DESC
LIMIT 10;

--  Players with Most Sixes t20
SELECT p.name, SUM(d.runs_batsman = 6) AS sixes
FROM deliveries d
JOIN players p ON d.batsman_id = p.player_id
JOIN matches m ON d.match_id = m.match_id
WHERE m.format = 'T20'
GROUP BY p.name
ORDER BY sixes DESC
LIMIT 10;


--  Top 10 test team totals
SELECT m.match_id, m.team1, SUM(d.runs_total) AS team_runs
FROM deliveries d
JOIN matches m ON d.match_id = m.match_id
WHERE m.format = 'TEST'
GROUP BY m.match_id, m.team1
ORDER BY team_runs DESC
LIMIT 10;


--  Average Runs per Match per Player
SELECT p.name, AVG(player_runs) AS avg_runs
FROM (
    SELECT d.batsman_id, SUM(d.runs_batsman) AS player_runs, d.match_id
    FROM deliveries d
    JOIN matches m ON d.match_id = m.match_id
    WHERE m.format = 'ODI'   -- filter here
    GROUP BY d.batsman_id, d.match_id
) AS match_runs
JOIN players p ON match_runs.batsman_id = p.player_id
GROUP BY p.name
ORDER BY avg_runs DESC
LIMIT 10;


--  Team Win Counts by Format
SELECT winner AS team, format, COUNT(*) AS wins
FROM matches
GROUP BY winner, format
ORDER BY wins DESC;

--  Most Common Dismissal Type
SELECT dismissal_kind, COUNT(*) AS count
FROM deliveries
WHERE dismissal_kind IS NOT NULL
GROUP BY dismissal_kind
ORDER BY count DESC;

-- Listing the first 10 matches that have a winner
SELECT match_id, winner
FROM matches
WHERE winner IS NOT NULL
ORDER BY match_id
LIMIT 10;


--  Players who Opened in Most Matches
SELECT p.name, COUNT(*) AS open_matches
FROM match_players mp
JOIN players p ON mp.player_id = p.player_id
WHERE mp.role = 'player'
GROUP BY p.name
ORDER BY open_matches DESC
LIMIT 10;

-- Top 10 individual scores in ODI matches
SELECT p.name, d.match_id, SUM(d.runs_batsman) AS score
FROM deliveries d
JOIN players p ON d.batsman_id = p.player_id
JOIN matches m ON d.match_id = m.match_id
WHERE m.format = 'ODI'
GROUP BY p.name, d.match_id
ORDER BY score DESC
LIMIT 10;


--  Bowler Economy Rate (All Formats)
SELECT p.name, SUM(d.runs_total)/COUNT(DISTINCT CONCAT(d.match_id,'-',d.over_num)) AS economy
FROM deliveries d
JOIN players p ON d.bowler_id = p.player_id
GROUP BY p.name
ORDER BY economy ASC
LIMIT 10;

--  Matches Played Per Season
SELECT season, COUNT(*) AS matches_played
FROM matches
GROUP BY season
ORDER BY season;

--  Player Most Dismissed
SELECT p.name, COUNT(*) AS times_out
FROM deliveries d
JOIN players p ON d.player_dismissed_id = p.player_id
GROUP BY p.name
ORDER BY times_out DESC
LIMIT 10;

--  Teams with Highest Win Percentage
SELECT winner AS team, 
       COUNT(*)*100.0/(SELECT COUNT(*) FROM matches WHERE team1=winner OR team2=winner) AS win_percentage
FROM matches
GROUP BY winner
ORDER BY win_percentage DESC
LIMIT 10;


--  Top Partnerships (Two Batsmen)
SELECT batsman_id, non_striker_id, SUM(runs_total) AS partnership_runs
FROM deliveries
GROUP BY batsman_id, non_striker_id
ORDER BY partnership_runs DESC
LIMIT 10;

--  Most Wickets in a Single Match by a Player
SELECT p.name, d.match_id, COUNT(*) AS wickets
FROM deliveries d
JOIN players p ON d.bowler_id = p.player_id
WHERE d.dismissal_kind IS NOT NULL
GROUP BY p.name, d.match_id
ORDER BY wickets DESC
LIMIT 10;


# Players with Most Dot Balls
SELECT p.name, COUNT(*) AS dot_balls
FROM deliveries d
JOIN players p ON d.batsman_id = p.player_id
WHERE d.runs_batsman = 0
GROUP BY p.name
ORDER BY dot_balls DESC
LIMIT 10;

# Most Frequent Match Venues
SELECT venue, COUNT(*) AS matches_played
FROM matches
GROUP BY venue
ORDER BY matches_played DESC
LIMIT 10;

# Teams with Best Win/Loss Ratio
SELECT winner AS team, 
       COUNT(*) / (SELECT COUNT(*) FROM matches WHERE team1=winner OR team2=winner) AS win_loss_ratio
FROM matches
WHERE winner IS NOT NULL
GROUP BY winner
ORDER BY win_loss_ratio DESC
LIMIT 10;


--  Top 10 Batsmen by Total Runs (T20)
SELECT p.name, SUM(d.runs_batsman) AS total_runs
FROM deliveries d
JOIN players p ON d.batsman_id = p.player_id
JOIN matches m ON d.match_id = m.match_id
WHERE m.format = 'T20'
GROUP BY p.name
ORDER BY total_runs DESC
LIMIT 10;