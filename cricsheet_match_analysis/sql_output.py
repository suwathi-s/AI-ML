import pandas as pd
import mysql.connector
import os

# ----------------------------
# Database Connection
# ----------------------------
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="suwathi@2004",
    database="cricsheet_analysis"
)

# ----------------------------
# Queries Dictionary (All 20 queries)
# ----------------------------
queries = {
    "top_10_batsmen_ODI": """
        SELECT p.name, SUM(d.runs_batsman) AS total_runs
        FROM deliveries d
        JOIN players p ON d.batsman_id = p.player_id
        JOIN matches m ON d.match_id = m.match_id
        WHERE m.format = 'ODI'
        GROUP BY p.name
        ORDER BY total_runs DESC
        LIMIT 10;
    """,
    "leading_wicket_takers": """
        SELECT p.name, COUNT(*) AS wickets
        FROM deliveries d
        JOIN players p ON d.player_dismissed_id = p.player_id
        WHERE d.dismissal_kind IS NOT NULL
        GROUP BY p.name
        ORDER BY wickets DESC
        LIMIT 10;
    """,
    "most_sixes_T20": """
        SELECT p.name, SUM(d.runs_batsman = 6) AS sixes
        FROM deliveries d
        JOIN players p ON d.batsman_id = p.player_id
        JOIN matches m ON d.match_id = m.match_id
        WHERE m.format = 'T20'
        GROUP BY p.name
        ORDER BY sixes DESC
        LIMIT 10;
    """,
    "top_10_test_team_totals": """
        SELECT m.match_id, m.team1, SUM(d.runs_total) AS team_runs
        FROM deliveries d
        JOIN matches m ON d.match_id = m.match_id
        WHERE m.format = 'TEST'
        GROUP BY m.match_id, m.team1
        ORDER BY team_runs DESC
        LIMIT 10;
    """,
    "average_runs_per_match_ODI": """
        SELECT p.name, AVG(player_runs) AS avg_runs
        FROM (
            SELECT d.batsman_id, SUM(d.runs_batsman) AS player_runs, d.match_id
            FROM deliveries d
            JOIN matches m ON d.match_id = m.match_id
            WHERE m.format = 'ODI'
            GROUP BY d.batsman_id, d.match_id
        ) AS match_runs
        JOIN players p ON match_runs.batsman_id = p.player_id
        GROUP BY p.name
        ORDER BY avg_runs DESC
        LIMIT 10;
    """,
    "team_win_counts_by_format": """
        SELECT winner AS team, format, COUNT(*) AS wins
        FROM matches
        GROUP BY winner, format
        ORDER BY wins DESC;
    """,
    "most_common_dismissal": """
        SELECT dismissal_kind, COUNT(*) AS count
        FROM deliveries
        WHERE dismissal_kind IS NOT NULL
        GROUP BY dismissal_kind
        ORDER BY count DESC;
    """,
    "first_10_matches_with_winner": """
        SELECT match_id, winner
        FROM matches
        WHERE winner IS NOT NULL
        ORDER BY match_id
        LIMIT 10;
    """,
    "players_opened_most_matches": """
        SELECT p.name, COUNT(*) AS open_matches
        FROM match_players mp
        JOIN players p ON mp.player_id = p.player_id
        WHERE mp.role = 'player'
        GROUP BY p.name
        ORDER BY open_matches DESC
        LIMIT 10;
    """,
    "top_10_individual_scores_ODI": """
        SELECT p.name, d.match_id, SUM(d.runs_batsman) AS score
        FROM deliveries d
        JOIN players p ON d.batsman_id = p.player_id
        JOIN matches m ON d.match_id = m.match_id
        WHERE m.format = 'ODI'
        GROUP BY p.name, d.match_id
        ORDER BY score DESC
        LIMIT 10;
    """,
    "bowler_economy_all_formats": """
        SELECT p.name, SUM(d.runs_total)/COUNT(DISTINCT CONCAT(d.match_id,'-',d.over_num)) AS economy
        FROM deliveries d
        JOIN players p ON d.bowler_id = p.player_id
        GROUP BY p.name
        ORDER BY economy ASC
        LIMIT 10;
    """,
    "matches_played_per_season": """
        SELECT season, COUNT(*) AS matches_played
        FROM matches
        GROUP BY season
        ORDER BY season;
    """,
    "player_most_dismissed": """
        SELECT p.name, COUNT(*) AS times_out
        FROM deliveries d
        JOIN players p ON d.player_dismissed_id = p.player_id
        GROUP BY p.name
        ORDER BY times_out DESC
        LIMIT 10;
    """,
    "teams_highest_win_percentage": """
        SELECT winner AS team, 
               COUNT(*)*100.0/(SELECT COUNT(*) FROM matches WHERE team1=winner OR team2=winner) AS win_percentage
        FROM matches
        GROUP BY winner
        ORDER BY win_percentage DESC
        LIMIT 10;
    """,
    "top_partnerships": """
        SELECT batsman_id, non_striker_id, SUM(runs_total) AS partnership_runs
        FROM deliveries
        GROUP BY batsman_id, non_striker_id
        ORDER BY partnership_runs DESC
        LIMIT 10;
    """,
    "most_wickets_single_match": """
        SELECT p.name, d.match_id, COUNT(*) AS wickets
        FROM deliveries d
        JOIN players p ON d.bowler_id = p.player_id
        WHERE d.dismissal_kind IS NOT NULL
        GROUP BY p.name, d.match_id
        ORDER BY wickets DESC
        LIMIT 10;
    """,
    "players_most_dot_balls": """
        SELECT p.name, COUNT(*) AS dot_balls
        FROM deliveries d
        JOIN players p ON d.batsman_id = p.player_id
        WHERE d.runs_batsman = 0
        GROUP BY p.name
        ORDER BY dot_balls DESC
        LIMIT 10;
    """,
    "most_frequent_venues": """
        SELECT venue, COUNT(*) AS matches_played
        FROM matches
        GROUP BY venue
        ORDER BY matches_played DESC
        LIMIT 10;
    """,
    "teams_best_win_loss_ratio": """
        SELECT winner AS team, 
               COUNT(*) / (SELECT COUNT(*) FROM matches WHERE team1=winner OR team2=winner) AS win_loss_ratio
        FROM matches
        WHERE winner IS NOT NULL
        GROUP BY winner
        ORDER BY win_loss_ratio DESC
        LIMIT 10;
    """,
    "top_10_batsmen_T20": """
        SELECT p.name, SUM(d.runs_batsman) AS total_runs
        FROM deliveries d
        JOIN players p ON d.batsman_id = p.player_id
        JOIN matches m ON d.match_id = m.match_id
        WHERE m.format = 'T20'
        GROUP BY p.name
        ORDER BY total_runs DESC
        LIMIT 10;
    """
}

# ----------------------------
# Create directory for CSVs
# ----------------------------
os.makedirs("query_outputs", exist_ok=True)

# ----------------------------
# Run queries sequentially
# ----------------------------
for name, query in queries.items():
    try:
        print(f"Running query: {name}")
        df = pd.read_sql(query, conn)
        file_path = f"query_outputs/{name}.csv"
        df.to_csv(file_path, index=False)
        print(f"✅ Saved {file_path}")
    except Exception as e:
        print(f"❌ Failed to run {name}: {e}")

# ----------------------------
# Close connection
# ----------------------------
conn.close()
print("🎉 All 20 queries executed and exported successfully!")
