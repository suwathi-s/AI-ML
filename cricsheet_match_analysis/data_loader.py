import os
import json
import mysql.connector

# ----------------------------
# Database Connection
# ----------------------------
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="suwathi@2004",
        database="cricsheet_analysis"
    )

# ----------------------------
# Player Insert (avoid duplicates)
# ----------------------------
def get_or_create_player(cursor, player_name, cache):
    if player_name in cache:
        return cache[player_name]

    cursor.execute("SELECT player_id FROM players WHERE name = %s", (player_name,))
    result = cursor.fetchone()
    if result:
        player_id = result[0]
    else:
        cursor.execute("INSERT INTO players (name) VALUES (%s)", (player_name,))
        player_id = cursor.lastrowid
    cache[player_name] = player_id
    return player_id

# ----------------------------
# Insert Match Data
# ----------------------------
def insert_match(cursor, match_id, info):
    cursor.execute("""
        INSERT INTO matches (match_id, format, season, venue, team1, team2, toss_winner, toss_decision, winner, result)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE match_id = match_id
    """, (
        match_id,
        info.get("match_type"),
        str(info.get("season", "")),
        info.get("venue", ""),
        info.get("teams", ["", ""])[0],
        info.get("teams", ["", ""])[1],
        info.get("toss", {}).get("winner", ""),
        info.get("toss", {}).get("decision", ""),
        info.get("outcome", {}).get("winner", ""),
        str(info.get("outcome", {}).get("result", ""))
    ))

# ----------------------------
# Insert Players for Match
# ----------------------------
def insert_match_players(cursor, match_id, info, player_cache):
    if "players" not in info:
        return
    for team, players in info["players"].items():
        for player in players:
            player_id = get_or_create_player(cursor, player, player_cache)
            cursor.execute("""
                INSERT INTO match_players (match_id, player_id, role)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE match_id = match_id
            """, (match_id, player_id, "player"))

# ----------------------------
# Insert Deliveries
# ----------------------------
def insert_deliveries(cursor, match_id, innings, player_cache):
    for inning in innings:
        inning_num = inning.get("inning", 0)
        overs = inning.get("overs", [])
        for over in overs:
            over_num = over.get("over", 0)
            for delivery in over.get("deliveries", []):
                batsman = get_or_create_player(cursor, delivery.get("batter"), player_cache)
                bowler = get_or_create_player(cursor, delivery.get("bowler"), player_cache)
                non_striker = get_or_create_player(cursor, delivery.get("non_striker"), player_cache)

                runs = delivery.get("runs", {})
                extras = delivery.get("extras", 0)
                dismissal = delivery.get("wickets", [])

                player_dismissed_id = None
                dismissal_kind = None
                if dismissal:
                    out_info = dismissal[0]
                    dismissal_kind = out_info.get("kind")
                    if out_info.get("player_out"):
                        player_dismissed_id = get_or_create_player(cursor, out_info["player_out"], player_cache)

                cursor.execute("""
                    INSERT INTO deliveries (match_id, inning, over_num, ball_num,
                        batsman_id, bowler_id, non_striker_id,
                        runs_batsman, runs_extras, runs_total,
                        dismissal_kind, player_dismissed_id)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    match_id,
                    inning_num,
                    over_num,
                    delivery.get("ball", 0),
                    batsman,
                    bowler,
                    non_striker,
                    runs.get("batter", 0),
                    runs.get("extras", 0),
                    runs.get("total", 0),
                    dismissal_kind,
                    player_dismissed_id
                ))

# ----------------------------
# Process a JSON Match File
# ----------------------------
def process_match_file(cursor, file_path, player_cache):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    match_id = str(data.get("info", {}).get("match_id", os.path.basename(file_path).replace(".json", "")))
    info = data.get("info", {})
    innings = data.get("innings", [])

    insert_match(cursor, match_id, info)
    insert_match_players(cursor, match_id, info, player_cache)
    insert_deliveries(cursor, match_id, innings, player_cache)

    print(f"Inserted match {match_id}")

# ----------------------------
# Main Loader
# ----------------------------
def load_json_files(folder="data/matches"):
    conn = get_connection()
    cursor = conn.cursor()
    player_cache = {}

    for file_name in os.listdir(folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder, file_name)
            try:
                process_match_file(cursor, file_path, player_cache)
                conn.commit()
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                conn.rollback()

    conn.close()
    print("All matches loaded successfully!")

if __name__ == "__main__":
    load_json_files()
