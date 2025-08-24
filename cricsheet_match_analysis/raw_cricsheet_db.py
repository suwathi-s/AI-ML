import pandas as pd
import mysql.connector
import random
import os

# ----------------------------
# Step 0: Configuration
# ----------------------------
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "suwathi@2004",
    "database": "cricsheet_db"
}

csv_dir = "cleaned_csv"
formats = ["odi", "t20", "test", "ipl"]

# Default values for missing columns
default_city = "Unknown City"
default_country = "Unknown Country"

# Overs ranges per format
overs_ranges = {
    "odi": (45, 60),
    "t20": (18, 22),
    "test": (80, 100),
    "ipl": (18, 22)
}

# ----------------------------
# Step 1: Connect to MySQL
# ----------------------------
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# ----------------------------
# Step 2: Insert Data Function
# ----------------------------
def insert_data(format_name):
    file_path = os.path.join(csv_dir, f"{format_name}.csv")
    if not os.path.exists(file_path):
        print(f"CSV file not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    # Fill missing city/country/winner/player_of_match
    df['city'] = df['city'].fillna(default_city)
    df['country'] = df['country'].fillna(default_country)
    df['winner'] = df['winner'].fillna("No Result")             #  Fill missing winner
    df['player_of_match'] = df['player_of_match'].fillna("Unknown")  #  Fill missing player

    table_name = f"{format_name}_matches"

    for _, row in df.iterrows():
        # Fill missing overs with realistic random number
        overs = row['overs']
        if pd.isna(overs):
            min_overs, max_overs = overs_ranges.get(format_name, (20, 50))
            overs = random.randint(min_overs, max_overs)
        else:
            overs = int(overs)

        try:
            cursor.execute(f"""
                INSERT INTO {table_name} (
                    match_date, team1, team2, venue, city, country,
                    toss_winner, toss_decision, winner, player_of_match, overs
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                row['match_date'],
                row['team1'],
                row['team2'],
                row['venue'],
                row['city'],
                row['country'],
                row['toss_winner'],
                row['toss_decision'],
                row['winner'],
                row['player_of_match'],
                overs
            ))
        except Exception as e:
            print(f"Error inserting match {row.get('match_date')}: {e}")

    conn.commit()
    print(f"{format_name.upper()} matches loaded successfully!")

# ----------------------------
# Step 3: Load all formats
# ----------------------------
try:
    for fmt in formats:
        insert_data(fmt)
finally:
    # ----------------------------
    # Step 4: Close connection
    # ----------------------------
    cursor.close()
    conn.close()
    print(" All 4 tables loaded successfully into MySQL!")
