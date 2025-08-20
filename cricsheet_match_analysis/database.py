import mysql.connector

def create_database():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="suwathi@2004"
    )
    cursor = conn.cursor()
    cursor.execute("DROP DATABASE IF EXISTS cricsheet_analysis;")
    cursor.execute("CREATE DATABASE cricsheet_analysis;")
    cursor.execute("USE cricsheet_analysis;")

    cursor.execute("""
    CREATE TABLE matches (
        match_id VARCHAR(50) PRIMARY KEY,
        format VARCHAR(10),
        season VARCHAR(10),
        venue VARCHAR(255),
        team1 VARCHAR(100),
        team2 VARCHAR(100),
        toss_winner VARCHAR(100),
        toss_decision VARCHAR(50),
        winner VARCHAR(100),
        result VARCHAR(50)
    );
    """)

    cursor.execute("""
    CREATE TABLE players (
        player_id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) UNIQUE
    );
    """)

    cursor.execute("""
    CREATE TABLE match_players (
        match_id VARCHAR(50),
        player_id INT,
        role VARCHAR(50),
        FOREIGN KEY (match_id) REFERENCES matches(match_id),
        FOREIGN KEY (player_id) REFERENCES players(player_id)
    );
    """)

    cursor.execute("""
    CREATE TABLE deliveries (
        delivery_id INT AUTO_INCREMENT PRIMARY KEY,
        match_id VARCHAR(50),
        inning INT,
        over_num INT,
        ball_num INT,
        batsman_id INT,
        bowler_id INT,
        non_striker_id INT,
        runs_batsman INT,
        runs_extras INT,
        runs_total INT,
        dismissal_kind VARCHAR(50),
        player_dismissed_id INT,
        FOREIGN KEY (match_id) REFERENCES matches(match_id),
        FOREIGN KEY (batsman_id) REFERENCES players(player_id),
        FOREIGN KEY (bowler_id) REFERENCES players(player_id),
        FOREIGN KEY (non_striker_id) REFERENCES players(player_id),
        FOREIGN KEY (player_dismissed_id) REFERENCES players(player_id)
    );
    """)

    print("✅ Database and tables created successfully!")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
