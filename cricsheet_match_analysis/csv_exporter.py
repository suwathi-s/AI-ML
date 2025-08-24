import os
import mysql.connector
import pandas as pd

def export_tables():
    # Connect to MySQL
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="suwathi@2004",
        database="cricsheet_analysis"
    )
    cursor = conn.cursor()

    # Directory to save exports
    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)

    # Tables to export
    tables = ["matches", "players", "match_players", "deliveries"]

    for table in tables:
        query = f"SELECT * FROM {table};"
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=columns)

        # Save to CSV
        file_path = os.path.join(export_dir, f"{table}.csv")
        df.to_csv(file_path, index=False)
        print(f"Exported {table} → {file_path}")

    cursor.close()
    conn.close()
    print("All tables exported successfully!")

if __name__ == "__main__":
    export_tables()
