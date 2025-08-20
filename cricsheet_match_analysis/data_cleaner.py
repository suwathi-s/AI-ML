import os
import zipfile
import json
import pandas as pd

# ----------------------------
# Step 1: Extract Zip Files
# ----------------------------
formats = ["odi", "test", "t20", "ipl"]
base_dir = "datasets"
extract_dir = "extracted"

os.makedirs(extract_dir, exist_ok=True)

for fmt in formats:
    zip_path = os.path.join(base_dir, f"{fmt}.zip")
    out_dir = os.path.join(extract_dir, fmt)
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)
    print(f"Extracted {fmt.upper()} matches into {out_dir}")

# ----------------------------
# Step 2: Function to Clean JSON
# ----------------------------
def parse_match(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    info = data.get("info", {})
    
    return {
        "match_date": info.get("dates", [None])[0],
        "team1": info.get("teams", [None, None])[0],
        "team2": info.get("teams", [None, None])[1],
        "venue": info.get("venue"),
        "city": info.get("city"),
        "country": info.get("country"),
        "toss_winner": info.get("toss", {}).get("winner"),
        "toss_decision": info.get("toss", {}).get("decision"),
        "winner": info.get("outcome", {}).get("winner"),
        "player_of_match": (info.get("player_of_match", [None])[0]
                            if isinstance(info.get("player_of_match"), list)
                            else info.get("player_of_match")),
        "overs": info.get("overs")
    }

# ----------------------------
# Step 3: Loop through all formats
# ----------------------------
clean_dir = "cleaned_csv"
os.makedirs(clean_dir, exist_ok=True)

for fmt in formats:
    matches_dir = os.path.join(extract_dir, fmt)
    rows = []
    for file in os.listdir(matches_dir):
        if file.endswith(".json"):
            filepath = os.path.join(matches_dir, file)
            try:
                rows.append(parse_match(filepath))
            except Exception as e:
                print(f"Error in {file}: {e}")
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(clean_dir, f"{fmt}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Cleaned {fmt.upper()} data saved to {csv_path}")
