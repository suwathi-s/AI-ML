import os
import zipfile

# Folders
save_dir = "datasets"
extract_dir = "data/matches"
os.makedirs(extract_dir, exist_ok=True)

# Dataset formats
formats = ["odi", "test", "t20", "ipl"]

# Extract each dataset
for fmt in formats:
    zip_path = os.path.join(save_dir, f"{fmt}.zip")
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        print(f"⚠️ {zip_path} not found, skipping...")

print(f"✅ All datasets extracted into {extract_dir}")
