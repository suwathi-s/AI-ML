import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

# ----------------------------
# Step 1: Setup Chrome Driver
# ----------------------------
options = Options()
options.add_argument("--headless")  # run without opening browser
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ----------------------------
# Step 2: Go to Cricsheet
# ----------------------------
url = "https://cricsheet.org/"
driver.get(url)
print("Page title:", driver.title)

# ----------------------------
# Step 3: Find Dataset Links
# ----------------------------
# The datasets are under the "Data" section
dataset_links = {
    "odi": "https://cricsheet.org/downloads/odis_json.zip",
    "test": "https://cricsheet.org/downloads/tests_json.zip",
    "t20": "https://cricsheet.org/downloads/t20s_json.zip",
    "ipl": "https://cricsheet.org/downloads/ipl_json.zip"
}

# ----------------------------
# Step 4: Download Files
# ----------------------------
save_dir = "datasets"
os.makedirs(save_dir, exist_ok=True)

for format_name, link in dataset_links.items():
    print(f"Downloading {format_name.upper()} data...")
    response = requests.get(link)
    file_path = os.path.join(save_dir, f"{format_name}.zip")
    with open(file_path, "wb") as f:
        f.write(response.content)
    print(f"{format_name.upper()} saved at {file_path}")

driver.quit()
print("✅ All datasets downloaded successfully!")
