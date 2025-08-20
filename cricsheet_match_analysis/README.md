#  CricSheet Match Analysis  

##  Abstract  
This project performs an **end-to-end analysis of cricket matches** using the **CricSheet dataset**.  
It integrates **data extraction, cleaning, storage in MySQL, Exploratory Data Analysis (EDA), Selenium automation, and visualization using Power BI**.  
The goal is to provide meaningful insights about teams, players, and match outcomes across different formats (ODI, T20, Test, IPL).  

---

## 📂 Project Structure  

cricsheet_match_analysis/
│
├── cleaned_csv/ # Cleaned CSV files after preprocessing
├── data/ # Raw + processed data
├── datasets/ # CricSheet datasets (ODI, T20, Test, IPL zips)
├── eda_outputs/ # PNG charts generated from Python EDA
├── exports/ # Exported outputs (CSV, Excel)
├── extracted/ # Extracted match JSON files
├── query_outputs/ # SQL query results (CSV/Excel)
│
├── cricsheet_analysis_queries.sql # 20 SQL queries for insights
├── cricsheet_eda.ipynb # Jupyter Notebook for EDA
├── cricsheet_match_analysis.pbix # Power BI dashboard
├── csv_exporter.py # Script to export cleaned data
├── data_cleaner.py # Script for cleaning raw CricSheet data
├── data_loader.py # Script to load data into MySQL
├── database.py # Database connection utilities
├── extractor.py # Extracts CricSheet zip/json files
├── raw_cricsheet_db.py # Load raw CricSheet JSON → MySQL
├── raw_cricsheet_db.sql # SQL schema for CricSheet DB
├── scraper.py # Selenium/Helper script for dataset handling
├── sql_output.py # SQL execution & export script
│
├── Exploratory Data Analysis of Cricket Matches.pptx # Final presentation


---

##  Tech Stack  

- **Python** → Data cleaning, transformation, and EDA  
- **MySQL** → Database to store structured CricSheet data  
- **Power BI** → Interactive dashboard for visualization  
- **Jupyter Notebook** → Exploratory data analysis (EDA + PNG charts)  
- **SQL** → 20+ analytical queries for insights  
- **Selenium** → Automated downloading and handling of CricSheet datasets  

---

##  Features  

1. **Data Extraction** – Parses CricSheet zip files (ODI, T20, Test, IPL).  
2. **Selenium Automation** – Automates dataset downloads and preprocessing tasks.  
3. **Data Cleaning** – Removes nulls, restructures JSON, outputs cleaned CSV.  
4. **Database Integration** – Stores structured match data in MySQL.  
5. **EDA** – Generates visual insights (team performance, match outcomes, player stats).  
6. **SQL Queries** – 20+ queries (top players, highest scores, wickets, season stats).  
7. **Power BI Dashboard** –  
   - Page 1: Batting Insights  
   - Page 2: Bowling Insights  
   - Page 3: Team Insights  
 

---

##  Example Insights  

- Win percentages of teams across formats  
- Season-wise match distribution  
- Top run scorers and wicket-takers  
- Most sixes, dot balls, and dismissal types  
- Team win-loss ratios & venue statistics  

---

##  How to Run  

### 1️⃣ Setup Environment  

pip install -r requirements.txt

2️⃣ Database Setup

Import raw_cricsheet_db.sql into MySQL

Run raw_cricsheet_db.py to insert CricSheet JSON data

3️⃣ Data Cleaning & Export
python data_cleaner.py
python csv_exporter.py

4️⃣ Run SQL Queries
mysql -u root -p cricsheet_analysis < cricsheet_analysis_queries.sql

5️⃣ Exploratory Data Analysis

Open and run cricsheet_eda.ipynb in Jupyter Notebook

6️⃣ Visualization

Open cricsheet_match_analysis.pbix in Power BI

Use slicers for format, year, player, team

##  Deliverables

✔️ Cleaned CSV files

✔️ SQL queries & results

✔️ Python scripts (OOP + Selenium)

✔️ Power BI dashboard (.pbix)

✔️ EDA outputs (PNG charts)

✔️ Final Presentation (PPTX)

## Conclusion

This project successfully integrates data engineering, analytics, and visualization to provide rich cricket insights.
It showcases the use of Python, SQL, Selenium automation, MySQL, and Power BI to deliver an end-to-end data analysis pipeline.

👨‍💻 Author : 
SUWATHI S


