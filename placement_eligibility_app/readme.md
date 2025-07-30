# 🎓 Placement Eligibility Streamlit Application

##  Project Overview
This is a complete end-to-end **Placement Eligibility Web Application** built using:

-  Python (with OOP concepts)
-  MySQL
-  Streamlit
-  Faker (for synthetic data generation)

It enables users to:
- Filter eligible students based on technical and soft skills.
- Run SQL-based placement insights.
- Download filtered student data as CSV.

---

##  Technologies Used

| Layer         | Technology         |
|---------------|--------------------|
| Frontend      | Streamlit          |
| Backend       | MySQL              |
| Data Generation | Faker (Python)    |
| Data Handling | Pandas             |
| Architecture  | OOP in Python      |

---

## 🗂️ Project Structure

placement_eligibility_app/
│
├── app.py # Main Streamlit app
├── data_generator_mysql.py # Faker-based data generator with OOP
├── placement_db.sql # SQL schema for table creation
├── data/
│ ├── students.csv
│ ├── programming.csv
│ ├── soft_skills.csv
│ └── placements.csv
├── README.md # Project documentation

---

##  Features

### 1️⃣ Eligibility Checker
-  Filter students based on:
  - Problems solved
  - Average soft skills score
  - Mock interview score
-  Batch-wise filtering
-  Download eligible student list as CSV

### 2️⃣ Placement Insights
-  Average project score per batch
-  Top 5 not placed eligible students
-  Placement status distribution
-  Most popular programming languages
-  Highest placement package by batch

---

## 💻 How to Run

### Prerequisites
- MySQL Server installed and running
- Python 3.8+ and `pip` installed
- Install required packages
pip install streamlit mysql-connector-python pandas faker
### Setup Steps
### 1.Create the database:
Run: placement_db.sql inside MySQL
### 2.Update your MySQL credentials in:
data_generator_mysql.py
app.py
###  Generate Data
python data_generator_mysql.py
### Run Streamlit App
streamlit run app.py
### 📁Downloadable CSVs (Auto-generated)
students.csv
programming.csv
soft_skills.csv
placements.csv

### Sample SQL Queries Used
Average project score per batch

Top 5 not placed eligible students

Placement status distribution

Most popular programming languages

Highest package per batch

Batch-wise soft skills average

Internship count by batch

Students with certifications

### 👩‍💻 Author
Name: Suwathi
Tech Stack: Python | Streamlit | MySQL | Faker | OOP
