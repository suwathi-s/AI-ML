import streamlit as st
import mysql.connector
import pandas as pd

# --- MySQL Connection ---
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="suwathi@2004",
        database="placement_db"
    )

# --- Page Setup ---
st.set_page_config(page_title="Placement Eligibility App", layout="wide")
st.title("🎓 Placement Eligibility Application")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Global Filters")
selected_batch = st.sidebar.selectbox(
    "Select Course Batch",
    ["All"] + [f"Batch-{i}" for i in range(1, 11)]
)

# --- Eligibility Filters ---
st.sidebar.header("🎯 Eligibility Criteria")
min_problems = st.sidebar.slider("Minimum Problems Solved", 0, 300, 200)
min_soft_skills = st.sidebar.slider("Minimum Avg Soft Skills Score", 0, 100, 70)
min_mock = st.sidebar.slider("Minimum Mock Interview Score", 0, 100, 70)

# --- Load Eligible Students ---
def load_filtered_students(min_problems, min_soft_skills, min_mock):
    query = f"""
    SELECT s.student_id, s.name, s.email, s.course_batch,
           p.problems_solved,
           ROUND((ss.communication + ss.teamwork + ss.presentation + ss.leadership + ss.critical_thinking + ss.interpersonal_skills)/6, 2) AS avg_soft_skills,
           pl.mock_interview_score,
           pl.internships_completed,
           pl.placement_status
    FROM students s
    JOIN programming p ON s.student_id = p.student_id
    JOIN soft_skills ss ON s.student_id = ss.student_id
    JOIN placements pl ON s.student_id = pl.student_id
    WHERE p.problems_solved >= {min_problems}
      AND ((ss.communication + ss.teamwork + ss.presentation + ss.leadership + ss.critical_thinking + ss.interpersonal_skills)/6) >= {min_soft_skills}
      AND pl.mock_interview_score >= {min_mock}
    """
    if selected_batch != "All":
        query += f" AND s.course_batch = '{selected_batch}'"
    query += " ORDER BY pl.mock_interview_score DESC"

    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --- Eligibility Checker Section ---
st.subheader(" 🧾 Eligibility Checker")

if st.sidebar.button("Show Eligible Students"):
    df_eligible = load_filtered_students(min_problems, min_soft_skills, min_mock)
    if df_eligible.empty:
        st.warning("❌ No students match the given criteria.")
    else:
        st.success(f"✅ Found {len(df_eligible)} eligible students.")
        st.dataframe(df_eligible)
        csv = df_eligible.to_csv(index=False)
        st.download_button("📥 Save & Download CSV", data=csv, file_name="eligible_students.csv", mime="text/csv")

# --- SQL Insights Section ---
st.markdown("##  📍 Run SQL Insights")
query_options = {
    "1. Total number of students": "SELECT COUNT(*) AS total_students FROM students",
    "2. Number of placed students": "SELECT COUNT(*) AS placed_students FROM placements WHERE placement_status = 'Placed'",
    "3. Top 5 mock interview scores": """
        SELECT s.name, pl.mock_interview_score
        FROM students s JOIN placements pl ON s.student_id = pl.student_id
        ORDER BY pl.mock_interview_score DESC LIMIT 5
    """,
    "4. Average project score per batch": """
        SELECT s.course_batch, AVG(p.latest_project_score) AS avg_score
        FROM students s JOIN programming p ON s.student_id = p.student_id
        GROUP BY s.course_batch
    """,
    "5. Average soft skills by batch": """
        SELECT s.course_batch, ROUND(AVG((ss.communication + ss.teamwork + ss.presentation + ss.leadership + ss.critical_thinking + ss.interpersonal_skills)/6), 2) AS avg_soft_skills
        FROM students s JOIN soft_skills ss ON s.student_id = ss.student_id
        GROUP BY s.course_batch
    """,
    "6. Students not placed": """
        SELECT s.name, s.course_batch FROM students s
        JOIN placements p ON s.student_id = p.student_id
        WHERE p.placement_status = 'Not Placed'
    """,
    "7. Most used programming languages": """
        SELECT p.language, COUNT(*) AS count
        FROM programming p
        JOIN students s ON p.student_id = s.student_id
        GROUP BY p.language
        ORDER BY count DESC
    """,
    "8. Internship count by batch": """
        SELECT s.course_batch, SUM(p.internships_completed) AS total_internships
        FROM students s JOIN placements p ON s.student_id = p.student_id
        GROUP BY s.course_batch
    """,
    "9. Highest placement package per batch": """
        SELECT s.course_batch, MAX(p.placement_package) AS max_package
        FROM students s JOIN placements p ON s.student_id = p.student_id
        GROUP BY s.course_batch
    """,
    "10. Students with certifications": """
        SELECT s.name, p.certifications_earned
        FROM students s JOIN programming p ON s.student_id = p.student_id
        WHERE p.certifications_earned > 0
    """
}

selected_query_label = st.selectbox("Choose a Query", list(query_options.keys()))
if st.button("🔍 Show Query Result"):
    query = query_options[selected_query_label]

    # --- Fix: Smart batch filter injection ---
    if selected_batch != "All" and "s.course_batch" in query:
        if "WHERE" in query.upper():
            if "GROUP BY" in query.upper():
                parts = query.upper().split("GROUP BY")
                query = parts[0] + f" AND s.course_batch = '{selected_batch}' GROUP BY" + parts[1]
            else:
                parts = query.upper().split("WHERE")
                query = parts[0] + f"WHERE s.course_batch = '{selected_batch}' AND" + parts[1]
        else:
            if "GROUP BY" in query.upper():
                parts = query.upper().split("GROUP BY")
                query = parts[0] + f" WHERE s.course_batch = '{selected_batch}' GROUP BY" + parts[1]
            else:
                query += f" WHERE s.course_batch = '{selected_batch}'"

    try:
        conn = get_connection()
        df_result = pd.read_sql(query, conn)
        conn.close()
        st.dataframe(df_result)
    except Exception as e:
        st.error(f"Query Error: {e}")
