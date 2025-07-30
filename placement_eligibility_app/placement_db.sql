-----SQL database schema---
CREATE DATABASE IF NOT EXISTS placement_db;
USE placement_db;

-- Students Table
CREATE TABLE students (
    student_id INT PRIMARY KEY,  -- NOT NULL implied
    name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(10) NOT NULL,
    email VARCHAR(100) NOT NULL,
    phone VARCHAR(15) NOT NULL,
    enrollment_year INT NOT NULL,
    course_batch VARCHAR(50) NOT NULL,
    city VARCHAR(50) NOT NULL,
    graduation_year INT NOT NULL
);

-- Programming Table
CREATE TABLE programming (
    programming_id INT PRIMARY KEY,
    student_id INT NOT NULL,
    language VARCHAR(50) NOT NULL,
    problems_solved INT NOT NULL,
    assessments_completed INT NOT NULL,
    mini_projects INT NOT NULL,
    certifications_earned INT NOT NULL,
    latest_project_score INT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- Soft Skills Table
CREATE TABLE soft_skills (
    soft_skill_id INT PRIMARY KEY,
    student_id INT NOT NULL,
    communication INT NOT NULL,
    teamwork INT NOT NULL,
    presentation INT NOT NULL,
    leadership INT NOT NULL,
    critical_thinking INT NOT NULL,
    interpersonal_skills INT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- Placements Table 
CREATE TABLE placements (
    placement_id INT PRIMARY KEY,
    student_id INT NOT NULL,
    mock_interview_score INT NOT NULL,
    internships_completed INT NOT NULL,
    placement_status VARCHAR(20) NOT NULL,
    company_name VARCHAR(100) NOT NULL DEFAULT 'Not Placed',
    placement_package DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    interview_rounds_cleared INT NOT NULL,
    placement_date DATE NOT NULL DEFAULT '1970-01-01',
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

