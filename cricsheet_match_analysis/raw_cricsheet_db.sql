-- Drop & create database
DROP DATABASE IF EXISTS cricsheet_db;
CREATE DATABASE cricsheet_db;
USE cricsheet_db;

-- ---------------- Matches Tables ----------------

CREATE TABLE odi_matches (
    match_id INT AUTO_INCREMENT PRIMARY KEY,
    match_date DATE NOT NULL,
    team1 VARCHAR(100) NOT NULL,
    team2 VARCHAR(100) NOT NULL,
    venue VARCHAR(255) NOT NULL,
    city VARCHAR(100) NOT NULL,
    country VARCHAR(100) NOT NULL,
    toss_winner VARCHAR(100) NOT NULL,
    toss_decision ENUM('bat','field') NOT NULL,
    winner VARCHAR(100) NOT NULL,
    player_of_match VARCHAR(150) NOT NULL,
    overs INT NOT NULL
);

CREATE TABLE t20_matches LIKE odi_matches;
CREATE TABLE test_matches LIKE odi_matches;
CREATE TABLE ipl_matches LIKE odi_matches;
