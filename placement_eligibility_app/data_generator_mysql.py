# data_generator_mysql.py
import mysql.connector
from faker import Faker
import random
from datetime import datetime
from abc import ABC, abstractmethod #Supports abstract base classes
import pandas as pd

fake = Faker()

class DBConnector:#database interactions
    def __init__(self):
        self.conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="suwathi@2004",
            database="placement_db"
        )
        self.cursor = self.conn.cursor()

    def fetch_df(self, query):
        return pd.read_sql(query, self.conn)

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


class DataGenerator(ABC):#abstract blueprint for all data generators (students, programming, etc.)

    def __init__(self, db: DBConnector):
        self.db = db

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def export_csv(self):
        pass


class StudentGenerator(DataGenerator):
    def generate(self):
        for student_id in range(1, 501):
            self.db.cursor.execute("""
                INSERT INTO students VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                student_id,
                fake.name(),
                random.randint(18, 25),
                random.choice(['Male', 'Female', 'Other']),
                fake.email(),
                fake.msisdn()[:10],
                random.randint(2019, 2022),
                f"Batch-{random.randint(1, 10)}",
                fake.city(),
                datetime.now().year + 1
            ))
        self.db.commit()

    def export_csv(self):
        df = self.db.fetch_df("SELECT * FROM students")
        df.to_csv("students.csv", index=False)
        print("✅ students.csv saved!")


class ProgrammingGenerator(DataGenerator):
    def generate(self):
        for prog_id in range(1, 501):
            self.db.cursor.execute("""
                INSERT INTO programming VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                prog_id,
                prog_id,
                random.choice(['Python', 'Java', 'C++', 'JavaScript', 'SQL', 'AIML', 'Data science']),
                random.randint(50, 300),
                random.randint(1, 10),
                random.randint(1, 5),
                random.randint(0, 3),
                random.randint(50, 100)
            ))
        self.db.commit()

    def export_csv(self):
        df = self.db.fetch_df("SELECT * FROM programming")
        df.to_csv("programming.csv", index=False)
        print("✅ programming.csv saved!")


class SoftSkillsGenerator(DataGenerator):
    def generate(self):
        for sid in range(1, 501):
            self.db.cursor.execute("""
                INSERT INTO soft_skills VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                sid, sid,
                *[random.randint(1, 100) for _ in range(6)]
            ))
        self.db.commit()

    def export_csv(self):
        df = self.db.fetch_df("SELECT * FROM soft_skills")
        df.to_csv("soft_skills.csv", index=False)
        print("✅ soft_skills.csv saved!")


class PlacementsGenerator(DataGenerator):
    def generate(self):
        for pid in range(1, 501):
            status = random.choice(['Placed', 'Not Placed'])
            company = fake.company() if status == 'Placed' else 'NA'
            package = round(random.uniform(3.0, 15.0), 2) if status == 'Placed' else 0.00
            rounds = random.randint(1, 5) if status == 'Placed' else 0
            date = fake.date_this_year() if status == 'Placed' else datetime.now()

            self.db.cursor.execute("""
                INSERT INTO placements VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                pid, pid,
                random.randint(40, 100),
                random.randint(0, 3),
                status, company, package, rounds, date
            ))
        self.db.commit()

    def export_csv(self):
        df = self.db.fetch_df("SELECT * FROM placements")
        df.to_csv("placements.csv", index=False)
        print("✅ placements.csv saved!")


def main():#Creates a DB connection
    db = DBConnector()
    try:
        student_gen = StudentGenerator(db)
        prog_gen = ProgrammingGenerator(db)
        soft_gen = SoftSkillsGenerator(db)
        place_gen = PlacementsGenerator(db)

        # Generate data
        student_gen.generate()
        prog_gen.generate()
        soft_gen.generate()
        place_gen.generate()

        print("✅ 500 records inserted into all tables!")

        # Export data
        student_gen.export_csv()
        prog_gen.export_csv()
        soft_gen.export_csv()
        place_gen.export_csv()
        print("📁 All CSV files saved successfully!")

    finally:
        db.close()

if __name__ == "__main__": #script entry point
    main()
