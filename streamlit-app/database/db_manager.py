import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "travel_mind.db")

def init_db():
    """Initializes the database and creates the predictions table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            plan_name TEXT,
            recommendation TEXT,
            match_confidence REAL,
            age INTEGER,
            gender TEXT,
            adults INTEGER,
            children INTEGER,
            budget TEXT,
            month TEXT,
            prefs TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(plan_name, recommendation, match_confidence, age, gender, adults, children, budget, month, prefs):
    """Saves a prediction record to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            plan_name, recommendation, match_confidence, age, gender, adults, children, budget, month, prefs
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (plan_name, recommendation, match_confidence, age, gender, adults, children, budget, month, "; ".join(prefs)))
    conn.commit()
    conn.close()

def get_history(limit=50):
    """Retrieves the prediction history from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, plan_name, recommendation, match_confidence, age, budget, month FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "Time": row[0],
            "Plan Name": row[1],
            "Recommendation": row[2],
            "Match": f"{row[3]:.1f}%",
            "Age": row[4],
            "Budget": row[5],
            "Month": row[6]
        })
    return history

def clear_history():
    """Clears all prediction records from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
