import sqlite3
import os
from datetime import datetime

DB_PATH = "data/experiments.db"

def create_database():
    os.makedirs("data", exist_ok=True)
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            exp_id          TEXT PRIMARY KEY,
            timestamp       TEXT,
            n_estimators    INTEGER,
            max_depth       INTEGER,
            learning_rate   REAL,
            accuracy        REAL,
            auc_score       REAL,
            time_seconds    REAL,
            status          TEXT,
            failure_reason  TEXT
        )
    """)
    conn.commit()
    conn.close()
    print(f" Database ready at: {DB_PATH}")


def save_experiment(exp_id, n_estimators, max_depth,
                    learning_rate, accuracy, auc_score,
                    time_seconds, status, failure_reason=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO experiments
        (exp_id, timestamp, n_estimators, max_depth,
         learning_rate, accuracy, auc_score,
         time_seconds, status, failure_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        exp_id, timestamp, n_estimators, max_depth,
        learning_rate, accuracy, auc_score,
        time_seconds, status, failure_reason
    ))
    conn.commit()
    conn.close()


def load_all_experiments():
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("SELECT * FROM experiments", conn)
    conn.close()
    return df


def get_summary():
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM experiments")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM experiments WHERE status='success'")
    successful = cursor.fetchone()[0]

    cursor.execute("SELECT MAX(auc_score) FROM experiments WHERE status='success'")
    best_auc = cursor.fetchone()[0]

    conn.close()

    print(f"\n====== DATABASE SUMMARY ======")
    print(f"Total experiments : {total}")
    print(f"Successful        : {successful}")
    print(f"Failed            : {total - successful}")
    if best_auc:
        print(f"Best AUC          : {best_auc}")


if __name__ == "__main__":
    create_database()
    get_summary()