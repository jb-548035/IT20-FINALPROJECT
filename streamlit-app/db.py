"""
db.py
-----
Manages a local SQLite database (streamlit_predictions.db) that stores
every prediction made through the Streamlit app.

This allows the app to:
  - Save predictions with a timestamp for record-keeping
  - Show a history table of past predictions
  - Support simple analytics (class distribution, confidence trends)
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

# ─── Database file path ───────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "streamlit_predictions.db"


def get_connection() -> sqlite3.Connection:
    """
    Opens a connection to the SQLite database.
    Creates the file if it does not yet exist.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row   # Rows behave like dicts
    return conn


def init_db():
    """
    Creates the predictions table if it doesn't already exist.
    Call this once at app startup.
    """
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            predicted_class  TEXT    NOT NULL,
            confidence       REAL    NOT NULL,
            probabilities    TEXT    NOT NULL,   -- JSON string
            model_used       TEXT    NOT NULL,
            input_features   TEXT    NOT NULL    -- JSON string of raw inputs
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(
    predicted_class : str,
    confidence      : float,
    probabilities   : dict,
    model_used      : str,
    input_features  : dict
) -> int:
    """
    Inserts one prediction record into the database.

    Parameters
    ----------
    predicted_class : str   — 'Online', 'Store', or 'Hybrid'
    confidence      : float — probability of the winning class
    probabilities   : dict  — {class: probability} for all classes
    model_used      : str   — model name string
    input_features  : dict  — raw input values from the form

    Returns
    -------
    int — the newly inserted row ID
    """
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO predictions
            (timestamp, predicted_class, confidence, probabilities, model_used, input_features)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            predicted_class,
            confidence,
            json.dumps(probabilities),
            model_used,
            json.dumps(input_features)
        )
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_all_predictions(limit: int = 200) -> list[dict]:
    """
    Retrieves the most recent predictions from the database.

    Parameters
    ----------
    limit : int — max number of records to return (default 200)

    Returns
    -------
    list of dicts, newest first
    """
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT id, timestamp, predicted_class, confidence, probabilities, model_used
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        r = dict(row)
        r["probabilities"] = json.loads(r["probabilities"])
        results.append(r)
    return results


def get_summary_stats() -> dict:
    """
    Returns aggregate statistics for the history page.
    """
    conn = get_connection()

    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    class_counts = conn.execute(
        """
        SELECT predicted_class, COUNT(*) as count
        FROM predictions
        GROUP BY predicted_class
        """
    ).fetchall()

    avg_confidence = conn.execute(
        "SELECT AVG(confidence) FROM predictions"
    ).fetchone()[0]

    conn.close()

    return {
        "total_predictions" : total,
        "class_distribution": {row["predicted_class"]: row["count"] for row in class_counts},
        "avg_confidence"    : round(avg_confidence, 4) if avg_confidence else 0
    }


def delete_all_predictions():
    """
    Clears all predictions from the database.
    Used for the 'Reset History' button.
    """
    conn = get_connection()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()