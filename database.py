import os
import sqlite3
import datetime
from typing import List, Dict, Any

import numpy as np

# Path to the SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), "smart_attend.db")


# ─────────────────────────────────────────
# DB INIT
# ─────────────────────────────────────────
def init_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Students table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            emoji TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    # Face embeddings table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        )
        """
    )

    # Attendance table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            emotion TEXT,
            emoji TEXT,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        )
        """
    )

    conn.commit()
    conn.close()


# ─────────────────────────────────────────
# HELPERS FOR EMBEDDINGS
# ─────────────────────────────────────────
def _embedding_to_blob(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr


# ─────────────────────────────────────────
# STUDENTS / FACES
# ─────────────────────────────────────────
def add_student(name: str, emoji: str, db_path: str = DB_PATH) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    created_at = datetime.datetime.now().isoformat(timespec="seconds")

    cur.execute(
        """
        INSERT INTO students (name, emoji, created_at)
        VALUES (?, ?, ?)
        """,
        (name, emoji, created_at),
    )

    student_id = cur.lastrowid
    conn.commit()
    conn.close()
    return student_id


def add_face_embedding(student_id: int, embedding: np.ndarray, db_path: str = DB_PATH) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    created_at = datetime.datetime.now().isoformat(timespec="seconds")
    blob = _embedding_to_blob(embedding)

    cur.execute(
        """
        INSERT INTO faces (student_id, embedding, created_at)
        VALUES (?, ?, ?)
        """,
        (student_id, blob, created_at),
    )

    face_id = cur.lastrowid
    conn.commit()
    conn.close()
    return face_id


def get_all_embeddings(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Returns a list of:
    {
        "face_id": int,
        "student_id": int,
        "name": str,
        "emoji": str,
        "embedding": np.ndarray
    }
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT 
            f.id,
            f.student_id,
            f.embedding,
            s.name,
            s.emoji
        FROM faces f
        JOIN students s ON f.student_id = s.id
        """
    )

    rows = cur.fetchall()
    conn.close()

    results: List[Dict[str, Any]] = []
    for face_id, student_id, emb_blob, name, emoji in rows:
        emb = _blob_to_embedding(emb_blob)
        results.append(
            {
                "face_id": face_id,
                "student_id": student_id,
                "name": name,
                "emoji": emoji or "🙂",
                "embedding": emb,
            }
        )

    return results


# ─────────────────────────────────────────
# ATTENDANCE
# ─────────────────────────────────────────
def log_attendance(
    student_id: int,
    emotion: str,
    emoji: str,
    db_path: str = DB_PATH,
) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    ts = datetime.datetime.now().isoformat(timespec="seconds")

    cur.execute(
        """
        INSERT INTO attendance (student_id, timestamp, emotion, emoji)
        VALUES (?, ?, ?, ?)
        """,
        (student_id, ts, emotion, emoji),
    )

    att_id = cur.lastrowid
    conn.commit()
    conn.close()
    return att_id


def get_recent_attendance(limit: int = 30, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT 
            a.timestamp,
            s.name,
            s.emoji,
            a.emotion
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        ORDER BY a.timestamp DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cur.fetchall()
    conn.close()

    results: List[Dict[str, Any]] = []
    for ts, name, emoji, emotion in rows:
        results.append(
            {
                "timestamp": ts,
                "name": name,
                "emoji": emoji or "",
                "emotion": emotion or "",
            }
        )

    return results
