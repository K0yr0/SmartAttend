# database.py
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "smart_attend.db")


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they do not exist."""
    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            code TEXT UNIQUE NOT NULL,
            emoji TEXT DEFAULT '🙂',
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            name TEXT,
            status TEXT,
            emotion TEXT,
            mode TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


def add_student(name: str, code: str, emoji: str = "🙂", db_path: str = DB_PATH) -> int:
    conn = _connect(db_path)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO students (name, code, emoji, created_at) VALUES (?, ?, ?, ?)",
        (name, code, emoji, now),
    )
    student_id = cur.lastrowid
    conn.commit()
    conn.close()
    return student_id


def add_face_embedding(student_id: int, embedding: np.ndarray, db_path: str = DB_PATH) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    # store as binary
    cur.execute(
        "INSERT INTO faces (student_id, embedding, created_at) VALUES (?, ?, ?)",
        (student_id, embedding.astype("float32").tobytes(), now),
    )
    conn.commit()
    conn.close()


def get_all_embeddings(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Return list of dicts: {student_id, name, code, emoji, embedding}."""
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT s.id, s.name, s.code, s.emoji, f.embedding
        FROM students s
        JOIN faces f ON s.id = f.student_id
        """
    )
    rows = cur.fetchall()
    conn.close()

    result: List[Dict[str, Any]] = []
    for r in rows:
        student_id, name, code, emoji, emb_blob = r
        emb_array = np.frombuffer(emb_blob, dtype="float32")
        result.append(
            {
                "student_id": student_id,
                "name": name,
                "code": code,
                "emoji": emoji,
                "embedding": emb_array,
            }
        )
    return result


def log_attendance(
    student_id: Optional[int],
    name: str,
    status: str,
    emotion: str,
    mode: str,
    db_path: str = DB_PATH,
) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        """
        INSERT INTO attendance (student_id, name, status, emotion, mode, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (student_id, name, status, emotion, mode, now),
    )
    conn.commit()
    conn.close()


def get_recent_attendance(limit: int = 20, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name, status, emotion, mode, created_at
        FROM attendance
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "name": r[0],
            "status": r[1],
            "emotion": r[2],
            "mode": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]
