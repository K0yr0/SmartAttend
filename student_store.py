# student_store.py
import json
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
STUDENTS_FILE = DATA_DIR / "students.json"

def load_students():
    if not STUDENTS_FILE.exists():
        return {}
    with open(STUDENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_students(students):
    with open(STUDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(students, f, indent=2)

def add_student_record(student_id, name, image_path):
    students = load_students()
    students[student_id] = {
        "name": name,
        "image_path": image_path
    }
    save_students(students)
    return students
