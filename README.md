# SmartAttend – Face + Emotion Attendance App

SmartAttend is a desktop app that takes **attendance using face recognition and emotion detection**.

It uses the laptop camera to:
- Detect faces live
- Recognize **registered students**
- Detect basic emotions (**happy / neutral / sad / angry**)
- Mark attendance (optionally sending it to an **n8n webhook**)

The app is designed with a **macOS-style dark UI** using CustomTkinter.

---

## ✨ Features

- 🎥 **Live camera preview**  
  Start / stop camera with buttons, large preview in the center.

- 🧑‍🎓 **Student registration**
  - Type a **student name**
  - Capture a face crop from the camera
  - Store face embedding + name in the local database
  - Only **registered faces** are accepted as known students

- 🧠 **Face recognition**
  - Compares the current face embedding with the stored ones
  - If distance is small → recognized as that student  
  - Otherwise → marked as **“Unregistered”**

- 😊 **Emotion detection**
  - Detects the dominant emotion and shows it in the UI
  - Supports at least: `happy`, `neutral`, `sad`, `angry`
  - Emotion appears next to the detected student

- 🏫 **Standard vs Classroom mode**
  - **Standard** – mark attendance one by one
  - **Classroom** – better suited for quickly marking many students

- 📡 **Optional n8n integration**
  - Toggle “Send to n8n”
  - When enabled, attendance entries are POSTed to an n8n webhook URL
  - (The URL can be configured in `api_client.py`)

- 🕒 **Attendance history**
  - Recent attendance is shown in the sidebar
  - Uses a local SQLite database (`smart_attend.db`)

---

## 🧩 Tech Stack

- Python 3.11
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) – modern Tk UI
- OpenCV (`cv2`) – camera + basic image processing
- NumPy – vector math
- A small neural network / model for face embeddings & emotion detection
- SQLite – local database
- Requests – for sending data to n8n

---

## 🛠 Installation

These instructions work on **macOS** and **Windows** (with Python 3.11 installed).

### 1. Clone or extract the project

If using ZIP:
1. Download the project ZIP
2. Extract it (for example to `Desktop/mega_project`)
3. You should end up with:

   ```text
   mega_project/
     smart_attend/
       app.py
       ...
       requirements.txt
