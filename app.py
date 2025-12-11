# app.py
import os
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

import cv2
import customtkinter as ctk
import numpy as np
import requests
from PIL import Image, ImageTk

from database import (
    DB_PATH,
    init_db,
    add_student,
    add_face_embedding,
    get_all_embeddings,
    log_attendance,
    get_recent_attendance,
)
from face_utils import detect_faces_and_emotions

# --------- CONFIG ---------
WEBHOOK_URL = "https://adam143-20143.wykr.es/webhook/attendance"
CAM_WIDTH = 960
CAM_HEIGHT = 540

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def send_attendance_to_n8n(payload: Dict[str, Any]) -> None:
    """Best-effort POST to n8n (non-blocking from UI)."""
    def _worker():
        try:
            requests.post(WEBHOOK_URL, json=payload, timeout=5)
        except Exception:
            # Silent fail – we still log locally
            pass

    threading.Thread(target=_worker, daemon=True).start()


class SmartAttendApp(ctk.CTk):

    def __init__(self) -> None:
        super().__init__()

        # Basic window config
        self.title("SmartAttend 🎓")
        self.geometry("1280x720")
        self.minsize(1100, 650)

        # DB & known faces
        init_db(DB_PATH)
        self.known_faces: List[Dict[str, Any]] = []
        self._reload_known_faces()

        # Camera state
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_running = False
        self.last_frame: Optional[np.ndarray] = None
        self.mode: str = "single"  # "single" or "classroom"
        self.last_detections: List[Dict[str, Any]] = []

        self._build_layout()

    # ---------- DB helpers ----------
    def _reload_known_faces(self) -> None:
        self.known_faces = get_all_embeddings(DB_PATH)

    # ---------- UI layout ----------
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0, minsize=320)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left panel (controls / info)
        left_frame = ctk.CTkFrame(self, corner_radius=20)
        left_frame.grid(row=0, column=0, sticky="nswe", padx=16, pady=16)
        left_frame.grid_rowconfigure(4, weight=1)

        title_label = ctk.CTkLabel(
            left_frame,
            text="SmartAttend 🎓",
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        title_label.pack(pady=(18, 6))

        subtitle_label = ctk.CTkLabel(
            left_frame,
            text="Face & emotion based attendance\nmade for classrooms 👩‍🏫",
            font=ctk.CTkFont(size=13),
        )
        subtitle_label.pack(pady=(0, 10))

        # Mode switch
        mode_label = ctk.CTkLabel(
            left_frame,
            text="Mode",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        mode_label.pack(pady=(10, 4))

        self.mode_segment = ctk.CTkSegmentedButton(
            left_frame,
            values=["Single 🧍", "Classroom 👥"],
            command=self._on_mode_change,
        )
        self.mode_segment.set("Single 🧍")
        self.mode_segment.pack(pady=(0, 12), fill="x", padx=12)

        # Register section
        reg_frame = ctk.CTkFrame(left_frame, corner_radius=16)
        reg_frame.pack(fill="x", padx=12, pady=(4, 12))

        reg_title = ctk.CTkLabel(
            reg_frame,
            text="Register student ✨",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        reg_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.name_entry = ctk.CTkEntry(
            reg_frame,
            placeholder_text="Full name (e.g. Kayra)",
        )
        self.name_entry.pack(fill="x", padx=12, pady=(0, 6))

        self.code_entry = ctk.CTkEntry(
            reg_frame,
            placeholder_text="Student code / ID",
        )
        self.code_entry.pack(fill="x", padx=12, pady=(0, 6))

        self.emoji_entry = ctk.CTkEntry(
            reg_frame,
            placeholder_text="Emoji (e.g. 😎)",
        )
        self.emoji_entry.insert(0, "🙂")
        self.emoji_entry.pack(fill="x", padx=12, pady=(0, 10))

        reg_hint = ctk.CTkLabel(
            reg_frame,
            text="Tip: look straight at the camera.\nOnly 1 face should be visible.",
            font=ctk.CTkFont(size=11),
            text_color=("gray70", "gray80"),
            justify="left",
        )
        reg_hint.pack(anchor="w", padx=12, pady=(0, 6))

        reg_btn = ctk.CTkButton(
            reg_frame,
            text="Capture & Register 📸",
            command=self._on_register_student,
        )
        reg_btn.pack(fill="x", padx=12, pady=(4, 12))

        # Attendance actions
        action_frame = ctk.CTkFrame(left_frame, corner_radius=16)
        action_frame.pack(fill="x", padx=12, pady=(0, 12))

        att_title = ctk.CTkLabel(
            action_frame,
            text="Attendance",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        att_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.mark_btn = ctk.CTkButton(
            action_frame,
            text="Mark attendance now ✅",
            command=self._on_mark_attendance,
        )
        self.mark_btn.pack(fill="x", padx=12, pady=(4, 8))

        # Recent logs
        recent_title = ctk.CTkLabel(
            left_frame,
            text="Recent attendance 📝",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        recent_title.pack(anchor="w", padx=18, pady=(4, 4))

        self.recent_box = ctk.CTkTextbox(
            left_frame,
            height=160,
            activate_scrollbars=True,
            corner_radius=12,
        )
        self.recent_box.pack(fill="both", expand=False, padx=16, pady=(0, 12))
        self.recent_box.configure(state="disabled")

        self._refresh_recent_attendance()

        # Status
        self.status_label = ctk.CTkLabel(
            left_frame,
            text="Ready. Start camera to begin 👀",
            font=ctk.CTkFont(size=12),
            text_color=("gray75", "gray80"),
            wraplength=280,
            justify="left",
        )
        self.status_label.pack(anchor="w", padx=16, pady=(4, 10))

        # Right panel (camera + info)
        right_frame = ctk.CTkFrame(self, corner_radius=20)
        right_frame.grid(row=0, column=1, sticky="nswe", padx=(0, 16), pady=16)
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Camera header
        cam_header = ctk.CTkFrame(right_frame, fg_color="transparent")
        cam_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 4))
        cam_header.grid_columnconfigure(0, weight=1)

        cam_title = ctk.CTkLabel(
            cam_header,
            text="Live camera 🎥",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        cam_title.grid(row=0, column=0, sticky="w")

        self.cam_toggle_btn = ctk.CTkButton(
            cam_header,
            text="Start camera",
            width=140,
            command=self._toggle_camera,
        )
        self.cam_toggle_btn.grid(row=0, column=1, sticky="e")

        # Camera view
        cam_frame = ctk.CTkFrame(right_frame, corner_radius=18)
        cam_frame.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        cam_frame.grid_rowconfigure(0, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        self.camera_label = ctk.CTkLabel(cam_frame, text="")
        self.camera_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Info bar under camera
        info_frame = ctk.CTkFrame(right_frame, corner_radius=16)
        info_frame.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        info_frame.grid_columnconfigure(0, weight=1)
        info_frame.grid_columnconfigure(1, weight=1)

        self.name_info_label = ctk.CTkLabel(
            info_frame,
            text="Student: —",
            font=ctk.CTkFont(size=14),
            anchor="w",
        )
        self.name_info_label.grid(row=0, column=0, sticky="w", padx=12, pady=8)

        self.emotion_info_label = ctk.CTkLabel(
            info_frame,
            text="Emotion: —",
            font=ctk.CTkFont(size=14),
            anchor="e",
        )
        self.emotion_info_label.grid(row=0, column=1, sticky="e", padx=12, pady=8)

        # Blank camera display initially
        self._show_blank_camera()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- Mode ----------
    def _on_mode_change(self, value: str) -> None:
        if "Classroom" in value:
            self.mode = "classroom"
            self.status_label.configure(
                text="Classroom mode 👥 – multiple registered faces will be accepted.",
                text_color="#9FE870",
            )
        else:
            self.mode = "single"
            self.status_label.configure(
                text="Single mode 🧍 – strongest match only.",
                text_color=("gray75", "gray80"),
            )

    # ---------- Camera control ----------
    def _toggle_camera(self) -> None:
        if self.camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self) -> None:
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.configure(
                text="Could not open camera. Check permissions & try again.",
                text_color="#FF6B6B",
            )
            self.cap = None
            return

        self.camera_running = True
        self.cam_toggle_btn.configure(text="Stop camera")
        self.status_label.configure(
            text="Camera running. Look at the camera 😄",
            text_color="#9FE870",
        )
        self._update_camera()

    def _stop_camera(self) -> None:
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cam_toggle_btn.configure(text="Start camera")
        self._show_blank_camera()
        self.name_info_label.configure(text="Student: —")
        self.emotion_info_label.configure(text="Emotion: —")
        self.last_detections = []

    def _show_blank_camera(self) -> None:
        blank = Image.new("RGB", (CAM_WIDTH, CAM_HEIGHT), (12, 12, 16))
        draw = Image.new("RGB", (CAM_WIDTH, CAM_HEIGHT), (12, 12, 16))
        img_tk = ImageTk.PhotoImage(blank)
        self.camera_label.configure(image=img_tk)
        self.camera_label.image = img_tk
        self.camera_label.configure(text="Camera stopped", font=ctk.CTkFont(size=16))

    def _update_camera(self) -> None:
        if not self.camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.configure(
                text="Lost camera signal. Stopping.",
                text_color="#FF6B6B",
            )
            self._stop_camera()
            return

        self.last_frame = frame.copy()

        processed, detections = detect_faces_and_emotions(frame, self.known_faces, self.mode)
        self.last_detections = detections

        # Update info labels from first detection if any
        if detections:
            d0 = detections[0]
            if d0["registered"]:
                self.name_info_label.configure(
                    text=f"Student: {d0['name']} {d0['emoji']}",
                )
            else:
                self.name_info_label.configure(text="Student: Unregistered ❔")
            self.emotion_info_label.configure(text=f"Emotion: {d0['emotion']}")
        else:
            self.name_info_label.configure(text="Student: —")
            self.emotion_info_label.configure(text="Emotion: —")

        # Convert to Tk image
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape

        # Keep aspect ratio, fit into CAM_WIDTH x CAM_HEIGHT
        scale = min(CAM_WIDTH / w, CAM_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h))

        canvas_img = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        x_offset = (CAM_WIDTH - new_w) // 2
        y_offset = (CAM_HEIGHT - new_h) // 2
        canvas_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        img_pil = Image.fromarray(canvas_img)
        img_tk = ImageTk.PhotoImage(img_pil)

        self.camera_label.configure(image=img_tk, text="")
        self.camera_label.image = img_tk

        # Schedule next frame
        self.after(40, self._update_camera)  # ~25 fps

    # ---------- Register ----------
    def _on_register_student(self) -> None:
        name = self.name_entry.get().strip()
        code = self.code_entry.get().strip()
        emoji = self.emoji_entry.get().strip() or "🙂"

        if not name or not code:
            self.status_label.configure(
                text="Please fill in name and student code to register.",
                text_color="#FF6B6B",
            )
            return

        if self.last_frame is None:
            self.status_label.configure(
                text="Camera must be running and showing your face to register.",
                text_color="#FF6B6B",
            )
            return

        frame = self.last_frame.copy()
        processed, detections = detect_faces_and_emotions(
            frame, self.known_faces, mode="single"
        )

        if not detections:
            self.status_label.configure(
                text="No face detected. Make sure only your face is visible.",
                text_color="#FF6B6B",
            )
            return

        # We already asked face_utils to only use largest face in "single" mode
        d = detections[0]
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        face_crop = frame[max(y, 0) : y + h, max(x, 0) : x + w]

        from face_utils import compute_embedding  # local import to avoid circular

        emb = compute_embedding(face_crop)

        try:
            student_id = add_student(name, code, emoji)
            add_face_embedding(student_id, emb)
            self._reload_known_faces()
            self.status_label.configure(
                text=f"Registered {name} {emoji} successfully 🎉",
                text_color="#9FE870",
            )
            self.name_entry.delete(0, "end")
            self.code_entry.delete(0, "end")
        except Exception as e:
            self.status_label.configure(
                text=f"Error while registering: {e}",
                text_color="#FF6B6B",
            )

    # ---------- Attendance ----------
    def _on_mark_attendance(self) -> None:
        if not self.last_detections:
            self.status_label.configure(
                text="No faces to mark. Make sure camera sees a registered student.",
                text_color="#FF6B6B",
            )
            return

        now_str = datetime.utcnow().isoformat()
        mode_label = "Classroom" if self.mode == "classroom" else "Single"

        unique_by_id: Dict[Any, Dict[str, Any]] = {}

        for d in self.last_detections:
            student_id = d["student_id"]
            name = d["name"]
            emotion = d["emotion"]

            if d["registered"]:
                key = student_id
            else:
                key = f"unreg-{id(d)}"  # unique key per unregistered detection

            if key not in unique_by_id:
                unique_by_id[key] = {
                    "student_id": student_id,
                    "name": name,
                    "emotion": emotion,
                }

        marked_list = []

        for v in unique_by_id.values():
            student_id = v["student_id"]
            name = v["name"]
            emotion = v["emotion"]

            if student_id is None:
                status = "Unregistered"
            else:
                status = "Present"

            # Log to DB
            log_attendance(
                student_id=student_id,
                name=name,
                status=status,
                emotion=emotion,
                mode=mode_label,
            )

            # Send to n8n (non-blocking)
            payload = {
                "timestamp": now_str,
                "mode": mode_label,
                "name": name,
                "status": status,
                "emotion": emotion,
            }
            send_attendance_to_n8n(payload)

            marked_list.append(f"{name} – {status} – {emotion}")

        if marked_list:
            msg = "Marked:\n" + "\n".join(marked_list)
            self.status_label.configure(
                text=msg,
                text_color="#9FE870",
            )
        else:
            self.status_label.configure(
                text="Nothing to mark.",
                text_color="#FF6B6B",
            )

        self._refresh_recent_attendance()

    def _refresh_recent_attendance(self) -> None:
        rows = get_recent_attendance(limit=10)
        self.recent_box.configure(state="normal")
        self.recent_box.delete("1.0", "end")
        for r in rows:
            line = f"{r['created_at']}  •  {r['name']}  •  {r['status']}  •  {r['emotion']}\n"
            self.recent_box.insert("end", line)
        self.recent_box.configure(state="disabled")

    # ---------- Closing ----------
    def _on_close(self) -> None:
        self._stop_camera()
        self.destroy()


if __name__ == "__main__":
    app = SmartAttendApp()
    app.mainloop()
