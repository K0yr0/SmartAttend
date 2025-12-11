# face_utils.py
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple

# --- Haar cascades ---
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

SMILE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)


def detect_faces(gray_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces on a gray frame. Returns list of (x, y, w, h)."""
    faces = FACE_CASCADE.detectMultiScale(
        gray_frame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return list(faces)


def compute_embedding(face_img: np.ndarray) -> np.ndarray:
    """
    Simple embedding: resize to 32x32 gray, standardize, then L2-normalize.
    No dlib / deep model needed.
    """
    if len(face_img.shape) == 3:
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_img

    resized = cv2.resize(face_gray, (32, 32))  # 32x32 -> 1024 dims
    vec = resized.flatten().astype("float32")

    # standardize
    vec = vec - float(np.mean(vec))
    std = float(np.std(vec)) + 1e-6
    vec = vec / std

    # L2 normalize
    norm = float(np.linalg.norm(vec)) + 1e-6
    vec = vec / norm
    return vec


def match_embedding(
    emb: np.ndarray,
    known_faces: List[Dict[str, Any]],
    threshold: float = 0.68,  # slightly loosened
) -> Tuple[bool, Dict[str, Any] | None, float]:
    """
    Compare emb to all known embeddings. Uses cosine similarity.
    If best similarity >= threshold, return (True, best_face, best_sim).
    Otherwise (False, None, best_sim).
    """
    if not known_faces:
        return False, None, 0.0

    sims: List[float] = []
    for f in known_faces:
        kemb = f["embedding"]
        sim = float(np.dot(emb, kemb))  # embeddings are normalized
        sims.append(sim)

    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]
    if best_sim >= threshold:
        return True, known_faces[best_idx], best_sim
    else:
        return False, None, best_sim


def _rule_based_emotion(face_img: np.ndarray) -> str:
    """
    Lightweight emotion heuristic using:
    - brightness & contrast
    - upper vs lower face brightness (mouth/eyebrow hint)
    - smile & eye Haar cascades

    Returns label with emoji, e.g. "Happy 😊".
    """
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    gray = cv2.resize(gray, (120, 120))
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    upper = gray[:60, :]
    lower = gray[60:, :]

    # positive -> lower part brighter / more open
    mouth_open = float(np.mean(lower) - np.mean(upper))

    # --- Detect eyes & smile using cascades ---

    eyes_roi = upper
    eyes = EYE_CASCADE.detectMultiScale(
        eyes_roi,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(12, 12),
    )

    smile_roi = lower
    smiles = SMILE_CASCADE.detectMultiScale(
        smile_roi,
        scaleFactor=1.5,
        minNeighbors=12,
        minSize=(15, 15),
    )

    num_eyes = len(eyes)
    num_smiles = len(smiles)

    # ---- Heuristic rules ----
    # Priority order: Happy > Surprised > Angry > Sad > Fear > Disgust > Neutral

    # 1) Happy: clear smile, or bright face with open mouth
    if num_smiles > 0 or (mouth_open > 1.5 and brightness > 65):
        return "Happy 😊"

    # 2) Surprised: wide eyes + high contrast + mouth_open
    if num_eyes >= 2 and contrast > 55 and mouth_open > 1:
        return "Surprised 😮"

    # 3) Angry: make this easier to trigger (frowny, tense face)
    #   - mouth_open slightly negative (lower part darker)
    #   - decent contrast
    #   - not super dark overall
    if mouth_open < -0.4 and contrast > 40 and 50 < brightness < 170:
        return "Angry 😠"

    # 4) Sad: darker / low-energy face, not too much mouth movement
    if brightness < 115 and contrast < 55 and mouth_open < 0.5:
        return "Sad 😢"

    # 5) Fear: high contrast, slightly dark, mouth a bit open
    if contrast > 60 and brightness < 120 and mouth_open > 0.5:
        return "Fear 😱"

    # 6) Disgust: keep this quite rare
    if brightness < 70 and mouth_open < -8:
        return "Disgust 🤢"

    # 7) Default
    return "Neutral 😐"


def detect_faces_and_emotions(
    frame_bgr: np.ndarray,
    known_faces: List[Dict[str, Any]],
    mode: str = "single",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Detect faces, calculate embeddings & match to known faces.
    Also estimate emotion heuristically.

    Returns:
        processed_frame_bgr, detections

        detections: list of dicts
            {
              "x", "y", "w", "h",
              "name",
              "emoji",
              "label_text",   # e.g. "Kayra 😊 • Happy"
              "emotion",      # e.g. "Happy 😊"
              "student_id",   # or None
              "registered",   # bool
            }
    """
    frame_out = frame_bgr.copy()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    detections: List[Dict[str, Any]] = []

    if not len(faces):
        return frame_out, detections

    # For "single" mode, pick the largest face; for "classroom", keep all.
    if mode == "single":
        areas = [w * h for (x, y, w, h) in faces]
        idx = int(np.argmax(areas))
        faces_to_use = [faces[idx]]
    else:
        faces_to_use = faces

    for (x, y, w, h) in faces_to_use:
        x, y, w, h = int(x), int(y), int(w), int(h)
        face_crop = frame_bgr[max(y, 0) : y + h, max(x, 0) : x + w]
        if face_crop.size == 0:
            continue

        emb = compute_embedding(face_crop)
        registered, best_face, sim = match_embedding(emb, known_faces)

        emotion = _rule_based_emotion(face_crop)

        if registered and best_face is not None:
            name = best_face["name"]
            emoji = best_face.get("emoji") or "🙂"
            student_id = best_face["student_id"]
            label_text = f"{name} {emoji}  •  {emotion}  ({sim:.2f})"
            color = (0, 200, 0)  # green box
        else:
            name = "Unregistered"
            emoji = "❔"
            student_id = None
            label_text = f"Unregistered {emoji}  •  {emotion}"
            color = (0, 0, 255)  # red box

        # Draw rectangle & label
        cv2.rectangle(frame_out, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame_out, (x, y - 24), (x + w, y), color, -1)
        cv2.putText(
            frame_out,
            label_text,
            (x + 4, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        detections.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "name": name,
                "emoji": emoji,
                "label_text": label_text,
                "emotion": emotion,
                "student_id": student_id,
                "registered": registered,
            }
        )

    return frame_out, detections
