# face_utils.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

import mediapipe as mp


# -----------------------------
# MediaPipe singletons
# -----------------------------
_mp_fd = mp.solutions.face_detection
_mp_fm = mp.solutions.face_mesh

_face_detector = _mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.6)

_face_mesh = _mp_fm.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# simple temporal smoothing (per "track id" built from bbox)
_EMO_SMOOTH: Dict[str, Dict[str, Any]] = {}  # {"key": {"emo": str, "ttl": int}}


# -----------------------------
# Utilities
# -----------------------------
def draw_box(
    img_bgr: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: Tuple[int, int, int],
    label: str = "",
) -> None:
    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (x, y - th - 10), (x + tw + 8, y), color, -1)
        cv2.putText(
            img_bgr,
            label,
            (x + 4, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def _safe_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


def _lm_to_px(lm, w: int, h: int) -> Tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def _dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


# -----------------------------
# Face embedding (NO dlib)
# -----------------------------
def compute_embedding(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0

    dct = cv2.dct(gray)
    dct_block = dct[:16, :16].flatten()  # 256

    small = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA).flatten()  # 576

    vec = np.concatenate([dct_block, small], axis=0).astype(np.float32)
    vec /= (np.linalg.norm(vec) + 1e-8)
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def match_face(
    embedding: np.ndarray,
    known_faces: List[Dict[str, Any]],
    similarity_threshold: float = 0.78,
) -> Tuple[bool, Optional[Dict[str, Any]], float]:
    best = None
    best_sim = -1.0

    for k in known_faces:
        emb_k = k.get("embedding")
        if emb_k is None:
            continue

        if isinstance(emb_k, (list, tuple)):
            emb_k = np.array(emb_k, dtype=np.float32)
        elif isinstance(emb_k, (bytes, bytearray)):
            emb_k = np.frombuffer(emb_k, dtype=np.float32)

        if not isinstance(emb_k, np.ndarray) or emb_k.size == 0:
            continue

        sim = _cosine_similarity(embedding, emb_k.astype(np.float32))
        if sim > best_sim:
            best_sim = sim
            best = k

    if best is not None and best_sim >= similarity_threshold:
        return True, best, best_sim
    return False, None, best_sim


# -----------------------------
# Emotion (more sensitive)
# -----------------------------
def _emotion_from_landmarks(face_landmarks, img_w: int, img_h: int) -> str:
    """
    More sensitive heuristics to get Happy/Sad/Angry instead of Neutral.
    Uses:
      - smile curvature (mouth corners vs lips center)
      - mouth openness
      - brow tension (brows lower & closer)
    """
    lms = face_landmarks.landmark
    n = len(lms)

    def get(idx: int) -> Optional[Tuple[int, int]]:
        if 0 <= idx < n:
            return _lm_to_px(lms[idx], img_w, img_h)
        return None

    # stable FaceMesh ids
    p_mL = get(61)    # mouth left
    p_mR = get(291)   # mouth right
    p_u = get(13)     # upper lip
    p_l = get(14)     # lower lip
    p_n = get(1)      # nose ref
    p_eL = get(33)    # left eye outer
    p_eR = get(263)   # right eye outer

    # brows
    p_bL = get(65)
    p_bR = get(295)
    # inner brows (tension indicator)
    p_biL = get(55)
    p_biR = get(285)

    needed = [p_mL, p_mR, p_u, p_l, p_n, p_eL, p_eR, p_bL, p_bR, p_biL, p_biR]
    if any(p is None for p in needed):
        return "Neutral"

    face_w = max(_dist(p_eL, p_eR), 1.0)

    mouth_w = _dist(p_mL, p_mR)
    mouth_open = _dist(p_u, p_l)

    # smile curvature: compare mouth corners y vs mouth center y
    mouth_center_y = (p_u[1] + p_l[1]) / 2.0
    corners_y = (p_mL[1] + p_mR[1]) / 2.0
    smile_curve = (mouth_center_y - corners_y) / face_w
    # >0 means corners are higher than center (smile)

    # corners relative to nose as extra signal
    corner_lift = (p_n[1] - corners_y) / face_w

    # brow drop relative to nose
    brow_y = (p_bL[1] + p_bR[1]) / 2.0
    brow_drop = (brow_y - p_n[1]) / face_w

    # brow inner distance (smaller => tension, often angry)
    brow_inner_dist = _dist(p_biL, p_biR) / face_w

    # normalized
    mouth_w_r = mouth_w / face_w
    mouth_open_r = mouth_open / face_w

    # ---- more permissive thresholds ----
    # Happy: visible smile curve OR lifted corners + wide mouth
    if (smile_curve > 0.020 and mouth_w_r > 0.36) or (corner_lift > 0.030 and mouth_w_r > 0.40):
        return "Happy"

    # Angry: brows lower + inner brows closer + mouth not too open
    if brow_drop > 0.10 and brow_inner_dist < 0.38 and mouth_open_r < 0.10:
        return "Angry"

    # Sad: corners droop / no smile + mouth not wide
    if (smile_curve < 0.005 and corner_lift < 0.018 and mouth_w_r < 0.40) or (corner_lift < 0.012):
        return "Sad"

    return "Neutral"


def _smooth_emotion(key: str, raw: str) -> str:
    """
    Prevents flicker and helps 'Neutral-only' issues:
    - if we already had an emotion recently, keep it briefly unless strong change.
    """
    # decay ttl for all
    dead = []
    for k, v in _EMO_SMOOTH.items():
        v["ttl"] -= 1
        if v["ttl"] <= 0:
            dead.append(k)
    for k in dead:
        _EMO_SMOOTH.pop(k, None)

    prev = _EMO_SMOOTH.get(key)
    if prev is None:
        _EMO_SMOOTH[key] = {"emo": raw, "ttl": 8}
        return raw

    # If raw is Neutral but we had something else, keep old a bit
    if raw == "Neutral" and prev["emo"] in ("Happy", "Sad", "Angry"):
        prev["ttl"] = 6
        return prev["emo"]

    # If change, accept it but stabilize
    prev["emo"] = raw
    prev["ttl"] = 8
    return raw


# -----------------------------
# Main API used by app.py
# -----------------------------
def detect_faces_and_emotions(
    frame_bgr: np.ndarray,
    known_faces: List[Dict[str, Any]],
    mode: str = "single",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    try:
        img_h, img_w = frame_bgr.shape[:2]
        processed = frame_bgr.copy()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        fd_res = _face_detector.process(rgb)
        bboxes: List[Tuple[int, int, int, int]] = []

        if fd_res.detections:
            for det in fd_res.detections:
                r = det.location_data.relative_bounding_box
                x = _safe_int(r.xmin * img_w, 0, img_w - 1)
                y = _safe_int(r.ymin * img_h, 0, img_h - 1)
                w = _safe_int(r.width * img_w, 1, img_w - x)
                h = _safe_int(r.height * img_h, 1, img_h - y)
                bboxes.append((x, y, w, h))

        if not bboxes:
            return processed, []

        if mode == "single":
            bboxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            bboxes = bboxes[:1]

        fm_res = _face_mesh.process(rgb)
        mesh_faces = fm_res.multi_face_landmarks or []

        detections: List[Dict[str, Any]] = []

        for (x, y, w, h) in bboxes:
            face_crop = frame_bgr[y : y + h, x : x + w]
            if face_crop.size == 0:
                continue

            emb = compute_embedding(face_crop)
            is_reg, best, sim = match_face(emb, known_faces, similarity_threshold=0.78)

            if is_reg and best is not None:
                name = str(best.get("name", "Registered"))
                emoji = str(best.get("emoji", "🙂"))
                student_id = best.get("student_id", None)
                registered = True
                box_color = (60, 220, 120)  # green
                label = f"{name} {emoji}"
            else:
                name = "Unregistered"
                emoji = "❔"
                student_id = None
                registered = False
                box_color = (80, 80, 255)  # red
                label = "Unregistered ❌"

            # Emotion
            raw_emotion = "Neutral"
            if len(mesh_faces) >= 1:
                raw_emotion = _emotion_from_landmarks(mesh_faces[0], img_w, img_h)

            # Smooth using bbox key
            key = f"{x//20}-{y//20}-{w//20}-{h//20}"
            emotion = _smooth_emotion(key, raw_emotion)

            draw_box(processed, x, y, w, h, box_color, f"{label} • {emotion}")

            detections.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "registered": registered,
                    "student_id": student_id,
                    "name": name if registered else "Unregistered",
                    "emoji": emoji if registered else "❔",
                    "emotion": emotion,
                    "similarity": float(sim),
                }
            )

        return processed, detections

    except Exception as e:
        print(f"Error in detect_faces_and_emotions: {e}")
        return frame_bgr, []
