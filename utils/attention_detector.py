"""
Attentiveness detection engine — MediaPipe 0.10.x Tasks API (FaceLandmarker).

v1.7 changes
  - Event debouncing is TIME-based (seconds), so results no longer depend on
    webcam fps or on how many frames a video is sampled at.
  - `process_frame(frame, ts=None)`: pass the video timestamp for uploaded
    videos so durations/timelines reflect video time, not processing time.
  - "No face" is counted as events (a continuous absence = 1 event), and the
    raw frame count is kept separately as `no_face_frames`.
  - Class-wide attentiveness uses a majority rule (>= half of visible
    students attentive) instead of "every student at once", so the score no
    longer collapses as more students appear.
  - Eye-closure is ignored while the head is pitched down/up (looking at a
    desk shrinks the eye-aspect ratio and is not drowsiness).
  - Students seen for fewer than MIN_STUDENT_FRAMES frames are treated as
    tracking noise and left out of the per-student table / student count.
  - Timeline in the summary is bucketed to ~1 point per second.
"""

import cv2
import numpy as np
import math
import time
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import mediapipe as mp
from mediapipe.tasks.python.core import base_options as mp_base
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode,
)

from utils.timeline import bucket_timeline

# ── Model auto-download (a copy is also bundled in /models) ──────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")


def _ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("[ClassPulse] Downloading face_landmarker.task (~4 MB) ...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            raise RuntimeError(
                f"Could not download MediaPipe model: {e}\n"
                f"Download it manually from:\n  {MODEL_URL}\n"
                f"and place it at: {os.path.abspath(MODEL_PATH)}"
            )
    return MODEL_PATH


# ── Landmark indices ──────────────────────────────────────────────────────────
LEFT_EYE_TOP      = [159, 160, 161]
LEFT_EYE_BOTTOM   = [144, 145, 163]
LEFT_EYE_CORNERS  = [33, 133]
RIGHT_EYE_TOP     = [386, 387, 388]
RIGHT_EYE_BOTTOM  = [373, 374, 380]
RIGHT_EYE_CORNERS = [263, 362]

MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0,  -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)
POSE_LM_IDS = [1, 152, 263, 33, 287, 57]

# ── Thresholds ────────────────────────────────────────────────────────────────
YAW_THRESHOLD    = 25
PITCH_THRESHOLD  = 22
EYE_OPEN_MIN     = 0.21     # scale-invariant EAR threshold (open eyes ~0.28-0.35)
EAR_SMOOTH_N     = 4
MAX_FACES        = 8

EYE_CLOSE_SECONDS  = 0.4    # eyes must stay closed this long to count (blinks are shorter)
POSE_EVENT_SECONDS = 0.5    # head turned / head down must persist this long to log an event
NO_FACE_SECONDS    = 1.0    # absence must last this long to count as a "no face" event
MIN_STUDENT_FRAMES = 15     # fewer frames than this = tracking noise, not a student


def frame_is_attentive(states: list) -> bool:
    """Class-wide verdict for one frame: at least half of visible students attentive."""
    if not states:
        return False
    return sum(1 for s in states if s.is_attentive) * 2 >= len(states)


# ── Face identity tracker ─────────────────────────────────────────────────────
class _FaceTracker:
    """
    Centroid tracker in normalised [0,1] image coordinates. All detections in a
    frame are matched against all known slots in one greedy nearest-first pass,
    so two faces in the same frame never take the same ID, and a student who
    leaves and returns near the same seat gets their old ID back.
    """
    def __init__(self, dist_threshold: float = 0.22):
        # dist_threshold: max Euclidean distance in normalised (x, y) units
        self._slots: dict = {}
        self._next_id = 1
        self._dist_thresh = dist_threshold

    def update_batch(self, centroids: list) -> list:
        n = len(centroids)
        assigned = [None] * n

        if self._slots and n:
            candidates = []
            for i, (cx, cy) in enumerate(centroids):
                for sid, (ex, ey) in self._slots.items():
                    d = math.hypot(cx - ex, cy - ey)
                    if d <= self._dist_thresh:
                        candidates.append((d, i, sid))
            candidates.sort(key=lambda t: t[0])

            used_dets, used_slots = set(), set()
            for d, i, sid in candidates:
                if i in used_dets or sid in used_slots:
                    continue
                assigned[i] = sid
                used_dets.add(i)
                used_slots.add(sid)

        for i, (cx, cy) in enumerate(centroids):
            if assigned[i] is None:
                assigned[i] = self._next_id
                self._next_id += 1
            sid = assigned[i]
            if sid in self._slots:
                ex, ey = self._slots[sid]
                self._slots[sid] = (ex * 0.7 + cx * 0.3, ey * 0.7 + cy * 0.3)
            else:
                self._slots[sid] = (cx, cy)

        return assigned

    def reset(self):
        self._slots.clear()
        self._next_id = 1


@dataclass
class AttentionState:
    is_attentive: bool    = True
    confidence: float     = 1.0
    head_yaw: float       = 0.0
    head_pitch: float     = 0.0
    eye_open_ratio: float = 1.0
    reason: str           = "Attentive"
    face_detected: bool   = False
    student_id: int       = 1


@dataclass
class _StudentRecord:
    student_id: int
    frames_seen: int        = 0
    attentive_frames: int   = 0
    distracted_frames: int  = 0
    eyes_closed_events: int = 0
    head_turned_events: int = 0
    head_down_events: int   = 0
    first_seen_ts: float    = 0.0
    last_seen_ts: float     = 0.0

    # smoothing history + time-based debounce state (per student)
    _yaw_hist: list   = field(default_factory=list, repr=False)
    _pitch_hist: list = field(default_factory=list, repr=False)
    _ear_hist: list   = field(default_factory=list, repr=False)
    _closed_since: Optional[float] = None
    _closed_logged: bool = False
    _turn_since: Optional[float] = None
    _turn_logged: bool = False
    _pitch_since: Optional[float] = None
    _pitch_logged: bool = False

    @property
    def attentiveness_score(self) -> float:
        if self.frames_seen == 0:
            return 0.0
        return round(self.attentive_frames / self.frames_seen * 100, 1)


def _track_streak(rec, name: str, active: bool, ts: float, min_seconds: float):
    """
    Time-based debounce. Returns (sustained, new_event):
      sustained  -> condition has now held for >= min_seconds
      new_event  -> True exactly once per continuous stretch, when it first becomes sustained
    """
    since_attr, logged_attr = f"_{name}_since", f"_{name}_logged"
    if not active:
        setattr(rec, since_attr, None)
        setattr(rec, logged_attr, False)
        return False, False
    since = getattr(rec, since_attr)
    if since is None:
        since = ts
        setattr(rec, since_attr, ts)
    sustained = (ts - since) >= min_seconds
    new_event = sustained and not getattr(rec, logged_attr)
    if new_event:
        setattr(rec, logged_attr, True)
    return sustained, new_event


@dataclass
class SessionStats:
    total_frames: int       = 0
    attentive_frames: int   = 0
    distracted_frames: int  = 0
    eyes_closed_events: int = 0
    head_turned_events: int = 0
    head_down_events: int   = 0
    no_face_events: int     = 0
    no_face_frames: int     = 0
    timeline: list          = field(default_factory=list)
    start_time: float       = field(default_factory=time.time)
    last_ts: float          = 0.0     # seconds of session/video time covered so far

    @property
    def attentiveness_score(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return round(self.attentive_frames / self.total_frames * 100, 1)

    @property
    def elapsed_minutes(self) -> float:
        return round(self.last_ts / 60, 2)


class AttentionDetector:
    def __init__(
        self,
        yaw_thresh: float   = YAW_THRESHOLD,
        pitch_thresh: float = PITCH_THRESHOLD,
        eye_min: float      = EYE_OPEN_MIN,
        max_faces: int      = MAX_FACES,
    ):
        model_path = _ensure_model()
        options = FaceLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.session    = SessionStats()

        self._yaw_thresh   = yaw_thresh
        self._pitch_thresh = pitch_thresh
        self._eye_min      = eye_min
        self._smooth_n     = 5
        self._ear_smooth_n = EAR_SMOOTH_N

        self._students: dict = {}
        self._tracker = _FaceTracker(dist_threshold=0.22)
        self._no_face_since: Optional[float] = None
        self._no_face_logged = False

    def reset_session(self):
        self.session = SessionStats()
        self._students.clear()
        self._tracker.reset()
        self._no_face_since = None
        self._no_face_logged = False

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _eye_open_ratio(self, lm, ids_top, ids_bot, ids_corners, h, w) -> float:
        top_y = np.mean([lm[i].y * h for i in ids_top])
        bot_y = np.mean([lm[i].y * h for i in ids_bot])
        eye_height = abs(bot_y - top_y)
        cx0, cy0 = lm[ids_corners[0]].x * w, lm[ids_corners[0]].y * h
        cx1, cy1 = lm[ids_corners[1]].x * w, lm[ids_corners[1]].y * h
        eye_width = math.hypot(cx1 - cx0, cy1 - cy0)
        return eye_height / (eye_width + 1e-6)

    def _head_pose(self, lm, h, w):
        img_pts = np.array(
            [(lm[i].x * w, lm[i].y * h) for i in POSE_LM_IDS], dtype=np.float64,
        )
        focal   = float(w)
        cam_mat = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1))
        ok, rvec, _ = cv2.solvePnP(
            MODEL_POINTS_3D, img_pts, cam_mat, dist, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        sy    = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
        return float(yaw), float(pitch)

    def _face_centroid(self, lm) -> tuple:
        xs = [l.x for l in lm]
        ys = [l.y for l in lm]
        return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)

    def _face_bbox_px(self, lm, h, w) -> tuple:
        xs = [l.x * w for l in lm]
        ys = [l.y * h for l in lm]
        pad = 12
        return (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
                min(w, int(max(xs)) + pad), min(h, int(max(ys)) + pad))

    def _evaluate_student(self, rec: "_StudentRecord", lm, h, w, ts: float) -> AttentionState:
        state = AttentionState(student_id=rec.student_id, face_detected=True)

        # Eye open ratio (scale-invariant, smoothed)
        left_ear  = self._eye_open_ratio(lm, LEFT_EYE_TOP,  LEFT_EYE_BOTTOM,  LEFT_EYE_CORNERS,  h, w)
        right_ear = self._eye_open_ratio(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_CORNERS, h, w)
        rec._ear_hist.append((left_ear + right_ear) / 2)
        if len(rec._ear_hist) > self._ear_smooth_n:
            rec._ear_hist.pop(0)
        ear = sum(rec._ear_hist) / len(rec._ear_hist)
        state.eye_open_ratio = round(ear, 3)

        # Head pose (smoothed)
        raw_yaw, raw_pitch = self._head_pose(lm, h, w)
        rec._yaw_hist.append(raw_yaw)
        rec._pitch_hist.append(raw_pitch)
        if len(rec._yaw_hist) > self._smooth_n:
            rec._yaw_hist.pop(0)
            rec._pitch_hist.pop(0)
        yaw   = sum(rec._yaw_hist) / len(rec._yaw_hist)
        pitch = sum(rec._pitch_hist) / len(rec._pitch_hist)
        state.head_yaw   = round(yaw, 1)
        state.head_pitch = round(pitch, 1)

        reasons = []

        turned = abs(yaw) > self._yaw_thresh
        if turned:
            reasons.append(f"Head turned {'left' if yaw < 0 else 'right'}")
        _, new_turn = _track_streak(rec, "turn", turned, ts, POSE_EVENT_SECONDS)
        if new_turn:
            rec.head_turned_events += 1

        pitched = abs(pitch) > self._pitch_thresh
        if pitched:
            reasons.append(f"Head {'down' if pitch > 0 else 'up'}")
        _, new_pitch = _track_streak(rec, "pitch", pitched, ts, POSE_EVENT_SECONDS)
        if new_pitch:
            rec.head_down_events += 1

        # Eyes closed: only trusted while the head is roughly level — looking
        # down at a desk makes the eye-aspect ratio drop without being drowsiness.
        eyes_low = (ear < self._eye_min) and not pitched
        closed_sustained, new_closed = _track_streak(rec, "closed", eyes_low, ts, EYE_CLOSE_SECONDS)
        if closed_sustained:
            reasons.append("Eyes closed / drowsy")
        if new_closed:
            rec.eyes_closed_events += 1

        state.is_attentive = len(reasons) == 0
        state.reason       = ", ".join(reasons) if reasons else "Attentive"

        yaw_conf   = max(0.0, 1.0 - abs(yaw)   / (self._yaw_thresh   * 2))
        pitch_conf = max(0.0, 1.0 - abs(pitch) / (self._pitch_thresh * 2))
        eye_conf   = min(1.0, ear / (self._eye_min + 1e-6))
        state.confidence = round((yaw_conf + pitch_conf + eye_conf) / 3, 3)
        return state

    # ── Main ──────────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray, ts: Optional[float] = None):
        """
        Process one BGR frame. Returns (annotated_frame, states).
        ts: seconds since session start. Leave None for live use (wall clock);
        pass frame_index / fps for uploaded videos.
        """
        h, w = frame.shape[:2]
        if ts is None:
            ts = time.time() - self.session.start_time
        self.session.last_ts = max(self.session.last_ts, ts)
        self.session.total_frames += 1

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            self.session.no_face_frames    += 1
            self.session.distracted_frames += 1
            if self._no_face_since is None:
                self._no_face_since = ts
            if not self._no_face_logged and (ts - self._no_face_since) >= NO_FACE_SECONDS:
                self.session.no_face_events += 1
                self._no_face_logged = True
            self.session.timeline.append((ts, False))
            self._draw_top_bar(frame, [], w, h)
            return frame, []

        self._no_face_since = None
        self._no_face_logged = False

        centroids = [self._face_centroid(lm) for lm in result.face_landmarks]
        ids = self._tracker.update_batch(centroids)

        states = []
        for lm, sid in zip(result.face_landmarks, ids):
            rec = self._students.setdefault(sid, _StudentRecord(student_id=sid, first_seen_ts=ts))
            rec.frames_seen += 1
            rec.last_seen_ts = ts

            state = self._evaluate_student(rec, lm, h, w, ts)
            if state.is_attentive:
                rec.attentive_frames += 1
            else:
                rec.distracted_frames += 1
            states.append(state)
            self._draw_face_box(frame, lm, state, h, w)

        attentive = frame_is_attentive(states)
        if attentive:
            self.session.attentive_frames += 1
        else:
            self.session.distracted_frames += 1
        self.session.timeline.append((ts, attentive))

        self._draw_top_bar(frame, states, w, h)
        return frame, states

    def _draw_face_box(self, frame, lm, state: AttentionState, h, w):
        x1, y1, x2, y2 = self._face_bbox_px(lm, h, w)
        color = (52, 168, 83) if state.is_attentive else (234, 67, 53)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"S{state.student_id:02d}  {'OK' if state.is_attentive else 'DISTRACTED'}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(0, y1 - th - 10)
        cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_top_bar(self, frame, states: list, w, h):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (15, 17, 22), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        n = len(states)
        if n == 0:
            text, color = "NO FACE DETECTED", (234, 67, 53)
        else:
            n_attentive = sum(1 for s in states if s.is_attentive)
            text  = f"{n} STUDENT{'S' if n != 1 else ''} DETECTED   |   {n_attentive}/{n} ATTENTIVE"
            color = (52, 168, 83) if n_attentive == n else (251, 188, 4)
        cv2.putText(frame, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

    def get_session_summary(self) -> dict:
        s = self.session

        # Ignore very short-lived "students" (usually a false face detection).
        sids = sorted(self._students)
        real = [sid for sid in sids if self._students[sid].frames_seen >= MIN_STUDENT_FRAMES]
        sids = real or sids

        students = []
        agg_eyes = agg_turn = agg_down = 0
        for sid in sids:
            rec = self._students[sid]
            agg_eyes += rec.eyes_closed_events
            agg_turn += rec.head_turned_events
            agg_down += rec.head_down_events
            students.append({
                "student_id":          sid,
                "label":               f"S{sid:02d}",
                "frames_seen":         rec.frames_seen,
                "attentiveness_score": rec.attentiveness_score,
                "eyes_closed_events":  rec.eyes_closed_events,
                "head_turned_events":  rec.head_turned_events,
                "head_down_events":    rec.head_down_events,
            })

        return {
            "duration_minutes":    s.elapsed_minutes,
            "total_frames":        s.total_frames,
            "attentiveness_score": s.attentiveness_score,
            "attentive_frames":    s.attentive_frames,
            "distracted_frames":   s.distracted_frames,
            "eyes_closed_events":  agg_eyes,
            "head_turned_events":  agg_turn,
            "head_down_events":    agg_down,
            "no_face_events":      s.no_face_events,
            "no_face_frames":      s.no_face_frames,
            "timeline":            bucket_timeline(s.timeline, 1.0),
            "num_students":        len(students),
            "students":            students,
        }
