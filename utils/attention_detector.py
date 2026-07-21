"""
Attentiveness detection engine — compatible with MediaPipe 0.10.x+ (Tasks API).

Key improvements over v1.5:
  - Multi-face support: tracks every student visible in frame at once
    (previously hard-capped at a single face via num_faces=1).
  - Frame-wide identity matching: when several faces are visible in the
    same frame, IDs are assigned via a single greedy nearest-neighbour pass
    over ALL detections vs ALL known slots, so two students never collide
    onto the same ID in one frame. A student who leaves and re-enters is
    matched back to their original slot instead of being counted as a new
    student, as long as they reappear close to where they were last seen.
  - Per-student tracking state: eye/yaw/pitch smoothing history and event
    debouncing are now kept per student ID, not globally — so one
    student's drowsiness event doesn't get attributed to another.
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
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode,
)

# ── Model auto-download ───────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

def _ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("[ClassPulse] Downloading face_landmarker.task (~5 MB) …")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[ClassPulse] Model downloaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Could not download MediaPipe model: {e}\n"
                f"Please download manually from:\n  {MODEL_URL}\n"
                f"and place it at: {os.path.abspath(MODEL_PATH)}"
            )
    return MODEL_PATH


# ── Landmark indices ──────────────────────────────────────────────────────────
LEFT_EYE_TOP     = [159, 160, 161]
LEFT_EYE_BOTTOM  = [144, 145, 163]
LEFT_EYE_CORNERS = [33, 133]      # outer, inner corner — for width normalisation
RIGHT_EYE_TOP    = [386, 387, 388]
RIGHT_EYE_BOTTOM = [373, 374, 380]
RIGHT_EYE_CORNERS = [263, 362]    # outer, inner corner

MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0,  -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)
POSE_LM_IDS = [1, 152, 263, 33, 287, 57]

YAW_THRESHOLD   = 25
PITCH_THRESHOLD = 22
EYE_OPEN_MIN    = 0.21          # scale-invariant EAR threshold (typical open EAR ~0.28-0.35)
EYE_CLOSE_FRAMES = 6            # consecutive low-EAR frames required (~0.2-0.3s) before
                                 # counting as a real closure — filters out normal blinks
EAR_SMOOTH_N     = 4            # EMA smoothing window for EAR, same idea as yaw/pitch
MAX_FACES        = 8            # cap on simultaneous faces tracked per frame


# ── Face identity tracker ─────────────────────────────────────────────────────
# Keeps a registry of known face centroids so a face that leaves and re-enters
# is recognized as the same student rather than a new one, and so multiple
# students visible at once never collide onto the same ID.
class _FaceTracker:
    """
    Centroid-based multi-object tracker. Each known student is represented
    by a (cx, cy) centroid computed from the bounding box of their face
    landmarks, in coordinates normalised to [0, 1] (so it works regardless
    of frame resolution).

    Assignment is done once per frame, across ALL detections in that frame
    against ALL known slots simultaneously (greedy nearest-neighbour by
    increasing distance). This is what prevents two different students
    detected in the same frame from both matching the same existing slot —
    a per-detection "find nearest" loop (as used in earlier versions) can't
    guarantee that, since each face is matched independently of the others.
    """
    def __init__(self, dist_threshold: float = 0.22):
        """
        dist_threshold: max normalised distance (fraction of frame diagonal)
        before a detection is treated as a brand-new person rather than a
        returning one.
        """
        self._slots: dict[int, tuple] = {}   # id -> (cx, cy) in normalised coords
        self._next_id = 1
        self._dist_thresh = dist_threshold

    def update_batch(self, centroids: list) -> list:
        """
        Assign a student ID to each centroid detected in the current frame.
        Returns a list of IDs, same order/length as `centroids`.
        """
        n = len(centroids)
        assigned = [None] * n

        if self._slots and n:
            # Build every (distance, detection_index, slot_id) pair that's
            # within the match threshold, then greedily claim the closest
            # pairs first — each detection and each slot can be used once.
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

        # Anything left unmatched (new student, or no slots existed yet)
        # gets a freshly minted ID.
        for i, (cx, cy) in enumerate(centroids):
            if assigned[i] is None:
                sid = self._next_id
                self._next_id += 1
                assigned[i] = sid

            # Update (or create) the slot centroid with light EMA smoothing
            # so small frame-to-frame jitter doesn't drift the match point.
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
    """Per-student running stats and smoothing/debounce state. Not exposed
    directly — get_session_summary() flattens this into plain dicts."""
    student_id: int
    frames_seen: int        = 0
    attentive_frames: int   = 0
    distracted_frames: int  = 0
    eyes_closed_events: int = 0
    head_turned_events: int = 0
    head_down_events: int   = 0
    first_seen_ts: float    = 0.0
    last_seen_ts: float     = 0.0

    # internal smoothing/debounce state — kept per student so one
    # student's blink/turn streak never bleeds into another's count
    _yaw_hist: list   = field(default_factory=list, repr=False)
    _pitch_hist: list = field(default_factory=list, repr=False)
    _ear_hist: list   = field(default_factory=list, repr=False)
    _closed_streak: int        = 0
    _closed_event_logged: bool = False
    _turn_streak: int           = 0
    _turn_event_logged: bool    = False
    _pitch_streak: int          = 0
    _pitch_event_logged: bool   = False

    @property
    def attentiveness_score(self) -> float:
        if self.frames_seen == 0:
            return 0.0
        return round(self.attentive_frames / self.frames_seen * 100, 1)


@dataclass
class SessionStats:
    total_frames: int       = 0
    attentive_frames: int   = 0
    distracted_frames: int  = 0
    eyes_closed_events: int = 0
    head_turned_events: int = 0
    head_down_events: int   = 0
    no_face_events: int     = 0
    timeline: list          = field(default_factory=list)
    start_time: float       = field(default_factory=time.time)

    @property
    def attentiveness_score(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return round(self.attentive_frames / self.total_frames * 100, 1)

    @property
    def elapsed_minutes(self) -> float:
        return round((time.time() - self.start_time) / 60, 2)


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
            min_tracking_confidence=0.6,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.session    = SessionStats()

        self._yaw_thresh   = yaw_thresh
        self._pitch_thresh = pitch_thresh
        self._eye_min      = eye_min
        self._smooth_n      = 5
        self._ear_smooth_n  = EAR_SMOOTH_N

        # One record per distinct student seen this session, keyed by the
        # stable ID handed out by _FaceTracker.
        self._students: dict = {}

        # Identity persistence tracker
        self._tracker = _FaceTracker(dist_threshold=0.22)

    def reset_session(self):
        self.session = SessionStats()
        self._students.clear()
        self._tracker.reset()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _eye_open_ratio(self, lm, ids_top, ids_bot, ids_corners, h, w) -> float:
        top_y = np.mean([lm[i].y * h for i in ids_top])
        bot_y = np.mean([lm[i].y * h for i in ids_bot])
        eye_height = abs(bot_y - top_y)

        # Normalise by the eye's own horizontal width (outer-to-inner corner
        # distance) instead of a fixed fraction of frame width. This makes the
        # ratio invariant to distance-from-camera and face scale, so moving
        # slightly closer/farther no longer falsely reads as eyes closing.
        cx0, cy0 = lm[ids_corners[0]].x * w, lm[ids_corners[0]].y * h
        cx1, cy1 = lm[ids_corners[1]].x * w, lm[ids_corners[1]].y * h
        eye_width = math.hypot(cx1 - cx0, cy1 - cy0)

        return eye_height / (eye_width + 1e-6)

    def _head_pose(self, lm, h, w):
        img_pts = np.array(
            [(lm[i].x * w, lm[i].y * h) for i in POSE_LM_IDS],
            dtype=np.float64,
        )
        focal   = float(w)
        cam_mat = np.array(
            [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]],
            dtype=np.float64,
        )
        dist = np.zeros((4, 1))
        ok, rvec, _ = cv2.solvePnP(
            MODEL_POINTS_3D, img_pts, cam_mat, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        sy    = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
        return float(yaw), float(pitch)

    def _face_centroid(self, lm) -> tuple:
        """Return normalised (cx, cy) centroid of face bounding box —
        used only for cross-frame identity matching."""
        xs = [l.x for l in lm]
        ys = [l.y for l in lm]
        return (
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
        )

    def _face_bbox_px(self, lm, h, w) -> tuple:
        """Return pixel-space (x1, y1, x2, y2) bounding box, padded slightly,
        for drawing — separate from the normalised centroid used for tracking."""
        xs = [l.x * w for l in lm]
        ys = [l.y * h for l in lm]
        pad = 12
        x1 = max(0, int(min(xs)) - pad)
        y1 = max(0, int(min(ys)) - pad)
        x2 = min(w, int(max(xs)) + pad)
        y2 = min(h, int(max(ys)) + pad)
        return x1, y1, x2, y2

    def _evaluate_student(self, rec: "_StudentRecord", lm, h, w) -> AttentionState:
        """Run the full attentiveness decision for one student's landmarks,
        reading/writing only that student's smoothing + debounce state."""
        state = AttentionState(student_id=rec.student_id, face_detected=True)

        # ── Eye open ratio (scale-invariant, smoothed) ──────────────────────
        left_ear  = self._eye_open_ratio(lm, LEFT_EYE_TOP,  LEFT_EYE_BOTTOM,  LEFT_EYE_CORNERS,  h, w)
        right_ear = self._eye_open_ratio(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_CORNERS, h, w)
        raw_ear = (left_ear + right_ear) / 2

        rec._ear_hist.append(raw_ear)
        if len(rec._ear_hist) > self._ear_smooth_n:
            rec._ear_hist.pop(0)
        ear = sum(rec._ear_hist) / len(rec._ear_hist)
        state.eye_open_ratio = round(ear, 3)

        # ── Head pose ────────────────────────────────────────────────────────
        raw_yaw, raw_pitch = self._head_pose(lm, h, w)
        rec._yaw_hist.append(raw_yaw)
        rec._pitch_hist.append(raw_pitch)
        if len(rec._yaw_hist) > self._smooth_n:
            rec._yaw_hist.pop(0)
            rec._pitch_hist.pop(0)
        yaw   = sum(rec._yaw_hist) / len(rec._yaw_hist)
        pitch = sum(rec._pitch_hist) / len(rec._pitch_hist)
        state.head_yaw   = round(yaw,   1)
        state.head_pitch = round(pitch, 1)

        # ── Attention decision ───────────────────────────────────────────────
        reasons = []
        if abs(yaw) > self._yaw_thresh:
            reasons.append(f"Head turned {'left' if yaw < 0 else 'right'}")
            rec._turn_streak += 1
            if rec._turn_streak >= EYE_CLOSE_FRAMES and not rec._turn_event_logged:
                rec.head_turned_events += 1
                rec._turn_event_logged = True
        else:
            rec._turn_streak = 0
            rec._turn_event_logged = False

        if abs(pitch) > self._pitch_thresh:
            reasons.append(f"Head {'down' if pitch > 0 else 'up'}")
            rec._pitch_streak += 1
            if rec._pitch_streak >= EYE_CLOSE_FRAMES and not rec._pitch_event_logged:
                rec.head_down_events += 1
                rec._pitch_event_logged = True
        else:
            rec._pitch_streak = 0
            rec._pitch_event_logged = False

        # Eyes-closed handling: count a STREAK of consecutive low-EAR frames,
        # not every individual frame. A normal blink lasts ~3-8 frames at
        # typical webcam fps and should NOT be logged as a drowsiness event —
        # only a sustained closure (>= EYE_CLOSE_FRAMES) qualifies. Exactly
        # one event is logged per closure (on the frame it crosses the
        # threshold), preventing a single yawn/blink from inflating the count
        # into dozens of "events".
        if ear < self._eye_min:
            rec._closed_streak += 1
            if rec._closed_streak >= EYE_CLOSE_FRAMES:
                reasons.append("Eyes closed / drowsy")
                if not rec._closed_event_logged:
                    rec.eyes_closed_events += 1
                    rec._closed_event_logged = True
        else:
            rec._closed_streak = 0
            rec._closed_event_logged = False

        state.is_attentive = len(reasons) == 0
        state.reason       = ", ".join(reasons) if reasons else "Attentive"

        yaw_conf   = max(0.0, 1.0 - abs(yaw)   / (self._yaw_thresh   * 2))
        pitch_conf = max(0.0, 1.0 - abs(pitch)  / (self._pitch_thresh * 2))
        eye_conf   = min(1.0, ear / (self._eye_min + 1e-6))
        state.confidence = round((yaw_conf + pitch_conf + eye_conf) / 3, 3)

        return state

    # ── Main ──────────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray):
        """
        Process one frame. Returns (frame, states) where `states` is a list
        of AttentionState — one per face currently visible (may be empty).
        """
        h, w = frame.shape[:2]
        ts   = time.time() - self.session.start_time
        self.session.total_frames += 1

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            self.session.no_face_events    += 1
            self.session.distracted_frames += 1
            self.session.timeline.append((ts, False))
            self._draw_top_bar(frame, [], w, h)
            return frame, []

        # ── Assign stable identities to every face in this frame at once ────
        centroids = [self._face_centroid(lm) for lm in result.face_landmarks]
        ids = self._tracker.update_batch(centroids)

        states = []
        for lm, sid in zip(result.face_landmarks, ids):
            rec = self._students.setdefault(sid, _StudentRecord(student_id=sid, first_seen_ts=ts))
            rec.frames_seen  += 1
            rec.last_seen_ts  = ts

            state = self._evaluate_student(rec, lm, h, w)
            if state.is_attentive:
                rec.attentive_frames += 1
            else:
                rec.distracted_frames += 1

            states.append(state)
            self._draw_face_box(frame, lm, state, h, w)

        # Frame-level (classroom-wide) attentiveness: counts as attentive
        # only if every currently visible student is attentive.
        all_attentive = all(s.is_attentive for s in states)
        if all_attentive:
            self.session.attentive_frames += 1
        else:
            self.session.distracted_frames += 1
        self.session.timeline.append((ts, all_attentive))

        self._draw_top_bar(frame, states, w, h)
        return frame, states

    def _draw_face_box(self, frame, lm, state: AttentionState, h, w):
        """Draw a per-student bounding box + status label — used for every
        detected face so multiple students are each clearly marked."""
        x1, y1, x2, y2 = self._face_bbox_px(lm, h, w)
        color = (52, 168, 83) if state.is_attentive else (234, 67, 53)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"S{state.student_id:02d}  {'OK' if state.is_attentive else 'DISTRACTED'}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(0, y1 - th - 10)
        cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_top_bar(self, frame, states: list, w, h):
        """Thin top status bar summarising how many students are currently
        visible and how many of them are attentive right now."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (15, 17, 22), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        n = len(states)
        if n == 0:
            text  = "NO FACE DETECTED"
            color = (234, 67, 53)
        else:
            n_attentive = sum(1 for s in states if s.is_attentive)
            text  = f"{n} STUDENT{'S' if n != 1 else ''} DETECTED   .   {n_attentive}/{n} ATTENTIVE"
            color = (52, 168, 83) if n_attentive == n else (251, 188, 4)

        cv2.putText(frame, text, (12, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

    def get_session_summary(self) -> dict:
        s = self.session

        students = []
        agg_eyes_closed = agg_head_turned = agg_head_down = 0
        for sid in sorted(self._students):
            rec = self._students[sid]
            agg_eyes_closed += rec.eyes_closed_events
            agg_head_turned += rec.head_turned_events
            agg_head_down   += rec.head_down_events
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
            "eyes_closed_events":  agg_eyes_closed,
            "head_turned_events":  agg_head_turned,
            "head_down_events":    agg_head_down,
            "no_face_events":      s.no_face_events,
            "timeline":            s.timeline,
            "num_students":        len(self._students),
            "students":            students,
        }
