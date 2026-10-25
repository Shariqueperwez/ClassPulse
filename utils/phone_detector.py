"""
Phone detection with per-student attribution (YOLOv8n + face identity).

v1.8
  - Every student gets a persistent ID (S01, S02, ...) based on their FACE, so the
    ID survives leaving the frame, coming back, or moving seats (utils/identity.py).
  - Each detected phone is attributed to the nearest student, so the report can say
    "S02 used a phone for 0:42 across 2 incidents".
  - Phones with no student near them are kept as "Unidentified".
  - If the face-recognition model can't load, phone detection still works and
    the report simply has no per-student breakdown (identity_backend = "unavailable").
"""

import cv2
import numpy as np
import time
import os
import math
from dataclasses import dataclass, field
from typing import Optional

from utils.timeline import bucket_timeline

COCO_PHONE_CLASS  = 67
IMGSZ             = 800    # multiple of 32; larger = better for small phones, slower on CPU
PHONE_GAP_SECONDS = 1.0    # a phone must vanish this long before the next sighting is a NEW incident
ASSIGN_MAX_FACES  = 4.0    # a phone counts as "theirs" if within ~4 face-widths of the face
RECENT_HOLD_S     = 2.0    # a student who just left view (e.g. head down) can still own a phone this long


@dataclass
class PhoneDetection:
    detected: bool = False
    confidence: float = 0.0
    bbox: Optional[tuple] = None          # best phone box (x1, y1, x2, y2)
    num_phones: int = 0
    phones: list = field(default_factory=list)   # [(conf, bbox, student_id or None)]
    tracks: list = field(default_factory=list)   # one dict per confirmed student seen this frame


@dataclass
class StudentPhoneStats:
    sid: int
    first_seen: float = 0.0
    last_seen: float = 0.0
    frames_seen: int = 0
    phone_frames: int = 0
    incidents: list = field(default_factory=list)   # [[start, end, max_conf], ...]
    last_phone_ts: Optional[float] = None


@dataclass
class PhoneSessionStats:
    total_frames: int = 0
    frames_with_phone: int = 0
    timeline: list = field(default_factory=list)    # [(timestamp, detected)]
    students: dict = field(default_factory=dict)    # sid -> StudentPhoneStats
    unattributed: list = field(default_factory=list)  # incidents with no student: [[start, end, conf]]
    unattributed_last_ts: Optional[float] = None
    start_time: float = field(default_factory=time.time)
    last_ts: float = 0.0

    @property
    def incident_count(self) -> int:
        return sum(len(s.incidents) for s in self.students.values()) + len(self.unattributed)

    @property
    def detection_rate(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return round(self.frames_with_phone / self.total_frames * 100, 1)

    @property
    def elapsed_minutes(self) -> float:
        return round(self.last_ts / 60, 2)


def _fmt(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


class PhoneDetector:
    def __init__(self, conf_threshold: float = 0.35, allow_heuristic: bool = False,
                 track_students: bool = True):
        self.conf_threshold = conf_threshold
        self.allow_heuristic = allow_heuristic
        self.session = PhoneSessionStats()
        self._model = None
        self._backend = "unavailable"
        self.load_error = ""
        self._dt = 0.1                     # running estimate of seconds between processed frames
        self._prev_ts: Optional[float] = None

        # identity (optional)
        self._detector = None
        self._embedder = None
        self._registry = None
        self.identity_error = ""
        self._load_model()
        if track_students:
            self._load_identity()

    # ── properties ────────────────────────────────────────────────────────────
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def identity_backend(self) -> str:
        return "ready" if self._registry is not None else "unavailable"

    # ── loading ───────────────────────────────────────────────────────────────
    def _load_model(self):
        try:
            from ultralytics import YOLO  # type: ignore
            weights = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt"))
            self._model = YOLO(weights if os.path.exists(weights) else "yolov8n.pt")
            self._backend = "yolov8"
        except Exception as e:
            self.load_error = f"YOLOv8 could not be loaded ({e})."
            self._backend = "heuristic" if self.allow_heuristic else "unavailable"

    def _load_identity(self):
        try:
            from utils.identity import FaceDetector, FaceEmbedder, StudentRegistry
            self._detector = FaceDetector()
            self._embedder = FaceEmbedder()
            self._registry = StudentRegistry()
        except Exception as e:
            self._registry = None
            self.identity_error = (
                f"Student identification is unavailable ({e}). Phones are still detected, "
                "but they can't be linked to individual students."
            )

    def reset_session(self):
        self.session = PhoneSessionStats()
        self._prev_ts = None
        if self._registry is not None:
            from utils.identity import StudentRegistry
            self._registry = StudentRegistry()

    # ── main ──────────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray, ts: Optional[float] = None):
        if ts is None:
            ts = time.time() - self.session.start_time
        if self._prev_ts is not None and 0 < ts - self._prev_ts <= 1.0:
            self._dt = 0.8 * self._dt + 0.2 * (ts - self._prev_ts)
        self._prev_ts = ts
        s = self.session
        s.last_ts = max(s.last_ts, ts)
        s.total_frames += 1

        # 1) phones
        if self._backend == "yolov8":
            phones = self._yolo_detect(frame)
        elif self._backend == "heuristic":
            phones = self._heuristic_detect(frame)
        else:
            phones = []

        # 2) who is in the frame (persistent IDs)
        tracks = self._identify_students(frame, ts)

        # 3) which student does each phone belong to
        owner = self._assign_phones(phones, tracks, ts)

        # 4) update per-student and unattributed incident state
        owned = {}
        for (conf, box), sid in zip(phones, owner):
            if sid is None:
                self._note_unattributed(ts, conf)
            else:
                owned[sid] = max(owned.get(sid, 0.0), conf)
        for sid, conf in owned.items():
            st = s.students[sid]
            st.phone_frames += 1
            if st.last_phone_ts is None or (ts - st.last_phone_ts) >= PHONE_GAP_SECONDS:
                st.incidents.append([ts, ts, conf])
            else:
                inc = st.incidents[-1]
                inc[1] = ts
                inc[2] = max(inc[2], conf)
            st.last_phone_ts = ts

        detected = len(phones) > 0
        if detected:
            s.frames_with_phone += 1
        s.timeline.append((ts, detected))

        best = max(phones, key=lambda p: p[0]) if phones else None
        det = PhoneDetection(
            detected=detected,
            confidence=round(best[0], 3) if best else 0.0,
            bbox=best[1] if best else None,
            num_phones=len(phones),
            phones=[(round(c, 3), b, o) for (c, b), o in zip(phones, owner)],
            tracks=self._track_infos(tracks, ts),
        )
        self._draw_overlay(frame, det, tracks)
        return frame, det

    # ── identity ──────────────────────────────────────────────────────────────
    def _identify_students(self, frame, ts) -> list:
        """Confirmed students visible this frame: [{sid, bbox, cx, cy, w, confirmed}] plus unconfirmed faces."""
        if self._registry is None:
            return []
        from utils.identity import FaceObs, MIN_FACE_PX
        obs = []
        for bbox, kps, score in self._detector.detect(frame):
            emb = None
            if (bbox[2] - bbox[0]) >= MIN_FACE_PX:
                emb = self._embedder.embed(frame, kps)
            obs.append(FaceObs(bbox=bbox, emb=emb))
        sids = self._registry.identify(obs, ts)

        tracks = []
        for o, sid in zip(obs, sids):
            if sid is not None:
                st = self.session.students.setdefault(sid, StudentPhoneStats(sid=sid, first_seen=ts))
                st.last_seen = ts
                st.frames_seen += 1
            tracks.append({"sid": sid, "bbox": o.bbox, "cx": o.cx, "cy": o.cy, "w": o.w})
        return tracks

    def _assign_phones(self, phones, tracks, ts) -> list:
        """Return a student id (or None) for each phone, by distance to the nearest face."""
        if self._registry is None or not phones:
            return [None] * len(phones)
        # candidates: faces visible now + students seen in the last RECENT_HOLD_S (head down, turned away...)
        cands = {}
        for t in tracks:
            if t["sid"] is not None:
                cands[t["sid"]] = (t["cx"], t["cy"], t["w"])
        for sid, st in self._registry.students.items():
            if sid not in cands and (ts - st.last_ts) <= RECENT_HOLD_S:
                cands[sid] = (st.cx, st.cy, st.w)
        owners = []
        for conf, (x1, y1, x2, y2) in phones:
            px, py = (x1 + x2) / 2, (y1 + y2) / 2
            best, best_d = None, ASSIGN_MAX_FACES
            for sid, (cx, cy, fw) in cands.items():
                d = math.hypot(px - cx, py - cy) / max(fw, 1.0)
                if d < best_d:
                    best, best_d = sid, d
            owners.append(best)
        return owners

    def _note_unattributed(self, ts, conf):
        s = self.session
        if s.unattributed_last_ts is None or (ts - s.unattributed_last_ts) >= PHONE_GAP_SECONDS:
            s.unattributed.append([ts, ts, conf])
        else:
            inc = s.unattributed[-1]
            inc[1] = ts
            inc[2] = max(inc[2], conf)
        s.unattributed_last_ts = ts

    def _incident_seconds(self, inc) -> float:
        return max(0.0, inc[1] - inc[0]) + self._dt

    def _phone_seconds(self, st: StudentPhoneStats) -> float:
        return sum(self._incident_seconds(i) for i in st.incidents)

    def phone_seconds(self, sid: int) -> float:
        st = self.session.students.get(sid)
        return round(self._phone_seconds(st), 1) if st else 0.0

    def live_incidents(self, n: int = 6) -> list:
        """Most recent incidents: [(start_s, 'S01' | 'Unidentified', seconds, confidence)]."""
        rows = []
        for sid, st in self.session.students.items():
            for a, b, c in st.incidents:
                rows.append((a, f"S{sid:02d}", self._incident_seconds([a, b]), c))
        for a, b, c in self.session.unattributed:
            rows.append((a, "Unidentified", self._incident_seconds([a, b]), c))
        rows.sort(key=lambda r: r[0])
        return rows[-n:]

    def _track_infos(self, tracks, ts) -> list:
        out = []
        for t in tracks:
            if t["sid"] is None:
                continue
            st = self.session.students[t["sid"]]
            using = st.last_phone_ts is not None and (ts - st.last_phone_ts) < PHONE_GAP_SECONDS
            out.append({
                "sid": t["sid"], "label": f"S{t['sid']:02d}", "bbox": t["bbox"],
                "using_phone": using, "incidents": len(st.incidents),
                "phone_seconds": round(self._phone_seconds(st), 1),
            })
        return out

    # ── detection backends ────────────────────────────────────────────────────
    def _yolo_detect(self, frame: np.ndarray) -> list:
        results = self._model(
            frame, verbose=False, conf=self.conf_threshold,
            imgsz=IMGSZ, classes=[COCO_PHONE_CLASS],
        )
        phones = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                phones.append((float(box.conf[0]), (x1, y1, x2, y2)))
        return phones

    def _heuristic_detect(self, frame: np.ndarray) -> list:
        """Demo-only shape heuristic (opt-in via allow_heuristic=True). Not reliable."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800 or area > h * w * 0.35:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            if 1.5 < ch / (cw + 1e-6) < 3.5 and 60 < cw < 300 and (best is None or area > best[0]):
                best = (area, (x, y, x + cw, y + ch))
        if best is None:
            return []
        return [(min(0.72, best[0] / (h * w * 0.1)), best[1])]

    # ── drawing ───────────────────────────────────────────────────────────────
    def _draw_overlay(self, frame, det: PhoneDetection, tracks: list):
        h, w = frame.shape[:2]
        info = {t["sid"]: t for t in det.tracks}

        # student boxes
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            sid = t["sid"]
            if sid is None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
                continue
            ti = info.get(sid)
            using = bool(ti and ti["using_phone"])
            color = (60, 60, 235) if using else (83, 168, 52)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"S{sid:02d}" + (f"  PHONE {_fmt(ti['phone_seconds'])}" if using else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(52, y1 - th - 10)
            cv2.rectangle(frame, (x1, ly), (x1 + tw + 10, ly + th + 8), color, -1)
            cv2.putText(frame, label, (x1 + 5, ly + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # phone boxes, with a line to the owner
        centers = {t["sid"]: (int(t["cx"]), int(t["cy"])) for t in tracks if t["sid"] is not None}
        for conf, (x1, y1, x2, y2), owner in det.phones:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 80, 255), 2)
            tag = f"phone {int(conf * 100)}%" + (f" -> S{owner:02d}" if owner is not None else " (unassigned)")
            cv2.putText(frame, tag, (x1, min(h - 6, y2 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (60, 80, 255), 1, cv2.LINE_AA)
            if owner in centers:
                cv2.line(frame, ((x1 + x2) // 2, (y1 + y2) // 2), centers[owner], (60, 80, 255), 1, cv2.LINE_AA)

        # top banner
        n_students = len(info)
        n_using = sum(1 for t in info.values() if t["using_phone"])
        if self._backend == "unavailable":
            banner, text, color = (60, 60, 60), "PHONE MODEL UNAVAILABLE", (200, 200, 200)
        elif det.detected:
            banner, color = (30, 50, 220), (255, 255, 255)
            text = f"PHONE DETECTED   |   {n_using} OF {n_students} STUDENTS" if n_students else "PHONE DETECTED"
        else:
            banner, color = (10, 14, 20), (0, 220, 120)
            text = f"{n_students} STUDENTS TRACKED   |   NO PHONE" if n_students else "NO PHONE DETECTED"
        cv2.rectangle(frame, (0, 0), (w, 44), banner, -1)
        cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Incidents: {self.session.incident_count}", (w - 160, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 130, 180), 1)

    # ── summary ───────────────────────────────────────────────────────────────
    def get_session_summary(self) -> dict:
        s = self.session
        students, log = [], []
        for sid in sorted(s.students):
            st = s.students[sid]
            label = f"S{sid:02d}"
            secs = self._phone_seconds(st)
            students.append({
                "student_id": sid, "label": label,
                "first_seen": round(st.first_seen, 1), "last_seen": round(st.last_seen, 1),
                "frames_seen": st.frames_seen, "phone_frames": st.phone_frames,
                "incident_count": len(st.incidents),
                "phone_seconds": round(secs, 1),
                "incidents": [(round(a, 1), round(b + self._dt, 1), round(c, 3)) for a, b, c in st.incidents],
            })
            for a, b, c in st.incidents:
                log.append({"student": label, "start": round(a, 1), "end": round(b + self._dt, 1),
                            "seconds": round(self._incident_seconds([a, b]), 1), "confidence": round(c, 3)})
        for a, b, c in s.unattributed:
            log.append({"student": "Unidentified", "start": round(a, 1), "end": round(b + self._dt, 1),
                        "seconds": round(self._incident_seconds([a, b]), 1), "confidence": round(c, 3)})
        log.sort(key=lambda r: r["start"])

        return {
            "duration_minutes":  s.elapsed_minutes,
            "total_frames":      s.total_frames,
            "detection_rate":    s.detection_rate,
            "frames_with_phone": s.frames_with_phone,
            "incident_count":    s.incident_count,
            "incidents":         [(r["start"], r["confidence"]) for r in log],   # legacy shape
            "incident_log":      log,
            "students":          students,
            "num_students":      len(students),
            "unattributed_incidents": len(s.unattributed),
            "timeline":          bucket_timeline(s.timeline, 1.0, any_mode=True),
            "backend":           self._backend,
            "identity_backend":  self.identity_backend,
        }
