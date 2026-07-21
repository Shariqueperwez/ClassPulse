"""
Phone / mobile device detection engine.
Primary: YOLOv8n (ultralytics) — class 67 = cell phone
Fallback: OpenCV DNN with YOLO v4-tiny weights
"""

import cv2
import numpy as np
import time
import os
from dataclasses import dataclass, field
from typing import Optional

# COCO class names — index 67 is 'cell phone'
COCO_PHONE_CLASS = 67
COCO_CLASSES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]


@dataclass
class PhoneDetection:
    detected: bool = False
    confidence: float = 0.0
    bbox: Optional[tuple] = None   # (x1, y1, x2, y2)
    num_phones: int = 0


@dataclass
class PhoneSessionStats:
    total_frames: int = 0
    frames_with_phone: int = 0
    incident_count: int = 0       # rising edges (phone appears)
    incidents: list = field(default_factory=list)   # [(timestamp, confidence)]
    timeline: list = field(default_factory=list)    # [(timestamp, detected)]
    start_time: float = field(default_factory=time.time)
    _prev_detected: bool = False
    _miss_streak: int = 0

    @property
    def detection_rate(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return round(self.frames_with_phone / self.total_frames * 100, 1)

    @property
    def elapsed_minutes(self) -> float:
        return round((time.time() - self.start_time) / 60, 2)


class PhoneDetector:
    """
    Tries to use ultralytics YOLOv8n. If unavailable, falls back to a
    colour-heuristic detector so the app always runs without a model download.
    """
    def __init__(self, conf_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.session = PhoneSessionStats()
        self._model = None
        self._backend = "heuristic"
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO  # type: ignore
            # Resolve relative to this file's project root rather than the
            # process CWD, so the bundled weights are always found instead
            # of silently re-downloading (or failing offline).
            weights_path = os.path.join(
                os.path.dirname(__file__), "..", "yolov8n.pt"
            )
            weights_path = os.path.abspath(weights_path)
            self._model = YOLO(weights_path if os.path.exists(weights_path) else "yolov8n.pt")
            self._backend = "yolov8"
        except Exception:
            # Fallback: simple shape heuristic as last resort so the UI
            # keeps working without the model.
            self._backend = "heuristic"

    def reset_session(self):
        self.session = PhoneSessionStats()

    def process_frame(self, frame: np.ndarray) -> "tuple[np.ndarray, PhoneDetection]":
        self.session.total_frames += 1
        ts = time.time() - self.session.start_time

        if self._backend == "yolov8":
            detection = self._yolo_detect(frame)
        else:
            detection = self._heuristic_detect(frame)

        # Session tracking — debounced so a single dropped/missed YOLO frame
        # in the middle of someone holding a phone doesn't get logged as the
        # phone "disappearing and reappearing" (a new incident). The phone
        # must be absent for MISS_TOLERANCE consecutive frames before the
        # incident is considered truly over.
        MISS_TOLERANCE = 5
        if detection.detected:
            self.session.frames_with_phone += 1
            self.session._miss_streak = 0
            if not self.session._prev_detected:
                self.session.incident_count += 1
                self.session.incidents.append((ts, detection.confidence))
            self.session._prev_detected = True
        else:
            self.session._miss_streak += 1
            if self.session._miss_streak >= MISS_TOLERANCE:
                self.session._prev_detected = False
        self.session.timeline.append((ts, detection.detected))

        # Draw
        self._draw_overlay(frame, detection)
        return frame, detection

    def _yolo_detect(self, frame: np.ndarray) -> PhoneDetection:
        results = self._model(frame, verbose=False, conf=self.conf_threshold)
        phones = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == COCO_PHONE_CLASS:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    phones.append((conf, (x1, y1, x2, y2)))

        if not phones:
            return PhoneDetection(detected=False)
        best = max(phones, key=lambda p: p[0])
        return PhoneDetection(
            detected=True,
            confidence=round(best[0], 3),
            bbox=best[1],
            num_phones=len(phones),
        )

    def _heuristic_detect(self, frame: np.ndarray) -> PhoneDetection:
        """
        Simple fallback: detect rectangular dark objects in portrait orientation.
        Not production-ready — just keeps the UI functional for demo / testing.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800 or area > h * w * 0.35:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = ch / (cw + 1e-6)
            # Phone-like: tall rectangle with moderate size
            if 1.5 < aspect < 3.5 and 60 < cw < 300:
                candidates.append((area, (x, y, x + cw, y + ch)))

        if not candidates:
            return PhoneDetection(detected=False)

        # Largest candidate
        best = max(candidates, key=lambda c: c[0])
        conf = min(0.72, best[0] / (h * w * 0.1))
        return PhoneDetection(
            detected=True,
            confidence=round(conf, 3),
            bbox=best[1],
            num_phones=len(candidates),
        )

    def _draw_overlay(self, frame: np.ndarray, detection: PhoneDetection):
        h, w = frame.shape[:2]
        # Top banner
        banner_col = (30, 50, 220) if detection.detected else (10, 14, 20)
        cv2.rectangle(frame, (0, 0), (w, 48), banner_col, -1)

        if detection.detected:
            label = f"⚠ PHONE DETECTED  ({int(detection.confidence*100)}%)"
            cv2.putText(frame, label, (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 80, 80), 2, cv2.LINE_AA)
            # Bounding box
            if detection.bbox:
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 80, 255), 2)
                cv2.rectangle(frame, (x1, y1 - 24), (x1 + 130, y1), (60, 80, 255), -1)
                cv2.putText(frame, f"cell phone {int(detection.confidence*100)}%",
                            (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "✓ NO PHONE DETECTED", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 120), 2, cv2.LINE_AA)

        # Incident counter bottom-right
        incident_text = f"Incidents: {self.session.incident_count}"
        cv2.putText(frame, incident_text, (w - 160, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 130, 180), 1)

        # Backend watermark
        cv2.putText(frame, f"[{self._backend.upper()}]", (w - 160, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 80, 100), 1)

    def get_session_summary(self) -> dict:
        s = self.session
        return {
            "duration_minutes": s.elapsed_minutes,
            "total_frames": s.total_frames,
            "detection_rate": s.detection_rate,
            "frames_with_phone": s.frames_with_phone,
            "incident_count": s.incident_count,
            "incidents": s.incidents,
            "timeline": s.timeline,
            "backend": self._backend,
        }
