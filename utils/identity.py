"""
Persistent student identity ("this is Student 1, whenever he is in the frame").

How it works
  1. MediaPipe FaceLandmarker finds every face and its landmarks.
  2. Five landmarks (eyes, nose, mouth corners) are used to align the face and a
     small face-recognition network (MobileFaceNet, 512-d) turns it into an
     embedding - a numeric "face fingerprint".
  3. StudentRegistry compares each fingerprint with the ones it has already
     seen. A match keeps the old ID; a face that is consistently new becomes the
     next student (S01, S02, ...). Because matching is by face, not position,
     a student who leaves the frame and returns is still the same student.

Privacy: embeddings live in memory for the session only. Nothing is saved.
"""

import math
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
EMBED_MODEL_PATH = os.path.join(MODEL_DIR, "w600k_mbf.onnx")
# Official InsightFace release (small "buffalo_sc" pack contains the model below).
EMBED_ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
EMBED_NAME_IN_ZIP = "w600k_mbf.onnx"

# Standard 112x112 alignment template for ArcFace-family models
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],   # eye (image-left)
    [73.5318, 51.5014],   # eye (image-right)
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # mouth corner (image-left)
    [70.7299, 92.2041],   # mouth corner (image-right)
], dtype=np.float32)

MIN_FACE_PX = 28          # faces smaller than this are too blurry to recognise


DET_MODEL_PATH = os.path.join(MODEL_DIR, "det_500m.onnx")
DET_NAME_IN_ZIP = "det_500m.onnx"


def ensure_models() -> tuple:
    """Return (detector_path, embedder_path), downloading the ~15 MB pack once if needed."""
    ok = lambda p: os.path.exists(p) and os.path.getsize(p) > 500_000
    if ok(DET_MODEL_PATH) and ok(EMBED_MODEL_PATH):
        return DET_MODEL_PATH, EMBED_MODEL_PATH
    os.makedirs(MODEL_DIR, exist_ok=True)
    zip_path = os.path.join(MODEL_DIR, "_buffalo_sc.zip")
    try:
        urllib.request.urlretrieve(EMBED_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            for name, dest in ((DET_NAME_IN_ZIP, DET_MODEL_PATH), (EMBED_NAME_IN_ZIP, EMBED_MODEL_PATH)):
                with z.open(name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
    except Exception as e:
        raise RuntimeError(
            f"Could not download the face models ({e}). Download\n  {EMBED_ZIP_URL}\n"
            f"unzip it, and place {DET_NAME_IN_ZIP} and {EMBED_NAME_IN_ZIP} in: {os.path.abspath(MODEL_DIR)}"
        )
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
    return DET_MODEL_PATH, EMBED_MODEL_PATH


def five_points(lm, w: int, h: int) -> np.ndarray:
    """5 alignment points (px) from MediaPipe FaceLandmarker landmarks."""
    def p(i):
        return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)
    return np.stack([
        (p(33) + p(133)) / 2,     # image-left eye centre
        (p(362) + p(263)) / 2,    # image-right eye centre
        p(1),                     # nose tip
        p(61),                    # mouth corner, image-left
        p(291),                   # mouth corner, image-right
    ])


class FaceDetector:
    """
    SCRFD-500M face detector (multi-scale, good at small faces) that also returns the
    5 alignment points. Runs on a letter-boxed copy of the frame of at most
    `max_side` pixels on the long edge.
    """
    _STRIDES = (8, 16, 32)
    _ANCHORS = 2

    def __init__(self, score_thresh: float = 0.5, nms_thresh: float = 0.4, max_side: int = 960):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3          # the model declares fixed shapes but runs at any size; hide the harmless warnings
        self._sess = ort.InferenceSession(ensure_models()[0], sess_options=so, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        self._outs = [o.name for o in self._sess.get_outputs()]
        self.score_thresh, self.nms_thresh, self.max_side = score_thresh, nms_thresh, max_side
        self._anchor_cache = {}

    def _centers(self, h, w, stride):
        key = (h, w, stride)
        if key not in self._anchor_cache:
            g = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            c = (g * stride).reshape(-1, 2)
            self._anchor_cache[key] = np.stack([c] * self._ANCHORS, axis=1).reshape(-1, 2)
        return self._anchor_cache[key]

    def detect(self, frame_bgr: np.ndarray) -> list:
        """Returns [(bbox(x1,y1,x2,y2), kps5 (5,2) px, score)] in original-frame coordinates."""
        H, W = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(H, W))
        nh, nw = int(round(H * scale)), int(round(W * scale))
        ph, pw = (nh + 31) // 32 * 32, (nw + 31) // 32 * 32          # model wants multiples of 32
        canvas = np.zeros((ph, pw, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame_bgr, (nw, nh)) if scale != 1.0 else frame_bgr
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (pw, ph), (127.5, 127.5, 127.5), swapRB=True)
        outs = self._sess.run(self._outs, {self._input: blob})

        boxes, kpss, scores = [], [], []
        for i, stride in enumerate(self._STRIDES):
            sc = outs[i].reshape(-1)
            bp = outs[i + 3].reshape(-1, 4) * stride
            kp = outs[i + 6].reshape(-1, 10) * stride
            cen = self._centers(ph // stride, pw // stride, stride)
            keep = np.where(sc >= self.score_thresh)[0]
            if keep.size == 0:
                continue
            c, b, k = cen[keep], bp[keep], kp[keep]
            boxes.append(np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], 1))
            kpss.append(np.stack([c + k[:, 2 * j:2 * j + 2] for j in range(5)], 1))
            scores.append(sc[keep])
        if not boxes:
            return []
        boxes, kpss, scores = np.concatenate(boxes), np.concatenate(kpss), np.concatenate(scores)
        boxes, kpss = boxes / scale, kpss / scale
        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        idx = cv2.dnn.NMSBoxes(xywh, scores.tolist(), self.score_thresh, self.nms_thresh)
        out = []
        for j in np.array(idx).reshape(-1):
            x1, y1, x2, y2 = boxes[j]
            out.append(((max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))),
                        kpss[j].astype(np.float32), float(scores[j])))
        return out


class FaceEmbedder:
    def __init__(self):
        import onnxruntime as ort  # imported lazily: the app still runs without it
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._sess = ort.InferenceSession(ensure_models()[1], sess_options=so, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name

    def embed(self, frame_bgr: np.ndarray, pts5: np.ndarray) -> Optional[np.ndarray]:
        M, _ = cv2.estimateAffinePartial2D(pts5, _ARCFACE_DST, method=cv2.LMEDS)
        if M is None:
            return None
        face = cv2.warpAffine(frame_bgr, M, (112, 112), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(face, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        vec = self._sess.run(None, {self._input: blob})[0].reshape(-1).astype(np.float32)
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else None


@dataclass
class FaceObs:
    """One detected face in one frame."""
    bbox: tuple                      # (x1, y1, x2, y2) px
    emb: Optional[np.ndarray]        # None when the face was too small / unreadable
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.cx, self.cy, self.w = (x1 + x2) / 2, (y1 + y2) / 2, max(1.0, x2 - x1)


@dataclass
class _Student:
    sid: int
    gallery: list = field(default_factory=list)
    cx: float = 0.0
    cy: float = 0.0
    w: float = 1.0
    last_ts: float = -1e9
    first_ts: float = 0.0
    frames_seen: int = 0


class StudentRegistry:
    """
    Assigns persistent IDs to faces.

    match_thresh   cosine similarity needed to call two faces the same person
    relax_thresh   lower bar, used only when the face is also right where that
                   student was a moment ago (handles head turns / blur)
    confirm_frames a new face must be seen this many frames before it becomes a
                   new student (stops one-frame false detections making phantoms)
    """
    def __init__(self, match_thresh=0.38, relax_thresh=0.25, confirm_frames=4,
                 max_gallery=10, recent_s=1.0):
        self.match_thresh = match_thresh
        self.relax_thresh = relax_thresh
        self.confirm_frames = confirm_frames
        self.max_gallery = max_gallery
        self.recent_s = recent_s
        self.students: dict = {}
        self._tent: list = []
        self._next_id = 1

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _sim(emb, gallery) -> float:
        return float(max(np.dot(emb, g) for g in gallery)) if gallery else -1.0

    def _remember(self, st: _Student, obs: FaceObs, ts: float, confident: bool):
        st.cx, st.cy, st.w, st.last_ts = obs.cx, obs.cy, obs.w, ts
        st.frames_seen += 1
        if confident and obs.emb is not None:
            # keep the gallery varied: only add embeddings that differ from what we have
            if self._sim(obs.emb, st.gallery) < 0.85:
                st.gallery.append(obs.emb)
                if len(st.gallery) > self.max_gallery:
                    st.gallery.pop(1)          # keep the very first (cleanest) one

    # ── main entry ────────────────────────────────────────────────────────────
    def identify(self, faces: list, ts: float) -> list:
        """Return one student id (or None while unconfirmed) per FaceObs."""
        n = len(faces)
        result = [None] * n

        # 1. match to known students
        cands = []
        for i, f in enumerate(faces):
            if f.emb is None:
                continue
            for sid, st in self.students.items():
                sim = self._sim(f.emb, st.gallery)
                near = (ts - st.last_ts) <= self.recent_s and \
                    math.hypot(f.cx - st.cx, f.cy - st.cy) / max(st.w, 1.0) < 1.0
                if sim >= self.match_thresh or (near and sim >= self.relax_thresh):
                    cands.append((sim, i, sid))
        cands.sort(reverse=True)
        used_f, used_s = set(), set()
        for sim, i, sid in cands:
            if i in used_f or sid in used_s:
                continue
            used_f.add(i); used_s.add(sid)
            result[i] = sid
            self._remember(self.students[sid], faces[i], ts, confident=sim >= 0.5)

        # 2. unmatched faces -> tentative tracks -> maybe a brand-new student
        self._tent = [t for t in self._tent if ts - t["last_ts"] <= 1.5]
        for i, f in enumerate(faces):
            if result[i] is not None or f.emb is None:
                continue
            track = None
            for t in self._tent:
                if t.get("_taken"):
                    continue
                same_place = math.hypot(f.cx - t["cx"], f.cy - t["cy"]) / max(t["w"], 1.0) < 1.5
                same_face = self._sim(f.emb, t["embs"]) >= 0.5
                if same_place or same_face:
                    track = t
                    break
            if track is None:
                track = {"embs": [], "frames": 0, "cx": f.cx, "cy": f.cy, "w": f.w, "last_ts": ts, "first_ts": ts}
                self._tent.append(track)
            track["_taken"] = True
            track["embs"].append(f.emb)
            track["embs"] = track["embs"][-6:]
            track["frames"] += 1
            track.update(cx=f.cx, cy=f.cy, w=f.w, last_ts=ts)

            if track["frames"] >= self.confirm_frames:
                mean = np.mean(track["embs"], axis=0)
                mean /= (np.linalg.norm(mean) + 1e-9)
                # last check: is this actually someone we already know?
                best_sid, best_sim = None, -1.0
                for sid, st in self.students.items():
                    if sid in used_s:
                        continue
                    s = self._sim(mean, st.gallery)
                    if s > best_sim:
                        best_sid, best_sim = sid, s
                if best_sid is not None and best_sim >= self.match_thresh:
                    st = self.students[best_sid]
                    st.gallery.extend(track["embs"][-2:])
                    sid = best_sid
                else:
                    sid = self._next_id
                    self._next_id += 1
                    st = _Student(sid=sid, gallery=list(track["embs"][-5:]), first_ts=track["first_ts"])
                    self.students[sid] = st
                self._tent.remove(track)
                used_s.add(sid)
                result[i] = sid
                self._remember(self.students[sid], f, ts, confident=False)
        for t in self._tent:
            t.pop("_taken", None)
        return result

    def label(self, sid: int) -> str:
        return f"S{sid:02d}"
