# 🎓 ClassPulse

Classroom engagement monitoring — attentiveness tracking and phone detection, built on MediaPipe, YOLOv8 and Streamlit.

## Features
- **Attentiveness Monitor** — head pose (yaw/pitch via solvePnP) and eye-open ratio from MediaPipe's 478-point face landmarks, for up to 8 students at once, each with a bounding box, live status and score.
- **Phone Detection, per student** — YOLOv8n finds phones; a face-recognition model gives every student a persistent ID (S01, S02, …) that survives leaving and re-entering the frame. The report says exactly who used a phone, for how long, and how many times — e.g. *“S02 used a phone for 0:42 across 2 incidents.”*
- **Session Reports** — KPIs, charts, per-student table, PDF/CSV export, and an optional AI-written summary (Google Gemini free tier).
- **Multi-student identity** — Attentiveness Monitor tracks students by position; Phone Detection identifies them by face (see below).

Input options: **Webcam (live)** via WebRTC (local runs), **Webcam (snapshots)** (works anywhere), or **Upload video**.

## Screenshots
| Dashboard | Attentiveness Monitor |
|---|---|
| ![Home](docs/home.png) | ![Attentiveness](docs/attentiveness.png) |

| Session report (demo data) | Phone detection report |
|---|---|
| ![Report](docs/report.png) | ![Phone report](docs/phone-report.png) |

*No webcam? Open **Session Reports** and click **Load demo session** to explore the report UI with sample data.*

## How student identity works (Phone Detection)
1. **SCRFD** (a small multi-scale face detector) finds every face, including small ones at the back of a room, and returns 5 landmark points.
2. Each face is aligned and passed through a **MobileFaceNet** model that outputs a 512-number "face fingerprint".
3. A registry compares each fingerprint with the students seen so far. A match keeps the old ID; a face that is consistently new becomes the next student. Because matching is by face, **a student who leaves and returns — even at a different seat — is still the same student.** One-frame false detections never create phantom students.
4. Each detected phone is linked to the nearest student's face. Phones with nobody close are reported as *Unidentified*.

Tested on real photos: the same person scores ≈ 0.72 similarity across different photos, different people ≈ 0.0. Fingerprints are kept in memory for the session only and are never saved.

> The two face models (~15 MB) download automatically on first use from the official InsightFace release. Their licence permits **non-commercial research/educational use** only, which is why they are not committed to this repo.

## How scoring works
- A student is *distracted* when |yaw| or |pitch| exceeds the threshold, or eyes stay closed for 0.4 s+ while the head is level.
- Events (head turn, head down/up, eyes closed, no face) are debounced by **time**, so results don't depend on fps.
- **Class attentiveness** = share of frames where at least half of the visible students are attentive. Per-student scores are in the table.
- Uploaded videos use *video* time for durations and timelines. Every 3rd frame is analysed.

## Run it locally
Requires **Python 3.10 – 3.12**.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run Home.py
```
Open http://localhost:8501. Allow camera access when asked.

Optional AI summary: get a free key at https://aistudio.google.com/apikey, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set `GEMINI_API_KEY`.

## Deploy (Streamlit Community Cloud)
Push to GitHub, create an app with main file `Home.py`. `packages.txt` installs the system libraries OpenCV needs. Live WebRTC is often blocked on cloud hosts — use **Webcam (snapshots)** or **Upload video** there.

## Limitations & privacy
- Head pose and eye state are a rough proxy for attention, not a measure of learning. Treat results as a prompt for a teacher, not a verdict on a student.
- Phone detection depends on camera angle and resolution; small or partly hidden phones can be missed.
- Phone Detection computes face fingerprints (biometric data) in memory to tell students apart. Nothing is stored, but get consent before using it on real people.
- Identity needs a reasonably clear face: students looking away from the camera for long stretches may be briefly unrecognised (a phone is still linked to them if they were seen in the last 2 s).
- No video is stored: frames are processed in memory and discarded. Only aggregate numbers are kept for the report (and sent to Google Gemini if you enable the AI summary — never video or images).
- Monitoring people on camera needs consent and must follow your school's and region's privacy rules.
