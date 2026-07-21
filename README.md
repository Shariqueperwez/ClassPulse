🎓 ClassPulse

Real-time classroom engagement monitoring — attentiveness tracking and phone detection, built on MediaPipe, YOLOv8, and Streamlit.


📌 About

ClassPulse is a computer-vision app that watches a classroom feed — live webcam or recorded video — and turns it into actual signal: who's paying attention, who's on their phone, and how the session went overall. Everything runs locally, frame by frame, with no video ever leaving the machine.

✨ Features
👀 Attentiveness Monitor

Tracks head pose (yaw/pitch) and eye-open ratio in real time using MediaPipe Face Mesh's 468-point landmarks — for up to 8 students at once, each with their own bounding box, live status, and attentiveness score.

📱 Phone Detection

Detects mobile phones in frame using YOLOv8n (COCO class 67), with a shape-based fallback heuristic if the model can't load.

📊 Session Reports

Turns a session into KPIs, charts, a per-student breakdown table, and exportable PDF/CSV reports — plus an optional AI-written narrative summary powered by Groq.

🧑‍🤝‍🧑 Multi-Student Identity Tracking

A greedy nearest-neighbour matcher assigns and re-assigns student IDs frame by frame, so re-entries are matched back correctly and two students in frame at once never collide onto the same ID.


🧠 How it works
Frame → MediaPipe Face Mesh → 468 landmarks (up to 8 faces)
         │
         ├─ Head pose (solvePnP)
         │     |yaw|   > threshold → looking away
         │     |pitch| > threshold → looking down/up
         │
         └─ Eye-open ratio (EAR)
               EAR < threshold → eyes closed / drowsy

