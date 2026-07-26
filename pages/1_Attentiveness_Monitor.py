import streamlit as st
import cv2
import numpy as np
import time
import sys
import os
import threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.attention_detector import AttentionDetector
from utils.report_generator import generate_attention_pdf, summary_to_csv
from utils import theme, strip_chart

# streamlit-webrtc: lets the VISITOR'S BROWSER send its webcam feed to this
# server for processing. cv2.VideoCapture(0) only works when Streamlit is
# running on the same machine as the camera (e.g. your own laptop) — it can
# never work on a cloud server, which has no camera attached.
import av
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase,
    RTCConfiguration,
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Attentiveness Monitor — ClassPulse", page_icon="▦", layout="wide")

theme.inject(page_accent="attentiveness")

st.markdown(f"""
<style>
.readout-box {{ border-bottom: 1px solid {theme.RULE}; padding-bottom: 14px; margin-bottom: 14px; }}
.strip-wrap {{ border: 1px solid {theme.RULE_DARK}; padding: 10px 14px 4px 14px; background: {theme.PAPER_DEEP}; }}
.strip-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: {theme.INK_FAINT}; letter-spacing: 0.06em; margin-bottom: 4px; }}
</style>
""", unsafe_allow_html=True)

theme.render_sidebar("attn")
with st.sidebar:
    st.divider()
    st.markdown(f"<span style='font-family:\"IBM Plex Mono\",monospace;font-size:0.85rem;color:{theme.INK_FAINT};letter-spacing:0.05em;'>SETTINGS</span>", unsafe_allow_html=True)
    source = st.radio("Input source", ["Webcam", "Upload video"], index=0)
    st.markdown("---")
    yaw_thresh   = st.slider("Yaw threshold (°)",   10, 45, 25)
    pitch_thresh = st.slider("Pitch threshold (°)", 10, 40, 22)
    eye_thresh   = st.slider("Eye ratio threshold", 0.10, 0.45, 0.21, 0.01)
    st.caption("Lower = stricter (flags drowsiness sooner). For Webcam mode, thresholds apply the next time you click START below.")

st.markdown(f"<div class='section-label'>01 · ATTENTIVENESS MONITOR</div>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{theme.INK_SOFT};margin-bottom:20px;font-size:1.05rem;'>Head pose and eye-state, read from MediaPipe Face Mesh.</p>", unsafe_allow_html=True)

# Session state
if "attn_detector" not in st.session_state:
    st.session_state.attn_detector = None
if "attn_running" not in st.session_state:
    st.session_state.attn_running = False
if "attn_summary" not in st.session_state:
    st.session_state.attn_summary = None
if "attn_strip" not in st.session_state:
    st.session_state.attn_strip = []

STRIP_WINDOW = 200  # ~ frames shown in the live strip, trimmed each frame


def _render_students_table(states):
    if not states:
        return f"<span style='font-family:\"IBM Plex Mono\",monospace;font-size:0.85rem;color:{theme.INK_FAINT};'>No student in frame</span>"
    rows = ""
    for st_ in states:
        color = theme.CHALK if st_.is_attentive else theme.PEN_RED
        status = "ATTENTIVE" if st_.is_attentive else "DISTRACTED"
        rows += (
            f"<tr style='border-bottom:1px solid {theme.RULE};'>"
            f"<td style='padding:4px 8px;font-family:\"IBM Plex Mono\",monospace;font-size:0.82rem;color:{theme.INK};'>S{st_.student_id:02d}</td>"
            f"<td style='padding:4px 8px;font-family:\"IBM Plex Mono\",monospace;font-size:0.82rem;color:{color};font-weight:600;'>{status}</td>"
            f"<td style='padding:4px 8px;font-family:\"IBM Plex Mono\",monospace;font-size:0.78rem;color:{theme.INK_SOFT};'>Yaw {st_.head_yaw:+.0f}&deg;</td>"
            f"<td style='padding:4px 8px;font-family:\"IBM Plex Mono\",monospace;font-size:0.78rem;color:{theme.INK_SOFT};'>Pitch {st_.head_pitch:+.0f}&deg;</td>"
            f"<td style='padding:4px 8px;font-family:\"IBM Plex Mono\",monospace;font-size:0.78rem;color:{theme.INK_SOFT};'>Eye {st_.eye_open_ratio:.2f}</td>"
            f"</tr>"
        )
    return f"<table style='width:100%; border-collapse:collapse;'>{rows}</table>"


# ── Live webcam via streamlit-webrtc ────────────────────────────────────────
class AttentionVideoProcessor(VideoProcessorBase):
    """
    Runs in a background thread managed by streamlit-webrtc, once per
    browser frame. We keep the AttentionDetector + rolling strip chart
    data here, behind a lock, so the main Streamlit script (running in a
    different thread) can safely read the latest results to update the UI.
    """
    def __init__(self, yaw_thresh, pitch_thresh, eye_thresh):
        self.detector = AttentionDetector(
            yaw_thresh=yaw_thresh, pitch_thresh=pitch_thresh, eye_min=eye_thresh
        )
        self.lock = threading.Lock()
        self.latest_states = []
        self.strip = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        annotated, states = self.detector.process_frame(img)

        n_faces = len(states)
        n_attentive = sum(1 for s in states if s.is_attentive)
        frame_attentive = n_faces > 0 and n_attentive == n_faces

        with self.lock:
            self.latest_states = states
            self.strip.append(frame_attentive)
            if len(self.strip) > STRIP_WINDOW:
                self.strip.pop(0)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


if source == "Webcam":
    col_feed, col_stats = st.columns([3, 2])

    with col_feed:
        st.caption("Click START below (in the video box) to allow camera access and begin a live session.")
        ctx = webrtc_streamer(
            key="attn-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: AttentionVideoProcessor(yaw_thresh, pitch_thresh, eye_thresh),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.markdown(f"<div class='strip-label' style='margin-top:14px;'>LIVE TRACE — LAST ~20S, GREEN = ATTENTIVE</div>", unsafe_allow_html=True)
        strip_wrap = st.empty()

    with col_stats:
        st.markdown(f"<div class='section-label'>LIVE READING</div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            score_ph   = st.empty()
            faces_ph   = st.empty()
        with m2:
            frames_ph  = st.empty()
            attentive_ph = st.empty()
        reason_ph  = st.empty()
        st.markdown(f"<div class='strip-label' style='margin-top:10px;'>STUDENTS IN FRAME</div>", unsafe_allow_html=True)
        students_table_ph = st.empty()

    # While the browser is streaming, poll the processor and refresh the UI.
    while ctx.state.playing and ctx.video_processor:
        with ctx.video_processor.lock:
            states = list(ctx.video_processor.latest_states)
            strip = list(ctx.video_processor.strip)
            s = ctx.video_processor.detector.session

        n_faces = len(states)
        n_attentive = sum(1 for st_ in states if st_.is_attentive)

        svg = strip_chart.render_strip(strip)
        strip_wrap.markdown(f"<div class='strip-wrap'>{svg}</div>", unsafe_allow_html=True)

        score_ph.metric("Score", f"{s.attentiveness_score}%")
        frames_ph.metric("Frames", f"{s.total_frames:,}")
        faces_ph.metric("Faces in frame", n_faces)
        attentive_ph.metric("Attentive now", f"{n_attentive}/{n_faces}" if n_faces else "—")
        reason_ph.markdown(
            f"<span style='font-family:\"IBM Plex Mono\",monospace;font-size:0.85rem;color:{theme.INK_SOFT};'>"
            f"{', '.join(st_.reason for st_ in states if not st_.is_attentive) or ('Attentive' if n_faces else 'No face detected')}</span>",
            unsafe_allow_html=True,
        )
        students_table_ph.markdown(_render_students_table(states), unsafe_allow_html=True)
        time.sleep(0.3)

    # Stream just stopped (or was never started) — if we have a processor
    # with recorded frames, surface the session summary once.
    if ctx.video_processor and ctx.video_processor.detector.session.total_frames > 0 \
            and not ctx.state.playing and st.session_state.attn_summary is None:
        st.session_state.attn_summary = ctx.video_processor.detector.get_session_summary()

else:
    st.markdown("<hr style='margin:4px 0 24px 0;'>", unsafe_allow_html=True)
    col_feed, col_stats = st.columns([3, 2])

    with col_feed:
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        st.markdown(f"<div class='strip-label' style='margin-top:14px;'>LIVE TRACE — LAST ~20S, GREEN = ATTENTIVE</div>", unsafe_allow_html=True)
        strip_wrap = st.empty()

    with col_stats:
        st.markdown(f"<div class='section-label'>LIVE READING</div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            score_ph   = st.empty()
            faces_ph   = st.empty()
        with m2:
            frames_ph  = st.empty()
            attentive_ph = st.empty()
        reason_ph  = st.empty()
        st.markdown(f"<div class='strip-label' style='margin-top:10px;'>STUDENTS IN FRAME</div>", unsafe_allow_html=True)
        students_table_ph = st.empty()

    uploaded = st.file_uploader("Upload classroom video", type=["mp4", "avi", "mov"])
    if uploaded:
        import tempfile, pathlib
        detector = AttentionDetector(yaw_thresh=yaw_thresh, pitch_thresh=pitch_thresh, eye_min=eye_thresh)
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(uploaded.name).suffix) as f:
            f.write(uploaded.read())
            tmp_path = f.name
        cap = cv2.VideoCapture(tmp_path)
        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_idx = 0
        strip = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 3 == 0:
                frame, states = detector.process_frame(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                n_faces     = len(states)
                n_attentive = sum(1 for st_ in states if st_.is_attentive)
                frame_attentive = n_faces > 0 and n_attentive == n_faces

                if n_faces == 0:
                    status_text, status_cls = "NO FACE DETECTED", "status-bad"
                else:
                    status_text = f"{n_attentive}/{n_faces} ATTENTIVE"
                    status_cls  = "status-good" if frame_attentive else "status-bad"
                status_placeholder.markdown(
                    f"<span class='status-line {status_cls}'>{status_text}</span>",
                    unsafe_allow_html=True,
                )

                strip.append(frame_attentive)
                if len(strip) > STRIP_WINDOW:
                    strip.pop(0)
                svg = strip_chart.render_strip(strip)
                strip_wrap.markdown(f"<div class='strip-wrap'>{svg}</div>", unsafe_allow_html=True)

                s = detector.session
                score_ph.metric("Score", f"{s.attentiveness_score}%")
                frames_ph.metric("Frames", f"{s.total_frames:,}")
                faces_ph.metric("Faces in frame", n_faces)
                attentive_ph.metric("Attentive now", f"{n_attentive}/{n_faces}" if n_faces else "—")
                students_table_ph.markdown(_render_students_table(states), unsafe_allow_html=True)
                progress.progress(min(frame_idx / total_frames, 1.0))
        cap.release()
        os.unlink(tmp_path)
        st.session_state.attn_summary = detector.get_session_summary()
        st.success("Video processed.")

# ── Post-session summary ───────────────────────────────────────────────────────
if st.session_state.attn_summary:
    s = st.session_state.attn_summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>SESSION SUMMARY</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Class Attentiveness", f"{s['attentiveness_score']}%")
    c2.metric("Duration",      f"{s['duration_minutes']:.1f} min")
    c3.metric("Students",      s.get('num_students', 0))
    c4.metric("Eye Events",    s['eyes_closed_events'])
    c5.metric("Head Turns",    s['head_turned_events'])
    c6.metric("Head Down/Up",  s.get('head_down_events', 0))

    students = s.get("students", [])
    if students:
        st.markdown(f"<div class='section-label' style='margin-top:18px;'>PER-STUDENT BREAKDOWN</div>", unsafe_allow_html=True)
        import pandas as pd
        df_students = pd.DataFrame(students)[
            ["label", "attentiveness_score", "frames_seen", "eyes_closed_events", "head_turned_events", "head_down_events"]
        ]
        df_students.columns = ["Student", "Attentiveness %", "Frames Seen", "Eye Events", "Head Turns", "Head Down/Up"]
        st.dataframe(df_students, use_container_width=True, hide_index=True)
        st.caption(
            "\"Class Attentiveness\" above counts a frame as attentive only when every "
            "visible student is attentive at once — the per-student table shows each "
            "student's own score individually."
        )

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        pdf_bytes = generate_attention_pdf(s, "Attentiveness")
        st.download_button("⬇ Download PDF Report", pdf_bytes, "attentiveness_report.pdf", "application/pdf", use_container_width=True)
    with col_dl2:
        csv_str = summary_to_csv(s, "attention")
        st.download_button("⬇ Download CSV Data", csv_str, "attentiveness_data.csv", "text/csv", use_container_width=True)

    import pandas as pd
    import plotly.express as px
    timeline = s.get("timeline", [])
    if timeline:
        df = pd.DataFrame(timeline, columns=["time", "attentive"])
        df["status"] = df["attentive"].map({True: "Attentive", False: "Distracted"})
        fig = px.scatter(df, x="time", y="attentive", color="status",
                         color_discrete_map={"Attentive": theme.CHALK, "Distracted": theme.PEN_RED},
                         labels={"time": "Time (s)", "attentive": ""},
                         title="Attentiveness Timeline",
                         height=200)
        fig.update_layout(
            paper_bgcolor=theme.PAPER, plot_bgcolor=theme.PAPER_DEEP,
            font_color=theme.INK_SOFT, font_family="IBM Plex Sans", showlegend=True,
            yaxis=dict(tickvals=[0, 1], ticktext=["Distracted", "Attentive"]),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Open Session Reports for a fuller breakdown and an optional AI-written summary.")
