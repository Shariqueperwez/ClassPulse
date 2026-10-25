import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import time
import sys
import os
import threading
import tempfile
import pathlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.attention_detector import AttentionDetector, frame_is_attentive
from utils.report_generator import generate_attention_pdf, summary_to_csv
from utils import theme, strip_chart
from utils.webrtc_support import (
    HAS_WEBRTC, RTC_CONFIGURATION, webrtc_streamer, WebRtcMode, VideoProcessorBase, av,
)

st.set_page_config(page_title="Attentiveness Monitor — ClassPulse", page_icon="▦", layout="wide")
theme.inject(page_accent="attentiveness")

st.markdown(f"""
<style>
.strip-wrap {{ border: 1px solid {theme.RULE_DARK}; padding: 10px 14px 4px 14px; background: {theme.PAPER_DEEP}; }}
.strip-label {{ font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: {theme.INK_FAINT}; letter-spacing: 0.06em; margin-bottom: 4px; }}
</style>
""", unsafe_allow_html=True)

MODES = (["Webcam (live)"] if HAS_WEBRTC else []) + ["Webcam (snapshots)", "Upload video"]

theme.render_sidebar("attn")
with st.sidebar:
    st.divider()
    st.markdown(f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.85rem;color:{theme.INK_FAINT};letter-spacing:0.05em;'>SETTINGS</span>", unsafe_allow_html=True)
    source = st.radio("Input source", MODES, index=0)
    st.markdown("---")
    yaw_thresh   = st.slider("Yaw threshold (°)",   10, 45, 25)
    pitch_thresh = st.slider("Pitch threshold (°)", 10, 40, 22)
    eye_thresh   = st.slider("Eye ratio threshold", 0.10, 0.45, 0.21, 0.01)
    st.caption("Lower eye ratio = stricter. For live webcam, thresholds apply the next time you press START.")
    if not HAS_WEBRTC:
        st.caption("Live WebRTC webcam isn't available here — use snapshots or upload a video.")

st.markdown("<div class='section-label'>01 · ATTENTIVENESS MONITOR</div>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{theme.INK_SOFT};margin-bottom:20px;font-size:1.05rem;'>Head pose and eye-state, read from MediaPipe Face Mesh.</p>", unsafe_allow_html=True)

for key in ("attn_summary", "attn_summary_src", "attn_last_proc",
            "attn_snap_det", "attn_snap_last", "attn_snap_img", "attn_snap_states"):
    st.session_state.setdefault(key, None)

STRIP_WINDOW = 200


def _render_students_table(states):
    if not states:
        return f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.85rem;color:{theme.INK_FAINT};'>No student in frame</span>"
    rows = ""
    cell = "padding:4px 8px;font-family:\"JetBrains Mono\",monospace;font-size:0.8rem;"
    for s_ in states:
        color  = theme.CHALK if s_.is_attentive else theme.PEN_RED
        status = "ATTENTIVE" if s_.is_attentive else "DISTRACTED"
        rows += (
            f"<tr style='border-bottom:1px solid {theme.RULE};'>"
            f"<td style='{cell}color:{theme.INK};'>S{s_.student_id:02d}</td>"
            f"<td style='{cell}color:{color};font-weight:600;'>{status}</td>"
            f"<td style='{cell}color:{theme.INK_SOFT};'>Yaw {s_.head_yaw:+.0f}&deg;</td>"
            f"<td style='{cell}color:{theme.INK_SOFT};'>Pitch {s_.head_pitch:+.0f}&deg;</td>"
            f"<td style='{cell}color:{theme.INK_SOFT};'>Eye {s_.eye_open_ratio:.2f}</td>"
            f"</tr>"
        )
    return f"<table style='width:100%; border-collapse:collapse;'>{rows}</table>"


def _make_readout():
    st.markdown("<div class='section-label'>LIVE READING</div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        score_ph, faces_ph = st.empty(), st.empty()
    with m2:
        frames_ph, attentive_ph = st.empty(), st.empty()
    reason_ph = st.empty()
    st.markdown("<div class='strip-label' style='margin-top:10px;'>STUDENTS IN FRAME</div>", unsafe_allow_html=True)
    table_ph = st.empty()
    score_ph.metric("Class score", "—")
    faces_ph.metric("Faces in frame", "—")
    frames_ph.metric("Frames", "—")
    attentive_ph.metric("Attentive now", "—")
    return dict(score=score_ph, faces=faces_ph, frames=frames_ph,
                attentive=attentive_ph, reason=reason_ph, table=table_ph)


def _update_readout(ph, session, states):
    n = len(states)
    n_att = sum(1 for s_ in states if s_.is_attentive)
    ph["score"].metric("Class score", f"{session.attentiveness_score}%")
    ph["frames"].metric("Frames", f"{session.total_frames:,}")
    ph["faces"].metric("Faces in frame", n)
    ph["attentive"].metric("Attentive now", f"{n_att}/{n}" if n else "—")
    reasons = ", ".join(s_.reason for s_ in states if not s_.is_attentive)
    ph["reason"].markdown(
        f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.85rem;color:{theme.INK_SOFT};'>"
        f"{reasons or ('Attentive' if n else 'No face detected')}</span>",
        unsafe_allow_html=True,
    )
    ph["table"].markdown(_render_students_table(states), unsafe_allow_html=True)


# ── Live webcam via streamlit-webrtc (works when run locally) ────────────────
class AttentionVideoProcessor(VideoProcessorBase):
    def __init__(self, yaw, pitch, eye):
        self.detector = AttentionDetector(yaw_thresh=yaw, pitch_thresh=pitch, eye_min=eye)
        self.lock = threading.Lock()
        self.latest_states = []
        self.strip = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            annotated, states = self.detector.process_frame(img)
            self.latest_states = states
            self.strip.append(frame_is_attentive(states))
            if len(self.strip) > STRIP_WINDOW:
                self.strip.pop(0)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


if source == "Webcam (live)":
    col_feed, col_stats = st.columns([3, 2])
    with col_feed:
        st.caption("Press START in the video box to allow camera access and begin. Press STOP to finish and get the report.")
        ctx = webrtc_streamer(
            key="attn-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: AttentionVideoProcessor(yaw_thresh, pitch_thresh, eye_thresh),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.markdown("<div class='strip-label' style='margin-top:14px;'>LIVE TRACE — LAST ~200 FRAMES, GREEN = ATTENTIVE</div>", unsafe_allow_html=True)
        strip_wrap = st.empty()
    with col_stats:
        ph = _make_readout()

    while ctx.state.playing and ctx.video_processor:
        proc = ctx.video_processor
        st.session_state.attn_last_proc = proc
        with proc.lock:
            states = list(proc.latest_states)
            strip = list(proc.strip)
        strip_wrap.markdown(f"<div class='strip-wrap'>{strip_chart.render_strip(strip)}</div>", unsafe_allow_html=True)
        _update_readout(ph, proc.detector.session, states)
        time.sleep(0.3)

    # Stream stopped: build the summary ONCE per session (keyed by processor).
    proc = st.session_state.attn_last_proc
    if (proc is not None and not ctx.state.playing
            and proc.detector.session.total_frames > 0
            and st.session_state.attn_summary_src != id(proc)):
        st.session_state.attn_summary = proc.detector.get_session_summary()
        st.session_state.attn_summary_src = id(proc)

# ── Snapshot webcam (works on any host, no WebRTC needed) ────────────────────
elif source == "Webcam (snapshots)":
    col_feed, col_stats = st.columns([3, 2])
    with col_feed:
        st.caption("Take a photo — every snapshot is added to the current session. Press 'Finish session' when you're done.")
        shot = st.camera_input("Classroom camera", label_visibility="collapsed")
        img_ph = st.empty()
    with col_stats:
        ph = _make_readout()
        finish = st.button("Finish session & build report", width="stretch")

    if shot is not None:
        raw = shot.getvalue()
        key = hash(raw)
        if st.session_state.attn_snap_last != key:
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                if st.session_state.attn_snap_det is None:
                    st.session_state.attn_snap_det = AttentionDetector(
                        yaw_thresh=yaw_thresh, pitch_thresh=pitch_thresh, eye_min=eye_thresh)
                annotated, states = st.session_state.attn_snap_det.process_frame(frame)
                st.session_state.attn_snap_last = key
                st.session_state.attn_snap_states = states
                st.session_state.attn_snap_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    det = st.session_state.attn_snap_det
    if st.session_state.attn_snap_img is not None:
        img_ph.image(st.session_state.attn_snap_img, width="stretch")
    if det is not None:
        _update_readout(ph, det.session, st.session_state.attn_snap_states or [])
    if finish:
        if det is not None and det.session.total_frames > 0:
            st.session_state.attn_summary = det.get_session_summary()
            st.session_state.attn_snap_det = None
            st.session_state.attn_snap_img = None
            st.session_state.attn_snap_states = None
        else:
            st.warning("Take at least one snapshot first.")

# ── Uploaded video ───────────────────────────────────────────────────────────
else:
    st.markdown("<hr style='margin:4px 0 24px 0;'>", unsafe_allow_html=True)
    col_feed, col_stats = st.columns([3, 2])
    with col_feed:
        frame_placeholder  = st.empty()
        status_placeholder = st.empty()
        st.markdown("<div class='strip-label' style='margin-top:14px;'>LIVE TRACE — GREEN = ATTENTIVE</div>", unsafe_allow_html=True)
        strip_wrap = st.empty()
    with col_stats:
        ph = _make_readout()

    uploaded = st.file_uploader("Upload classroom video", type=["mp4", "avi", "mov"])
    if uploaded and st.button("Analyse video", type="primary"):
        detector = AttentionDetector(yaw_thresh=yaw_thresh, pitch_thresh=pitch_thresh, eye_min=eye_thresh)
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(uploaded.name).suffix) as f:
            f.write(uploaded.read())
            tmp_path = f.name
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        progress = st.progress(0)
        frame_idx, strip = 0, []
        SAMPLE_EVERY = 3
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % SAMPLE_EVERY:
                continue
            fh, fw = frame.shape[:2]
            if fw > 960:                                   # keep CPU time reasonable
                frame = cv2.resize(frame, (960, int(fh * 960 / fw)))
            frame, states = detector.process_frame(frame, ts=frame_idx / fps)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")

            n_faces = len(states)
            n_att = sum(1 for s_ in states if s_.is_attentive)
            ok = frame_is_attentive(states)
            status_placeholder.markdown(
                f"<span class='status-line {'status-good' if ok else 'status-bad'}'>"
                f"{'NO FACE DETECTED' if n_faces == 0 else f'{n_att}/{n_faces} ATTENTIVE'}</span>",
                unsafe_allow_html=True)
            strip.append(ok)
            if len(strip) > STRIP_WINDOW:
                strip.pop(0)
            strip_wrap.markdown(f"<div class='strip-wrap'>{strip_chart.render_strip(strip)}</div>", unsafe_allow_html=True)
            _update_readout(ph, detector.session, states)
            progress.progress(min(frame_idx / total_frames, 1.0))
        cap.release()
        os.unlink(tmp_path)
        st.session_state.attn_summary = detector.get_session_summary()
        st.success("Video processed.")

# ── Post-session summary ──────────────────────────────────────────────────────
if st.session_state.attn_summary:
    s = st.session_state.attn_summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>SESSION SUMMARY</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Class Attentiveness", f"{s['attentiveness_score']}%")
    c2.metric("Duration",      f"{s['duration_minutes']:.1f} min")
    c3.metric("Students",      s.get("num_students", 0))
    c4.metric("Eye Events",    s["eyes_closed_events"])
    c5.metric("Head Turns",    s["head_turned_events"])
    c6.metric("Head Down/Up",  s.get("head_down_events", 0))

    students = s.get("students", [])
    if students:
        st.markdown("<div class='section-label' style='margin-top:18px;'>PER-STUDENT BREAKDOWN</div>", unsafe_allow_html=True)
        df_students = pd.DataFrame(students)[
            ["label", "attentiveness_score", "frames_seen", "eyes_closed_events", "head_turned_events", "head_down_events"]
        ]
        df_students.columns = ["Student", "Attentiveness %", "Frames Seen", "Eye Events", "Head Turns", "Head Down/Up"]
        st.dataframe(df_students, width="stretch", hide_index=True)
        st.caption(
            "Class Attentiveness counts a frame as attentive when at least half of the visible "
            "students are attentive. The table shows each student's own score."
        )

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("⬇ Download PDF Report", generate_attention_pdf(s, "Attentiveness"),
                           "attentiveness_report.pdf", "application/pdf", width="stretch")
    with col_dl2:
        st.download_button("⬇ Download CSV Data", summary_to_csv(s, "attention"),
                           "attentiveness_data.csv", "text/csv", width="stretch")

    timeline = s.get("timeline", [])
    if timeline:
        df = pd.DataFrame(timeline, columns=["time", "attentive"])
        df["status"] = df["attentive"].map({True: "Attentive", False: "Distracted"})
        fig = px.scatter(df, x="time", y="attentive", color="status",
                         color_discrete_map={"Attentive": theme.CHALK, "Distracted": theme.PEN_RED},
                         labels={"time": "Time (s)", "attentive": ""},
                         title="Attentiveness Timeline", height=200)
        fig.update_layout(
            paper_bgcolor=theme.PAPER, plot_bgcolor=theme.PAPER_DEEP,
            font_color=theme.INK_SOFT, font_family="Inter, Segoe UI, Arial, sans-serif", showlegend=True,
            yaxis=dict(tickvals=[0, 1], ticktext=["Distracted", "Attentive"]),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, width="stretch")

    st.caption("Open Session Reports for a fuller breakdown and an optional AI-written summary.")
