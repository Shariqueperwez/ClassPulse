import streamlit as st
import cv2
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.phone_detector import PhoneDetector
from utils.report_generator import generate_phone_pdf, summary_to_csv
from utils import theme, strip_chart

st.set_page_config(page_title="Phone Detection — ClassPulse", page_icon="▦", layout="wide")

theme.inject(page_accent="phone")

st.markdown(f"""
<style>
.strip-wrap {{ border: 1px solid {theme.RULE_DARK}; padding: 10px 14px 4px 14px; background: {theme.PAPER_DEEP}; }}
.strip-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: {theme.INK_FAINT}; letter-spacing: 0.06em; margin-bottom: 4px; }}
.incident-log {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem; color: {theme.INK_SOFT}; border-top: 1px solid {theme.RULE}; padding-top: 10px; line-height: 1.9; }}
</style>
""", unsafe_allow_html=True)

theme.render_sidebar("phone")
with st.sidebar:
    st.divider()
    st.markdown(f"<span style='font-family:\"IBM Plex Mono\",monospace;font-size:0.85rem;color:{theme.INK_FAINT};letter-spacing:0.05em;'>SETTINGS</span>", unsafe_allow_html=True)
    source = st.radio("Input source", ["Webcam", "Upload video"], index=0)
    conf_thresh = st.slider("Detection confidence", 0.25, 0.90, 0.45, 0.05)
    st.markdown("---")
    st.info("YOLOv8n loads from the bundled weights file. Falls back to a shape heuristic if unavailable.")

st.markdown(f"<div class='section-label'>02 · PHONE DETECTION</div>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{theme.INK_SOFT};margin-bottom:20px;font-size:1.05rem;'>YOLOv8 object detection, watching for COCO class 67 — cell phone.</p>", unsafe_allow_html=True)

# Session state
if "phone_detector" not in st.session_state:
    st.session_state.phone_detector = None
if "phone_running" not in st.session_state:
    st.session_state.phone_running = False
if "phone_summary" not in st.session_state:
    st.session_state.phone_summary = None
if "phone_strip" not in st.session_state:
    st.session_state.phone_strip = []

col1, col2, _ = st.columns([1, 1, 3])
with col1:
    if st.button("▶  Start Session", use_container_width=True):
        st.session_state.phone_detector = PhoneDetector(conf_threshold=conf_thresh)
        st.session_state.phone_running  = True
        st.session_state.phone_summary  = None
        st.session_state.phone_strip    = []
with col2:
    if st.button("■  Stop Session", use_container_width=True):
        st.session_state.phone_running = False
        if st.session_state.phone_detector:
            st.session_state.phone_summary = st.session_state.phone_detector.get_session_summary()

st.markdown("<hr style='margin:4px 0 24px 0;'>", unsafe_allow_html=True)

col_feed, col_stats = st.columns([3, 2])
with col_feed:
    frame_ph  = st.empty()
    status_ph = st.empty()
    st.markdown(f"<div class='strip-label' style='margin-top:14px;'>LIVE TRACE — LAST ~20S, RED = PHONE VISIBLE</div>", unsafe_allow_html=True)
    strip_wrap = st.empty()

with col_stats:
    st.markdown(f"<div class='section-label'>LIVE READING</div>", unsafe_allow_html=True)
    ma, mb = st.columns(2)
    with ma:
        incidents_ph = st.empty()
        rate_ph      = st.empty()
    with mb:
        frames_ph = st.empty()
        conf_ph   = st.empty()
    backend_ph  = st.empty()
    log_ph      = st.empty()

STRIP_WINDOW = 200

# ── Detection loop ────────────────────────────────────────────────────────────
if st.session_state.phone_running and st.session_state.phone_detector:
    detector = st.session_state.phone_detector

    if source == "Webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
            st.session_state.phone_running = False
        else:
            while st.session_state.phone_running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame, det = detector.process_frame(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_ph.image(rgb, channels="RGB", use_container_width=True)

                status_cls  = "status-bad" if det.detected else "status-good"
                status_text = "PHONE DETECTED" if det.detected else "CLEAR"
                status_ph.markdown(f"<span class='status-line {status_cls}'>{status_text}</span>", unsafe_allow_html=True)

                st.session_state.phone_strip.append(not det.detected)  # True = "clear" (good) for strip coloring
                if len(st.session_state.phone_strip) > STRIP_WINDOW:
                    st.session_state.phone_strip.pop(0)
                svg = strip_chart.render_strip(st.session_state.phone_strip)
                strip_wrap.markdown(f"<div class='strip-wrap'>{svg}</div>", unsafe_allow_html=True)

                s = detector.session
                incidents_ph.metric("Incidents", s.incident_count)
                rate_ph.metric("Detection Rate", f"{s.detection_rate}%")
                frames_ph.metric("Frames", f"{s.total_frames:,}")
                conf_ph.metric("Confidence", f"{int(det.confidence*100)}%")
                backend_ph.markdown(
                    f"<span style='font-family:\"IBM Plex Mono\",monospace;font-size:0.85rem;color:{theme.INK_FAINT};'>BACKEND: {detector._backend.upper()}</span>",
                    unsafe_allow_html=True,
                )

                if s.incidents:
                    log_lines = []
                    for i, (ts, c) in enumerate(s.incidents[-6:], 1):
                        m = int(ts // 60); sec = int(ts % 60)
                        log_lines.append(f"{m:02d}:{sec:02d} — {int(c*100)}% confidence")
                    log_ph.markdown(
                        f"<div class='incident-log'>" + "<br>".join(log_lines) + "</div>",
                        unsafe_allow_html=True)
                time.sleep(0.04)
            cap.release()
            st.session_state.phone_summary = detector.get_session_summary()
            st.session_state.phone_running = False

    else:
        uploaded = st.file_uploader("Upload classroom video", type=["mp4", "avi", "mov"])
        if uploaded:
            import tempfile, pathlib
            with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(uploaded.name).suffix) as f:
                f.write(uploaded.read())
                tmp = f.name
            cap = cv2.VideoCapture(tmp)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            prog  = st.progress(0)
            idx   = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                if idx % 3 == 0:
                    frame, det = detector.process_frame(frame)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_ph.image(rgb, channels="RGB", use_container_width=True)

                    status_cls  = "status-bad" if det.detected else "status-good"
                    status_text = "PHONE DETECTED" if det.detected else "CLEAR"
                    status_ph.markdown(f"<span class='status-line {status_cls}'>{status_text}</span>", unsafe_allow_html=True)

                    st.session_state.phone_strip.append(not det.detected)
                    if len(st.session_state.phone_strip) > STRIP_WINDOW:
                        st.session_state.phone_strip.pop(0)
                    svg = strip_chart.render_strip(st.session_state.phone_strip)
                    strip_wrap.markdown(f"<div class='strip-wrap'>{svg}</div>", unsafe_allow_html=True)

                    s = detector.session
                    incidents_ph.metric("Incidents", s.incident_count)
                    rate_ph.metric("Detection Rate", f"{s.detection_rate}%")
                    frames_ph.metric("Frames", f"{s.total_frames:,}")
                    conf_ph.metric("Confidence", f"{int(det.confidence*100)}%")
                    prog.progress(min(idx / total, 1.0))
            cap.release()
            os.unlink(tmp)
            st.session_state.phone_summary = detector.get_session_summary()
            st.session_state.phone_running = False
            st.success("Video processed.")
elif st.session_state.phone_running and not st.session_state.phone_detector:
    st.session_state.phone_running = False
    st.warning("Session state was reset. Click Start Session again.")

# ── Post-session ──────────────────────────────────────────────────────────────
if st.session_state.phone_summary:
    s = st.session_state.phone_summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>SESSION SUMMARY</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidents",       s["incident_count"])
    c2.metric("Detection Rate",  f"{s['detection_rate']}%")
    c3.metric("Duration",        f"{s['duration_minutes']:.1f} min")
    c4.metric("Frames Analysed", f"{s['total_frames']:,}")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        pdf = generate_phone_pdf(s, "Phone Detection")
        st.download_button("⬇ Download PDF Report", pdf, "phone_detection_report.pdf", "application/pdf", use_container_width=True)
    with col_dl2:
        csv = summary_to_csv(s, "phone")
        st.download_button("⬇ Download CSV Data", csv, "phone_data.csv", "text/csv", use_container_width=True)

    import pandas as pd
    import plotly.express as px
    timeline = s.get("timeline", [])
    if timeline:
        df = pd.DataFrame(timeline, columns=["time", "detected"])
        df["status"] = df["detected"].map({True: "Phone Detected", False: "Clear"})
        fig = px.area(df, x="time", y="detected", color="status",
                      color_discrete_map={"Phone Detected": theme.PEN_RED, "Clear": theme.CHALK},
                      labels={"time": "Time (s)", "detected": ""},
                      title="Phone Detection Timeline", height=200)
        fig.update_layout(paper_bgcolor=theme.PAPER, plot_bgcolor=theme.PAPER_DEEP,
                          font_color=theme.INK_SOFT, font_family="IBM Plex Sans",
                          margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Open Session Reports for a fuller breakdown and an optional AI-written summary.")
