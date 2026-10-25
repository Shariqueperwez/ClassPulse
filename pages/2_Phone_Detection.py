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

from utils.phone_detector import PhoneDetector
from utils.timeline import fmt_mmss
from utils.report_generator import generate_phone_pdf, summary_to_csv
from utils import theme, strip_chart
from utils.webrtc_support import (
    HAS_WEBRTC, RTC_CONFIGURATION, webrtc_streamer, WebRtcMode, VideoProcessorBase, av,
)

st.set_page_config(page_title="Phone Detection — ClassPulse", page_icon="▦", layout="wide")
theme.inject(page_accent="phone")

st.markdown(f"""
<style>
.strip-wrap {{ border: 1px solid {theme.RULE_DARK}; padding: 10px 14px 4px 14px; background: {theme.PAPER_DEEP}; }}
.strip-label {{ font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: {theme.INK_FAINT}; letter-spacing: 0.06em; margin-bottom: 4px; }}
.incident-log {{ font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; color: {theme.INK_SOFT}; border-top: 1px solid {theme.RULE}; padding-top: 10px; line-height: 1.9; }}
</style>
""", unsafe_allow_html=True)

MODES = (["Webcam (live)"] if HAS_WEBRTC else []) + ["Webcam (snapshots)", "Upload video"]

theme.render_sidebar("phone")
with st.sidebar:
    st.divider()
    st.markdown(f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.85rem;color:{theme.INK_FAINT};letter-spacing:0.05em;'>SETTINGS</span>", unsafe_allow_html=True)
    source = st.radio("Input source", MODES, index=0)
    conf_thresh = st.slider("Detection confidence", 0.20, 0.90, 0.35, 0.05)
    st.caption("Lower = finds more (and smaller) phones but may add false alarms.")
    st.markdown("---")
    st.info("YOLOv8n finds phones. A face-recognition model gives each student a persistent ID (S01, S02...), so phone use is credited to the right person even if they leave and return.")

st.markdown("<div class='section-label'>02 · PHONE DETECTION</div>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{theme.INK_SOFT};margin-bottom:20px;font-size:1.05rem;'>YOLOv8 phone detection, linked to individual students by face recognition.</p>", unsafe_allow_html=True)

for key in ("phone_summary", "phone_summary_src", "phone_last_proc",
            "phone_snap_det", "phone_snap_last", "phone_snap_img", "phone_snap_det_last"):
    st.session_state.setdefault(key, None)

STRIP_WINDOW = 200


def _make_readout():
    st.markdown("<div class='section-label'>LIVE READING</div>", unsafe_allow_html=True)
    ma, mb = st.columns(2)
    with ma:
        incidents_ph, rate_ph = st.empty(), st.empty()
    with mb:
        frames_ph, students_ph = st.empty(), st.empty()
    incidents_ph.metric("Incidents", "—")
    rate_ph.metric("Detection Rate", "—")
    frames_ph.metric("Frames", "—")
    students_ph.metric("Students tracked", "—")
    st.markdown("<div class='strip-label' style='margin-top:10px;'>STUDENTS</div>", unsafe_allow_html=True)
    table_ph = st.empty()
    return dict(incidents=incidents_ph, rate=rate_ph, frames=frames_ph, students=students_ph,
                table=table_ph, backend=st.empty(), log=st.empty())


def _students_table(detector, det):
    s = detector.session
    if not s.students:
        msg = ("Looking for faces..." if detector.identity_backend == "ready"
               else "Student identification unavailable")
        return f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.85rem;color:{theme.INK_FAINT};'>{msg}</span>"
    visible = {t["sid"]: t for t in (det.tracks if det else [])}
    cell = "padding:4px 8px;font-family:\"JetBrains Mono\",monospace;font-size:0.8rem;"
    rows = ""
    for sid in sorted(s.students):
        t = visible.get(sid)
        n_inc = len(s.students[sid].incidents)
        secs = detector.phone_seconds(sid)
        if t is None:
            status, color = "OUT OF FRAME", theme.INK_FAINT
        elif t["using_phone"]:
            status, color = "USING PHONE", theme.PEN_RED
        else:
            status, color = "OK", theme.CHALK
        total = f"{fmt_mmss(secs)} · {n_inc}x" if n_inc else "—"
        rows += (f"<tr style='border-bottom:1px solid {theme.RULE};'>"
                 f"<td style='{cell}color:{theme.INK};'>S{sid:02d}</td>"
                 f"<td style='{cell}color:{color};font-weight:600;'>{status}</td>"
                 f"<td style='{cell}color:{theme.INK_SOFT};'>{total}</td></tr>")
    return f"<table style='width:100%; border-collapse:collapse;'>{rows}</table>"


def _update_readout(ph, detector, det):
    s = detector.session
    ph["incidents"].metric("Incidents", s.incident_count)
    ph["rate"].metric("Detection Rate", f"{s.detection_rate}%")
    ph["frames"].metric("Frames", f"{s.total_frames:,}")
    ph["students"].metric("Students tracked", len(s.students))
    ph["table"].markdown(_students_table(detector, det), unsafe_allow_html=True)
    ph["backend"].markdown(
        f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.85rem;color:{theme.INK_FAINT};'>"
        f"PHONE MODEL: {detector.backend.upper()} · FACE ID: {detector.identity_backend.upper()}</span>",
        unsafe_allow_html=True)
    recent = detector.live_incidents(6)
    if recent:
        lines = [f"{fmt_mmss(a)} — {who} · {fmt_mmss(d)} · {int(c * 100)}%" for a, who, d, c in recent]
        ph["log"].markdown("<div class='incident-log'>" + "<br>".join(lines) + "</div>", unsafe_allow_html=True)


class PhoneVideoProcessor(VideoProcessorBase):
    def __init__(self, conf):
        self.detector = PhoneDetector(conf_threshold=conf)
        self.lock = threading.Lock()
        self.latest_det = None
        self.strip = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            annotated, det = self.detector.process_frame(img)
            self.latest_det = det
            self.strip.append(not det.detected)      # True = clear (green)
            if len(self.strip) > STRIP_WINDOW:
                self.strip.pop(0)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


if source == "Webcam (live)":
    col_feed, col_stats = st.columns([3, 2])
    with col_feed:
        st.caption("Press START in the video box to allow camera access and begin. Press STOP to finish and get the report.")
        ctx = webrtc_streamer(
            key="phone-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: PhoneVideoProcessor(conf_thresh),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.markdown("<div class='strip-label' style='margin-top:14px;'>LIVE TRACE — LAST ~200 FRAMES, RED = PHONE VISIBLE</div>", unsafe_allow_html=True)
        strip_wrap = st.empty()
    with col_stats:
        ph = _make_readout()

    while ctx.state.playing and ctx.video_processor:
        proc = ctx.video_processor
        st.session_state.phone_last_proc = proc
        with proc.lock:
            det = proc.latest_det
            strip = list(proc.strip)
        strip_wrap.markdown(f"<div class='strip-wrap'>{strip_chart.render_strip(strip)}</div>", unsafe_allow_html=True)
        _update_readout(ph, proc.detector, det)
        time.sleep(0.3)

    proc = st.session_state.phone_last_proc
    if (proc is not None and not ctx.state.playing
            and proc.detector.session.total_frames > 0
            and st.session_state.phone_summary_src != id(proc)):
        st.session_state.phone_summary = proc.detector.get_session_summary()
        st.session_state.phone_summary_src = id(proc)

elif source == "Webcam (snapshots)":
    col_feed, col_stats = st.columns([3, 2])
    with col_feed:
        st.caption("Take a photo — every snapshot is checked for phones and added to the session. Press 'Finish session' when done.")
        shot = st.camera_input("Classroom camera", label_visibility="collapsed")
        img_ph = st.empty()
    with col_stats:
        ph = _make_readout()
        finish = st.button("Finish session & build report", width="stretch")

    if shot is not None:
        raw = shot.getvalue()
        key = hash(raw)
        if st.session_state.phone_snap_last != key:
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                if st.session_state.phone_snap_det is None:
                    st.session_state.phone_snap_det = PhoneDetector(conf_threshold=conf_thresh)
                annotated, det = st.session_state.phone_snap_det.process_frame(frame)
                st.session_state.phone_snap_last = key
                st.session_state.phone_snap_det_last = det
                st.session_state.phone_snap_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    detector = st.session_state.phone_snap_det
    if st.session_state.phone_snap_img is not None:
        img_ph.image(st.session_state.phone_snap_img, width="stretch")
    if detector is not None:
        if detector.backend == "unavailable":
            st.warning(detector.load_error or "Phone model unavailable.")
        if detector.identity_error:
            st.warning(detector.identity_error)
        _update_readout(ph, detector, st.session_state.phone_snap_det_last)
    if finish:
        if detector is not None and detector.session.total_frames > 0:
            st.session_state.phone_summary = detector.get_session_summary()
            st.session_state.phone_snap_det = None
            st.session_state.phone_snap_img = None
            st.session_state.phone_snap_det_last = None
        else:
            st.warning("Take at least one snapshot first.")

else:
    st.markdown("<hr style='margin:4px 0 24px 0;'>", unsafe_allow_html=True)
    col_feed, col_stats = st.columns([3, 2])
    with col_feed:
        frame_ph  = st.empty()
        status_ph = st.empty()
        st.markdown("<div class='strip-label' style='margin-top:14px;'>LIVE TRACE — RED = PHONE VISIBLE</div>", unsafe_allow_html=True)
        strip_wrap = st.empty()
    with col_stats:
        ph = _make_readout()

    uploaded = st.file_uploader("Upload classroom video", type=["mp4", "avi", "mov"])
    if uploaded and st.button("Analyse video", type="primary"):
        detector = PhoneDetector(conf_threshold=conf_thresh)
        if detector.backend == "unavailable":
            st.warning(detector.load_error or "Phone model unavailable.")
        if detector.identity_error:
            st.warning(detector.identity_error)
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(uploaded.name).suffix) as f:
            f.write(uploaded.read())
            tmp = f.name
        cap = cv2.VideoCapture(tmp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        prog = st.progress(0)
        idx, strip = 0, []
        SAMPLE_EVERY = 3
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if idx % SAMPLE_EVERY:
                continue
            fh, fw = frame.shape[:2]
            if fw > 1280:
                frame = cv2.resize(frame, (1280, int(fh * 1280 / fw)))
            frame, det = detector.process_frame(frame, ts=idx / fps)
            frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
            status_ph.markdown(
                f"<span class='status-line {'status-bad' if det.detected else 'status-good'}'>"
                f"{'PHONE DETECTED' if det.detected else 'CLEAR'}</span>", unsafe_allow_html=True)
            strip.append(not det.detected)
            if len(strip) > STRIP_WINDOW:
                strip.pop(0)
            strip_wrap.markdown(f"<div class='strip-wrap'>{strip_chart.render_strip(strip)}</div>", unsafe_allow_html=True)
            _update_readout(ph, detector, det)
            prog.progress(min(idx / total, 1.0))
        cap.release()
        os.unlink(tmp)
        st.session_state.phone_summary = detector.get_session_summary()
        st.success("Video processed.")

# ── Post-session ──────────────────────────────────────────────────────────────
if st.session_state.phone_summary:
    s = st.session_state.phone_summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>SESSION SUMMARY</div>", unsafe_allow_html=True)
    if s.get("backend") == "unavailable":
        st.warning("The phone model was not available for this session, so no phones could be detected.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidents",       s["incident_count"])
    c2.metric("Detection Rate",  f"{s['detection_rate']}%")
    c3.metric("Duration",        f"{s['duration_minutes']:.1f} min")
    c4.metric("Frames Analysed", f"{s['total_frames']:,}")

    students = s.get("students", [])
    if students:
        st.markdown("<div class='section-label' style='margin-top:18px;'>PHONE USE BY STUDENT</div>", unsafe_allow_html=True)
        df_st = pd.DataFrame([{
            "Student": x["label"], "Incidents": x["incident_count"],
            "Total phone time": fmt_mmss(x["phone_seconds"]) if x["incident_count"] else "-",
            "First seen": fmt_mmss(x["first_seen"]), "Last seen": fmt_mmss(x["last_seen"]),
        } for x in students])
        st.dataframe(df_st, width="stretch", hide_index=True)
        any_use = False
        for x in students:
            if x["incident_count"]:
                any_use = True
                n = x["incident_count"]
                st.markdown(f"**{x['label']}** used a phone for **{fmt_mmss(x['phone_seconds'])}** across "
                            f"{n} incident{'s' if n != 1 else ''} (first at {fmt_mmss(x['incidents'][0][0])}).")
        if not any_use:
            st.caption("No student was seen using a phone.")
        if s.get("unattributed_incidents"):
            st.caption(f"{s['unattributed_incidents']} phone sighting(s) couldn't be linked to a student.")
    elif s.get("identity_backend") == "unavailable":
        st.info("Per-student identification wasn't available, so phones aren't linked to individual students.")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("⬇ Download PDF Report", generate_phone_pdf(s, "Phone Detection"),
                           "phone_detection_report.pdf", "application/pdf", width="stretch")
    with col_dl2:
        st.download_button("⬇ Download CSV Data", summary_to_csv(s, "phone"),
                           "phone_data.csv", "text/csv", width="stretch")

    timeline = s.get("timeline", [])
    if timeline:
        df = pd.DataFrame(timeline, columns=["time", "detected"])
        df["status"] = df["detected"].map({True: "Phone Detected", False: "Clear"})
        fig = px.area(df, x="time", y="detected", color="status",
                      color_discrete_map={"Phone Detected": theme.PEN_RED, "Clear": theme.CHALK},
                      labels={"time": "Time (s)", "detected": ""},
                      title="Phone Detection Timeline", height=200)
        fig.update_layout(paper_bgcolor=theme.PAPER, plot_bgcolor=theme.PAPER_DEEP,
                          font_color=theme.INK_SOFT, font_family="Inter, Segoe UI, Arial, sans-serif",
                          margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, width="stretch")

    st.caption("Open Session Reports for a fuller breakdown and an optional AI-written summary.")
