import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils import theme

st.set_page_config(
    page_title="ClassPulse",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject()
theme.render_sidebar("home")

# ── Page header ──────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown(f"""
    <h1 style='margin-bottom:8px;'>Classroom Monitoring</h1>
    <p class='page-subtitle'>
        Real-time attentiveness and device-use tracking — all processing is local, nothing is uploaded.
    </p>
    """, unsafe_allow_html=True)
with col_badge:
    st.markdown(f"""
    <div style='margin-top:16px; text-align:right;'>
        <span style='background:{theme.GREEN_LIGHT}; color:{theme.GREEN};
                     font-size:0.88rem; font-weight:700; padding:7px 16px;
                     border-radius:20px; letter-spacing:0.03em;'>● LIVE READY</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Module cards ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Modules</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class='cv-card' style='background:{theme.WHITE}; border:1px solid {theme.BORDER};
                border-radius:14px; padding:26px 26px; height:215px;
                box-shadow:0 1px 3px rgba(15,23,42,0.05);'>
        <div style='width:48px; height:48px; background:{theme.BLUE_LIGHT}; border-radius:12px;
                    display:flex; align-items:center; justify-content:center;
                    font-size:1.7rem; margin-bottom:16px;'>👁</div>
        <div style='font-weight:700; font-size:1.2rem; color:{theme.INK};
                    margin-bottom:8px;'>Attentiveness Monitor</div>
        <div style='font-size:0.98rem; color:{theme.INK_SOFT}; line-height:1.6;'>
            Tracks head yaw, pitch and eye-open ratio using MediaPipe Face Mesh.
            Scores each session frame-by-frame.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='cv-card' style='background:{theme.WHITE}; border:1px solid {theme.BORDER};
                border-radius:14px; padding:26px 26px; height:215px;
                box-shadow:0 1px 3px rgba(15,23,42,0.05);'>
        <div style='width:48px; height:48px; background:{theme.RED_LIGHT}; border-radius:12px;
                    display:flex; align-items:center; justify-content:center;
                    font-size:1.7rem; margin-bottom:16px;'>📵</div>
        <div style='font-weight:700; font-size:1.2rem; color:{theme.INK};
                    margin-bottom:8px;'>Phone Detection</div>
        <div style='font-size:0.98rem; color:{theme.INK_SOFT}; line-height:1.6;'>
            Runs YOLOv8 over each frame to detect mobile phones (COCO class 67)
            and logs timestamped incidents.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='cv-card' style='background:{theme.WHITE}; border:1px solid {theme.BORDER};
                border-radius:14px; padding:26px 26px; height:215px;
                box-shadow:0 1px 3px rgba(15,23,42,0.05);'>
        <div style='width:48px; height:48px; background:{theme.AMBER_LIGHT}; border-radius:12px;
                    display:flex; align-items:center; justify-content:center;
                    font-size:1.7rem; margin-bottom:16px;'>📋</div>
        <div style='font-weight:700; font-size:1.2rem; color:{theme.INK};
                    margin-bottom:8px;'>Session Reports</div>
        <div style='font-size:0.98rem; color:{theme.INK_SOFT}; line-height:1.6;'>
            Exports a clean PDF or CSV with score, event log and timeline.
            Optional AI-written summary via Groq.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Quick start ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Quick Start</div>", unsafe_allow_html=True)
qa, qb, qc = st.columns(3)

def _step(col, num, title, body):
    with col:
        st.markdown(f"""
        <div style='display:flex; gap:16px; align-items:flex-start;'>
            <div style='min-width:34px; height:34px; background:{theme.BLUE}; color:{theme.WHITE};
                        border-radius:50%; display:flex; align-items:center;
                        justify-content:center; font-weight:700; font-size:0.95rem;
                        box-shadow:0 2px 6px rgba(37,99,235,0.3);'>
                {num}
            </div>
            <div>
                <div style='font-weight:700; font-size:1.05rem; color:{theme.INK};
                            margin-bottom:5px;'>{title}</div>
                <div style='font-size:0.95rem; color:{theme.INK_SOFT}; line-height:1.55;'>{body}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

_step(qa, "1", "Open a module", "Select Attentiveness Monitor or Phone Detection from the sidebar.")
_step(qb, "2", "Start a session", "Choose webcam or upload a video file, then press Start Session.")
_step(qc, "3", "Export the report", "Press Stop Session, then open Session Reports to download PDF or CSV.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"<div style='font-family:\"JetBrains Mono\",monospace; color:{theme.INK_FAINT};"
    f"font-size:0.8rem; letter-spacing:0.04em;'>"
    f"CLASSPULSE v1.6 · MediaPipe FaceLandmarker + YOLOv8n · All processing is local</div>",
    unsafe_allow_html=True,
)
