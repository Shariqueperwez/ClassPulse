import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os, json, time
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.report_generator import generate_attention_pdf, generate_phone_pdf, summary_to_csv
from utils import theme, ai_summary

st.set_page_config(page_title="Session Reports — ClassPulse", page_icon="▦", layout="wide")

theme.inject(page_accent="reports")

st.markdown(f"""
<style>
.ai-summary-box {{
    background: {theme.PAPER_DEEP};
    border: 1px solid {theme.RULE_DARK};
    border-left: 3px solid {theme.CHALK};
    padding: 18px 22px;
    font-size: 0.92rem;
    line-height: 1.65;
    color: {theme.INK};
    font-family: 'IBM Plex Sans', sans-serif;
}}
</style>
""", unsafe_allow_html=True)

theme.render_sidebar(
    "reports",
    extra=f"<span style='font-size:0.95rem;color:{theme.INK_SOFT};'>Run a live session first, then come here to view and export.</span>",
)

st.markdown(f"<div class='section-label'>03 · SESSION REPORTS</div>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{theme.INK_SOFT};margin-bottom:24px;font-size:1.05rem;'>The most recent session from each module, ready to read or export.</p>", unsafe_allow_html=True)

# Pull saved summaries from session state
attn_summary  = st.session_state.get("attn_summary")
phone_summary = st.session_state.get("phone_summary")

if not attn_summary and not phone_summary:
    st.markdown(f"""
    <div style='border:1px dashed {theme.RULE_DARK};padding:44px;text-align:center;'>
        <p style='font-family:Fraunces;font-size:1.3rem;font-weight:600;color:{theme.INK};margin:0 0 8px 0;'>No sessions recorded yet</p>
        <p style='color:{theme.INK_FAINT};font-size:1.02rem;margin:0;'>Start a monitoring session from the Attentiveness Monitor or Phone Detection pages, then return here.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Optional AI-written narrative (Groq) ───────────────────────────────────
st.markdown(f"<div class='section-label'>AI SUMMARY</div>", unsafe_allow_html=True)
if ai_summary.is_configured():
    if "ai_narrative" not in st.session_state:
        st.session_state.ai_narrative = None
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("Generate Summary", use_container_width=True):
            with st.spinner("Asking Groq for a summary..."):
                try:
                    st.session_state.ai_narrative = ai_summary.generate_session_narrative(
                        attn_summary, phone_summary
                    )
                except RuntimeError as e:
                    st.session_state.ai_narrative = None
                    st.error(str(e))
    if st.session_state.ai_narrative:
        st.markdown(f"<div class='ai-summary-box'>{st.session_state.ai_narrative}</div>", unsafe_allow_html=True)
else:
    st.info(
        "Configure a Groq API key (see .streamlit/secrets.toml.example) to enable a short "
        "written summary of the session here."
    )

st.markdown("<br>", unsafe_allow_html=True)

tabs = []
if attn_summary:  tabs.append("Attentiveness")
if phone_summary: tabs.append("Phone Detection")
if attn_summary and phone_summary: tabs.append("Combined")

selected_tabs = st.tabs(tabs)
tab_idx = 0

# ── Attentiveness tab ─────────────────────────────────────────────────────────
if attn_summary:
    with selected_tabs[tab_idx]:
        tab_idx += 1
        s = attn_summary
        score = s["attentiveness_score"]
        score_color = theme.CHALK if score >= 70 else theme.AMBER if score >= 45 else theme.PEN_RED

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%", "font": {"size": 38, "color": theme.INK, "family": "IBM Plex Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": theme.INK_FAINT},
                "bar":  {"color": score_color},
                "bgcolor": theme.PAPER_DEEP,
                "steps": [
                    {"range": [0, 45],  "color": "#E0D4CC"},
                    {"range": [45, 70], "color": "#E2DBC4"},
                    {"range": [70, 100], "color": "#D6DECB"},
                ],
                "threshold": {"line": {"color": score_color, "width": 3}, "thickness": 0.8, "value": score},
            },
            title={"text": "Attentiveness Score", "font": {"color": theme.INK_SOFT, "size": 14, "family": "IBM Plex Sans"}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor=theme.PAPER, font_color=theme.INK,
            height=270, margin=dict(l=30, r=30, t=30, b=10),
        )

        col_gauge, col_kpi = st.columns([2, 3])
        with col_gauge:
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col_kpi:
            st.markdown("<br>", unsafe_allow_html=True)
            k1, k2 = st.columns(2)
            k1.metric("Duration",       f"{s['duration_minutes']:.1f} min")
            k2.metric("Total Frames",   f"{s['total_frames']:,}")
            k3, k4 = st.columns(2)
            k3.metric("Eyes Closed",    s["eyes_closed_events"])
            k4.metric("Head Turns",     s["head_turned_events"])
            k5, k6 = st.columns(2)
            k5.metric("Head Down/Up",   s.get("head_down_events", 0))
            k6.metric("No Face",        s["no_face_events"])
            k7, k8 = st.columns(2)
            k7.metric("Distracted Frames", f"{s['distracted_frames']:,}")
            k8.metric("Students Detected", s.get("num_students", 0))

        students = s.get("students", [])
        if students:
            st.markdown("**Per-Student Breakdown**")
            df_students = pd.DataFrame(students)[
                ["label", "attentiveness_score", "frames_seen", "eyes_closed_events", "head_turned_events", "head_down_events"]
            ]
            df_students.columns = ["Student", "Attentiveness %", "Frames Seen", "Eye Events", "Head Turns", "Head Down/Up"]
            st.dataframe(df_students, use_container_width=True, hide_index=True)
            st.caption(
                "The gauge above shows class-wide attentiveness — a frame only counts as "
                "attentive when every visible student is attentive at once. Use this table "
                "for each student's individual score."
            )

        pie_fig = px.pie(
            names=["Attentive", "Distracted"],
            values=[s["attentive_frames"], s["distracted_frames"]],
            color=["Attentive", "Distracted"],
            color_discrete_map={"Attentive": theme.CHALK, "Distracted": theme.PEN_RED},
            hole=0.55,
            title="Frame Distribution",
        )
        pie_fig.update_layout(paper_bgcolor=theme.PAPER, font_color=theme.INK_SOFT,
                              font_family="IBM Plex Sans",
                              height=300, margin=dict(l=20, r=20, t=40, b=20))

        timeline = s.get("timeline", [])
        if timeline:
            df = pd.DataFrame(timeline, columns=["time", "attentive"])
            df["smooth"] = df["attentive"].astype(float).rolling(30, min_periods=1).mean()
            line_fig = px.line(df, x="time", y="smooth",
                               labels={"time": "Time (s)", "smooth": "Attention Level"},
                               title="Attention Level Over Time",
                               color_discrete_sequence=[theme.CHALK])
            line_fig.update_traces(fill="tozeroy", fillcolor="rgba(59,91,69,0.13)")
            line_fig.update_layout(paper_bgcolor=theme.PAPER, plot_bgcolor=theme.PAPER_DEEP,
                                   font_color=theme.INK_SOFT, font_family="IBM Plex Sans", height=220,
                                   margin=dict(l=20, r=20, t=40, b=20),
                                   yaxis=dict(range=[0, 1.1], tickformat=".0%"))
            col_pie, col_line = st.columns([1, 2])
            with col_pie: st.plotly_chart(pie_fig, use_container_width=True)
            with col_line: st.plotly_chart(line_fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Export Report**")
        d1, d2 = st.columns(2)
        with d1:
            pdf = generate_attention_pdf(s, f"Session — {datetime.now().strftime('%d %b %Y')}")
            st.download_button("⬇ PDF Report", pdf, "attentiveness_report.pdf", "application/pdf", use_container_width=True)
        with d2:
            csv = summary_to_csv(s, "attention")
            st.download_button("⬇ CSV Data", csv, "attentiveness_data.csv", "text/csv", use_container_width=True)

# ── Phone tab ─────────────────────────────────────────────────────────────────
if phone_summary:
    with selected_tabs[tab_idx]:
        tab_idx += 1
        s = phone_summary
        incidents = s["incident_count"]
        sev_label = "High" if incidents > 3 else "Medium" if incidents > 0 else "None"

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Incidents", incidents)
        col_b.metric("Detection Rate", f"{s['detection_rate']}%")
        col_c.metric("Duration", f"{s['duration_minutes']:.1f} min")
        col_d.metric("Severity", sev_label)

        inc_list = s.get("incidents", [])
        if inc_list:
            st.markdown("**Incident Log**")
            df_inc = pd.DataFrame(inc_list, columns=["Timestamp (s)", "Confidence"])
            df_inc["Time"] = df_inc["Timestamp (s)"].apply(lambda t: f"{int(t//60):02d}:{int(t%60):02d}")
            df_inc["Confidence %"] = (df_inc["Confidence"] * 100).round(1)
            df_inc["Severity"] = df_inc["Confidence %"].apply(lambda c: "High" if c > 75 else "Medium" if c > 50 else "Low")
            df_inc = df_inc[["Time", "Confidence %", "Severity"]]
            st.dataframe(df_inc, use_container_width=True, hide_index=True)

        timeline = s.get("timeline", [])
        if timeline:
            df = pd.DataFrame(timeline, columns=["time", "detected"])
            df["smooth"] = df["detected"].astype(float).rolling(20, min_periods=1).mean()
            area_fig = px.area(df, x="time", y="smooth",
                               labels={"time": "Time (s)", "smooth": "Phone Presence"},
                               title="Phone Detection Timeline",
                               color_discrete_sequence=[theme.PEN_RED])
            area_fig.update_layout(paper_bgcolor=theme.PAPER, plot_bgcolor=theme.PAPER_DEEP,
                                   font_color=theme.INK_SOFT, font_family="IBM Plex Sans", height=220,
                                   margin=dict(l=20, r=20, t=40, b=20),
                                   yaxis=dict(range=[0, 1.1]))
            st.plotly_chart(area_fig, use_container_width=True)

        st.markdown("---")
        d1, d2 = st.columns(2)
        with d1:
            pdf = generate_phone_pdf(s, f"Session — {datetime.now().strftime('%d %b %Y')}")
            st.download_button("⬇ PDF Report", pdf, "phone_report.pdf", "application/pdf", use_container_width=True)
        with d2:
            csv = summary_to_csv(s, "phone")
            st.download_button("⬇ CSV Data", csv, "phone_data.csv", "text/csv", use_container_width=True)

# ── Combined tab ──────────────────────────────────────────────────────────────
if attn_summary and phone_summary:
    with selected_tabs[tab_idx]:
        s1, s2 = attn_summary, phone_summary
        st.markdown(f"<div class='section-label'>COMBINED OVERVIEW</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Attentiveness",  f"{s1['attentiveness_score']}%")
        c2.metric("Phone Incidents", s2["incident_count"])
        c3.metric("Distracted Frames", f"{s1['distracted_frames']:,}")
        c4.metric("Detection Rate", f"{s2['detection_rate']}%")

        st.markdown("<br>", unsafe_allow_html=True)
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(name="Attentive",   x=["Frames"], y=[s1["attentive_frames"]],   marker_color=theme.CHALK))
        bar_fig.add_trace(go.Bar(name="Distracted",  x=["Frames"], y=[s1["distracted_frames"]],  marker_color=theme.PEN_RED))
        bar_fig.add_trace(go.Bar(name="Phone Frames", x=["Frames"], y=[s2["frames_with_phone"]], marker_color=theme.AMBER))
        bar_fig.update_layout(
            barmode="group", paper_bgcolor=theme.PAPER, plot_bgcolor=theme.PAPER_DEEP,
            font_color=theme.INK_SOFT, font_family="IBM Plex Sans", title="Frame Comparison",
            height=300, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(bar_fig, use_container_width=True)
