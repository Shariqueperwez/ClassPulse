"""
ClassPulse — visual theme v2.

Design: a real EdTech monitoring dashboard. Clean, white, typographically
considered. Not paper-toned, not over-styled with glow or gradients.
Think Google Workspace + a school whiteboard, not a terminal readout.
"""

from typing import Optional
import streamlit as st

# ── Palette ──────────────────────────────────────────────────────────────────
WHITE      = "#FFFFFF"
BG         = "#F8FAFC"      # near-white page bg
BG_SURFACE = "#F1F5F9"      # slightly deeper surface (sidebar, input bg)
BG_CARD    = "#FFFFFF"      # cards are white on the grey page

BORDER     = "#E2E8F0"      # hairline borders
BORDER_MED = "#CBD5E1"

INK        = "#0F172A"      # primary text — dark slate
INK_SOFT   = "#475569"      # secondary text
INK_FAINT  = "#94A3B8"      # placeholder / captions

BLUE       = "#2563EB"      # primary brand colour — clear blue
BLUE_DARK  = "#1D4ED8"
BLUE_LIGHT = "#EFF6FF"

GREEN      = "#16A34A"
GREEN_LIGHT= "#DCFCE7"
RED        = "#DC2626"
RED_LIGHT  = "#FEE2E2"
AMBER      = "#D97706"
AMBER_LIGHT= "#FEF3C7"

GOOD = GREEN
WARN = AMBER
BAD  = RED

# Aliases used in page files
CHALK      = GREEN
CHALK_DEEP = "#15803D"
PEN_RED    = RED
PAPER      = BG
PAPER_DEEP = BG_SURFACE
RULE       = BORDER
RULE_DARK  = BORDER_MED
INK_SOFT   = INK_SOFT

_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Inter:wght@400;500;600;700&'
    'family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'
)

_BASE_CSS = f"""
<style>
/* ── Base ── */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 16px;
}}
.stApp {{
    background: {BG};
    color: {INK};
}}
.block-container {{
    padding-top: 2.2rem !important;
    max-width: 1280px;
}}
p, li, div {{
    font-size: 1rem;
    line-height: 1.6;
}}
.stMarkdown p {{
    font-size: 1rem;
    color: {INK_SOFT};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {WHITE} !important;
    border-right: 1px solid {BORDER};
    min-width: 280px !important;
}}
[data-testid="stSidebarNav"] {{ display: none; }}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {{
    color: {INK_SOFT} !important;
    font-size: 0.98rem;
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] p {{
    font-size: 1.02rem !important;
    font-weight: 500;
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] {{
    border-radius: 8px;
    padding: 4px 2px;
    margin-bottom: 2px;
    transition: background 0.15s;
}}
[data-testid="stSidebar"] [data-testid="stPageLink"]:hover {{
    background: {BG_SURFACE};
}}

/* ── Typography ── */
h1 {{
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: {INK} !important;
    letter-spacing: -0.025em;
    line-height: 1.2;
}}
h2 {{
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: {INK} !important;
    letter-spacing: -0.01em;
}}
h3 {{
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: {INK} !important;
}}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 22px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    transition: box-shadow 0.15s, transform 0.15s;
}}
[data-testid="metric-container"]:hover {{
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    transform: translateY(-1px);
}}
[data-testid="metric-container"] label {{
    color: {INK_FAINT} !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {INK} !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-weight: 700 !important;
    font-size: 2.1rem !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: {BLUE};
    color: {WHITE};
    border: none;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.65rem 1.5rem;
    box-shadow: 0 1px 2px rgba(37, 99, 235, 0.25);
    transition: background 0.15s, transform 0.1s, box-shadow 0.15s;
}}
.stButton > button:hover {{
    background: {BLUE_DARK};
    color: {WHITE};
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3);
    transform: translateY(-1px);
}}
.stButton > button:active {{
    transform: translateY(0);
}}
.stButton > button:disabled {{
    background: {BORDER_MED};
    color: {INK_FAINT};
    box-shadow: none;
}}
.stDownloadButton > button {{
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.65rem 1.5rem;
}}

/* ── Alerts ── */
.stAlert {{
    border-radius: 8px;
    border: 1px solid {BORDER_MED};
    border-left: 4px solid {BLUE};
    background: {BLUE_LIGHT} !important;
    color: {INK_SOFT} !important;
    font-size: 1rem;
    padding: 14px 18px;
}}

/* ── Dividers ── */
hr {{ border-color: {BORDER} !important; border-width: 1px 0 0 0; margin: 1.5rem 0; }}

/* ── Sliders ── */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {{
    background: {BLUE} !important;
}}
[data-testid="stSlider"] label {{
    font-size: 0.98rem !important;
    font-weight: 500;
}}

/* ── Radio buttons ── */
[data-testid="stRadio"] label {{
    font-size: 0.98rem !important;
}}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px;
    font-size: 0.95rem;
}}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab"] {{
    font-size: 1.02rem;
    font-weight: 600;
    padding: 12px 18px;
}}

/* ── Scrollbars ── */
::-webkit-scrollbar {{ width: 7px; height: 7px; }}
::-webkit-scrollbar-track {{ background: {BG_SURFACE}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER_MED}; border-radius: 4px; }}

/* ── Status pill ── */
.status-line {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 7px 16px;
    border-radius: 20px;
    border: 1.5px solid currentColor;
}}
.status-good {{ color: {GREEN}; background: {GREEN_LIGHT}; }}
.status-bad  {{ color: {RED};   background: {RED_LIGHT}; }}

/* ── Section eyebrow label ── */
.section-label {{
    font-size: 0.85rem;
    font-weight: 700;
    color: {INK_FAINT};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid {BORDER};
}}

/* ── Page title block ── */
.page-title {{
    font-size: 1.8rem;
    font-weight: 700;
    color: {INK};
    margin: 0 0 6px 0;
}}
.page-subtitle {{
    font-size: 1.05rem;
    color: {INK_SOFT};
    margin-bottom: 28px;
    line-height: 1.6;
}}

/* ── Card hover lift, used by module / step cards ── */
.cv-card {{
    transition: box-shadow 0.18s, transform 0.18s;
}}
.cv-card:hover {{
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.09) !important;
    transform: translateY(-2px);
}}
</style>
"""

PAGE_ACCENTS = {
    "attentiveness": BLUE,
    "phone":         RED,
    "reports":       INK,
}


def inject(page_accent: Optional[str] = None):
    st.markdown(_FONT_LINK, unsafe_allow_html=True)
    st.markdown(_BASE_CSS, unsafe_allow_html=True)
    if page_accent and page_accent in PAGE_ACCENTS:
        color = PAGE_ACCENTS[page_accent]
        st.markdown(f"""
        <style>
        [data-testid="metric-container"] [data-testid="stMetricValue"] {{
            color: {color} !important;
        }}
        </style>
        """, unsafe_allow_html=True)


def render_sidebar(active: str, extra: str = ""):
    with st.sidebar:
        st.markdown(f"""
        <div style='padding: 22px 8px 18px 8px; border-bottom: 1px solid {BORDER}; margin-bottom: 12px;'>
            <div style='display:flex; align-items:center; gap:12px;'>
                <div style='width:40px; height:40px; background:{BLUE}; border-radius:10px;
                            display:flex; align-items:center; justify-content:center;
                            box-shadow:0 2px 6px rgba(37,99,235,0.3);'>
                    <span style='color:{WHITE}; font-weight:700; font-size:1.05rem;'>CP</span>
                </div>
                <div>
                    <div style='font-weight:700; font-size:1.2rem; color:{INK};'>ClassPulse</div>
                    <div style='font-size:0.78rem; color:{INK_FAINT}; letter-spacing:0.04em;
                                font-family:"JetBrains Mono",monospace;'>v1.6 · LOCAL</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.page_link("Home.py",                               label="🏠  Dashboard")
        st.page_link("pages/1_Attentiveness_Monitor.py",     label="👁  Attentiveness Monitor")
        st.page_link("pages/2_Phone_Detection.py",           label="📵  Phone Detection")
        st.page_link("pages/3_Session_Reports.py",           label="📋  Session Reports")

        if extra:
            st.divider()
            st.markdown(extra, unsafe_allow_html=True)
