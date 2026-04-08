"""
IoT Energy Analytics Platform - Streamlit dashboard
===================================================
Main application with three views:
1. Real-time monitoring
2. ML predictions
3. Load optimization
"""

import os
from datetime import datetime

import streamlit as st

from pages import monitoring, optimization, predictions

st.set_page_config(
    page_title="GreenIoT Control Center",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg: #f4efe6;
        --paper: rgba(255, 252, 247, 0.82);
        --paper-strong: #fffdf8;
        --ink: #1b2a22;
        --muted: #61756b;
        --line: rgba(27, 42, 34, 0.10);
        --accent: #166a4a;
        --accent-2: #d88b2b;
        --accent-soft: rgba(22, 106, 74, 0.10);
        --shadow: 0 18px 50px rgba(39, 52, 45, 0.10);
    }

    html, body, [class*="css"] {
        font-family: "IBM Plex Sans", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(216, 139, 43, 0.18), transparent 24%),
            radial-gradient(circle at top right, rgba(22, 106, 74, 0.15), transparent 28%),
            linear-gradient(180deg, #f6f1e7 0%, #eef3ed 52%, #f7f4ef 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(19, 51, 40, 0.98), rgba(18, 41, 36, 0.96));
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #edf5ef;
    }

    [data-testid="stSidebar"] .stRadio label p,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption {
        color: #edf5ef !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #d88b2b, #f0b560);
        color: #10251c;
        border: none;
        border-radius: 999px;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(216, 139, 43, 0.25);
    }

    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 0.5rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .shell-hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(27, 42, 34, 0.08);
        border-radius: 28px;
        padding: 1.35rem 1.5rem 1.3rem 1.5rem;
        margin-bottom: 1.3rem;
        background:
            radial-gradient(circle at 15% 15%, rgba(216, 139, 43, 0.16), transparent 24%),
            radial-gradient(circle at 85% 20%, rgba(22, 106, 74, 0.16), transparent 28%),
            linear-gradient(135deg, rgba(255,255,255,0.94), rgba(248, 244, 236, 0.86));
        box-shadow: var(--shadow);
    }

    .shell-kicker {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-weight: 700;
        color: var(--accent);
        margin-bottom: 0.55rem;
    }

    .shell-title {
        font-family: "Space Grotesk", sans-serif;
        font-size: clamp(2rem, 4vw, 3.8rem);
        line-height: 0.96;
        letter-spacing: -0.04em;
        font-weight: 700;
        color: var(--ink);
        margin: 0;
    }

    .shell-subtitle {
        margin-top: 0.9rem;
        max-width: 68rem;
        font-size: 1rem;
        line-height: 1.65;
        color: var(--muted);
    }

    .shell-meta {
        display: inline-flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }

    .shell-pill {
        border-radius: 999px;
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(27, 42, 34, 0.08);
        padding: 0.45rem 0.8rem;
        font-size: 0.84rem;
        color: var(--ink);
    }

    .section-card {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
    }

    .section-label {
        font-size: 0.78rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--accent);
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .section-title {
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 0.2rem;
    }

    .section-copy {
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.55;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.93), rgba(251, 247, 240, 0.90)) !important;
        border: 1px solid rgba(27, 42, 34, 0.08) !important;
        border-radius: 22px;
        padding: 1rem 1.1rem !important;
        box-shadow: var(--shadow);
        color: var(--ink) !important;
    }

    [data-testid="stMetricValue"] {
        font-family: "Space Grotesk", sans-serif;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--ink) !important;
        letter-spacing: -0.04em;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 700 !important;
    }

    [data-testid="stDataFrame"], .stPlotlyChart, [data-testid="stExpander"] {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 22px;
        box-shadow: var(--shadow);
    }

    .stAlert {
        border-radius: 18px;
        border: 1px solid rgba(27, 42, 34, 0.08);
    }

    h1, h2, h3 {
        font-family: "Space Grotesk", sans-serif;
        color: var(--ink);
        letter-spacing: -0.03em;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255,255,255,0.45);
        border-radius: 999px;
        padding: 0.35rem;
        border: 1px solid var(--line);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.45rem 0.95rem;
    }
</style>
""",
    unsafe_allow_html=True,
)





DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

sidebar = st.sidebar
sidebar.markdown("### GreenIoT Control Center")
sidebar.caption("Orchestration energetique pour une infrastructure numerique plus resiliente")
sidebar.divider()

page = sidebar.radio(
    "Navigation",
    ["Monitoring", "Predictions", "Optimization"],
    index=0,
)

sidebar.caption(f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")
if sidebar.button("Rafraichir les donnees", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

if page == "Monitoring":
    monitoring.render()
elif page == "Predictions":
    predictions.render()
else:
    optimization.render()
