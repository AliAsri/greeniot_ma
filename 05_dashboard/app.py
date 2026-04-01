"""
IoT Energy Analytics Platform — Dashboard Streamlit
====================================
Application principale avec 3 pages :
1. Monitoring temps réel — KPIs énergétiques
2. Prédictions ML — LSTM vs XGBoost
3. Optimisation charge — Décalage solaire
"""

import streamlit as st
import os
from datetime import datetime
from pages import monitoring, predictions, optimization

st.set_page_config(
    page_title="IoT Energy Analytics Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé (compatible dark theme) ──────────────────
st.markdown("""
<style>
    /* Headers with Gradient */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #0288d1, #43a047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: fadeIn 0.8s ease-in-out;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #546e7a;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* True Glassmorphism / Modern Card Light Theme for Metric Cards */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px 0 rgba(0, 0, 0, 0.03);
        color: #263238 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    /* Interactive Hover Effect */
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px 0 rgba(2, 136, 209, 0.12);
        border: 1px solid rgba(2, 136, 209, 0.3) !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800;
        color: #0f172a !important;
    }

    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-size: 0.85rem !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600;
        font-size: 0.9rem !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Gestion du mode Demo Global
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
if DEMO_MODE:
    st.info("🎯 Mode démonstration actif — les données affichées sont simulées localement (sans MinIO/DeltaLake).", icon="ℹ️")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("# ⚡ GreenIoT Analytics")
st.sidebar.markdown("**Système d'Optimisation Énergétique par IA**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["🔋 Monitoring temps réel", "📈 Prédictions ML", "☀️ Optimisation charge"],
    index=0,
)

st.sidebar.divider()

# ── Bouton de rafraîchissement global (une seule fois, dans la sidebar) ──
st.sidebar.caption(f"⏱️ Dernière mise à jour  : {datetime.now().strftime('%H:%M:%S')}")
if st.sidebar.button("🔄 Rafraîchir les données", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

# ── Informations contextuelles ────────────────────────────────
if page == "🔋 Monitoring temps réel":
    st.sidebar.info("💡 **Contexte :**\nSurveillez la santé thermodynamique et computationnelle des baies en quasi temps réel (raffraîchissement 30s).")
elif page == "📈 Prédictions ML":
    st.sidebar.success("💡 **Contexte :**\nL'inférence PyTorch en temps réel offre une visibilité projetée à +15 minutes sur la charge énergétique.")
elif page == "☀️ Optimisation charge":
    st.sidebar.warning("💡 **Contexte :**\nLe Green Shift consiste à déplacer les workloads massifs vers les fenêtres d'ensoleillement maximal.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Projet :** Optimisation Énergétique  
**Cible :** Neutralité carbone 2035  
**Site :** Centre de Données Principal
""")
st.sidebar.caption("IoT Energy Analytics v1.0")

# ── Router vers les pages ─────────────────────────────────────
if page == "🔋 Monitoring temps réel":
    monitoring.render()
elif page == "📈 Prédictions ML":
    predictions.render()
elif page == "☀️ Optimisation charge":
    optimization.render()
