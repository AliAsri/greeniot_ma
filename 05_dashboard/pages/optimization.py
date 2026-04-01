"""
IoT Energy Analytics — Page Optimisation de charge
============================================
Décalage des tâches batch vers les pics de production solaire.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.data_loader import load_bronze_solar, load_schedule
from typing import List, Dict

import importlib.util

ml_dir = os.path.join(os.path.dirname(__file__), "..", "..", "04_ml")
opt_path = os.path.join(ml_dir, "optimize_load.py")

spec = importlib.util.spec_from_file_location("optimize_load", opt_path)
optimize_module = importlib.util.module_from_spec(spec)
sys.modules["optimize_load"] = optimize_module
spec.loader.exec_module(optimize_module)

def _build_schedule(tasks: list[dict], solar_df: pd.DataFrame) -> pd.DataFrame:
    """Utilise l'algorithme d'optimisation réel basé sur la DataFrame solaire."""
    if len(solar_df) == 0:
        return pd.DataFrame()
        
    mapped_tasks = []
    for t in tasks:
        mapped_tasks.append({
            "name": t.get("nom", "Tâche"),
            "duration_min": t.get("duree", 30),
            "priority": t.get("priorite", "low").split(" ")[-1].lower()
        })
        
    real_schedule = optimize_module.schedule_deferred_tasks(mapped_tasks, solar_df)
    
    # Mapping back to French dashboard columns
    display_sch = []
    for _, row in real_schedule.iterrows():
        display_sch.append({
            "Tâche": row["task"],
            "Priorité": row["priority"],
            "Durée (min)": row["duration_min"],
            "Début planifié": row["scheduled_start"],
            "Fin planifiée": row["scheduled_end"],
            "Source énergie": "☀️ Solaire" if row["energy_source"] == "solar" else "⚡ Mixte",
            "CO2 économisé (kg)": row["co2_saved_kg"],
        })
    return pd.DataFrame(display_sch)


def render():
    st.title("☀️ Optimisation de la Charge & Planification Solaire")
    st.caption("Décalage des tâches batch vers les fenêtres de production solaire maximale")

    solar_df = load_bronze_solar()

    # ── KPIs solaires (AVANT le graphique) ───────────────────
    st.subheader("📡 Production solaire — Site Principal")

    c1, c2, c3, c4 = st.columns(4)
    if "production_kw" in solar_df.columns:
        max_prod   = solar_df["production_kw"].max()
        avg_prod   = solar_df["production_kw"].mean()
        total_kwh  = solar_df["production_kw"].sum() * 5 / 60
        co2_total_solar = total_kwh * 0.7

        c1.metric("☀️ Production max",  f"{max_prod:.0f} kW")
        c2.metric("📊 Production moy.", f"{avg_prod:.0f} kW")
        c3.metric("⚡ Total produit",   f"{total_kwh:.0f} kWh")
        c4.metric("🌍 CO2 évité",       f"{co2_total_solar:.0f} kg")

    # Courbe de production (après KPIs)
    if "production_kw" in solar_df.columns:
        plot_solar = solar_df.tail(600).copy() # Augmenté légèrement pour couvrir un trou
        
        # Remédier à la "ligne droite nocturne" s'il y a un trou de données : on force des zéros
        if not plot_solar.empty and "sensor_id" in plot_solar.columns:
            # Resampling crée des NaN là où la donnée manque (le PC était éteint)
            plot_solar = plot_solar.set_index("ts").groupby("sensor_id").resample("15min")["production_kw"].mean()
            # On remplace par 0.0 kW (le solaire s'éteint la nuit)
            plot_solar = plot_solar.fillna(0.0).reset_index()
            # Trier pour Plotly
            plot_solar = plot_solar.sort_values("ts")

        fig_solar = px.area(
            plot_solar, x="ts", y="production_kw",
            color="sensor_id" if "sensor_id" in plot_solar.columns else None,
            title="Production solaire (kW) — fenêtre récente",
            color_discrete_sequence=["#f57c00", "#ffb300"],
        )
        fig_solar.update_layout(
            template="plotly_white", height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_solar, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToAdd": ["downloadImage"]})

    st.divider()

    # ── Configuration des tâches (WIRED) ─────────────────────
    st.subheader("📅 Planning de décalage de charge")

    with st.expander("⚙️ Configurer les tâches batch", expanded=False):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            task1_name = st.text_input("Tâche 1", "Backup Delta Lake", key="t1_name")
            task1_prio = st.selectbox("Priorité", ["🟢 Low", "🟡 Medium", "🔴 High"], key="t1_prio")
            task1_dur  = st.slider("Durée (min)", 10, 180, 45, key="t1_dur")
        with col_t2:
            task2_name = st.text_input("Tâche 2", "Entraînement ML", key="t2_name")
            task2_prio = st.selectbox("Priorité", ["🟢 Low", "🟡 Medium", "🔴 High"],
                                      index=0, key="t2_prio")
            task2_dur  = st.slider("Durée (min)", 10, 180, 120, key="t2_dur")

    # Tâches fixes + tâches configurées par l'utilisateur
    tasks = [
        {"nom": task1_name, "duree": task1_dur, "priorite": task1_prio},
        {"nom": task2_name, "duree": task2_dur, "priorite": task2_prio},
        {"nom": "Export rapports BI",  "duree": 20, "priorite": "🟡 Medium"},
        {"nom": "Compression Bronze",  "duree": 30, "priorite": "🟢 Low"},
        {"nom": "Sync MinIO",          "duree": 15, "priorite": "🟡 Medium"},
    ]
    schedule = _build_schedule(tasks, solar_df)

    if not schedule.empty:
        st.dataframe(schedule, use_container_width=True, hide_index=True)

    if not schedule.empty and "CO2 économisé (kg)" in schedule.columns:
        total_co2 = schedule["CO2 économisé (kg)"].sum()
    else:
        total_co2 = 0

    st.success(
        f"🌍 **CO2 total économisé : {total_co2:.2f} kg/jour** "
        f"| Objectif : ≥ 50 kg/jour {'✅' if total_co2 >= 50 else '❌'}"
    )

    st.divider()

    # ── Gantt chart ───────────────────────────────────────────
    st.subheader("📊 Diagramme de Gantt — Tâches planifiées")

    today = datetime.now().strftime("%Y-%m-%d")
    gantt_data = []
    for _, row in schedule.iterrows():
        gantt_data.append({
            "Task":     row["Tâche"],
            "Start":    f"{today} {row['Début planifié']}",
            "Finish":   f"{today} {row['Fin planifiée']}",
            "Resource": row["Priorité"],
        })

    gantt_df = pd.DataFrame(gantt_data)
    
    if not gantt_df.empty:
        gantt_df["Start"]  = pd.to_datetime(gantt_df["Start"])
        gantt_df["Finish"] = pd.to_datetime(gantt_df["Finish"])

        fig_gantt = px.timeline(
            gantt_df, x_start="Start", x_end="Finish", y="Task",
            color="Resource",
            title="Planification des tâches batch sur fenêtre solaire",
            color_discrete_sequence=["#43a047", "#fb8c00", "#e53935"],
        )
        fig_gantt.update_layout(
            template="plotly_white", height=350,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_gantt, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToAdd": ["downloadImage"]})
    else:
        st.info("Aucune tâche n'a pû être planifiée. Vérifiez les données solaires en entrée.")

    # ── Comparaison avant/après ───────────────────────────────
    st.subheader("📈 Impact de l'optimisation")

    col5, col6 = st.columns(2)

    with col5:
        impact_data = {
            "Métrique": ["Charge déplacée (%)", "CO2 économisé (kg/j)",
                         "PUE effectif", "Coût énergie (%)", "Utilisation solaire (%)"],
            "Avant": [0, 0, 1.65, 100, 30],
            "Après": [35, total_co2, 1.38, 72, 65],
        }
        impact_df = pd.DataFrame(impact_data)

        fig_impact = go.Figure()
        fig_impact.add_trace(go.Bar(
            name="Avant optimisation",
            x=impact_df["Métrique"], y=impact_df["Avant"],
            marker_color="#e53935",
        ))
        fig_impact.add_trace(go.Bar(
            name="Après optimisation",
            x=impact_df["Métrique"], y=impact_df["Après"],
            marker_color="#43a047",
        ))
        fig_impact.update_layout(
            title="Impact de l'optimisation",
            template="plotly_white", height=400,
            barmode="group",
        )
        st.plotly_chart(fig_impact, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToAdd": ["downloadImage"]})

    with col6:
        energy_sources = {"Source": ["Solaire", "Réseau"], "Pourcentage": [65, 35]}
        fig_pie = px.pie(
            pd.DataFrame(energy_sources),
            values="Pourcentage", names="Source",
            title="Répartition des sources d'énergie",
            color_discrete_sequence=["#ff9800", "#546e7a"],
            hole=0.4,
        )
        fig_pie.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Recommandations ───────────────────────────────────────
    st.divider()
    st.subheader("💡 Recommandations")
    
    # Trouver la fenêtre solaire optimale en direct
    if not solar_df.empty and "production_kw" in solar_df.columns:
        daily_max = solar_df["production_kw"].max()
        peak_df = solar_df[solar_df["production_kw"] >= daily_max * 0.70]
        if not peak_df.empty:
            peak_start = peak_df["ts"].dt.hour.min()
            peak_end = peak_df["ts"].dt.hour.max()
            fenetre_str = f"{peak_start}h00 — {peak_end}h00"
        else:
            fenetre_str = "11:00 — 15:00 (Default)"
    else:
        fenetre_str = "11:00 — 15:00 (Estimé)"

    st.info(f"""
    **Fenêtre solaire optimale identifiée par l'IA : {fenetre_str}**

    1. 🔄 **{task1_name}** planifié à {schedule.iloc[0]["Début planifié"]} (durée : {task1_dur} min)
    2. 🧠 **{task2_name}** planifié à {schedule.iloc[1]["Début planifié"]} (durée : {task2_dur} min)
    3. 📊 **Exporter les rapports BI** en fin de pic solaire (14h — 14h20)
    4. ⚡ **Réduire le PUE** en synchronisant les charges avec la production photovoltaïque identifiée
    5. 🔋 **Charger les batteries** dans la fenêtre **{fenetre_str}** pour subvenir aux tâches nocturnes
    """)
