"""
IoT Energy Analytics — Page Monitoring temps réel
==========================================
Affiche les KPIs énergétiques et les graphiques en quasi-temps réel.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import load_bronze_servers


def render():
    st.title("🔋 Télémétrie et Analyse Énergétique en Temps Réel")
    st.caption("Surveillance de l'infrastructure Cloud — Centre de Données Principal")

    df = load_bronze_servers()

    # ── Système d'Alertes Temps Réel ──────────────────────────
    SEUILS = {"cpu_pct": 90, "temp_c": 70, "power_kw": 85}
    alertes_actives = False
    for col, seuil in SEUILS.items():
        if col in df.columns and df[col].max() > seuil:
            alertes_actives = True
            racks_alerte = df[df[col] > seuil]['sensor_id'].unique()
            st.error(f"🚨 **ALERTE CRITIQUE :** `{col}` a franchi le seuil de sécurité (> {seuil}) sur les baies {list(racks_alerte)} !")
    
    if not alertes_actives:
        st.success("✅ Aucun problème critique de surchauffe ou de surcharge CPU détecté.")

    # ── Panneau de Contrôle (Filtres) ─────────────────────────
    with st.expander("🎛️ Panneau de Contrôle & Filtres", expanded=True):
        st.markdown("<style>div[data-testid='stExpander'] { border-radius: 12px; border: 1px solid #e2e8f0; }</style>", unsafe_allow_html=True)
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            racks_dispos = sorted(df["sensor_id"].unique().tolist()) if "sensor_id" in df.columns else []
            racks = st.multiselect("🎯 Cibler baies serveurs", racks_dispos, default=racks_dispos)
        with col_f2:
            period = st.select_slider("⏱️ Fenêtre d'analyse", ["15 min", "30 min", "1h", "2h"], value="2h")
        with col_f3:
            smoothing = st.selectbox("Type d'affichage (Lissage)", ["Données brutes", "Lissage (Moyenne Mobile)"], index=0)
    
    # Garder une version non tronquée par le temps pour des KPIs (H vs H-1) équitables
    df_racks = df[df["sensor_id"].isin(racks)].copy() if racks else df.copy()
    
    # Filtrer temporellement pour les graphiques
    df_filtered = df_racks.copy()
    if not df_filtered.empty and "ts" in df_filtered.columns:
        if period == "15 min":
            cutoff = df_filtered["ts"].max() - pd.Timedelta(minutes=15)
        elif period == "30 min":
            cutoff = df_filtered["ts"].max() - pd.Timedelta(minutes=30)
        elif period == "1h":
            cutoff = df_filtered["ts"].max() - pd.Timedelta(hours=1)
        else:
            cutoff = df_filtered["ts"].min() # 2h max
        df_filtered = df_filtered[df_filtered["ts"] >= cutoff]
        
        # Application du lissage si demandé (purement visuel pour les graphiques)
        if smoothing == "Lissage (Moyenne Mobile)":
            # BUG FIX : Ne SURTOUT PAS lisser les catégories booléennes !
            numeric_cols = [c for c in df_filtered.select_dtypes(include=[np.number]).columns if c not in ["anomaly_flag", "Score Criticité"]]
            for col in numeric_cols:
                df_filtered[col] = df_filtered.groupby("sensor_id")[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
    else:
        df_filtered = df_racks.copy()
    # ── KPIs ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    # Total power
    total_power = df_filtered.groupby("sensor_id")["power_kw"].mean().sum()
    avg_pue = df_filtered["pue"].mean() if "pue" in df_filtered.columns else 1.45
    n_anomalies = int(df_filtered["anomaly_flag"].sum()) if "anomaly_flag" in df_filtered.columns else 0
    
    # Baseline power proportionnelle au nombre de racks (Max théorique réaliste ~85kW par baie)
    n_active_racks = len(df_filtered["sensor_id"].unique()) if "sensor_id" in df_filtered.columns and not df_filtered.empty else 3
    baseline_power = n_active_racks * 85.0
    co2_saved = max(0, (baseline_power - total_power)) * 0.7

    # Trend rigoureux : H courante vs H précédente (Généré sur le dataset complet 2h, indépendant du filtre UI)
    if "ts" in df_racks.columns and not df_racks.empty:
        now = df_racks["ts"].max()
        h_courante = df_racks[df_racks["ts"] >= now - pd.Timedelta(hours=1)]
        h_precedente = df_racks[(df_racks["ts"] >= now - pd.Timedelta(hours=2)) & (df_racks["ts"] < now - pd.Timedelta(hours=1))]
        recent_p = h_courante.groupby("sensor_id")["power_kw"].mean().sum() if not h_courante.empty else total_power
        older_p = h_precedente.groupby("sensor_id")["power_kw"].mean().sum() if not h_precedente.empty else recent_p
        power_trend_pct = ((recent_p - older_p) / max(1, abs(older_p))) * 100
    else:
        power_trend_pct = 0.0

    c1.metric(
        "⚡ Consommation",
        f"{total_power:.0f} kW",
        delta=f"{power_trend_pct:+.1f}% vs H-1",
        delta_color="inverse",
    )
    c2.metric(
        "📊 PUE moyen",
        f"{avg_pue:.2f}",
        delta=f"{avg_pue - 1.40:+.2f} vs objectif 1.40",
        delta_color="inverse",
    )
    c3.metric(
        "⚠️ Anomalies",
        str(n_anomalies),
        delta=f"sur {len(df_filtered):,} points",
        delta_color="off",
    )
    c4.metric(
        "🌍 CO2 économisé",
        f"{co2_saved:.0f} kg/h",
        delta=f"vs baseline {baseline_power:.0f} kW",
        delta_color="off",
    )
    
    st.divider()

    col_csv, _ = st.columns([1, 4])
    with col_csv:
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Exporter vue CSV", csv, "monitoring_telemetry.csv", "text/csv")

    st.divider()

    # ── Graphique consommation par rack ────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        fig_power = px.line(
            df_filtered, x="ts", y="power_kw", color="sensor_id",
            title="Consommation par rack (kW)",
            labels={"power_kw": "Puissance (kW)", "ts": "Temps"},
            color_discrete_sequence=["#0288d1", "#43a047", "#f57c00"],
        )
        fig_power.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=400,
        )
        st.plotly_chart(fig_power, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToAdd": ["downloadImage"]})

    with col2:
        fig_dist = px.histogram(
            df_filtered, x="power_kw", color="sensor_id",
            title="Distribution",
            nbins=30,
            color_discrete_sequence=["#0288d1", "#43a047", "#f57c00"],
        )
        fig_dist.update_layout(template="plotly_white", height=400, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToAdd": ["downloadImage"]})

    # ── CPU & Température ─────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        if "cpu_pct" in df_filtered.columns:
            fig_cpu = px.line(
                df_filtered, x="ts", y="cpu_pct", color="sensor_id",
                title="Utilisation CPU (%)",
                color_discrete_sequence=["#0288d1", "#43a047", "#f57c00"],
            )
            fig_cpu.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig_cpu, use_container_width=True)

    with col4:
        if "temp_c" in df_filtered.columns:
            df_heat = df_filtered.copy()
            df_heat["ts_bin"] = df_heat["ts"].dt.floor("10min")
            heat_df = df_heat.groupby(["sensor_id", "ts_bin"])["temp_c"].mean().unstack(level="ts_bin")
            
            fig_temp = go.Figure(data=go.Heatmap(
                z=heat_df.values,
                x=heat_df.columns,
                y=heat_df.index,
                colorscale="YlOrRd", # Plus lumineux et "alerte" (Yellow/Orange/Red)
                colorbar=dict(title="°C", thickness=15),
                hoverongaps=False,
                zmin=30, zmax=55 # Plage dynamique réaliste pour des serveurs (au lieu de 20-80 qui écrasait le contraste)
            ))
            fig_temp.update_layout(
                title=dict(text="🌡️ Carte Thermique", font=dict(size=14)),
                template="plotly_white", 
                height=350,
                xaxis_title="Heure",
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_temp, use_container_width=True)

    # ── PUE Gauge ─────────────────────────────────────────────
    st.subheader("📊 Power Usage Effectiveness (PUE)")
    col5, col6 = st.columns([1, 2])

    with col5:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_pue,
            delta={"reference": 1.40, "increasing": {"color": "red"}, "decreasing": {"color": "#43a047"}},
            title={"text": "PUE actuel"},
            gauge={
                "axis": {"range": [1.0, 2.5]},
                "bar": {"color": "#0288d1"},
                "steps": [
                    {"range": [1.0, 1.2], "color": "#e8f5e9"},
                    {"range": [1.2, 1.5], "color": "#fff3e0"},
                    {"range": [1.5, 2.0], "color": "#ffccbc"},
                    {"range": [2.0, 2.5], "color": "#ef9a9a"},
                ],
                "threshold": {"line": {"color": "#d32f2f", "width": 4}, "value": 1.40},
            },
        ))
        fig_gauge.update_layout(height=300, template="plotly_white")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col6:
        if "pue" in df_filtered.columns:
            pue_by_time = df_filtered.groupby(df_filtered["ts"].dt.floor("5min"))["pue"].mean().reset_index()
            fig_pue = px.area(
                pue_by_time, x="ts", y="pue",
                title="Évolution du PUE consolidé",
                color_discrete_sequence=["#0288d1"],
            )
            fig_pue.add_hline(y=1.40, line_dash="dash", line_color="red",
                              annotation_text="Objectif 1.40")
            fig_pue.update_yaxes(range=[1.2, 1.8]) # Zoom vertical sur la plage d'intérêt du PUE
            fig_pue.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig_pue, use_container_width=True)

    # ── Anomalies ─────────────────────────────────────────────
    if "anomaly_flag" in df_filtered.columns:
        anomalies = df_filtered[df_filtered["anomaly_flag"] == 1].copy()
        if not anomalies.empty:
            st.divider()
            st.subheader("⚠️ Console d'Alertes et Diagnostics")
            st.warning(f"🔴 {len(anomalies)} anomalies détectées dans les 2 dernières heures")

            # --- Classification Métier des Anomalies ---
            # 1. Calcul du Score de Criticité (de 1 à 10) basé sur l'amplitude du problème
            if "power_delta" in anomalies.columns:
                anomalies["Score Criticité"] = (anomalies["power_delta"].abs() / 2).clip(1, 10).round(1)
            else:
                anomalies["Score Criticité"] = 5.0
                
            # 2. Identification de la cause (Type d'Anomalie)
            def classify_anomaly(row):
                if row.get("temp_c", 0) > 35:
                    return "🔥 Surchauffe (Risque Thermique)"
                elif row.get("cpu_pct", 0) > 90:
                    return "💻 Surcharge CPU Critique"
                elif row.get("power_delta", 0) > 15:
                    return "⚡ Pic de Consommation Anormal"
                elif row.get("power_delta", 0) < -15:
                    return "📉 Chute Brutale (Panne possible)"
                else:
                    return "🧩 Comportement Suspect (Désalignement PUE/Modèle)"
                    
            anomalies["Type d'Anomalie"] = anomalies.apply(classify_anomaly, axis=1)
            
            # 3. Tri pour afficher les plus récentes en haut
            anomalies = anomalies.sort_values("ts", ascending=False)
            
            PAGE_SIZE = 15
            total = len(anomalies)
            if total > PAGE_SIZE:
                st.caption(f"Affichage des {PAGE_SIZE} événements les plus récents sur {total} anomalies totales.")

            display_cols = ["ts", "sensor_id", "Type d'Anomalie", "Score Criticité", "power_kw", "temp_c", "cpu_pct"]
            available_cols = [c for c in display_cols if c in anomalies.columns]

            # Custom styling pour la dataframe Streamlit
            st.dataframe(
                anomalies[available_cols].head(PAGE_SIZE),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score Criticité": st.column_config.ProgressColumn(
                        "Gravité (1-10)", min_value=1, max_value=10, format="%.1f"
                    ),
                    "temp_c": st.column_config.NumberColumn("Température (°C)", format="%.1f"),
                    "power_kw": st.column_config.NumberColumn("Puissance (kW)", format="%.1f"),
                    "cpu_pct": st.column_config.NumberColumn("Charge CPU (%)", format="%.1f"),
                }
            )
