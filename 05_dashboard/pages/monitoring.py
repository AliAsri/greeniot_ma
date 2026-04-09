"""
IoT Energy Analytics - Real-time monitoring page
================================================
Displays live energy KPIs and real-time diagnostics.
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import detect_runtime_mode, load_bronze_servers, summarize_dataframe_freshness
from utils.ui_blocks import render_section_card, render_status_card, render_takeaway_card


def _build_status_banner(df: pd.DataFrame) -> tuple[str, str, str]:
    thresholds = {"cpu_pct": 90, "temp_c": 70, "power_kw": 85}
    issues = []

    for col, threshold in thresholds.items():
        if col in df.columns and not df.empty and df[col].max() > threshold:
            racks = df[df[col] > threshold]["sensor_id"].unique().tolist()
            issues.append(f"{col} > {threshold} on {', '.join(racks)}")

    if issues:
        return (
            "Critical signals detected",
            "Live telemetry shows one or more racks above their expected operating envelope.",
            " | ".join(issues),
        )

    return (
        "System remains within normal range",
        "No critical overload or thermal excursion is currently visible in the latest telemetry window.",
        "Infrastructure status is stable across the selected racks.",
    )


def _classify_anomaly(row):
    temp = row["temp_c"] if "temp_c" in row.index else 0
    cpu = row["cpu_pct"] if "cpu_pct" in row.index else 0
    delt = row["power_delta"] if "power_delta" in row.index else 0

    if temp > 35:
        return "Thermal risk"
    if cpu > 90:
        return "Critical CPU saturation"
    if delt > 15:
        return "Abnormal power spike"
    if delt < -15:
        return "Abrupt drop"
    return "Suspicious behavior"


def render():
    st.title("Télémétrie & Monitoring Temps Réel")

    df = load_bronze_servers()
    runtime_mode = detect_runtime_mode()
    freshness = summarize_dataframe_freshness(df, ts_col="ts")
    st.caption(f"Supervision énergétique • Mode: {runtime_mode.upper()} • {freshness.replace('Latest point', 'Dernier point')}")

    banner_title, banner_copy, banner_detail = _build_status_banner(df)
    is_alert = "critical" in banner_title.lower()

    render_status_card(
        label="État d'alerte" if is_alert else "État de stabilité",
        title=banner_title,
        copy=banner_copy,
        detail=banner_detail,
        detail_color="#9f2d2d" if is_alert else "#166a4a",
        background="linear-gradient(135deg, rgba(178,39,39,0.08), rgba(255,255,255,0.92))"
        if is_alert
        else "linear-gradient(135deg, rgba(22,106,74,0.08), rgba(255,255,255,0.92))",
    )

    with st.expander("Panneau de contrôle & filtres", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            racks_dispos = sorted(df["sensor_id"].unique().tolist()) if "sensor_id" in df.columns else []
            racks = st.multiselect("Baies ciblées", racks_dispos, default=racks_dispos)
        with col_f2:
            period = st.select_slider("Fenêtre d'analyse", ["15 min", "30 min", "1h", "2h"], value="2h")
        with col_f3:
            smoothing = st.selectbox("Mode d'affichage", ["Signal brut", "Signal lisse"], index=0)

    df_racks = df[df["sensor_id"].isin(racks)].copy() if racks else df.copy()
    df_filtered = df_racks.copy()

    if not df_filtered.empty and "ts" in df_filtered.columns:
        if period == "15 min":
            cutoff = df_filtered["ts"].max() - pd.Timedelta(minutes=15)
        elif period == "30 min":
            cutoff = df_filtered["ts"].max() - pd.Timedelta(minutes=30)
        elif period == "1h":
            cutoff = df_filtered["ts"].max() - pd.Timedelta(hours=1)
        else:
            cutoff = df_filtered["ts"].min()

        df_filtered = df_filtered[df_filtered["ts"] >= cutoff]

        if smoothing == "Signal lisse":
            numeric_cols = [
                c
                for c in df_filtered.select_dtypes(include=[np.number]).columns
                if c not in ["anomaly_flag", "Score Criticite"]
            ]
            for col in numeric_cols:
                df_filtered[col] = df_filtered.groupby("sensor_id")[col].transform(
                    lambda x: x.rolling(3, min_periods=1).mean()
                )

    c1, c2, c3, c4 = st.columns(4)
    total_power = df_filtered.groupby("sensor_id")["power_kw"].last().sum() if not df_filtered.empty else 0
    avg_pue = df_filtered.groupby("sensor_id")["pue"].last().mean() if "pue" in df_filtered.columns and not df_filtered.empty else 1.45
    n_anomalies = int(df_filtered["anomaly_flag"].sum()) if "anomaly_flag" in df_filtered.columns else 0

    n_active_racks = len(df_filtered["sensor_id"].unique()) if "sensor_id" in df_filtered.columns and not df_filtered.empty else 3
    baseline_power = n_active_racks * 85.0
    co2_saved = max(0, baseline_power - total_power) * 0.7

    if "ts" in df_racks.columns and not df_racks.empty:
        now = df_racks["ts"].max()
        h_current = df_racks[df_racks["ts"] >= now - pd.Timedelta(hours=1)]
        h_previous = df_racks[
            (df_racks["ts"] >= now - pd.Timedelta(hours=2))
            & (df_racks["ts"] < now - pd.Timedelta(hours=1))
        ]
        recent_p = h_current.groupby("sensor_id")["power_kw"].mean().sum() if not h_current.empty else total_power
        older_p = h_previous.groupby("sensor_id")["power_kw"].mean().sum() if not h_previous.empty else recent_p
        power_trend_pct = ((recent_p - older_p) / max(1, abs(older_p))) * 100
    else:
        power_trend_pct = 0.0

    c1.metric("Consommation", f"{total_power:.0f} kW", delta=f"{power_trend_pct:+.1f}% vs H-1", delta_color="inverse")
    c2.metric("PUE moyen", f"{avg_pue:.2f}", delta=f"{avg_pue - 1.40:+.2f} vs objectif", delta_color="inverse")
    c3.metric("Anomalies", str(n_anomalies), delta=f"{len(df_filtered):,} points", delta_color="off")
    c4.metric("Gain CO2 estime", f"{co2_saved:.0f} kg/h", delta=f"vs baseline {baseline_power:.0f} kW", delta_color="off")

    render_section_card(
        label="Opérations",
        title="Signaux de Télémétrie",
        copy="",
        style="margin-top: 1rem;",
    )

    col_csv, _ = st.columns([1, 4])
    with col_csv:
        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Exporter CSV", csv, "monitoring_telemetry.csv", "text/csv")

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_power = px.line(
            df_filtered,
            x="ts",
            y="power_kw",
            color="sensor_id",
            title="Consommation par baie",
            labels={"power_kw": "Puissance (kW)", "ts": "Temps"},
            color_discrete_sequence=["#166a4a", "#d88b2b", "#2a7da7"],
        )
        fig_power.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02), height=410)
        st.plotly_chart(fig_power, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToAdd": ["downloadImage"]})

    with col2:
        fig_dist = px.histogram(
            df_filtered,
            x="power_kw",
            color="sensor_id",
            title="Distribution de charge",
            nbins=30,
            color_discrete_sequence=["#166a4a", "#d88b2b", "#2a7da7"],
        )
        fig_dist.update_layout(template="plotly_white", height=410, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToAdd": ["downloadImage"]})

    col3, col4 = st.columns(2)
    with col3:
        if "cpu_pct" in df_filtered.columns:
            fig_cpu = px.line(
                df_filtered,
                x="ts",
                y="cpu_pct",
                color="sensor_id",
                title="Utilisation CPU",
                color_discrete_sequence=["#166a4a", "#d88b2b", "#2a7da7"],
            )
            fig_cpu.update_layout(template="plotly_white", height=360)
            st.plotly_chart(fig_cpu, use_container_width=True)

    with col4:
        if "temp_c" in df_filtered.columns and not df_filtered.empty:
            df_heat = df_filtered.copy()
            df_heat["ts_bin"] = df_heat["ts"].dt.floor("10min")
            heat_df = df_heat.groupby(["sensor_id", "ts_bin"])["temp_c"].mean().unstack(level="ts_bin")

            fig_temp = go.Figure(
                data=go.Heatmap(
                    z=heat_df.values,
                    x=heat_df.columns,
                    y=heat_df.index,
                    colorscale="YlOrBr",
                    colorbar=dict(title="C", thickness=15),
                    hoverongaps=False,
                    zmin=30,
                    zmax=55,
                )
            )
            fig_temp.update_layout(
                title=dict(text="Carte thermique", font=dict(size=14)),
                template="plotly_white",
                height=360,
                xaxis_title="Temps",
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_temp, use_container_width=True)

    render_section_card(
        label="Efficacité",
        title="Power Usage Effectiveness (PUE)",
        copy="",
        style="margin-top: 0.8rem;",
    )

    col5, col6 = st.columns([1, 2])
    with col5:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=avg_pue,
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={"reference": 1.40, "increasing": {"color": "red"}, "decreasing": {"color": "#166a4a"}},
                title={"text": "PUE actuel"},
                gauge={
                    "axis": {"range": [1.0, 2.5]},
                    "bar": {"color": "#166a4a"},
                    "steps": [
                        {"range": [1.0, 1.2], "color": "#e8f4eb"},
                        {"range": [1.2, 1.5], "color": "#fff1dc"},
                        {"range": [1.5, 2.0], "color": "#f8d9c3"},
                        {"range": [2.0, 2.5], "color": "#efb0a7"},
                    ],
                    "threshold": {"line": {"color": "#d14d3f", "width": 4}, "value": 1.40},
                },
            )
        )
        fig_gauge.update_layout(
            height=310, 
            template="plotly_white",
            margin=dict(l=30, r=30, t=50, b=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col6:
        if "pue" in df_filtered.columns and not df_filtered.empty:
            pue_by_time = df_filtered.groupby(df_filtered["ts"].dt.floor("5min"))["pue"].mean().reset_index()

            fig_pue = px.line(
                pue_by_time,
                x="ts",
                y="pue",
                title="Evolution consolidee du PUE",
                color_discrete_sequence=["#166a4a"],
            )
            fig_pue.update_traces(line=dict(width=3))
            fig_pue.add_hline(y=1.40, line_dash="dash", line_color="#d14d3f", annotation_text="Objectif 1.40")
            fig_pue.update_yaxes(range=[1.2, 1.8])
            fig_pue.update_layout(template="plotly_white", height=310)
            st.plotly_chart(fig_pue, use_container_width=True)
            if not pue_by_time.empty and pue_by_time["pue"].std() < 0.01:
                st.caption("Le PUE varie peu sur la fenetre selectionnee ; la courbe affiche donc un signal reel presque plat.")

    if "anomaly_flag" in df_filtered.columns:
        anomalies = df_filtered[df_filtered["anomaly_flag"] == 1].copy()
        if not anomalies.empty:
            render_section_card(
                label="Diagnostics",
                title="Anomalies Actives",
                copy="",
                style="margin-top: 0.8rem;",
            )

            stat1, stat2, stat3 = st.columns(3)
            stat1.metric("Evenements detectes", f"{len(anomalies)}")
            stat2.metric("Baies observees", f"{anomalies['sensor_id'].nunique() if 'sensor_id' in anomalies.columns else 0}")
            stat3.metric("Fenetre", period)

            if "power_delta" in anomalies.columns:
                anomalies["Score Criticite"] = (anomalies["power_delta"].abs() / 2).clip(1, 10).round(1)
            else:
                anomalies["Score Criticite"] = 5.0

            anomalies["Type d'Anomalie"] = anomalies.apply(_classify_anomaly, axis=1)
            anomalies = anomalies.sort_values("ts", ascending=False)

            col_a, col_b = st.columns([1.1, 1.2])
            with col_a:
                display_cols = ["ts", "sensor_id", "Type d'Anomalie", "Score Criticite", "power_kw", "temp_c", "cpu_pct"]
                available_cols = [c for c in display_cols if c in anomalies.columns]
                st.dataframe(
                    anomalies[available_cols].head(15),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Score Criticite": st.column_config.ProgressColumn("Gravite", min_value=1, max_value=10, format="%.1f"),
                        "temp_c": st.column_config.NumberColumn("Temperature", format="%.1f"),
                        "power_kw": st.column_config.NumberColumn("Puissance", format="%.1f"),
                        "cpu_pct": st.column_config.NumberColumn("CPU", format="%.1f"),
                    },
                )

            with col_b:
                cause_df = (
                    anomalies["Type d'Anomalie"]
                    .value_counts()
                    .rename_axis("Cause")
                    .reset_index(name="Count")
                )
                fig_causes = px.bar(
                    cause_df,
                    x="Count",
                    y="Cause",
                    orientation="h",
                    title="Repartition des causes",
                    color="Count",
                    color_continuous_scale=["#f4d8b6", "#d88b2b", "#9f2d2d"],
                )
                fig_causes.update_layout(template="plotly_white", height=360, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_causes, use_container_width=True)
