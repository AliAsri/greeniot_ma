"""
IoT Energy Analytics - Load optimization page
=============================================
Shifts batch workloads toward the best solar production slots.
"""

import importlib.util
import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import detect_runtime_mode, load_bronze_solar, load_gold_solar, summarize_dataframe_freshness
from utils.ui_blocks import render_section_card, render_takeaway_card


@st.cache_resource
def _load_optimize_module():
    """Load optimize_load.py once and reuse it."""
    ml_dir = os.path.join(os.path.dirname(__file__), "..", "..", "04_ml")
    opt_path = os.path.join(ml_dir, "optimize_load.py")
    spec = importlib.util.spec_from_file_location("optimize_load", opt_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["optimize_load"] = mod
    spec.loader.exec_module(mod)
    return mod


def _normalize_priority(raw_priority: str) -> str:
    label = str(raw_priority).split(" ")[-1].lower()
    return label if label in {"low", "medium", "high"} else "low"


def _priority_badge(priority: str) -> str:
    return {
        "low": "Low",
        "medium": "Medium",
        "high": "High",
    }.get(priority, priority.title())


def _build_tasks(task1_name, task1_dur, task1_prio, task2_name, task2_dur, task2_prio):
    return [
        {
            "name": task1_name,
            "duration_min": task1_dur,
            "priority": _normalize_priority(task1_prio),
            "power_kw_required": 40 if _normalize_priority(task1_prio) == "low" else 55,
        },
        {
            "name": task2_name,
            "duration_min": task2_dur,
            "priority": _normalize_priority(task2_prio),
            "power_kw_required": 75 if _normalize_priority(task2_prio) == "high" else 60,
        },
        {
            "name": "Export rapports BI",
            "duration_min": 20,
            "priority": "medium",
            "power_kw_required": 25,
        },
        {
            "name": "Compression Bronze",
            "duration_min": 30,
            "priority": "low",
            "power_kw_required": 30,
        },
        {
            "name": "Sync MinIO",
            "duration_min": 15,
            "priority": "medium",
            "power_kw_required": 35,
        },
    ]


def _build_display_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame()

    display = schedule.copy()
    display["Priorite"] = display["priority"].map(_priority_badge)
    display["Debut planifie"] = pd.to_datetime(display["scheduled_start_ts"]).dt.strftime("%Y-%m-%d %H:%M")
    display["Fin planifiee"] = pd.to_datetime(display["scheduled_end_ts"]).dt.strftime("%Y-%m-%d %H:%M")
    display["Source energie"] = display["energy_source"].map(
        {
            "solar": "Solaire",
            "hybrid": "Hybride",
            "mixed": "Mixte",
        }
    )

    return display[
        [
            "task",
            "Priorite",
            "duration_min",
            "power_kw_required",
            "Debut planifie",
            "Fin planifiee",
            "Source energie",
            "solar_share_pct",
            "solar_energy_kwh",
            "grid_energy_kwh",
            "co2_saved_kg",
        ]
    ].rename(
        columns={
            "task": "Tache",
            "duration_min": "Duree (min)",
            "power_kw_required": "Puissance requise (kW)",
            "solar_share_pct": "Part solaire (%)",
            "solar_energy_kwh": "Energie solaire (kWh)",
            "grid_energy_kwh": "Energie reseau (kWh)",
            "co2_saved_kg": "CO2 economise (kg)",
        }
    )


def render():
    st.title("Optimisation de la Charge & Planification Solaire")
    st.caption("Planification des taches batch sur les meilleurs creneaux de production solaire")

    solar_df = load_bronze_solar()
    solar_gold_df = load_gold_solar()
    if solar_gold_df.empty:
        solar_gold_df = solar_df
    runtime_mode = detect_runtime_mode()
    solar_freshness = summarize_dataframe_freshness(solar_df, ts_col="ts")
    st.caption(f"Optimisation & Solaire • Mode: {runtime_mode.upper()} • {solar_freshness.replace('Latest point', 'Dernier point')}")



    render_section_card(
        label="Énergie",
        title="Capacité de Production Solaire",
        copy="",
    )

    c1, c2, c3, c4 = st.columns(4)
    if "production_kw" in solar_df.columns:
        solar_day = solar_df[solar_df["ts"].dt.hour.between(6, 18)] if "ts" in solar_df.columns else solar_df
        solar_day = solar_day if not solar_day.empty else solar_df

        max_prod = solar_day["production_kw"].max()
        avg_prod = solar_day[solar_day["production_kw"] > 0]["production_kw"].mean()
        total_kwh = solar_df["production_kw"].sum() * 5 / 60
        co2_total_solar = total_kwh * 0.7

        c1.metric("Pic de production", f"{max_prod:.0f} kW")
        c2.metric("Moyenne diurne", f"{avg_prod:.0f} kW")
        c3.metric("Total produit", f"{total_kwh:.0f} kWh")
        c4.metric("CO2 evite", f"{co2_total_solar:.0f} kg")

    if "production_kw" in solar_df.columns:
        plot_solar = solar_df.copy()
        if not plot_solar.empty and "sensor_id" in plot_solar.columns:
            plot_solar = (
                plot_solar.set_index("ts")
                .groupby("sensor_id")
                .resample("15min")["production_kw"]
                .mean()
                .fillna(0.0)
                .reset_index()
                .sort_values("ts")
            )

        fig_solar = px.area(
            plot_solar,
            x="ts",
            y="production_kw",
            color="sensor_id" if "sensor_id" in plot_solar.columns else None,
            title="Production solaire (kW) - Fenetre 24h",
            color_discrete_sequence=["#f57c00", "#ffb300"],
        )
        fig_solar.update_layout(
            template="plotly_white",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_title="Heure",
            yaxis_title="Production (kW)",
        )
        fig_solar.add_vline(
            x=datetime.now().timestamp() * 1000,
            line_dash="dash",
            line_color="#546e7a",
            annotation_text="Maintenant",
            annotation_position="top left",
        )
        st.plotly_chart(
            fig_solar,
            use_container_width=True,
            config={"displayModeBar": True, "modeBarButtonsToAdd": ["downloadImage"]},
        )

    st.divider()
    render_section_card(
        label="Configuration",
        title="Paramètres de Décalage (Load Shifting)",
        copy="",
    )

    with st.expander("Configurer les taches batch", expanded=False):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            task1_name = st.text_input("Tache 1", "Backup Delta Lake", key="t1_name")
            task1_prio = st.selectbox("Priorite", ["Low", "Medium", "High"], key="t1_prio")
            task1_dur = st.slider("Duree (min)", 10, 180, 45, key="t1_dur")
        with col_t2:
            task2_name = st.text_input("Tache 2", "Entrainement ML", key="t2_name")
            task2_prio = st.selectbox("Priorite ", ["Low", "Medium", "High"], index=2, key="t2_prio")
            task2_dur = st.slider("Duree (min) ", 10, 180, 120, key="t2_dur")

    tasks = _build_tasks(task1_name, task1_dur, task1_prio, task2_name, task2_dur, task2_prio)
    opt_mod = _load_optimize_module()
    result = opt_mod.optimize_schedule(tasks, solar_gold_df)
    peaks = result["peaks"]
    schedule = result["schedule"]
    summary = result["summary"]

    peak_col1, peak_col2, peak_col3, peak_col4 = st.columns(4)
    peak_col1.metric("Fenetre optimale", f"{peaks['optimal_start']} - {peaks['optimal_end']}")
    peak_col2.metric("Puissance solaire attendue", f"{peaks['expected_kw']:.1f} kW")
    peak_col3.metric("Energie fenetre", f"{summary['window_energy_kwh']:.1f} kWh")
    peak_col4.metric("CO2 economise", f"{summary['total_co2_saved_kg']:.2f} kg/j")

    st.success(
        f"Charge couverte par le solaire : {summary['load_shift_pct']:.1f}% | "
        f"Part solaire moyenne par tache : {summary['avg_solar_share_pct']:.1f}%"
    )

    display_schedule = _build_display_schedule(schedule)
    tab_plan, tab_impact, tab_notes = st.tabs(["Plan d'execution", "Impact", "Recommandations"])

    with tab_plan:
        render_section_card(
            label="Exécution",
            title="Allocation détaillée des charges",
            copy="",
        )
        if not display_schedule.empty:
            st.dataframe(display_schedule, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune tache n'a pu etre planifiee. Verifiez les donnees solaires en entree.")

        if not schedule.empty:
            gantt_df = pd.DataFrame(
                {
                    "Task": schedule["task"],
                    "Start": pd.to_datetime(schedule["scheduled_start_ts"]),
                    "Finish": pd.to_datetime(schedule["scheduled_end_ts"]),
                    "Resource": schedule["priority"].map(_priority_badge),
                    "Source": schedule["energy_source"],
                }
            )

            fig_gantt = px.timeline(
                gantt_df,
                x_start="Start",
                x_end="Finish",
                y="Task",
                color="Source",
                title="Planification des taches batch sur la meilleure capacite solaire",
                color_discrete_map={
                    "solar": "#2f7d4a",
                    "hybrid": "#d88b2b",
                    "mixed": "#6f7d76",
                },
            )
            fig_gantt.update_layout(
                template="plotly_white",
                height=380,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(
                fig_gantt,
                use_container_width=True,
                config={"displayModeBar": True, "modeBarButtonsToAdd": ["downloadImage"]},
            )

    with tab_impact:
        render_section_card(
            label="Mesures",
            title="Impact Énergétique & Environnemental",
            copy="",
        )
        col5, col6 = st.columns(2)

        with col5:
            impact_df = pd.DataFrame(
                {
                    "Metrique": [
                        "Charge deplacee (%)",
                        "CO2 economise (kg/j)",
                        "Part solaire moyenne (%)",
                        "Energie solaire (kWh)",
                        "Energie reseau (kWh)",
                    ],
                    "Valeur": [
                        summary["load_shift_pct"],
                        summary["total_co2_saved_kg"],
                        summary["avg_solar_share_pct"],
                        summary["total_solar_kwh"],
                        summary["total_grid_kwh"],
                    ],
                }
            )

            fig_impact = px.bar(
                impact_df,
                x="Metrique",
                y="Valeur",
                title="Impact derive du planning calcule",
                color="Metrique",
                color_discrete_sequence=["#2f7d4a", "#55a870", "#1f6e8c", "#d88b2b", "#6f7d76"],
            )
            fig_impact.update_layout(template="plotly_white", height=420, showlegend=False)
            st.plotly_chart(fig_impact, use_container_width=True)

        with col6:
            energy_mix = pd.DataFrame(
                {
                    "Source": ["Solaire", "Reseau"],
                    "kWh": [summary["total_solar_kwh"], summary["total_grid_kwh"]],
                }
            )
            fig_pie = px.pie(
                energy_mix,
                values="kWh",
                names="Source",
                title="Repartition energetique reelle du planning",
                color_discrete_sequence=["#d88b2b", "#6f7d76"],
                hole=0.58,
            )
            fig_pie.update_layout(height=420, template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab_notes:
        render_section_card(
            label="Rapports",
            title="Journal de Synthèse du Planning",
            copy="",
        )
        if not schedule.empty:
            lines = []
            for _, row in schedule.head(3).iterrows():
                lines.append(
                    f"- {row['task']} -> {pd.to_datetime(row['scheduled_start_ts']).strftime('%Y-%m-%d %H:%M')} "
                    f"({row['solar_share_pct']:.0f}% solaire, {row['energy_source']})"
                )

            st.info(
                "\n".join(
                    [
                        f"Fenetre optimale identifiee: {peaks['optimal_start_ts']} -> {peaks['optimal_end_ts']}",
                        f"Taches planifiees: {summary['tasks_scheduled']} | charge deplacee: {summary['load_shift_pct']:.1f}%",
                        f"Energie solaire mobilisee: {summary['total_solar_kwh']:.1f} kWh | reseau: {summary['total_grid_kwh']:.1f} kWh",
                        "Top recommandations:",
                        *lines,
                    ]
                )
            )
        else:
            st.warning("Donnees solaires insuffisantes pour generer des recommandations de planification.")
