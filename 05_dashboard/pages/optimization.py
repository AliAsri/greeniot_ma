"""
IoT Energy Analytics - Load optimization page
=============================================
Shifts batch workloads toward the best solar production slots.
"""

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

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


def _build_total_solar_metrics(opt_mod, solar_df: pd.DataFrame) -> dict:
    profile = opt_mod.build_solar_profile(solar_df)
    if profile.empty:
        return {
            "max_prod_kw": 0.0,
            "avg_prod_kw": 0.0,
            "total_kwh": 0.0,
            "co2_total_solar": 0.0,
        }

    positive_profile = profile[profile["available_kw"] > 0]
    max_prod_kw = float(profile["available_kw"].max())
    avg_prod_kw = float(positive_profile["available_kw"].mean()) if not positive_profile.empty else 0.0
    total_kwh = float(profile["available_kwh"].sum())

    return {
        "max_prod_kw": max_prod_kw,
        "avg_prod_kw": avg_prod_kw,
        "total_kwh": total_kwh,
        "co2_total_solar": total_kwh * opt_mod.CO2_FACTOR,
    }


def _normalize_solar_frame(solar_df: pd.DataFrame) -> pd.DataFrame:
    if solar_df is None or solar_df.empty or "production_kw" not in solar_df.columns:
        return pd.DataFrame()

    df = solar_df.copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], format="mixed", errors="coerce")
    elif "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    else:
        return pd.DataFrame()

    return df.dropna(subset=["ts", "production_kw"]).sort_values("ts")


def _load_local_solar_candidates() -> dict[str, pd.DataFrame]:
    data_dir = Path(__file__).resolve().parents[2] / "data"
    candidates = {}
    for label, filename in {
        "local_gold": "gold_solar.parquet",
        "local_raw": "raw_solar.parquet",
    }.items():
        path = data_dir / filename
        if not path.exists():
            continue
        try:
            candidates[label] = pd.read_parquet(path)
        except Exception:
            continue
    return candidates


def _describe_solar_source(opt_mod, solar_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = _normalize_solar_frame(solar_df)
    if df.empty:
        return df, {
            "profile_rows": 0,
            "coverage_hours": 0.0,
            "freshness_minutes": float("inf"),
        }

    profile = opt_mod.build_solar_profile(df)
    coverage_hours = 0.0
    if len(profile) > 1:
        coverage_hours = (profile["slot_ts"].max() - profile["slot_ts"].min()).total_seconds() / 3600

    latest_ts = df["ts"].max()
    if getattr(latest_ts, "tzinfo", None) is not None:
        now_ts = pd.Timestamp.now(tz=latest_ts.tzinfo)
    else:
        now_ts = pd.Timestamp.now()
    freshness_minutes = max(0.0, (now_ts - latest_ts).total_seconds() / 60)

    return df, {
        "profile_rows": int(len(profile)),
        "coverage_hours": round(float(coverage_hours), 2),
        "freshness_minutes": round(float(freshness_minutes), 2),
    }


def _summarize_recent_days(opt_mod, solar_df: pd.DataFrame, max_days: int = 5) -> pd.DataFrame:
    df = _normalize_solar_frame(solar_df)
    if df.empty:
        return pd.DataFrame(columns=["day", "positive_slots", "total_kwh", "last_ts", "is_today"])

    daily_rows = []
    for day_label, day_df in df.groupby(df["ts"].dt.strftime("%Y-%m-%d"), sort=True):
        profile = opt_mod.build_solar_profile(day_df)
        positive_profile = profile[profile["available_kw"] > 0]
        daily_rows.append(
            {
                "day": day_label,
                "positive_slots": int(len(positive_profile)),
                "total_kwh": float(profile["available_kwh"].sum()) if not profile.empty else 0.0,
                "last_ts": day_df["ts"].max(),
            }
        )

    daily_df = pd.DataFrame(daily_rows).sort_values("day", ascending=False).head(max_days).copy()
    if daily_df.empty:
        return pd.DataFrame(columns=["day", "positive_slots", "total_kwh", "last_ts", "is_today"])

    latest_ts = df["ts"].max()
    if getattr(latest_ts, "tzinfo", None) is not None:
        today_label = pd.Timestamp.now(tz=latest_ts.tzinfo).strftime("%Y-%m-%d")
    else:
        today_label = pd.Timestamp.now().strftime("%Y-%m-%d")

    daily_df["is_today"] = daily_df["day"] == today_label
    return daily_df.reset_index(drop=True)


def _select_reference_day(opt_mod, solar_df: pd.DataFrame, selected_day: str | None = None) -> tuple[pd.DataFrame, dict]:
    df = _normalize_solar_frame(solar_df)
    daily_df = _summarize_recent_days(opt_mod, df, max_days=5)
    if df.empty or daily_df.empty:
        return df, {
            "selected_day": "N/A",
            "positive_slots": 0,
            "total_kwh": 0.0,
            "is_today": False,
            "available_days": 0,
        }

    if selected_day and selected_day in set(daily_df["day"].astype(str)):
        chosen = daily_df[daily_df["day"].astype(str) == str(selected_day)].iloc[0]
    else:
        eligible_days = daily_df[daily_df["positive_slots"] >= 8]
        chosen = eligible_days.iloc[0] if not eligible_days.empty else daily_df.iloc[0]

    selected_day = str(chosen["day"])
    day_df = df[df["ts"].dt.strftime("%Y-%m-%d") == selected_day].copy()

    return day_df, {
        "selected_day": selected_day,
        "positive_slots": int(chosen["positive_slots"]),
        "total_kwh": round(float(chosen["total_kwh"]), 2),
        "is_today": bool(chosen["is_today"]),
        "available_days": int(len(daily_df)),
    }


def _select_best_solar_source(opt_mod, sources: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame, dict]:
    source_priority = {
        "gold_live": 40,
        "local_gold": 30,
        "bronze_live": 20,
        "local_raw": 10,
    }
    best_name = "none"
    best_df = pd.DataFrame()
    best_meta = {
        "profile_rows": 0,
        "coverage_hours": 0.0,
        "freshness_minutes": float("inf"),
    }
    best_score = float("-inf")

    for name, raw_df in sources.items():
        normalized_df, meta = _describe_solar_source(opt_mod, raw_df)
        score = (
            min(meta["profile_rows"], 97) * 1000
            - min(meta["freshness_minutes"], 1440)
            + source_priority.get(name, 0)
        )
        if score > best_score:
            best_score = score
            best_name = name
            best_df = normalized_df
            best_meta = meta

    return best_name, best_df, best_meta


def _build_solar_plot_frame(solar_df: pd.DataFrame) -> pd.DataFrame:
    plot_solar = _normalize_solar_frame(solar_df)
    if plot_solar.empty:
        return pd.DataFrame()

    cutoff_24h = plot_solar["ts"].max() - pd.Timedelta(hours=24)
    plot_solar = plot_solar[plot_solar["ts"] >= cutoff_24h]
    if plot_solar.empty:
        return pd.DataFrame()

    if "sensor_id" in plot_solar.columns:
        return (
            plot_solar.set_index("ts")
            .groupby("sensor_id")
            .resample("15min")["production_kw"]
            .mean()
            .fillna(0.0)
            .reset_index()
            .sort_values("ts")
        )

    return (
        plot_solar.set_index("ts")
        .resample("15min")["production_kw"]
        .mean()
        .fillna(0.0)
        .reset_index()
        .sort_values("ts")
    )


def render():
    st.title("Optimisation de la Charge & Planification Solaire")
    st.caption("Planification des taches batch sur les meilleurs creneaux de production solaire")

    opt_mod = _load_optimize_module()
    solar_live_df = load_bronze_solar()
    solar_gold_df = load_gold_solar()
    candidate_sources = {
        "gold_live": solar_gold_df,
        "bronze_live": solar_live_df,
        **_load_local_solar_candidates(),
    }
    selected_source_name, solar_source_df, solar_source_meta = _select_best_solar_source(opt_mod, candidate_sources)
    recent_days = _summarize_recent_days(opt_mod, solar_source_df, max_days=5)
    runtime_mode = detect_runtime_mode()
    selected_day = None
    if not recent_days.empty:
        eligible_recent = recent_days[recent_days["positive_slots"] >= 8]
        default_day = (
            str(eligible_recent.iloc[0]["day"])
            if not eligible_recent.empty
            else str(recent_days.iloc[0]["day"])
        )
        day_options = recent_days["day"].astype(str).tolist()
        selected_day_key = "optimization_selected_day"
        if st.session_state.get(selected_day_key) not in day_options:
            st.session_state[selected_day_key] = default_day

        st.caption("Historique d'optimisation sur les 5 derniers jours disponibles")
        day_cols = st.columns(len(day_options))
        for idx, day in enumerate(day_options):
            day_row = recent_days[recent_days["day"].astype(str) == day].iloc[0]
            button_label = f"{pd.to_datetime(day).strftime('%d/%m')} | {float(day_row['total_kwh']):.0f} kWh"
            if day_cols[idx].button(
                button_label,
                key=f"opt_day_{day}",
                use_container_width=True,
                type="primary" if st.session_state[selected_day_key] == day else "secondary",
                help=f"{int(day_row['positive_slots'])} slots actifs",
            ):
                st.session_state[selected_day_key] = day

        selected_day = st.session_state[selected_day_key]

    solar_day_df, solar_day_meta = _select_reference_day(opt_mod, solar_source_df, selected_day=selected_day)
    solar_freshness = summarize_dataframe_freshness(solar_day_df, ts_col="ts")
    source_labels = {
        "gold_live": "Gold live",
        "bronze_live": "Bronze live",
        "local_gold": "Gold local",
        "local_raw": "Raw local",
        "none": "Aucune source",
    }
    st.caption(
        f"Optimisation & Solaire • Mode: {runtime_mode.upper()} • Source: {source_labels.get(selected_source_name, selected_source_name)} • "
        f"{solar_freshness.replace('Latest point', 'Dernier point')}"
    )

    if selected_source_name.startswith("local_"):
        st.info("La page utilise la source solaire la plus complete disponible, car la source live ne couvre pas correctement la derniere journee.")
    elif solar_source_meta["coverage_hours"] < 20:
        st.warning("Les donnees live couvrent une fenetre partielle. Les KPI sont calcules sur la meilleure plage disponible et non sur une journee complete.")

    if solar_day_meta["selected_day"] != "N/A":
        day_copy = f"Optimisation journaliere calculee sur la date {solar_day_meta['selected_day']}."
        if not solar_day_meta["is_today"]:
            day_copy += " Cette date correspond a la derniere journee solaire exploitable disponible."
        if solar_day_meta["available_days"] > 1:
            day_copy += f" Historique disponible: {solar_day_meta['available_days']} jours."
        st.caption(day_copy)



    render_section_card(
        label="Énergie",
        title="Capacité de Production Solaire",
        copy="",
    )

    c1, c2, c3, c4 = st.columns(4)
    capacity_metrics = _build_total_solar_metrics(opt_mod, solar_day_df)
    energy_label = "Energie du jour"
    co2_label = "CO2 potentiel du jour"
    c1.metric("Pic de production", f"{capacity_metrics['max_prod_kw']:.0f} kW")
    c2.metric("Moyenne diurne", f"{capacity_metrics['avg_prod_kw']:.0f} kW")
    c3.metric(energy_label, f"{capacity_metrics['total_kwh']:.0f} kWh")
    c4.metric(co2_label, f"{capacity_metrics['co2_total_solar']:.0f} kg")

    if "production_kw" in solar_day_df.columns:
        plot_solar = _build_solar_plot_frame(solar_day_df)

        plot_title = f"Production solaire (kW) - Journee du {solar_day_meta['selected_day']}"

        fig_solar = px.area(
            plot_solar,
            x="ts",
            y="production_kw",
            color="sensor_id" if "sensor_id" in plot_solar.columns else None,
            title=plot_title,
            color_discrete_sequence=["#f57c00", "#ffb300"],
        )
        fig_solar.update_layout(
            template="plotly_white",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_title="Heure",
            yaxis_title="Production (kW)",
        )
        if not plot_solar.empty:
            marker_ts = plot_solar["ts"].max()
            marker_label = "Maintenant" if solar_day_meta["is_today"] else "Dernier point"
            fig_solar.add_vline(
                x=marker_ts.timestamp() * 1000,
                line_dash="dash",
                line_color="#546e7a",
                annotation_text=marker_label,
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
    result = opt_mod.optimize_schedule(tasks, solar_day_df)
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
