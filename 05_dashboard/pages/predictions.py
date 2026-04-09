"""
IoT Energy Analytics - Prediction page
=====================================
Visualizes consumption forecasting and model behavior.
"""

import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn

from utils.data_loader import (
    detect_runtime_mode,
    load_anomalies,
    load_gold_servers,
    summarize_dataframe_freshness,
)
from utils.ui_blocks import render_section_card, render_takeaway_card


class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden=64, layers=2, horizon=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, horizon),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _format_metric(value, suffix: str = "", precision: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}{suffix}"


def _build_live_predictions(df: pd.DataFrame):
    root_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    models_dir = os.path.join(root_dir, "models")
    xgb_path = os.path.join(models_dir, "xgboost_predictor.pkl")
    scaler_path = os.path.join(models_dir, "scaler_prediction.pkl")
    lstm_path = os.path.join(models_dir, "lstm_predictor.pt")

    n_pts = min(200, len(df))
    sample_df = df.sort_values("ts").tail(n_pts).copy() if "power_kw" in df.columns else pd.DataFrame()

    xgb_metrics = {}
    lstm_metrics = {}
    xgb_payload_for_fi = None
    features_for_fi = None
    inference_status = "Prediction artifacts are unavailable. Live forecasts are disabled until models are trained."

    if os.path.exists(xgb_path) and os.path.exists(scaler_path) and not sample_df.empty:
        try:
            xgb_payload = joblib.load(xgb_path)
            scaler_payload = joblib.load(scaler_path)

            models = xgb_payload["models"]
            features = xgb_payload["features"]
            scaler = scaler_payload["scaler"]

            xgb_payload_for_fi = xgb_payload
            features_for_fi = features

            lstm_model = None
            if os.path.exists(lstm_path):
                lstm_model = LSTMPredictor(input_size=len(features))
                lstm_model.load_state_dict(torch.load(lstm_path, map_location=torch.device("cpu")))
                lstm_model.eval()

            saved_metrics = xgb_payload.get("metrics", {})
            if saved_metrics:
                xgb_metrics.update(saved_metrics)

            lstm_metrics_path = os.path.join(models_dir, "lstm_metrics.pkl")
            if os.path.exists(lstm_metrics_path):
                saved_lstm_metrics = joblib.load(lstm_metrics_path)
                lstm_metrics.update(saved_lstm_metrics)

            df_sorted = df.sort_values(["sensor_id", "ts"])
            results = []

            for _, group in df_sorted.groupby("sensor_id"):
                group = group.tail(150)
                if len(group) < 25:
                    continue

                clean_group = group[features].ffill().bfill().fillna(0)
                group_scaled = scaler.transform(clean_group)

                model = models[0] if isinstance(models, list) else list(models.values())[0]
                X_seq = []
                valid_idx = []
                for i in range(24, len(group_scaled)):
                    X_seq.append(group_scaled[i - 24 : i].flatten())
                    valid_idx.append(group.index[i])

                if not X_seq:
                    continue

                preds_scaled = model.predict(np.array(X_seq))
                dummy = np.zeros((len(preds_scaled), len(features)))
                dummy[:, 0] = preds_scaled
                preds_unscaled = scaler.inverse_transform(dummy)[:, 0]

                res_df = group.loc[valid_idx].copy()
                res_df["pred_xgb"] = preds_unscaled

                if lstm_model is not None:
                    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32).view(-1, 24, len(features))
                    with torch.no_grad():
                        preds_lstm_scaled = lstm_model(X_tensor).numpy()[:, 0]

                    dummy_lstm = np.zeros((len(preds_lstm_scaled), len(features)))
                    dummy_lstm[:, 0] = preds_lstm_scaled
                    res_df["pred_lstm"] = scaler.inverse_transform(dummy_lstm)[:, 0]

                results.append(res_df)

            if results:
                sample_df = pd.concat(results)
                if "pred_xgb" in sample_df.columns:
                    xgb_metrics.setdefault("mae", float(np.mean(np.abs(sample_df["power_kw"] - sample_df["pred_xgb"]))))
                    xgb_metrics.setdefault("rmse", float(np.sqrt(np.mean((sample_df["power_kw"] - sample_df["pred_xgb"]) ** 2))))
                if "pred_lstm" in sample_df.columns:
                    lstm_metrics.setdefault("mae", float(np.mean(np.abs(sample_df["power_kw"] - sample_df["pred_lstm"]))))
                    lstm_metrics.setdefault("rmse", float(np.sqrt(np.mean((sample_df["power_kw"] - sample_df["pred_lstm"]) ** 2))))
                inference_status = "Live inference available from local model artifacts."
            else:
                inference_status = "Model artifacts loaded, but not enough recent sequences were available."

        except Exception as exc:
            inference_status = f"Live inference temporarily unavailable: {exc}"

    if "pred_xgb" not in sample_df.columns and "pred_lstm" not in sample_df.columns:
        sample_df = pd.DataFrame()

    return sample_df, lstm_metrics, xgb_metrics, xgb_payload_for_fi, features_for_fi, inference_status


def _extract_feature_importance(xgb_payload, features):
    if xgb_payload is None or features is None:
        return None, None

    try:
        models = xgb_payload.get("models", None)
        if models is None:
            return None, None

        model = models[0] if isinstance(models, list) else list(models.values())[0]
        if not hasattr(model, "feature_importances_"):
            return None, None

        raw_imp = model.feature_importances_
        n_features = len(features)
        if n_features <= 0 or len(raw_imp) % n_features != 0:
            return None, None

        aggregated = {}
        for flat_idx, importance in enumerate(raw_imp):
            _, feature_idx = divmod(flat_idx, n_features)
            feature_name = features[feature_idx]
            aggregated[feature_name] = aggregated.get(feature_name, 0.0) + float(importance)

        ranked = sorted(aggregated.items(), key=lambda item: item[1], reverse=True)[: min(10, len(aggregated))]
        return [name for name, _ in ranked], [round(score, 4) for _, score in ranked]
    except Exception:
        return None, None


def render():
    st.title("Analyse Predictive de la Consommation")
    st.caption("Prevision de charge electrique a court horizon avec LSTM et XGBoost")

    df = load_gold_servers()
    exported_anomalies = load_anomalies()
    runtime_mode = detect_runtime_mode()
    freshness = summarize_dataframe_freshness(df, ts_col="ts")

    st.caption(f"Studio prédictif • Mode: {runtime_mode.upper()} • {freshness.replace('Latest point', 'Dernier point')}")

    sample_df, lstm_metrics, xgb_metrics, xgb_payload, fi_features_src, inference_status = _build_live_predictions(df)
    st.info(inference_status)

    lstm_mae = lstm_metrics.get("mae")
    xgb_mae = xgb_metrics.get("mae")
    lstm_won = lstm_mae is not None and xgb_mae is not None and lstm_mae < xgb_mae
    xgb_won = lstm_mae is not None and xgb_mae is not None and xgb_mae <= lstm_mae



    render_section_card(
        label="Modèles",
        title="Métriques de Performance",
        copy="",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### LSTM {'Gagnant' if lstm_won else ''}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", _format_metric(lstm_metrics.get("mae"), " kW"))
        m2.metric("RMSE", _format_metric(lstm_metrics.get("rmse"), " kW"))
        m3.metric("R2", _format_metric(lstm_metrics.get("r2")))
        m4.metric("MAPE", _format_metric(lstm_metrics.get("mape"), "%", precision=1))

    with col2:
        st.markdown(f"### XGBoost {'Gagnant' if xgb_won else ''}")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("MAE", _format_metric(xgb_metrics.get("mae"), " kW"))
        m6.metric("RMSE", _format_metric(xgb_metrics.get("rmse"), " kW"))
        m7.metric("R2", _format_metric(xgb_metrics.get("r2")))
        m8.metric("MAPE", _format_metric(xgb_metrics.get("mape"), "%", precision=1))

    tab_live, tab_errors, tab_signals = st.tabs(["Prevision live", "Profil d'erreur", "Signaux & drivers"])

    with tab_live:
        render_section_card(
            label="Prévision live",
            title="Charge observée vs charge prédite",
            copy="",
        )

        if sample_df is not None and not sample_df.empty:
            racks_dispos = sorted(sample_df["sensor_id"].unique().tolist()) if "sensor_id" in sample_df.columns else []
            selected_rack = st.selectbox("Baie ciblee", racks_dispos) if racks_dispos else None
            smooth_signal = st.toggle("Lisser le signal observe", value=True)

            plot_df = sample_df[sample_df["sensor_id"] == selected_rack].sort_values("ts") if selected_rack else sample_df.sort_values("ts")
            if not plot_df.empty:
                plot_df = plot_df.set_index("ts").resample("15s").mean(numeric_only=True).reset_index()
            if smooth_signal and not plot_df.empty:
                plot_df["power_kw"] = plot_df["power_kw"].rolling(window=4, min_periods=1).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df["ts"], y=plot_df["power_kw"], name="Observed", mode="lines", line=dict(color="#166a4a", width=3)))
            if "pred_lstm" in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df["ts"], y=plot_df["pred_lstm"], name="LSTM +1", mode="lines", line=dict(color="#d88b2b", width=2, dash="dash")))
            if "pred_xgb" in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df["ts"], y=plot_df["pred_xgb"], name="XGBoost +1", mode="lines", line=dict(color="#2a7da7", width=2, dash="dot")))
            fig.update_layout(
                title=f"Puissance observee vs predite {f'- {selected_rack}' if selected_rack else ''}",
                xaxis_title="Temps",
                yaxis_title="Puissance (kW)",
                template="plotly_white",
                height=470,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToAdd": ["downloadImage"]})

            csv = sample_df.to_csv(index=False).encode("utf-8")
            st.download_button("Exporter predictions CSV", csv, "predictions_ml.csv", "text/csv")
        else:
            st.warning("Aucune vue predictive recente n'est disponible.")

    with tab_errors:
        render_section_card(
            label="Profil d'erreur",
            title="Distribution des résidus",
            copy="",
        )

        if sample_df is not None and not sample_df.empty:
            hist_data = []
            if "pred_lstm" in sample_df.columns:
                hist_data.append(pd.DataFrame({"Error (kW)": sample_df["power_kw"] - sample_df["pred_lstm"], "Model": "LSTM"}))
            if "pred_xgb" in sample_df.columns:
                hist_data.append(pd.DataFrame({"Error (kW)": sample_df["power_kw"] - sample_df["pred_xgb"], "Model": "XGBoost"}))

            if hist_data:
                hist_df = pd.concat(hist_data)
                col_a, col_b = st.columns(2)
                with col_a:
                    fig_err = px.histogram(
                        hist_df,
                        x="Error (kW)",
                        color="Model",
                        barmode="overlay",
                        nbins=50,
                        opacity=0.72,
                        title="Superposition des residus",
                        color_discrete_map={"LSTM": "#d88b2b", "XGBoost": "#2a7da7"},
                    )
                    fig_err.update_layout(template="plotly_white", height=410, legend=dict(orientation="h", yanchor="bottom", y=1.02))
                    st.plotly_chart(fig_err, use_container_width=True)

                with col_b:
                    if lstm_metrics and xgb_metrics:
                        summary_df = pd.DataFrame(
                            {
                                "Metric": ["MAE", "RMSE", "MAPE"],
                                "LSTM": [lstm_metrics.get("mae"), lstm_metrics.get("rmse"), lstm_metrics.get("mape")],
                                "XGBoost": [xgb_metrics.get("mae"), xgb_metrics.get("rmse"), xgb_metrics.get("mape")],
                            }
                        )
                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Bar(name="LSTM", x=summary_df["Metric"], y=summary_df["LSTM"], marker_color="#d88b2b"))
                        fig_compare.add_trace(go.Bar(name="XGBoost", x=summary_df["Metric"], y=summary_df["XGBoost"], marker_color="#2a7da7"))
                        fig_compare.update_layout(template="plotly_white", height=410, barmode="group", title="Comparaison des metriques")
                        st.plotly_chart(fig_compare, use_container_width=True)
                    else:
                        st.info("Les metriques de validation ne sont pas disponibles tant que les artefacts d'entrainement ne sont pas presents.")
            else:
                st.warning("L'analyse des residus n'est pas disponible.")

    with tab_signals:
        render_section_card(
            label="Signaux",
            title="Analyse des features & Anomalies",
            copy="",
        )

        col_left, col_right = st.columns(2)
        with col_left:
            if "anomaly_flag" in df.columns:
                n_total = len(df)
                n_anomalies = int(df["anomaly_flag"].sum())
                anomaly_rate = n_anomalies / max(1, n_total) * 100

                a1, a2, a3 = st.columns(3)
                a1.metric("Total points", f"{n_total:,}")
                a2.metric("Anomalies", f"{n_anomalies:,}")
                a3.metric("Taux", f"{anomaly_rate:.1f}%")

                if "power_kw" in df.columns and "cpu_pct" in df.columns:
                    anomalies_df = df[df["anomaly_flag"] == 1]
                    normal_df = df[df["anomaly_flag"] == 0]
                    sampled_normals = normal_df.sample(n=min(3000, len(normal_df)), random_state=42) if not normal_df.empty else normal_df
                    plot_df = pd.concat([anomalies_df, sampled_normals])
                    plot_df["status"] = plot_df["anomaly_flag"].map({0: "Normal", 1: "Anomaly"})

                    fig_scatter = px.scatter(
                        plot_df,
                        x="cpu_pct",
                        y="power_kw",
                        color="status",
                        title="Champ d'anomalies CPU vs consommation",
                        color_discrete_map={"Normal": "#2a7da7", "Anomaly": "#d14d3f"},
                        opacity=0.62,
                    )
                    fig_scatter.update_layout(template="plotly_white", height=410)
                    st.plotly_chart(fig_scatter, use_container_width=True)
            elif not exported_anomalies.empty:
                st.dataframe(exported_anomalies.head(25), use_container_width=True, hide_index=True)
            else:
                st.info("Aucun signal d'anomalie disponible.")

        with col_right:
            fi_features, fi_importance = _extract_feature_importance(xgb_payload, fi_features_src)
            if fi_features is None:
                st.info("L'importance des features sera disponible une fois les artefacts XGBoost charges.")
            else:
                fig_imp = px.bar(
                    x=fi_importance,
                    y=fi_features,
                    orientation="h",
                    title="Importance des features - XGBoost",
                    labels={"x": "Importance", "y": "Feature"},
                    color_discrete_sequence=["#166a4a"],
                )
                fig_imp.update_layout(template="plotly_white", height=410, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)
