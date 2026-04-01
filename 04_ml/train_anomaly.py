"""
GreenIoT-MA — Détection d'anomalies (Isolation Forest)
=========================================================
Détecte automatiquement les pics de consommation anormaux :
- Surchauffe
- Fuite d'énergie
- Comportement anormal

L'Isolation Forest est adapté aux données IoT non supervisées.
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
from dotenv import load_dotenv
from deltalake import DeltaTable

load_dotenv()
warnings.filterwarnings("ignore")

MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_ANOMALY", "greeniot_anomaly_detection")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
GOLD_PATH = os.getenv("DELTA_GOLD", "s3a://greeniot/gold").replace("s3a://", "s3://") + "/servers"

STORAGE_OPTIONS = {
    "AWS_ACCESS_KEY_ID":        os.getenv("MINIO_ACCESS_KEY", "greeniot"),
    "AWS_SECRET_ACCESS_KEY":    os.getenv("MINIO_SECRET_KEY", "greeniot2030"),
    "AWS_ENDPOINT_URL":         os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
    "AWS_REGION":               "us-east-1",
    "AWS_ALLOW_HTTP":           "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}


def train_anomaly_detector():
    """Entraîne le détecteur d'anomalies Isolation Forest."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    print("🌿 GreenIoT-MA — Entraînement du détecteur d'anomalies")
    print(f"   MLflow: {MLFLOW_URI}")
    print(f"   Expérience: {EXPERIMENT}\n")

    # ── Chargement données Gold ───────────────────────────────
    print(f"   📡 Lecture Gold Delta depuis : {GOLD_PATH}")
    try:
        dt = DeltaTable(GOLD_PATH, storage_options=STORAGE_OPTIONS)
        available = [f.name for f in dt.schema().fields]
        
        # Features pour la détection d'anomalies
        features = ["power_kw", "cpu_pct", "ram_pct", "power_delta", "power_kw_std5", "temp_c"]
        cols_to_load = [f for f in features if f in available] + ["sensor_id", "ts"]
        if "anomaly_flag" in available:
            cols_to_load.append("anomaly_flag")
            
        df = dt.to_pyarrow_table(columns=list(set(cols_to_load))).to_pandas()
        
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values(["sensor_id", "ts"]).reset_index(drop=True)
        df = df.dropna(subset=[f for f in features if f in df.columns])
    except Exception as e:
        print(f"   ⚠️  Erreur lors de la lecture de {GOLD_PATH} : {e}")
        return

    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        print(f"   ⚠️  Colonnes manquantes: {set(features) - set(available_features)}")
        features = available_features

    print(f"   📊 Dataset: {len(df)} lignes, {len(features)} features")
    print(f"   📋 Features: {features}\n")

    X = df[features].fillna(0).values

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with mlflow.start_run(run_name="IsolationForest_greeniot"):
        # ── Hyperparamètres ───────────────────────────────────
        params = {
            "contamination": 0.05,
            "n_estimators": 200,
            "max_samples": "auto",
            "max_features": 1.0,
            "random_state": 42,
        }
        mlflow.log_params(params)

        # ── Entraînement ─────────────────────────────────────
        model = IsolationForest(**params)
        preds = model.fit_predict(X_scaled)

        # -1 = anomalie, 1 = normal
        n_anomalies = (preds == -1).sum()
        n_normal = (preds == 1).sum()
        anomaly_rate = n_anomalies / len(preds) * 100

        # Scores d'anomalie (plus négatif = plus anormal)
        scores = model.decision_function(X_scaled)

        # ── Métriques ────────────────────────────────────────
        mlflow.log_metrics({
            "n_anomalies_detected": int(n_anomalies),
            "n_normal": int(n_normal),
            "anomaly_rate_pct": round(anomaly_rate, 2),
            "mean_anomaly_score": float(scores[preds == -1].mean()) if n_anomalies > 0 else 0,
            "mean_normal_score": float(scores[preds == 1].mean()),
        })

        # ── Comparaison avec anomaly_flag existant ───────────
        if "anomaly_flag" in df.columns:
            true_labels = df["anomaly_flag"].values
            pred_labels = (preds == -1).astype(int)

            report = classification_report(true_labels, pred_labels, output_dict=True)
            mlflow.log_metrics({
                "precision": report["1"]["precision"] if "1" in report else 0,
                "recall": report["1"]["recall"] if "1" in report else 0,
                "f1_score": report["1"]["f1-score"] if "1" in report else 0,
            })

            cm = confusion_matrix(true_labels, pred_labels)
            if cm.shape == (2, 2):
                fp_rate = cm[0, 1] / max(1, cm[0, 0] + cm[0, 1]) * 100
                mlflow.log_metric("false_positive_rate_pct", round(fp_rate, 2))

            print("   📊 Classification Report :")
            print(classification_report(true_labels, pred_labels,
                                        target_names=["Normal", "Anomalie"]))

        # ── Sauvegarde ───────────────────────────────────────
        mlflow.sklearn.log_model(model, "isolation_forest")

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "anomaly_detector.pkl")
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "features": features,
            "params": params,
        }, model_path)

        # ── Annoter le DataFrame avec les prédictions ────────
        df["if_anomaly"] = (preds == -1).astype(int)
        df["anomaly_score"] = scores
        anomaly_df = df[df["if_anomaly"] == 1]

        # Export des anomalies pour le rapport
        report_path = os.path.join(DATA_DIR, "anomalies_detected.csv")
        anomaly_df.to_csv(report_path, index=False)

        print(f"   ✅ Anomalies détectées : {n_anomalies} / {len(preds)} "
              f"({anomaly_rate:.1f}%)")
        print(f"   📂 Modèle sauvé : {model_path}")
        print(f"   📄 Anomalies exportées : {report_path}")


if __name__ == "__main__":
    train_anomaly_detector()
