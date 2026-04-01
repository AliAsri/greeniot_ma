"""
GreenIoT-MA — Détection d'anomalies (XGBoost / Isolation Forest)
=========================================================
Détecte automatiquement les pics de consommation anormaux :
- Surchauffe
- Fuite d'énergie
- Comportement anormal

Le script utilise un XGBoost Supervisé s'il détecte des labels,
sinon il utilise une approche non supervisée (Isolation Forest).
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

    has_labels = "anomaly_flag" in df.columns

    if has_labels:
        print("   🎯 Labels 'anomaly_flag' détectés : Entraînement Supervisé (XGBoost Classifier)")
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        
        y = df["anomaly_flag"].values
        # Gérer le déséquilibre de classe
        pos_weight = (len(y) - sum(y)) / max(1, sum(y))
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

        with mlflow.start_run(run_name="XGBoost_Anomaly_Supervised"):
            params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "scale_pos_weight": pos_weight,
                "random_state": 42,
            }
            mlflow.log_params(params)

            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            preds = model.predict(X_scaled)
            scores = model.predict_proba(X_scaled)[:, 1] # Probabilité d'être anomalie
            
            n_anomalies = int(sum(preds))
            anomaly_rate = n_anomalies / len(preds) * 100

            true_labels = df["anomaly_flag"].values
            report = classification_report(true_labels, preds, output_dict=True)
            
            mlflow.log_metrics({
                "n_anomalies_detected": n_anomalies,
                "anomaly_rate_pct": round(anomaly_rate, 2),
                "precision": report["1"]["precision"] if "1" in report else 0,
                "recall": report["1"]["recall"] if "1" in report else 0,
                "f1_score": report["1"]["f1-score"] if "1" in report else 0,
            })

            cm = confusion_matrix(true_labels, preds)
            if cm.shape == (2, 2):
                fp_rate = cm[0, 1] / max(1, cm[0, 0] + cm[0, 1]) * 100
                mlflow.log_metric("false_positive_rate_pct", round(fp_rate, 2))

            print("   📊 Classification Report (XGBoost Supervisé) :")
            print(classification_report(true_labels, preds, target_names=["Normal", "Anomalie"]))

            mlflow.sklearn.log_model(model, "xgboost_anomaly")
            model_type = "supervised_xgboost"

    else:
        print("   🕵️‍♂️ Aucun label détecté : Entraînement Non-Supervisé (Isolation Forest)")
        with mlflow.start_run(run_name="IsolationForest_greeniot"):
            params = {
                "contamination": 0.05,
                "n_estimators": 200,
                "max_samples": "auto",
                "max_features": 1.0,
                "random_state": 42,
            }
            mlflow.log_params(params)

            model = IsolationForest(**params)
            preds_if = model.fit_predict(X_scaled)
            
            # Convertir -1/1 en 1/0
            preds = (preds_if == -1).astype(int)
            scores = -model.decision_function(X_scaled) # Plus c'est positif, plus c'est anormal
            
            n_anomalies = int(sum(preds))
            anomaly_rate = n_anomalies / len(preds) * 100
            
            mlflow.log_metrics({
                "n_anomalies_detected": n_anomalies,
                "anomaly_rate_pct": round(anomaly_rate, 2),
            })
            
            mlflow.sklearn.log_model(model, "isolation_forest")
            model_type = "unsupervised_isolation_forest"

    # ── Sauvegarde ───────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "anomaly_detector.pkl")
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "features": features,
        "params": params,
        "type": model_type
    }, model_path)

    # ── Annoter le DataFrame avec les prédictions ────────
    df["if_anomaly"] = preds
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
