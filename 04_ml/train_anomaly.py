"""
GreenIoT-MA - Anomaly detection
===============================
Detects abnormal consumption patterns:
- overheating
- energy leaks
- unusual behaviour

Uses supervised XGBoost when labels are available,
otherwise falls back to Isolation Forest.
"""

import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from deltalake import DeltaTable
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler

load_dotenv()
warnings.filterwarnings("ignore")

MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_ANOMALY", "greeniot_anomaly_detection")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
GOLD_PATH = os.getenv("DELTA_GOLD", "s3a://greeniot/gold").replace("s3a://", "s3://") + "/servers"

STORAGE_OPTIONS = {
    "AWS_ACCESS_KEY_ID": os.getenv("MINIO_ACCESS_KEY", "greeniot"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("MINIO_SECRET_KEY", "greeniot2030"),
    "AWS_ENDPOINT_URL": os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}


def train_anomaly_detector():
    """Train the anomaly detector."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    print("GreenIoT-MA - Entrainement du detecteur d'anomalies")
    print(f"   MLflow: {MLFLOW_URI}")
    print(f"   Experience: {EXPERIMENT}\n")

    print(f"   Reading Gold Delta from: {GOLD_PATH}")
    try:
        dt = DeltaTable(GOLD_PATH, storage_options=STORAGE_OPTIONS)
        available = [f.name for f in dt.schema().fields]

        supervised_candidates = [
            "power_kw",
            "cpu_pct",
            "ram_pct",
            "temp_c",
            "power_kw_avg5",
            "cpu_avg5",
            "power_kw_avg12",
            "power_kw_avg24",
            "power_kw_lag1",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "is_business_hours",
            "is_weekend",
        ]
        rule_features = ["power_delta", "power_kw_std5"]
        cols_to_load = [f for f in supervised_candidates + rule_features if f in available] + ["sensor_id", "ts"]
        if "anomaly_flag" in available:
            cols_to_load.append("anomaly_flag")

        df = dt.to_pyarrow_table(columns=list(dict.fromkeys(cols_to_load))).to_pandas()
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values(["sensor_id", "ts"]).reset_index(drop=True)
        available_candidates = [f for f in supervised_candidates if f in df.columns]
        available_rule_features = [f for f in rule_features if f in df.columns]
        df = df.dropna(subset=available_candidates or available_rule_features)
    except Exception as exc:
        print(f"   Error while reading {GOLD_PATH}: {exc}")
        return

    has_labels = "anomaly_flag" in df.columns
    features = available_candidates.copy() if has_labels else available_candidates + available_rule_features
    missing_candidates = set(supervised_candidates) - set(available_candidates)
    if missing_candidates:
        print(f"   Missing columns: {missing_candidates}")

    if not has_labels and {"power_delta", "power_kw_std5"}.issubset(df.columns):
        std_safe = df["power_kw_std5"].abs().replace(0, np.nan)
        df["power_delta_abs"] = df["power_delta"].abs()
        df["delta_std_ratio"] = (df["power_delta_abs"] / std_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        features.extend(["power_delta_abs", "delta_std_ratio"])

    print(f"   Dataset: {len(df)} lignes, {len(features)} features")
    print(f"   Features: {features}\n")

    X = df[features].fillna(0).values

    if has_labels:
        print("   Labels anomaly_flag detectes: entrainement supervise XGBoost")
        print("   Les variables qui servent a fabriquer le label heuristique sont exclues des features supervisees.")
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier

        y = df["anomaly_flag"].values
        positive_count = int(y.sum())
        positive_rate = positive_count / max(1, len(y)) * 100
        print(f"   Labels positifs: {positive_count} / {len(y)} ({positive_rate:.2f}%)")
        if positive_count < 150 or positive_rate < 0.5:
            print("   Avertissement: le dataset reste tres desequilibre; les metriques de la classe anomalie seront instables.")
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.25,
            stratify=y_train_full,
            random_state=42,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)
        X_scaled = scaler.transform(X)

        pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))

        with mlflow.start_run(run_name="XGBoost_Anomaly_Supervised"):
            params = {
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "min_child_weight": 3,
                "gamma": 0.2,
                "reg_lambda": 2.0,
                "scale_pos_weight": pos_weight,
                "random_state": 42,
                "eval_metric": "aucpr",
            }
            mlflow.log_params(params)

            model = XGBClassifier(**params)
            model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_valid_scaled, y_valid)],
                verbose=False,
            )

            valid_scores = model.predict_proba(X_valid_scaled)[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(y_valid, valid_scores)
            threshold = 0.5
            best_f1 = -1.0
            min_recall = 0.70

            for precision, recall, current_threshold in zip(precisions[:-1], recalls[:-1], thresholds):
                if recall < min_recall:
                    continue
                f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    threshold = float(current_threshold)

            if best_f1 < 0:
                for precision, recall, current_threshold in zip(precisions[:-1], recalls[:-1], thresholds):
                    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        threshold = float(current_threshold)

            test_scores = model.predict_proba(X_test_scaled)[:, 1]
            test_preds = (test_scores >= threshold).astype(int)
            preds = (model.predict_proba(X_scaled)[:, 1] >= threshold).astype(int)
            scores = model.predict_proba(X_scaled)[:, 1]

            n_anomalies = int(sum(test_preds))
            anomaly_rate = n_anomalies / len(test_preds) * 100
            report = classification_report(y_test, test_preds, output_dict=True)
            avg_precision = average_precision_score(y_test, test_scores)

            mlflow.log_metrics(
                {
                    "decision_threshold": threshold,
                    "n_anomalies_detected": n_anomalies,
                    "anomaly_rate_pct": round(anomaly_rate, 2),
                    "precision": report["1"]["precision"] if "1" in report else 0,
                    "recall": report["1"]["recall"] if "1" in report else 0,
                    "f1_score": report["1"]["f1-score"] if "1" in report else 0,
                    "average_precision": avg_precision,
                    "validation_best_f1": best_f1,
                }
            )

            cm = confusion_matrix(y_test, test_preds)
            if cm.shape == (2, 2):
                fp_rate = cm[0, 1] / max(1, cm[0, 0] + cm[0, 1]) * 100
                mlflow.log_metric("false_positive_rate_pct", round(fp_rate, 2))

            print(f"   Decision threshold calibre: {threshold:.3f}")
            print("   Classification Report (XGBoost supervise) :")
            print(classification_report(y_test, test_preds, target_names=["Normal", "Anomalie"]))

            mlflow.sklearn.log_model(model, "xgboost_anomaly")
            model_type = "supervised_xgboost"
    else:
        print("   Aucun label detecte: entrainement non supervise Isolation Forest")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
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

            preds = (preds_if == -1).astype(int)
            scores = -model.decision_function(X_scaled)

            n_anomalies = int(sum(preds))
            anomaly_rate = n_anomalies / len(preds) * 100

            mlflow.log_metrics(
                {
                    "n_anomalies_detected": n_anomalies,
                    "anomaly_rate_pct": round(anomaly_rate, 2),
                }
            )

            mlflow.sklearn.log_model(model, "isolation_forest")
            model_type = "unsupervised_isolation_forest"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "anomaly_detector.pkl")
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": features,
            "params": params,
            "type": model_type,
            "threshold": threshold if has_labels else None,
        },
        model_path,
    )

    df["if_anomaly"] = preds
    df["anomaly_score"] = scores
    anomaly_df = df[df["if_anomaly"] == 1]

    report_path = os.path.join(DATA_DIR, "anomalies_detected.csv")
    anomaly_df.to_csv(report_path, index=False)

    print(f"   Anomalies detectees : {int(sum(preds))} / {len(preds)}")
    print(f"   Modele sauve : {model_path}")
    print(f"   Anomalies exportees : {report_path}")


if __name__ == "__main__":
    train_anomaly_detector()
