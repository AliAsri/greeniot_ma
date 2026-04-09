import os
import sys

import joblib
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ROOT_DIR)
DASHBOARD_DIR = os.path.join(PROJECT_DIR, "05_dashboard")
sys.path.insert(0, DASHBOARD_DIR)

from utils.data_loader import load_gold_servers


def main():
    df = load_gold_servers()
    models_dir = os.path.join(PROJECT_DIR, "models")
    xgb_path = os.path.join(models_dir, "xgboost_predictor.pkl")
    scaler_path = os.path.join(models_dir, "scaler_prediction.pkl")

    if df.empty:
        raise RuntimeError("No Gold server data available to inspect predictions.")
    if not os.path.exists(xgb_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Prediction artifacts are missing from the models directory.")

    xgb_payload = joblib.load(xgb_path)
    scaler_payload = joblib.load(scaler_path)

    models = xgb_payload["models"]
    features = xgb_payload["features"]
    scaler = scaler_payload["scaler"]

    df_sorted = df.sort_values(["sensor_id", "ts"])
    for sid, group in df_sorted.groupby("sensor_id"):
        group = group.tail(150)
        clean_group = group[features].ffill().bfill().fillna(0)
        group_scaled = scaler.transform(clean_group)

        model = models[0] if isinstance(models, list) else models.get(sid, list(models.values())[0])

        X_seq = []
        valid_idx = []
        for idx in range(24, len(group_scaled)):
            X_seq.append(group_scaled[idx - 24 : idx].flatten())
            valid_idx.append(group.index[idx])

        if not X_seq:
            continue

        preds_scaled = model.predict(np.array(X_seq))

        dummy = np.zeros((len(preds_scaled), len(features)))
        dummy[:, 0] = preds_scaled
        preds_unscaled = scaler.inverse_transform(dummy)[:, 0]

        res_df = group.loc[valid_idx].copy()
        print(f"Sensor {sid}:")
        print("  Real power_kw: Mean =", res_df["power_kw"].mean(), "| Std =", res_df["power_kw"].std())
        print("  Pred XGB: Mean =", preds_unscaled.mean(), "| Std =", preds_unscaled.std())
        break


if __name__ == "__main__":
    main()
