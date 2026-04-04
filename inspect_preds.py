import os
import sys
import pandas as pd
import numpy as np
import joblib
import torch

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)
from utils.data_loader import load_gold_servers

df = load_gold_servers()
models_dir = os.path.join(root_dir, "models")
xgb_path = os.path.join(models_dir, "xgboost_predictor.pkl")
scaler_path = os.path.join(models_dir, "scaler_prediction.pkl")

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
    
    m = models[0] if isinstance(models, list) else models.get(sid, list(models.values())[0])
    
    X_seq = []
    valid_idx = []
    for i in range(24, len(group_scaled)):
        X_seq.append(group_scaled[i-24:i].flatten())
        valid_idx.append(group.index[i])
        
    preds_scaled = m.predict(np.array(X_seq))
    
    dummy = np.zeros((len(preds_scaled), len(features)))
    dummy[:, 0] = preds_scaled
    preds_unscaled = scaler.inverse_transform(dummy)[:, 0]
    
    res_df = group.loc[valid_idx].copy()
    print(f"Sensor {sid}:")
    print("  Real power_kw: Mean =", res_df["power_kw"].mean(), "| Std =", res_df["power_kw"].std())
    print("  Pred XGB: Mean =", preds_unscaled.mean(), "| Std =", preds_unscaled.std())
    break
