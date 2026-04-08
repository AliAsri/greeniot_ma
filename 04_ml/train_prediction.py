"""
GreenIoT-MA - Prediction model training
=======================================
Trains and compares two models via MLflow:
- PyTorch LSTM for temporal dependencies
- XGBoost as a strong tabular baseline

Source: Gold Delta Lake on MinIO (s3a://greeniot/gold/servers)
Goal: predict consumption at +5 min, +10 min and +15 min
"""

import os
import warnings

import joblib
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from deltalake import DeltaTable
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

# Fix potential OpenMP deadlock on Windows with PyTorch/XGBoost
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()
warnings.filterwarnings("ignore")

MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")
EXPERIMENT = os.getenv(
    "MLFLOW_EXPERIMENT_PREDICTION",
    "greeniot_consumption_prediction",
)
GOLD_PATH = os.getenv("DELTA_GOLD", "s3a://greeniot/gold").replace("s3a://", "s3://") + "/servers"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

STORAGE_OPTIONS = {
    "AWS_ACCESS_KEY_ID": os.getenv("MINIO_ACCESS_KEY", "greeniot"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("MINIO_SECRET_KEY", "greeniot2030"),
    "AWS_ENDPOINT_URL": os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

WINDOW_SIZE = 24
HORIZON = 3
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 2

FEATURES = [
    "power_kw",
    "cpu_pct",
    "ram_pct",
    "temp_c",
    "power_kw_avg5",
    "power_kw_std5",
    "cpu_avg5",
    "power_delta",
    "power_kw_lag1",
    "power_kw_lag3",
    "power_kw_lag6",
    "power_kw_avg12",
    "power_kw_avg24",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_business_hours",
    "is_weekend",
]


def load_gold() -> pd.DataFrame:
    """Read the Gold Delta table directly from MinIO."""
    print(f"   Reading Gold Delta from: {GOLD_PATH}")
    dt = DeltaTable(GOLD_PATH, storage_options=STORAGE_OPTIONS)
    available = [f.name for f in dt.schema().fields]
    cols = [f for f in FEATURES if f in available] + ["sensor_id", "ts"]
    df = dt.to_pyarrow_table(columns=list(dict.fromkeys(cols))).to_pandas()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["sensor_id", "ts"]).reset_index(drop=True)
    df = df.dropna(subset=[f for f in FEATURES if f in df.columns])
    return df


class LSTMPredictor(nn.Module):
    """LSTM network for sequence prediction."""

    def __init__(self, input_size: int, hidden=64, layers=2, horizon=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden,
            layers,
            batch_first=True,
            dropout=0.2,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, horizon),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def create_sequences(data: np.ndarray, window: int, horizon: int):
    """Create sliding windows for sequence training."""
    X, y = [], []
    for i in range(len(data) - window - horizon):
        X.append(data[i : i + window])
        y.append(data[i + window : i + window + horizon, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_lstm(X_tr, y_tr, X_val, y_val, features, scaler):
    with mlflow.start_run(run_name="LSTM_greeniot"):
        mlflow.log_params(
            {
                "model_type": "LSTM",
                "window_size": WINDOW_SIZE,
                "horizon": HORIZON,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "n_features": len(features),
                "features": str(features),
                "data_source": GOLD_PATH,
            }
        )

        tr_ds = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        model = LSTMPredictor(input_size=len(features))
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=5,
            factor=0.5,
        )

        try:
            from tqdm import tqdm

            epoch_iter = tqdm(range(EPOCHS), desc="  LSTM Epochs", unit="epoch", position=0)
        except ImportError:
            tqdm = None
            epoch_iter = range(EPOCHS)

        def unscale_horizon(arr):
            res = []
            for horizon_idx in range(arr.shape[1]):
                dummy = np.zeros((len(arr), len(features)))
                dummy[:, 0] = arr[:, horizon_idx]
                res.append(scaler.inverse_transform(dummy)[:, 0])
            return np.column_stack(res)

        y_val_unscaled = unscale_horizon(y_val)

        best_mae = float("inf")
        n_batches = len(tr_ds)
        mae = rmse = r2 = mape_val = 0.0

        for epoch in epoch_iter:
            model.train()
            epoch_loss = 0.0

            if tqdm is not None and hasattr(epoch_iter, "set_postfix"):
                batch_iter = tqdm(
                    tr_ds,
                    desc=f"    Epoch {epoch + 1}/{EPOCHS}",
                    leave=False,
                    position=1,
                )
            else:
                batch_iter = tr_ds

            for xb, yb in batch_iter:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

                if hasattr(batch_iter, "set_postfix"):
                    batch_iter.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = epoch_loss / n_batches

            model.eval()
            with torch.no_grad():
                val_pred = model(torch.tensor(X_val)).numpy()

            val_pred_unscaled = unscale_horizon(val_pred)

            mae = mean_absolute_error(y_val_unscaled, val_pred_unscaled)
            rmse = np.sqrt(mean_squared_error(y_val_unscaled, val_pred_unscaled))
            r2 = r2_score(y_val.flatten(), val_pred.flatten())
            mape_val = mape(y_val_unscaled.flatten(), val_pred_unscaled.flatten())

            mlflow.log_metrics(
                {
                    "val_mae": mae,
                    "val_rmse": rmse,
                    "val_r2": r2,
                    "val_mape": mape_val,
                    "train_loss": avg_loss,
                },
                step=epoch,
            )

            scheduler.step(mae)
            if mae < best_mae:
                best_mae = mae

            if hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=f"{avg_loss:.4f}", MAE=f"{mae:.4f}", R2=f"{r2:.4f}")
            else:
                print(
                    f"  Epoch {epoch + 1:3d}/{EPOCHS} - loss: {avg_loss:.4f} | "
                    f"MAE: {mae:.4f} | R2: {r2:.4f} | MAPE: {mape_val:.1f}%"
                )

        mlflow.pytorch.log_model(model, "lstm_model")
        mlflow.log_metric("best_mae", best_mae)

        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_predictor.pt"))

        joblib.dump(
            {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mape": mape_val,
            },
            os.path.join(MODEL_DIR, "lstm_metrics.pkl"),
        )

        print(f"\n  LSTM - Best MAE: {best_mae:.4f}")
        return model, best_mae


def train_xgboost(X_tr, y_tr, X_val, y_val, features, scaler):
    """Train XGBoost baseline and log it in MLflow."""
    X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    with mlflow.start_run(run_name="XGBoost_greeniot"):
        params = {
            "model_type": "XGBoost",
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "window_size": WINDOW_SIZE,
            "horizon": HORIZON,
            "n_features": len(features),
            "data_source": GOLD_PATH,
        }
        mlflow.log_params(params)

        models = []

        for horizon_idx in range(HORIZON):
            xgb = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            xgb.fit(
                X_tr_flat,
                y_tr[:, horizon_idx],
                eval_set=[(X_val_flat, y_val[:, horizon_idx])],
                verbose=False,
            )
            models.append(xgb)

            pred = xgb.predict(X_val_flat)
            mae_h = mean_absolute_error(y_val[:, horizon_idx], pred)
            mlflow.log_metric(f"val_mae_h{horizon_idx + 1}", mae_h)

        val_pred = np.column_stack([model.predict(X_val_flat) for model in models])

        def unscale_horizon(arr):
            res = []
            for horizon_idx in range(arr.shape[1]):
                dummy = np.zeros((len(arr), len(features)))
                dummy[:, 0] = arr[:, horizon_idx]
                res.append(scaler.inverse_transform(dummy)[:, 0])
            return np.column_stack(res)

        y_val_unscaled = unscale_horizon(y_val)
        val_pred_unscaled = unscale_horizon(val_pred)

        overall_mae = mean_absolute_error(y_val_unscaled, val_pred_unscaled)
        overall_rmse = np.sqrt(mean_squared_error(y_val_unscaled, val_pred_unscaled))
        overall_r2 = r2_score(y_val.flatten(), val_pred.flatten())
        overall_mape = mape(y_val_unscaled.flatten(), val_pred_unscaled.flatten())

        mlflow.log_metrics(
            {
                "val_mae": overall_mae,
                "val_rmse": overall_rmse,
                "val_r2": overall_r2,
                "val_mape": overall_mape,
            }
        )

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "xgboost_predictor.pkl")
        joblib.dump(
            {
                "models": models,
                "features": features,
                "metrics": {
                    "mae": overall_mae,
                    "rmse": overall_rmse,
                    "r2": overall_r2,
                    "mape": overall_mape,
                },
            },
            model_path,
        )
        mlflow.sklearn.log_model(models[0], "xgboost_model")

        print(
            f"\n  XGBoost - MAE: {overall_mae:.4f} | "
            f"RMSE: {overall_rmse:.4f} | R2: {overall_r2:.4f} | MAPE: {overall_mape:.1f}%"
        )
        return models, overall_mae


def train():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    print("GreenIoT-MA - Entrainement des modeles de prediction")
    print(f"   MLflow   : {MLFLOW_URI}")
    print(f"   Experience: {EXPERIMENT}\n")

    df = load_gold()

    features = [f for f in FEATURES if f in df.columns]
    missing = set(FEATURES) - set(features)
    if missing:
        print(f"   Features absentes du Gold (ignorees) : {missing}")

    print(f"   Dataset  : {len(df):,} lignes | {len(features)} features")
    print(f"   Features : {features}\n")

    train_frames = []
    sensor_splits = []
    for sid in df["sensor_id"].unique():
        sub_df = (
            df[df["sensor_id"] == sid]
            .sort_values("ts")[features]
            .ffill()
            .bfill()
            .reset_index(drop=True)
        )

        min_points = WINDOW_SIZE + HORIZON + 2
        if len(sub_df) < min_points:
            continue

        split_idx = max(WINDOW_SIZE + 1, int(0.8 * len(sub_df)))
        if split_idx >= len(sub_df) - HORIZON:
            split_idx = len(sub_df) - HORIZON - 1
        if split_idx <= WINDOW_SIZE:
            continue

        train_frames.append(sub_df.iloc[:split_idx])
        sensor_splits.append((sub_df, split_idx))

    if not sensor_splits:
        print("   Pas assez de donnees par capteur pour creer un split temporel. Arret.")
        return

    scaler = MinMaxScaler()
    scaler.fit(pd.concat(train_frames, ignore_index=True))

    train_X, train_y, val_X, val_y = [], [], [], []
    for sub_df, split_idx in sensor_splits:
        train_raw = sub_df.iloc[:split_idx].values
        val_raw = sub_df.iloc[max(0, split_idx - WINDOW_SIZE) :].values

        train_scaled = scaler.transform(train_raw)
        val_scaled = scaler.transform(val_raw)

        X_train_sid, y_train_sid = create_sequences(train_scaled, WINDOW_SIZE, HORIZON)
        X_val_sid, y_val_sid = create_sequences(val_scaled, WINDOW_SIZE, HORIZON)

        if len(X_train_sid) > 0:
            train_X.append(X_train_sid)
            train_y.append(y_train_sid)
        if len(X_val_sid) > 0:
            val_X.append(X_val_sid)
            val_y.append(y_val_sid)

    if not train_X or not val_X:
        print("   Pas assez de donnees pour creer des sequences. Arret.")
        return

    X_tr = np.concatenate(train_X)
    y_tr = np.concatenate(train_y)
    X_val = np.concatenate(val_X)
    y_val = np.concatenate(val_y)

    print(f"   Sequences: {len(X_tr) + len(X_val):,} total | {len(X_tr):,} train | {len(X_val):,} val")
    print(
        f"   Window: {WINDOW_SIZE} pts ({WINDOW_SIZE * 5} min) | "
        f"Horizon: {HORIZON} pts (+{HORIZON * 5} min)\n"
    )

    print("-" * 55)
    print("Entrainement LSTM...")
    print("-" * 55)
    _, lstm_mae = train_lstm(X_tr, y_tr, X_val, y_val, features, scaler)

    print(f"\n{'-' * 55}")
    print("Entrainement XGBoost...")
    print("-" * 55)
    _, xgb_mae = train_xgboost(X_tr, y_tr, X_val, y_val, features, scaler)

    print(f"\n{'=' * 55}")
    print("Comparaison des modeles :")
    print(f"   LSTM    - MAE: {lstm_mae:.4f}")
    print(f"   XGBoost - MAE: {xgb_mae:.4f}")
    winner = "LSTM" if lstm_mae < xgb_mae else "XGBoost"
    print(f"   Meilleur : {winner}")
    print(f"{'=' * 55}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "features": features},
        os.path.join(MODEL_DIR, "scaler_prediction.pkl"),
    )
    print(f"\n   Modeles sauvegardes dans : {MODEL_DIR}")


if __name__ == "__main__":
    train()
