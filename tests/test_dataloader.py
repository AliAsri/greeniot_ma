from datetime import datetime, timedelta

import pandas as pd

from utils import data_loader as dl


def test_detect_runtime_mode_demo(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    assert dl.detect_runtime_mode() == "demo"


def test_detect_runtime_mode_fallback_without_deltalake(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setattr(dl, "DeltaTable", None)
    assert dl.detect_runtime_mode() == "fallback"


def test_summarize_dataframe_freshness_recent():
    now = datetime.now()
    df = pd.DataFrame({"ts": [now - timedelta(seconds=20)]})
    summary = dl.summarize_dataframe_freshness(df, ts_col="ts")
    assert "Latest point at" in summary


def test_enrich_bronze_adds_expected_columns():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="5min").astype(str),
            "sensor_id": ["rack_A1"] * 6,
            "power_kw": [50.0, 52.0, 51.0, 60.0, 54.0, 53.0],
            "cpu_pct": [30.0, 35.0, 33.0, 45.0, 36.0, 37.0],
            "ram_pct": [60.0, 61.0, 60.5, 62.0, 61.0, 60.0],
            "temp_c": [24.0, 24.5, 24.2, 25.0, 24.8, 24.6],
        }
    )

    enriched = dl._enrich_bronze(df)

    assert {"ts", "power_kw_avg5", "power_delta", "anomaly_flag", "pue"}.issubset(enriched.columns)
    assert len(enriched) == len(df)
