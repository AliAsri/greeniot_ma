import pytest

from utils import data_loader as dl


def test_load_bronze_filtered_has_clear_error_without_deltalake(monkeypatch):
    monkeypatch.setattr(dl, "DeltaTable", None)

    with pytest.raises(RuntimeError, match="deltalake is not installed"):
        dl._load_bronze_filtered(2)
