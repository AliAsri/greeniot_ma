import os

import pytest

RUN_MINIO_TESTS = os.getenv("RUN_MINIO_TESTS", "false").lower() == "true"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_MINIO_TESTS,
        reason="Set RUN_MINIO_TESTS=true to enable live MinIO integration tests.",
    ),
]

DeltaTable = pytest.importorskip("deltalake").DeltaTable


def _storage_options():
    return {
        "AWS_ACCESS_KEY_ID": os.getenv("MINIO_ACCESS_KEY", "greeniot"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("MINIO_SECRET_KEY", "greeniot2030"),
        "AWS_ENDPOINT_URL": os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        "AWS_REGION": os.getenv("AWS_REGION", "us-east-1"),
        "AWS_ALLOW_HTTP": os.getenv("AWS_ALLOW_HTTP", "true"),
        "AWS_S3_ALLOW_UNSAFE_RENAME": os.getenv("AWS_S3_ALLOW_UNSAFE_RENAME", "true"),
        "AWS_S3_FORCE_PATH_STYLE": os.getenv("AWS_S3_FORCE_PATH_STYLE", "true"),
    }


def test_minio_delta_schema_is_accessible():
    dt = DeltaTable("s3://greeniot/bronze/servers", storage_options=_storage_options())
    assert len(dt.schema().fields) > 0
