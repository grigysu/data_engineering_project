"""Storage abstraction for the bronze layer.

Two backends:
  - `local`: writes JSON to a local directory (default while Docker isn't up).
  - `s3`:    writes JSON to MinIO / S3 via boto3.

Selected via the `STORAGE_BACKEND` env var. The Silver Spark job reads from the
same logical key space regardless of backend.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol


class BronzeStorage(Protocol):
    def write_json(self, key: str, payload: dict) -> str:
        """Write payload as JSON under `key`. Returns the resolved location (path or s3 URI)."""

    def describe(self) -> str:
        """Short, human-readable description of where this storage points."""


class LocalBronzeStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, key: str, payload: dict) -> str:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return str(path)

    def describe(self) -> str:
        return f"local://{self.root.resolve()}"


class S3BronzeStorage:
    def __init__(
        self,
        bucket: str,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        region: str,
    ) -> None:
        import boto3  # imported lazily so local-only users don't need boto3 at runtime

        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    def write_json(self, key: str, payload: dict) -> str:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return f"s3://{self.bucket}/{key}"

    def describe(self) -> str:
        return f"s3://{self.bucket} (endpoint={self.endpoint_url})"


def storage_from_env() -> BronzeStorage:
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    if backend == "local":
        root = Path(os.getenv("LOCAL_LAKE_PATH", "./data/lake"))
        return LocalBronzeStorage(root)
    if backend == "s3":
        return S3BronzeStorage(
            bucket=os.environ["MINIO_BUCKET"],
            endpoint_url=os.environ["MINIO_ENDPOINT"],
            access_key=os.environ["MINIO_ROOT_USER"],
            secret_key=os.environ["MINIO_ROOT_PASSWORD"],
            region=os.getenv("MINIO_REGION", "us-east-1"),
        )
    raise ValueError(f"Unknown STORAGE_BACKEND: {backend!r} (expected 'local' or 's3')")
