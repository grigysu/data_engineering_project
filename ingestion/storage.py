"""S3/MinIO storage for the bronze layer.

The local-FS backend was removed in Phase 2a (the project standardised on
MinIO as the single source of truth). The S3 client speaks the AWS S3 wire
protocol, so the same code works against MinIO locally and any real S3
bucket in the cloud.
"""

from __future__ import annotations

import json
import os
from typing import Protocol


class BronzeStorage(Protocol):
    def write_json(self, key: str, payload: dict) -> str:
        """Write payload as JSON under `key`. Returns the resolved s3:// URI."""

    def describe(self) -> str:
        """Short, human-readable description of where this storage points."""


class S3BronzeStorage:
    def __init__(
        self,
        bucket: str,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        region: str,
    ) -> None:
        import boto3

        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    @property
    def client(self):
        return self._client

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


def storage_from_env() -> S3BronzeStorage:
    """Build the bronze S3 storage from env vars. All are required."""
    return S3BronzeStorage(
        bucket=os.environ["MINIO_BUCKET"],
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ROOT_USER"],
        secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        region=os.getenv("MINIO_REGION", "us-east-1"),
    )
