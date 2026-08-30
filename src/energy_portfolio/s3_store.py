"""Minimal S3 read/write helpers for market data and optimization results.

Credentials: never pass keys explicitly. boto3 picks up, in order, env vars
(AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY — local dev only, via a gitignored
.env), or an EC2 instance role (preferred once deployed — see training doc
AWS Mini-project A/B). This module assumes one of those is already configured.
"""

from __future__ import annotations

import io

import boto3
import pandas as pd


def get_client(region: str = "eu-west-3"):
    return boto3.client("s3", region_name=region)


def upload_dataframe(df: pd.DataFrame, bucket: str, key: str, region: str = "eu-west-3") -> None:
    buffer = io.StringIO()
    df.to_csv(buffer)
    get_client(region).put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def download_dataframe(bucket: str, key: str, region: str = "eu-west-3", **read_csv_kwargs) -> pd.DataFrame:
    obj = get_client(region).get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), **read_csv_kwargs)


def list_keys(bucket: str, prefix: str = "", region: str = "eu-west-3") -> list[str]:
    resp = get_client(region).list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [item["Key"] for item in resp.get("Contents", [])]
