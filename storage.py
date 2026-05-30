"""
MLB Strikeout Pipeline — Storage Abstraction

A small swappable interface for reading/writing the daily artifacts (Statcast
snapshots, feature parquets) so the pipeline does not depend on the ephemeral
local filesystem.

Backends
--------
  local  (default)  — writes under data/storage/.  Zero config; works with no
                      credentials.  Good for dev and as a safe fallback.
  s3                — any S3-compatible object store (Cloudflare R2, AWS S3,
                      Backblaze B2).  Selected by setting STORAGE_BACKEND=s3.

Environment variables (s3 backend)
----------------------------------
  STORAGE_BACKEND   = s3
  S3_BUCKET         = your-bucket-name
  S3_ENDPOINT_URL   = https://<accountid>.r2.cloudflarestorage.com   (R2)
                      (omit for AWS S3 to use the default endpoint)
  S3_ACCESS_KEY     = ...
  S3_SECRET_KEY     = ...
  S3_REGION         = auto            (R2 uses "auto"; S3 uses e.g. us-east-1)

Interface
---------
  storage = get_storage()
  storage.put_parquet(df, "snapshots/snapshot_20260530.parquet")
  df  = storage.get_parquet("snapshots/snapshot_20260530.parquet")
  ok  = storage.exists("snapshots/snapshot_20260530.parquet")
  key = storage.latest("snapshots/snapshot_")   # most recent matching key
"""

import io
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)


class Storage(ABC):
    """Abstract artifact store. Keys are '/'-delimited paths (no leading slash)."""

    @abstractmethod
    def put_parquet(self, df: pd.DataFrame, key: str) -> None: ...

    @abstractmethod
    def get_parquet(self, key: str) -> Optional[pd.DataFrame]: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list_keys(self, prefix: str) -> List[str]: ...

    def latest(self, prefix: str) -> Optional[str]:
        """Return the lexicographically-greatest key under `prefix`, or None.

        Snapshot keys embed a YYYYMMDD date, so lexicographic max == newest.
        """
        keys = self.list_keys(prefix)
        return max(keys) if keys else None


# ── Local filesystem backend ─────────────────────────────────────────────────

class LocalStorage(Storage):
    """Stores artifacts under data/storage/. Default backend (no credentials)."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else (DATA_DIR / "storage")
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Storage backend: local ({self.root})")

    def _path(self, key: str) -> Path:
        return self.root / key

    def put_parquet(self, df: pd.DataFrame, key: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Stored {len(df):,} rows -> local:{key}")

    def get_parquet(self, key: str) -> Optional[pd.DataFrame]:
        path = self._path(key)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list_keys(self, prefix: str) -> List[str]:
        # Split prefix into directory part + filename-stem part
        pfx_path = self._path(prefix)
        search_dir = pfx_path.parent
        stem = pfx_path.name
        if not search_dir.exists():
            return []
        out = []
        for p in search_dir.iterdir():
            if p.is_file() and p.name.startswith(stem):
                out.append(str(p.relative_to(self.root)).replace("\\", "/"))
        return out


# ── S3-compatible backend (R2 / S3 / B2) ─────────────────────────────────────

class S3Storage(Storage):
    """S3-compatible object store. Lazily imports boto3 so local needs no deps."""

    def __init__(self):
        try:
            import boto3  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for the s3 storage backend: pip install boto3"
            ) from exc
        import boto3

        self.bucket = os.environ["S3_BUCKET"]
        endpoint = os.environ.get("S3_ENDPOINT_URL") or None
        region = os.environ.get("S3_REGION", "auto")

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ["S3_ACCESS_KEY"],
            aws_secret_access_key=os.environ["S3_SECRET_KEY"],
            region_name=region,
        )
        logger.info(f"Storage backend: s3 (bucket={self.bucket}, endpoint={endpoint})")

    def put_parquet(self, df: pd.DataFrame, key: str) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())
        logger.info(f"Stored {len(df):,} rows -> s3:{key}")

    def get_parquet(self, key: str) -> Optional[pd.DataFrame]:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.NoSuchKey:
            return None
        except Exception as exc:
            logger.warning(f"s3 get_parquet failed for {key}: {exc}")
            return None
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def list_keys(self, prefix: str) -> List[str]:
        keys: List[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys


# ── Factory ──────────────────────────────────────────────────────────────────

_INSTANCE: Optional[Storage] = None


def get_storage() -> Storage:
    """Return the configured storage backend (cached singleton).

    STORAGE_BACKEND=s3 selects the object store; anything else (default) is local.
    Falls back to local if the s3 backend cannot initialize, so the pipeline
    never hard-fails on a storage misconfiguration.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE

    backend = os.environ.get("STORAGE_BACKEND", "local").strip().lower()
    if backend == "s3":
        try:
            _INSTANCE = S3Storage()
            return _INSTANCE
        except Exception as exc:
            logger.warning(f"s3 storage init failed ({exc}); falling back to local")

    _INSTANCE = LocalStorage()
    return _INSTANCE


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    s = get_storage()
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    s.put_parquet(df, "snapshots/snapshot_20260529.parquet")
    s.put_parquet(df, "snapshots/snapshot_20260530.parquet")

    print("exists:", s.exists("snapshots/snapshot_20260530.parquet"))
    print("latest:", s.latest("snapshots/snapshot_"))
    got = s.get_parquet("snapshots/snapshot_20260530.parquet")
    print("round-trip rows:", 0 if got is None else len(got))
    print("missing returns None:", s.get_parquet("snapshots/nope.parquet") is None)
