"""GCS object helpers."""

import asyncio
from typing import cast


async def file_exists(uri: str) -> bool:
    """Return True when a GCS object exists."""

    bucket_name, blob_name = _parse_gcs_uri(uri)

    def _exists() -> bool:
        from google.cloud import storage  # type: ignore[attr-defined]

        client = storage.Client()
        return cast(bool, client.bucket(bucket_name).blob(blob_name).exists())

    return await asyncio.to_thread(_exists)


async def upload_bytes(data: bytes, destination_uri: str, content_type: str) -> str:
    """Upload bytes to GCS and return the destination URI."""

    bucket_name, blob_name = _parse_gcs_uri(destination_uri)

    def _upload() -> None:
        from google.cloud import storage  # type: ignore[attr-defined]

        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.upload_from_string(data, content_type=content_type)

    await asyncio.to_thread(_upload)
    return destination_uri


async def download_bytes(uri: str, *, max_bytes: int | None = None) -> bytes:
    """Download bytes from a GCS URI."""

    bucket_name, blob_name = _parse_gcs_uri(uri)

    def _download() -> bytes:
        from google.cloud import storage  # type: ignore[attr-defined]

        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        if max_bytes is not None:
            blob.reload()
            if blob.size is None or blob.size > max_bytes or blob.generation is None:
                raise ValueError("Recording exceeds bounded audio verification limits")
            return cast(bytes, blob.download_as_bytes(if_generation_match=blob.generation))
        return cast(bytes, blob.download_as_bytes())

    return await asyncio.to_thread(_download)


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse a `gs://bucket/object` URI."""

    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got {uri!r}")

    path = uri.removeprefix("gs://")
    bucket, separator, blob = path.partition("/")
    if not bucket or not separator or not blob:
        raise ValueError(f"Expected a full gs://bucket/object URI, got {uri!r}")

    return bucket, blob
