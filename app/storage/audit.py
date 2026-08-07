"""PostgreSQL audit log helpers."""

from collections.abc import Sequence
from typing import Any

import asyncpg

from app.config import get_settings

AUDIT_STATUS_LEGACY_UNPROCESSED = "legacy_unprocessed"
AUDIT_STATUS_PROCESSING = "processing"
AUDIT_STATUS_SUCCEEDED = "succeeded"
AUDIT_STATUS_FAILED = "failed"
PROCESSING_RECLAIM_TTL_SECONDS = 30 * 60

_ALLOWED_COLUMNS = {
    "call_id",
    "source",
    "status",
    "workspace",
    "crm_record_id",
    "phone_from",
    "phone_to",
    "duration_sec",
    "gcs_audio_uri",
    "gcs_transcript_uri",
    "match_confidence",
    "match_method",
    "matched_on_phone",
    "note_created",
    "review_required",
    "error_message",
    "processed_at",
    "reserved_at",
    "updated_at",
}

DEEPGRAM_NO_TRANSCRIPT_ERROR_MESSAGE = "Deepgram returned no transcript"
DEEPGRAM_BAD_REQUEST_ERROR_MESSAGE = (
    "Deepgram bad request: corrupt or unsupported audio"
)

IS_PROCESSED_QUERY = """
SELECT EXISTS (
    SELECT 1
    FROM call_audit_log
    WHERE call_id = $1
      AND processed_at IS NOT NULL
      AND status IN ('succeeded', 'failed')
)
"""

PROCESSED_CALL_IDS_QUERY = """
SELECT call_id
FROM call_audit_log
WHERE call_id = ANY($1)
  AND processed_at IS NOT NULL
  AND status IN ('succeeded', 'failed')
"""

RECLAIMABLE_PROCESSING_CALL_IDS_QUERY = """
SELECT call_id
FROM call_audit_log
WHERE status = 'processing'
  AND updated_at < NOW() - ($1::double precision * INTERVAL '1 second')
ORDER BY updated_at ASC
LIMIT $2
"""


async def upsert_call_log(row: dict[str, Any]) -> None:
    """Upsert a row in call_audit_log by call_id using asyncpg directly."""

    if "call_id" not in row:
        raise ValueError("call_id is required to upsert call_audit_log.")

    unknown_columns = set(row) - _ALLOWED_COLUMNS
    if unknown_columns:
        raise ValueError(f"Unsupported call_audit_log columns: {sorted(unknown_columns)}")

    columns = list(row)
    values = [row[column] for column in columns]
    placeholders = ", ".join(f"${index}" for index in range(1, len(columns) + 1))
    column_sql = ", ".join(columns)
    updates = ", ".join(
        f"{column} = EXCLUDED.{column}" for column in columns if column != "call_id"
    )
    conflict_action = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
    query = (
        f"INSERT INTO call_audit_log ({column_sql}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (call_id) {conflict_action}"
    )

    connection = await _connect()
    try:
        await connection.execute(query, *values)
    finally:
        await connection.close()


async def try_reserve_call_id(
    call_id: str,
    source: str,
    *,
    reclaim_after_seconds: int = PROCESSING_RECLAIM_TTL_SECONDS,
) -> bool:
    """Atomically claim call_id before processing.

    Returns True for a new claim or an expired processing reclaim.
    Terminal and legacy-unprocessed rows are never automatically reopened.
    """

    query = (
        "INSERT INTO call_audit_log (call_id, source, status, reserved_at, updated_at) "
        "VALUES ($1, $2, 'processing', NOW(), NOW()) "
        "ON CONFLICT (call_id) DO UPDATE SET "
        "source = EXCLUDED.source, "
        "status = EXCLUDED.status, "
        "reserved_at = EXCLUDED.reserved_at, "
        "updated_at = EXCLUDED.updated_at, "
        "processed_at = NULL, "
        "error_message = NULL "
        "WHERE call_audit_log.status = 'processing' "
        "AND call_audit_log.updated_at < "
        "NOW() - ($3::double precision * INTERVAL '1 second') "
        "RETURNING call_id"
    )
    connection = await _connect()
    try:
        row = await connection.fetchrow(query, call_id, source, reclaim_after_seconds)
        return row is not None
    finally:
        await connection.close()


async def is_processed(call_id: str) -> bool:
    """Return True if call_audit_log already has a successful row for call_id."""

    connection = await _connect()
    try:
        value = await connection.fetchval(IS_PROCESSED_QUERY, call_id)
    finally:
        await connection.close()

    return bool(value)


async def processed_call_ids(call_ids: Sequence[str]) -> set[str]:
    """Return candidate call IDs that already have successful audit-log rows."""

    if not call_ids:
        return set()

    connection = await _connect()
    try:
        rows = await connection.fetch(PROCESSED_CALL_IDS_QUERY, list(call_ids))
    finally:
        await connection.close()

    return {str(row["call_id"]) for row in rows}


async def _connect() -> asyncpg.Connection:
    """Create an asyncpg connection from DATABASE_URL."""

    database_url = get_settings().database.url
    if not database_url:
        raise RuntimeError("DATABASE_URL must be configured before audit logging.")
    return await asyncpg.connect(database_url)
