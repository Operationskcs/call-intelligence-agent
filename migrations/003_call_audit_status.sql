ALTER TABLE call_audit_log
    ADD COLUMN IF NOT EXISTS status VARCHAR(32),
    ADD COLUMN IF NOT EXISTS reserved_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

UPDATE call_audit_log
SET status = CASE
    WHEN processed_at IS NULL THEN 'legacy_unprocessed'
    WHEN error_message IS NULL THEN 'succeeded'
    ELSE 'failed'
END
WHERE status IS NULL;

UPDATE call_audit_log
SET
    reserved_at = COALESCE(reserved_at, created_at),
    updated_at = COALESCE(updated_at, processed_at, created_at)
WHERE reserved_at IS NULL
   OR updated_at IS NULL;

ALTER TABLE call_audit_log
    ALTER COLUMN status SET DEFAULT 'processing',
    ALTER COLUMN reserved_at SET DEFAULT NOW(),
    ALTER COLUMN updated_at SET DEFAULT NOW(),
    ALTER COLUMN status SET NOT NULL,
    ALTER COLUMN reserved_at SET NOT NULL,
    ALTER COLUMN updated_at SET NOT NULL;

ALTER TABLE call_audit_log
    DROP CONSTRAINT IF EXISTS call_audit_log_status_check;

ALTER TABLE call_audit_log
    ADD CONSTRAINT call_audit_log_status_check
    CHECK (status IN ('legacy_unprocessed', 'processing', 'succeeded', 'failed'));

CREATE INDEX IF NOT EXISTS idx_call_audit_processing_updated_at
    ON call_audit_log(updated_at)
    WHERE status = 'processing';
