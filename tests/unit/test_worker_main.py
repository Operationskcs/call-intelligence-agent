"""Tests for worker polling loop behavior."""

import logging

import pytest
from deepgram.errors.bad_request_error import BadRequestError

from app.adapters.crm.base import TwentyCRMError
from app.adapters.stt.deepgram import DeepgramCreditsExhaustedError
from app.models.call_event import CallEvent, CallSource
from app.storage.audit import (
    DEEPGRAM_BAD_REQUEST_ERROR_MESSAGE,
    DEEPGRAM_NO_TRANSCRIPT_ERROR_MESSAGE,
)
from app.worker import main, pipeline


def test_configure_logging_adds_info_stream_handler_to_root() -> None:
    """App INFO logs should have a root stream handler for Cloud Run."""

    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    try:
        for handler in original_handlers:
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.WARNING)

        main._configure_logging()

        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) == 1
        handler = root_logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.level == logging.INFO
        assert handler.formatter is not None

        record = logging.LogRecord(
            "app.services.example",
            logging.INFO,
            __file__,
            1,
            "hello",
            (),
            None,
        )
        assert handler.formatter.format(record) == "INFO:app.services.example:hello"
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)


def _event(call_id: str) -> CallEvent:
    """Build a minimal call event for worker-loop tests."""

    return CallEvent(
        call_id=call_id,
        source=CallSource.PHONEBURNER,
        workspace="intake",
        phone_from="+15550000001",
        phone_to="+15550000002",
        duration_sec=60,
        agent_id=None,
        gcs_audio_uri="gs://bucket/audio.mp3",
        raw_payload={},
    )


async def test_process_polled_calls_continues_after_manual_review(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Manual-review calls should not make the worker exit non-zero."""

    events = [_event("review-call"), _event("next-call")]
    processed: list[str] = []

    async def fake_poll_new_calls() -> list[CallEvent]:
        return events

    async def fake_pipeline_run(event: CallEvent) -> None:
        processed.append(event.call_id)
        if event.call_id == "review-call":
            raise pipeline.ManualReviewRequiredError(pipeline.MANUAL_REVIEW_ERROR_MESSAGE)

    monkeypatch.setattr(main, "poll_new_calls", fake_poll_new_calls)
    monkeypatch.setattr(main.pipeline, "run", fake_pipeline_run)

    with caplog.at_level("WARNING"):
        await main.process_polled_calls()

    assert processed == ["review-call", "next-call"]
    assert "call_id=review-call" in caplog.text
    assert pipeline.MANUAL_REVIEW_ERROR_MESSAGE in caplog.text


async def test_process_polled_calls_continues_after_twenty_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Twenty rate limits should not make the worker exit non-zero."""

    events = [_event("rate-limited-call"), _event("next-call")]
    processed: list[str] = []

    async def fake_poll_new_calls() -> list[CallEvent]:
        return events

    async def fake_pipeline_run(event: CallEvent) -> None:
        processed.append(event.call_id)
        if event.call_id == "rate-limited-call":
            raise TwentyCRMError(
                "Limit reached: token rate exceeded",
                code="BAD_USER_INPUT",
            )

    monkeypatch.setattr(main, "poll_new_calls", fake_poll_new_calls)
    monkeypatch.setattr(main.pipeline, "run", fake_pipeline_run)

    with caplog.at_level("WARNING"):
        await main.process_polled_calls()

    assert processed == ["rate-limited-call", "next-call"]
    assert "call_id=rate-limited-call" in caplog.text
    assert "Limit reached" in caplog.text


async def test_process_polled_calls_continues_after_deepgram_no_transcript(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Deepgram no-transcript calls should be audited and skipped per event."""

    events = [_event("silent-call"), _event("next-call")]
    processed: list[str] = []
    audit_rows: list[dict[str, object]] = []

    async def fake_poll_new_calls() -> list[CallEvent]:
        return events

    async def fake_pipeline_run(event: CallEvent) -> None:
        processed.append(event.call_id)
        if event.call_id == "silent-call":
            raise ValueError("Deepgram response did not include a transcript.")

    async def fake_log_result(
        event: CallEvent,
        match: object,
        note: object,
        error: str | None,
    ) -> None:
        audit_rows.append(
            {
                "call_id": event.call_id,
                "requires_review": getattr(match, "requires_review"),
                "error": error,
                "note": note,
            }
        )

    monkeypatch.setattr(main, "poll_new_calls", fake_poll_new_calls)
    monkeypatch.setattr(main.pipeline, "run", fake_pipeline_run)
    monkeypatch.setattr(main, "log_result", fake_log_result)

    with caplog.at_level("WARNING"):
        await main.process_polled_calls()

    assert processed == ["silent-call", "next-call"]
    assert audit_rows == [
        {
            "call_id": "silent-call",
            "requires_review": False,
            "error": DEEPGRAM_NO_TRANSCRIPT_ERROR_MESSAGE,
            "note": None,
        }
    ]
    assert "call_id=silent-call" in caplog.text


async def test_process_polled_calls_continues_after_deepgram_bad_request(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Deepgram bad requests should be audited and skipped per event."""

    events = [_event("corrupt-call"), _event("next-call")]
    processed: list[str] = []
    audit_rows: list[dict[str, object]] = []

    async def fake_poll_new_calls() -> list[CallEvent]:
        return events

    async def fake_pipeline_run(event: CallEvent) -> None:
        processed.append(event.call_id)
        if event.call_id == "corrupt-call":
            raise BadRequestError({"err_msg": "Bad Request"})

    async def fake_log_result(
        event: CallEvent,
        match: object,
        note: object,
        error: str | None,
    ) -> None:
        audit_rows.append(
            {
                "call_id": event.call_id,
                "requires_review": getattr(match, "requires_review"),
                "error": error,
                "note": note,
            }
        )

    monkeypatch.setattr(main, "poll_new_calls", fake_poll_new_calls)
    monkeypatch.setattr(main.pipeline, "run", fake_pipeline_run)
    monkeypatch.setattr(main, "log_result", fake_log_result)

    with caplog.at_level("WARNING"):
        await main.process_polled_calls()

    assert processed == ["corrupt-call", "next-call"]
    assert audit_rows == [
        {
            "call_id": "corrupt-call",
            "requires_review": False,
            "error": DEEPGRAM_BAD_REQUEST_ERROR_MESSAGE,
            "note": None,
        }
    ]
    assert "call_id=corrupt-call" in caplog.text


async def test_process_polled_calls_stops_on_deepgram_credits_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deepgram credit exhaustion should bubble to the worker entrypoint."""

    events = [_event("credits-call"), _event("next-call")]
    processed: list[str] = []

    async def fake_poll_new_calls() -> list[CallEvent]:
        return events

    async def fake_pipeline_run(event: CallEvent) -> None:
        processed.append(event.call_id)
        raise DeepgramCreditsExhaustedError("Deepgram credits exhausted")

    monkeypatch.setattr(main, "poll_new_calls", fake_poll_new_calls)
    monkeypatch.setattr(main.pipeline, "run", fake_pipeline_run)

    with pytest.raises(DeepgramCreditsExhaustedError):
        await main.process_polled_calls()

    assert processed == ["credits-call"]


def test_main_pauses_scheduler_after_deepgram_credits_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The top-level worker entrypoint pauses scheduling and exits successfully."""

    paused = False

    async def fake_process_polled_calls() -> None:
        raise DeepgramCreditsExhaustedError("Deepgram credits exhausted")

    def fake_pause_scheduler() -> None:
        nonlocal paused
        paused = True

    monkeypatch.setattr(main, "process_polled_calls", fake_process_polled_calls)
    monkeypatch.setattr(main, "_pause_call_intelligence_worker_scheduler", fake_pause_scheduler)

    with caplog.at_level("CRITICAL"):
        exit_code = main.main()

    assert exit_code == 0
    assert paused
    assert "Deepgram credits exhausted — pausing Cloud Scheduler and stopping worker" in caplog.text


def test_pause_scheduler_calls_cloud_scheduler_pause_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduler pausing should target the configured project, region, and job."""

    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            captured["raised_for_status"] = True

    class FakeSession:
        def __init__(self, credentials: object) -> None:
            captured["credentials"] = credentials

        def post(self, url: str, timeout: int) -> FakeResponse:
            captured["url"] = url
            captured["timeout"] = timeout
            return FakeResponse()

    def fake_google_auth_default(scopes: list[str]) -> tuple[str, str]:
        captured["scopes"] = scopes
        return "credentials", "project"

    monkeypatch.setattr(main, "google_auth_default", fake_google_auth_default)
    monkeypatch.setattr(main, "AuthorizedSession", FakeSession)

    main._pause_call_intelligence_worker_scheduler()

    assert captured == {
        "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
        "credentials": "credentials",
        "url": (
            "https://cloudscheduler.googleapis.com/v1/projects/keep-calm-database"
            "/locations/us-central1/jobs/call-intelligence-worker-scheduler:pause"
        ),
        "timeout": 30,
        "raised_for_status": True,
    }
