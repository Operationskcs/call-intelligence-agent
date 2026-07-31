"""Worker Cloud Run entry point for webhook-triggered call processing."""

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from deepgram.errors.bad_request_error import BadRequestError
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response, status
from google.auth import default as google_auth_default
from google.auth.transport.requests import AuthorizedSession

from app.adapters.crm.base import TwentyCRMError, is_twenty_rate_limit_error
from app.adapters.stt.deepgram import DeepgramCreditsExhaustedError
from app.config import get_settings
from app.models.call_event import CallEvent
from app.models.match_result import MatchMethod, MatchResult
from app.storage.audit import (
    DEEPGRAM_BAD_REQUEST_ERROR_MESSAGE,
    DEEPGRAM_NO_TRANSCRIPT_ERROR_MESSAGE,
)
from app.worker import pipeline
from app.worker.steps.s1_ingest import poll_legacy_phoneburner_calls, poll_new_calls
from app.worker.steps.s8_audit import log_result

logger = logging.getLogger(__name__)

__all__ = [
    "_pause_call_intelligence_worker_scheduler",
    "app",
    "create_app",
    "main",
    "pipeline",
    "process_legacy_intake_calls",
    "process_polled_calls",
]

_SCHEDULER_PROJECT_ID = "keep-calm-database"
_SCHEDULER_REGION = "us-central1"
_SCHEDULER_JOB_ID = "call-intelligence-worker-scheduler"
_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_CLOUD_SCHEDULER_API_BASE_URL = "https://cloudscheduler.googleapis.com/v1"


async def process_polled_calls() -> None:
    """Poll BigQuery once and run the pipeline for each unprocessed call."""

    events = await poll_new_calls()
    logger.info("BigQuery polling found %d unprocessed call(s).", len(events))
    await _process_events(events)


async def process_legacy_intake_calls() -> int:
    """Poll legacy PhoneBurner BigQuery rows once for manual Intake reprocessing."""

    events = await poll_legacy_phoneburner_calls()
    logger.info("Legacy PhoneBurner polling found %d unprocessed call(s).", len(events))
    await _process_events(events)
    return len(events)


async def _process_events(events: Sequence[CallEvent]) -> None:
    """Run the pipeline for a sequence of already-normalized call events."""

    for event in events:
        try:
            await pipeline.run(event)
        except pipeline.ManualReviewRequiredError as exc:
            logger.warning(
                "Call requires manual review; continuing to next event. call_id=%s reason=%s",
                event.call_id,
                exc,
            )
            continue
        except TwentyCRMError as exc:
            if not is_twenty_rate_limit_error(exc):
                raise

            logger.warning(
                "Twenty CRM rate limit reached; continuing to next event. call_id=%s reason=%s",
                event.call_id,
                exc,
            )
            continue
        except BadRequestError as exc:
            logger.warning(
                "Deepgram bad request; continuing to next event. call_id=%s reason=%s",
                event.call_id,
                exc,
            )
            await _log_terminal_deepgram_error(
                event,
                DEEPGRAM_BAD_REQUEST_ERROR_MESSAGE,
            )
            continue
        except ValueError as exc:
            if not _is_deepgram_no_transcript_error(exc):
                raise

            logger.warning(
                "Deepgram returned no transcript; continuing to next event. call_id=%s reason=%s",
                event.call_id,
                exc,
            )
            await _log_terminal_deepgram_error(
                event,
                DEEPGRAM_NO_TRANSCRIPT_ERROR_MESSAGE,
            )
            continue


def create_app() -> FastAPI:
    """Create the Cloud Run Service application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Log startup config and register the RingCentral webhook subscription."""

        _ = app
        logging.basicConfig(level=logging.INFO)
        settings = get_settings()
        logger.info("Worker service configuration loaded: %s", settings.safe_summary())

        if settings.telephony.ringcentral.webhook_url:
            from app.services.ringcentral_subscription import subscription_manager

            logger.info("Starting RingCentral subscription manager from FastAPI lifespan.")
            try:
                await subscription_manager.startup()
            except Exception:
                logger.exception(
                    "RingCentral subscription manager startup failed; "
                    "continuing Cloud Run service startup."
                )
            else:
                logger.info("RingCentral subscription manager started from FastAPI lifespan.")
        else:
            logger.warning(
                "RINGCENTRAL_WEBHOOK_URL is not configured; skipping subscription startup."
            )

        try:
            yield
        finally:
            from app.services.ringcentral_subscription import subscription_manager

            await subscription_manager.shutdown()

    app = FastAPI(
        title="Call Intelligence Webhook Worker",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Return service health status."""

        return {"status": "ok"}

    @app.post("/webhook/phoneburner", status_code=status.HTTP_200_OK)
    async def phoneburner_webhook(
        request: Request,
        x_webhook_token: str | None = Header(default=None, alias="X-Webhook-Token"),
    ) -> dict[str, Any]:
        """Receive PhoneBurner call completion events for Intake."""

        _validate_webhook_token(x_webhook_token)
        payload = await _json_object(request)

        from app.webhook.phoneburner import handle_phoneburner_webhook

        return await handle_phoneburner_webhook(payload)

    @app.post(
        "/webhook/ringcentral",
        status_code=status.HTTP_200_OK,
        response_model=None,
    )
    async def ringcentral_webhook(
        request: Request,
        background_tasks: BackgroundTasks,
        validation_token: str | None = Header(default=None, alias="Validation-Token"),
    ) -> Response | dict[str, Any]:
        """Receive RingCentral notifications for MedHub."""

        if validation_token:
            return Response(
                status_code=status.HTTP_200_OK,
                headers={"Validation-Token": validation_token},
            )

        payload = await _json_object(request)

        from app.webhook.ringcentral import handle_ringcentral_webhook

        return await handle_ringcentral_webhook(payload, background_tasks)

    @app.post("/reprocess/intake", status_code=status.HTTP_200_OK)
    async def reprocess_intake(
        x_webhook_token: str | None = Header(default=None, alias="X-Webhook-Token"),
    ) -> dict[str, object]:
        """Run one legacy PhoneBurner polling cycle for Intake reprocessing."""

        _validate_webhook_token(x_webhook_token)
        processed_count = await process_legacy_intake_calls()
        return {"status": "accepted", "processed_count": processed_count}

    return app


def _validate_webhook_token(token: str | None) -> None:
    """Validate the shared webhook token header."""

    expected_token = get_settings().webhook.secret_token
    if not expected_token or token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook token",
        )


async def _json_object(request: Request) -> dict[str, Any]:
    """Parse a webhook request body and require a JSON object."""

    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Expected JSON object")
    return payload


def _is_deepgram_no_transcript_error(error: ValueError) -> bool:
    """Return True for Deepgram responses that contain no transcript text."""

    return str(error).startswith("Deepgram response did not include a transcript")


async def _log_terminal_deepgram_error(event: CallEvent, error_message: str) -> None:
    """Write a terminal Deepgram failure row to the audit log."""

    await log_result(
        event,
        MatchResult(
            crm_record_id=None,
            workspace=event.workspace,
            confidence=0.0,
            method=MatchMethod.UNMATCHED,
            requires_review=False,
            agent_name=event.agent_name,
        ),
        note=None,
        error=error_message,
    )


def _pause_call_intelligence_worker_scheduler() -> None:
    """Pause the Cloud Scheduler job that invokes the worker."""

    credentials, _ = google_auth_default(scopes=[_CLOUD_PLATFORM_SCOPE])
    session = AuthorizedSession(credentials)  # type: ignore[no-untyped-call]
    job_name = (
        f"projects/{_SCHEDULER_PROJECT_ID}/locations/{_SCHEDULER_REGION}/jobs/{_SCHEDULER_JOB_ID}"
    )
    response = session.post(
        f"{_CLOUD_SCHEDULER_API_BASE_URL}/{job_name}:pause",
        timeout=30,
    )
    response.raise_for_status()
    logger.info("Paused Cloud Scheduler job %s.", job_name)


def main() -> int:
    """Run one BigQuery polling cycle for Cloud Run Jobs or scheduled invocations."""

    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(process_polled_calls())
    except DeepgramCreditsExhaustedError:
        logger.critical("Deepgram credits exhausted — pausing Cloud Scheduler and stopping worker")
        try:
            _pause_call_intelligence_worker_scheduler()
        except Exception:
            logger.exception(
                "Failed to pause Cloud Scheduler job after Deepgram credits exhaustion."
            )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


app = create_app()
