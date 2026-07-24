"""RingCentral webhook subscription registration and renewal."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import httpx

from app.adapters.telephony.ringcentral import RingCentralAdapter
from app.config import get_settings

logger = logging.getLogger(__name__)

PLATFORM_BASE_URL = "https://platform.ringcentral.com"
RINGCENTRAL_EVENT_FILTER = "/restapi/v1.0/account/~/telephony/sessions"
RENEWAL_INTERVAL_SECONDS = 23 * 60 * 60


class RingCentralAuthAdapter(Protocol):
    """Minimal adapter surface needed for RingCentral authenticated requests."""

    async def get_access_token(self) -> str:
        """Return a RingCentral access token."""


SettingsProvider = Callable[[], object]
RequestSender = Callable[
    [str, str, dict[str, object], dict[str, str]],
    Awaitable[dict[str, Any]],
]
SleepFunc = Callable[[float], Awaitable[None]]


class RingCentralSubscriptionManager:
    """Manage one in-memory RingCentral webhook subscription for this service process."""

    def __init__(
        self,
        *,
        auth_adapter: RingCentralAuthAdapter | None = None,
        settings_provider: SettingsProvider | None = None,
        request_sender: RequestSender | None = None,
        sleep: SleepFunc = asyncio.sleep,
        platform_base_url: str = PLATFORM_BASE_URL,
        renewal_interval_seconds: float = RENEWAL_INTERVAL_SECONDS,
    ) -> None:
        self._auth_adapter = auth_adapter or RingCentralAdapter()
        self._settings_provider = settings_provider
        self._request_sender = request_sender or _send_json
        self._sleep = sleep
        self._platform_base_url = platform_base_url.rstrip("/")
        self._renewal_interval_seconds = renewal_interval_seconds
        self._subscription_id: str | None = None
        self._renewal_task: asyncio.Task[None] | None = None

    @property
    def subscription_id(self) -> str | None:
        """Return the RingCentral subscription id held by this process, if any."""

        return self._subscription_id

    @property
    def renewal_task(self) -> asyncio.Task[None] | None:
        """Return the active renewal task, if one has been started."""

        return self._renewal_task

    async def startup(self) -> None:
        """Register the subscription and start the renewal loop.

        This is intended to be called from a FastAPI startup hook once the receiver
        service is ready to advertise its public RingCentral webhook URL.
        """

        await self.register()
        self.start_renewal_loop()

    async def shutdown(self) -> None:
        """Cancel the renewal loop if it is running."""

        task = self._renewal_task
        if task is None:
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._renewal_task = None

    async def register(self) -> str:
        """Register a RingCentral webhook subscription and store its id in memory."""

        logger.info("Starting RingCentral webhook subscription registration.")
        try:
            webhook_url = self.webhook_url()
            logger.info(
                "Registering RingCentral webhook subscription. event_filter=%s webhook_url=%s",
                RINGCENTRAL_EVENT_FILTER,
                webhook_url,
            )
            response = await self._request(
                "POST",
                f"{self._platform_base_url}/restapi/v1.0/subscription",
                self._subscription_payload(webhook_url),
            )
            subscription_id = _extract_subscription_id(response)
            self._subscription_id = subscription_id
            logger.info(
                "Finished RingCentral webhook subscription registration. subscription_id=%s",
                subscription_id,
            )
            return subscription_id
        except Exception:
            logger.exception("RingCentral webhook subscription registration failed.")
            raise
        finally:
            logger.info("Ended RingCentral webhook subscription registration attempt.")

    async def renew(self) -> str:
        """Renew the currently stored RingCentral webhook subscription."""

        subscription_id = self._subscription_id
        if subscription_id is None:
            raise RuntimeError("RingCentral subscription has not been registered.")

        response = await self._request(
            "PUT",
            f"{self._platform_base_url}/restapi/v1.0/subscription/{subscription_id}",
            self._subscription_payload(self.webhook_url()),
        )
        renewed_id = _extract_subscription_id(response, default=subscription_id)
        self._subscription_id = renewed_id
        logger.info("Renewed RingCentral webhook subscription id=%s", renewed_id)
        return renewed_id

    def start_renewal_loop(self) -> asyncio.Task[None]:
        """Start a single background task that renews the subscription every 23 hours."""

        if self._renewal_task is not None and not self._renewal_task.done():
            return self._renewal_task

        self._renewal_task = asyncio.create_task(self.run_renewal_loop())
        return self._renewal_task

    async def run_renewal_loop(self) -> None:
        """Renew the registered subscription until the task is cancelled."""

        while True:
            await self._sleep(self._renewal_interval_seconds)
            await self.renew()

    def webhook_url(self) -> str:
        """Resolve the public RingCentral webhook URL from settings or SERVICE_URL."""

        return resolve_webhook_url(self._settings())

    def _settings(self) -> object:
        if self._settings_provider is not None:
            return self._settings_provider()
        return get_settings()

    async def _request(
        self,
        method: str,
        url: str,
        payload: dict[str, object],
    ) -> dict[str, Any]:
        access_token = await self._auth_adapter.get_access_token()
        return await self._request_sender(
            method,
            url,
            payload,
            {"Authorization": f"Bearer {access_token}"},
        )

    @staticmethod
    def _subscription_payload(webhook_url: str) -> dict[str, object]:
        return {
            "eventFilters": [RINGCENTRAL_EVENT_FILTER],
            "deliveryMode": {
                "transportType": "WebHook",
                "address": webhook_url,
            },
        }


def resolve_webhook_url(settings: object | None = None) -> str:
    """Resolve RINGCENTRAL_WEBHOOK_URL, falling back to SERVICE_URL when available."""

    resolved_settings = settings if settings is not None else get_settings()
    ringcentral_settings = getattr(getattr(resolved_settings, "telephony", None), "ringcentral", None)
    configured_webhook_url = _string_attr(ringcentral_settings, "webhook_url")
    if configured_webhook_url:
        return configured_webhook_url

    service_url = _string_attr(resolved_settings, "service_url") or os.getenv("SERVICE_URL", "")
    service_url = service_url.strip().rstrip("/")
    if service_url:
        if service_url.startswith(("http://", "https://")):
            return f"{service_url}/webhook/ringcentral"
        return f"https://{service_url}/webhook/ringcentral"

    raise RuntimeError("RINGCENTRAL_WEBHOOK_URL or SERVICE_URL must be configured.")


async def register_ringcentral_subscription() -> str:
    """Register the process-global RingCentral subscription manager."""

    return await subscription_manager.register()


async def start_ringcentral_subscription_manager() -> None:
    """Register and start renewal for the process-global subscription manager."""

    await subscription_manager.startup()


async def stop_ringcentral_subscription_manager() -> None:
    """Stop renewal for the process-global subscription manager."""

    await subscription_manager.shutdown()


async def _send_json(
    method: str,
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.request(method, url, json=payload, headers=headers)
        response.raise_for_status()

    response_payload = response.json()
    if not isinstance(response_payload, dict):
        raise RuntimeError("RingCentral subscription response was not a JSON object.")
    return response_payload


def _extract_subscription_id(payload: dict[str, Any], *, default: str | None = None) -> str:
    subscription_id = payload.get("id")
    if isinstance(subscription_id, str) and subscription_id.strip():
        return subscription_id.strip()
    if default:
        return default
    raise RuntimeError("RingCentral subscription response did not include id.")


def _string_attr(value: object | None, name: str) -> str:
    if value is None:
        return ""
    attribute = getattr(value, name, "")
    if isinstance(attribute, str):
        return attribute.strip()
    return ""


subscription_manager = RingCentralSubscriptionManager()
