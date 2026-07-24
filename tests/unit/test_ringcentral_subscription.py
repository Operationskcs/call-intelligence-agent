"""Tests for RingCentral webhook subscription management."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import Settings, get_settings
from app.services.ringcentral_subscription import (
    RENEWAL_INTERVAL_SECONDS,
    RINGCENTRAL_EVENT_FILTER,
    RingCentralSubscriptionManager,
    resolve_webhook_url,
)


class FakeRingCentralAuthAdapter:
    """Fake auth adapter matching the RingCentral token interface."""

    async def get_access_token(self) -> str:
        return "access-token"


def _settings(
    *,
    webhook_url: str = "",
    service_url: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        service_url=service_url,
        telephony=SimpleNamespace(
            ringcentral=SimpleNamespace(webhook_url=webhook_url),
        ),
    )


def test_settings_loads_ringcentral_webhook_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """RINGCENTRAL_WEBHOOK_URL is available through nested RingCentral settings."""

    monkeypatch.setenv("RINGCENTRAL_WEBHOOK_URL", "https://example.com/webhook/ringcentral")
    get_settings.cache_clear()
    try:
        assert Settings().telephony.ringcentral.webhook_url == (
            "https://example.com/webhook/ringcentral"
        )
    finally:
        get_settings.cache_clear()


def test_resolve_webhook_url_prefers_ringcentral_webhook_url() -> None:
    assert (
        resolve_webhook_url(_settings(webhook_url="https://service.example/webhook/ringcentral"))
        == "https://service.example/webhook/ringcentral"
    )


def test_resolve_webhook_url_falls_back_to_service_url() -> None:
    assert (
        resolve_webhook_url(_settings(service_url="call-service.example"))
        == "https://call-service.example/webhook/ringcentral"
    )


async def test_register_creates_subscription_and_stores_id() -> None:
    requests: list[dict[str, Any]] = []

    async def fake_request_sender(
        method: str,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        requests.append(
            {
                "method": method,
                "url": url,
                "payload": payload,
                "headers": headers,
            }
        )
        return {"id": "sub-123"}

    manager = RingCentralSubscriptionManager(
        auth_adapter=FakeRingCentralAuthAdapter(),
        settings_provider=lambda: _settings(
            webhook_url="https://receiver.example/webhook/ringcentral"
        ),
        request_sender=fake_request_sender,
    )

    subscription_id = await manager.register()

    assert subscription_id == "sub-123"
    assert manager.subscription_id == "sub-123"
    assert requests == [
        {
            "method": "POST",
            "url": "https://platform.ringcentral.com/restapi/v1.0/subscription",
            "payload": {
                "eventFilters": [RINGCENTRAL_EVENT_FILTER],
                "deliveryMode": {
                    "transportType": "WebHook",
                    "address": "https://receiver.example/webhook/ringcentral",
                },
            },
            "headers": {"Authorization": "Bearer access-token"},
        }
    ]


async def test_renew_uses_put_with_stored_subscription_id() -> None:
    requests: list[dict[str, Any]] = []

    async def fake_request_sender(
        method: str,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        requests.append(
            {
                "method": method,
                "url": url,
                "payload": payload,
                "headers": headers,
            }
        )
        if method == "POST":
            return {"id": "sub-123"}
        return {"id": "sub-456"}

    manager = RingCentralSubscriptionManager(
        auth_adapter=FakeRingCentralAuthAdapter(),
        settings_provider=lambda: _settings(
            webhook_url="https://receiver.example/webhook/ringcentral"
        ),
        request_sender=fake_request_sender,
    )
    await manager.register()

    renewed_id = await manager.renew()

    assert renewed_id == "sub-456"
    assert manager.subscription_id == "sub-456"
    assert requests[-1] == {
        "method": "PUT",
        "url": "https://platform.ringcentral.com/restapi/v1.0/subscription/sub-123",
        "payload": {
            "eventFilters": [RINGCENTRAL_EVENT_FILTER],
            "deliveryMode": {
                "transportType": "WebHook",
                "address": "https://receiver.example/webhook/ringcentral",
            },
        },
        "headers": {"Authorization": "Bearer access-token"},
    }


async def test_run_renewal_loop_sleeps_for_23_hours_before_renewing() -> None:
    class StopLoop(Exception):
        pass

    sleeps: list[float] = []
    methods: list[str] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) > 1:
            raise StopLoop

    async def fake_request_sender(
        method: str,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        _ = url, payload, headers
        methods.append(method)
        return {"id": "sub-123"}

    manager = RingCentralSubscriptionManager(
        auth_adapter=FakeRingCentralAuthAdapter(),
        settings_provider=lambda: _settings(
            webhook_url="https://receiver.example/webhook/ringcentral"
        ),
        request_sender=fake_request_sender,
        sleep=fake_sleep,
    )
    await manager.register()

    with pytest.raises(StopLoop):
        await manager.run_renewal_loop()

    assert sleeps == [RENEWAL_INTERVAL_SECONDS, RENEWAL_INTERVAL_SECONDS]
    assert methods == ["POST", "PUT"]


async def test_startup_registers_and_starts_cancellable_renewal_task() -> None:
    sleep_started = asyncio.Event()

    async def sleeping_until_cancelled(seconds: float) -> None:
        assert seconds == RENEWAL_INTERVAL_SECONDS
        sleep_started.set()
        await asyncio.Future()

    async def fake_request_sender(
        method: str,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        _ = method, url, payload, headers
        return {"id": "sub-123"}

    manager = RingCentralSubscriptionManager(
        auth_adapter=FakeRingCentralAuthAdapter(),
        settings_provider=lambda: _settings(
            webhook_url="https://receiver.example/webhook/ringcentral"
        ),
        request_sender=fake_request_sender,
        sleep=sleeping_until_cancelled,
    )

    await manager.startup()
    await asyncio.wait_for(sleep_started.wait(), timeout=1.0)
    await manager.shutdown()

    assert manager.subscription_id == "sub-123"
    assert manager.renewal_task is None
