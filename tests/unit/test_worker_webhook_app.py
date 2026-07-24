"""Tests for the composed worker FastAPI webhook service."""

from __future__ import annotations

import httpx
import pytest

from app.config import get_settings
from app.worker import main


async def _request(
    method: str,
    path: str,
    *,
    json: object | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=main.create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, json=json, headers=headers)


async def test_health_returns_ok() -> None:
    response = await _request("GET", "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_phoneburner_webhook_requires_shared_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBHOOK_SECRET_TOKEN", "secret")
    get_settings.cache_clear()
    try:
        response = await _request(
            "POST",
            "/webhook/phoneburner",
            json={"call_id": "pb-1"},
            headers={"X-Webhook-Token": "wrong"},
        )
    finally:
        get_settings.cache_clear()

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid webhook token"}


async def test_ringcentral_webhook_ignores_shared_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBHOOK_SECRET_TOKEN", "secret")
    get_settings.cache_clear()
    payload = {"body": {"id": "rc-1"}}

    async def fake_handle_ringcentral_webhook(
        received_payload: dict[str, object],
        background_tasks: object,
    ) -> dict[str, str]:
        assert received_payload == payload
        assert background_tasks is not None
        return {"status": "accepted", "call_id": "rc-1"}

    monkeypatch.setattr(
        "app.webhook.ringcentral.handle_ringcentral_webhook",
        fake_handle_ringcentral_webhook,
    )
    try:
        response = await _request(
            "POST",
            "/webhook/ringcentral",
            json=payload,
            headers={"X-Webhook-Token": "wrong"},
        )
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "call_id": "rc-1"}


async def test_ringcentral_validation_token_bypasses_shared_token() -> None:
    response = await _request(
        "POST",
        "/webhook/ringcentral",
        headers={"Validation-Token": "validation-token-123"},
    )

    assert response.status_code == 200
    assert response.headers["Validation-Token"] == "validation-token-123"
    assert response.content == b""


async def test_reprocess_intake_uses_legacy_phoneburner_polling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBHOOK_SECRET_TOKEN", "secret")
    get_settings.cache_clear()

    async def fake_process_legacy_intake_calls() -> int:
        return 2

    monkeypatch.setattr(main, "process_legacy_intake_calls", fake_process_legacy_intake_calls)
    try:
        response = await _request(
            "POST",
            "/reprocess/intake",
            headers={"X-Webhook-Token": "secret"},
        )
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "processed_count": 2}
