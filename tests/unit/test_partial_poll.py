from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.worker import main
from app.worker.steps import s1_ingest as poll


def failure(code=429, path="/restapi/v1.0/account/~/call-log"):
    request = httpx.Request("GET", "https://platform.ringcentral.com" + path)
    return httpx.HTTPStatusError(
        "fixture", request=request, response=httpx.Response(code, request=request)
    )


async def test_intake_is_drained_without_swallowing_provider_failure(monkeypatch, mock_call_event):
    event = mock_call_event
    other = event.model_copy(update={"call_id": "already-done"})
    monkeypatch.setattr(
        poll,
        "get_settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(enable_phoneburner_polling=True)),
    )
    monkeypatch.setattr(poll, "_poll_phoneburner", AsyncMock(return_value=[event, event, other]))
    error = failure()
    monkeypatch.setattr(poll, "_poll_ringcentral_api", AsyncMock(side_effect=error))
    checked = AsyncMock(return_value={"already-done"})
    monkeypatch.setattr(poll, "processed_call_ids", checked)
    run = AsyncMock()
    monkeypatch.setattr(main.pipeline, "run", run)
    with pytest.raises(poll.PartialPollRateLimit) as caught:
        await main.process_polled_calls()
    checked.assert_awaited_once_with([event.call_id, "already-done"])
    run.assert_awaited_once_with(event)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("code,path", [(403, "/call-log"), (429, "/restapi/oauth/token")])
async def test_other_errors_propagate_unchanged(monkeypatch, code, path):
    error = failure(code, path)
    monkeypatch.setattr(poll, "_poll_ringcentral_api", AsyncMock(side_effect=error))
    with pytest.raises(httpx.HTTPStatusError) as caught:
        await poll._ringcentral_poll_result()
    assert caught.value is error
