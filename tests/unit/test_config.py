"""Tests for application configuration."""

import pytest

from app.config import PipelineSettings


def test_phoneburner_polling_enabled_by_default() -> None:
    """PhoneBurner polling should stay enabled unless env disables it."""

    settings = PipelineSettings(_env_file=None)

    assert settings.enable_phoneburner_polling is True


def test_phoneburner_polling_can_be_disabled_by_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production env var should be able to disable PhoneBurner polling."""

    monkeypatch.setenv("ENABLE_PHONEBURNER_POLLING", "false")

    settings = PipelineSettings(_env_file=None)

    assert settings.enable_phoneburner_polling is False
