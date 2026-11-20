"""Tests for application configuration."""

import pytest

from app.config import PipelineSettings, STTSettings


def test_phoneburner_has_no_implicit_channel_role_map() -> None:
    assert STTSettings(_env_file=None).phoneburner_channel_roles == {}


def test_phoneburner_explicit_channel_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHONEBURNER_CHANNEL_ROLES", '{"0":"Lead","1":"Agent"}')
    assert STTSettings(_env_file=None).phoneburner_channel_roles == {"0": "Lead", "1": "Agent"}


@pytest.mark.parametrize("mapping", ['{"x":"Agent"}', '{"0":"Manager"}', '{"-1":"Lead"}'])
def test_phoneburner_invalid_channel_maps_fail_configuration(
    mapping: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHONEBURNER_CHANNEL_ROLES", mapping)
    with pytest.raises(ValueError):
        STTSettings(_env_file=None)


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
