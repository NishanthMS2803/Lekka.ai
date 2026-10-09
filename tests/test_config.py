"""Tests for application settings."""

import pytest
from pydantic import ValidationError

from lekka_ai.config import Settings, get_settings


def test_missing_required_settings_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instantiating Settings with missing required fields raises ValidationError."""
    for key in [
        "DATABASE_URL",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_EMBEDDING_MODEL",
        "SEC_USER_AGENT",
    ]:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    errors = exc_info.value.errors()
    missing_fields = {err["loc"][0] for err in errors if err["type"] == "missing"}
    assert "DATABASE_URL" in missing_fields
    assert "GEMINI_API_KEY" in missing_fields
    assert "GEMINI_MODEL" in missing_fields
    assert "GEMINI_EMBEDDING_MODEL" in missing_fields
    assert "SEC_USER_AGENT" in missing_fields


def test_valid_settings_instantiation() -> None:
    """Instantiating Settings with all required fields succeeds."""
    settings = Settings(
        _env_file=None,
        DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/lekka",
        GEMINI_API_KEY="test-gemini-key",
        GEMINI_MODEL="gemini-2.0-flash",
        GEMINI_EMBEDDING_MODEL="text-embedding-004",
        SEC_USER_AGENT="LekkaAI Research bot@lekka.ai",
    )
    assert settings.ENV == "development"
    assert (
        settings.DATABASE_URL == "postgresql+asyncpg://user:pass@localhost:5432/lekka"
    )
    assert settings.GEMINI_API_KEY == "test-gemini-key"
    assert settings.GEMINI_MODEL == "gemini-2.0-flash"
    assert settings.GEMINI_EMBEDDING_MODEL == "text-embedding-004"
    assert settings.SEC_USER_AGENT == "LekkaAI Research bot@lekka.ai"
    assert settings.MARKET_DATA_API_KEY is None
    assert settings.NEWS_API_KEY is None
    assert settings.ESTIMATES_API_KEY is None


def test_get_settings_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings returns the same cached instance across multiple calls."""
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost:5432/db")
    monkeypatch.setenv("GEMINI_API_KEY", "key123")
    monkeypatch.setenv("GEMINI_MODEL", "model123")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "embed123")
    monkeypatch.setenv("SEC_USER_AGENT", "Agent contact@lekka.ai")

    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()
