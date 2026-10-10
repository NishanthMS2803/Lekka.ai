"""Configuration settings for Lekka AI."""

from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENV: str = Field(default="development", description="Environment name")
    DATABASE_URL: str = Field(..., description="PostgreSQL database connection URL")
    GEMINI_API_KEY: str = Field(..., description="API key for Google Gemini")
    GEMINI_MODEL: str = Field(..., description="Gemini LLM model name")
    GEMINI_EMBEDDING_MODEL: str = Field(
        ..., description="Gemini text embedding model name"
    )
    SEC_USER_AGENT: str = Field(
        ..., description="User-Agent header required for SEC EDGAR queries"
    )
    MARKET_DATA_API_KEY: str | None = Field(
        default=None, description="Optional API key for market data providers"
    )
    NEWS_API_KEY: str | None = Field(
        default=None, description="Optional API key for financial news"
    )
    ESTIMATES_API_KEY: str | None = Field(
        default=None, description="Optional API key for consensus estimates"
    )

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
