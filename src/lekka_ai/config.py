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
    EMBEDDING_DIMENSION: int = Field(
        default=768,
        description="Dimension of embeddings matching GEMINI_EMBEDDING_MODEL",
    )
    SEC_USER_AGENT: str = Field(
        ..., description="User-Agent header required for SEC EDGAR queries"
    )
    MARKET_DATA_API_KEY: str | None = Field(
        default=None, description="Optional API key for market data providers"
    )
    NEWS_PROVIDER: str = Field(
        default="finnhub",
        description="Default news provider (finnhub, tavily, newsapi)",
    )
    NEWS_API_KEY: str | None = Field(
        default=None, description="Optional API key for financial news"
    )
    ESTIMATES_API_KEY: str | None = Field(
        default=None, description="Optional API key for consensus estimates"
    )
    VERIFIER_MODEL: str = Field(
        default="gemini-flash-lite-latest",
        description="Cheap model setting for verifier tasks",
    )
    VERIFIER_BATCH_SIZE: int = Field(
        default=5,
        description="Batch size for claim verification LLM calls",
    )
    INPUT_TOKEN_PRICE_PER_M: float = Field(
        default=0.15,
        description="USD cost per 1M input tokens",
    )
    OUTPUT_TOKEN_PRICE_PER_M: float = Field(
        default=0.60,
        description="USD cost per 1M output tokens",
    )
    DEFAULT_RUN_BUDGET_TOOL_CALLS: int = Field(
        default=50,
        description="Default run-level tool calls budget limit",
    )
    DEFAULT_RUN_BUDGET_TOKENS: int = Field(
        default=300000,
        description="Default run-level token budget limit",
    )
    DEFAULT_RUN_BUDGET_SECONDS: float = Field(
        default=180.0,
        description="Default run-level execution seconds budget limit",
    )
    DEFAULT_RUN_BUDGET_USD: float = Field(
        default=1.0,
        description="Default run-level USD cost budget limit",
    )

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
