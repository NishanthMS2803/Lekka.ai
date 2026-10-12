"""Query understanding module converting user questions into structured ParsedQuery."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lekka_ai.config import get_settings
from lekka_ai.data.sec import SecClient
from lekka_ai.llm.base import LLMClient, Message
from lekka_ai.llm.gemini import GeminiClient
from lekka_ai.tools.base import Tool, default_registry


class QueryIntent(StrEnum):
    """Allowed intent classifications for financial research queries."""

    INVESTMENT_EVALUATION = "investment_evaluation"
    VALUATION_CHECK = "valuation_check"
    MOVE_EXPLANATION = "move_explanation"
    RISK_REVIEW = "risk_review"
    OTHER = "other"


QueryIntentLiteral = Literal[
    "investment_evaluation",
    "valuation_check",
    "move_explanation",
    "risk_review",
    "other",
]


class UnknownTickerError(ValueError):
    """Raised when query understanding detects an invalid stock ticker."""


class ValidateTickerInput(BaseModel):
    """Input payload for validate_ticker tool."""

    ticker: str = Field(..., description="Stock ticker symbol to validate, e.g. 'AAPL'")


class ValidateTickerOutput(BaseModel):
    """Output payload from validate_ticker tool."""

    model_config = ConfigDict(extra="allow")

    ticker: str = Field(..., description="Queried ticker symbol")
    is_valid: bool = Field(..., description="True if ticker is recognized")
    cik: str | None = Field(default=None, description="10-digit CIK if found")
    company_name: str | None = Field(
        default=None, description="Company name if available"
    )
    error: str | None = Field(default=None, description="Error reason if invalid")


class ValidateTickerTool(Tool[ValidateTickerInput, ValidateTickerOutput]):
    """Tool for validating ticker symbols against SEC EDGAR and known universes."""

    name: str = "validate_ticker"
    description: str = (
        "Validate if a stock ticker symbol exists in the SEC EDGAR directory "
        "or recognized financial universe."
    )
    input_model: type[ValidateTickerInput] = ValidateTickerInput
    output_model: type[ValidateTickerOutput] = ValidateTickerOutput

    # Well-known fallback universe for offline/test environments
    KNOWN_TICKERS: set[str] = {
        "AAPL",
        "MSFT",
        "GOOG",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "SPY",
        "QQQ",
        "BRK.B",
        "JPM",
        "V",
        "JNJ",
        "WMT",
        "PG",
        "XOM",
        "UNH",
        "MA",
        "HD",
        "ORCL",
        "ADBE",
        "CRM",
        "NFLX",
        "AMD",
        "INTC",
        "DIS",
    }

    def __init__(
        self,
        sec_client: SecClient | None = None,
        allowed_tickers: set[str] | list[str] | None = None,
    ) -> None:
        self.sec_client = sec_client
        self.allowed_tickers = (
            {t.strip().upper() for t in allowed_tickers}
            if allowed_tickers is not None
            else None
        )

    async def run(self, input_data: ValidateTickerInput) -> ValidateTickerOutput:
        """Validate ticker symbol existence."""
        ticker = input_data.ticker.strip().upper()

        if not ticker:
            return ValidateTickerOutput(
                ticker=ticker,
                is_valid=False,
                error="Empty ticker symbol provided",
            )

        # 1. Custom allowed tickers set takes precedence
        if self.allowed_tickers is not None:
            if ticker in self.allowed_tickers:
                return ValidateTickerOutput(ticker=ticker, is_valid=True)
            return ValidateTickerOutput(
                ticker=ticker,
                is_valid=False,
                error=f"Ticker '{ticker}' not found in allowed universe.",
            )

        # 2. Known static universe check (instant offline path)
        if ticker in self.KNOWN_TICKERS:
            return ValidateTickerOutput(ticker=ticker, is_valid=True)

        # 3. Live SEC EDGAR verification
        client = self.sec_client or SecClient()
        try:
            cik = await client.get_cik(ticker)
            return ValidateTickerOutput(ticker=ticker, is_valid=True, cik=cik)
        except Exception as exc:
            return ValidateTickerOutput(
                ticker=ticker,
                is_valid=False,
                error=f"Ticker '{ticker}' not found in SEC EDGAR: {exc}",
            )


# Register default validate_ticker tool globally
default_validate_ticker_tool = ValidateTickerTool()
default_registry.register(default_validate_ticker_tool, overwrite=True)


class ParsedQuery(BaseModel):
    """Structured decomposition and intent understanding of a user question."""

    model_config = ConfigDict(extra="ignore")

    intent: QueryIntent | QueryIntentLiteral = Field(
        ...,
        description=(
            "Primary research intent: 'investment_evaluation', 'valuation_check', "
            "'move_explanation', 'risk_review', or 'other'"
        ),
    )
    tickers: list[str] = Field(
        default_factory=list,
        description="Extracted stock ticker symbols normalized to uppercase",
    )
    horizon: str | None = Field(
        default=None,
        description="Time horizon (e.g. 'short-term', '1 year', 'long-term')",
    )
    risk_focus: str | None = Field(
        default=None,
        description="Specific risk dimension (e.g. 'volatility', 'downside')",
    )
    as_of_date: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d"),
        description="Point-in-time cutoff date formatted as YYYY-MM-DD",
    )
    ambiguities: list[str] = Field(
        default_factory=list,
        description="List of ambiguous points or underspecified elements in query",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if query is ambiguous or lacks required assets/scope",
    )

    @field_validator("tickers", mode="before")
    @classmethod
    def _normalize_tickers(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            parts = [t.strip().upper() for t in v.split(",") if t.strip()]
            return list(dict.fromkeys(parts))
        if isinstance(v, (list, tuple, set)):
            cleaned: list[str] = []
            for item in v:
                s = str(item).strip().upper()
                if s and s not in cleaned:
                    cleaned.append(s)
            return cleaned
        return []

    @field_validator("intent", mode="before")
    @classmethod
    def _validate_intent(cls, v: Any) -> str:
        s = str(v).strip().lower()
        valid = {
            "investment_evaluation",
            "valuation_check",
            "move_explanation",
            "risk_review",
            "other",
        }
        if s not in valid:
            return "other"
        return s

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _validate_as_of_date(cls, v: Any) -> str:
        if not v:
            return datetime.now(UTC).strftime("%Y-%m-%d")
        return str(v).strip()


QUERY_UNDERSTANDING_SYSTEM_PROMPT = """You are a financial query parser.
Analyze the user's research question and extract a conforming ParsedQuery:

1. intent: Exactly one of:
   - "investment_evaluation": Evaluate whether to buy, hold, or sell.
   - "valuation_check": DCF intrinsic valuation, target price, multiples.
   - "move_explanation": Explaining price changes, earnings reactions, catalysts.
   - "risk_review": Quant risk, volatility, max drawdown, beta, or exposure.
   - "other": Broad industry trends, macro overviews, or unclassified.

2. tickers: Extract all US stock ticker symbols mentioned or implied
   (e.g. Apple -> AAPL, Microsoft -> MSFT, Tesla -> TSLA, Nvidia -> NVDA).
   Always uppercase.

3. horizon: Specific investment horizon (e.g. "short-term", "1 year", "3 years",
   "long-term", or null).

4. risk_focus: Specific risk dimension (e.g. "volatility", "downside", "beta",
   "margin compression", or null).

5. as_of_date: Cutoff date in YYYY-MM-DD format if explicitly specified or
   implied. If not mentioned, return null or the default date.

6. ambiguities: List any ambiguous points, missing ticker names, or
   underspecified scopes.

7. needs_clarification: Set to true if the question is ambiguous, lacks the
   necessary company/ticker, or cannot be confidently answered without
   clarifying details. Set to false if actionable."""


async def understand_query(
    question: str,
    llm: LLMClient | None = None,
    *,
    as_of_date: str | None = None,
    ticker_tool: ValidateTickerTool | None = None,
    validate_tickers: bool = True,
) -> ParsedQuery:
    """Analyze and decompose user question into a validated ParsedQuery.

    Args:
        question: Natural language question from the user.
        llm: Optional LLM client (defaults to GeminiClient with API key).
        as_of_date: Default point-in-time date if query lacks an explicit date.
        ticker_tool: Optional ValidateTickerTool for validating tickers.
        validate_tickers: When True, validates tickers and rejects unknown ones.

    Returns:
        Structured ParsedQuery with intent, tickers, horizon, and ambiguities.

    Raises:
        UnknownTickerError: If a ticker is rejected by validation tool.
    """
    default_date = as_of_date or datetime.now(UTC).strftime("%Y-%m-%d")

    # 1. Resolve LLM client
    if llm is None:
        settings = get_settings()
        llm = GeminiClient(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
        )

    # 2. Issue structured output generation
    user_prompt = (
        f"User Question: {question.strip()}\nDefault as_of_date: {default_date}"
    )
    response = await llm.generate(
        messages=[Message(role="user", content=user_prompt)],
        response_schema=ParsedQuery,
        system=QUERY_UNDERSTANDING_SYSTEM_PROMPT,
    )

    # 3. Parse LLM response into ParsedQuery instance
    parsed: ParsedQuery
    if isinstance(response.structured_output, ParsedQuery):
        parsed = response.structured_output
    elif isinstance(response.structured_output, dict):
        parsed = ParsedQuery.model_validate(response.structured_output)
    elif response.text:
        clean_text = response.text.strip()
        # Strip potential markdown fences
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        if fence_match:
            clean_text = fence_match.group(1).strip()
        try:
            data = json.loads(clean_text)
            parsed = ParsedQuery.model_validate(data)
        except Exception:
            parsed = ParsedQuery(
                intent="other",
                tickers=[],
                as_of_date=default_date,
                ambiguities=["Failed to parse structured LLM response"],
                needs_clarification=True,
            )
    else:
        parsed = ParsedQuery(
            intent="other",
            tickers=[],
            as_of_date=default_date,
            ambiguities=["Empty response from model"],
            needs_clarification=True,
        )

    # 4. Fallback to default as_of_date if empty
    if not parsed.as_of_date:
        parsed.as_of_date = default_date

    # 5. Ticker allowlist & directory validation
    settings = get_settings()
    allowlist = settings.TICKER_ALLOWLIST
    is_allowlist_enforced = settings.TICKER_ALLOWLIST_ENABLED or (
        allowlist is not None and len(allowlist) > 0
    )
    allowed_set = {t.strip().upper() for t in allowlist} if allowlist else set()

    if parsed.tickers and is_allowlist_enforced:
        for ticker in parsed.tickers:
            if ticker.strip().upper() not in allowed_set:
                raise UnknownTickerError(
                    f"Ticker '{ticker}' is not in the allowed ticker universe."
                )

    if validate_tickers and parsed.tickers:
        validator = ticker_tool or default_validate_ticker_tool
        for ticker in parsed.tickers:
            check_output = await validator.run(ValidateTickerInput(ticker=ticker))
            if not check_output.is_valid:
                raise UnknownTickerError(
                    f"Unknown ticker symbol '{ticker}'. "
                    f"{check_output.error or 'Ticker is not recognized.'}"
                )

    return parsed
