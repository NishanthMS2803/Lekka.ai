"""Tools for analyst consensus estimates and peer/industry multiples."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lekka_ai.data.estimates import (
    EstimatesProvider,
    create_estimates_provider,
)
from lekka_ai.logging import get_logger
from lekka_ai.tools.base import Tool, default_registry

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Models for get_analyst_estimates
# ---------------------------------------------------------------------------


class AnalystEstimatesPayload(BaseModel):
    """Structured analyst consensus price targets and forward estimates."""

    model_config = ConfigDict(extra="allow")

    ticker: str = Field(..., description="Stock ticker symbol")
    target_price_mean: float | None = Field(
        default=None, description="Average target price"
    )
    target_price_high: float | None = Field(
        default=None, description="High target price"
    )
    target_price_low: float | None = Field(default=None, description="Low target price")
    target_price_median: float | None = Field(
        default=None, description="Median target price"
    )
    consensus_rating: str | None = Field(
        default=None, description="Analyst consensus rating (e.g. Buy, Hold, Sell)"
    )
    num_analysts: int | None = Field(
        default=None, description="Number of covering analysts"
    )
    eps_estimate_current_year: float | None = Field(
        default=None, description="Forward EPS consensus estimate"
    )
    revenue_estimate_current_year: float | None = Field(
        default=None, description="Forward revenue consensus estimate"
    )
    period: str | None = Field(default=None, description="Fiscal period of estimate")
    source: str = Field(default="EstimatesProvider", description="Data provider source")
    as_of_date: str | None = Field(
        default=None, description="Point-in-time date of estimates"
    )


class GetAnalystEstimatesInput(BaseModel):
    """Input payload for get_analyst_estimates tool."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. 'AAPL')")
    as_of_date: str | None = Field(
        default=None,
        description="Point-in-time cutoff date (YYYY-MM-DD)",
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def _validate_ticker(cls, v: Any) -> str:
        clean = str(v).strip().upper()
        if not clean:
            raise ValueError("Ticker symbol cannot be empty")
        return clean

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _validate_as_of_date(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, (date, datetime)):
            return v.strftime("%Y-%m-%d")
        clean = str(v).strip()
        return clean or None


class GetAnalystEstimatesOutput(BaseModel):
    """Output payload for get_analyst_estimates tool with graceful degradation."""

    model_config = ConfigDict(extra="allow")

    status: str = Field(
        ...,
        description="Status indicator: 'available' or 'unavailable'",
    )
    available: bool = Field(
        ...,
        description="Boolean flag indicating whether estimates data was found",
    )
    ticker: str = Field(..., description="Stock ticker queried")
    as_of_date: str | None = Field(
        default=None, description="Point-in-time cutoff applied"
    )
    message: str | None = Field(
        default=None,
        description="Informative message explaining unavailability reason",
    )
    estimates: AnalystEstimatesPayload | None = Field(
        default=None,
        description="Analyst estimates payload when available",
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class GetAnalystEstimatesTool(
    Tool[GetAnalystEstimatesInput, GetAnalystEstimatesOutput]
):
    """Tool for fetching consensus analyst price targets and recommendations."""

    name: str = "get_analyst_estimates"
    description: str = (
        "Retrieve consensus analyst price targets, ratings, and forward estimates. "
        "Returns a structured 'unavailable' result if no API key is configured or the "
        "call fails, allowing agents to degrade gracefully without exceptions."
    )
    input_model: type[GetAnalystEstimatesInput] = GetAnalystEstimatesInput
    output_model: type[GetAnalystEstimatesOutput] = GetAnalystEstimatesOutput

    def __init__(self, provider: EstimatesProvider | None = None) -> None:
        self._provider = provider

    def _get_provider(self) -> EstimatesProvider:
        if self._provider is not None:
            return self._provider
        return create_estimates_provider()

    async def run(
        self, input_data: GetAnalystEstimatesInput
    ) -> GetAnalystEstimatesOutput:
        """Fetch analyst estimates, returning unavailable result on failure."""
        ticker = input_data.ticker
        as_of_date_obj: date | None = None
        if input_data.as_of_date:
            try:
                as_of_date_obj = date.fromisoformat(input_data.as_of_date)
            except ValueError:
                logger.warning(
                    "Invalid as_of_date format", as_of_date=input_data.as_of_date
                )

        try:
            provider = self._get_provider()
            estimates_data = await provider.get_estimates(
                ticker=ticker,
                as_of_date=as_of_date_obj,
            )
        except Exception as exc:
            logger.info(
                "Estimates query failed gracefully",
                ticker=ticker,
                error=str(exc),
            )
            return GetAnalystEstimatesOutput(
                status="unavailable",
                available=False,
                ticker=ticker,
                as_of_date=input_data.as_of_date,
                message=f"Estimates provider query failed: {exc}",
                estimates=None,
            )

        if estimates_data is None:
            logger.info("Estimates data unavailable or unconfigured", ticker=ticker)
            return GetAnalystEstimatesOutput(
                status="unavailable",
                available=False,
                ticker=ticker,
                as_of_date=input_data.as_of_date,
                message=(
                    "Estimates unavailable (no data or ESTIMATES_API_KEY unconfigured)."
                ),
                estimates=None,
            )

        payload = AnalystEstimatesPayload(
            ticker=estimates_data.ticker,
            target_price_mean=estimates_data.target_price_mean,
            target_price_high=estimates_data.target_price_high,
            target_price_low=estimates_data.target_price_low,
            target_price_median=estimates_data.target_price_median,
            consensus_rating=estimates_data.consensus_rating,
            num_analysts=estimates_data.num_analysts,
            eps_estimate_current_year=estimates_data.eps_estimate_current_year,
            revenue_estimate_current_year=estimates_data.revenue_estimate_current_year,
            period=estimates_data.period,
            source=estimates_data.source,
            as_of_date=estimates_data.as_of_date or input_data.as_of_date,
        )

        return GetAnalystEstimatesOutput(
            status="available",
            available=True,
            ticker=ticker,
            as_of_date=input_data.as_of_date,
            message=None,
            estimates=payload,
        )


# ---------------------------------------------------------------------------
# Models for get_industry_data
# ---------------------------------------------------------------------------


class IndustryDataPayload(BaseModel):
    """Structured sector, industry, peer group, and valuation multiples."""

    model_config = ConfigDict(extra="allow")

    ticker: str = Field(..., description="Stock ticker symbol")
    sector: str = Field(..., description="Economic sector classification")
    industry: str = Field(..., description="Specific industry classification")
    peer_tickers: list[str] = Field(
        default_factory=list, description="Direct industry peer ticker symbols"
    )
    pe_ratio_median: float | None = Field(
        default=None, description="Median Price-to-Earnings ratio of peers"
    )
    ev_ebitda_median: float | None = Field(
        default=None, description="Median EV/EBITDA multiple of peers"
    )
    price_to_sales_median: float | None = Field(
        default=None, description="Median Price-to-Sales multiple of peers"
    )
    source: str = Field(default="EstimatesProvider", description="Data source provider")
    as_of_date: str | None = Field(
        default=None, description="Point-in-time date of multiples"
    )


class GetIndustryDataInput(BaseModel):
    """Input payload for get_industry_data tool."""

    ticker: str = Field(
        ...,
        description="Company stock ticker to retrieve industry peers and multiples for",
    )
    as_of_date: str | None = Field(
        default=None,
        description="Point-in-time cutoff date (YYYY-MM-DD)",
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def _validate_ticker(cls, v: Any) -> str:
        clean = str(v).strip().upper()
        if not clean:
            raise ValueError("Ticker symbol cannot be empty")
        return clean

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _validate_as_of_date(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, (date, datetime)):
            return v.strftime("%Y-%m-%d")
        clean = str(v).strip()
        return clean or None


class GetIndustryDataOutput(BaseModel):
    """Output payload for get_industry_data tool with graceful degradation."""

    model_config = ConfigDict(extra="allow")

    status: str = Field(
        ...,
        description="Status indicator: 'available' or 'unavailable'",
    )
    available: bool = Field(
        ...,
        description="Boolean flag indicating whether industry data was found",
    )
    ticker: str = Field(..., description="Stock ticker queried")
    as_of_date: str | None = Field(
        default=None, description="Point-in-time cutoff applied"
    )
    message: str | None = Field(
        default=None,
        description="Informative message explaining unavailability reason",
    )
    industry_data: IndustryDataPayload | None = Field(
        default=None,
        description="Industry and peer multiples payload when available",
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class GetIndustryDataTool(Tool[GetIndustryDataInput, GetIndustryDataOutput]):
    """Tool for fetching sector/industry classification, peers, and multiples."""

    name: str = "get_industry_data"
    description: str = (
        "Retrieve sector, industry group, direct peers, and valuation multiples. "
        "Returns 'unavailable' result if no API key configured or call fails."
    )
    input_model: type[GetIndustryDataInput] = GetIndustryDataInput
    output_model: type[GetIndustryDataOutput] = GetIndustryDataOutput

    def __init__(self, provider: EstimatesProvider | None = None) -> None:
        self._provider = provider

    def _get_provider(self) -> EstimatesProvider:
        if self._provider is not None:
            return self._provider
        return create_estimates_provider()

    async def run(self, input_data: GetIndustryDataInput) -> GetIndustryDataOutput:
        """Fetch industry data, returning structured unavailable result on failure."""
        ticker = input_data.ticker
        as_of_date_obj: date | None = None
        if input_data.as_of_date:
            try:
                as_of_date_obj = date.fromisoformat(input_data.as_of_date)
            except ValueError:
                logger.warning(
                    "Invalid as_of_date format", as_of_date=input_data.as_of_date
                )

        try:
            provider = self._get_provider()
            ind_data = await provider.get_industry_data(
                ticker=ticker,
                as_of_date=as_of_date_obj,
            )
        except Exception as exc:
            logger.info(
                "Industry data query failed gracefully",
                ticker=ticker,
                error=str(exc),
            )
            return GetIndustryDataOutput(
                status="unavailable",
                available=False,
                ticker=ticker,
                as_of_date=input_data.as_of_date,
                message=f"Industry data provider query failed: {exc}",
                industry_data=None,
            )

        if ind_data is None:
            logger.info("Industry data unavailable or unconfigured", ticker=ticker)
            return GetIndustryDataOutput(
                status="unavailable",
                available=False,
                ticker=ticker,
                as_of_date=input_data.as_of_date,
                message=(
                    "Industry data unavailable "
                    "(no data or ESTIMATES_API_KEY unconfigured)."
                ),
                industry_data=None,
            )

        payload = IndustryDataPayload(
            ticker=ind_data.ticker,
            sector=ind_data.sector,
            industry=ind_data.industry,
            peer_tickers=ind_data.peer_tickers,
            pe_ratio_median=ind_data.pe_ratio_median,
            ev_ebitda_median=ind_data.ev_ebitda_median,
            price_to_sales_median=ind_data.price_to_sales_median,
            source=ind_data.source,
            as_of_date=ind_data.as_of_date or input_data.as_of_date,
        )

        return GetIndustryDataOutput(
            status="available",
            available=True,
            ticker=ticker,
            as_of_date=input_data.as_of_date,
            message=None,
            industry_data=payload,
        )


AnalystEstimatesTool = GetAnalystEstimatesTool
IndustryDataTool = GetIndustryDataTool

# Register default instances in default tool registry
default_analyst_estimates_tool = GetAnalystEstimatesTool()
default_industry_data_tool = GetIndustryDataTool()
default_registry.register(default_analyst_estimates_tool, overwrite=True)
default_registry.register(default_industry_data_tool, overwrite=True)
