"""Tool retrieving standardized and derived financial statements facts."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lekka_ai.data.sec import SecClient
from lekka_ai.data.xbrl import MetricSource, parse_company_facts
from lekka_ai.quant.fundamentals import (
    DerivedFinancials,
    compute_derived_financials,
)
from lekka_ai.tools.base import Tool, default_registry


class FinancialPeriodSummary(BaseModel):
    """Standardized and derived metrics for a fiscal period with provenance."""

    model_config = ConfigDict(extra="allow")

    fy: int = Field(..., description="Fiscal year")
    fp: str = Field(..., description="Fiscal period (e.g. 'FY', 'Q1', 'Q2', 'Q3')")
    period: str = Field(..., description="Period identifier (e.g. '2023-FY')")
    end_date: str | None = Field(default=None, description="Fiscal period end date")
    filed_date: str | None = Field(default=None, description="Filing submission date")
    form: str | None = Field(default=None, description="Primary filing form")

    # Standardized GAAP statement facts
    revenues: float | int | None = Field(
        default=None, description="Total revenues or sales"
    )
    operating_income: float | int | None = Field(
        default=None, description="Operating income (EBIT)"
    )
    net_income: float | int | None = Field(
        default=None, description="Net income or loss"
    )
    operating_cash_flow: float | int | None = Field(
        default=None, description="Operating cash flow (CFO)"
    )
    capex: float | int | None = Field(
        default=None, description="Capital expenditures (PPE payments)"
    )
    cash_and_cash_equivalents: float | int | None = Field(
        default=None, description="Cash and cash equivalents"
    )
    debt: float | int | None = Field(
        default=None, description="Total or long-term debt"
    )

    # Derived fundamental metrics
    derived: DerivedFinancials = Field(
        default_factory=DerivedFinancials,
        description="Derived fundamental and profitability metrics",
    )

    # Source provenance per metric: maps metric to tag, accession, filed date
    source_info: dict[str, MetricSource] = Field(
        default_factory=dict,
        description="Source provenance (tag, accession_number, filed_date, form)",
    )

    @property
    def revenue_yoy(self) -> float | None:
        return self.derived.revenue_yoy

    @property
    def net_income_yoy(self) -> float | None:
        return self.derived.net_income_yoy

    @property
    def operating_margin(self) -> float | None:
        return self.derived.operating_margin

    @property
    def net_margin(self) -> float | None:
        return self.derived.net_margin

    @property
    def free_cash_flow(self) -> float | int | None:
        return self.derived.free_cash_flow

    @property
    def net_debt(self) -> float | int | None:
        return self.derived.net_debt

    def __getitem__(self, item: str) -> Any:
        norm = item.strip().lower()
        if norm in (
            "revenue_yoy",
            "net_income_yoy",
            "operating_margin",
            "net_margin",
            "free_cash_flow",
            "net_debt",
        ):
            return getattr(self.derived, norm)
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"Key '{item}' not found in FinancialPeriodSummary.")


class GetFinancialsInput(BaseModel):
    """Input payload for get_financials tool."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. 'AAPL') or CIK")
    periods: int = Field(
        default=4,
        description="Number of recent periods to retrieve (default: 4)",
    )
    as_of_date: str | None = Field(
        default=None,
        description=(
            "Point-in-time cutoff date (YYYY-MM-DD). "
            "Only facts filed on or before this date are included."
        ),
    )
    period_type: str = Field(
        default="annual",
        description="Period frequency: 'annual' (default) or 'quarterly'",
    )

    @field_validator("periods", mode="before")
    @classmethod
    def _validate_periods(cls, v: Any) -> int:
        try:
            val = int(v)
            return max(1, val)
        except (ValueError, TypeError):
            return 4

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _validate_as_of_date(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, (date, datetime)):
            return v.strftime("%Y-%m-%d")
        s = str(v).strip()
        return s if s else None

    @field_validator("period_type", mode="before")
    @classmethod
    def _validate_period_type(cls, v: Any) -> str:
        s = str(v).strip().lower() if v else "annual"
        return "quarterly" if "quarter" in s else "annual"


class GetFinancialsOutput(BaseModel):
    """Output payload for get_financials tool."""

    model_config = ConfigDict(extra="allow")

    ticker: str = Field(..., description="Stock ticker symbol queried")
    cik: str = Field(..., description="Zero-padded 10-digit CIK")
    company_name: str | None = Field(
        default=None, description="Official legal entity name"
    )
    as_of_date: str | None = Field(
        default=None, description="Point-in-time cutoff date applied"
    )
    period_type: str = Field(
        default="annual", description="Frequency of periods returned"
    )
    total_periods: int = Field(
        default=0, description="Total periods returned in this payload"
    )
    periods: list[FinancialPeriodSummary] = Field(
        default_factory=list, description="Chronological financial period summaries"
    )

    def __getitem__(self, item: str | int) -> Any:
        if isinstance(item, int):
            return self.periods[item]
        return getattr(self, item)

    def __len__(self) -> int:
        return len(self.periods)

    def __iter__(self) -> Any:
        return iter(self.periods)


class GetFinancialsTool(Tool[GetFinancialsInput, GetFinancialsOutput]):
    """Tool retrieving official SEC XBRL company facts and computed derived metrics."""

    name: str = "get_financials"
    description: str = (
        "Retrieve standardized company financials (Revenues, Net Income, Operating "
        "Income, Operating Cash Flow, Capex, Cash, Debt) with source provenance (tag, "
        "accession number, filing date) and computed derived metrics (YoY growth, "
        "operating margin, net margin, free cash flow, net debt). Enforces zero "
        "lookahead bias via point-in-time filtering (as_of_date)."
    )
    input_model: type[GetFinancialsInput] = GetFinancialsInput
    output_model: type[GetFinancialsOutput] = GetFinancialsOutput

    def __init__(self, sec_client: SecClient | None = None) -> None:
        self.sec_client: SecClient = sec_client or SecClient()

    async def run(self, input_data: GetFinancialsInput) -> GetFinancialsOutput:
        """Execute financial extraction and fundamental metrics computation."""
        ticker = input_data.ticker.strip().upper()

        # 1. Resolve ticker to CIK
        cik = await self.sec_client.get_cik(ticker)

        # 2. Fetch raw XBRL company facts from SEC EDGAR
        raw_facts = await self.sec_client.get_company_facts(cik)
        company_name = raw_facts.get(
            "entityName"
        ) or await self.sec_client.get_company_name(ticker)

        # 3. Parse facts with point-in-time cutoff: filed <= as_of_date
        parsed = parse_company_facts(raw_facts, as_of_date=input_data.as_of_date)

        # 4. Filter target period frequency (annual vs quarterly)
        if input_data.period_type == "quarterly":
            target_list = parsed.quarterly()
        else:
            target_list = parsed.annual()

        # Build summary list with derived metrics (calculating YoY against history)
        summaries: list[FinancialPeriodSummary] = []
        for p in target_list:
            # Look up prior comparable period: (p.fy - 1, p.fp)
            prior_p = parsed.get((p.fy - 1, p.fp))
            derived_metrics = compute_derived_financials(p, prior=prior_p)

            item = FinancialPeriodSummary(
                fy=p.fy,
                fp=p.fp,
                period=f"{p.fy}-{p.fp}",
                end_date=p.end,
                filed_date=p.filed,
                form=p.form,
                revenues=p.revenues,
                operating_income=p.operating_income,
                net_income=p.net_income,
                operating_cash_flow=p.operating_cash_flow,
                capex=p.capex,
                cash_and_cash_equivalents=p.cash_and_cash_equivalents,
                debt=p.debt,
                derived=derived_metrics,
                source_info=p.sources,
            )
            summaries.append(item)

        # Sort descending by period (newest first)
        summaries.sort(key=lambda s: (s.fy, s.fp), reverse=True)

        # Slice to requested number of periods
        sliced = summaries[: input_data.periods]

        return GetFinancialsOutput(
            ticker=ticker,
            cik=cik,
            company_name=company_name,
            as_of_date=input_data.as_of_date,
            period_type=input_data.period_type,
            total_periods=len(sliced),
            periods=sliced,
        )


FinancialsTool = GetFinancialsTool

# Register default instance in tool registry
default_financials_tool = GetFinancialsTool()
default_registry.register(default_financials_tool, overwrite=True)
