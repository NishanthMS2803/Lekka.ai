"""SEC EDGAR filings listing tool with point-in-time cutoff filtering."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lekka_ai.data.sec import SecClient
from lekka_ai.tools.base import Tool, default_registry


class FilingItem(BaseModel):
    """Metadata for a single SEC EDGAR filing."""

    model_config = ConfigDict(extra="allow")

    form: str = Field(..., description="SEC form type (e.g. 10-K, 10-Q, 8-K)")
    filing_date: str = Field(..., description="Date filing was submitted (YYYY-MM-DD)")
    report_date: str | None = Field(
        default=None, description="Fiscal period end date reported in filing"
    )
    accession_number: str = Field(..., description="SEC EDGAR unique accession number")
    primary_document_url: str = Field(
        ..., description="Canonical URL to view document on SEC.gov"
    )
    primary_document: str | None = Field(
        default=None, description="Filename of the primary document"
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class ListFilingsInput(BaseModel):
    """Input payload for list_filings tool."""

    ticker: str = Field(
        ..., description="Stock ticker symbol (e.g. 'AAPL') or numeric CIK"
    )
    form_types: list[str] = Field(
        default_factory=lambda: ["10-K", "10-Q", "8-K"],
        description="List of SEC form types to include (default: 10-K, 10-Q, 8-K)",
    )
    limit: int = Field(default=10, description="Maximum number of filings to return")
    as_of_date: str | None = Field(
        default=None,
        description=(
            "Point-in-time cutoff date (YYYY-MM-DD). "
            "Only filings on or before this date are returned."
        ),
    )

    @field_validator("form_types", mode="before")
    @classmethod
    def _validate_form_types(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [f.strip().upper() for f in v.split(",") if f.strip()]
        if isinstance(v, (list, tuple)):
            return [str(f).strip().upper() for f in v if str(f).strip()]
        return ["10-K", "10-Q", "8-K"]

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _validate_as_of_date(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, (date, datetime)):
            return v.strftime("%Y-%m-%d")
        return str(v).strip()


class ListFilingsOutput(BaseModel):
    """Output payload for list_filings tool."""

    model_config = ConfigDict(extra="allow")

    ticker: str = Field(..., description="Stock ticker symbol queried")
    cik: str = Field(..., description="Zero-padded 10-digit CIK")
    company_name: str | None = Field(
        default=None, description="Official company legal name"
    )
    as_of_date: str | None = Field(
        default=None, description="Point-in-time cutoff applied"
    )
    total_matches: int = Field(
        default=0, description="Total matching filings found after filtering"
    )
    filings: list[FilingItem] = Field(
        default_factory=list, description="List of matching SEC filings"
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __len__(self) -> int:
        return len(self.filings)

    def __iter__(self) -> Any:
        return iter(self.filings)


class ListFilingsTool(Tool[ListFilingsInput, ListFilingsOutput]):
    """Tool that retrieves official SEC filings with point-in-time filtering."""

    name: str = "list_filings"
    description: str = (
        "Retrieve official SEC EDGAR filings (e.g. 10-K, 10-Q, 8-K). "
        "Supports point-in-time filtering (as_of_date) to enforce zero lookahead bias. "
        "Returns form, filing date, report date, accession number, and document URL."
    )
    input_model: type[ListFilingsInput] = ListFilingsInput
    output_model: type[ListFilingsOutput] = ListFilingsOutput

    def __init__(self, sec_client: SecClient | None = None) -> None:
        self.sec_client: SecClient = sec_client or SecClient()

    async def run(self, input_data: ListFilingsInput) -> ListFilingsOutput:
        """Execute filing discovery with form type and point-in-time filtering."""
        ticker = input_data.ticker.strip().upper()

        # 1. Resolve ticker to 10-digit CIK
        cik = await self.sec_client.get_cik(ticker)

        # 2. Fetch submission history from SEC EDGAR
        submissions = await self.sec_client.get_submissions(cik)
        company_name = submissions.get("name")

        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        target_forms = {f.strip().upper() for f in input_data.form_types}
        as_of = input_data.as_of_date.strip() if input_data.as_of_date else None

        matched_items: list[FilingItem] = []
        n_rows = len(forms)

        for i in range(n_rows):
            form_val = str(forms[i]).strip().upper()
            if target_forms and form_val not in target_forms:
                continue

            filing_date_val = str(filing_dates[i]).strip()
            # Enforce point-in-time filter: filingDate <= as_of_date
            if as_of is not None and filing_date_val > as_of:
                continue

            report_date_val = (
                str(report_dates[i]).strip()
                if i < len(report_dates) and report_dates[i]
                else None
            )
            accession_val = str(accessions[i]).strip()
            doc_val = str(primary_docs[i]).strip() if i < len(primary_docs) else ""

            doc_url = SecClient.build_document_url(
                cik=cik,
                accession_number=accession_val,
                primary_document=doc_val,
            )

            item = FilingItem(
                form=form_val,
                filing_date=filing_date_val,
                report_date=report_date_val,
                accession_number=accession_val,
                primary_document_url=doc_url,
                primary_document=doc_val,
            )
            matched_items.append(item)

        # Ensure sorted by filing_date descending (newest first)
        matched_items.sort(key=lambda x: x.filing_date, reverse=True)

        sliced_items = matched_items[: input_data.limit]

        return ListFilingsOutput(
            ticker=ticker,
            cik=cik,
            company_name=company_name,
            as_of_date=input_data.as_of_date,
            total_matches=len(matched_items),
            filings=sliced_items,
        )


FilingsTool = ListFilingsTool

# Register default instance in tool registry
default_filings_tool = ListFilingsTool()
default_registry.register(default_filings_tool, overwrite=True)
