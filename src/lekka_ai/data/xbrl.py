"""XBRL company facts parser with tag fallbacks and fiscal period extraction."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, overload

from pydantic import BaseModel, ConfigDict, Field

from lekka_ai.logging import get_logger

logger = get_logger(__name__)

# Standard tag fallbacks mapping canonical concepts to lists of candidate XBRL tags
DEFAULT_TAG_FALLBACKS: dict[str, list[str]] = {
    "revenues": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
    ],
    "cash_and_cash_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsAtCarryingValue",
        "CashAndCashEquivalents",
    ],
    "debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "CommercialPaper",
        "DebtCurrent",
        "ShortTermBorrowings",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "short_term_debt": [
        "LongTermDebtCurrent",
        "CommercialPaper",
        "ShortTermBorrowings",
        "DebtCurrent",
    ],
}


class PeriodFinancials(BaseModel):
    """Normalized financial statements facts for a single fiscal period."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    fy: int = Field(..., description="Fiscal year (e.g. 2023)")
    fp: str = Field(
        ..., description="Fiscal period (e.g. 'FY', 'Q1', 'Q2', 'Q3', 'Q4')"
    )
    form: str | None = Field(
        default=None, description="Primary filing form (e.g. '10-K', '10-Q')"
    )
    filed: str | None = Field(
        default=None, description="Filing submission date (YYYY-MM-DD)"
    )
    end: str | None = Field(
        default=None, description="Fiscal period end date (YYYY-MM-DD)"
    )

    # Core financial statement metrics
    revenues: float | int | None = Field(
        default=None, description="Total revenues or net sales"
    )
    net_income: float | int | None = Field(
        default=None, description="Net income or loss"
    )
    operating_income: float | int | None = Field(
        default=None, description="Operating income or loss"
    )
    operating_cash_flow: float | int | None = Field(
        default=None, description="Cash flow from operating activities"
    )
    capex: float | int | None = Field(
        default=None, description="Capital expenditures (PPE purchases)"
    )
    cash_and_cash_equivalents: float | int | None = Field(
        default=None, description="Cash and cash equivalents"
    )
    debt: float | int | None = Field(
        default=None, description="Total debt or primary debt tags"
    )

    # Provenance metadata: maps canonical metric to the actual XBRL tag resolved
    tag_sources: dict[str, str] = Field(
        default_factory=dict, description="Resolved XBRL tag name per metric"
    )

    @property
    def period(self) -> str:
        return f"{self.fy}-{self.fp}"

    @property
    def filed_date(self) -> str | None:
        return self.filed

    @property
    def end_date(self) -> str | None:
        return self.end

    @property
    def revenue(self) -> float | int | None:
        return self.revenues

    @property
    def cash(self) -> float | int | None:
        return self.cash_and_cash_equivalents

    # Concept name aliases matching prompt specifications
    @property
    def Revenues(self) -> float | int | None:
        return self.revenues

    @property
    def NetIncomeLoss(self) -> float | int | None:
        return self.net_income

    @property
    def OperatingIncomeLoss(self) -> float | int | None:
        return self.operating_income

    @property
    def NetCashProvidedByOperatingActivities(self) -> float | int | None:
        return self.operating_cash_flow

    @property
    def PaymentsToAcquirePropertyPlantAndEquipment(self) -> float | int | None:
        return self.capex

    @property
    def CashAndCashEquivalents(self) -> float | int | None:
        return self.cash_and_cash_equivalents

    def __getitem__(self, item: str) -> Any:
        norm = item.strip().lower()
        key_map = {
            "revenues": self.revenues,
            "revenue": self.revenues,
            "netincomeloss": self.net_income,
            "net_income": self.net_income,
            "operatingincomeloss": self.operating_income,
            "operating_income": self.operating_income,
            "netcashprovidedbyoperatingactivities": self.operating_cash_flow,
            "operating_cash_flow": self.operating_cash_flow,
            "paymentstoacquirepropertyplantandequipment": self.capex,
            "capex": self.capex,
            "cashandcashequivalents": self.cash_and_cash_equivalents,
            "cash_and_cash_equivalents": self.cash_and_cash_equivalents,
            "cash": self.cash_and_cash_equivalents,
            "debt": self.debt,
            "filed": self.filed,
            "filed_date": self.filed,
            "end": self.end,
            "end_date": self.end,
            "fy": self.fy,
            "fp": self.fp,
            "form": self.form,
            "tag_sources": self.tag_sources,
        }
        if norm in key_map:
            return key_map[norm]
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"Metric '{item}' not found on PeriodFinancials.")


class CompanyFactsResult(list[PeriodFinancials]):
    """Ordered container of parsed period financials with multi-index lookup."""

    def __init__(self, periods: Sequence[PeriodFinancials] = ()) -> None:
        super().__init__(periods)
        self._by_period: dict[tuple[int, str], PeriodFinancials] = {
            (p.fy, p.fp.upper()): p for p in self
        }

    @property
    def by_period(self) -> dict[tuple[int, str], PeriodFinancials]:
        return dict(self._by_period)

    def annual(self) -> list[PeriodFinancials]:
        """Return all annual (FY) period records sorted chronologically."""
        return [p for p in self if p.fp.upper() == "FY"]

    def quarterly(self) -> list[PeriodFinancials]:
        """Return all quarterly period records (Q1..Q4) sorted chronologically."""
        return [p for p in self if p.fp.upper() in ("Q1", "Q2", "Q3", "Q4")]

    def get(
        self,
        key: int | str | tuple[int, str],
        default: PeriodFinancials | None = None,
    ) -> PeriodFinancials | None:
        """Lookup period by (fy, fp) tuple, integer fy, or string 'YYYY-FP'."""
        try:
            return self._lookup(key)
        except KeyError:
            return default

    def _lookup(self, key: int | str | tuple[int, str]) -> PeriodFinancials:
        if isinstance(key, tuple) and len(key) == 2:
            fy, fp = int(key[0]), str(key[1]).strip().upper()
            if (fy, fp) in self._by_period:
                return self._by_period[(fy, fp)]
        elif isinstance(key, int):
            if (key, "FY") in self._by_period:
                return self._by_period[(key, "FY")]
        elif isinstance(key, str):
            clean = key.strip().upper()
            if "-" in clean:
                parts = clean.split("-", 1)
                if parts[0].isdigit():
                    fy, fp = int(parts[0]), parts[1]
                    if (fy, fp) in self._by_period:
                        return self._by_period[(fy, fp)]
            elif clean.isdigit():
                fy = int(clean)
                if (fy, "FY") in self._by_period:
                    return self._by_period[(fy, "FY")]
        raise KeyError(f"No financial facts found for period key '{key}'.")

    @overload  # type: ignore[override]
    def __getitem__(self, index: int) -> PeriodFinancials: ...

    @overload
    def __getitem__(self, index: slice) -> list[PeriodFinancials]: ...

    @overload
    def __getitem__(self, index: tuple[int, str] | str) -> PeriodFinancials: ...

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, (int, slice)):
            # If standard list index in bounds
            if isinstance(index, int) and (index < 0 or index < len(self)):
                return super().__getitem__(index)
            # If integer fiscal year (e.g. 2023)
            if isinstance(index, int) and (index, "FY") in self._by_period:
                return self._by_period[(index, "FY")]
            return super().__getitem__(index)
        return self._lookup(index)


class CompanyFactsParser:
    """Parser for SEC EDGAR XBRL company facts JSON responses."""

    def __init__(
        self,
        tag_fallbacks: dict[str, list[str]] | None = None,
    ) -> None:
        self.tag_fallbacks = (
            dict(tag_fallbacks) if tag_fallbacks is not None else DEFAULT_TAG_FALLBACKS
        )

    def parse(
        self,
        raw_facts: dict[str, Any],
        as_of_date: str | None = None,
    ) -> CompanyFactsResult:
        """Parse raw SEC EDGAR facts payload into normalized period financials.

        Args:
            raw_facts: Dictionary response from SEC EDGAR companyfacts endpoint.
            as_of_date: Point-in-time cutoff date (YYYY-MM-DD). Only facts filed
                on or before this date will be considered.

        Returns:
            CompanyFactsResult containing normalized PeriodFinancials.
        """
        # 1. Extract GAAP taxonomy
        facts_dict = raw_facts.get("facts", {})
        gaap = facts_dict.get("us-gaap") or raw_facts.get("us-gaap") or {}
        if not isinstance(gaap, dict):
            logger.warning("No us-gaap taxonomy found in company facts payload.")
            return CompanyFactsResult([])

        cutoff = as_of_date.strip() if as_of_date else None

        # 2. Discover all available periods (fy, fp)
        discovered_periods: set[tuple[int, str]] = set()
        for concept_data in gaap.values():
            if not isinstance(concept_data, dict):
                continue
            units_dict = concept_data.get("units", {})
            for unit_list in units_dict.values():
                if not isinstance(unit_list, list):
                    continue
                for obs in unit_list:
                    if not isinstance(obs, dict):
                        continue
                    fy = obs.get("fy")
                    fp = obs.get("fp")
                    filed = obs.get("filed")

                    if fy is None or fp is None:
                        continue
                    if cutoff is not None and filed and filed > cutoff:
                        continue

                    fp_str = str(fp).strip().upper()
                    if fp_str in ("FY", "Q1", "Q2", "Q3", "Q4"):
                        try:
                            discovered_periods.add((int(fy), fp_str))
                        except (ValueError, TypeError):
                            continue

        # Sort periods chronologically
        fp_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 5}
        sorted_periods = sorted(
            discovered_periods, key=lambda p: (p[0], fp_order.get(p[1], 0))
        )

        # 3. Extract financial facts per period
        periods: list[PeriodFinancials] = []
        for fy, fp in sorted_periods:
            period_record = self._extract_period_financials(
                gaap=gaap,
                fy=fy,
                fp=fp,
                as_of_date=cutoff,
            )
            periods.append(period_record)

        return CompanyFactsResult(periods)

    def _extract_period_financials(
        self,
        gaap: dict[str, Any],
        fy: int,
        fp: str,
        as_of_date: str | None,
    ) -> PeriodFinancials:
        """Extract canonical metrics for a given (fy, fp) using fallback hierarchy."""
        form_expected = "10-K" if fp == "FY" else "10-Q"

        extracted_values: dict[str, Any] = {}
        tag_sources: dict[str, str] = {}
        primary_filed: str | None = None
        primary_end: str | None = None
        primary_form: str | None = None

        for metric_name, candidate_tags in self.tag_fallbacks.items():
            for tag in candidate_tags:
                if tag not in gaap or not isinstance(gaap[tag], dict):
                    continue

                concept = gaap[tag]
                units_dict = concept.get("units", {})
                # Look in USD units first, or fallback to first available currency unit
                raw_units = (
                    units_dict.get("USD")
                    or units_dict.get("usd")
                    or (next(iter(units_dict.values())) if units_dict else [])
                )
                unit_list: list[Any] = raw_units if isinstance(raw_units, list) else []
                if not unit_list:
                    continue

                # Filter observations for target fiscal period and point-in-time cutoff
                matching = [
                    u
                    for u in unit_list
                    if isinstance(u, dict)
                    and u.get("fy") == fy
                    and str(u.get("fp", "")).strip().upper() == fp
                    and (
                        as_of_date is None
                        or not u.get("filed")
                        or u["filed"] <= as_of_date
                    )
                ]
                if not matching:
                    continue

                # Prefer observations from canonical form (10-K for FY, 10-Q)
                form_matching = [
                    m
                    for m in matching
                    if form_expected in str(m.get("form", "")).upper()
                ] or matching

                # Find observation representing current period (max end date in filing)
                max_end = max(str(m.get("end", "")) for m in form_matching)
                at_max_end = [
                    m for m in form_matching if str(m.get("end", "")) == max_end
                ]

                # Check if duration facts (have start and end)
                duration_entries: list[tuple[int, dict[str, Any]]] = []
                for m in at_max_end:
                    start_str = m.get("start")
                    end_str = m.get("end")
                    if start_str and end_str:
                        try:
                            d1 = date.fromisoformat(start_str)
                            d2 = date.fromisoformat(end_str)
                            duration_entries.append(((d2 - d1).days, m))
                        except (ValueError, TypeError):
                            pass

                chosen: dict[str, Any] | None = None
                if duration_entries:
                    if fp == "FY":
                        # Full-year duration (~365 days)
                        duration_entries.sort(key=lambda x: x[0], reverse=True)
                        chosen = duration_entries[0][1]
                    else:
                        # Quarter duration (~90 days)
                        duration_entries.sort(key=lambda x: abs(x[0] - 90))
                        chosen = duration_entries[0][1]
                else:
                    # Instantaneous facts (balance sheet): pick latest filed date
                    at_max_end.sort(key=lambda x: str(x.get("filed", "")), reverse=True)
                    chosen = at_max_end[0]

                if chosen is not None and "val" in chosen:
                    extracted_values[metric_name] = chosen["val"]
                    tag_sources[metric_name] = tag

                    filed_val = chosen.get("filed")
                    if filed_val and (not primary_filed or filed_val > primary_filed):
                        primary_filed = str(filed_val)

                    end_val = chosen.get("end")
                    if end_val and (not primary_end or end_val > primary_end):
                        primary_end = str(end_val)

                    form_val = chosen.get("form")
                    if form_val and not primary_form:
                        primary_form = str(form_val)

                    # Successful resolution for this metric; do not check next fallbacks
                    break

        return PeriodFinancials(
            fy=fy,
            fp=fp,
            form=primary_form,
            filed=primary_filed,
            end=primary_end,
            revenues=extracted_values.get("revenues"),
            net_income=extracted_values.get("net_income"),
            operating_income=extracted_values.get("operating_income"),
            operating_cash_flow=extracted_values.get("operating_cash_flow"),
            capex=extracted_values.get("capex"),
            cash_and_cash_equivalents=extracted_values.get("cash_and_cash_equivalents"),
            debt=extracted_values.get("debt"),
            tag_sources=tag_sources,
        )


def parse_company_facts(
    raw_facts: dict[str, Any],
    tag_fallbacks: dict[str, list[str]] | None = None,
    as_of_date: str | None = None,
) -> CompanyFactsResult:
    """Parse raw SEC EDGAR facts payload into normalized period financials.

    Args:
        raw_facts: JSON dictionary from SEC EDGAR company facts endpoint.
        tag_fallbacks: Optional mapping overriding concept tag fallbacks.
        as_of_date: Point-in-time cutoff date (YYYY-MM-DD).

    Returns:
        CompanyFactsResult with extracted financial facts per period.
    """
    parser = CompanyFactsParser(tag_fallbacks=tag_fallbacks)
    return parser.parse(raw_facts, as_of_date=as_of_date)
