"""Fact cache (lite memory) built on top of the ledger with TTL policies."""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from lekka_ai.db.models import ClaimModel, FactCache
from lekka_ai.db.session import get_session_factory
from lekka_ai.logging import get_logger
from lekka_ai.schemas.claims import Claim
from lekka_ai.verify.conflicts import (
    extract_claim_metric_and_value,
    normalize_metric,
)

logger = get_logger(__name__)

# TTL: price 5m, news 6h, estimates 24h, quarterly 90d, filings never
FACT_TTL_POLICIES: dict[str, timedelta | None] = {
    "price": timedelta(minutes=5),
    "market_data": timedelta(minutes=5),
    "news": timedelta(hours=6),
    "estimates": timedelta(hours=24),
    "analyst_estimates": timedelta(hours=24),
    "quarterly financials": timedelta(days=90),
    "quarterly_financials": timedelta(days=90),
    "financials": timedelta(days=90),
    "filings": None,
    "filing": None,
    "sec_filing": None,
    "filing_doc": None,
}


def get_ttl_for_fact_type(fact_type: str) -> timedelta | None:
    """Return timedelta TTL for a fact type, or None if it never expires."""
    normalized = fact_type.strip().lower()
    if normalized in FACT_TTL_POLICIES:
        return FACT_TTL_POLICIES[normalized]

    if "price" in normalized or "market" in normalized:
        return timedelta(minutes=5)
    if "news" in normalized:
        return timedelta(hours=6)
    if "estimate" in normalized:
        return timedelta(hours=24)
    if "financial" in normalized or "quarter" in normalized:
        return timedelta(days=90)
    if "filing" in normalized or "sec" in normalized:
        return None
    return timedelta(hours=24)


def compute_expires_at(
    fact_type: str,
    fetched_at: datetime | None = None,
) -> datetime | None:
    """Compute expiration timestamp based on fact_type TTL policy."""
    ttl = get_ttl_for_fact_type(fact_type)
    if ttl is None:
        return None
    base_time = fetched_at or datetime.now(UTC)
    if base_time.tzinfo is None:
        base_time = base_time.replace(tzinfo=UTC)
    return base_time + ttl


class RecalledFact(BaseModel):
    """An unexpired verified fact retrieved from facts_cache."""

    model_config = ConfigDict(extra="ignore")

    metric_key: str = Field(
        ...,
        description="Canonical metric identifier, e.g. 'revenue' or 'price'",
    )
    value: Any = Field(
        ...,
        description="Fact value (numeric, string, or structured object)",
    )
    source: str | None = Field(
        default=None,
        description="Source document or reference",
    )
    fact_type: str = Field(
        ...,
        description="Fact category (price, news, estimates, financials, filings)",
    )
    as_of_date: str | None = Field(
        default=None,
        description="Point-in-time cutoff date (YYYY-MM-DD)",
    )
    fetched_at: str | None = Field(
        default=None,
        description="ISO retrieval timestamp",
    )
    expires_at: str | None = Field(
        default=None,
        description="ISO expiration timestamp if applicable",
    )
    claim_id: str | None = Field(
        default=None,
        description="Linked claim identifier",
    )
    period: str | None = Field(
        default=None,
        description="Fiscal period if applicable",
    )
    ticker: str | None = Field(
        default=None,
        description="Ticker symbol",
    )


class RecallFactsOutput(BaseModel):
    """Output payload from recall_facts tool."""

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(..., description="Ticker symbol")
    unexpired_facts: list[RecalledFact] = Field(
        default_factory=list,
        description="List of unexpired verified facts for the requested ticker/metrics",
    )
    needs_revalidation: list[str] = Field(
        default_factory=list,
        description="List of metric keys that have expired or need fresh retrieval",
    )


class FactCacheRepository:
    """Repository managing verified facts in memory and database."""

    def __init__(self, session: Session | None = None) -> None:
        self.session = session
        self._in_memory: list[dict[str, Any]] = []

    def _resolve_session(
        self, explicit_session: Session | None
    ) -> tuple[Session | None, bool]:
        if explicit_session is not None:
            return explicit_session, False
        if self.session is not None:
            return self.session, False
        try:
            factory = get_session_factory()
            return factory(), True
        except Exception:
            return None, False

    def store_fact(
        self,
        ticker: str,
        metric_key: str,
        value: Any,
        fact_type: str,
        *,
        source: str | None = None,
        as_of_date: date | str | None = None,
        fetched_at: datetime | str | None = None,
        expires_at: datetime | str | None = None,
        claim_id: str | None = None,
        period: str | None = None,
        session: Session | None = None,
    ) -> dict[str, Any]:
        """Store a single verified fact into cache with TTL calculation."""
        ticker_clean = ticker.strip().upper()
        metric_clean = metric_key.strip()

        # Parse fetched_at
        fetched_dt: datetime
        if fetched_at is None:
            fetched_dt = datetime.now(UTC)
        elif isinstance(fetched_at, datetime):
            fetched_dt = (
                fetched_at.replace(tzinfo=UTC)
                if fetched_at.tzinfo is None
                else fetched_at
            )
        else:
            try:
                fetched_dt = datetime.fromisoformat(str(fetched_at).strip())
                if fetched_dt.tzinfo is None:
                    fetched_dt = fetched_dt.replace(tzinfo=UTC)
            except ValueError:
                fetched_dt = datetime.now(UTC)

        # Parse as_of_date
        as_of_d: date | None = None
        if as_of_date is not None:
            if isinstance(as_of_date, date) and not isinstance(as_of_date, datetime):
                as_of_d = as_of_date
            elif isinstance(as_of_date, datetime):
                as_of_d = as_of_date.date()
            else:
                try:
                    as_of_d = date.fromisoformat(str(as_of_date).strip()[:10])
                except ValueError:
                    as_of_d = None

        # Compute expiration
        exp_dt: datetime | None = None
        if expires_at is not None:
            if isinstance(expires_at, datetime):
                exp_dt = (
                    expires_at.replace(tzinfo=UTC)
                    if expires_at.tzinfo is None
                    else expires_at
                )
            else:
                try:
                    exp_dt = datetime.fromisoformat(str(expires_at).strip())
                    if exp_dt.tzinfo is None:
                        exp_dt = exp_dt.replace(tzinfo=UTC)
                except ValueError:
                    exp_dt = None
        else:
            exp_dt = compute_expires_at(fact_type, fetched_dt)

        record_id = uuid.uuid4()
        fact_dict: dict[str, Any] = {
            "id": record_id,
            "ticker": ticker_clean,
            "metric_key": metric_clean,
            "value": value,
            "source": source,
            "fact_type": fact_type,
            "as_of_date": as_of_d,
            "fetched_at": fetched_dt,
            "expires_at": exp_dt,
            "claim_id": claim_id,
            "period": period,
            "created_at": datetime.now(UTC),
        }

        # Keep in memory
        self._in_memory.append(fact_dict)

        # Persist to database if available
        sess, should_close = self._resolve_session(session)
        if sess is not None:
            try:
                model = FactCache(
                    id=record_id,
                    ticker=ticker_clean,
                    metric_key=metric_clean,
                    value=value,
                    source=source,
                    fact_type=fact_type,
                    as_of_date=as_of_d,
                    fetched_at=fetched_dt,
                    expires_at=exp_dt,
                    claim_id=claim_id,
                    period=period,
                )
                sess.add(model)
                sess.commit()
            except Exception as exc:
                sess.rollback()
                logger.warning(
                    "Failed to persist fact to database facts_cache",
                    error=str(exc),
                )
            finally:
                if should_close:
                    sess.close()

        return fact_dict

    def store_verified_claims(
        self,
        claims: Sequence[Claim | ClaimModel | dict[str, Any]],
        *,
        ticker: str | None = None,
        session: Session | None = None,
        default_fact_type: str | None = None,
        default_source: str | None = None,
    ) -> list[dict[str, Any]]:
        """Persist verified claims into the fact cache with calculated TTLs."""
        stored: list[dict[str, Any]] = []

        for item in claims:
            # Check if claim has active contradictions; if contradicted, skip
            contr = getattr(item, "contradicted_by", None)
            if isinstance(item, dict):
                contr = item.get("contradicted_by")
            if contr and len(contr) > 0:
                continue

            # Extract fields
            claim_id: str | None = None
            if isinstance(item, dict):
                claim_id = item.get("claim_id")
            else:
                claim_id = getattr(item, "claim_id", None)

            # Determine metric and value
            canonical_metric, canonical_period, norm_val = (
                extract_claim_metric_and_value(item)
            )

            metric_key = canonical_metric
            if not metric_key:
                if isinstance(item, dict):
                    metric_key = item.get("metric") or item.get("metric_key")
                else:
                    metric_key = getattr(item, "metric_key", None)
            if not metric_key:
                metric_key = normalize_metric(str(getattr(item, "text", "")))
            if not metric_key:
                metric_key = "general_fact"

            # Determine value
            val = norm_val
            if val is None:
                if isinstance(item, dict):
                    val = item.get("value") or item.get("text")
                else:
                    val = getattr(item, "text", None)

            # Evidence inspection
            evidence_list: list[Any] = []
            if isinstance(item, dict):
                evidence_list = list(item.get("evidence", []) or [])
            else:
                evidence_list = list(getattr(item, "evidence", []) or [])

            src = default_source
            as_of_val: str | None = None
            retrieved_at_val: str | None = None
            inferred_fact_type = default_fact_type

            if evidence_list:
                first_ev = evidence_list[0]
                ev_src_type = (
                    first_ev.get("source_type")
                    if isinstance(first_ev, dict)
                    else getattr(first_ev, "source_type", None)
                )
                ev_doc_id = (
                    first_ev.get("document_id")
                    if isinstance(first_ev, dict)
                    else getattr(first_ev, "document_id", None)
                )
                ev_as_of = (
                    first_ev.get("as_of_date")
                    if isinstance(first_ev, dict)
                    else getattr(first_ev, "as_of_date", None)
                )
                ev_retrieved = (
                    first_ev.get("retrieved_at")
                    if isinstance(first_ev, dict)
                    else getattr(first_ev, "retrieved_at", None)
                )

                if ev_doc_id:
                    src = ev_doc_id
                elif ev_src_type:
                    src = ev_src_type

                if ev_as_of:
                    as_of_val = str(ev_as_of)
                if ev_retrieved:
                    retrieved_at_val = str(ev_retrieved)

                if not inferred_fact_type and ev_src_type:
                    st_clean = ev_src_type.lower()
                    if "market" in st_clean or "price" in st_clean:
                        inferred_fact_type = "price"
                    elif "news" in st_clean:
                        inferred_fact_type = "news"
                    elif "estimate" in st_clean:
                        inferred_fact_type = "estimates"
                    elif "filing" in st_clean or "sec" in st_clean:
                        inferred_fact_type = "filings"
                    elif "financial" in st_clean or "xbrl" in st_clean:
                        inferred_fact_type = "quarterly financials"

            # Fallback fact_type inference
            if not inferred_fact_type:
                if metric_key in ("price", "market_cap", "volume"):
                    inferred_fact_type = "price"
                elif "filing" in metric_key:
                    inferred_fact_type = "filings"
                elif "news" in metric_key:
                    inferred_fact_type = "news"
                else:
                    inferred_fact_type = "quarterly financials"

            # Resolve ticker
            fact_ticker = ticker
            if not fact_ticker:
                # Try evidence doc id if it's ticker-shaped (e.g. 1-5 letters)
                if src and re.fullmatch(r"[A-Za-z]{1,5}", src):
                    fact_ticker = src
                else:
                    fact_ticker = "GENERAL"

            rec = self.store_fact(
                ticker=fact_ticker,
                metric_key=metric_key,
                value=val,
                fact_type=inferred_fact_type,
                source=src,
                as_of_date=as_of_val,
                fetched_at=retrieved_at_val,
                claim_id=claim_id,
                period=canonical_period,
                session=session,
            )
            stored.append(rec)

        return stored

    def recall_facts(
        self,
        ticker: str,
        metrics: Sequence[str] | None = None,
        *,
        session: Session | None = None,
        as_of_time: datetime | None = None,
    ) -> RecallFactsOutput:
        """Recall unexpired facts for ticker and detect expired metrics."""
        ticker_clean = ticker.strip().upper()
        req_metrics: set[str] | None = None
        if metrics is not None:
            req_metrics = {m.strip().lower() for m in metrics}

        now_dt = as_of_time or datetime.now(UTC)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=UTC)

        # Retrieve records from DB or memory
        records: list[dict[str, Any]] = []
        sess, should_close = self._resolve_session(session)

        if sess is not None:
            try:
                stmt = select(FactCache).where(FactCache.ticker == ticker_clean)
                db_results = (
                    sess.execute(stmt.order_by(FactCache.fetched_at.desc()))
                    .scalars()
                    .all()
                )
                for row in db_results:
                    records.append(
                        {
                            "ticker": row.ticker,
                            "metric_key": row.metric_key,
                            "value": row.value,
                            "source": row.source,
                            "fact_type": row.fact_type,
                            "as_of_date": (
                                row.as_of_date.isoformat() if row.as_of_date else None
                            ),
                            "fetched_at": (
                                row.fetched_at.isoformat() if row.fetched_at else None
                            ),
                            "expires_at": (
                                row.expires_at.isoformat() if row.expires_at else None
                            ),
                            "claim_id": row.claim_id,
                            "period": row.period,
                            "_expires_dt": row.expires_at,
                        }
                    )
            except Exception as exc:
                logger.warning(
                    "Database query for facts_cache failed; falling back to memory",
                    error=str(exc),
                )
                records = []
            finally:
                if should_close:
                    sess.close()

        # If DB records are empty, use memory
        if not records:
            for r in reversed(self._in_memory):
                if r.get("ticker", "").upper() == ticker_clean:
                    records.append(
                        {
                            "ticker": r.get("ticker"),
                            "metric_key": r.get("metric_key"),
                            "value": r.get("value"),
                            "source": r.get("source"),
                            "fact_type": r.get("fact_type"),
                            "as_of_date": (
                                r["as_of_date"].isoformat()
                                if isinstance(r.get("as_of_date"), (date, datetime))
                                else str(r.get("as_of_date") or "") or None
                            ),
                            "fetched_at": (
                                r["fetched_at"].isoformat()
                                if isinstance(r.get("fetched_at"), datetime)
                                else str(r.get("fetched_at") or "") or None
                            ),
                            "expires_at": (
                                r["expires_at"].isoformat()
                                if isinstance(r.get("expires_at"), datetime)
                                else str(r.get("expires_at") or "") or None
                            ),
                            "claim_id": r.get("claim_id"),
                            "period": r.get("period"),
                            "_expires_dt": r.get("expires_at"),
                        }
                    )

        # Filter by metrics if requested
        if req_metrics is not None:
            records = [
                r
                for r in records
                if r.get("metric_key", "").lower() in req_metrics
                or normalize_metric(r.get("metric_key", "")) in req_metrics
            ]

        # Group by metric_key to pick the latest record per metric
        latest_by_metric: dict[str, dict[str, Any]] = {}
        for r in records:
            m_key = r.get("metric_key", "")
            if m_key not in latest_by_metric:
                latest_by_metric[m_key] = r

        unexpired_facts: list[RecalledFact] = []
        needs_revalidation: list[str] = []

        for m_key, r in latest_by_metric.items():
            exp_dt = r.get("_expires_dt")
            if exp_dt is not None and isinstance(exp_dt, str):
                try:
                    exp_dt = datetime.fromisoformat(exp_dt)
                except ValueError:
                    exp_dt = None
            if exp_dt is not None and exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=UTC)

            # Check expiration rule
            if exp_dt is None:
                # Never expires (e.g. filings)
                unexpired_facts.append(
                    RecalledFact(
                        metric_key=m_key,
                        value=r.get("value"),
                        source=r.get("source"),
                        fact_type=r.get("fact_type", "filings"),
                        as_of_date=r.get("as_of_date"),
                        fetched_at=r.get("fetched_at"),
                        expires_at=r.get("expires_at"),
                        claim_id=r.get("claim_id"),
                        period=r.get("period"),
                        ticker=r.get("ticker"),
                    )
                )
            elif exp_dt > now_dt:
                # Unexpired
                unexpired_facts.append(
                    RecalledFact(
                        metric_key=m_key,
                        value=r.get("value"),
                        source=r.get("source"),
                        fact_type=r.get("fact_type", "general"),
                        as_of_date=r.get("as_of_date"),
                        fetched_at=r.get("fetched_at"),
                        expires_at=r.get("expires_at"),
                        claim_id=r.get("claim_id"),
                        period=r.get("period"),
                        ticker=r.get("ticker"),
                    )
                )
            else:
                # Expired -> DO NOT RETURN as valid fact; flag for revalidation
                if m_key not in needs_revalidation:
                    needs_revalidation.append(m_key)

        # Check for requested metrics that were not found or unexpired
        if metrics is not None:
            found_keys = {f.metric_key.lower() for f in unexpired_facts}
            for req in metrics:
                req_norm = req.strip().lower()
                if (
                    req_norm not in found_keys
                    and req not in needs_revalidation
                    and req_norm not in [n.lower() for n in needs_revalidation]
                ):
                    needs_revalidation.append(req)

        return RecallFactsOutput(
            ticker=ticker_clean,
            unexpired_facts=unexpired_facts,
            needs_revalidation=needs_revalidation,
        )

    def clear(self, ticker: str | None = None, session: Session | None = None) -> None:
        """Clear memory and optional database records for testing."""
        if ticker:
            t_upper = ticker.strip().upper()
            self._in_memory = [
                r for r in self._in_memory if r.get("ticker", "").upper() != t_upper
            ]
        else:
            self._in_memory.clear()

        sess, should_close = self._resolve_session(session)
        if sess is not None:
            try:
                from sqlalchemy import delete

                stmt = delete(FactCache)
                if ticker:
                    stmt = stmt.where(FactCache.ticker == ticker.strip().upper())
                sess.execute(stmt)
                sess.commit()
            except Exception as exc:
                sess.rollback()
                logger.warning("Failed to clear FactCache table", error=str(exc))
            finally:
                if should_close:
                    sess.close()


_default_fact_cache_repo: FactCacheRepository | None = None


def get_fact_cache_repository(
    session: Session | None = None,
) -> FactCacheRepository:
    """Return singleton or configured FactCacheRepository instance."""
    global _default_fact_cache_repo
    if session is not None:
        return FactCacheRepository(session=session)
    if _default_fact_cache_repo is None:
        _default_fact_cache_repo = FactCacheRepository()
    return _default_fact_cache_repo


def store_verified_claims_in_cache(
    claims: Sequence[Claim | ClaimModel | dict[str, Any]],
    *,
    ticker: str | None = None,
    session: Session | None = None,
    default_fact_type: str | None = None,
    default_source: str | None = None,
    repository: FactCacheRepository | None = None,
) -> list[dict[str, Any]]:
    """Store verified claims into the fact cache with fact_type TTL policies."""
    repo = repository or get_fact_cache_repository(session=session)
    return repo.store_verified_claims(
        claims,
        ticker=ticker,
        session=session,
        default_fact_type=default_fact_type,
        default_source=default_source,
    )


def get_recalled_facts(
    ticker: str,
    metrics: Sequence[str] | None = None,
    *,
    session: Session | None = None,
    repository: FactCacheRepository | None = None,
    as_of_time: datetime | None = None,
) -> RecallFactsOutput:
    """Retrieve unexpired facts and needs_revalidation list for a ticker."""
    repo = repository or get_fact_cache_repository(session=session)
    return repo.recall_facts(
        ticker=ticker,
        metrics=metrics,
        session=session,
        as_of_time=as_of_time,
    )
