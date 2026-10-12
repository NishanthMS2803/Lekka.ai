"""Tests for search_filings tool, citation objects, and filing ingestion."""

import contextlib
import uuid
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import delete, text

from lekka_ai.data.chunking import chunk_filing_document
from lekka_ai.data.filings import parse_filing_document
from lekka_ai.data.ingest import ingest_filings_for_ticker
from lekka_ai.db import (
    Filing,
    FilingChunk,
    get_engine,
    get_session,
    save_filing_chunks,
    save_filing_document,
)
from lekka_ai.tools.base import default_registry
from lekka_ai.tools.search_filings import (
    Citation,
    Passage,
    SearchFilingsInput,
    SearchFilingsTool,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_10k.html"


@pytest.fixture
def sample_10k_html() -> str:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return f.read()


def test_search_filings_tool_registered() -> None:
    """Verify search_filings is registered in the tool registry with correct schema."""
    tool = default_registry.get("search_filings")
    assert tool is not None
    assert tool.name == "search_filings"
    spec = tool.to_spec()
    assert spec.name == "search_filings"
    assert "query" in spec.json_schema["properties"]
    assert "ticker" in spec.json_schema["properties"]
    assert "as_of_date" in spec.json_schema["properties"]


def test_search_filings_input_validation() -> None:
    """Verify input normalization and validation."""
    inp = SearchFilingsInput(
        ticker="  aapl  ",
        query="  key risks  ",
        form_types="10-k, 10-q",  # type: ignore[arg-type]
        sections=["Item 1A"],
        as_of_date="2024-01-15",
        top_k=10,
    )
    assert inp.ticker == "AAPL"
    assert inp.query == "key risks"
    assert inp.form_types == ["10-K", "10-Q"]
    assert inp.sections == ["Item 1A"]
    assert inp.as_of_date == "2024-01-15"
    assert inp.top_k == 10

    # Empty query raises ValueError
    with pytest.raises(ValueError):
        SearchFilingsInput(ticker="AAPL", query="   ")


def test_citation_and_passage_models() -> None:
    """Verify Citation and Passage model construction and attributes."""
    cit = Citation(
        document_id="doc-123",
        accession="0001234567-24-000001",
        section="Item 1A",
        filing_date="2024-02-01",
        chunk_id="chunk-456",
    )
    assert cit.document_id == "doc-123"
    assert cit.accession == "0001234567-24-000001"
    assert cit.section == "Item 1A"
    assert cit.filing_date == "2024-02-01"
    assert cit.chunk_id == "chunk-456"

    pas = Passage(text="Risk factors text", score=0.92, citation=cit)
    assert pas.text == "Risk factors text"
    assert pas.score == 0.92
    assert pas.citation.section == "Item 1A"


@pytest.mark.asyncio
async def test_search_filings_risk_factors_returns_item_1a(
    sample_10k_html: str,
) -> None:
    """Verify a query about risk factors returns Item 1A passages with citations."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL database is unreachable; skipping DB test.")

    test_ticker = "ACME"
    test_form = "10-K"
    test_date = date(2023, 11, 15)
    test_accession = f"0001234567-23-{uuid.uuid4().hex[:8]}"

    # Parse fixture into sections
    cleaned_text, sections = parse_filing_document(sample_10k_html)
    assert "Item 1A" in sections
    assert "Item 7" in sections

    # Chunk sections
    chunks = chunk_filing_document(
        ticker=test_ticker,
        form=test_form,
        filing_date=test_date,
        accession=test_accession,
        sections=sections,
    )

    # Synthetic 768-dim embeddings:
    # Dimension 0 for Item 1A (Risk factors)
    # Dimension 1 for Item 7 (MD&A)
    # Dimension 2 for other items
    dim = 768
    fake_embeddings: list[list[float]] = []
    for c in chunks:
        vec = [0.0] * dim
        if c.section_name == "Item 1A":
            vec[0] = 1.0  # Dominant in dim 0
        elif c.section_name == "Item 7":
            vec[1] = 1.0  # Dominant in dim 1
        else:
            vec[2] = 1.0
        fake_embeddings.append(vec)

    # Mock embed function returning dim 0 dominant vector for risk queries
    async def fake_embed_fn(texts: list[str]) -> list[list[float]]:
        results = []
        for t in texts:
            vec = [0.0] * dim
            t_lower = t.lower()
            if "risk" in t_lower or "cybersecurity" in t_lower:
                vec[0] = 0.95
                vec[5] = 0.05
            elif "revenue" in t_lower or "financial" in t_lower:
                vec[1] = 0.95
                vec[5] = 0.05
            else:
                vec[2] = 0.95
            results.append(vec)
        return results

    tool = SearchFilingsTool(embed_fn=fake_embed_fn)

    with contextlib.contextmanager(get_session)() as session:
        try:
            # Ingest filing and chunks
            save_filing_document(
                session=session,
                ticker=test_ticker,
                form=test_form,
                filing_date=test_date,
                accession=test_accession,
                sections=sections,
            )
            save_filing_chunks(
                session=session,
                chunks=chunks,
                embeddings=fake_embeddings,
            )

            # 1. Query about risk factors
            output = await tool.run(
                SearchFilingsInput(
                    ticker=test_ticker,
                    query="What are primary risk factors and cybersecurity concerns?",
                    top_k=3,
                )
            )

            assert len(output.passages) > 0
            top_passage = output.passages[0]

            # Verify top passage is Item 1A
            assert top_passage.citation.section == "Item 1A"
            assert "Item 1A" in top_passage.citation.section
            assert "cybersecurity" in top_passage.text
            assert "semiconductor hardware availability" in top_passage.text
            assert top_passage.citation.accession == test_accession
            assert top_passage.citation.filing_date == "2023-11-15"
            assert top_passage.citation.chunk_id is not None
            assert top_passage.score > 0.8

            # 2. Point-in-time cutoff: as_of_date before filing returns no passages
            past_output = await tool.run(
                SearchFilingsInput(
                    ticker=test_ticker,
                    query="risk factors",
                    as_of_date="2023-01-01",
                    top_k=3,
                )
            )
            assert len(past_output.passages) == 0

            # 3. Test section filter: requesting Item 7 only returns Item 7 passages
            mda_output = await tool.run(
                SearchFilingsInput(
                    ticker=test_ticker,
                    query="risk factors",
                    sections=["Item 7"],
                    top_k=3,
                )
            )
            assert len(mda_output.passages) > 0
            for p in mda_output.passages:
                assert p.citation.section == "Item 7"

        finally:
            # Teardown
            session.execute(
                delete(FilingChunk).where(FilingChunk.accession == test_accession)
            )
            session.execute(delete(Filing).where(Filing.accession == test_accession))
            session.commit()


@pytest.mark.asyncio
async def test_ingest_filings_pipeline(sample_10k_html: str) -> None:
    """Verify ingest_filings_for_ticker fetches, splits, chunks, and embeds filings."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL database is unreachable; skipping DB test.")

    test_ticker = "INTEST"
    test_cik = "0009999888"
    acc_10k = f"0009999888-23-{uuid.uuid4().hex[:8]}"
    acc_10q = f"0009999888-24-{uuid.uuid4().hex[:8]}"

    # Mock SecClient
    mock_sec = MagicMock()
    mock_sec.lookup_cik = AsyncMock(return_value=test_cik)
    mock_sec.get_submissions = AsyncMock(
        return_value={
            "filings": {
                "recent": {
                    "form": ["10-Q", "8-K", "10-K"],
                    "accessionNumber": [acc_10q, "0009999888-24-000002", acc_10k],
                    "filingDate": ["2024-01-20", "2024-01-10", "2023-11-10"],
                    "primaryDocument": ["intest-10q.htm", "doc.htm", "intest-10k.htm"],
                }
            }
        }
    )
    mock_sec.get_filing_document = AsyncMock(return_value=sample_10k_html)

    # Mock embedding function
    async def mock_embed_fn(texts: list[str]) -> list[list[float]]:
        return [[0.05] * 768 for _ in texts]

    with contextlib.contextmanager(get_session)() as session:
        try:
            summary = await ingest_filings_for_ticker(
                ticker=test_ticker,
                forms=("10-K", "10-Q"),
                sec_client=mock_sec,
                embed_fn=mock_embed_fn,
            )

            assert summary["ticker"] == test_ticker
            assert summary["cik"] == test_cik
            assert "10-K" in summary["forms"]
            assert "10-Q" in summary["forms"]

            details_10k = summary["forms"]["10-K"]
            assert details_10k["accession"] == acc_10k
            assert details_10k["sections_saved"] > 0
            assert details_10k["chunks_saved"] > 0

            details_10q = summary["forms"]["10-Q"]
            assert details_10q["accession"] == acc_10q
            assert details_10q["sections_saved"] > 0
            assert details_10q["chunks_saved"] > 0

        finally:
            session.execute(
                delete(FilingChunk).where(FilingChunk.accession.in_([acc_10k, acc_10q]))
            )
            session.execute(
                delete(Filing).where(Filing.accession.in_([acc_10k, acc_10q]))
            )
            session.commit()
