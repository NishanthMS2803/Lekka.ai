"""Tests for SEC filing document cleaning, section splitting, and DB persistence."""

import contextlib
from datetime import date
from pathlib import Path

import httpx
import pytest

from lekka_ai.data.filings import (
    clean_html_to_text,
    get_section,
    parse_filing_document,
    split_filing_sections,
)
from lekka_ai.data.http import create_http_client
from lekka_ai.data.sec import SecClient
from lekka_ai.db import (
    Filing,
    get_filing_section,
    get_filing_sections,
    get_session,
    save_filing_document,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_10k.html"


@pytest.fixture
def sample_10k_html() -> str:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return f.read()


def test_clean_html_to_text(sample_10k_html: str) -> None:
    """clean_html_to_text strips non-content tags and normalizes whitespace."""
    cleaned = clean_html_to_text(sample_10k_html)

    # Must contain readable text
    assert "UNITED STATES SECURITIES AND EXCHANGE COMMISSION" in cleaned
    assert "Acme Corporation is a leading provider" in cleaned
    assert "Interest Rate Sensitivity" in cleaned

    # Must not contain raw html tags
    assert "<html>" not in cleaned
    assert "</div>" not in cleaned
    assert "<a href=" not in cleaned
    assert "&nbsp;" not in cleaned

    # Empty string handling
    assert clean_html_to_text("") == ""
    assert clean_html_to_text("   ") == ""


def test_split_filing_sections_expected_items(sample_10k_html: str) -> None:
    """Verify fixture splits into expected 10-K sections without TOC interference."""
    cleaned_text, sections = parse_filing_document(sample_10k_html)

    # Core sections present in fixture
    assert "Item 1" in sections
    assert "Item 1A" in sections
    assert "Item 2" in sections
    assert "Item 7" in sections
    assert "Item 7A" in sections
    assert "Item 8" in sections

    # Item 1 content check: contains body text, not Table of Contents page numbers
    sec_1 = sections["Item 1"]
    assert "Acme Corporation is a leading provider" in sec_1
    assert "45,000 full-time employees" in sec_1
    assert "Page 1" not in sec_1

    # Item 1A content check: Risk factors
    sec_1a = sections["Item 1A"]
    assert "Adverse macroeconomic conditions" in sec_1a
    assert "Failures in cybersecurity" in sec_1a
    assert "Page 8" not in sec_1a

    # Item 7 content check: MD&A
    sec_7 = sections["Item 7"]
    assert "Total revenue grew 14% year-over-year to $45.2 billion" in sec_7
    assert "free cash flow of $11.8 billion" in sec_7
    assert "Page 25" not in sec_7

    # Item 7A content check: Market Risk
    sec_7a = sections["Item 7A"]
    assert "Interest Rate Sensitivity" in sec_7a
    assert "100-basis-point increase" in sec_7a

    # Item 8 content check: Financial Statements
    sec_8 = sections["Item 8"]
    assert "consolidated balance sheets" in sec_8
    assert "Report of Independent Registered Public Accounting Firm" in sec_8


def test_handle_missing_sections_gracefully(sample_10k_html: str) -> None:
    """Missing sections return None without raising exceptions."""
    cleaned = clean_html_to_text(sample_10k_html)
    sections = split_filing_sections(cleaned)

    # Sections absent from fixture
    assert sections.get("Item 1B") is None
    assert sections.get("Item 3") is None
    assert sections.get("Item 9") is None
    assert sections.get("Item 9A") is None
    assert sections.get("Item 15") is None

    # get_section helper with normalization
    assert get_section(sections, "Item 1B") is None
    assert get_section(sections, "item 3") is None
    assert get_section(sections, "9A") is None

    # Normalized lookup for present sections
    assert get_section(sections, "item 1a") is not None
    assert get_section(sections, "ITEM 7") is not None
    assert get_section(sections, "7a") is not None
    assert get_section(sections, "Item 8") is not None


@pytest.mark.asyncio
async def test_sec_client_get_filing_document_mocked() -> None:
    """SecClient fetches filing document HTML using constructed URL."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, text="<html><body>Filing Content</body></html>")

    transport = httpx.MockTransport(handler)
    http_client = create_http_client(transport=transport)
    sec_client = SecClient(http_client=http_client)

    doc_text = await sec_client.get_filing_document(
        cik="0000320193",
        accession_number="0000320193-23-000106",
        primary_document="aapl-20230930.htm",
    )

    assert "Filing Content" in doc_text
    assert len(captured) == 1
    req = captured[0]
    expected_url = (
        "https://www.sec.gov/Archives/edgar/data/"
        "320193/000032019323000106/aapl-20230930.htm"
    )
    assert str(req.url) == expected_url


def test_filings_table_db_persistence(sample_10k_html: str) -> None:
    """Store raw and cleaned text in filings table and query back sections."""
    from sqlalchemy import text

    from lekka_ai.db.session import get_engine

    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL database is unreachable; skipping DB test.")

    cleaned_text, sections = parse_filing_document(sample_10k_html)

    import uuid

    test_ticker = "ACME"
    test_form = "10-K"
    test_filing_date = date(2023, 11, 15)
    test_accession = f"0001234567-23-{uuid.uuid4().hex[:8]}"

    with contextlib.contextmanager(get_session)() as session:
        try:
            # Save sections in DB
            records = save_filing_document(
                session=session,
                ticker=test_ticker,
                form=test_form,
                filing_date=test_filing_date,
                accession=test_accession,
                sections=sections,
                raw_text=sample_10k_html[:500],  # sample raw snippet
            )
            assert len(records) == len(sections)

            # Query back single section
            item_1 = get_filing_section(
                session=session,
                ticker=test_ticker,
                accession=test_accession,
                section_name="Item 1",
            )
            assert item_1 is not None
            assert item_1.ticker == test_ticker
            assert item_1.form == test_form
            assert item_1.accession == test_accession
            assert "Acme Corporation is a leading provider" in item_1.text

            # Query all sections
            all_secs = get_filing_sections(
                session=session,
                ticker=test_ticker,
                accession=test_accession,
            )
            assert len(all_secs) == len(sections)
            sec_names = [s.section_name for s in all_secs]
            assert "Item 1" in sec_names
            assert "Item 1A" in sec_names
            assert "Item 7" in sec_names
        finally:
            from sqlalchemy import delete

            session.execute(delete(Filing).where(Filing.accession == test_accession))
            session.commit()
