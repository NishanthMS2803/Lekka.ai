"""Ingestion pipeline to fetch, clean, split, chunk, and embed SEC filings."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

from lekka_ai.data.chunking import FilingChunkData, chunk_filing_document
from lekka_ai.data.filings import parse_filing_document
from lekka_ai.data.http import create_http_client
from lekka_ai.data.sec import SecClient
from lekka_ai.db import get_session, save_filing_chunks, save_filing_document
from lekka_ai.llm.gemini import embed_texts
from lekka_ai.logging import get_logger

logger = get_logger(__name__)


async def ingest_filings_for_ticker(
    ticker: str,
    *,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    sec_client: SecClient | None = None,
    embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
) -> dict[str, Any]:
    """Fetch, split, chunk, and embed a company's latest filings.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        forms: SEC form types to ingest (defaults to ('10-K', '10-Q')).
        sec_client: Optional preconfigured SecClient instance.
        embed_fn: Optional embedding function (defaults to embed_texts).

    Returns:
        Summary dict containing ingestion statistics per form type.
    """
    clean_ticker = ticker.strip().upper()
    client = sec_client or SecClient(http_client=create_http_client())
    embedding_func = embed_fn or embed_texts

    logger.info("Starting filing ingestion pipeline", ticker=clean_ticker, forms=forms)

    # 1. Look up CIK
    cik = await client.lookup_cik(clean_ticker)
    logger.info("Resolved CIK for ticker", ticker=clean_ticker, cik=cik)

    # 2. Retrieve recent submissions
    submissions = await client.get_submissions(cik)
    recent = submissions.get("filings", {}).get("recent", {})

    form_list = recent.get("form", [])
    acc_list = recent.get("accessionNumber", [])
    date_list = recent.get("filingDate", [])
    doc_list = recent.get("primaryDocument", [])

    results_summary: dict[str, Any] = {
        "ticker": clean_ticker,
        "cik": cik,
        "forms": {},
    }

    # 3. For each target form type, locate the newest filing
    for target_form in forms:
        target_norm = target_form.strip().upper()
        found_idx: int | None = None
        for i, f_val in enumerate(form_list):
            if str(f_val).strip().upper() == target_norm:
                found_idx = i
                break

        if found_idx is None:
            logger.warning(
                "No filing found for form type",
                ticker=clean_ticker,
                form=target_norm,
            )
            continue

        accession = str(acc_list[found_idx]).strip()
        filing_date = date.fromisoformat(str(date_list[found_idx]).strip())
        primary_doc = str(doc_list[found_idx]).strip()

        logger.info(
            "Fetching filing document",
            ticker=clean_ticker,
            form=target_norm,
            accession=accession,
            doc=primary_doc,
        )

        # 4. Fetch HTML document
        html_content = await client.get_filing_document(
            cik=cik,
            accession_number=accession,
            primary_document=primary_doc,
        )

        # 5. Parse document and split into sections
        cleaned_text, sections = parse_filing_document(html_content)
        logger.info(
            "Parsed filing sections",
            ticker=clean_ticker,
            form=target_norm,
            section_count=len(sections),
        )

        # 6. Chunk sections
        chunks_data: list[FilingChunkData] = chunk_filing_document(
            ticker=clean_ticker,
            form=target_norm,
            filing_date=filing_date,
            accession=accession,
            sections=sections,
        )
        logger.info(
            "Generated text chunks",
            ticker=clean_ticker,
            form=target_norm,
            chunk_count=len(chunks_data),
        )

        # 7. Generate embeddings
        chunk_texts = [c.text for c in chunks_data]
        embeddings = await embedding_func(chunk_texts) if chunk_texts else []

        # 8. Store filing sections and chunks in database
        with contextlib.contextmanager(get_session)() as session:
            # Save filing sections
            saved_sections = save_filing_document(
                session=session,
                ticker=clean_ticker,
                form=target_norm,
                filing_date=filing_date,
                accession=accession,
                sections=sections,
                raw_text=html_content[:5000],  # preserve initial raw header
            )

            # Save chunks with embeddings
            saved_chunks = save_filing_chunks(
                session=session,
                chunks=chunks_data,
                embeddings=embeddings,
            )

        results_summary["forms"][target_norm] = {
            "accession": accession,
            "filing_date": filing_date.isoformat(),
            "sections_saved": len(saved_sections),
            "chunks_saved": len(saved_chunks),
        }

        logger.info(
            "Ingestion completed for form",
            ticker=clean_ticker,
            form=target_norm,
            sections_count=len(saved_sections),
            chunks_count=len(saved_chunks),
        )

    return results_summary


def main() -> None:
    """CLI entrypoint for filing ingestion command."""
    parser = argparse.ArgumentParser(
        description="Ingest, clean, split, chunk, and embed SEC filings for a ticker."
    )
    parser.add_argument(
        "--ticker",
        "-t",
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)",
    )
    args = parser.parse_args()

    summary = asyncio.run(ingest_filings_for_ticker(args.ticker))
    print(f"Successfully ingested filings for {summary['ticker']}:")
    for form_name, details in summary.get("forms", {}).items():
        print(
            f"  {form_name}: {details['sections_saved']} sections, "
            f"{details['chunks_saved']} chunks saved "
            f"(Accession: {details['accession']})"
        )


if __name__ == "__main__":
    main()
