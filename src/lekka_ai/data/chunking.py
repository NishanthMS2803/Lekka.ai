"""Text chunking for SEC filings with metadata preservation."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from lekka_ai.logging import get_logger

logger = get_logger(__name__)

# Default chunking parameters (tokens/words)
DEFAULT_CHUNK_SIZE: int = 800
DEFAULT_CHUNK_OVERLAP: int = 100


def count_tokens(text: str) -> int:
    """Estimate token count for a text string using whitespace word tokenization.

    Args:
        text: Plain text string.

    Returns:
        Integer token/word count.
    """
    if not text:
        return 0
    return len(text.split())


@dataclass
class FilingChunkData:
    """Container for a text chunk with filing and section metadata."""

    ticker: str
    form: str
    filing_date: date
    accession: str
    section_name: str
    chunk_index: int
    text: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    token_counter: Callable[[str], int] | None = None,
) -> list[str]:
    """Split text into chunks of target token size with specified overlap.

    Args:
        text: Raw or cleaned text string to split.
        chunk_size: Target maximum tokens per chunk (default: 800).
        chunk_overlap: Overlapping tokens between adjacent chunks (default: 100).
        token_counter: Optional custom token counting function.

    Returns:
        List of chunk strings.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    words = clean_text.split()
    total_words = len(words)
    if total_words == 0:
        return []

    if total_words <= chunk_size:
        return [clean_text]

    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size

    chunks: list[str] = []
    i = 0
    while i < total_words:
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= total_words:
            break
        i += step

    return chunks


def chunk_filing_section(
    *,
    ticker: str,
    form: str,
    filing_date: date,
    accession: str,
    section_name: str,
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    token_counter: Callable[[str], int] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[FilingChunkData]:
    """Chunk a specific filing section while preserving filing and section metadata.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        form: Filing form type (e.g. '10-K', '10-Q').
        filing_date: SEC filing date.
        accession: SEC EDGAR accession number.
        section_name: Item section heading (e.g. 'Item 1', 'Item 7').
        text: Narrative section body text.
        chunk_size: Target maximum tokens per chunk (default: 800).
        chunk_overlap: Overlapping tokens between chunks (default: 100).
        token_counter: Optional custom token counting function.
        extra_metadata: Additional metadata key-values to attach to each chunk.

    Returns:
        List of FilingChunkData objects with metadata and sequential chunk indices.
    """
    raw_chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        token_counter=token_counter,
    )
    if not raw_chunks:
        return []

    count_fn = token_counter or count_tokens
    norm_ticker = ticker.strip().upper()
    norm_form = form.strip().upper()
    norm_acc = accession.strip()
    norm_sec = section_name.strip()
    base_meta = dict(extra_metadata or {})

    chunks_data: list[FilingChunkData] = []
    for idx, chunk_str in enumerate(raw_chunks):
        t_count = count_fn(chunk_str)
        chunks_data.append(
            FilingChunkData(
                ticker=norm_ticker,
                form=norm_form,
                filing_date=filing_date,
                accession=norm_acc,
                section_name=norm_sec,
                chunk_index=idx,
                text=chunk_str,
                token_count=t_count,
                metadata={
                    **base_meta,
                    "ticker": norm_ticker,
                    "form": norm_form,
                    "filing_date": str(filing_date),
                    "accession": norm_acc,
                    "section_name": norm_sec,
                    "chunk_index": idx,
                    "token_count": t_count,
                },
            )
        )

    logger.debug(
        "Chunked section into segments",
        ticker=norm_ticker,
        accession=norm_acc,
        section=norm_sec,
        chunk_count=len(chunks_data),
    )
    return chunks_data


def chunk_filing_document(
    *,
    ticker: str,
    form: str,
    filing_date: date,
    accession: str,
    sections: dict[str, str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    token_counter: Callable[[str], int] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[FilingChunkData]:
    """Chunk all sections of a filing document, preserving metadata per chunk.

    Args:
        ticker: Stock ticker symbol.
        form: Filing form type.
        filing_date: SEC filing date.
        accession: SEC EDGAR accession number.
        sections: Dictionary mapping section names (e.g. 'Item 1') to body text.
        chunk_size: Target maximum tokens per chunk.
        chunk_overlap: Overlapping tokens between chunks.
        token_counter: Optional custom token counting function.
        extra_metadata: Optional additional metadata dict.

    Returns:
        Consolidated list of FilingChunkData objects across all sections.
    """
    all_chunks: list[FilingChunkData] = []
    for sec_name, sec_text in sections.items():
        section_chunks = chunk_filing_section(
            ticker=ticker,
            form=form,
            filing_date=filing_date,
            accession=accession,
            section_name=sec_name,
            text=sec_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            token_counter=token_counter,
            extra_metadata=extra_metadata,
        )
        all_chunks.extend(section_chunks)

    return all_chunks
