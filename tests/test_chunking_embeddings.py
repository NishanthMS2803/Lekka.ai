"""Tests for filing chunking, Gemini embeddings, and pgvector persistence."""

import contextlib
import uuid
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import delete, text

from lekka_ai.data.chunking import (
    FilingChunkData,
    chunk_filing_document,
    chunk_filing_section,
    chunk_text,
    count_tokens,
)
from lekka_ai.db import (
    FilingChunk,
    find_similar_chunks,
    get_engine,
    get_filing_chunks,
    get_session,
    save_filing_chunks,
)
from lekka_ai.llm.gemini import embed_texts


def test_count_tokens() -> None:
    """Verify whitespace token count estimation."""
    assert count_tokens("") == 0
    assert count_tokens("   ") == 0
    assert count_tokens("Apple reports record quarterly revenue.") == 5


def test_chunk_text_boundaries_and_overlap() -> None:
    """Test chunk boundaries and exact token overlap on known sequence."""
    # Create synthetic text of 1500 words: 'word_0 word_1 ... word_1499'
    words = [f"w{i}" for i in range(1500)]
    text_content = " ".join(words)

    chunks = chunk_text(text_content, chunk_size=800, chunk_overlap=100)

    # 1500 words with 800 size and 700 step:
    # Chunk 0: 0 to 800 (800 words)
    # Chunk 1: 700 to 1500 (800 words)
    assert len(chunks) == 2

    chunk_0_words = chunks[0].split()
    chunk_1_words = chunks[1].split()

    assert len(chunk_0_words) == 800
    assert len(chunk_1_words) == 800

    # Overlap between chunk 0 end and chunk 1 start must be exactly 100 words
    # (w700 to w799)
    assert chunk_0_words[-100:] == chunk_1_words[:100]
    assert chunk_0_words[700] == "w700"
    assert chunk_1_words[0] == "w700"
    assert chunk_0_words[-1] == "w799"
    assert chunk_1_words[99] == "w799"


def test_chunk_text_shorter_than_limit() -> None:
    """Text shorter than chunk_size produces exactly one chunk."""
    text_content = "Short section content that fits easily."
    chunks = chunk_text(text_content, chunk_size=800, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == text_content


def test_chunk_text_empty_input() -> None:
    """Empty or whitespace text produces empty list."""
    assert chunk_text("") == []
    assert chunk_text("    \n\t  ") == []


def test_chunk_filing_section_preserves_metadata() -> None:
    """chunk_filing_section preserves all filing metadata across chunks."""
    words = [f"item1_word_{i}" for i in range(1200)]
    section_text = " ".join(words)

    filing_date = date(2024, 2, 1)
    chunks_data = chunk_filing_section(
        ticker="AAPL",
        form="10-K",
        filing_date=filing_date,
        accession="0000320193-24-000006",
        section_name="Item 1",
        text=section_text,
        chunk_size=500,
        chunk_overlap=50,
        extra_metadata={"source": "edgar"},
    )

    # 1200 words with step = 450:
    # Chunk 0: 0..500
    # Chunk 1: 450..950
    # Chunk 2: 900..1200 (300 words)
    assert len(chunks_data) == 3

    for idx, cd in enumerate(chunks_data):
        assert cd.ticker == "AAPL"
        assert cd.form == "10-K"
        assert cd.filing_date == filing_date
        assert cd.accession == "0000320193-24-000006"
        assert cd.section_name == "Item 1"
        assert cd.chunk_index == idx
        assert cd.token_count > 0
        assert cd.metadata["source"] == "edgar"
        assert cd.metadata["ticker"] == "AAPL"


def test_chunk_filing_document_multiple_sections() -> None:
    """chunk_filing_document processes all document sections."""
    sections = {
        "Item 1": "Business description and overview.",
        "Item 1A": "Risk factors regarding supply chain and global tariffs.",
        "Item 7": "Management discussion of operations and financial results.",
    }

    chunks = chunk_filing_document(
        ticker="MSFT",
        form="10-Q",
        filing_date=date(2024, 1, 25),
        accession="0000789019-24-000001",
        sections=sections,
    )

    assert len(chunks) == 3
    sec_names = [c.section_name for c in chunks]
    assert sec_names == ["Item 1", "Item 1A", "Item 7"]


@pytest.mark.asyncio
async def test_embed_texts_batching_and_retries() -> None:
    """embed_texts handles batching across chunk size limits and retries."""
    from google.genai import errors

    # Mock client and aio.models.embed_content
    mock_client = MagicMock()
    mock_aio_models = MagicMock()
    mock_client.aio.models = mock_aio_models

    attempt_count = 0

    async def fake_embed_content(
        model: str, contents: list[str], config: object = None
    ) -> object:
        nonlocal attempt_count
        attempt_count += 1

        # Simulate 1 transient rate limit on first attempt of first batch
        if attempt_count == 1:
            raise errors.APIError(
                code=429,
                response_json={"error": {"message": "Rate limit exceeded"}},
            )

        # Return mock embeddings matching batch size
        mock_response = MagicMock()
        mock_embeddings = []
        for _ in contents:
            mock_emb = MagicMock()
            mock_emb.values = [0.1, 0.2, 0.3]
            mock_embeddings.append(mock_emb)
        mock_response.embeddings = mock_embeddings
        return mock_response

    mock_aio_models.embed_content = AsyncMock(side_effect=fake_embed_content)

    # 125 sample texts with batch_size=50 -> 3 batches (50, 50, 25)
    test_texts = [f"Sample sentence {i}" for i in range(125)]

    embeddings = await embed_texts(
        texts=test_texts,
        client=mock_client,
        batch_size=50,
        model="text-embedding-004",
        output_dimensionality=3,
    )

    assert len(embeddings) == 125
    assert len(embeddings[0]) == 3
    assert embeddings[0] == [0.1, 0.2, 0.3]
    # Total calls: 1 failed + 3 successful batches = 4 calls
    assert attempt_count == 4


@pytest.mark.asyncio
async def test_embed_texts_empty_input() -> None:
    """Empty list returns empty embedding results immediately."""
    result = await embed_texts([])
    assert result == []


def test_filing_chunks_db_round_trip_with_fake_embeddings() -> None:
    """Store chunks with fake embeddings and query back with vector similarity."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL database is unreachable; skipping DB test.")

    test_ticker = "TESTCORP"
    test_form = "10-K"
    test_date = date(2024, 3, 15)
    test_accession = f"0009999999-24-{uuid.uuid4().hex[:8]}"

    chunk_1 = FilingChunkData(
        ticker=test_ticker,
        form=test_form,
        filing_date=test_date,
        accession=test_accession,
        section_name="Item 1",
        chunk_index=0,
        text="TestCorp manufactures advanced high performance semiconductor chips.",
        token_count=8,
    )
    chunk_2 = FilingChunkData(
        ticker=test_ticker,
        form=test_form,
        filing_date=test_date,
        accession=test_accession,
        section_name="Item 7",
        chunk_index=0,
        text="Revenues increased by twenty percent driven by enterprise cloud demand.",
        token_count=10,
    )

    # Fake 768-dimensional embeddings
    fake_dim = 768
    fake_emb_1 = [0.0] * fake_dim
    fake_emb_1[0] = 1.0  # Vector pointing along dim 0

    fake_emb_2 = [0.0] * fake_dim
    fake_emb_2[1] = 1.0  # Vector pointing along dim 1

    with contextlib.contextmanager(get_session)() as session:
        try:
            # 1. Store chunks with fake embeddings
            stored_records = save_filing_chunks(
                session=session,
                chunks=[chunk_1, chunk_2],
                embeddings=[fake_emb_1, fake_emb_2],
            )
            assert len(stored_records) == 2
            assert stored_records[0].id is not None
            assert stored_records[0].ticker == test_ticker
            assert stored_records[0].accession == test_accession

            # 2. Query back stored chunks by ticker and accession
            queried = get_filing_chunks(
                session=session,
                ticker=test_ticker,
                accession=test_accession,
            )
            assert len(queried) == 2
            assert queried[0].section_name == "Item 1"
            assert queried[0].text == chunk_1.text
            assert queried[0].token_count == 8
            assert queried[0].embedding is not None
            assert len(queried[0].embedding) == fake_dim
            assert queried[0].embedding[0] == pytest.approx(1.0)

            # 3. Vector similarity search using pgvector cosine distance
            # Query vector close to emb_1
            query_vector = [0.0] * fake_dim
            query_vector[0] = 0.99
            query_vector[2] = 0.05

            similar = find_similar_chunks(
                session=session,
                query_embedding=query_vector,
                limit=2,
                ticker=test_ticker,
            )
            assert len(similar) == 2
            top_chunk, distance = similar[0]
            assert top_chunk.section_name == "Item 1"
            assert top_chunk.text == chunk_1.text
            assert distance < 0.1  # Very small cosine distance to query_vector

        finally:
            # Cleanup test records
            session.execute(
                delete(FilingChunk).where(FilingChunk.accession == test_accession)
            )
            session.commit()
