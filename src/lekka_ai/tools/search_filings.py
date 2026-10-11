"""SEC filing semantic vector search tool with citation provenance."""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lekka_ai.db.repository import find_similar_chunks
from lekka_ai.db.session import get_session
from lekka_ai.llm.gemini import embed_texts
from lekka_ai.logging import get_logger
from lekka_ai.tools.base import Tool, default_registry

logger = get_logger(__name__)


class Citation(BaseModel):
    """Citation provenance identifying source filing and chunk."""

    model_config = ConfigDict(extra="allow")

    document_id: str | None = Field(
        default=None, description="Primary document or filing record ID"
    )
    accession: str = Field(..., description="SEC EDGAR unique accession number")
    section: str = Field(..., description="Filing Item heading (e.g. 'Item 1A')")
    filing_date: str = Field(..., description="Filing date (YYYY-MM-DD)")
    chunk_id: str = Field(..., description="Unique chunk UUID string")

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class Passage(BaseModel):
    """Search result passage with similarity score and citation."""

    model_config = ConfigDict(extra="allow")

    text: str = Field(..., description="Passage body text")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    citation: Citation = Field(..., description="Citation source object")

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class SearchFilingsInput(BaseModel):
    """Input parameters for search_filings tool."""

    ticker: str = Field(..., description="Stock ticker symbol to search (e.g. 'AAPL')")
    query: str = Field(..., description="Natural language semantic search query")
    form_types: list[str] | None = Field(
        default=None,
        description="Optional list of SEC forms to include (e.g. ['10-K', '10-Q'])",
    )
    sections: list[str] | None = Field(
        default=None,
        description="Optional section names to filter by (e.g. ['Item 1A', 'Item 7'])",
    )
    as_of_date: str | None = Field(
        default=None,
        description=(
            "Point-in-time cutoff date (YYYY-MM-DD). "
            "Only filings on or before this date are searched."
        ),
    )
    top_k: int = Field(
        default=5,
        description="Maximum number of relevant passages to return (default: 5)",
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def _validate_ticker(cls, v: Any) -> str:
        return str(v).strip().upper()

    @field_validator("query", mode="before")
    @classmethod
    def _validate_query(cls, v: Any) -> str:
        query_str = str(v).strip()
        if not query_str:
            raise ValueError("Search query cannot be empty")
        return query_str

    @field_validator("form_types", mode="before")
    @classmethod
    def _validate_form_types(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, str):
            return [f.strip().upper() for f in v.split(",") if f.strip()]
        if isinstance(v, (list, tuple)):
            return [str(f).strip().upper() for f in v if str(f).strip()]
        return None

    @field_validator("sections", mode="before")
    @classmethod
    def _validate_sections(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, (list, tuple)):
            return [str(s).strip() for s in v if str(s).strip()]
        return None

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _validate_as_of_date(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, (date, datetime)):
            return v.strftime("%Y-%m-%d")
        return str(v).strip()

    @field_validator("top_k", mode="before")
    @classmethod
    def _validate_top_k(cls, v: Any) -> int:
        try:
            val = int(v)
            return max(1, min(val, 50))
        except (ValueError, TypeError):
            return 5


class SearchFilingsOutput(BaseModel):
    """Output payload for search_filings tool."""

    model_config = ConfigDict(extra="allow")

    ticker: str = Field(..., description="Stock ticker symbol queried")
    query: str = Field(..., description="Search query executed")
    top_k: int = Field(..., description="Requested top_k results limit")
    total_results: int = Field(..., description="Number of passages returned")
    as_of_date: str | None = Field(
        default=None, description="Point-in-time cutoff date applied"
    )
    passages: list[Passage] = Field(
        default_factory=list, description="Ranked relevant passages with citations"
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __len__(self) -> int:
        return len(self.passages)

    def __iter__(self) -> Any:
        return iter(self.passages)


class SearchFilingsTool(Tool[SearchFilingsInput, SearchFilingsOutput]):
    """Semantic vector search across SEC filing narrative sections."""

    name: str = "search_filings"
    description: str = (
        "Search SEC 10-K and 10-Q filing narrative sections using vector semantic "
        "similarity. Returns relevant passages with full citation metadata "
        "(accession, section, filing date, chunk ID)."
    )
    input_model: type[SearchFilingsInput] = SearchFilingsInput
    output_model: type[SearchFilingsOutput] = SearchFilingsOutput

    def __init__(
        self,
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
    ) -> None:
        self._embed_fn = embed_fn or embed_texts

    async def run(self, input_data: SearchFilingsInput) -> SearchFilingsOutput:
        """Embed the query and retrieve closest filing chunks using pgvector."""
        # 1. Parse point-in-time cutoff date
        as_of_date_obj: date | None = None
        if input_data.as_of_date:
            try:
                as_of_date_obj = date.fromisoformat(input_data.as_of_date)
            except ValueError:
                logger.warning(
                    "Invalid as_of_date format, ignoring point-in-time filter",
                    as_of_date=input_data.as_of_date,
                )

        # 2. Embed user query
        query_embeddings = await self._embed_fn([input_data.query])
        if not query_embeddings or not query_embeddings[0]:
            logger.error(
                "Failed to generate embedding for query", query=input_data.query
            )
            return SearchFilingsOutput(
                ticker=input_data.ticker,
                query=input_data.query,
                top_k=input_data.top_k,
                total_results=0,
                as_of_date=input_data.as_of_date,
                passages=[],
            )

        query_vector = query_embeddings[0]

        # 3. Query pgvector via repository
        passages: list[Passage] = []
        with contextlib.contextmanager(get_session)() as session:
            rows = find_similar_chunks(
                session=session,
                query_embedding=query_vector,
                limit=input_data.top_k,
                ticker=input_data.ticker,
                sections=input_data.sections,
                form_types=input_data.form_types,
                as_of_date=as_of_date_obj,
            )

            for chunk, distance in rows:
                # Cosine similarity score = max(0.0, 1.0 - distance)
                similarity_score = max(0.0, round(1.0 - distance, 4))
                citation = Citation(
                    document_id=str(chunk.filing_id) if chunk.filing_id else None,
                    accession=chunk.accession,
                    section=chunk.section_name,
                    filing_date=chunk.filing_date.isoformat(),
                    chunk_id=str(chunk.id),
                )
                passages.append(
                    Passage(
                        text=chunk.text,
                        score=similarity_score,
                        citation=citation,
                    )
                )

        logger.info(
            "Executed filing semantic search",
            ticker=input_data.ticker,
            query=input_data.query,
            results_count=len(passages),
        )

        return SearchFilingsOutput(
            ticker=input_data.ticker,
            query=input_data.query,
            top_k=input_data.top_k,
            total_results=len(passages),
            as_of_date=input_data.as_of_date,
            passages=passages,
        )


# Register default instance in default tool registry
default_search_filings_tool = SearchFilingsTool()
default_registry.register(default_search_filings_tool, overwrite=True)
