"""Tool for recalling verified unexpired facts with revalidation detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from lekka_ai.tools.base import Tool, default_registry
from lekka_ai.verify.facts_cache import (
    FactCacheRepository,
    get_fact_cache_repository,
)
from lekka_ai.verify.facts_cache import (
    RecalledFact as RecalledFact,
)
from lekka_ai.verify.facts_cache import (
    RecallFactsOutput as RecallFactsOutput,
)

if TYPE_CHECKING:
    pass


class RecallFactsInput(BaseModel):
    """Input payload for recalling unexpired facts for a ticker."""

    ticker: str = Field(
        ...,
        description="Stock ticker symbol to recall facts for, e.g. 'AAPL' or 'NVDA'",
    )
    metrics: list[str] | None = Field(
        default=None,
        description="Optional list of specific metric keys, e.g. ['revenue', 'price']",
    )


class RecallFactsTool(Tool[RecallFactsInput, RecallFactsOutput]):
    """Tool that queries the fact cache and returns unexpired facts."""

    name: str = "recall_facts"
    description: str = (
        "Recall previously verified facts from the fact cache for a ticker. "
        "Returns valid unexpired facts and a 'needs_revalidation' list."
    )
    input_model: type[RecallFactsInput] = RecallFactsInput
    output_model: type[RecallFactsOutput] = RecallFactsOutput

    def __init__(
        self,
        repository: FactCacheRepository | None = None,
        session: Session | None = None,
    ) -> None:
        self.repository = repository
        self.session = session

    async def run(self, input_data: RecallFactsInput) -> RecallFactsOutput:
        """Recall unexpired facts for the requested ticker and metrics."""
        repo = self.repository or get_fact_cache_repository(session=self.session)
        return repo.recall_facts(
            ticker=input_data.ticker,
            metrics=input_data.metrics,
            session=self.session,
        )


default_recall_facts_tool = RecallFactsTool()
default_registry.register(default_recall_facts_tool, overwrite=True)

__all__ = [
    "RecallFactsInput",
    "RecallFactsOutput",
    "RecallFactsTool",
    "RecalledFact",
    "default_recall_facts_tool",
]
