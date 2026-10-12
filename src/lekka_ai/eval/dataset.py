"""Evaluation dataset format (JSONL) and persistence utilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvalExample(BaseModel):
    """A single evaluation benchmark example."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Unique identifier for the benchmark item")
    question: str = Field(..., description="Financial question or research query")
    ticker: str | None = Field(
        default=None, description="Primary ticker symbol if applicable"
    )
    as_of_date: str | None = Field(
        default=None, description="Point-in-time cutoff date (YYYY-MM-DD)"
    )
    type: str = Field(
        default="factual",
        description=(
            "Query type (e.g. 'factual', 'calculation', "
            "'point_in_time', 'unanswerable')"
        ),
    )
    metric: str | None = Field(
        default=None, description="Target financial metric evaluated"
    )
    expected_value: Any = Field(default=None, description="Ground truth target value")
    tolerance: float | None = Field(
        default=None, description="Allowed deviation ratio (e.g. 0.02 for 2%)"
    )
    source_accession: str | None = Field(
        default=None, description="SEC accession reference number"
    )
    expected: Any = Field(
        default=None,
        description=(
            "Ground truth expectation (e.g. target number, "
            "dict with tolerance, or text)"
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Descriptive tags (e.g. ['market_data', '10-k', 'ratios'])",
    )


def load_dataset(path: str | Path) -> list[EvalExample]:
    """Load evaluation examples from a JSONL file.

    Args:
        path: Path to the JSONL dataset file.

    Returns:
        List of parsed EvalExample objects.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset file not found: {path}")

    examples: list[EvalExample] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            try:
                example = EvalExample.model_validate_json(clean)
                examples.append(example)
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse EvalExample on line {line_num} of {path}: {exc}"
                ) from exc

    return examples


def save_dataset(examples: Sequence[EvalExample], path: str | Path) -> None:
    """Save evaluation examples to a JSONL file.

    Args:
        examples: Sequence of EvalExample objects.
        path: Destination file path.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")
