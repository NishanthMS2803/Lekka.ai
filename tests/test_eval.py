"""Tests for evaluation harness: dataset, variants, metrics, and runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from lekka_ai.db import get_engine, get_session
from lekka_ai.db.models import EvalResultModel
from lekka_ai.eval.dataset import EvalExample, load_dataset, save_dataset
from lekka_ai.eval.metrics import (
    latency_metric,
    success_metric,
)
from lekka_ai.eval.runner import EvalRunner, EvalSummary
from lekka_ai.eval.variant import RunResult, StubVariant, Variant
from lekka_ai.schemas.claims import Claim, ClaimType


def is_database_reachable() -> bool:
    """Check if the PostgreSQL database is reachable."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def test_eval_example_serialization_and_jsonl(tmp_path: Path) -> None:
    """Test saving and loading EvalExamples using the JSONL dataset format."""
    dataset_file = tmp_path / "test_dataset.jsonl"

    examples = [
        EvalExample(
            id="bench_01",
            question="What was Apple's total revenue for FY2024?",
            ticker="AAPL",
            as_of_date="2024-09-30",
            type="factual",
            expected={"value": 391.0, "tolerance": 0.05},
            tags=["revenue", "10-k"],
        ),
        EvalExample(
            id="bench_02",
            question="Who is the Chief Executive Officer of Microsoft?",
            ticker="MSFT",
            as_of_date="2024-06-30",
            type="factual",
            expected="Satya Nadella",
            tags=["governance", "leadership"],
        ),
    ]

    save_dataset(examples, dataset_file)
    assert dataset_file.exists()

    loaded = load_dataset(dataset_file)
    assert len(loaded) == 2
    assert loaded[0].id == "bench_01"
    assert loaded[0].ticker == "AAPL"
    assert loaded[0].expected == {"value": 391.0, "tolerance": 0.05}
    assert loaded[1].id == "bench_02"
    assert loaded[1].expected == "Satya Nadella"
    assert loaded[1].tags == ["governance", "leadership"]


def test_load_dataset_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    """Test that dataset loader ignores whitespace and comment lines."""
    dataset_file = tmp_path / "annotated.jsonl"
    content = (
        "# Benchmark Dataset Header Comment\n"
        "\n"
        '{"id": "q1", "question": "Question 1", "type": "factual"}\n'
        "  # Inline indented comment\n"
        "\n"
        '{"id": "q2", "question": "Question 2", "type": "calc", "expected": 42.0}\n'
    )
    dataset_file.write_text(content, encoding="utf-8")

    loaded = load_dataset(dataset_file)
    assert len(loaded) == 2
    assert loaded[0].id == "q1"
    assert loaded[1].id == "q2"
    assert loaded[1].expected == 42.0


def test_load_dataset_invalid_format_raises_error(tmp_path: Path) -> None:
    """Test that malformed JSON in dataset file raises ValueError with line number."""
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("not-a-valid-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to parse EvalExample on line 1"):
        load_dataset(bad_file)


@pytest.mark.asyncio
async def test_stub_variant_execution() -> None:
    """Test StubVariant returns configured mock outputs."""
    claim = Claim(
        claim_id="clm_1",
        text="Apple reported $391B in revenue.",
        type=ClaimType.FACT,
        produced_by="direct_llm",
        confidence=0.95,
    )
    stub = StubVariant(
        name="stub_analyst",
        answer="Apple reported $391B in revenue.",
        claims=[claim],
        cost=0.0025,
        latency_ms=120.0,
        tool_calls=3,
    )

    example = EvalExample(
        id="q_100",
        question="What was Apple's FY2024 revenue?",
        ticker="AAPL",
    )

    result = await stub.run(example)
    assert isinstance(result, RunResult)
    assert result.example_id == "q_100"
    assert result.variant == "stub_analyst"
    assert result.answer == "Apple reported $391B in revenue."
    assert len(result.claims) == 1
    assert result.cost == 0.0025
    assert result.latency_ms == 120.0
    assert result.tool_calls == 3


@pytest.mark.asyncio
async def test_runner_executes_two_question_dataset(tmp_path: Path) -> None:
    """Test EvalRunner executes a 2-question dataset over multiple variants."""
    dataset_file = tmp_path / "two_questions.jsonl"
    examples = [
        EvalExample(
            id="q_apple_rev",
            question="What is Apple's FY2024 revenue?",
            ticker="AAPL",
            expected={"value": 391.0, "tolerance": 0.05},
        ),
        EvalExample(
            id="q_msft_ceo",
            question="Who is the CEO of Microsoft?",
            ticker="MSFT",
            expected="Satya Nadella",
        ),
    ]
    save_dataset(examples, dataset_file)

    # Variant 1: Accurate stub
    accurate_claim = {
        "claim_id": "c1",
        "text": "Apple FY2024 revenue was $391.0B",
        "evidence": [{"raw_value": {"value": 391.0}}],
    }
    v1 = StubVariant(
        name="v1_accurate",
        answer="Satya Nadella is the CEO of Microsoft and Apple revenue is 391.0.",
        claims=[accurate_claim],
        cost=0.005,
        latency_ms=150.0,
        tool_calls=2,
    )

    # Variant 2: Inaccurate stub
    v2 = StubVariant(
        name="v2_inaccurate",
        answer="I do not know the answer.",
        claims=[],
        cost=0.001,
        latency_ms=50.0,
        tool_calls=0,
    )

    runner = EvalRunner()
    summary: EvalSummary = await runner.run(
        dataset=dataset_file,
        variants=[v1, v2],
        eval_run_id="run_test_001",
        print_summary=True,
    )

    assert summary.eval_run_id == "run_test_001"
    assert len(summary.results) == 4  # 2 questions * 2 variants

    # Check v1 aggregate stats
    v1_agg = summary.variants["v1_accurate"]
    assert v1_agg.count == 2
    assert v1_agg.success_rate == 1.0
    assert v1_agg.accuracy == 1.0  # Both 391.0 and "Satya Nadella" match
    assert v1_agg.avg_claims == 1.0
    assert v1_agg.avg_tool_calls == 2.0
    assert v1_agg.total_cost == pytest.approx(0.01)

    # Check v2 aggregate stats
    v2_agg = summary.variants["v2_inaccurate"]
    assert v2_agg.count == 2
    assert v2_agg.accuracy == 0.0
    assert v2_agg.avg_claims == 0.0


@pytest.mark.asyncio
async def test_pluggable_metrics() -> None:
    """Test injecting custom metric functions into EvalRunner."""

    def custom_char_count(res: RunResult, ex: EvalExample) -> int:
        return len(res.answer)

    custom_metrics = {
        "char_count": custom_char_count,
        "success": success_metric,
        "latency": latency_metric,
    }

    runner = EvalRunner(metrics=custom_metrics)
    examples = [
        EvalExample(id="ex1", question="Q1"),
        EvalExample(id="ex2", question="Q2"),
    ]
    stub = StubVariant(name="custom_stub", answer="Hello world")

    summary = await runner.run(examples, [stub], print_summary=False)
    assert len(summary.results) == 2
    assert summary.results[0].metrics["char_count"] == 11
    assert summary.results[1].metrics["char_count"] == 11

    agg = summary.variants["custom_stub"]
    assert agg.metric_averages["char_count"] == 11.0


@pytest.mark.asyncio
async def test_variant_failure_handled_gracefully() -> None:
    """Test that crashing variants do not crash the runner and log failure."""

    class FailingVariant(Variant):
        name = "failing_variant"

        async def run(
            self,
            question: str | EvalExample,
            *,
            as_of_date: str | None = None,
            example_id: str | None = None,
            **kwargs: Any,
        ) -> RunResult:
            raise RuntimeError("Simulated API failure or quota limit")

    runner = EvalRunner()
    examples = [EvalExample(id="ex_crash", question="Crash me")]
    summary = await runner.run(examples, [FailingVariant()], print_summary=False)

    assert len(summary.results) == 1
    result = summary.results[0]
    assert result.error is not None
    assert "Simulated API failure" in result.error
    assert result.metrics["success"] == 0.0
    assert summary.variants["failing_variant"].success_rate == 0.0


@pytest.mark.db
@pytest.mark.asyncio
async def test_runner_persists_to_eval_results_table() -> None:
    """Test EvalRunner persists evaluated results to PostgreSQL eval_results."""
    if not is_database_reachable():
        pytest.skip("Database is unreachable")

    session_gen = get_session()
    session = next(session_gen)
    try:
        run_id = "test_eval_persist_99"
        example = EvalExample(
            id="q_db_test",
            question="What is the gross margin for NVDA?",
            ticker="NVDA",
            expected={"value": 75.0, "tolerance": 0.05},
        )
        stub = StubVariant(
            name="stub_db_variant",
            answer="NVDA gross margin was 75.0%.",
            claims=[{"text": "Gross margin 75.0%", "evidence": [{"raw_value": 75.0}]}],
            cost=0.003,
            latency_ms=90.0,
            tool_calls=1,
        )

        runner = EvalRunner(session=session)
        summary = await runner.run(
            dataset=[example],
            variants=[stub],
            eval_run_id=run_id,
            print_summary=False,
        )
        assert len(summary.results) == 1

        # Query database directly to confirm persistence
        db_rows = (
            session.query(EvalResultModel)
            .filter(EvalResultModel.eval_run_id == run_id)
            .all()
        )
        assert len(db_rows) == 1
        row = db_rows[0]
        assert row.example_id == "q_db_test"
        assert row.variant == "stub_db_variant"
        assert row.question == "What is the gross margin for NVDA?"
        assert row.answer == "NVDA gross margin was 75.0%."
        assert row.tool_calls == 1
        assert row.latency_ms == 90.0
        assert row.cost == 0.003
        assert row.metrics.get("success") == 1.0
        assert row.metrics.get("accuracy") == 1.0

        # Cleanup test row
        session.delete(row)
        session.commit()
    finally:
        try:
            next(session_gen)
        except StopIteration:
            pass
