"""Tests for programmatic numeric grader: units, scales, and edge cases."""

from __future__ import annotations

import pytest

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.grader import ExtractedNumber, NumericGrader, factual_accuracy_metric
from lekka_ai.eval.variant import RunResult
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence


def test_parse_number_text_units_and_scales() -> None:
    """Test extracting numbers with unit abbreviations, prefixes, and scales."""
    grader = NumericGrader()

    # Billions
    parsed_b = grader.parse_number_text("Apple generated $391.035 billion in revenue.")
    assert len(parsed_b) >= 1
    assert parsed_b[0].value == 391.035
    assert parsed_b[0].base_value == pytest.approx(391_035_000_000.0)
    assert parsed_b[0].unit == "Billion"

    # Millions
    parsed_m = grader.parse_number_text("Operating expenses were $55,100M.")
    assert len(parsed_m) >= 1
    assert parsed_m[0].value == 55100.0
    assert parsed_m[0].base_value == pytest.approx(55_100_000_000.0)
    assert parsed_m[0].unit == "M"

    # Trillions
    parsed_t = grader.parse_number_text("Market capitalization reached $3.4 trillion.")
    assert len(parsed_t) >= 1
    assert parsed_t[0].value == 3.4
    assert parsed_t[0].base_value == pytest.approx(3_400_000_000_000.0)
    assert parsed_t[0].unit == "Trillion"

    # Thousands
    parsed_k = grader.parse_number_text("Headcount increased by 500k employees.")
    assert len(parsed_k) >= 1
    assert parsed_k[0].value == 500.0
    assert parsed_k[0].base_value == pytest.approx(500_000.0)
    assert parsed_k[0].unit == "K"

    # Percentages
    parsed_pct = grader.parse_number_text("Gross margin was 72.7%.")
    assert len(parsed_pct) >= 1
    assert parsed_pct[0].value == 72.7
    assert parsed_pct[0].is_percentage is True
    assert parsed_pct[0].base_value == pytest.approx(0.727)

    # Accounting negative parentheses
    parsed_neg = grader.parse_number_text("Operating loss was ($12.5 billion).")
    assert len(parsed_neg) >= 1
    assert parsed_neg[0].value == -12.5
    assert parsed_neg[0].base_value == pytest.approx(-12_500_000_000.0)

    # Negative sign
    parsed_minus = grader.parse_number_text("Net adjustment was -$45.2M.")
    assert len(parsed_minus) >= 1
    assert parsed_minus[0].value == -45.2
    assert parsed_minus[0].base_value == pytest.approx(-45_200_000.0)


def test_context_multiplier_in_millions_and_billions() -> None:
    """Test extracting numbers with sentence-level scale contexts like 'in millions'."""
    grader = NumericGrader()

    text_m = "Total consolidated revenue was 391,035 (in millions)."
    nums_m = grader.parse_number_text(text_m)
    assert len(nums_m) >= 1
    assert nums_m[0].value == 391035.0
    assert nums_m[0].base_value == pytest.approx(391_035_000_000.0)

    text_b = "Consolidated revenue was 391.04 (in billions)."
    nums_b = grader.parse_number_text(text_b)
    assert len(nums_b) >= 1
    assert nums_b[0].value == 391.04
    assert nums_b[0].base_value == pytest.approx(391_040_000_000.0)


def test_millions_vs_billions_equivalence() -> None:
    """Test that $391,000 million correctly matches $391 billion and 391,000,000,000."""
    grader = NumericGrader(default_tolerance=0.02)
    cand_m = ExtractedNumber(
        raw_text="$391,000M",
        value=391000.0,
        base_value=391_000_000_000.0,
        unit="M",
        scale=1e6,
    )

    # Target in billions
    target_b = ExtractedNumber(
        raw_text="$391.0B",
        value=391.0,
        base_value=391_000_000_000.0,
        unit="B",
        scale=1e9,
    )
    matched, err = grader.compare_values(cand_m, target_b, tolerance=0.02)
    assert matched is True
    assert err == pytest.approx(0.0)

    # Target in raw dollars
    target_raw = ExtractedNumber(
        raw_text="391035000000",
        value=391035000000.0,
        base_value=391035000000.0,
        unit=None,
        scale=1.0,
    )
    matched_raw, err_raw = grader.compare_values(cand_m, target_raw, tolerance=0.02)
    assert matched_raw is True
    assert err_raw is not None and err_raw < 0.001


def test_sub_billion_millions_equivalence() -> None:
    """Test that $50M matches $0.05B."""
    grader = NumericGrader(default_tolerance=0.01)
    cand = ExtractedNumber(
        raw_text="$50M",
        value=50.0,
        base_value=50_000_000.0,
        unit="M",
        scale=1e6,
    )
    target = ExtractedNumber(
        raw_text="$0.05B",
        value=0.05,
        base_value=50_000_000.0,
        unit="B",
        scale=1e9,
    )
    matched, err = grader.compare_values(cand, target, tolerance=0.01)
    assert matched is True
    assert err == pytest.approx(0.0)


def test_percentage_comparisons() -> None:
    """Test comparing percentages in points (72.7%) vs ratios (0.727)."""
    grader = NumericGrader(default_tolerance=0.01)

    cand_pct = ExtractedNumber(
        raw_text="72.7%",
        value=72.7,
        base_value=0.727,
        unit="%",
        scale=0.01,
        is_percentage=True,
    )

    # Target given as decimal ratio 0.727
    target_ratio = ExtractedNumber(
        raw_text="0.727",
        value=0.727,
        base_value=0.727,
        unit=None,
    )
    matched, err = grader.compare_values(cand_pct, target_ratio, tolerance=0.01)
    assert matched is True

    # Target given as percentage points 72.7
    target_pts = ExtractedNumber(
        raw_text="72.7",
        value=72.7,
        base_value=72.7,
        unit=None,
    )
    matched2, err2 = grader.compare_values(cand_pct, target_pts, tolerance=0.01)
    assert matched2 is True


def test_tolerance_threshold_boundaries() -> None:
    """Test that tolerance comparison respects relative threshold boundaries."""
    grader = NumericGrader(default_tolerance=0.05)
    target = ExtractedNumber(
        raw_text="100.0",
        value=100.0,
        base_value=100.0,
    )

    # 104.9 (+4.9% error) -> inside 5%
    c_inside, _ = grader.compare_values(
        ExtractedNumber(raw_text="104.9", value=104.9, base_value=104.9),
        target,
        tolerance=0.05,
    )
    assert c_inside is True

    # 105.0 (+5.0% error) -> boundary inside 5%
    c_boundary, _ = grader.compare_values(
        ExtractedNumber(raw_text="105.0", value=105.0, base_value=105.0),
        target,
        tolerance=0.05,
    )
    assert c_boundary is True

    # 105.5 (+5.5% error) -> outside 5%
    c_outside, _ = grader.compare_values(
        ExtractedNumber(raw_text="105.5", value=105.5, base_value=105.5),
        target,
        tolerance=0.05,
    )
    assert c_outside is False


def test_grade_structured_claims_and_evidence() -> None:
    """Test grading RunResult extracting numbers from claims and evidence."""
    grader = NumericGrader()

    example = EvalExample(
        id="fact_test_01",
        question="What was Apple's FY2024 revenue?",
        ticker="AAPL",
        expected_value=391035000000.0,
        tolerance=0.02,
    )

    # Variant output with structured claim and raw_value evidence
    ev = Evidence(
        source_type="sec_filing",
        tool_call_id="call_sec_01",
        raw_value={"value": 391035000000.0},
    )
    claim = Claim(
        claim_id="clm_aapl_rev",
        text="Apple reported $391.035 billion in net sales for FY2024.",
        type=ClaimType.FACT,
        evidence=[ev],
        confidence=0.98,
    )
    res = RunResult(
        example_id="fact_test_01",
        variant="test_variant",
        question="What was Apple's FY2024 revenue?",
        answer="Apple reported revenue of $391.035B.",
        claims=[claim],
    )

    grade = grader.grade(res, example)
    assert grade.is_correct is True
    assert grade.score == 1.0
    assert grade.source in ("evidence_raw", "claim_text")


def test_grade_answer_text_fallback() -> None:
    """Test grading RunResult extracting candidate from answer text when no claims."""
    grader = NumericGrader()

    example = EvalExample(
        id="fact_test_02",
        question="What was Apple's diluted EPS for FY2024?",
        ticker="AAPL",
        expected_value=6.08,
        tolerance=0.02,
    )

    res = RunResult(
        example_id="fact_test_02",
        variant="direct_llm",
        question="What was Apple's diluted EPS for FY2024?",
        answer="Apple reported diluted earnings per share of $6.08 for FY2024.",
        claims=[],
    )

    grade = grader.grade(res, example)
    assert grade.is_correct is True
    assert grade.score == 1.0
    assert grade.source == "answer_text"
    assert grade.matched_number is not None
    assert grade.matched_number.value == 6.08


def test_grade_incorrect_value_fails() -> None:
    """Test that hallucinatory numbers fail the grader."""
    grader = NumericGrader(default_tolerance=0.02)

    example = EvalExample(
        id="fact_test_03",
        question="What was Apple's FY2024 revenue?",
        ticker="AAPL",
        expected_value=391035000000.0,
        tolerance=0.02,
    )

    res = RunResult(
        example_id="fact_test_03",
        variant="hallucinating",
        question="What was Apple's FY2024 revenue?",
        answer="Apple reported revenue of $200 billion.",
        claims=[],
    )

    grade = grader.grade(res, example)
    assert grade.is_correct is False
    assert grade.score == 0.0


def test_factual_accuracy_metric_function() -> None:
    """Test pluggable factual_accuracy_metric function interface."""
    example = EvalExample(
        id="fact_test_04",
        question="What was Microsoft's net income for FY2024?",
        ticker="MSFT",
        expected_value=88136000000.0,
        tolerance=0.02,
    )

    # Correct run
    good_res = RunResult(
        example_id="fact_test_04",
        variant="analyst",
        question="What was Microsoft's net income for FY2024?",
        answer="Microsoft reported net income of $88.14 billion.",
    )
    assert factual_accuracy_metric(good_res, example) == 1.0

    # Incorrect run
    bad_res = RunResult(
        example_id="fact_test_04",
        variant="analyst",
        question="What was Microsoft's net income for FY2024?",
        answer="Microsoft reported net income of $50 billion.",
    )
    assert factual_accuracy_metric(bad_res, example) == 0.0


@pytest.mark.asyncio
async def test_runner_evaluates_facts_v0_dataset() -> None:
    """Test EvalRunner evaluating the facts_v0.jsonl dataset with StubVariant."""
    from lekka_ai.eval.runner import EvalRunner
    from lekka_ai.eval.variant import StubVariant

    stub = StubVariant(
        name="stub_analyst",
        answer="The total reported metric is 391035000000.",
        tool_calls=1,
    )
    runner = EvalRunner()
    summary = await runner.run(
        dataset="eval/datasets/facts_v0.jsonl",
        variants=[stub],
        eval_run_id="run_facts_v0_test",
        print_summary=False,
    )

    assert summary.eval_run_id == "run_facts_v0_test"
    assert len(summary.results) == 20
    assert "stub_analyst" in summary.variants
    agg = summary.variants["stub_analyst"]
    assert agg.count == 20
    assert "factual_accuracy" in agg.metric_averages
