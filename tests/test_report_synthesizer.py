"""Tests for Report Synthesizer, citations, and FakeLLM regeneration."""

from __future__ import annotations

import pytest

from lekka_ai.llm.base import LLMResponse
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.report.synthesize import (
    ReportValidationError,
    SynthesizedReport,
    synthesize_report,
    validate_report,
)
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence


@pytest.fixture
def sample_ledger_claims() -> list[Claim]:
    """Return realistic grounded ledger claims for report synthesis tests."""
    return [
        Claim(
            claim_id="claim_rev_001",
            text="Apple reported quarterly revenue of $85.8B for Q3 2024.",
            type=ClaimType.FACT,
            confidence=1.0,
            evidence=[
                Evidence(
                    source_type="quarterly_financials",
                    document_id="AAPL",
                    tool_call_id="tc_001",
                    as_of_date="2024-06-29",
                    raw_value={"revenue": 85780000000.0},
                )
            ],
        ),
        Claim(
            claim_id="claim_margin_002",
            text="Operating margin remained stable at 29.8% in Q3 2024.",
            type=ClaimType.FACT,
            confidence=1.0,
            evidence=[
                Evidence(
                    source_type="quarterly_financials",
                    document_id="AAPL",
                    tool_call_id="tc_002",
                    as_of_date="2024-06-29",
                    raw_value={"operating_margin": 0.298},
                )
            ],
        ),
        Claim(
            claim_id="claim_dcf_003",
            text="Base DCF valuation model indicates a fair value of $210.00.",
            type=ClaimType.CALCULATION,
            confidence=0.9,
            derived_by="run_dcf",
            evidence=[
                Evidence(
                    source_type="dcf_model",
                    tool_call_id="tc_003",
                    raw_value={"fair_value": 210.0},
                )
            ],
        ),
        Claim(
            claim_id="claim_bull_004",
            text="AI adoption is projected to drive 12.0% revenue growth.",
            type=ClaimType.INTERPRETATION,
            confidence=0.85,
            depends_on=["claim_rev_001"],
            evidence=[
                Evidence(
                    source_type="analyst_estimates",
                    tool_call_id="tc_004",
                    raw_value={"growth": 0.12},
                )
            ],
        ),
        Claim(
            claim_id="claim_bear_005",
            text="Services growth decelerates to 3.0% under headwinds.",
            type=ClaimType.INTERPRETATION,
            confidence=0.8,
            depends_on=["claim_margin_002"],
            evidence=[
                Evidence(
                    source_type="sec_filing",
                    document_id="AAPL",
                    tool_call_id="tc_005",
                    raw_value={"bear_growth": 0.03},
                )
            ],
        ),
        Claim(
            claim_id="claim_risk_006",
            text="Greater China regional sales contracted by 6.5% year over year.",
            type=ClaimType.FACT,
            confidence=0.95,
            evidence=[
                Evidence(
                    source_type="quarterly_financials",
                    document_id="AAPL",
                    tool_call_id="tc_006",
                    as_of_date="2024-06-29",
                    raw_value={"china_decline": -0.065},
                )
            ],
        ),
        Claim(
            claim_id="claim_cat_007",
            text="Rollout of Apple Intelligence in Q4 2024 represents key catalyst.",
            type=ClaimType.BACKGROUND,
            confidence=0.9,
            evidence=[
                Evidence(
                    source_type="news",
                    document_id="AAPL",
                    tool_call_id="tc_007",
                    raw_value="Apple Intelligence launch",
                )
            ],
        ),
        Claim(
            claim_id="claim_gap_008",
            text="App Store litigation creates lingering regulatory uncertainty.",
            type=ClaimType.BACKGROUND,
            confidence=0.85,
            evidence=[
                Evidence(
                    source_type="sec_filing",
                    document_id="AAPL",
                    tool_call_id="tc_008",
                    raw_value="EU Digital Markets Act compliance",
                )
            ],
        ),
    ]


def build_valid_markdown_report() -> str:
    """Build a complete, syntactically valid 9-section markdown report."""
    return (
        "# Investment Research Report: Apple Inc. (AAPL)\n\n"
        "## Executive Summary\n"
        "Apple reported revenue of $85.8B for Q3 2024 [claim_rev_001]. "
        "Operating margins were 29.8% [claim_margin_002].\n\n"
        "## Business Quality\n"
        "The ecosystem generates durable revenue of $85.8B [claim_rev_001]. "
        "Execution is highlighted by 29.8% operating margins [claim_margin_002].\n\n"
        "## Valuation Scenarios\n"
        "Base DCF model indicates fair value of $210.00 per share [claim_dcf_003].\n\n"
        "## Bull Case\n"
        "Accelerated AI adoption drives 12.0% revenue growth [claim_bull_004].\n\n"
        "## Bear Case\n"
        "Services revenue growth decelerates to 3.0% [claim_bear_005].\n\n"
        "## Ranked Risks\n"
        "Greater China sales contracted by 6.5% [claim_risk_006].\n\n"
        "## Catalysts\n"
        "Apple Intelligence in Q4 2024 represents a catalyst [claim_cat_007].\n\n"
        "## What Would Invalidate the Thesis\n"
        "Deceleration below 3.0% would invalidate the thesis [claim_bear_005].\n\n"
        "## Uncertainty and Gaps\n"
        "App Store litigation remains an unquantified gap [claim_gap_008].\n"
    )


def test_validate_report_valid_passes(sample_ledger_claims: list[Claim]) -> None:
    """A valid report with all 9 sections and grounded claims passes validation."""
    valid_md = build_valid_markdown_report()
    res = validate_report(valid_md, sample_ledger_claims)
    assert res.passed is True
    assert res.errors == []
    assert res.missing_sections == []
    assert len(res.sentence_to_claims) >= 9


def test_post_check_rejects_invented_number(
    sample_ledger_claims: list[Claim],
) -> None:
    """VERIFY: Post-check rejects sentences containing invented numbers."""
    invented_num_md = (
        "# Investment Research Report: Apple Inc. (AAPL)\n\n"
        "## Executive Summary\n"
        "Apple reported quarterly revenue of $99.9B for Q3 2024 [claim_rev_001].\n\n"
        "## Business Quality\n"
        "Operating margin was 29.8% [claim_margin_002].\n\n"
        "## Valuation Scenarios\n"
        "Fair value is $210.00 [claim_dcf_003].\n\n"
        "## Bull Case\n"
        "AI drives 12.0% growth [claim_bull_004].\n\n"
        "## Bear Case\n"
        "Services decelerates to 3.0% [claim_bear_005].\n\n"
        "## Ranked Risks\n"
        "China sales dropped 6.5% [claim_risk_006].\n\n"
        "## Catalysts\n"
        "Apple Intelligence launches in Q4 2024 [claim_cat_007].\n\n"
        "## What Would Invalidate the Thesis\n"
        "Deceleration below 3.0% invalidates thesis [claim_bear_005].\n\n"
        "## Uncertainty and Gaps\n"
        "Litigation remains a gap [claim_gap_008].\n"
    )
    res = validate_report(invented_num_md, sample_ledger_claims)
    assert res.passed is False
    assert len(res.errors) > 0
    invented_err = [e for e in res.errors if "invented number" in e and "99.9" in e]
    assert len(invented_err) == 1
    assert "claim_rev_001" in invented_err[0]


def test_post_check_rejects_missing_claim_ids(
    sample_ledger_claims: list[Claim],
) -> None:
    """Sentences without claim citations are rejected by post-check."""
    no_citation_md = (
        "# Investment Research Report: Apple Inc. (AAPL)\n\n"
        "## Executive Summary\n"
        "Apple reported quarterly revenue of $85.8B for Q3 2024.\n\n"
        "## Business Quality\n"
        "Operating margin was 29.8% [claim_margin_002].\n\n"
        "## Valuation Scenarios\n"
        "Fair value is $210.00 [claim_dcf_003].\n\n"
        "## Bull Case\n"
        "AI drives 12.0% growth [claim_bull_004].\n\n"
        "## Bear Case\n"
        "Services decelerates to 3.0% [claim_bear_005].\n\n"
        "## Ranked Risks\n"
        "China sales dropped 6.5% [claim_risk_006].\n\n"
        "## Catalysts\n"
        "Apple Intelligence launches in Q4 2024 [claim_cat_007].\n\n"
        "## What Would Invalidate the Thesis\n"
        "Deceleration below 3.0% invalidates thesis [claim_bear_005].\n\n"
        "## Uncertainty and Gaps\n"
        "Litigation remains a gap [claim_gap_008].\n"
    )
    res = validate_report(no_citation_md, sample_ledger_claims)
    assert res.passed is False
    no_cite_err = [e for e in res.errors if "no claim citations" in e]
    assert len(no_cite_err) == 1


def test_post_check_rejects_missing_required_sections(
    sample_ledger_claims: list[Claim],
) -> None:
    """Reports omitting any of the 9 required sections are rejected."""
    partial_md = (
        "# Investment Research Report: Apple Inc. (AAPL)\n\n"
        "## Executive Summary\n"
        "Apple reported revenue of $85.8B for Q3 2024 [claim_rev_001].\n\n"
        "## Business Quality\n"
        "Operating margin was 29.8% [claim_margin_002].\n\n"
        "## Valuation Scenarios\n"
        "Fair value is $210.00 [claim_dcf_003].\n\n"
        "## Bull Case\n"
        "AI drives 12.0% growth [claim_bull_004].\n\n"
        "## Bear Case\n"
        "Services decelerates to 3.0% [claim_bear_005].\n\n"
        "## Ranked Risks\n"
        "China sales dropped 6.5% [claim_risk_006].\n\n"
        "## Catalysts\n"
        "Apple Intelligence launches in Q4 2024 [claim_cat_007].\n\n"
        "## What Would Invalidate the Thesis\n"
        "Deceleration below 3.0% invalidates thesis [claim_bear_005].\n"
    )
    res = validate_report(partial_md, sample_ledger_claims)
    assert res.passed is False
    assert "Uncertainty and Gaps" in res.missing_sections


@pytest.mark.asyncio
async def test_synthesize_report_with_fake_llm_invented_number_regeneration(
    sample_ledger_claims: list[Claim],
) -> None:
    """Verify FakeLLM draft with invented number triggers regeneration."""
    draft_1_bad = (
        "# Investment Research Report: Apple Inc. (AAPL)\n\n"
        "## Executive Summary\n"
        "Apple reported revenue of $125.0B for Q3 2024 [claim_rev_001].\n\n"
        "## Business Quality\n"
        "Operating margin was 29.8% [claim_margin_002].\n\n"
        "## Valuation Scenarios\n"
        "Fair value is $210.00 [claim_dcf_003].\n\n"
        "## Bull Case\n"
        "AI drives 12.0% growth [claim_bull_004].\n\n"
        "## Bear Case\n"
        "Services decelerates to 3.0% [claim_bear_005].\n\n"
        "## Ranked Risks\n"
        "China sales dropped 6.5% [claim_risk_006].\n\n"
        "## Catalysts\n"
        "Apple Intelligence launches in Q4 2024 [claim_cat_007].\n\n"
        "## What Would Invalidate the Thesis\n"
        "Deceleration below 3.0% invalidates thesis [claim_bear_005].\n\n"
        "## Uncertainty and Gaps\n"
        "Litigation remains a gap [claim_gap_008].\n"
    )
    draft_2_good = build_valid_markdown_report()

    fake_llm = FakeLLM(
        responses=[
            LLMResponse(text=draft_1_bad),
            LLMResponse(text=draft_2_good),
        ]
    )

    report = await synthesize_report(
        claims=sample_ledger_claims,
        ticker="AAPL",
        llm_client=fake_llm,
        max_retries=2,
    )

    assert isinstance(report, SynthesizedReport)
    assert report.validation_passed is True
    assert report.attempts == 2
    assert len(report.rejection_reasons) == 1
    assert "125.0" in report.rejection_reasons[0]
    assert "claim_rev_001" in report.claims_used
    assert len(report.sentence_to_claims) >= 9

    assert len(fake_llm.calls) == 2
    repair_call = fake_llm.calls[1]
    repair_prompt = repair_call["messages"][-1].content
    assert "REJECTED by validation" in repair_prompt
    assert "125.0" in repair_prompt


@pytest.mark.asyncio
async def test_synthesize_report_exhausts_max_retries(
    sample_ledger_claims: list[Claim],
) -> None:
    """When LLM keeps failing with ungrounded numbers, synthesis halts."""
    bad_draft = "## Executive Summary\nApple invented $500B [claim_rev_001].\n"
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(text=bad_draft),
            LLMResponse(text=bad_draft),
            LLMResponse(text=bad_draft),
        ]
    )

    report = await synthesize_report(
        claims=sample_ledger_claims,
        ticker="AAPL",
        llm_client=fake_llm,
        max_retries=2,
        raise_on_failure=False,
    )

    assert report.validation_passed is False
    assert report.attempts == 3
    assert len(report.rejection_reasons) == 3


@pytest.mark.asyncio
async def test_synthesize_report_raise_on_failure(
    sample_ledger_claims: list[Claim],
) -> None:
    """When raise_on_failure=True, exhausting retries raises ReportValidationError."""
    bad_draft = "No citations here."
    fake_llm = FakeLLM(responses=[LLMResponse(text=bad_draft)])

    with pytest.raises(ReportValidationError) as exc_info:
        await synthesize_report(
            claims=sample_ledger_claims,
            llm_client=fake_llm,
            max_retries=0,
            raise_on_failure=True,
        )
    assert "Report validation failed" in str(exc_info.value)
