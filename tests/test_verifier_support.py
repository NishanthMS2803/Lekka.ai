"""Tests for Source Support Verifier auditing factual claims against traces."""

from __future__ import annotations

import json
from typing import Any

import pytest

from lekka_ai.llm.base import LLMResponse, Usage
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence
from lekka_ai.tracing import Tracer
from lekka_ai.verify.support import (
    BatchSupportResponse,
    ClaimSupportCheck,
    SupportVerdict,
    fetch_claim_evidence_passage,
    verify_claim_support,
)


@pytest.mark.asyncio
async def test_verifier_seeded_true_and_false_claims() -> None:
    """Verify that seeded false claim is flagged as unsupported ('no')."""
    claim_true = Claim(
        claim_id="claim_rev_true",
        text="AAPL reported total revenue of $383.3B in fiscal year 2023.",
        type=ClaimType.FACT,
        confidence=0.98,
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="tc_sec_10k_01",
                raw_value="Total net sales: FY2023 $383,285 million.",
            )
        ],
    )

    claim_false = Claim(
        claim_id="claim_rev_false",
        text="AAPL reported total revenue of $850.0B in fiscal year 2023.",
        type=ClaimType.FACT,
        confidence=0.95,
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="tc_sec_10k_01",
                raw_value="Total net sales: FY2023 $383,285 million.",
            )
        ],
    )

    fake_response = BatchSupportResponse(
        verdicts=[
            ClaimSupportCheck(
                claim_id="claim_rev_true",
                supported=SupportVerdict.YES,
                reason="Directly corroborated by 10-K reported net sales of $383.3B.",
            ),
            ClaimSupportCheck(
                claim_id="claim_rev_false",
                supported=SupportVerdict.NO,
                reason=(
                    "Refuted by cited 10-K filing which discloses $383.3B, not $850.0B."
                ),
            ),
        ]
    )

    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                text=json.dumps(fake_response.model_dump()),
                structured_output=fake_response,
                usage=Usage(input_tokens=150, output_tokens=80),
            )
        ]
    )

    verdicts = await verify_claim_support(
        [claim_true, claim_false],
        llm=fake_llm,
    )

    assert len(verdicts) == 2

    # Verify true claim passed
    assert claim_true.verdict == "yes"
    assert claim_true.support_verdict == "yes"
    assert "383.3B" in (claim_true.verdict_reason or "")

    # Verify seeded false claim flagged
    assert claim_false.verdict == "no"
    assert claim_false.support_verdict == "no"
    assert "Refuted" in (claim_false.verdict_reason or "")

    v_false = next(v for v in verdicts if v.claim_id == "claim_rev_false")
    assert v_false.supported == SupportVerdict.NO


@pytest.mark.asyncio
async def test_verifier_batching() -> None:
    """Verify that multiple claims are split into batches of specified size."""
    claims: list[Claim] = [
        Claim(
            claim_id=f"claim_{i}",
            text=f"Company metric {i} equals {i * 10}.",
            type=ClaimType.FACT,
            confidence=0.9,
            evidence=[
                Evidence(
                    source_type="quant_tool",
                    tool_call_id=f"tc_{i}",
                    raw_value=f"metric_{i}={i * 10}",
                )
            ],
        )
        for i in range(5)
    ]

    # Batch 1 (claims 0, 1)
    resp_batch_1 = BatchSupportResponse(
        verdicts=[
            ClaimSupportCheck(
                claim_id="claim_0",
                supported=SupportVerdict.YES,
                reason="Corroborated by metric 0.",
            ),
            ClaimSupportCheck(
                claim_id="claim_1",
                supported=SupportVerdict.YES,
                reason="Corroborated by metric 1.",
            ),
        ]
    )

    # Batch 2 (claims 2, 3)
    resp_batch_2 = BatchSupportResponse(
        verdicts=[
            ClaimSupportCheck(
                claim_id="claim_2",
                supported=SupportVerdict.YES,
                reason="Corroborated by metric 2.",
            ),
            ClaimSupportCheck(
                claim_id="claim_3",
                supported=SupportVerdict.NO,
                reason="Discrepancy with metric 3.",
            ),
        ]
    )

    # Batch 3 (claim 4)
    resp_batch_3 = BatchSupportResponse(
        verdicts=[
            ClaimSupportCheck(
                claim_id="claim_4",
                supported=SupportVerdict.PARTIAL,
                reason="Partial match for metric 4.",
            ),
        ]
    )

    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                text=json.dumps(resp_batch_1.model_dump()),
                structured_output=resp_batch_1,
            ),
            LLMResponse(
                text=json.dumps(resp_batch_2.model_dump()),
                structured_output=resp_batch_2,
            ),
            LLMResponse(
                text=json.dumps(resp_batch_3.model_dump()),
                structured_output=resp_batch_3,
            ),
        ]
    )

    # With batch_size=2, 5 claims should require exactly 3 calls
    verdicts = await verify_claim_support(
        claims,
        llm=fake_llm,
        batch_size=2,
    )

    assert len(verdicts) == 5
    assert len(fake_llm.calls) == 3

    assert claims[0].verdict == "yes"
    assert claims[1].verdict == "yes"
    assert claims[2].verdict == "yes"
    assert claims[3].verdict == "no"
    assert claims[4].verdict == "partial"


@pytest.mark.asyncio
async def test_verifier_fetches_passage_from_tracer() -> None:
    """Verify that cited tool outputs are resolved from a Tracer instance."""
    tracer = Tracer()
    tool_output: dict[str, Any] = {
        "ticker": "AAPL",
        "volatility_annualized": 0.224,
        "max_drawdown": -0.318,
    }

    async with tracer.span(
        "compute_risk_stats",
        span_type="tool",
        tool_name="compute_risk_stats",
        output_json=tool_output,
    ) as span:
        span_id = str(span.id)

    claim = Claim(
        claim_id="claim_risk_dd",
        text="AAPL historical maximum drawdown reached -31.8%.",
        type=ClaimType.FACT,
        confidence=0.94,
        evidence=[
            Evidence(
                source_type="quant_engine",
                tool_call_id=span_id,
            )
        ],
    )

    passage = fetch_claim_evidence_passage(claim, trace=tracer)
    assert "-0.318" in passage
    assert "compute_risk_stats" in passage

    fake_response = BatchSupportResponse(
        verdicts=[
            ClaimSupportCheck(
                claim_id="claim_risk_dd",
                supported=SupportVerdict.YES,
                reason="Empirical drawdown matches tracer output (-0.318).",
            )
        ]
    )

    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                text=json.dumps(fake_response.model_dump()),
                structured_output=fake_response,
            )
        ]
    )

    verdicts = await verify_claim_support(
        [claim],
        trace=tracer,
        llm=fake_llm,
    )

    assert len(verdicts) == 1
    assert claim.verdict == "yes"


@pytest.mark.asyncio
async def test_verifier_ignores_non_fact_claims() -> None:
    """Verify that interpretation and calculation claims are skipped by verifier."""
    fact_claim = Claim(
        claim_id="claim_fact_01",
        text="Apple spent $29.9B on R&D in FY2023.",
        type=ClaimType.FACT,
        confidence=0.95,
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="tc_rd",
                raw_value="R&D expense: $29,915 million.",
            )
        ],
    )

    interp_claim = Claim(
        claim_id="claim_interp_01",
        text="Apple possesses durable competitive advantages in consumer hardware.",
        type=ClaimType.INTERPRETATION,
        confidence=0.85,
        depends_on=["claim_fact_01"],
    )

    calc_claim = Claim(
        claim_id="claim_calc_01",
        text="R&D intensity is 7.8% of net sales.",
        type=ClaimType.CALCULATION,
        confidence=0.92,
        derived_by="29915 / 383285",
    )

    fake_response = BatchSupportResponse(
        verdicts=[
            ClaimSupportCheck(
                claim_id="claim_fact_01",
                supported=SupportVerdict.YES,
                reason="Verified in 10-K expense disclosures.",
            )
        ]
    )

    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                text=json.dumps(fake_response.model_dump()),
                structured_output=fake_response,
            )
        ]
    )

    verdicts = await verify_claim_support(
        [fact_claim, interp_claim, calc_claim],
        llm=fake_llm,
    )

    # Only the fact claim should be audited
    assert len(verdicts) == 1
    assert fact_claim.verdict == "yes"
    assert interp_claim.verdict is None
    assert calc_claim.verdict is None


@pytest.mark.asyncio
async def test_verifier_dict_claims_support() -> None:
    """Verify support verification works seamlessly on dictionary claim inputs."""
    dict_claim_false = {
        "claim_id": "c_dict_false",
        "text": "Gross margin was 85%.",
        "type": "fact",
        "confidence": 0.9,
        "evidence": [
            {
                "source_type": "sec_filing",
                "tool_call_id": "tc_gm",
                "raw_value": "Gross margin: 44.1%",
            }
        ],
    }

    fake_response = BatchSupportResponse(
        verdicts=[
            ClaimSupportCheck(
                claim_id="c_dict_false",
                supported=SupportVerdict.NO,
                reason="Reported margin was 44.1%, which contradicts 85%.",
            )
        ]
    )

    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                text=json.dumps(fake_response.model_dump()),
                structured_output=fake_response,
            )
        ]
    )

    verdicts = await verify_claim_support([dict_claim_false], llm=fake_llm)

    assert len(verdicts) == 1
    assert dict_claim_false["verdict"] == "no"
    assert dict_claim_false["support_verdict"] == "no"
    assert "contradicts 85%" in str(dict_claim_false.get("verdict_reason", ""))
