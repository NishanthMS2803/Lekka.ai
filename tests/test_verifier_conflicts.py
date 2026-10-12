"""Tests for cross-agent conflict detection and gap-check integration."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from lekka_ai.db.base import Base
from lekka_ai.db.ledger import LedgerRepository
from lekka_ai.llm.base import LLMResponse, Usage
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.orchestrator.dispatch import TaskResult
from lekka_ai.orchestrator.gapcheck import Decision, GapCheckAction, gap_check
from lekka_ai.orchestrator.state import RunState
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence
from lekka_ai.verify.conflicts import (
    Conflict,
    detect_conflicts,
    normalize_metric,
    normalize_period,
)


def test_metric_and_period_normalization() -> None:
    """Verify that financial terms and period strings normalize reliably."""
    assert normalize_metric("Apple reported net sales of $383B") == "revenue"
    assert normalize_metric("Topline revenue reached $100B") == "revenue"
    assert normalize_metric("Operating income expanded 12%") == "operating_income"
    assert normalize_metric("EBIT margin was 30%") == "operating_income"
    assert normalize_metric("Diluted EPS grew to $6.13") == "eps"
    assert normalize_metric("FCF conversion is high") == "free_cash_flow"

    assert normalize_period("Results for FY2023") == "FY2023"
    assert normalize_period("Full year FY23 performance") == "FY2023"
    assert normalize_period("Q1 FY24 earnings") == "2024-Q1"
    assert normalize_period("In calendar 2024") == "FY2024"


def test_detect_conflicts_between_two_agents_reporting_different_revenue() -> None:
    """Verify two agents reporting differing revenue are flagged as a conflict."""
    # Agent 1 (fundamental analyst): $383.3B
    claim_fund = Claim(
        claim_id="claim_rev_fund_01",
        text="Apple reported FY2023 revenue of $383.3B.",
        type=ClaimType.FACT,
        produced_by="fundamental",
        confidence=0.98,
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="tc_sec_10k",
                as_of_date="2023-09-30",
            )
        ],
    )

    # Agent 2 (valuation analyst): $394.0B (discrepancy of ~$10.7B, ~2.8%)
    claim_val = Claim(
        claim_id="claim_rev_val_02",
        text="Apple reported FY2023 revenue of $394.0B.",
        type=ClaimType.FACT,
        produced_by="valuation",
        confidence=0.95,
        evidence=[
            Evidence(
                source_type="analyst_estimates",
                tool_call_id="tc_est_01",
                as_of_date="2023-09-30",
            )
        ],
    )

    state = RunState(question="Evaluate Apple investment prospects")

    conflicts = detect_conflicts([claim_fund, claim_val], state=state)

    # 1. Conflict detected
    assert len(conflicts) == 1
    conf: Conflict = conflicts[0]
    assert conf.metric == "revenue"
    assert conf.period == "FY2023"
    assert conf.agent_a == "fundamental"
    assert conf.agent_b == "valuation"
    assert conf.difference_pct > 0.02

    # 2. Marked contradicted_by reciprocally on claims
    assert "claim_rev_val_02" in claim_fund.contradicted_by
    assert "claim_rev_fund_01" in claim_val.contradicted_by

    # 3. Conflict question added to RunState.open_questions
    assert len(state.open_questions) == 1
    conflict_q = state.open_questions[0]
    assert "Reconcile conflict on revenue (FY2023)" in conflict_q
    assert "fundamental" in conflict_q
    assert "valuation" in conflict_q


@pytest.mark.asyncio
async def test_conflict_surfaced_in_gap_check_input() -> None:
    """Verify detected conflicts in RunState are surfaced in gap check input prompt."""
    claim_fund = Claim(
        claim_id="c_fund_rev",
        text="Apple reported FY2023 revenue of $383.3B.",
        type=ClaimType.FACT,
        produced_by="fundamental",
        confidence=0.98,
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="tc_sec_1",
            )
        ],
    )

    claim_val = Claim(
        claim_id="c_val_rev",
        text="Apple reported FY2023 revenue of $394.0B.",
        type=ClaimType.FACT,
        produced_by="valuation",
        confidence=0.95,
        evidence=[
            Evidence(
                source_type="analyst_estimates",
                tool_call_id="tc_est_1",
            )
        ],
    )

    state = RunState(question="Is AAPL attractive?")

    # Populate task results in state
    state.results["task_fund"] = TaskResult(
        task_id="task_fund",
        agent_type="fundamental",
        status="completed",
        claims=[claim_fund.model_dump()],
        confidence=0.98,
    )
    state.results["task_val"] = TaskResult(
        task_id="task_val",
        agent_type="valuation",
        status="completed",
        claims=[claim_val.model_dump()],
        confidence=0.95,
    )

    # Run conflict detection
    conflicts = detect_conflicts([claim_fund, claim_val], state=state)
    assert len(conflicts) == 1
    assert len(state.open_questions) >= 1

    # Configure FakeLLM for gap check
    fake_decision = Decision(
        action=GapCheckAction.FOLLOW_UP,
        reasons=["Reconcile discrepancy between fundamental and valuation revenue."],
        follow_up_tasks=[],
    )
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                text=json.dumps(fake_decision.model_dump()),
                structured_output=fake_decision,
                usage=Usage(input_tokens=200, output_tokens=50),
            )
        ]
    )

    # Invoke gap_check with state containing the surfaced conflict
    decision = await gap_check(state=state, llm=fake_llm)
    assert decision.action == GapCheckAction.FOLLOW_UP

    # Verify: conflict was explicitly surfaced in the prompt delivered to the LLM
    assert len(fake_llm.calls) == 1
    call_prompt = fake_llm.calls[0]["messages"][0].content

    assert "RUN-LEVEL OPEN QUESTIONS & CONFLICTS:" in call_prompt
    assert "Reconcile conflict on revenue (FY2023)" in call_prompt
    assert "fundamental" in call_prompt
    assert "valuation" in call_prompt


def test_unit_normalization_prevents_false_positive_conflict() -> None:
    """Verify that $383.3B and $383,285M match and do not trigger a false conflict."""
    claim_billions = Claim(
        claim_id="c_b_rev",
        text="Apple reported FY2023 revenue of $383.3B.",
        type=ClaimType.FACT,
        produced_by="agent_a",
        confidence=0.95,
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="tc_b_1",
            )
        ],
    )

    claim_millions = Claim(
        claim_id="c_m_rev",
        text="Apple reported FY2023 net sales of $383,285 million.",
        type=ClaimType.FACT,
        produced_by="agent_b",
        confidence=0.99,
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="tc_m_1",
            )
        ],
    )

    state = RunState(question="Apple check")
    conflicts = detect_conflicts(
        [claim_billions, claim_millions],
        state=state,
        tolerance=0.01,
    )

    # Difference is $15M out of $383.3B (< 0.004%), well within 1% tolerance
    assert len(conflicts) == 0
    assert len(state.open_questions) == 0
    assert len(claim_billions.contradicted_by) == 0
    assert len(claim_millions.contradicted_by) == 0


def test_conflict_marked_in_ledger_repository() -> None:
    """Verify that ledger repository receives reciprocal mark_contradiction updates."""
    import uuid

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        ledger = LedgerRepository(session)

        claim_1 = Claim(
            claim_id="claim_c1",
            text="Operating profit for FY23 was $114.3B.",
            type=ClaimType.FACT,
            produced_by="agent_1",
            confidence=0.9,
            evidence=[
                Evidence(
                    source_type="sec_filing",
                    tool_call_id="tc_1",
                )
            ],
        )

        claim_2 = Claim(
            claim_id="claim_c2",
            text="Operating profit for FY23 was $95.0B.",
            type=ClaimType.FACT,
            produced_by="agent_2",
            confidence=0.9,
            evidence=[
                Evidence(
                    source_type="sec_filing",
                    tool_call_id="tc_2",
                )
            ],
        )

        test_run_id = uuid.uuid4()
        ledger.add_claims([claim_1, claim_2], run_id=test_run_id)

        # Run conflict detection with ledger
        conflicts = detect_conflicts([claim_1, claim_2], ledger=ledger)
        assert len(conflicts) == 1

        # Query database models from ledger
        db_c1 = ledger.get_claim("claim_c1")
        db_c2 = ledger.get_claim("claim_c2")

        assert db_c1 is not None
        assert db_c2 is not None
        assert "claim_c2" in db_c1.contradicted_by
        assert "claim_c1" in db_c2.contradicted_by
