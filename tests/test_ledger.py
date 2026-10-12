"""Database tests for LedgerRepository, claims, and evidence persistence."""

from datetime import UTC, date, datetime
from typing import Any

import pytest
from sqlalchemy import text

from lekka_ai.db import (
    LedgerRepository,
    create_agent_run,
    create_run,
    create_task,
    create_tool_call,
    get_claims,
    get_engine,
    get_session,
    mark_contradiction,
)
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence


def is_database_reachable() -> bool:
    """Check if the database is reachable."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture
def db_session() -> Any:
    """Yield a database session and clean up after test."""
    if not is_database_reachable():
        pytest.skip("Database is unavailable")
    session_generator = get_session()
    session = next(session_generator)
    try:
        yield session
    finally:
        try:
            next(session_generator)
        except StopIteration:
            pass


@pytest.mark.db
def test_ledger_insert_and_get_claims(db_session: Any) -> None:
    """Insert claims across types with evidence and query them back."""
    run = create_run(db_session, question="Evaluate Microsoft revenue FY24")
    task = create_task(
        db_session,
        run_id=run.id,
        agent_type="analyst",
        instruction="Extract income statement facts",
    )
    agent_run = create_agent_run(db_session, task_id=task.id)

    # Tool call to ground the fact claim
    tool_call = create_tool_call(
        db_session,
        agent_run_id=agent_run.id,
        tool_name="get_sec_filing",
        output_json={"revenue": 245120000000},
    )

    fact_claim = Claim(
        claim_id="msft_rev_fact_01",
        text="Microsoft FY2024 revenue was $245.12B",
        type=ClaimType.FACT,
        confidence=0.98,
        produced_by="sec_analyst",
        evidence=[
            Evidence(
                source_type="sec_filing",
                document_id="0000950170-24-087843",
                section="Item 8",
                tool_call_id=str(tool_call.id),
                retrieved_at=datetime.now(UTC).isoformat(),
                as_of_date="2024-06-30",
                raw_value={"revenue_usd": 245120000000},
            )
        ],
    )

    calc_claim = Claim(
        claim_id="msft_rev_growth_calc_01",
        text="Microsoft FY2024 YoY revenue growth was 15.7%",
        type=ClaimType.CALCULATION,
        derived_by="((245.12 - 211.91) / 211.91) * 100",
        confidence=0.95,
        produced_by="quant_model",
    )

    interp_claim = Claim(
        claim_id="msft_growth_interp_01",
        text="Cloud services momentum drove double-digit top-line expansion",
        type=ClaimType.INTERPRETATION,
        depends_on=["msft_rev_fact_01", "msft_rev_growth_calc_01"],
        confidence=0.85,
        produced_by="synthesis_agent",
    )

    repo = LedgerRepository(db_session)
    persisted = repo.add_claims(
        [fact_claim, calc_claim, interp_claim],
        run_id=run.id,
        task_id=task.id,
        agent_run_id=agent_run.id,
    )
    assert len(persisted) == 3

    # Query all claims for this run
    claims = repo.get_claims(run.id)
    assert len(claims) == 3
    claim_ids = [c.claim_id for c in claims]
    assert "msft_rev_fact_01" in claim_ids
    assert "msft_rev_growth_calc_01" in claim_ids
    assert "msft_growth_interp_01" in claim_ids

    # Verify fact claim attributes and linked evidence
    fact_db = next(c for c in claims if c.claim_id == "msft_rev_fact_01")
    assert fact_db.type == "fact"
    assert fact_db.confidence == 0.98
    assert fact_db.run_id == run.id
    assert fact_db.task_id == task.id
    assert fact_db.agent_run_id == agent_run.id
    assert len(fact_db.evidence) == 1

    ev_db = fact_db.evidence[0]
    assert ev_db.source_type == "sec_filing"
    assert ev_db.document_id == "0000950170-24-087843"
    assert ev_db.section == "Item 8"
    assert ev_db.as_of_date == date(2024, 6, 30)
    assert ev_db.tool_call_id == tool_call.id
    assert ev_db.raw_value == {"revenue_usd": 245120000000}

    # Verify schema conversion roundtrip
    schema_claim = fact_db.to_schema()
    assert isinstance(schema_claim, Claim)
    assert schema_claim.claim_id == "msft_rev_fact_01"
    assert schema_claim.evidence[0].tool_call_id == str(tool_call.id)


@pytest.mark.db
def test_ledger_query_by_type_and_filters(db_session: Any) -> None:
    """Test filtering claims by type, confidence, and agent role."""
    run = create_run(db_session, question="Filter test run")
    task = create_task(
        db_session, run_id=run.id, agent_type="researcher", instruction="Task"
    )
    agent_run = create_agent_run(db_session, task_id=task.id)

    tool_call = create_tool_call(
        db_session,
        agent_run_id=agent_run.id,
        tool_name="get_filing",
    )

    claims_data = [
        Claim(
            claim_id="fact_1",
            text="Operating cash flow was $118.5B",
            type="fact",
            confidence=0.99,
            produced_by="auditor",
            evidence=[
                Evidence(source_type="sec_filing", tool_call_id=str(tool_call.id))
            ],
        ),
        Claim(
            claim_id="fact_2",
            text="Free cash flow was $74.1B",
            type="fact",
            confidence=0.80,
            produced_by="analyst",
            evidence=[
                Evidence(source_type="sec_filing", tool_call_id=str(tool_call.id))
            ],
        ),
        Claim(
            claim_id="calc_1",
            text="FCF conversion ratio is 62.5%",
            type="calculation",
            confidence=0.90,
            derived_by="74.1 / 118.5",
            produced_by="analyst",
        ),
        Claim(
            claim_id="interp_1",
            text="Capital expenditure intensity will rise next fiscal year",
            type="interpretation",
            confidence=0.70,
            depends_on=["fact_1", "calc_1"],
            produced_by="strategist",
        ),
    ]

    repo = LedgerRepository(db_session)
    repo.add_claims(claims_data, run_id=run.id, task_id=task.id)

    # 1. Filter by type 'fact'
    fact_claims = repo.get_claims(run.id, filters={"type": "fact"})
    assert len(fact_claims) == 2
    assert all(c.type == "fact" for c in fact_claims)

    # 2. Filter by type 'calculation' using keyword arg
    calc_claims = repo.get_claims(run.id, claim_type="calculation")
    assert len(calc_claims) == 1
    assert calc_claims[0].claim_id == "calc_1"

    # 3. Filter by min_confidence
    high_conf = repo.get_claims(run.id, min_confidence=0.95)
    assert len(high_conf) == 1
    assert high_conf[0].claim_id == "fact_1"

    # 4. Filter by produced_by
    analyst_claims = repo.get_claims(run.id, produced_by="analyst")
    assert len(analyst_claims) == 2
    assert {c.claim_id for c in analyst_claims} == {"fact_2", "calc_1"}

    # 5. Module-level helper get_claims
    interp_claims = get_claims(db_session, run.id, claim_type="interpretation")
    assert len(interp_claims) == 1
    assert interp_claims[0].claim_id == "interp_1"


@pytest.mark.db
def test_ledger_link_evidence_to_tool_calls(db_session: Any) -> None:
    """Link evidence to tool_calls and verify ORM relationship and filtering."""
    run = create_run(db_session, question="Tool call linkage test")
    task = create_task(
        db_session, run_id=run.id, agent_type="quant", instruction="Run DCF"
    )
    agent_run = create_agent_run(db_session, task_id=task.id)

    tool_call_1 = create_tool_call(
        db_session,
        agent_run_id=agent_run.id,
        tool_name="run_dcf",
        input_json={"ticker": "AAPL"},
        output_json={"fair_value": 240.50},
    )
    tool_call_2 = create_tool_call(
        db_session,
        agent_run_id=agent_run.id,
        tool_name="get_industry_data",
        input_json={"sector": "Technology"},
        output_json={"pe_ratio": 28.5},
    )

    claim_1 = Claim(
        claim_id="claim_dcf_val",
        text="DCF valuation indicates fair value of $240.50 per share",
        type="fact",
        confidence=0.92,
        evidence=[
            Evidence(
                source_type="dcf_tool",
                tool_call_id=str(tool_call_1.id),
                raw_value={"fair_value": 240.50},
            )
        ],
    )
    claim_2 = Claim(
        claim_id="claim_tech_mult",
        text="Tech sector median P/E ratio is 28.5x",
        type="fact",
        confidence=0.90,
        evidence=[
            Evidence(
                source_type="market_data",
                tool_call_id=str(tool_call_2.id),
                raw_value={"pe_ratio": 28.5},
            )
        ],
    )

    repo = LedgerRepository(db_session)
    repo.add_claims([claim_1, claim_2], run_id=run.id, agent_run_id=agent_run.id)

    # Filter claims by tool_call_1
    dcf_claims = repo.get_claims(run.id, tool_call_id=tool_call_1.id)
    assert len(dcf_claims) == 1
    assert dcf_claims[0].claim_id == "claim_dcf_val"

    ev = dcf_claims[0].evidence[0]
    assert ev.tool_call_id == tool_call_1.id
    # Test relationship from evidence to ToolCall model
    assert ev.tool_call is not None
    assert ev.tool_call.id == tool_call_1.id
    assert ev.tool_call.tool_name == "run_dcf"
    assert ev.tool_call.output_json == {"fair_value": 240.50}

    # Test reverse relationship from ToolCall to evidence
    db_session.refresh(tool_call_1)
    assert len(tool_call_1.evidence) == 1
    assert tool_call_1.evidence[0].id == ev.id


@pytest.mark.db
def test_ledger_mark_contradiction(db_session: Any) -> None:
    """Mark claims as contradicted and check persistence and errors."""
    run = create_run(db_session, question="Contradiction test")
    task = create_task(
        db_session, run_id=run.id, agent_type="analyst", instruction="Inspect"
    )
    agent_run = create_agent_run(db_session, task_id=task.id)

    tool_call = create_tool_call(
        db_session,
        agent_run_id=agent_run.id,
        tool_name="sec_search",
    )

    claim_original = Claim(
        claim_id="claim_orig_rev",
        text="Reported revenue was $100M",
        type="fact",
        confidence=0.85,
        evidence=[Evidence(source_type="sec_filing", tool_call_id=str(tool_call.id))],
    )
    claim_amended = Claim(
        claim_id="claim_amended_rev",
        text="Restated 10-K/A revenue was $92M",
        type="fact",
        confidence=0.99,
        evidence=[Evidence(source_type="sec_filing", tool_call_id=str(tool_call.id))],
    )

    repo = LedgerRepository(db_session)
    repo.add_claims([claim_original, claim_amended], run_id=run.id)

    # Mark contradiction
    updated = repo.mark_contradiction("claim_orig_rev", "claim_amended_rev")
    assert "claim_amended_rev" in updated.contradicted_by

    # Query from DB to verify persistence across sessions
    reloaded = repo.get_claim("claim_orig_rev")
    assert reloaded is not None
    assert "claim_amended_rev" in reloaded.contradicted_by

    # Mark another contradiction via module helper
    mark_contradiction(db_session, "claim_orig_rev", "claim_audit_note_03")
    reloaded_again = repo.get_claim("claim_orig_rev")
    assert reloaded_again is not None
    assert "claim_audit_note_03" in reloaded_again.contradicted_by
    assert "claim_amended_rev" in reloaded_again.contradicted_by

    # Marking contradiction on non-existent claim raises ValueError
    with pytest.raises(ValueError, match="Claim 'non_existent_id' not found"):
        repo.mark_contradiction("non_existent_id", "some_id")
