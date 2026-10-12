"""Unit and database tests for AgentRuntime inner loop execution."""

from uuid import uuid4

import pytest
from pydantic import BaseModel, Field
from sqlalchemy import text

from lekka_ai.db import (
    create_run,
    create_task,
    get_engine,
    get_session,
)
from lekka_ai.db.models import AgentRun
from lekka_ai.harness import (
    AgentProfile,
    AgentRuntime,
    AgentTask,
    Limits,
    ToolView,
)
from lekka_ai.llm.base import LLMResponse, ToolCall, Usage
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.tools.base import Tool, ToolRegistry
from lekka_ai.tracing import Tracer


class EchoInput(BaseModel):
    """Input payload for EchoTool."""

    message: str = Field(..., description="Message string")


class EchoOutput(BaseModel):
    """Output payload for EchoTool."""

    echoed: str = Field(..., description="Echoed response")


class EchoTool(Tool[EchoInput, EchoOutput]):
    """Test tool echoing input message."""

    name = "echo_tool"
    description = "Echoes a message"
    input_model = EchoInput
    output_model = EchoOutput

    async def run(self, input_data: EchoInput) -> EchoOutput:
        return EchoOutput(echoed=input_data.message)


class AddInput(BaseModel):
    """Input payload for AddTool."""

    x: int = Field(..., description="First operand")
    y: int = Field(..., description="Second operand")


class AddOutput(BaseModel):
    """Output payload for AddTool."""

    sum: int = Field(..., description="Calculated sum")


class AddTool(Tool[AddInput, AddOutput]):
    """Test tool adding two integers."""

    name = "add_tool"
    description = "Adds two numbers"
    input_model = AddInput
    output_model = AddOutput

    async def run(self, input_data: AddInput) -> AddOutput:
        return AddOutput(sum=input_data.x + input_data.y)


@pytest.fixture
def registry() -> ToolRegistry:
    """Fixture providing a test registry with EchoTool and AddTool."""
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(AddTool())
    return reg


@pytest.mark.asyncio
async def test_runtime_one_tool_call_then_text(registry: ToolRegistry) -> None:
    """Scripted FakeLLM test: one tool call followed by plain text answer."""
    task = AgentTask(
        task_id="task_calc_rev_01",
        agent_type="analyst",
        instruction="What was Apple's FY2024 revenue?",
        as_of_date="2024-09-30",
    )
    profile = AgentProfile(
        agent_type="analyst",
        allowed_tools=["echo_tool"],
    )
    tracer = Tracer()
    tool_view = ToolView(registry, profile, tracer=tracer)

    # Turn 1: model asks to execute echo_tool
    call_1 = LLMResponse(
        text="Looking up SEC data...",
        tool_calls=[
            ToolCall(id="tc_1", name="echo_tool", args={"message": "AAPL $391.0B"})
        ],
        usage=Usage(input_tokens=15, output_tokens=10),
    )
    # Turn 2: model sees tool output and returns final plain text
    call_2 = LLMResponse(
        text="Apple reported FY2024 revenue of $391.0B.",
        tool_calls=[],
        usage=Usage(input_tokens=25, output_tokens=15),
    )

    fake_llm = FakeLLM(responses=[call_1, call_2])
    result = await AgentRuntime.run(
        task=task,
        profile=profile,
        llm=fake_llm,
        tool_view=tool_view,
        limits=Limits(max_iterations=5),
    )

    assert result.status == "completed"
    assert result.iterations == 2
    assert result.tool_calls_count == 1
    assert result.tokens_in == 40
    assert result.tokens_out == 25
    assert result.final_text == "Apple reported FY2024 revenue of $391.0B."
    assert result.limits_hit == []

    # Check conversation history:
    # 1. user instruction, 2. assistant tool request,
    # 3. tool result, 4. assistant final text
    assert len(result.messages) == 4
    assert result.messages[0].role == "user"
    assert result.messages[1].role == "assistant"
    assert result.messages[2].role == "tool"
    assert "<untrusted_data" in result.messages[2].content
    assert "AAPL $391.0B" in result.messages[2].content
    assert result.messages[3].role == "assistant"
    assert result.messages[3].content == "Apple reported FY2024 revenue of $391.0B."

    # Tracing check
    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "echo_tool"
    assert tracer.spans[0].status == "ok"


@pytest.mark.asyncio
async def test_runtime_two_tool_calls_then_text(registry: ToolRegistry) -> None:
    """Scripted FakeLLM test: two sequential tool calls then plain text answer."""
    task = AgentTask(
        task_id="task_calc_sum_02",
        agent_type="quant",
        instruction="Calculate total and echo confirmation",
    )
    profile = AgentProfile(
        agent_type="quant",
        allowed_tools=["add_tool", "echo_tool"],
    )
    tracer = Tracer()
    tool_view = ToolView(registry, profile, tracer=tracer)

    # Turn 1: invoke add_tool
    step_1 = LLMResponse(
        text="Computing sum...",
        tool_calls=[ToolCall(id="tc_add", name="add_tool", args={"x": 100, "y": 250})],
        usage=Usage(input_tokens=20, output_tokens=8),
    )
    # Turn 2: invoke echo_tool
    step_2 = LLMResponse(
        text="Echoing verified calculation...",
        tool_calls=[
            ToolCall(id="tc_echo", name="echo_tool", args={"message": "Total: 350"})
        ],
        usage=Usage(input_tokens=30, output_tokens=10),
    )
    # Turn 3: final answer
    step_3 = LLMResponse(
        text="The final verified sum is 350.",
        tool_calls=[],
        usage=Usage(input_tokens=40, output_tokens=12),
    )

    fake_llm = FakeLLM(responses=[step_1, step_2, step_3])
    result = await AgentRuntime.run(
        task=task,
        profile=profile,
        llm=fake_llm,
        tool_view=tool_view,
        limits=Limits(max_iterations=5),
    )

    assert result.status == "completed"
    assert result.iterations == 3
    assert result.tool_calls_count == 2
    assert result.tokens_in == 90
    assert result.tokens_out == 30
    assert result.final_text == "The final verified sum is 350."

    # Check that both tools ran and logged spans
    assert len(tracer.spans) == 2
    span_names = [s.name for s in tracer.spans]
    assert span_names == ["add_tool", "echo_tool"]
    assert all(s.status == "ok" for s in tracer.spans)


@pytest.mark.asyncio
async def test_runtime_stops_on_max_iterations(registry: ToolRegistry) -> None:
    """Verify runtime terminates when max_iterations limit is reached."""
    task = AgentTask(
        task_id="task_loop_03",
        agent_type="analyst",
        instruction="Continuous querying",
    )
    profile = AgentProfile(agent_type="analyst", allowed_tools=["echo_tool"])
    tool_view = ToolView(registry, profile)

    infinite_step = LLMResponse(
        text="Calling again...",
        tool_calls=[ToolCall(id="tc_loop", name="echo_tool", args={"message": "loop"})],
        usage=Usage(input_tokens=10, output_tokens=5),
    )

    fake_llm = FakeLLM(responses=[infinite_step, infinite_step, infinite_step])
    result = await AgentRuntime.run(
        task=task,
        profile=profile,
        llm=fake_llm,
        tool_view=tool_view,
        limits=Limits(max_iterations=2),
    )

    assert result.status == "max_iterations"
    assert result.iterations == 2
    assert "max_iterations" in result.limits_hit


def is_database_reachable() -> bool:
    """Check if PostgreSQL database is reachable."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.mark.db
@pytest.mark.asyncio
async def test_runtime_logs_db_agent_runs_row(registry: ToolRegistry) -> None:
    """Verify runtime creates and updates an agent_runs table row in PostgreSQL."""
    if not is_database_reachable():
        pytest.skip("Database is unavailable")

    session_generator = get_session()
    session = next(session_generator)
    try:
        run = create_run(session, question="Test DB agent run logging")
        db_task = create_task(
            session,
            run_id=run.id,
            agent_type="researcher",
            instruction="Execute tool and finish",
        )

        task = AgentTask(
            task_id=str(db_task.id),
            agent_type="researcher",
            instruction="Execute tool and finish",
        )
        profile = AgentProfile(agent_type="researcher", allowed_tools=["echo_tool"])
        tool_view = ToolView(registry, profile)

        call_1 = LLMResponse(
            text="Querying...",
            tool_calls=[
                ToolCall(id=str(uuid4()), name="echo_tool", args={"message": "db_test"})
            ],
            usage=Usage(input_tokens=12, output_tokens=8),
        )
        call_2 = LLMResponse(
            text="Done.",
            tool_calls=[],
            usage=Usage(input_tokens=18, output_tokens=6),
        )

        fake_llm = FakeLLM(responses=[call_1, call_2])
        result = await AgentRuntime.run(
            task=task,
            profile=profile,
            llm=fake_llm,
            tool_view=tool_view,
            limits=Limits(max_iterations=5),
            session=session,
        )

        assert result.status == "completed"
        assert result.agent_run_id is not None

        # Verify agent_runs row exists in database
        persisted = session.query(AgentRun).filter_by(task_id=db_task.id).first()
        assert persisted is not None
        assert persisted.iterations == 2
        assert persisted.tokens_in == 30
        assert persisted.tokens_out == 14
        assert persisted.status == "completed"
    finally:
        try:
            next(session_generator)
        except StopIteration:
            pass


@pytest.mark.asyncio
async def test_runtime_valid_submit_ends_loop(registry: ToolRegistry) -> None:
    """Valid submit_answer call immediately ends runtime loop with completed status."""
    task = AgentTask(
        task_id="task_valid_sub_01",
        agent_type="analyst",
        instruction="Calculate Apple FY2024 revenue.",
    )
    profile = AgentProfile(agent_type="analyst")
    tracer = Tracer()
    tool_view = ToolView(registry, profile, tracer=tracer)

    submit_payload = {
        "task_id": "task_valid_sub_01",
        "agent_type": "analyst",
        "status": "completed",
        "claims": [
            {
                "claim_id": "claim_01",
                "text": "Apple FY2024 revenue was $391.0B.",
                "type": "fact",
                "confidence": 0.98,
                "evidence": [
                    {
                        "source_type": "sec_filing",
                        "document_id": "0000320193-24-000106",
                        "tool_call_id": "tc_echo_01",
                    }
                ],
            }
        ],
        "confidence": 0.98,
        "confidence_rationale": "Directly grounded in SEC 10-K filing.",
    }

    call = LLMResponse(
        text="Submitting finalized grounded answer.",
        tool_calls=[
            ToolCall(
                id="tc_sub_01",
                name="submit_answer",
                args=submit_payload,
            )
        ],
        usage=Usage(input_tokens=20, output_tokens=15),
    )

    fake_llm = FakeLLM(responses=[call])
    result = await AgentRuntime.run(
        task=task,
        profile=profile,
        llm=fake_llm,
        tool_view=tool_view,
        limits=Limits(max_iterations=5),
    )

    assert result.status == "completed"
    assert result.iterations == 1
    assert result.tool_calls_count == 1
    assert result.repair_attempts == 0
    assert result.submission is not None
    assert result.submission.task_id == "task_valid_sub_01"
    assert result.submission.confidence == 0.98
    assert result.final_text == "Directly grounded in SEC 10-K filing."
    assert "Answer submission accepted and validated." in result.messages[-1].content
    assert len(tracer.spans) == 1
    assert tracer.spans[0].status == "ok"


@pytest.mark.asyncio
async def test_runtime_invalid_submit_triggers_repair(
    registry: ToolRegistry,
) -> None:
    """Invalid submit_answer returns validation error observation and allows repair."""
    task = AgentTask(
        task_id="task_repair_01",
        agent_type="analyst",
        instruction="Determine Apple FY2024 revenue.",
    )
    profile = AgentProfile(agent_type="analyst")
    tracer = Tracer()
    tool_view = ToolView(registry, profile, tracer=tracer)

    # Turn 1: Invalid submission (status='completed' with 0 claims is rejected)
    invalid_payload = {
        "task_id": "task_repair_01",
        "agent_type": "analyst",
        "status": "completed",
        "claims": [],
        "confidence": 0.9,
        "confidence_rationale": "Premature submission without grounded claims.",
    }
    call_1 = LLMResponse(
        text="Submitting answer...",
        tool_calls=[
            ToolCall(id="tc_sub_inv", name="submit_answer", args=invalid_payload)
        ],
        usage=Usage(input_tokens=15, output_tokens=10),
    )

    # Turn 2: Repaired submission with grounded claim
    valid_payload = {
        "task_id": "task_repair_01",
        "agent_type": "analyst",
        "status": "completed",
        "claims": [
            {
                "claim_id": "claim_rep_01",
                "text": "Apple FY2024 revenue was $391.0B.",
                "type": "fact",
                "confidence": 0.95,
                "evidence": [
                    {
                        "source_type": "sec_filing",
                        "tool_call_id": "tc_step_01",
                    }
                ],
            }
        ],
        "confidence": 0.95,
        "confidence_rationale": "Repaired submission with verified SEC evidence.",
    }
    call_2 = LLMResponse(
        text="Repairing submission with required evidence claim.",
        tool_calls=[
            ToolCall(id="tc_sub_valid", name="submit_answer", args=valid_payload)
        ],
        usage=Usage(input_tokens=30, output_tokens=20),
    )

    fake_llm = FakeLLM(responses=[call_1, call_2])
    result = await AgentRuntime.run(
        task=task,
        profile=profile,
        llm=fake_llm,
        tool_view=tool_view,
        limits=Limits(max_iterations=5),
    )

    assert result.status == "completed"
    assert result.iterations == 2
    assert result.repair_attempts == 1
    assert result.submission is not None
    assert result.submission.confidence == 0.95

    # Check observation returned to agent in message 2
    assert result.messages[2].role == "tool"
    assert "Validation failed for submit_answer (1/3)" in result.messages[2].content
    assert (
        "Completed answers require at least one grounded Claim"
        in result.messages[2].content
    )

    # Check accepted submission in message 4
    assert result.messages[4].role == "tool"
    assert "Answer submission accepted and validated." in result.messages[4].content


@pytest.mark.asyncio
async def test_runtime_three_failures_end_gracefully(
    registry: ToolRegistry,
) -> None:
    """3 consecutive validation failures terminate runtime with status=partial."""
    task = AgentTask(
        task_id="task_fail_01",
        agent_type="analyst",
        instruction="Analyze complex market event.",
    )
    profile = AgentProfile(agent_type="analyst")
    tracer = Tracer()
    tool_view = ToolView(registry, profile, tracer=tracer)

    # Three invalid submissions with different validation flaws
    invalid_1 = {
        "task_id": "task_fail_01",
        "agent_type": "analyst",
        "status": "completed",
        "claims": [],  # Violates completed claims constraint
        "confidence": 0.8,
        "confidence_rationale": "Attempt 1 without claims.",
    }
    invalid_2 = {
        "task_id": "task_fail_01",
        "agent_type": "analyst",
        "status": "completed",
        "claims": [],
        "confidence": 2.5,  # Violates confidence le=1.0 constraint
        "confidence_rationale": "Attempt 2 invalid confidence.",
    }
    invalid_3 = {
        "task_id": "task_fail_01",
        "agent_type": "analyst",
        "status": "completed",
        "claims": [],
        "confidence": -0.5,  # Violates confidence ge=0.0 constraint
        "confidence_rationale": "Attempt 3 negative confidence.",
    }

    call_1 = LLMResponse(
        text="Attempt 1",
        tool_calls=[ToolCall(id="tc_1", name="submit_answer", args=invalid_1)],
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    call_2 = LLMResponse(
        text="Attempt 2",
        tool_calls=[ToolCall(id="tc_2", name="submit_answer", args=invalid_2)],
        usage=Usage(input_tokens=15, output_tokens=5),
    )
    call_3 = LLMResponse(
        text="Attempt 3",
        tool_calls=[ToolCall(id="tc_3", name="submit_answer", args=invalid_3)],
        usage=Usage(input_tokens=20, output_tokens=5),
    )

    fake_llm = FakeLLM(responses=[call_1, call_2, call_3])
    result = await AgentRuntime.run(
        task=task,
        profile=profile,
        llm=fake_llm,
        tool_view=tool_view,
        limits=Limits(max_iterations=10),
    )

    # Must terminate after 3 repair attempts, not running up to max_iterations=10
    assert result.status == "partial"
    assert result.iterations == 3
    assert result.repair_attempts == 3
    assert result.submission is None
    assert "submit_validation_failed" in result.limits_hit

    # Verify final gate error message
    assert (
        "Maximum repair attempts exceeded. Ending task." in result.messages[-1].content
    )
    assert tracer.spans[-1].status == "error"
