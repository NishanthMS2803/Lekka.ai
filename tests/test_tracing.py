"""Unit tests for Tracing framework and span wrappers."""

import pytest
from pydantic import BaseModel, Field

from lekka_ai.llm.base import LLMResponse, Message, Usage
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.tools.base import Tool
from lekka_ai.tools.echo import EchoOutput, EchoTool
from lekka_ai.tracing import Tracer, wrap_llm_call, wrap_tool_run


class FailingInput(BaseModel):
    """Input payload for FailingTool."""

    value: int = Field(..., description="Integer value to test")


class FailingOutput(BaseModel):
    """Output payload for FailingTool."""

    result: int = Field(..., description="Calculation result")


class FailingTool(Tool[FailingInput, FailingOutput]):
    """Tool that deliberately raises an error for failure testing."""

    name: str = "failing_calc"
    description: str = "A tool that throws an error during calculation."
    input_model: type[FailingInput] = FailingInput
    output_model: type[FailingOutput] = FailingOutput

    async def run(self, input_data: FailingInput) -> FailingOutput:
        """Raise an exception to test error tracing."""
        raise RuntimeError(f"Division by zero in financial formula: {input_data.value}")


@pytest.mark.asyncio
async def test_tracer_basic_span() -> None:
    """Tracer records name, start/end, status, and latency."""
    tracer = Tracer()

    async with tracer.span(name="step:init") as span:
        assert span.name == "step:init"
        assert span.status == "running"
        assert span.start_time is not None
        assert span.parent_span is None

    assert len(tracer.spans) == 1
    recorded = tracer.spans[0]
    assert recorded.status == "ok"
    assert recorded.end_time is not None
    assert recorded.latency_ms is not None
    assert recorded.latency_ms >= 0
    assert recorded.error is None


@pytest.mark.asyncio
async def test_tracer_nested_parent_spans() -> None:
    """Parent span relationship is maintained across nested spans."""
    tracer = Tracer()

    async with tracer.span(name="parent_operation") as parent:
        assert parent.parent_span is None

        async with tracer.span(name="child_operation") as child:
            assert child.parent_span is parent
            assert child.parent_id == parent.id

        # Back in parent context
        async with tracer.span(name="second_child") as second_child:
            assert second_child.parent_span is parent
            assert second_child.parent_id == parent.id

    assert len(tracer.spans) == 3
    # Completion order: child_operation, second_child, parent_operation
    assert tracer.spans[0].name == "child_operation"
    assert tracer.spans[1].name == "second_child"
    assert tracer.spans[2].name == "parent_operation"


@pytest.mark.asyncio
async def test_wrap_llm_call_records_tokens() -> None:
    """wrap_llm_call records LLM span and token usage metrics."""
    tracer = Tracer()
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                text="The net income is $100M.",
                usage=Usage(input_tokens=42, output_tokens=18),
            )
        ]
    )

    traced_generate = wrap_llm_call(tracer, fake_llm)
    response = await traced_generate(
        messages=[Message(role="user", content="Calculate net income")]
    )

    assert response.text == "The net income is $100M."
    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.name == "llm:FakeLLM"
    assert span.span_type == "llm"
    assert span.status == "ok"
    assert span.token_usage is not None
    assert isinstance(span.token_usage, Usage)
    assert span.token_usage.input_tokens == 42
    assert span.token_usage.output_tokens == 18
    assert span.latency_ms is not None
    assert span.latency_ms >= 0


@pytest.mark.asyncio
async def test_wrap_tool_run_success() -> None:
    """wrap_tool_run captures input, output, and execution metrics."""
    tracer = Tracer()
    tool = EchoTool()
    traced_tool = wrap_tool_run(tracer, tool)

    output = await traced_tool({"message": "test echo"})
    assert isinstance(output, EchoOutput)
    assert output.echoed == "test echo"

    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.name == "tool:echo"
    assert span.span_type == "tool"
    assert span.tool_name == "echo"
    assert span.status == "ok"
    assert span.input_json == {"message": "test echo"}
    assert span.output_json == {"echoed": "test echo"}
    assert span.error is None
    assert span.latency_ms is not None
    assert span.latency_ms >= 0


@pytest.mark.asyncio
async def test_wrap_tool_run_failing_records_error_and_latency() -> None:
    """A failing tool records an error span with error details and latency."""
    tracer = Tracer()
    failing_tool = FailingTool()
    traced_tool = wrap_tool_run(tracer, failing_tool)

    with pytest.raises(RuntimeError, match="Division by zero in financial formula"):
        await traced_tool({"value": 0})

    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.name == "tool:failing_calc"
    assert span.span_type == "tool"
    assert span.tool_name == "failing_calc"
    assert span.status == "error"
    assert span.error is not None
    assert "Division by zero in financial formula: 0" in span.error
    assert span.latency_ms is not None
    assert span.latency_ms >= 0
    assert span.input_json == {"value": 0}
