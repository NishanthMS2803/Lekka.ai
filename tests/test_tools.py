"""Tests for Tool framework, EchoTool, and ToolRegistry."""

import pytest
from pydantic import ValidationError

from lekka_ai.tools import EchoInput, EchoOutput, EchoTool, ToolRegistry


@pytest.mark.asyncio
async def test_tool_schema_export() -> None:
    """to_spec() generates a valid ToolSpec from the Pydantic input schema."""
    tool = EchoTool()
    spec = tool.to_spec()

    assert spec.name == "echo"
    assert spec.description == "Echo back the provided message string."
    assert spec.json_schema.get("type") == "object"
    assert "message" in spec.json_schema.get("properties", {})
    assert "message" in spec.json_schema.get("required", [])


@pytest.mark.asyncio
async def test_tool_execution_success() -> None:
    """Tool executes successfully with both Pydantic input and raw dictionary."""
    tool = EchoTool()

    # Direct model run
    result1 = await tool.run(EchoInput(message="hello world"))
    assert isinstance(result1, EchoOutput)
    assert result1.echoed == "hello world"

    # Execution with dictionary parsing
    result2 = await tool.execute({"message": "test dictionary"})
    assert isinstance(result2, EchoOutput)
    assert result2.echoed == "test dictionary"


@pytest.mark.asyncio
async def test_tool_input_validation_error() -> None:
    """Passing invalid input to execute() raises Pydantic ValidationError."""
    tool = EchoTool()

    with pytest.raises(ValidationError) as exc_info:
        await tool.execute({})

    errors = exc_info.value.errors()
    assert any(err["loc"] == ("message",) for err in errors)


def test_tool_registry_operations() -> None:
    """ToolRegistry supports registration, retrieval, listing, and errors."""
    registry = ToolRegistry()
    tool = EchoTool()

    assert len(registry.list()) == 0

    registry.register(tool)
    assert len(registry.list()) == 1
    assert registry.list()[0] is tool

    retrieved = registry.get("echo")
    assert retrieved is tool

    # Duplicate registration
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)

    # Missing tool lookup
    with pytest.raises(KeyError, match="not found"):
        registry.get("nonexistent_tool")
