"""Base models and abstract interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message in a conversation thread."""

    role: str
    content: str


class ToolSpec(BaseModel):
    """Declaration of a callable tool available to the LLM."""

    name: str
    description: str
    json_schema: dict[str, Any]


class ToolCall(BaseModel):
    """Tool invocation requested by the LLM."""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    """Token usage metrics for an LLM generation."""

    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    """Unified response structure returned by all LLM clients."""

    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    raw: Any = None


class LLMClient(ABC):
    """Abstract interface that all LLM clients must implement."""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | dict[str, Any] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given messages and tools."""
