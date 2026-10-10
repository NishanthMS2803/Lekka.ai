"""Fake LLM client returning scripted responses for unit tests."""

from typing import Any

from pydantic import BaseModel

from lekka_ai.llm.base import LLMClient, LLMResponse, Message, ToolSpec


class FakeLLM(LLMClient):
    """Mock LLM client returning predetermined responses in sequence."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.responses: list[LLMResponse] = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | dict[str, Any] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Record input arguments and return the next queued scripted response."""
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "response_schema": response_schema,
                "system": system,
            }
        )

        if self.responses:
            return self.responses.pop(0)

        return LLMResponse(text="default scripted response")
