"""Gemini LLM provider using google-genai SDK."""

from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from lekka_ai.config import get_settings
from lekka_ai.llm.base import (
    LLMClient,
    LLMResponse,
    Message,
    ToolSpec,
    Usage,
)


class GeminiClient(LLMClient):
    """Client for generating completions using Google Gemini models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self.client = genai.Client(api_key=self.api_key)

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | dict[str, Any] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Generate text from Gemini with history, system prompt, and usage."""
        system_parts: list[str] = []
        if system:
            system_parts.append(system)

        contents: list[types.Content] = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )

        config_kwargs: dict[str, Any] = {}
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata is not None:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        return LLMResponse(
            text=response.text,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            raw=response,
        )
