"""Gemini LLM provider using google-genai SDK with function calling and retries."""

from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from lekka_ai.config import get_settings
from lekka_ai.llm.base import (
    LLMClient,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)


def _is_transient_error(exc: BaseException) -> bool:
    """Identify retryable 429 rate limit or 5xx transient server errors."""
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        return code in {429, 500, 502, 503, 504}
    return False


@retry(
    retry=retry_if_exception(_is_transient_error),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: list[types.Content],
    config: types.GenerateContentConfig | None,
) -> Any:
    """Call generate_content with tenacity exponential backoff on transient errors."""
    return await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
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
        """Generate text, execute tools, or return structured output."""
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

        if tools:
            tool_declarations: list[types.FunctionDeclaration] = [
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=types.Schema.model_validate(tool.json_schema),
                )
                for tool in tools
            ]
            config_kwargs["tools"] = [
                types.Tool(function_declarations=tool_declarations)
            ]

        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response = await _generate_with_retry(
            self.client,
            self.model,
            contents,
            config,
        )

        parsed_tool_calls: list[ToolCall] = []
        if getattr(response, "function_calls", None):
            for i, fc in enumerate(response.function_calls):
                call_id = getattr(fc, "id", None) or f"call_{fc.name}_{i}"
                call_args = dict(fc.args) if fc.args else {}
                parsed_tool_calls.append(
                    ToolCall(
                        id=call_id,
                        name=fc.name,
                        args=call_args,
                    )
                )

        structured_obj: Any = None
        if response_schema is not None and getattr(response, "text", None):
            if isinstance(response_schema, type) and issubclass(
                response_schema, BaseModel
            ):
                structured_obj = response_schema.model_validate_json(response.text)
            elif hasattr(response, "parsed") and response.parsed is not None:
                structured_obj = response.parsed

        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata is not None:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        return LLMResponse(
            text=response.text,
            tool_calls=parsed_tool_calls,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            structured_output=structured_obj,
            raw=response,
        )
