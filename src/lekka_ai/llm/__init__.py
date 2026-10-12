"""LLM provider interface and implementations."""

from lekka_ai.llm.base import (
    LLMClient,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.llm.gemini import GeminiClient, embed_texts

__all__ = [
    "FakeLLM",
    "GeminiClient",
    "LLMClient",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "embed_texts",
]
