"""Tracing framework for instrumenting tools, LLM calls, and execution spans."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker

from lekka_ai.db.repository import create_tool_call
from lekka_ai.db.session import get_session_factory
from lekka_ai.llm.base import LLMClient, LLMResponse, Message, ToolSpec, Usage
from lekka_ai.logging import get_logger
from lekka_ai.tools.base import Tool

logger = get_logger(__name__)


@dataclass
class Span:
    """Represents a single execution span within a trace."""

    name: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    parent_span: Span | None = None
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    latency_ms: int | None = None
    status: str = "running"
    error: str | None = None
    token_usage: Usage | dict[str, int] | None = None
    span_type: str = "generic"
    tool_name: str | None = None
    agent_run_id: uuid.UUID | None = None
    tool_call_id: uuid.UUID | None = None
    input_json: dict[str, Any] | None = None
    output_json: dict[str, Any] | list[Any] | None = None
    cache_hit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def parent_id(self) -> uuid.UUID | None:
        """Return the unique ID of the parent span, if any."""
        return self.parent_span.id if self.parent_span else None


_current_span: ContextVar[Span | None] = ContextVar("current_span", default=None)


class Tracer:
    """Tracer for instrumenting execution flow and persisting tool calls."""

    def __init__(
        self,
        session: Session | None = None,
        session_factory: sessionmaker[Session] | None = None,
        agent_run_id: uuid.UUID | None = None,
        run_id: str | uuid.UUID | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self.session = session
        self.session_factory = session_factory
        self.agent_run_id = agent_run_id
        self.run_id = run_id
        self.event_bus = event_bus
        self.spans: list[Span] = []

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        span_type: str = "generic",
        tool_name: str | None = None,
        agent_run_id: uuid.UUID | None = None,
        input_json: dict[str, Any] | None = None,
        output_json: dict[str, Any] | list[Any] | None = None,
        cache_hit: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[Span]:
        """Record an execution span with parent tracking, latency, and status."""
        start_time = datetime.now(UTC)
        start_perf = time.perf_counter()
        parent = _current_span.get()

        span_obj = Span(
            name=name,
            parent_span=parent,
            start_time=start_time,
            span_type=span_type,
            tool_name=tool_name,
            agent_run_id=agent_run_id or self.agent_run_id,
            input_json=input_json,
            output_json=output_json,
            cache_hit=cache_hit,
            metadata=metadata if metadata is not None else {},
        )

        token = _current_span.set(span_obj)
        try:
            yield span_obj
        except Exception as exc:
            span_obj.status = "error"
            span_obj.error = str(exc)
            raise
        else:
            if span_obj.status == "running":
                span_obj.status = "ok"
        finally:
            _current_span.reset(token)
            end_time = datetime.now(UTC)
            latency_ms = max(0, int((time.perf_counter() - start_perf) * 1000))
            span_obj.end_time = end_time
            span_obj.latency_ms = latency_ms

            self.spans.append(span_obj)

            if span_obj.span_type == "tool" or span_obj.tool_name is not None:
                self._persist_tool_call(span_obj)
                if self.event_bus and self.run_id:
                    try:
                        import asyncio

                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            self.event_bus.emit(
                                str(self.run_id),
                                "tool_called",
                                {
                                    "tool_name": (span_obj.tool_name or span_obj.name),
                                    "input_json": span_obj.input_json,
                                    "latency_ms": span_obj.latency_ms,
                                    "cache_hit": span_obj.cache_hit,
                                },
                            )
                        )
                    except RuntimeError:
                        pass

    def _persist_tool_call(self, span_obj: Span) -> None:
        """Persist a tool call span to the tool_calls table via the repository."""
        target_agent_run_id = span_obj.agent_run_id or self.agent_run_id
        if not target_agent_run_id:
            return

        tool_name = span_obj.tool_name or span_obj.name

        if self.session is not None:
            tool_call = create_tool_call(
                self.session,
                agent_run_id=target_agent_run_id,
                tool_name=tool_name,
                input_json=span_obj.input_json,
                output_json=span_obj.output_json,
                error=span_obj.error,
                latency_ms=span_obj.latency_ms,
                cache_hit=span_obj.cache_hit,
            )
            span_obj.tool_call_id = tool_call.id
        elif self.session_factory is not None:
            with self.session_factory() as session:
                tool_call = create_tool_call(
                    session,
                    agent_run_id=target_agent_run_id,
                    tool_name=tool_name,
                    input_json=span_obj.input_json,
                    output_json=span_obj.output_json,
                    error=span_obj.error,
                    latency_ms=span_obj.latency_ms,
                    cache_hit=span_obj.cache_hit,
                )
                span_obj.tool_call_id = tool_call.id
        else:
            try:
                factory = get_session_factory()
                with factory() as session:
                    tool_call = create_tool_call(
                        session,
                        agent_run_id=target_agent_run_id,
                        tool_name=tool_name,
                        input_json=span_obj.input_json,
                        output_json=span_obj.output_json,
                        error=span_obj.error,
                        latency_ms=span_obj.latency_ms,
                        cache_hit=span_obj.cache_hit,
                    )
                    span_obj.tool_call_id = tool_call.id
            except Exception as exc:
                logger.warning(
                    "Skipping database tool call persistence",
                    error=str(exc),
                )

    def wrap_tool_run[InT: BaseModel, OutT: BaseModel](
        self,
        tool: Tool[InT, OutT],
        agent_run_id: uuid.UUID | None = None,
    ) -> Callable[[InT | dict[str, Any]], Coroutine[Any, Any, OutT]]:
        """Wrap a Tool execution inside a tool-call span."""
        return wrap_tool_run(self, tool, agent_run_id=agent_run_id)

    def wrap_llm_call(
        self,
        client: LLMClient,
    ) -> Callable[..., Coroutine[Any, Any, LLMResponse]]:
        """Wrap an LLMClient.generate call inside an LLM span."""
        return wrap_llm_call(self, client)

    async def trace_tool[InT: BaseModel, OutT: BaseModel](
        self,
        tool: Tool[InT, OutT],
        input_data: InT | dict[str, Any],
        agent_run_id: uuid.UUID | None = None,
    ) -> OutT:
        """Execute a tool wrapped in a span directly."""
        return await trace_tool_run(
            self,
            tool,
            input_data,
            agent_run_id=agent_run_id,
        )

    async def trace_llm(
        self,
        client: LLMClient,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | dict[str, Any] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Execute an LLM call wrapped in a span directly."""
        return await trace_llm_call(
            self,
            client,
            messages=messages,
            tools=tools,
            response_schema=response_schema,
            system=system,
        )


def wrap_tool_run[InT: BaseModel, OutT: BaseModel](
    tracer: Tracer,
    tool: Tool[InT, OutT],
    agent_run_id: uuid.UUID | None = None,
) -> Callable[[InT | dict[str, Any]], Coroutine[Any, Any, OutT]]:
    """Return a wrapper function around tool.run / execute that records spans."""

    async def wrapped(input_data: InT | dict[str, Any]) -> OutT:
        input_json: dict[str, Any] | None = None
        if isinstance(input_data, BaseModel):
            input_json = input_data.model_dump()
        elif isinstance(input_data, dict):
            input_json = input_data

        async with tracer.span(
            name=f"tool:{tool.name}",
            span_type="tool",
            tool_name=tool.name,
            agent_run_id=agent_run_id,
            input_json=input_json,
        ) as span:
            if isinstance(input_data, tool.input_model):
                result = await tool.run(input_data)
            else:
                result = await tool.execute(input_data)

            if isinstance(result, BaseModel):
                span.output_json = result.model_dump()
            elif isinstance(result, (dict, list)):
                span.output_json = result
            return result

    return wrapped


def wrap_llm_call(
    tracer: Tracer,
    client: LLMClient,
) -> Callable[..., Coroutine[Any, Any, LLMResponse]]:
    """Return a wrapper function around LLMClient.generate that records spans."""

    async def wrapped(
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | dict[str, Any] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        client_name = getattr(client, "model_name", client.__class__.__name__)
        async with tracer.span(
            name=f"llm:{client_name}",
            span_type="llm",
            metadata={"messages_count": len(messages)},
        ) as span:
            response = await client.generate(
                messages=messages,
                tools=tools,
                response_schema=response_schema,
                system=system,
            )
            span.token_usage = response.usage
            return response

    return wrapped


async def trace_tool_run[InT: BaseModel, OutT: BaseModel](
    tracer: Tracer,
    tool: Tool[InT, OutT],
    input_data: InT | dict[str, Any],
    agent_run_id: uuid.UUID | None = None,
) -> OutT:
    """Helper to run a tool wrapped in a span directly."""
    wrapped = wrap_tool_run(tracer, tool, agent_run_id=agent_run_id)
    return await wrapped(input_data)


async def trace_llm_call(
    tracer: Tracer,
    client: LLMClient,
    messages: list[Message],
    tools: list[ToolSpec] | None = None,
    response_schema: type[BaseModel] | dict[str, Any] | None = None,
    system: str | None = None,
) -> LLMResponse:
    """Helper to call an LLM wrapped in a span directly."""
    wrapped = wrap_llm_call(tracer, client)
    return await wrapped(
        messages=messages,
        tools=tools,
        response_schema=response_schema,
        system=system,
    )
