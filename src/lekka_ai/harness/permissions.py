"""Permission layer restricting tool access per agent profile with audit tracing."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lekka_ai.harness.errors import is_retryable_tool_error
from lekka_ai.harness.models import Limits
from lekka_ai.logging import get_logger
from lekka_ai.tracing import Tracer

if TYPE_CHECKING:
    from lekka_ai.llm.base import ToolSpec
    from lekka_ai.tools.base import Tool, ToolRegistry

logger = get_logger(__name__)


class PermissionDenied(PermissionError):
    """Raised when an agent attempts to access or execute a disallowed tool."""


class AgentProfile(BaseModel):
    """Configuration profile defining permissions and prompt for an agent type."""

    model_config = ConfigDict(extra="allow")

    agent_type: str = Field(
        ...,
        description="Specialist agent role identifier (e.g. 'analyst', 'quant')",
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="List of tool names this agent role is authorized to invoke",
    )
    system_prompt_path: str | None = Field(
        default=None,
        description="File path to the system prompt template for this agent role",
    )
    prompt_file: str | None = Field(
        default=None,
        description="File path to the system prompt template for this agent role",
    )
    answer_extension_schema: str | None = Field(
        default=None,
        description="Dotted import path or name of the typed answer extension schema",
    )
    default_limits: Limits = Field(
        default_factory=Limits,
        description="Default execution limits (max_iterations, max_tokens, etc.)",
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            prompt = (
                data.get("prompt_file")
                or data.get("system_prompt_path")
                or data.get("prompt_path")
            )
            if prompt:
                if not data.get("prompt_file"):
                    data["prompt_file"] = prompt
                if not data.get("system_prompt_path"):
                    data["system_prompt_path"] = prompt
            if "limits" in data and "default_limits" not in data:
                data["default_limits"] = data["limits"]
        return data

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check whether the tool name is in the allowed tools list."""
        clean = tool_name.strip()
        if "*" in self.allowed_tools or clean == "submit_answer":
            return True
        return clean in {t.strip() for t in self.allowed_tools}

    def load_system_prompt(self) -> str | None:
        """Read and return system prompt content from disk if path exists."""
        target = self.prompt_file or self.system_prompt_path
        if not target:
            return None
        path = Path(target)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        cwd_cand = Path.cwd() / target
        if cwd_cand.is_file():
            return cwd_cand.read_text(encoding="utf-8")
        pkg_root = Path(__file__).resolve().parent.parent
        candidate = pkg_root / target
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
        repo_root = pkg_root.parent.parent
        candidate = repo_root / target
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
        return None

    def get_extension_model(self) -> type[BaseModel] | None:
        """Resolve and return the registered or configured answer extension model."""
        from lekka_ai.schemas.submit import get_answer_extension

        registered = get_answer_extension(self.agent_type)
        if registered is not None:
            return registered
        if self.answer_extension_schema:
            from lekka_ai.harness.profiles import resolve_extension_model

            return resolve_extension_model(self.answer_extension_schema)
        return None


class PermittedToolWrapper:
    """Proxy around a Tool instance routing execution through ToolView permissions."""

    def __init__(self, tool: Tool[Any, Any], tool_view: ToolView) -> None:
        self._tool = tool
        self._tool_view = tool_view

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    @property
    def input_model(self) -> type[BaseModel]:
        return self._tool.input_model

    @property
    def output_model(self) -> type[BaseModel]:
        return self._tool.output_model

    def to_spec(self) -> ToolSpec:
        return self._tool.to_spec()

    async def execute(self, input_data: Any) -> Any:
        return await self._tool_view.execute(self._tool.name, input_data)

    async def run(self, input_data: Any) -> Any:
        return await self._tool_view.execute(self._tool.name, input_data)

    async def __call__(self, input_data: Any) -> Any:
        return await self._tool_view.execute(self._tool.name, input_data)


class ToolView:
    """Scoped view over a ToolRegistry enforcing role-based tool permissions."""

    def __init__(
        self,
        registry: ToolRegistry,
        profile: AgentProfile | Sequence[str],
        *,
        tracer: Tracer | None = None,
        agent_run_id: uuid.UUID | None = None,
        max_retries: int = 2,
        backoff_factor: float = 0.05,
        fault_plan: Any | None = None,
    ) -> None:
        self.registry = registry
        if isinstance(profile, AgentProfile):
            self.profile = profile
        else:
            self.profile = AgentProfile(
                agent_type="anonymous",
                allowed_tools=list(profile),
            )
        self.tracer = tracer or Tracer(agent_run_id=agent_run_id)
        self.agent_run_id = agent_run_id
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.fault_plan = fault_plan

    def is_allowed(self, tool_name: str) -> bool:
        """Verify if a tool is permitted under current agent profile."""
        return self.profile.is_tool_allowed(tool_name)

    def specs(self) -> list[ToolSpec]:
        """Return ToolSpecs only for authorized, registered tools."""
        return [
            tool.to_spec()
            for tool in self.registry.list()
            if self.is_allowed(tool.name)
        ]

    def list(self) -> list[Tool[Any, Any]]:
        """Return list of authorized, registered tools."""
        return [
            self.get(tool.name)
            for tool in self.registry.list()
            if self.is_allowed(tool.name)
        ]

    def get(self, name: str) -> Tool[Any, Any]:
        """Retrieve authorized tool.

        Raises PermissionDenied and records in trace if disallowed.
        """
        clean_name = name.strip()
        if not self.is_allowed(clean_name):
            msg = (
                f"PermissionDenied: Agent '{self.profile.agent_type}' "
                f"is not permitted to access tool '{clean_name}'."
            )
            logger.warning(
                "Tool access permission denied",
                agent_type=self.profile.agent_type,
                tool_name=clean_name,
            )
            self._record_denial_span(clean_name, msg)
            raise PermissionDenied(msg)

        raw_tool = self.registry.get(clean_name)
        return PermittedToolWrapper(raw_tool, self)  # type: ignore[return-value]

    async def execute(self, tool_name: str, input_data: Any) -> Any:
        """Execute a tool with permission checks, error tracing, and latency logging."""
        clean_name = tool_name.strip()
        payload = (
            input_data.model_dump()
            if hasattr(input_data, "model_dump")
            else input_data
            if isinstance(input_data, dict)
            else {"input": str(input_data)}
        )

        if not self.is_allowed(clean_name):
            msg = (
                f"PermissionDenied: Agent '{self.profile.agent_type}' "
                f"is not permitted to call tool '{clean_name}'."
            )
            logger.warning(
                "Disallowed tool execution attempted",
                agent_type=self.profile.agent_type,
                tool_name=clean_name,
            )
            async with self.tracer.span(
                clean_name,
                span_type="tool",
                tool_name=clean_name,
                agent_run_id=self.agent_run_id,
                input_json=payload,
            ):
                raise PermissionDenied(msg)

        tool = self.registry.get(clean_name)
        effective_fault_plan = self.fault_plan
        if effective_fault_plan is None:
            from lekka_ai.harness.faults import get_current_fault_plan

            effective_fault_plan = get_current_fault_plan()

        async with self.tracer.span(
            clean_name,
            span_type="tool",
            tool_name=clean_name,
            agent_run_id=self.agent_run_id,
            input_json=payload,
        ) as span_obj:
            last_exc: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:

                    async def _run_tool() -> Any:
                        if hasattr(tool, "execute"):
                            return await tool.execute(input_data)
                        return await tool(input_data)

                    if (
                        effective_fault_plan is not None
                        and effective_fault_plan.should_inject(clean_name)
                    ):
                        result = await effective_fault_plan.apply(clean_name, _run_tool)
                    else:
                        result = await _run_tool()

                    span_obj.output_json = (
                        result.model_dump()
                        if hasattr(result, "model_dump")
                        else result
                        if isinstance(result, (dict, list))
                        else {"result": str(result)}
                    )
                    return result
                except Exception as exc:
                    last_exc = exc
                    if is_retryable_tool_error(exc) and attempt < self.max_retries:
                        backoff = self.backoff_factor * (2**attempt)
                        logger.warning(
                            "Tool execution failed with retryable error, retrying",
                            tool_name=clean_name,
                            attempt=attempt + 1,
                            max_retries=self.max_retries,
                            backoff=backoff,
                            error=str(exc),
                        )
                        await asyncio.sleep(backoff)
                        continue
                    raise
            if last_exc is not None:
                raise last_exc

    async def call(self, tool_name: str, input_data: Any) -> Any:
        """Alias for execute()."""
        return await self.execute(tool_name, input_data)

    async def run(self, tool_name: str, input_data: Any) -> Any:
        """Alias for execute()."""
        return await self.execute(tool_name, input_data)

    async def __call__(self, tool_name: str, input_data: Any) -> Any:
        """Callable shorthand for execute()."""
        return await self.execute(tool_name, input_data)

    def __contains__(self, name: str) -> bool:
        """Check if tool name is permitted and present in the underlying registry."""
        return self.is_allowed(name) and name in self.registry

    def __getitem__(self, name: str) -> Tool[Any, Any]:
        """Support subscription lookup via get()."""
        return self.get(name)

    def _record_denial_span(self, tool_name: str, error_message: str) -> None:
        """Record a permission denied error span on the tracer."""
        from lekka_ai.tracing import Span

        span_obj = Span(
            name=tool_name,
            span_type="tool",
            tool_name=tool_name,
            agent_run_id=self.agent_run_id,
            status="error",
            error=error_message,
        )
        self.tracer.spans.append(span_obj)
        self.tracer._persist_tool_call(span_obj)
