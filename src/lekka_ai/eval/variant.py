"""Variant interface and standard implementations (Agent, Direct, Stub)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.harness.direct import DirectLLMRunner
from lekka_ai.harness.models import AgentTask, Limits
from lekka_ai.harness.permissions import ToolView
from lekka_ai.harness.profiles import create_analyst_profile
from lekka_ai.harness.runtime import AgentRuntime
from lekka_ai.llm.base import LLMClient
from lekka_ai.schemas.claims import Claim
from lekka_ai.tools.base import default_registry


class RunResult(BaseModel):
    """Execution output from a single variant evaluation run."""

    example_id: str = Field(..., description="ID of the benchmark example")
    variant: str = Field(..., description="Name of the executing variant")
    question: str = Field(..., description="Question investigated")
    answer: str = Field(default="", description="Answer text returned")
    claims: list[Any] = Field(
        default_factory=list,
        description="Structured claims returned by the variant",
    )
    cost: float = Field(default=0.0, description="Estimated monetary cost in USD")
    tokens_in: int = Field(default=0, description="Prompt input tokens")
    tokens_out: int = Field(default=0, description="Completion output tokens")
    tokens_total: int = Field(default=0, description="Total tokens consumed")
    latency_ms: float = Field(
        default=0.0, description="Execution latency in milliseconds"
    )
    tool_calls: int = Field(
        default=0, description="Total number of tool calls executed"
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Pluggable evaluation metric scores",
    )
    status: str = Field(
        default="completed",
        description="Execution status (e.g. 'completed', 'abstained', 'failed')",
    )
    is_abstention: bool = Field(
        default=False,
        description="Whether the run resulted in an explicit abstention",
    )
    recovered: bool = Field(
        default=False,
        description="Whether the run successfully recovered from an error or fault",
    )
    confidence_score: float | None = Field(
        default=None,
        description="Overall confidence score for the report/findings",
    )
    missing_evidence: list[str] = Field(
        default_factory=list,
        description="List of stated missing evidence items",
    )
    error: str | None = Field(default=None, description="Error message if run failed")
    raw: Any = Field(default=None, description="Underlying result payload")


class Variant(ABC):
    """Abstract interface defining an evaluation candidate system."""

    name: str
    description: str = ""

    @abstractmethod
    async def run(
        self,
        question: str | EvalExample,
        *,
        as_of_date: str | None = None,
        example_id: str | None = None,
        **kwargs: Any,
    ) -> RunResult:
        """Run the variant against a question or evaluation example.

        Returns:
            RunResult containing claims, cost, latency, and tool_calls.
        """
        ...


class StubVariant(Variant):
    """Deterministic stub variant for testing and calibration."""

    def __init__(
        self,
        name: str = "stub",
        *,
        answer: str = "Stub answer",
        claims: list[Claim | dict[str, Any]] | None = None,
        cost: float = 0.001,
        latency_ms: float = 15.0,
        tool_calls: int = 0,
        error: str | None = None,
    ) -> None:
        self.name = name
        self.answer = answer
        self.claims = claims or []
        self.cost = cost
        self.latency_ms = latency_ms
        self.tool_calls = tool_calls
        self.error = error

    async def run(
        self,
        question: str | EvalExample,
        *,
        as_of_date: str | None = None,
        example_id: str | None = None,
        **kwargs: Any,
    ) -> RunResult:
        """Return scripted stub output immediately."""
        q_text = question.question if isinstance(question, EvalExample) else question
        ex_id = (
            question.id
            if isinstance(question, EvalExample)
            else (example_id or "ex_stub")
        )
        return RunResult(
            example_id=ex_id,
            variant=self.name,
            question=q_text,
            answer=self.answer,
            claims=list(self.claims),
            cost=self.cost,
            latency_ms=self.latency_ms,
            tool_calls=self.tool_calls,
            error=self.error,
        )


class DirectVariant(Variant):
    """Direct LLM baseline (V1) variant without tools."""

    def __init__(
        self,
        llm: LLMClient,
        name: str = "direct_llm",
    ) -> None:
        self.name = name
        self.llm = llm

    async def run(
        self,
        question: str | EvalExample,
        *,
        as_of_date: str | None = None,
        example_id: str | None = None,
        **kwargs: Any,
    ) -> RunResult:
        """Execute question through DirectLLMRunner."""
        q_text = question.question if isinstance(question, EvalExample) else question
        cutoff = (
            question.as_of_date if isinstance(question, EvalExample) else as_of_date
        )
        ex_id = (
            question.id
            if isinstance(question, EvalExample)
            else (example_id or "ex_direct")
        )

        task = AgentTask(
            task_id=ex_id,
            instruction=q_text,
            as_of_date=cutoff,
            agent_type="direct",
        )

        start_t = time.perf_counter()
        try:
            agent_res = await DirectLLMRunner.run(task=task, llm=self.llm)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            claims = list(agent_res.submission.claims) if agent_res.submission else []
            # Estimate cost (~$0.15 per 1M in, $0.60 per 1M out on flash models)
            est_cost = (
                agent_res.tokens_in * 0.00000015 + agent_res.tokens_out * 0.00000060
            )

            return RunResult(
                example_id=ex_id,
                variant=self.name,
                question=q_text,
                answer=agent_res.final_text or "",
                claims=claims,
                cost=est_cost,
                tokens_in=agent_res.tokens_in,
                tokens_out=agent_res.tokens_out,
                tokens_total=agent_res.tokens_in + agent_res.tokens_out,
                latency_ms=elapsed_ms,
                tool_calls=agent_res.tool_calls_count,
                raw=agent_res,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return RunResult(
                example_id=ex_id,
                variant=self.name,
                question=q_text,
                answer="",
                claims=[],
                cost=0.0,
                latency_ms=elapsed_ms,
                tool_calls=0,
                error=str(exc),
            )


class AgentVariant(Variant):
    """Tool-augmented analyst agent (V2) variant."""

    def __init__(
        self,
        llm: LLMClient,
        name: str = "analyst_agent",
        limits: Limits | None = None,
    ) -> None:
        self.name = name
        self.llm = llm
        self.limits = limits or Limits(max_iterations=10, max_seconds=120.0)

    async def run(
        self,
        question: str | EvalExample,
        *,
        as_of_date: str | None = None,
        example_id: str | None = None,
        **kwargs: Any,
    ) -> RunResult:
        """Execute question through AgentRuntime with tools."""
        q_text = question.question if isinstance(question, EvalExample) else question
        cutoff = (
            question.as_of_date if isinstance(question, EvalExample) else as_of_date
        )
        ex_id = (
            question.id
            if isinstance(question, EvalExample)
            else (example_id or "ex_agent")
        )

        task = AgentTask(
            task_id=ex_id,
            instruction=q_text,
            as_of_date=cutoff,
            agent_type="analyst",
        )
        profile = create_analyst_profile()
        tool_view = ToolView(default_registry, profile=profile)

        start_t = time.perf_counter()
        try:
            agent_res = await AgentRuntime.run(
                task=task,
                profile=profile,
                llm=self.llm,
                tool_view=tool_view,
                limits=self.limits,
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            claims = list(agent_res.submission.claims) if agent_res.submission else []
            est_cost = (
                agent_res.tokens_in * 0.00000015 + agent_res.tokens_out * 0.00000060
            )
            answer_text = (
                agent_res.submission.confidence_rationale
                if agent_res.submission and agent_res.submission.confidence_rationale
                else agent_res.final_text or ""
            )

            return RunResult(
                example_id=ex_id,
                variant=self.name,
                question=q_text,
                answer=answer_text,
                claims=claims,
                cost=est_cost,
                tokens_in=agent_res.tokens_in,
                tokens_out=agent_res.tokens_out,
                tokens_total=agent_res.tokens_in + agent_res.tokens_out,
                latency_ms=elapsed_ms,
                tool_calls=agent_res.tool_calls_count,
                raw=agent_res,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return RunResult(
                example_id=ex_id,
                variant=self.name,
                question=q_text,
                answer="",
                claims=[],
                cost=0.0,
                latency_ms=elapsed_ms,
                tool_calls=0,
                error=str(exc),
            )
