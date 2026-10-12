"""Core execution harness data contracts: AgentTask, Limits, and Budget."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class BudgetExhaustedError(RuntimeError):
    """Raised when attempting to charge an exhausted budget under strict mode."""


class AgentTask(BaseModel):
    """Specification of a delegated task for an autonomous agent."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        default_factory=lambda: f"task_{uuid4().hex[:10]}",
        description="Unique identifier for this agent task",
    )
    agent_type: str = Field(
        ...,
        description="Specialist agent role (e.g. 'analyst', 'quant', 'synthesis')",
    )
    instruction: str = Field(
        ...,
        description="Detailed prompt instruction or objective for the subagent",
    )
    context_facts: list[Any] = Field(
        default_factory=list,
        validation_alias=AliasChoices("context_facts", "context", "facts"),
        description="Background claims, documents, or ground-truth facts for context",
    )
    as_of_date: date | str | None = Field(
        default=None,
        description="Point-in-time cutoff date (YYYY-MM-DD)",
    )

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _validate_as_of_date(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, (date, datetime)):
            return v.strftime("%Y-%m-%d")
        return str(v).strip() or None


class Limits(BaseModel):
    """Execution bounds and throttling limits for an agent run."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int | None = Field(
        default=None,
        ge=0,
        description="Maximum execution loops/turns allowed",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Maximum total tokens (input + output) allowed",
    )
    max_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum wall-clock execution time in seconds",
    )
    max_tool_calls: int | None = Field(
        default=None,
        ge=0,
        description="Maximum number of tool invocations allowed",
    )

    def is_exceeded(
        self,
        *,
        iterations: int = 0,
        tokens: int = 0,
        seconds: float = 0.0,
        tool_calls: int = 0,
    ) -> bool:
        """Check if any configured limit has been breached."""
        return (
            len(
                self.exceeded_limits(
                    iterations=iterations,
                    tokens=tokens,
                    seconds=seconds,
                    tool_calls=tool_calls,
                )
            )
            > 0
        )

    def exceeded_limits(
        self,
        *,
        iterations: int = 0,
        tokens: int = 0,
        seconds: float = 0.0,
        tool_calls: int = 0,
    ) -> list[str]:
        """Return the list of specific limit names that were breached."""
        breached: list[str] = []
        if self.max_iterations is not None and iterations >= self.max_iterations:
            breached.append("max_iterations")
        if self.max_tokens is not None and tokens >= self.max_tokens:
            breached.append("max_tokens")
        if self.max_seconds is not None and seconds >= self.max_seconds:
            breached.append("max_seconds")
        if self.max_tool_calls is not None and tool_calls >= self.max_tool_calls:
            breached.append("max_tool_calls")
        return breached

    def to_budget(self) -> Budget:
        """Convert limits into a fresh Budget instance."""
        return Budget(
            tool_calls=self.max_tool_calls or 0,
            tokens=self.max_tokens or 0,
            seconds=self.max_seconds or 0.0,
        )


class Budget(BaseModel):
    """Resource budget allowance with decrement tracking and non-negative balances."""

    model_config = ConfigDict(extra="allow")

    tool_calls: int = Field(default=0, ge=0, description="Allocated tool call budget")
    tokens: int = Field(default=0, ge=0, description="Allocated token budget")
    seconds: float = Field(default=0.0, ge=0.0, description="Allocated seconds budget")
    cost: float = Field(default=0.0, ge=0.0, description="Allocated cost budget (USD)")

    used_tool_calls: int = Field(
        default=0, ge=0, description="Cumulative tool calls used"
    )
    used_tokens: int = Field(default=0, ge=0, description="Cumulative tokens used")
    used_seconds: float = Field(
        default=0.0, ge=0.0, description="Cumulative seconds used"
    )
    used_cost: float = Field(
        default=0.0, ge=0.0, description="Cumulative cost incurred"
    )

    @field_validator(
        "tool_calls", "tokens", "used_tool_calls", "used_tokens", mode="before"
    )
    @classmethod
    def _validate_int_non_negative(cls, v: Any) -> int:
        val = int(v) if v is not None else 0
        if val < 0:
            msg = "Budget amounts cannot be negative."
            raise ValueError(msg)
        return val

    @field_validator("seconds", "cost", "used_seconds", "used_cost", mode="before")
    @classmethod
    def _validate_float_non_negative(cls, v: Any) -> float:
        val = float(v) if v is not None else 0.0
        if val < 0.0:
            msg = "Budget amounts cannot be negative."
            raise ValueError(msg)
        return val

    def charge(
        self,
        tool_calls: int | Budget = 0,
        tokens: int = 0,
        seconds: float = 0.0,
        cost: float = 0.0,
        *,
        strict: bool = False,
    ) -> Budget:
        """Charge resource usage against the budget.

        Args:
            tool_calls: Tool calls used, or a Budget instance with all delta metrics.
            tokens: Token count consumed.
            seconds: Wall-clock seconds spent.
            cost: Estimated cost incurred in USD.
            strict: If True, raises BudgetExhaustedError if any limit is exceeded.

        Returns:
            self for chaining.

        Raises:
            ValueError: If any charge parameter is negative.
            BudgetExhaustedError: If strict is True and charge exceeds
                the remaining balance.
        """
        if isinstance(tool_calls, Budget):
            tc = tool_calls.tool_calls
            tok = tool_calls.tokens
            sec = tool_calls.seconds
            c = tool_calls.cost
        else:
            tc = tool_calls
            tok = tokens
            sec = seconds
            c = cost

        if tc < 0 or tok < 0 or sec < 0.0 or c < 0.0:
            msg = "Charge values cannot be negative."
            raise ValueError(msg)

        if strict and not self.can_charge(
            tool_calls=tc, tokens=tok, seconds=sec, cost=c
        ):
            exhausted = []
            rem = self.remaining()
            if self.tool_calls > 0 and tc > rem.tool_calls:
                exhausted.append(f"tool_calls ({tc} > {rem.tool_calls})")
            if self.tokens > 0 and tok > rem.tokens:
                exhausted.append(f"tokens ({tok} > {rem.tokens})")
            if self.seconds > 0 and sec > rem.seconds:
                exhausted.append(f"seconds ({sec} > {rem.seconds})")
            if self.cost > 0 and c > rem.cost:
                exhausted.append(f"cost (${c} > ${rem.cost})")
            msg = f"Budget exhausted: {', '.join(exhausted)}"
            raise BudgetExhaustedError(msg)

        self.used_tool_calls += tc
        self.used_tokens += tok
        self.used_seconds = round(self.used_seconds + sec, 6)
        self.used_cost = round(self.used_cost + c, 6)
        return self

    def remaining(self) -> Budget:
        """Calculate remaining balance across all resources, clamped at zero."""
        rem_tool_calls = max(0, self.tool_calls - self.used_tool_calls)
        rem_tokens = max(0, self.tokens - self.used_tokens)
        rem_seconds = max(0.0, round(self.seconds - self.used_seconds, 6))
        rem_cost = max(0.0, round(self.cost - self.used_cost, 6))
        return Budget(
            tool_calls=rem_tool_calls,
            tokens=rem_tokens,
            seconds=rem_seconds,
            cost=rem_cost,
        )

    def is_exhausted(self, resource: str | None = None) -> bool:
        """Check if the budget is exhausted for a specific resource or overall."""
        rem = self.remaining()
        if resource is not None:
            match resource:
                case "tool_calls":
                    return self.tool_calls > 0 and rem.tool_calls <= 0
                case "tokens":
                    return self.tokens > 0 and rem.tokens <= 0
                case "seconds":
                    return self.seconds > 0 and rem.seconds <= 0
                case "cost":
                    return self.cost > 0 and rem.cost <= 0
                case _:
                    msg = f"Unknown budget resource: '{resource}'"
                    raise KeyError(msg)

        return (
            (self.tool_calls > 0 and rem.tool_calls <= 0)
            or (self.tokens > 0 and rem.tokens <= 0)
            or (self.seconds > 0 and rem.seconds <= 0)
            or (self.cost > 0 and rem.cost <= 0)
        )

    @property
    def exhausted(self) -> bool:
        """Boolean property indicating whether any configured resource is exhausted."""
        return self.is_exhausted()

    @property
    def exhausted_resources(self) -> list[str]:
        """List of resource names that have reached zero remaining balance."""
        rem = self.remaining()
        depleted: list[str] = []
        if self.tool_calls > 0 and rem.tool_calls <= 0:
            depleted.append("tool_calls")
        if self.tokens > 0 and rem.tokens <= 0:
            depleted.append("tokens")
        if self.seconds > 0 and rem.seconds <= 0:
            depleted.append("seconds")
        if self.cost > 0 and rem.cost <= 0:
            depleted.append("cost")
        return depleted

    def can_charge(
        self,
        *,
        tool_calls: int = 0,
        tokens: int = 0,
        seconds: float = 0.0,
        cost: float = 0.0,
    ) -> bool:
        """Verify whether the requested charge amounts can be accommodated."""
        rem = self.remaining()
        if self.tool_calls > 0 and tool_calls > rem.tool_calls:
            return False
        if self.tokens > 0 and tokens > rem.tokens:
            return False
        if self.seconds > 0 and seconds > rem.seconds:
            return False
        if self.cost > 0 and cost > rem.cost:
            return False
        return True

    def __getitem__(self, item: str) -> Any:
        """Allow dict-like subscription access."""
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)
