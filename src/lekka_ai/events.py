"""Event bus and Server-Sent Events (SSE) data contracts for run streaming."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from lekka_ai.logging import get_logger

logger = get_logger(__name__)


class EventType(StrEnum):
    """Event types emitted during multi-agent research run execution."""

    RUN_STARTED = "run_started"
    PLAN_CREATED = "plan_created"
    TASK_STARTED = "task_started"
    TOOL_CALLED = "tool_called"
    TASK_COMPLETED = "task_completed"
    GAP_CHECK = "gap_check"
    VERIFICATION_DONE = "verification_done"
    REPORT_READY = "report_ready"
    RUN_FAILED = "run_failed"


class RunEvent(BaseModel):
    """Single discrete execution event emitted during a research run."""

    event: str = Field(..., description="Categorical event type name")
    run_id: str = Field(..., description="Unique run identifier")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 event timestamp",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured payload metadata associated with event",
    )

    def to_sse(self) -> str:
        """Format event according to Server-Sent Events (SSE) wire protocol."""
        payload = json.dumps(
            {
                "event": self.event,
                "run_id": self.run_id,
                "timestamp": self.timestamp,
                "data": self.data,
            }
        )
        return f"event: {self.event}\ndata: {payload}\n\n"


class EventBus:
    """In-process asyncio queue event bus managing dispatch and SSE streams."""

    def __init__(self) -> None:
        self._history: dict[str, list[RunEvent]] = {}
        self._subscribers: dict[str, set[asyncio.Queue[RunEvent | None]]] = {}
        self._finished: set[str] = set()
        self._lock = asyncio.Lock()

    async def emit(
        self,
        run_id: str,
        event_name: str | EventType,
        data: dict[str, Any] | None = None,
    ) -> RunEvent:
        """Emit an event to the run's history and active subscribers."""
        event_str = str(event_name)
        event = RunEvent(
            event=event_str,
            run_id=run_id,
            timestamp=datetime.now(UTC).isoformat(),
            data=data or {},
        )

        async with self._lock:
            if run_id not in self._history:
                self._history[run_id] = []
            self._history[run_id].append(event)

            if event_str in (EventType.REPORT_READY.value, EventType.RUN_FAILED.value):
                self._finished.add(run_id)

            subscribers = self._subscribers.get(run_id, set()).copy()

        logger.info("Emitted run event", run_id=run_id, event_type=event_str)

        for queue in subscribers:
            await queue.put(event)
            if event_str in (
                EventType.REPORT_READY.value,
                EventType.RUN_FAILED.value,
            ):
                await queue.put(None)

        return event

    def get_history(self, run_id: str) -> list[RunEvent]:
        """Retrieve all recorded historical events for a run."""
        return list(self._history.get(run_id, []))

    async def subscribe(self, run_id: str) -> AsyncIterator[RunEvent]:
        """Subscribe to a run's event stream, replaying history first."""
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()

        async with self._lock:
            if run_id not in self._subscribers:
                self._subscribers[run_id] = set()
            self._subscribers[run_id].add(queue)

            past_events = list(self._history.get(run_id, []))
            already_done = run_id in self._finished

        # Yield historical events in order
        for ev in past_events:
            yield ev

        if already_done:
            async with self._lock:
                if run_id in self._subscribers:
                    self._subscribers[run_id].discard(queue)
            return

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            async with self._lock:
                if run_id in self._subscribers:
                    self._subscribers[run_id].discard(queue)


# Singleton event bus instance
default_event_bus = EventBus()
