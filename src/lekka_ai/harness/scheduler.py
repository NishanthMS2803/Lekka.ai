"""Parallel task scheduler with asyncio concurrency, timeouts, and scratch isolation."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from lekka_ai.harness.permissions import AgentProfile
from lekka_ai.llm.base import LLMClient
from lekka_ai.logging import get_logger
from lekka_ai.orchestrator.state import PlannedTask, RunState
from lekka_ai.tools.base import ToolRegistry

logger = get_logger(__name__)

TaskRunnerCallable = Callable[[PlannedTask, Path], Awaitable[Any]]


class TaskScheduler:
    """Schedules and executes ready agent tasks in parallel with strict isolation.

    Features:
    - Semaphore-bounded asyncio concurrency.
    - Per-task timeout enforcement.
    - Independent scratch workspace per agent task.
    - Coordinated sibling task cancellation upon run-level abort.
    - Result merging exclusively handled by the harness into RunState.
    """

    def __init__(
        self,
        max_concurrency: int = 5,
        default_timeout: float | None = 60.0,
        scratch_base_dir: str | Path | None = None,
        cleanup_scratch: bool = True,
        llm: LLMClient | None = None,
        tool_registry: ToolRegistry | None = None,
        task_runner: TaskRunnerCallable | None = None,
        session: Session | None = None,
    ) -> None:
        """Initialize the parallel task scheduler.

        Args:
            max_concurrency: Maximum number of simultaneous running tasks.
            default_timeout: Default execution timeout in seconds per task.
            scratch_base_dir: Base directory for per-task isolated scratch spaces.
            cleanup_scratch: Whether to remove scratch directories after task ends.
            llm: Optional default LLM client for agent execution.
            tool_registry: ToolRegistry for specialist tool authorization.
            task_runner: Optional custom async callable (task, scratch_dir) -> result.
            session: Optional database session for persisting task statuses.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

        self.max_concurrency = max_concurrency
        self.default_timeout = default_timeout
        self.scratch_base_dir = (
            Path(scratch_base_dir)
            if scratch_base_dir
            else Path(tempfile.gettempdir()) / "lekka_scratch"
        )
        self.cleanup_scratch = cleanup_scratch
        self.llm = llm
        self.tool_registry = tool_registry
        self.task_runner = task_runner
        self.session = session

        self._abort_event = asyncio.Event()

    def abort(self) -> None:
        """Trigger a run-level abort, cancelling all active sibling tasks."""
        self._abort_event.set()

    def _create_scratch_dir(self, run_id: str, task_id: str) -> Path:
        """Create an isolated, dedicated scratch directory for a task."""
        clean_task_id = task_id.replace("/", "_").replace(" ", "_")
        scratch_dir = self.scratch_base_dir / run_id / clean_task_id
        scratch_dir.mkdir(parents=True, exist_ok=True)
        return scratch_dir

    def _cleanup_scratch_dir(self, scratch_dir: Path) -> None:
        """Remove the isolated scratch directory."""
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)

    async def _default_runner(
        self,
        task: PlannedTask,
        scratch_dir: Path,
    ) -> Any:
        """Default task runner invoking AgentRuntime with isolated tool view."""
        from lekka_ai.agents import load_agent_profile
        from lekka_ai.config import get_settings
        from lekka_ai.harness.models import AgentTask, Limits
        from lekka_ai.harness.permissions import ToolView
        from lekka_ai.harness.runtime import AgentRuntime
        from lekka_ai.llm.gemini import GeminiClient
        from lekka_ai.tools.base import default_registry

        profile: AgentProfile = load_agent_profile(task.agent_type)
        reg = self.tool_registry or default_registry
        tool_view = ToolView(registry=reg, profile=profile)

        agent_task = AgentTask(
            task_id=task.task_id,
            agent_type=task.agent_type,
            instruction=task.instruction,
            as_of_date=task.metadata.get("as_of_date"),
        )
        agent_task.context_facts.append({"scratch_dir": str(scratch_dir)})

        limits = Limits(
            max_seconds=task.metadata.get(
                "max_seconds", profile.default_limits.max_seconds
            ),
            max_iterations=task.metadata.get(
                "max_iterations", profile.default_limits.max_iterations
            ),
            max_tokens=task.metadata.get(
                "max_tokens", profile.default_limits.max_tokens
            ),
            max_tool_calls=task.metadata.get(
                "max_tool_calls", profile.default_limits.max_tool_calls
            ),
        )

        llm = self.llm
        if llm is None:
            settings = get_settings()
            llm = GeminiClient(
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_MODEL,
            )

        run_result = await AgentRuntime.run(
            task=agent_task,
            profile=profile,
            llm=llm,
            tool_view=tool_view,
            limits=limits,
            session=self.session,
        )
        return run_result.submission or run_result

    async def _execute_single_task(
        self,
        task: PlannedTask,
        state: RunState,
        semaphore: asyncio.Semaphore,
        combined_abort_event: asyncio.Event,
        task_timeout: float | None,
    ) -> None:
        """Execute a single task with concurrency limiting, isolation, and timeout."""
        async with semaphore:
            if combined_abort_event.is_set():
                state.update_task_status(
                    task.task_id,
                    "cancelled",
                    session=self.session,
                    error="Run aborted prior to execution",
                )
                return

            scratch_dir = self._create_scratch_dir(state.run_id, task.task_id)
            state.update_task_status(
                task.task_id,
                "running",
                session=self.session,
            )

            runner = self.task_runner or self._default_runner
            effective_timeout = (
                task_timeout
                if task_timeout is not None
                else task.metadata.get("timeout", self.default_timeout)
            )

            try:
                coro = runner(task, scratch_dir)
                if effective_timeout is not None and effective_timeout > 0:
                    result = await asyncio.wait_for(
                        coro,
                        timeout=effective_timeout,
                    )
                else:
                    result = await coro

                # Harness merges results into RunState
                state.update_task_status(
                    task.task_id,
                    "completed",
                    session=self.session,
                    result=result,
                )
            except TimeoutError:
                err_msg = f"Task '{task.task_id}' timed out after {effective_timeout}s"
                logger.warning(err_msg, task_id=task.task_id)
                state.update_task_status(
                    task.task_id,
                    "failed",
                    session=self.session,
                    error=err_msg,
                )
            except asyncio.CancelledError:
                logger.info(
                    "Task cancelled due to run abort or timeout",
                    task_id=task.task_id,
                )
                state.update_task_status(
                    task.task_id,
                    "cancelled",
                    session=self.session,
                    error="Task was cancelled",
                )
                raise
            except Exception as exc:
                err_msg = f"Task '{task.task_id}' failed: {exc}"
                logger.error(err_msg, task_id=task.task_id, exc_info=True)
                state.update_task_status(
                    task.task_id,
                    "failed",
                    session=self.session,
                    error=err_msg,
                )
            finally:
                if self.cleanup_scratch:
                    self._cleanup_scratch_dir(scratch_dir)

    async def run_tasks(
        self,
        tasks: list[PlannedTask],
        state: RunState,
        *,
        timeout: float | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> dict[str, Any]:
        """Run tasks concurrently using asyncio, merging results into RunState.

        Args:
            tasks: List of ready PlannedTask instances to execute.
            state: Shared RunState where execution status and results are merged.
            timeout: Optional override for per-task execution timeout.
            abort_event: Optional external cancellation signal.

        Returns:
            Mapping of task_id to task result for all completed tasks in this round.
        """
        if not tasks:
            return state.results

        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Build combined abort monitor
        combined_abort = asyncio.Event()

        async def _forward_abort(src: asyncio.Event) -> None:
            await src.wait()
            combined_abort.set()

        abort_tasks: list[asyncio.Task[None]] = []
        abort_tasks.append(asyncio.create_task(_forward_abort(self._abort_event)))
        if abort_event is not None:
            abort_tasks.append(asyncio.create_task(_forward_abort(abort_event)))

        # Create asyncio task per planned task
        async_tasks: list[asyncio.Task[None]] = []
        for task in tasks:
            t = asyncio.create_task(
                self._execute_single_task(
                    task=task,
                    state=state,
                    semaphore=semaphore,
                    combined_abort_event=combined_abort,
                    task_timeout=timeout,
                )
            )
            async_tasks.append(t)

        # Monitor abort event to cancel active sibling tasks
        async def _abort_watcher() -> None:
            await combined_abort.wait()
            for t in async_tasks:
                if not t.done():
                    t.cancel()

        watcher_task = asyncio.create_task(_abort_watcher())

        # Wait for all task executions to settle
        try:
            await asyncio.gather(*async_tasks, return_exceptions=True)
        finally:
            watcher_task.cancel()
            for at in abort_tasks:
                at.cancel()
            # Clean up pending tasks if any remain
            for t in async_tasks:
                if not t.done():
                    t.cancel()

        return {
            t.task_id: state.results[t.task_id]
            for t in tasks
            if t.task_id in state.results
        }


async def run_ready_tasks_parallel(
    tasks: list[PlannedTask],
    state: RunState,
    *,
    max_concurrency: int = 5,
    timeout: float | None = 60.0,
    task_runner: TaskRunnerCallable | None = None,
    abort_event: asyncio.Event | None = None,
    session: Session | None = None,
) -> dict[str, Any]:
    """Convenience helper to run ready tasks with a TaskScheduler instance."""
    scheduler = TaskScheduler(
        max_concurrency=max_concurrency,
        default_timeout=timeout,
        task_runner=task_runner,
        session=session,
    )
    return await scheduler.run_tasks(
        tasks=tasks,
        state=state,
        timeout=timeout,
        abort_event=abort_event,
    )
