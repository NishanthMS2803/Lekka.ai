"""Unit and integration tests for the parallel task scheduler."""

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from lekka_ai.harness.scheduler import TaskScheduler, run_ready_tasks_parallel
from lekka_ai.orchestrator.state import Plan, PlannedTask, RunState


@pytest.fixture
def base_run_state() -> RunState:
    """Fixture providing an empty RunState."""
    return RunState(
        run_id="run_test_sched_01",
        question="What is the outlook for AAPL?",
        plan=Plan(),
    )


@pytest.mark.asyncio
async def test_three_agents_finish_in_parallel_time(base_run_state: RunState) -> None:
    """Verify three fake agents with different durations finish in parallel time.

    Sequential duration would be 0.20 + 0.35 + 0.25 = 0.80s.
    Parallel execution should finish in approximately max(0.20, 0.35, 0.25) ~ 0.35s.
    """
    durations = {
        "task_market": 0.20,
        "task_valuation": 0.35,
        "task_news": 0.25,
    }

    async def fake_runner(task: PlannedTask, scratch_dir: Path) -> dict[str, Any]:
        sleep_time = durations[task.task_id]
        await asyncio.sleep(sleep_time)
        return {"agent": task.agent_type, "duration": sleep_time}

    tasks = [
        PlannedTask(
            task_id="task_market",
            agent_type="market",
            instruction="Market price action and volatility",
            priority=2,
        ),
        PlannedTask(
            task_id="task_valuation",
            agent_type="valuation",
            instruction="DCF and multiple valuation",
            priority=2,
        ),
        PlannedTask(
            task_id="task_news",
            agent_type="news",
            instruction="Catalysts and SEC filings",
            priority=2,
        ),
    ]

    for t in tasks:
        base_run_state.plan.add_task(t)

    scheduler = TaskScheduler(
        max_concurrency=5,
        default_timeout=5.0,
        task_runner=fake_runner,
    )

    t0 = time.perf_counter()
    results = await scheduler.run_tasks(tasks, base_run_state)
    elapsed = time.perf_counter() - t0

    # Verification of parallelism:
    # 1. Total elapsed time must be noticeably less than the sequential sum (0.80s)
    #    Allow reasonable buffer for system scheduler jitter (e.g. 0.60s max)
    assert elapsed < 0.60, f"Execution took {elapsed:.3f}s, expected parallel (<0.60s)"
    assert elapsed >= 0.30, f"Execution took {elapsed:.3f}s, expected at least 0.30s"

    # 2. All tasks must be completed in RunState
    assert base_run_state.task_statuses["task_market"] == "completed"
    assert base_run_state.task_statuses["task_valuation"] == "completed"
    assert base_run_state.task_statuses["task_news"] == "completed"

    # 3. Results must be merged into RunState
    assert results["task_market"] == {"agent": "market", "duration": 0.20}
    assert results["task_valuation"] == {"agent": "valuation", "duration": 0.35}
    assert results["task_news"] == {"agent": "news", "duration": 0.25}
    assert base_run_state.results == results


@pytest.mark.asyncio
async def test_timeout_cancels_one_without_killing_others(
    base_run_state: RunState,
) -> None:
    """Verify a task timeout cancels only the slow task while siblings complete."""

    async def fake_runner(task: PlannedTask, scratch_dir: Path) -> dict[str, Any]:
        if task.task_id == "task_slow":
            # Hangs longer than its configured timeout
            await asyncio.sleep(2.0)
            return {"status": "slow_done"}

        if task.task_id == "task_fast_1":
            await asyncio.sleep(0.08)
            return {"status": "fast_1_done"}

        if task.task_id == "task_fast_2":
            await asyncio.sleep(0.12)
            return {"status": "fast_2_done"}

        return {"status": "unknown"}

    tasks = [
        PlannedTask(
            task_id="task_fast_1",
            agent_type="market",
            instruction="Fast market computation",
        ),
        PlannedTask(
            task_id="task_slow",
            agent_type="valuation",
            instruction="Slow valuation hanging task",
            metadata={"timeout": 0.20},  # Explicit 200ms timeout
        ),
        PlannedTask(
            task_id="task_fast_2",
            agent_type="news",
            instruction="Fast news extraction",
        ),
    ]

    for t in tasks:
        base_run_state.plan.add_task(t)

    scheduler = TaskScheduler(
        max_concurrency=5,
        default_timeout=5.0,
        task_runner=fake_runner,
    )

    t0 = time.perf_counter()
    results = await scheduler.run_tasks(tasks, base_run_state)
    elapsed = time.perf_counter() - t0

    # Total elapsed should be bounded around ~0.25s (not 2.0s!)
    assert elapsed < 0.60, f"Elapsed time was {elapsed:.3f}s"

    # Fast tasks must complete successfully
    assert base_run_state.task_statuses["task_fast_1"] == "completed"
    assert base_run_state.task_statuses["task_fast_2"] == "completed"
    assert results["task_fast_1"] == {"status": "fast_1_done"}
    assert results["task_fast_2"] == {"status": "fast_2_done"}

    # Slow task must be marked failed due to timeout
    assert base_run_state.task_statuses["task_slow"] == "failed"
    assert "task_slow" not in results

    # Error must be recorded in RunState
    assert any("timed out after 0.2" in err for err in base_run_state.errors)


@pytest.mark.asyncio
async def test_sibling_cancellation_on_run_abort(
    base_run_state: RunState,
) -> None:
    """Verify abort cancels all currently running sibling tasks."""
    cancelled_tasks: list[str] = []

    async def fake_runner(task: PlannedTask, scratch_dir: Path) -> dict[str, Any]:
        try:
            await asyncio.sleep(2.0)
            return {"status": "done"}
        except asyncio.CancelledError:
            cancelled_tasks.append(task.task_id)
            raise

    tasks = [
        PlannedTask(
            task_id="task_a",
            agent_type="market",
            instruction="Task A",
        ),
        PlannedTask(
            task_id="task_b",
            agent_type="valuation",
            instruction="Task B",
        ),
        PlannedTask(
            task_id="task_c",
            agent_type="news",
            instruction="Task C",
        ),
    ]

    for t in tasks:
        base_run_state.plan.add_task(t)

    scheduler = TaskScheduler(
        max_concurrency=5,
        default_timeout=5.0,
        task_runner=fake_runner,
    )

    abort_event = asyncio.Event()

    async def trigger_abort_later() -> None:
        await asyncio.sleep(0.08)
        abort_event.set()

    asyncio.create_task(trigger_abort_later())

    t0 = time.perf_counter()
    await scheduler.run_tasks(tasks, base_run_state, abort_event=abort_event)
    elapsed = time.perf_counter() - t0

    # Must abort quickly well before the 2.0s task sleep
    assert elapsed < 0.40, f"Aborted in {elapsed:.3f}s"

    # All tasks should be cancelled in RunState
    assert base_run_state.task_statuses["task_a"] == "cancelled"
    assert base_run_state.task_statuses["task_b"] == "cancelled"
    assert base_run_state.task_statuses["task_c"] == "cancelled"
    assert len(cancelled_tasks) == 3


@pytest.mark.asyncio
async def test_scratch_space_isolation(base_run_state: RunState) -> None:
    """Verify each agent runs in its own isolated scratch workspace."""

    observed_scratch_dirs: dict[str, Path] = {}

    async def isolated_runner(task: PlannedTask, scratch_dir: Path) -> dict[str, Any]:
        observed_scratch_dirs[task.task_id] = scratch_dir
        # Write file with task-specific content into scratch space
        file_path = scratch_dir / "scratch_test.txt"
        file_path.write_text(f"Content for {task.task_id}", encoding="utf-8")
        await asyncio.sleep(0.05)
        # Read back to ensure no sibling overwrite
        read_back = file_path.read_text(encoding="utf-8")
        assert read_back == f"Content for {task.task_id}"
        return {"file_ok": True}

    tasks = [
        PlannedTask(
            task_id="task_iso_1",
            agent_type="market",
            instruction="Isolation task 1",
        ),
        PlannedTask(
            task_id="task_iso_2",
            agent_type="valuation",
            instruction="Isolation task 2",
        ),
    ]

    for t in tasks:
        base_run_state.plan.add_task(t)

    # Don't cleanup scratch immediately so we can inspect uniqueness
    scheduler = TaskScheduler(
        max_concurrency=5,
        cleanup_scratch=False,
        task_runner=isolated_runner,
    )

    await scheduler.run_tasks(tasks, base_run_state)

    # Scratch directories must be strictly distinct
    dir_1 = observed_scratch_dirs["task_iso_1"]
    dir_2 = observed_scratch_dirs["task_iso_2"]
    assert dir_1 != dir_2
    assert "task_iso_1" in str(dir_1)
    assert "task_iso_2" in str(dir_2)

    # Clean up manually
    scheduler._cleanup_scratch_dir(dir_1)
    scheduler._cleanup_scratch_dir(dir_2)


@pytest.mark.asyncio
async def test_max_concurrency_semaphore_throttling(
    base_run_state: RunState,
) -> None:
    """Verify semaphore enforces max_concurrency by queuing extra tasks."""
    max_active = 0
    active_now = 0
    lock = asyncio.Lock()

    async def counting_runner(task: PlannedTask, scratch_dir: Path) -> dict[str, Any]:
        nonlocal max_active, active_now
        async with lock:
            active_now += 1
            if active_now > max_active:
                max_active = active_now

        await asyncio.sleep(0.10)

        async with lock:
            active_now -= 1
        return {"ok": True}

    tasks = [
        PlannedTask(
            task_id=f"task_throttled_{i}",
            agent_type="analyst",
            instruction=f"Throttled task {i}",
        )
        for i in range(4)
    ]

    for t in tasks:
        base_run_state.plan.add_task(t)

    # Concurrency limit of 2: 4 tasks should run in batches of 2
    scheduler = TaskScheduler(
        max_concurrency=2,
        task_runner=counting_runner,
    )

    t0 = time.perf_counter()
    await scheduler.run_tasks(tasks, base_run_state)
    elapsed = time.perf_counter() - t0

    # Max concurrent executions must not exceed 2
    assert max_active == 2
    # 2 waves of 0.10s should take at least ~0.20s
    assert elapsed >= 0.18


@pytest.mark.asyncio
async def test_run_ready_tasks_parallel_helper(base_run_state: RunState) -> None:
    """Verify run_ready_tasks_parallel convenience wrapper."""

    async def simple_runner(task: PlannedTask, scratch_dir: Path) -> str:
        return f"result_{task.task_id}"

    task = PlannedTask(
        task_id="task_helper",
        agent_type="market",
        instruction="Helper test task",
    )
    base_run_state.plan.add_task(task)

    results = await run_ready_tasks_parallel(
        tasks=[task],
        state=base_run_state,
        task_runner=simple_runner,
    )

    assert results == {"task_helper": "result_task_helper"}
    assert base_run_state.task_statuses["task_helper"] == "completed"
