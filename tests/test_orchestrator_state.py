"""Tests for orchestrator Plan, PlannedTask, RunState, and DAG logic."""

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from lekka_ai.db.base import Base
from lekka_ai.db.models import Run, Task
from lekka_ai.orchestrator.state import (
    CycleDetectedError,
    Plan,
    PlannedTask,
    RunState,
    UnknownDependencyError,
    detect_cycles,
    persist_task_status,
    ready_tasks,
    to_uuid,
    topological_sort,
)


@pytest.fixture
def sqlite_session() -> Generator[Session, None, None]:
    """Provide a fresh in-memory SQLite session with initialized tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_planned_task_creation_and_defaults() -> None:
    """Verify PlannedTask fields, defaults, and custom metadata."""
    task = PlannedTask(
        task_id="task_mkt_01",
        agent_type="market",
        instruction="Analyze AAPL market momentum and volatility.",
    )
    assert task.task_id == "task_mkt_01"
    assert task.agent_type == "market"
    assert task.instruction == "Analyze AAPL market momentum and volatility."
    assert task.depends_on == []
    assert task.priority == 0
    assert task.expected_cost == 0.0

    task_custom = PlannedTask(
        task_id="task_val_01",
        agent_type="valuation",
        instruction="Run DCF model for AAPL.",
        depends_on=["task_mkt_01"],
        priority=10,
        expected_cost=0.05,
        metadata={"ticker": "AAPL", "discount_rate": 0.09},
    )
    assert task_custom.depends_on == ["task_mkt_01"]
    assert task_custom.priority == 10
    assert task_custom.expected_cost == 0.05
    assert task_custom.metadata["ticker"] == "AAPL"


def test_plan_dag_validation_and_lookups() -> None:
    """Verify Plan indexing, iteration, and dangling dependency checks."""
    t1 = PlannedTask(
        task_id="task_1",
        agent_type="market",
        instruction="Task 1",
    )
    t2 = PlannedTask(
        task_id="task_2",
        agent_type="valuation",
        instruction="Task 2",
        depends_on=["task_1"],
    )

    plan = Plan(tasks=[t1, t2])
    assert len(plan) == 2
    assert plan.task_ids == ["task_1", "task_2"]
    assert plan[0].task_id == "task_1"
    assert plan["task_2"].agent_type == "valuation"
    assert plan.get_task("task_nonexistent") is None

    # Valid DAG
    plan.validate_dag()

    # Dangling dependency
    t3_dangling = PlannedTask(
        task_id="task_3",
        agent_type="news",
        instruction="Task 3",
        depends_on=["unknown_task_99"],
    )
    plan.add_task(t3_dangling)
    with pytest.raises(UnknownDependencyError, match="depends on unknown task"):
        plan.validate_dag()


def test_ready_tasks_independent_and_priority() -> None:
    """Verify ready_tasks returns independent pending tasks ordered by priority."""
    t_low = PlannedTask(
        task_id="task_low",
        agent_type="news",
        instruction="Low priority news check",
        priority=1,
    )
    t_high = PlannedTask(
        task_id="task_high",
        agent_type="valuation",
        instruction="High priority valuation",
        priority=10,
    )
    t_med = PlannedTask(
        task_id="task_med",
        agent_type="market",
        instruction="Medium priority market check",
        priority=5,
    )

    plan = Plan(tasks=[t_low, t_high, t_med])
    state = RunState(question="Evaluate AAPL", plan=plan)

    # All three independent tasks are pending -> all ready, sorted by priority
    ready = ready_tasks(plan, state)
    assert [t.task_id for t in ready] == ["task_high", "task_med", "task_low"]

    # Mark high as running -> only med and low ready
    state.update_task_status("task_high", "running")
    ready_after = ready_tasks(plan, state)
    assert [t.task_id for t in ready_after] == ["task_med", "task_low"]


def test_ready_tasks_sequential_and_diamond_dependencies() -> None:
    """Verify ready_tasks respects completion of dependencies in a diamond graph."""
    # Diamond graph:
    #      task_a
    #     /      \
    # task_b    task_c
    #     \      /
    #      task_d
    t_a = PlannedTask(task_id="task_a", agent_type="news", instruction="A")
    t_b = PlannedTask(
        task_id="task_b",
        agent_type="market",
        instruction="B",
        depends_on=["task_a"],
    )
    t_c = PlannedTask(
        task_id="task_c",
        agent_type="valuation",
        instruction="C",
        depends_on=["task_a"],
    )
    t_d = PlannedTask(
        task_id="task_d",
        agent_type="analyst",
        instruction="D",
        depends_on=["task_b", "task_c"],
    )

    plan = Plan(tasks=[t_a, t_b, t_c, t_d])
    state = RunState(question="Synthesize research", plan=plan)

    # 1. Initially only task_a is ready
    assert [t.task_id for t in ready_tasks(plan, state)] == ["task_a"]

    # 2. task_a runs and completes -> task_b and task_c become ready
    state.update_task_status("task_a", "completed")
    ready_step2 = [t.task_id for t in ready_tasks(plan, state)]
    assert set(ready_step2) == {"task_b", "task_c"}

    # 3. task_b completes, but task_c is running -> task_d is NOT yet ready
    state.update_task_status("task_b", "completed")
    state.update_task_status("task_c", "running")
    assert ready_tasks(plan, state) == []

    # 4. task_c completes -> task_d is ready
    state.update_task_status("task_c", "completed")
    assert [t.task_id for t in ready_tasks(plan, state)] == ["task_d"]

    # 5. task_d completes -> no tasks left ready
    state.update_task_status("task_d", "completed")
    assert ready_tasks(plan, state) == []
    assert state.is_complete() is True


def test_ready_tasks_blocked_by_failed_dependency() -> None:
    """Verify dependent task is not ready if upstream dependency fails."""
    t1 = PlannedTask(task_id="task_1", agent_type="market", instruction="Market")
    t2 = PlannedTask(
        task_id="task_2",
        agent_type="valuation",
        instruction="Valuation",
        depends_on=["task_1"],
    )

    plan = Plan(tasks=[t1, t2])
    state = RunState(question="Analysis", plan=plan)

    state.update_task_status("task_1", "failed", error="API timeout")
    ready = ready_tasks(plan, state)
    assert ready == []


def test_cycle_detection_simple_two_node_cycle() -> None:
    """Verify circular dependency is detected and raises CycleDetectedError."""
    t1 = PlannedTask(
        task_id="task_1",
        agent_type="market",
        instruction="Task 1",
        depends_on=["task_2"],
    )
    t2 = PlannedTask(
        task_id="task_2",
        agent_type="valuation",
        instruction="Task 2",
        depends_on=["task_1"],
    )

    plan = Plan(tasks=[t1, t2])
    cycle = detect_cycles(plan)
    assert cycle is not None
    assert set(cycle) == {"task_1", "task_2"}

    with pytest.raises(CycleDetectedError, match="Dependency cycle detected"):
        plan.validate_dag()

    with pytest.raises(CycleDetectedError, match="Dependency cycle detected"):
        topological_sort(plan)


def test_cycle_detection_three_node_and_self_cycle() -> None:
    """Verify 3-node cycle and direct self-dependency are detected."""
    # 3-node cycle: A -> B -> C -> A
    t_a = PlannedTask(
        task_id="task_a", agent_type="news", instruction="A", depends_on=["task_c"]
    )
    t_b = PlannedTask(
        task_id="task_b", agent_type="market", instruction="B", depends_on=["task_a"]
    )
    t_c = PlannedTask(
        task_id="task_c",
        agent_type="valuation",
        instruction="C",
        depends_on=["task_b"],
    )

    plan_3 = Plan(tasks=[t_a, t_b, t_c])
    assert detect_cycles(plan_3) is not None
    with pytest.raises(CycleDetectedError):
        plan_3.validate_dag()

    # Self-cycle: A -> A
    t_self = PlannedTask(
        task_id="task_self",
        agent_type="analyst",
        instruction="Self",
        depends_on=["task_self"],
    )
    plan_self = Plan(tasks=[t_self])
    assert detect_cycles(plan_self) is not None
    with pytest.raises(CycleDetectedError):
        plan_self.validate_dag()


def test_topological_sort_diamond_graph() -> None:
    """Verify topological sort orders diamond graph respecting dependencies."""
    t_a = PlannedTask(task_id="task_a", agent_type="news", instruction="A")
    t_b = PlannedTask(
        task_id="task_b", agent_type="market", instruction="B", depends_on=["task_a"]
    )
    t_c = PlannedTask(
        task_id="task_c",
        agent_type="valuation",
        instruction="C",
        depends_on=["task_a"],
    )
    t_d = PlannedTask(
        task_id="task_d",
        agent_type="analyst",
        instruction="D",
        depends_on=["task_b", "task_c"],
    )

    plan = Plan(tasks=[t_d, t_c, t_b, t_a])  # initially unordered
    ordered = topological_sort(plan)
    ordered_ids = [t.task_id for t in ordered]

    assert ordered_ids[0] == "task_a"
    assert set(ordered_ids[1:3]) == {"task_b", "task_c"}
    assert ordered_ids[3] == "task_d"


def test_run_state_task_status_updates_and_completion() -> None:
    """Verify RunState updates statuses, results, errors, and checks completion."""
    t1 = PlannedTask(task_id="task_1", agent_type="market", instruction="T1")
    t2 = PlannedTask(task_id="task_2", agent_type="valuation", instruction="T2")

    plan = Plan(tasks=[t1, t2])
    state = RunState(
        question="What is Apple fair value?",
        parsed_query={"ticker": "AAPL", "intent": "valuation"},
        plan=plan,
    )

    assert state.get_task_status("task_1") == "pending"
    assert state.is_complete() is False

    # Update task 1
    state.update_task_status("task_1", "running")
    assert state.get_task_status("task_1") == "running"

    state.update_task_status(
        "task_1",
        "completed",
        result={"trend": "bullish", "beta": 1.15},
    )
    assert state.get_task_status("task_1") == "completed"
    assert state.results["task_1"]["beta"] == 1.15
    assert state.is_complete() is False

    # Update task 2 to failed with error
    state.update_task_status("task_2", "failed", error="Execution timeout")
    assert state.get_task_status("task_2") == "failed"
    assert "Execution timeout" in state.errors
    assert state.is_complete() is True


def test_persist_task_status_to_database(sqlite_session: Session) -> None:
    """Verify persist_task_status persists run and task records to SQL tables."""
    run_id = "test_run_uuid_01"
    task_id = "test_task_uuid_01"

    # 1. Initial creation via persist_task_status
    db_task = persist_task_status(
        sqlite_session,
        run_id=run_id,
        task_id=task_id,
        status="running",
        agent_type="market",
        instruction="Evaluate market action",
    )
    assert db_task.status == "running"
    assert db_task.agent_type == "market"

    # Verify parent Run was created
    db_run = sqlite_session.get(Run, to_uuid(run_id))
    assert db_run is not None
    assert db_run.status == "running"

    # 2. Update status to completed
    updated_task = persist_task_status(
        sqlite_session,
        run_id=run_id,
        task_id=task_id,
        status="completed",
    )
    assert updated_task.status == "completed"

    # 3. Integration through RunState.update_task_status(session=...)
    plan = Plan(
        tasks=[
            PlannedTask(
                task_id=task_id,
                agent_type="market",
                instruction="Evaluate market action",
            )
        ]
    )
    state = RunState(
        run_id=run_id,
        question="AAPL review",
        plan=plan,
    )
    state.update_task_status(
        task_id=task_id,
        status="completed",
        session=sqlite_session,
        result={"latest_close": 228.20},
    )

    queried_task = sqlite_session.get(Task, to_uuid(task_id))
    assert queried_task is not None
    assert queried_task.status == "completed"
    assert state.results[task_id]["latest_close"] == 228.20
