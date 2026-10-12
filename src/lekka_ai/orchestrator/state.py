"""Orchestrator plan, shared-state models, and dependency resolution."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from lekka_ai.db.models import Run, Task


class CycleDetectedError(ValueError):
    """Raised when a circular dependency is detected in a plan."""


class UnknownDependencyError(ValueError):
    """Raised when a task depends on a task_id not present in the plan."""


def to_uuid(val: str | uuid.UUID) -> uuid.UUID:
    """Coerce string or UUID to uuid.UUID instance, using UUID5 for arbitrary slugs."""
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(val)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_DNS, str(val))


class PlannedTask(BaseModel):
    """Single sub-task planned for delegation to a specialist agent."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(..., description="Unique task identifier slug or UUID")
    agent_type: str = Field(..., description="Target specialist agent type")
    instruction: str = Field(..., description="Actionable instruction prompt")
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of task_ids that must complete prior to execution",
    )
    priority: int = Field(
        default=0,
        description="Scheduling priority (higher values scheduled first)",
    )
    expected_cost: float = Field(
        default=0.0,
        description="Estimated execution budget or dollar cost",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional execution context and arguments",
    )


class Plan(BaseModel):
    """Execution plan containing planned tasks with dependency DAG semantics."""

    model_config = ConfigDict(extra="allow")

    tasks: list[PlannedTask] = Field(
        default_factory=list,
        description="List of planned specialist tasks",
    )

    def add_task(self, task: PlannedTask) -> None:
        """Add a planned task to the plan."""
        self.tasks.append(task)

    def get_task(self, task_id: str) -> PlannedTask | None:
        """Find a planned task by task_id."""
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    @property
    def task_ids(self) -> list[str]:
        """Return list of all task_ids in plan."""
        return [t.task_id for t in self.tasks]

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Any:
        return iter(self.tasks)

    def __getitem__(self, item: int | str) -> PlannedTask:
        if isinstance(item, int):
            return self.tasks[item]
        task = self.get_task(item)
        if task is None:
            raise KeyError(f"Task '{item}' not found in plan")
        return task

    def validate_dag(self) -> None:
        """Validate that all dependencies exist and no cycles exist."""
        known_ids = set(self.task_ids)
        for t in self.tasks:
            for dep in t.depends_on:
                if dep not in known_ids:
                    raise UnknownDependencyError(
                        f"Task '{t.task_id}' depends on unknown task '{dep}'"
                    )

        cycle = detect_cycles(self)
        if cycle:
            cycle_str = " -> ".join(cycle)
            raise CycleDetectedError(f"Dependency cycle detected: {cycle_str}")


class RunState(BaseModel):
    """Shared execution state for a multi-round orchestrator run."""

    model_config = ConfigDict(extra="allow")

    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique execution run UUID string",
    )
    question: str = Field(..., description="Initial user question or research prompt")
    parsed_query: dict[str, Any] | None = Field(
        default=None,
        description="Structured query decomposition (ticker, intent, dates)",
    )
    plan: Plan = Field(
        default_factory=Plan,
        description="Current execution plan",
    )
    task_statuses: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of task_id to current execution status",
    )
    results: dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of task_id to structured result or submission",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Unresolved questions identified during research",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Errors or warning messages encountered during run",
    )
    round_number: int = Field(
        default=0,
        description="Current orchestration loop round count",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of run initialization",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of last state modification",
    )

    def get_task_status(self, task_id: str) -> str:
        """Return the current status for task_id, defaulting to 'pending'."""
        return self.task_statuses.get(task_id, "pending")

    def update_task_status(
        self,
        task_id: str,
        status: str,
        session: Session | None = None,
        result: Any | None = None,
        error: str | None = None,
    ) -> None:
        """Update task status in memory and optionally persist to tasks table."""
        self.task_statuses[task_id] = status
        self.updated_at = datetime.now(UTC)

        if result is not None:
            self.results[task_id] = result
        if error is not None:
            self.errors.append(error)

        if session is not None:
            planned = self.plan.get_task(task_id)
            agent_type = planned.agent_type if planned else "general"
            instruction = planned.instruction if planned else f"Task {task_id}"
            persist_task_status(
                session=session,
                run_id=self.run_id,
                task_id=task_id,
                status=status,
                agent_type=agent_type,
                instruction=instruction,
            )

    def is_complete(self) -> bool:
        """Return True if all planned tasks have reached terminal status."""
        if not self.plan.tasks:
            return False
        terminal = {"completed", "failed", "cancelled"}
        return all(self.get_task_status(t.task_id) in terminal for t in self.plan.tasks)


def persist_task_status(
    session: Session,
    *,
    run_id: str | uuid.UUID,
    task_id: str | uuid.UUID,
    status: str,
    agent_type: str = "general",
    instruction: str = "",
) -> Task:
    """Persist task status update to the database tasks table."""
    task_uuid = to_uuid(task_id)
    run_uuid = to_uuid(run_id)

    # Ensure parent run exists
    run = session.get(Run, run_uuid)
    if run is None:
        run = Run(
            id=run_uuid,
            question=f"Run {run_id}",
            status="running",
        )
        session.add(run)
        session.flush()

    task = session.get(Task, task_uuid)
    if task is None:
        task = Task(
            id=task_uuid,
            run_id=run_uuid,
            agent_type=agent_type,
            instruction=instruction or f"Task {task_id}",
            status=status,
        )
        session.add(task)
    else:
        task.status = status

    session.commit()
    session.refresh(task)
    return task


def ready_tasks(
    plan: Plan,
    state_or_statuses: RunState | dict[str, str],
) -> list[PlannedTask]:
    """Return tasks whose dependencies are completed and status is pending.

    Tasks are ordered by priority descending, breaking ties with plan order.
    """
    if isinstance(state_or_statuses, RunState):
        statuses = state_or_statuses.task_statuses
    else:
        statuses = state_or_statuses

    ready: list[PlannedTask] = []
    for task in plan.tasks:
        current_status = statuses.get(task.task_id, "pending")
        if current_status != "pending":
            continue

        # Check if all dependencies are 'completed'
        deps_satisfied = True
        for dep in task.depends_on:
            if statuses.get(dep) != "completed":
                deps_satisfied = False
                break

        if deps_satisfied:
            ready.append(task)

    # Sort primarily by priority descending
    ready.sort(key=lambda t: t.priority, reverse=True)
    return ready


def detect_cycles(plan: Plan) -> list[str] | None:
    """Detect cycles in plan dependency graph using depth-first search.

    Returns:
        List of task_ids forming a cycle if detected, or None if DAG is acyclic.
    """
    # Build graph: dep -> tasks that depend on dep
    # or task -> tasks that task depends on
    # Standard: task -> dependencies
    graph: dict[str, list[str]] = {t.task_id: list(t.depends_on) for t in plan.tasks}

    visited: set[str] = set()
    rec_stack: list[str] = []

    def dfs(node: str) -> list[str] | None:
        visited.add(node)
        rec_stack.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in graph:
                continue
            if neighbor not in visited:
                cycle = dfs(neighbor)
                if cycle:
                    return cycle
            elif neighbor in rec_stack:
                # Cycle found: slice rec_stack from neighbor
                idx = rec_stack.index(neighbor)
                return rec_stack[idx:] + [neighbor]

        rec_stack.pop()
        return None

    for task_id in graph:
        if task_id not in visited:
            cycle = dfs(task_id)
            if cycle:
                return cycle

    return None


def topological_sort(plan: Plan) -> list[PlannedTask]:
    """Return tasks in topologically sorted dependency order.

    Raises:
        CycleDetectedError: If circular dependencies exist in plan.
    """
    plan.validate_dag()

    # In-degree based topological sort (Kahn's algorithm)
    task_map = {t.task_id: t for t in plan.tasks}
    in_degree: dict[str, int] = {t.task_id: len(t.depends_on) for t in plan.tasks}
    dependents: dict[str, list[str]] = {t.task_id: [] for t in plan.tasks}

    for t in plan.tasks:
        for dep in t.depends_on:
            if dep in dependents:
                dependents[dep].append(t.task_id)

    # Queue of nodes with no incoming dependencies
    queue: list[str] = [tid for tid, deg in in_degree.items() if deg == 0]
    sorted_tasks: list[PlannedTask] = []

    while queue:
        # Sort queue by priority descending to respect priority in ties
        queue.sort(key=lambda tid: task_map[tid].priority, reverse=True)
        curr_id = queue.pop(0)
        sorted_tasks.append(task_map[curr_id])

        for dep_id in dependents.get(curr_id, []):
            in_degree[dep_id] -= 1
            if in_degree[dep_id] == 0:
                queue.append(dep_id)

    if len(sorted_tasks) != len(plan.tasks):
        raise CycleDetectedError("Failed to order all tasks due to circular dependency")

    return sorted_tasks
