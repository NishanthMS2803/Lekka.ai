"""Parent LLM gap check audit evaluating wave task results."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lekka_ai.config import get_settings
from lekka_ai.llm.base import LLMClient, Message
from lekka_ai.llm.gemini import GeminiClient
from lekka_ai.logging import get_logger
from lekka_ai.orchestrator.dispatch import TaskResult
from lekka_ai.orchestrator.state import PlannedTask, RunState

logger = get_logger(__name__)


class GapCheckAction(StrEnum):
    """Categorical action decision produced by parent orchestrator gap check."""

    DONE = "done"
    FOLLOW_UP = "follow_up"
    ABSTAIN = "abstain"


class FollowUpTaskSpec(BaseModel):
    """Specification of a targeted follow-up task to resolve gaps or conflicts."""

    model_config = ConfigDict(extra="ignore")

    task_id: str = Field(
        ...,
        description="Unique task slug, e.g. 'task_resolve_aapl_rev_conflict'",
    )
    agent_type: str = Field(
        ...,
        description="Target specialist agent type (market, news, fundamental, etc.)",
    )
    instruction: str = Field(
        ...,
        description=(
            "Specific, actionable instruction explaining what gap or conflict "
            "to investigate. Never use generic instructions."
        ),
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Prerequisite task IDs",
    )
    priority: int = Field(
        default=2,
        description="Scheduling priority integer (higher executed first)",
    )
    expected_cost: float = Field(
        default=0.0,
        description="Estimated token or budget cost",
    )

    def to_planned_task(self) -> PlannedTask:
        """Convert specification into an executable PlannedTask."""
        return PlannedTask(
            task_id=self.task_id,
            agent_type=self.agent_type,
            instruction=self.instruction,
            depends_on=self.depends_on,
            priority=self.priority,
            expected_cost=self.expected_cost,
        )


class Decision(BaseModel):
    """Structured parent orchestrator decision following a wave gap-check."""

    model_config = ConfigDict(extra="ignore")

    action: GapCheckAction = Field(
        ...,
        description="Parent decision action: 'done', 'follow_up', or 'abstain'",
    )
    follow_up_tasks: list[FollowUpTaskSpec] = Field(
        default_factory=list,
        description="Follow-up specialist tasks to dispatch if action is 'follow_up'",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Reasons justifying the decision, flagged conflicts, or gaps",
    )

    @field_validator("action", mode="before")
    @classmethod
    def _validate_action(cls, v: Any) -> GapCheckAction:
        if isinstance(v, GapCheckAction):
            return v
        if isinstance(v, str):
            clean = v.strip().lower()
            try:
                return GapCheckAction(clean)
            except ValueError:
                pass
        raise ValueError(
            f"Invalid action '{v}'. Must be one of: {[a.value for a in GapCheckAction]}"
        )

    def to_planned_tasks(self) -> list[PlannedTask]:
        """Convert all follow-up task specifications to PlannedTask objects."""
        return [task.to_planned_task() for task in self.follow_up_tasks]


GAP_CHECK_SYSTEM_PROMPT = """\
You are the Lead Investment Research Gap-Check Evaluator for Lekka AI.
Your role is to critically audit specialist agent research findings after an
execution wave, detecting factual gaps, conflicts, unsupported assumptions,
or missing evidence before final synthesis.

EVALUATION RULES:
1. CONFLICTING FACTS:
   - Carefully cross-examine claims across all specialists.
   - If two or more agents make contradictory statements (e.g. conflicting
     revenue figures, differing growth rates, or opposite catalyst sentiment),
     flag the conflict explicitly in 'reasons' with the exact conflicting
     values and claim IDs.
2. UNSUPPORTED ASSUMPTIONS:
   - Identify any explicit or implicit assumptions made by specialists that
     lack empirical grounding or conflict with verified evidence from other
     agents.
3. MISSING EVIDENCE:
   - Identify critical factual claims that lack tool-verified evidence
     citations.
   - Flag key aspects of the user's research question that have not yet been
     addressed by any specialist.
4. LOW-CONFIDENCE KEY CLAIMS:
   - Flag key claims with low confidence (< 0.70) or tasks returning partial
     or insufficient_evidence status.
5. DECISION ACTIONS:
   - 'done': All critical aspects of the research question are addressed
     with verified, high-confidence evidence; no material conflicts or
     ungrounded assumptions exist.
   - 'follow_up': Material conflicts, unsupported key assumptions, missing
     evidence, or low-confidence key claims exist that can be resolved by
     dispatching follow-up specialist tasks. In 'follow_up_tasks', provide
     specific, actionable instructions per specialist.
   - 'abstain': The question cannot be answered responsibly due to
     irreconcilable data contradictions, unavailable filings/data that cannot
     be resolved by further research, or unresolvable question ambiguity.
"""


def _format_task_results_for_prompt(
    task_results: list[TaskResult] | list[dict[str, Any]],
) -> str:
    """Format structured specialist results into clean text for LLM evaluation."""
    lines: list[str] = []

    for idx, tr in enumerate(task_results, 1):
        if isinstance(tr, TaskResult):
            task_id = tr.task_id
            agent_type = tr.agent_type
            status = tr.status
            confidence = tr.confidence
            rationale = tr.confidence_rationale
            claims = tr.claims
            assumptions = tr.assumptions
            open_questions = tr.open_questions
            error = tr.error
        elif isinstance(tr, dict):
            task_id = tr.get("task_id", f"task_{idx}")
            agent_type = tr.get("agent_type", "specialist")
            status = tr.get("status", "unknown")
            confidence = tr.get("confidence")
            rationale = tr.get("confidence_rationale")
            claims = tr.get("claims", [])
            assumptions = tr.get("assumptions", {})
            open_questions = tr.get("open_questions", [])
            error = tr.get("error")
        else:
            continue

        conf_str = (
            f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "N/A"
        )
        lines.append(
            f"### Task {idx}: {task_id} (Agent: {agent_type}, Status: {status}, "
            f"Confidence: {conf_str})"
        )
        if rationale:
            lines.append(f"  Confidence Rationale: {rationale}")
        if error:
            lines.append(f"  Error: {error}")

        if claims:
            lines.append("  Claims:")
            for c in claims:
                if isinstance(c, dict):
                    cid = c.get("claim_id", "claim")
                    ctype = c.get("type", "fact")
                    cconf = c.get("confidence", "N/A")
                    ctext = c.get("text", "")
                    cev = c.get("evidence", [])
                    ev_summary = (
                        f" (cites {len(cev)} evidence sources)"
                        if cev
                        else " (no evidence)"
                    )
                    lines.append(
                        f"    - [{cid}] ({ctype}, conf={cconf}): {ctext}{ev_summary}"
                    )
                elif hasattr(c, "text"):
                    cid = getattr(c, "claim_id", "claim")
                    ctype = getattr(c, "type", "fact")
                    cconf = getattr(c, "confidence", "N/A")
                    ctext = getattr(c, "text", "")
                    cev = getattr(c, "evidence", [])
                    ev_summary = (
                        f" (cites {len(cev)} evidence sources)"
                        if cev
                        else " (no evidence)"
                    )
                    lines.append(
                        f"    - [{cid}] ({ctype}, conf={cconf}): {ctext}{ev_summary}"
                    )
        else:
            lines.append("  Claims: (None submitted)")

        if assumptions:
            lines.append(f"  Assumptions: {assumptions}")
        else:
            lines.append("  Assumptions: (None)")

        if open_questions:
            lines.append("  Open Questions:")
            for oq in open_questions:
                lines.append(f"    - {oq}")
        else:
            lines.append("  Open Questions: (None)")

        lines.append("")

    return "\n".join(lines).strip()


def _parse_decision(raw_content: Any) -> Decision:
    """Parse raw LLM structured output or JSON text into a Decision model."""
    if isinstance(raw_content, Decision):
        return raw_content

    if isinstance(raw_content, dict):
        return Decision.model_validate(raw_content)

    if isinstance(raw_content, str):
        clean_text = raw_content.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        if fence_match:
            clean_text = fence_match.group(1).strip()
        data = json.loads(clean_text)
        return Decision.model_validate(data)

    raise ValueError(f"Unable to parse Decision from type {type(raw_content)}")


async def gap_check(
    task_results: list[TaskResult] | list[dict[str, Any]] | None = None,
    *,
    question: str | None = None,
    state: RunState | None = None,
    llm: LLMClient | None = None,
) -> Decision:
    """Audit completed wave task results and decide next orchestrator step.

    Reads claims (with types and confidence), open questions, and assumptions
    from specialist TaskResults. Flags conflicting facts, unsupported assumptions,
    missing evidence, and low-confidence claims.

    Args:
        task_results: List of completed TaskResult objects or result dictionaries.
        question: Original user research question or objective.
        state: Optional RunState from which to extract results or question.
        llm: Optional LLM client (defaults to GeminiClient).

    Returns:
        Structured Decision with action in {done, follow_up, abstain},
        follow_up_tasks, and reasons.
    """
    effective_question = question or (
        state.question if state else "Investment evaluation"
    )

    # Extract task results from state if not directly passed
    effective_results: list[Any] = []
    if task_results:
        effective_results = list(task_results)
    elif state and state.results:
        effective_results = list(state.results.values())

    if not effective_results:
        logger.warning("Gap check called with empty task results; returning follow_up")
        return Decision(
            action=GapCheckAction.FOLLOW_UP,
            follow_up_tasks=[],
            reasons=["No specialist task results were provided for gap evaluation."],
        )

    # 1. Initialize LLM if not provided
    if llm is None:
        settings = get_settings()
        llm = GeminiClient(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
        )

    # 2. Build user prompt
    formatted_results = _format_task_results_for_prompt(effective_results)
    prompt_parts = [
        f"RESEARCH QUESTION / OBJECTIVE:\n{effective_question}",
        f"SPECIALIST TASK RESULTS TO AUDIT:\n{formatted_results}",
    ]
    if state and state.open_questions:
        oq_lines = "\n".join(f"- {q}" for q in state.open_questions)
        prompt_parts.append(f"RUN-LEVEL OPEN QUESTIONS & CONFLICTS:\n{oq_lines}")

    prompt_parts.append(
        "Perform the gap check audit according to rules. Identify any conflicting "
        "facts, unsupported assumptions, missing evidence, or low-confidence key "
        "claims, and return your structured Decision."
    )
    user_prompt = "\n\n".join(prompt_parts)

    # 3. Call LLM for structured Decision
    response = await llm.generate(
        messages=[Message(role="user", content=user_prompt)],
        response_schema=Decision,
        system=GAP_CHECK_SYSTEM_PROMPT,
    )

    # 4. Parse Decision
    decision: Decision
    if response.structured_output is not None:
        decision = _parse_decision(response.structured_output)
    elif response.text:
        decision = _parse_decision(response.text)
    else:
        raise ValueError("LLM returned an empty response for gap check evaluation")

    logger.info(
        "Parent gap check evaluated",
        action=decision.action.value
        if hasattr(decision.action, "value")
        else str(decision.action),
        follow_up_count=len(decision.follow_up_tasks),
        reasons_count=len(decision.reasons),
    )

    return decision


run_gap_check = gap_check
