"""Direct LLM runner (V1 baseline) executing prompts without tools."""

from __future__ import annotations

import re
from uuid import uuid4

from pydantic import BaseModel, Field

from lekka_ai.harness.models import AgentTask
from lekka_ai.harness.runtime import AgentRunResult
from lekka_ai.llm.base import LLMClient, Message
from lekka_ai.schemas.claims import Claim, Evidence
from lekka_ai.schemas.submit import AnswerStatus, SubmitAnswer


class NumericClaimItem(BaseModel):
    """Structured extraction of a single numeric or factual claim."""

    text: str = Field(
        ...,
        description="Declarative claim statement containing the numeric figure",
    )
    type: str = Field(
        default="fact",
        description="Epistemic type: fact or calculation",
    )
    value: float | None = Field(
        default=None,
        description="Extracted numeric value if applicable",
    )
    metric: str | None = Field(
        default=None,
        description="Financial metric name (e.g. closing price, revenue, margin)",
    )
    as_of_date: str | None = Field(
        default=None,
        description="Point-in-time date of the figure in YYYY-MM-DD format",
    )


class ExtractedNumericClaims(BaseModel):
    """List container of extracted numeric claims for structured LLM decoding."""

    claims: list[NumericClaimItem] = Field(
        default_factory=list,
        description="List of extracted numeric claim items",
    )


def _heuristic_extract_numeric_claims(text: str) -> list[NumericClaimItem]:
    """Fallback rule-based extraction when LLM output lacks structure."""
    if not text:
        return []

    # Match sentences or clauses containing numbers, currencies, or percentages
    sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())
    extracted: list[NumericClaimItem] = []
    num_pattern = re.compile(r"(\$?\b\d+(?:,\d{3})*(?:\.\d+)?%?\b)")

    for sent in sentences:
        clean_sent = sent.strip().strip("*#-_ ")
        if not clean_sent or len(clean_sent) < 10:
            continue
        match = num_pattern.search(clean_sent)
        if match:
            # Parse raw number if simple float
            raw_val_str = (
                match.group(1).replace("$", "").replace("%", "").replace(",", "")
            )
            raw_float: float | None = None
            try:
                raw_float = float(raw_val_str)
            except ValueError:
                raw_float = None

            extracted.append(
                NumericClaimItem(
                    text=clean_sent,
                    type="fact",
                    value=raw_float,
                )
            )

    return extracted


class DirectLLMRunner:
    """Baseline V1 runner sending questions directly to an LLM without tools."""

    @classmethod
    async def run(
        cls,
        task: AgentTask | str,
        llm: LLMClient,
        *,
        as_of_date: str | None = None,
        extract_claims: bool = True,
    ) -> AgentRunResult:
        """Execute task through direct LLM invocation and extract numeric claims."""
        runner = cls()
        return await runner.execute(
            task=task,
            llm=llm,
            as_of_date=as_of_date,
            extract_claims=extract_claims,
        )

    async def execute(
        self,
        task: AgentTask | str,
        llm: LLMClient,
        *,
        as_of_date: str | None = None,
        extract_claims: bool = True,
    ) -> AgentRunResult:
        """Run direct LLM query and structure numeric claims for grading."""
        if isinstance(task, str):
            agent_task = AgentTask(
                instruction=task,
                as_of_date=as_of_date,
                agent_type="direct",
            )
        else:
            agent_task = task

        system_parts = [
            "You are a financial research assistant answering directly from your "
            "knowledge base. Provide a clear, detailed, and factual answer to the "
            "user's question without using external tools."
        ]
        if agent_task.as_of_date:
            system_parts.append(
                f"Point-in-time cutoff date: {agent_task.as_of_date}. "
                "Do not reference or assume knowledge after this date."
            )
        system_prompt = "\n\n".join(system_parts)

        # 1. Primary generation call with no tools
        messages = [Message(role="user", content=agent_task.instruction.strip())]
        response = await llm.generate(
            messages=messages,
            tools=None,
            system=system_prompt,
        )

        answer_text = response.text or ""
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens

        # 2. Best-effort extraction of numeric claims via structured output
        extracted_items: list[NumericClaimItem] = []
        if extract_claims and answer_text.strip():
            extract_prompt = (
                "Extract all specific numeric and quantitative financial claims "
                f"from the following text:\n\n{answer_text}"
            )
            extraction_response = await llm.generate(
                messages=[Message(role="user", content=extract_prompt)],
                tools=None,
                response_schema=ExtractedNumericClaims,
                system="Extract all specific factual and calculation numeric claims.",
            )
            tokens_in += extraction_response.usage.input_tokens
            tokens_out += extraction_response.usage.output_tokens

            if extraction_response.structured_output is not None:
                if isinstance(
                    extraction_response.structured_output, ExtractedNumericClaims
                ):
                    extracted_items = extraction_response.structured_output.claims
                elif isinstance(extraction_response.structured_output, dict):
                    raw_claims = extraction_response.structured_output.get("claims", [])
                    extracted_items = [
                        NumericClaimItem.model_validate(c) for c in raw_claims
                    ]
            elif extraction_response.text:
                try:
                    parsed = ExtractedNumericClaims.model_validate_json(
                        extraction_response.text
                    )
                    extracted_items = parsed.claims
                except Exception:
                    extracted_items = _heuristic_extract_numeric_claims(answer_text)
            else:
                extracted_items = _heuristic_extract_numeric_claims(answer_text)
        elif answer_text.strip():
            extracted_items = _heuristic_extract_numeric_claims(answer_text)

        # Convert extracted items into standardized Claim models
        claims: list[Claim] = []
        for item in extracted_items:
            effective_as_of = item.as_of_date or (
                str(agent_task.as_of_date) if agent_task.as_of_date else None
            )
            evidence: list[Evidence] = [
                Evidence(
                    source_type="direct_llm",
                    as_of_date=effective_as_of,
                    raw_value={"value": item.value, "metric": item.metric}
                    if item.metric
                    else item.value,
                )
            ]

            claims.append(
                Claim(
                    claim_id=f"claim_{uuid4().hex[:10]}",
                    text=item.text,
                    type=item.type if item.type in ("fact", "calculation") else "fact",
                    confidence=1.0,
                    evidence=evidence,
                    produced_by="direct_llm",
                )
            )

        submission = SubmitAnswer(
            task_id=agent_task.task_id,
            agent_type="direct",
            status=AnswerStatus.COMPLETED,
            confidence=1.0,
            confidence_rationale="Direct LLM completion without external tools.",
            claims=claims,
        )

        return AgentRunResult(
            task_id=agent_task.task_id,
            agent_type="direct",
            status="completed",
            final_text=answer_text,
            messages=[
                Message(role="user", content=agent_task.instruction),
                Message(role="assistant", content=answer_text),
            ],
            iterations=1,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tool_calls_count=0,
            submission=submission,
            limits_hit=[],
            stored_claims=[],
        )
