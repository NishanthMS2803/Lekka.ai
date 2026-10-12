"""Source support verifier evaluating fact claims against cited traces and tools."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lekka_ai.config import get_settings
from lekka_ai.llm.base import LLMClient, Message
from lekka_ai.llm.gemini import GeminiClient
from lekka_ai.logging import get_logger
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence
from lekka_ai.tracing import Span, Tracer

logger = get_logger(__name__)

SUPPORT_VERIFIER_SYSTEM_PROMPT = """\
You are an expert, independent financial auditor verifying factual claims against
cited source evidence passages and tool execution outputs.

Your task is to determine whether the cited passage or tool output directly supports
the specific factual claim.

VERDICT STANDARDS:
1. 'yes' (SUPPORTED): The cited evidence directly, factually, and accurately confirms
   the claim's core statement and numerical quantities.
2. 'no' (UNSUPPORTED): The cited evidence contradicts the claim, refutes the
   numbers, makes statements opposite to the claim, or contains no relevant evidence.
3. 'partial' (PARTIALLY SUPPORTED): The cited evidence provides directional or partial
   support (e.g. confirms the company/metric) but disagrees on the exact number,
   omits crucial qualifying context, or only verifies a portion of a compound claim.

For every claim, return a structured verdict with:
- claim_id: The exact ID of the evaluated claim.
- supported: Exactly one of 'yes', 'no', or 'partial'.
- reason: A concise, factual explanation for your determination.\
"""


class SupportVerdict(StrEnum):
    """Categorical verdict indicating whether cited source supports a claim."""

    YES = "yes"
    NO = "no"
    PARTIAL = "partial"


class ClaimSupportCheck(BaseModel):
    """Verification verdict evaluating source support for a single claim."""

    model_config = ConfigDict(extra="ignore")

    claim_id: str = Field(..., description="Unique identifier of the evaluated claim")
    supported: SupportVerdict = Field(
        ...,
        description="Support determination: 'yes', 'no', or 'partial'",
    )
    reason: str = Field(
        ...,
        description=(
            "Factual explanation justifying why the evidence supports or refutes"
        ),
    )

    @field_validator("supported", mode="before")
    @classmethod
    def _validate_supported(cls, v: Any) -> SupportVerdict:
        if isinstance(v, SupportVerdict):
            return v
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in ("yes", "true", "supported"):
                return SupportVerdict.YES
            elif clean in ("no", "false", "unsupported", "refuted"):
                return SupportVerdict.NO
            elif clean in ("partial", "partially_supported", "partially"):
                return SupportVerdict.PARTIAL
        raise ValueError(
            f"Invalid supported verdict '{v}'; must be 'yes', 'no', or 'partial'"
        )


class BatchSupportResponse(BaseModel):
    """Structured response containing verification verdicts for a batch of claims."""

    model_config = ConfigDict(extra="ignore")

    verdicts: list[ClaimSupportCheck] = Field(
        default_factory=list,
        description="List of verification verdicts for the batch of claims",
    )


def _format_data_snippet(data: Any, max_length: int = 2500) -> str:
    """Format structured tool output or evidence snippet into a readable string."""
    if data is None:
        return ""
    if isinstance(data, str):
        text = data.strip()
    else:
        try:
            text = json.dumps(data, indent=2, default=str)
        except Exception:
            text = str(data)
    if len(text) > max_length:
        return text[:max_length] + "\n... [truncated]"
    return text


def fetch_claim_evidence_passage(
    claim: Claim | dict[str, Any] | Any,
    trace: Tracer | Any | None = None,
) -> str:
    """Fetch cited evidence passages and tool outputs for a claim from trace storage.

    Inspects all Evidence objects linked to the claim. Resolves cited tool_call_ids
    against the provided trace (which may be a Tracer, SQLAlchemy Session, dict,
    or list of execution spans/records), and includes any cited raw_value snippets.

    Args:
        claim: Pydantic Claim, dictionary, or ORM ClaimModel to evaluate.
        trace: Optional execution trace (Tracer, Session, dict, or list).

    Returns:
        Formatted text containing all cited passages and tool outputs.
    """
    evidence_list: list[Any] = []
    if isinstance(claim, dict):
        evidence_list = list(claim.get("evidence", []) or [])
    elif hasattr(claim, "evidence"):
        evidence_list = list(getattr(claim, "evidence", []) or [])

    if not evidence_list:
        return "[No cited evidence items provided for this claim]"

    passages: list[str] = []

    for i, ev in enumerate(evidence_list, start=1):
        tool_call_id: str | None = None
        raw_val: Any = None
        source_type: str = "unknown"
        doc_id: str | None = None
        section: str | None = None

        if isinstance(ev, Evidence):
            tool_call_id = ev.tool_call_id
            raw_val = ev.raw_value
            source_type = ev.source_type
            doc_id = ev.document_id
            section = ev.section
        elif isinstance(ev, dict):
            tool_call_id = ev.get("tool_call_id")
            raw_val = ev.get("raw_value")
            source_type = ev.get("source_type", "unknown")
            doc_id = ev.get("document_id")
            section = ev.get("section")
        elif hasattr(ev, "tool_call_id"):
            tool_call_id = getattr(ev, "tool_call_id", None)
            raw_val = getattr(ev, "raw_value", None)
            source_type = getattr(ev, "source_type", "unknown")
            doc_id = getattr(ev, "document_id", None)
            section = getattr(ev, "section", None)

        tool_output: Any = None
        resolved_tool_name: str | None = None

        # Resolve tool call output from trace if tool_call_id is available
        if tool_call_id and trace is not None:
            clean_tid = str(tool_call_id).strip()

            # 1. Tracer instance
            if isinstance(trace, Tracer) or hasattr(trace, "spans"):
                spans: list[Span] = getattr(trace, "spans", [])
                for span in spans:
                    span_tid = (
                        str(span.tool_call_id).strip()
                        if span.tool_call_id
                        else str(span.id)
                    )
                    if (
                        span_tid == clean_tid
                        or str(span.id) == clean_tid
                        or span.name == clean_tid
                        or span.metadata.get("tool_call_id") == clean_tid
                    ):
                        tool_output = span.output_json
                        resolved_tool_name = span.tool_name or span.name
                        break

            # 2. Dictionary trace mapping
            elif isinstance(trace, dict):
                if clean_tid in trace:
                    tool_output = trace[clean_tid]
                elif "tool_calls" in trace and isinstance(trace["tool_calls"], dict):
                    tool_output = trace["tool_calls"].get(clean_tid)
                elif "spans" in trace and isinstance(trace["spans"], dict):
                    tool_output = trace["spans"].get(clean_tid)
                resolved_tool_name = trace.get("tool_name")

            # 3. List of spans or records
            elif isinstance(trace, list):
                for item in trace:
                    item_id = None
                    if isinstance(item, dict):
                        item_id = str(
                            item.get("tool_call_id") or item.get("id") or ""
                        ).strip()
                        if item_id == clean_tid:
                            tool_output = item.get("output_json") or item.get("output")
                            resolved_tool_name = item.get("tool_name") or item.get(
                                "name"
                            )
                            break
                    elif hasattr(item, "tool_call_id") or hasattr(item, "id"):
                        item_id = str(
                            getattr(item, "tool_call_id", None)
                            or getattr(item, "id", None)
                            or ""
                        ).strip()
                        if item_id == clean_tid:
                            tool_output = getattr(
                                item,
                                "output_json",
                                getattr(item, "output", None),
                            )
                            resolved_tool_name = getattr(
                                item,
                                "tool_name",
                                getattr(item, "name", None),
                            )
                            break

            # 4. SQLAlchemy Session
            elif hasattr(trace, "query") or hasattr(trace, "get"):
                try:
                    from lekka_ai.db.repository import get_tool_call

                    tc_uuid = uuid.UUID(clean_tid)
                    db_tc = get_tool_call(trace, tc_uuid)
                    if db_tc:
                        tool_output = db_tc.output_json
                        resolved_tool_name = db_tc.tool_name
                except Exception:
                    pass

        # Build passage description for this evidence item
        parts: list[str] = [
            f"[Evidence {i}] Source: {source_type}"
            + (f" | Tool: {resolved_tool_name}" if resolved_tool_name else "")
            + (f" | Document: {doc_id}" if doc_id else "")
            + (f" | Section: {section}" if section else "")
            + (f" | ToolCall: {tool_call_id}" if tool_call_id else "")
        ]

        if raw_val is not None and str(raw_val).strip():
            parts.append(f"Cited Snippet / Value:\n{_format_data_snippet(raw_val)}")

        if tool_output is not None:
            parts.append(
                f"Resolved Tool Output from Trace:\n{_format_data_snippet(tool_output)}"
            )

        if len(parts) == 1:
            parts.append("[No passage snippet or tool output found in trace]")

        passages.append("\n".join(parts))

    return "\n\n---\n\n".join(passages)


def _parse_batch_response(raw_content: Any) -> list[ClaimSupportCheck]:
    """Parse raw LLM output into a list of ClaimSupportCheck models."""
    if isinstance(raw_content, BatchSupportResponse):
        return raw_content.verdicts

    if isinstance(raw_content, list):
        return [
            item
            if isinstance(item, ClaimSupportCheck)
            else ClaimSupportCheck.model_validate(item)
            for item in raw_content
        ]

    if isinstance(raw_content, dict):
        if "verdicts" in raw_content:
            return BatchSupportResponse.model_validate(raw_content).verdicts
        if "claim_id" in raw_content and "supported" in raw_content:
            return [ClaimSupportCheck.model_validate(raw_content)]

    if isinstance(raw_content, str):
        clean_text = raw_content.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        if fence_match:
            clean_text = fence_match.group(1).strip()
        data = json.loads(clean_text)
        if isinstance(data, dict):
            if "verdicts" in data:
                return BatchSupportResponse.model_validate(data).verdicts
            if "claim_id" in data and "supported" in data:
                return [ClaimSupportCheck.model_validate(data)]
        elif isinstance(data, list):
            return [ClaimSupportCheck.model_validate(item) for item in data]

    raise ValueError(f"Unable to parse verifier response from type {type(raw_content)}")


def _extract_claim_info(
    claim: Claim | dict[str, Any] | Any,
) -> tuple[str, str, str]:
    """Extract (claim_id, text, type) from a Claim model or dictionary."""
    if isinstance(claim, dict):
        claim_id = str(claim.get("claim_id", f"claim_{uuid.uuid4().hex[:8]}"))
        text = str(claim.get("text", "")).strip()
        claim_type = str(claim.get("type", "fact")).strip().lower()
    else:
        claim_id = str(
            getattr(claim, "claim_id", None) or f"claim_{uuid.uuid4().hex[:8]}"
        )
        text = str(getattr(claim, "text", "")).strip()
        raw_type = getattr(claim, "type", "fact")
        claim_type = (
            raw_type.value
            if hasattr(raw_type, "value")
            else str(raw_type).strip().lower()
        )
    return claim_id, text, claim_type


def _store_verdict_on_claim(
    claim: Claim | dict[str, Any] | Any,
    verdict: ClaimSupportCheck,
) -> None:
    """Store support verdict and explanation directly on the claim object."""
    verdict_str = (
        verdict.supported.value
        if hasattr(verdict.supported, "value")
        else str(verdict.supported)
    )

    if isinstance(claim, dict):
        claim["verdict"] = verdict_str
        claim["verdict_reason"] = verdict.reason
        claim["support_verdict"] = verdict_str
        claim["support_reason"] = verdict.reason
    else:
        if hasattr(claim, "verdict"):
            claim.verdict = verdict_str
        if hasattr(claim, "verdict_reason"):
            claim.verdict_reason = verdict.reason
        try:
            claim.support_verdict = verdict_str
            claim.support_reason = verdict.reason
        except Exception:
            pass


async def verify_claim_support(
    claims: Sequence[Claim | dict[str, Any] | Any],
    *,
    trace: Tracer | Any | None = None,
    llm: LLMClient | None = None,
    model: str | None = None,
    batch_size: int | None = None,
) -> list[ClaimSupportCheck]:
    """Verify whether cited passages and tool outputs support factual claims.

    For each fact claim, fetches the cited passage/tool output from the trace
    and asks Gemini (structured output: supported yes/no/partial, reason) whether
    the passage supports the claim. Stores verdicts on the claim.

    Uses batching and the cheap model setting from config.

    Args:
        claims: List of Claim objects or claim dictionaries to verify.
        trace: Optional execution trace containing tool call outputs.
        llm: Optional LLM client instance (defaults to GeminiClient).
        model: Optional model override (defaults to Settings.VERIFIER_MODEL).
        batch_size: Optional batch size for grouped verification calls.

    Returns:
        List of structured ClaimSupportCheck verdicts.
    """
    # 1. Filter claims: verify FACT claims only
    fact_claims: list[Any] = []
    for c in claims:
        _, _, c_type = _extract_claim_info(c)
        if c_type in ("fact", ClaimType.FACT.value):
            fact_claims.append(c)

    if not fact_claims:
        logger.info("No fact claims provided for source support verification")
        return []

    # 2. Configure LLM client and parameters
    settings = get_settings()
    effective_model = (
        model or getattr(settings, "VERIFIER_MODEL", None) or settings.GEMINI_MODEL
    )
    effective_batch_size = (
        batch_size or getattr(settings, "VERIFIER_BATCH_SIZE", None) or 5
    )

    client = llm
    if client is None:
        client = GeminiClient(
            api_key=settings.GEMINI_API_KEY,
            model=effective_model,
        )

    all_verdicts: list[ClaimSupportCheck] = []
    verdicts_by_id: dict[str, ClaimSupportCheck] = {}

    # 3. Process fact claims in batches
    for i in range(0, len(fact_claims), effective_batch_size):
        chunk = fact_claims[i : i + effective_batch_size]

        claim_prompts: list[str] = []
        for c in chunk:
            c_id, c_text, _ = _extract_claim_info(c)
            passage = fetch_claim_evidence_passage(c, trace=trace)
            claim_prompts.append(
                f"CLAIM ID: {c_id}\n"
                f"CLAIM STATEMENT: {c_text}\n"
                f"CITED EVIDENCE PASSAGE / TOOL OUTPUT:\n"
                f"{passage}"
            )

        prompt_body = "\n\n========================================\n\n".join(
            claim_prompts
        )
        user_message = (
            f"Please verify the following {len(chunk)} factual claims against their "
            f"cited source evidence and tool outputs:\n\n"
            f"{prompt_body}\n\n"
            f"Return a structured verdict for each claim with 'claim_id', 'supported' "
            f"(yes/no/partial), and 'reason'."
        )

        response = await client.generate(
            messages=[Message(role="user", content=user_message)],
            response_schema=BatchSupportResponse,
            system=SUPPORT_VERIFIER_SYSTEM_PROMPT,
        )

        batch_verdicts: list[ClaimSupportCheck]
        if response.structured_output is not None:
            batch_verdicts = _parse_batch_response(response.structured_output)
        elif response.text:
            batch_verdicts = _parse_batch_response(response.text)
        else:
            raise ValueError(
                "Verifier LLM returned an empty response for claim support verification"
            )

        for v in batch_verdicts:
            verdicts_by_id[v.claim_id] = v
            all_verdicts.append(v)

    # 4. Store verdicts on the claims
    for c in fact_claims:
        c_id, _, _ = _extract_claim_info(c)
        if c_id in verdicts_by_id:
            _store_verdict_on_claim(c, verdicts_by_id[c_id])
        else:
            # Fallback if model missed an ID in batch
            fallback = ClaimSupportCheck(
                claim_id=c_id,
                supported=SupportVerdict.NO,
                reason="Verification omitted or claim ID missing from verifier output.",
            )
            _store_verdict_on_claim(c, fallback)
            all_verdicts.append(fallback)

    logger.info(
        "Source support verification complete",
        total_fact_claims=len(fact_claims),
        supported_yes=sum(1 for v in all_verdicts if v.supported == SupportVerdict.YES),
        supported_no=sum(1 for v in all_verdicts if v.supported == SupportVerdict.NO),
        supported_partial=sum(
            1 for v in all_verdicts if v.supported == SupportVerdict.PARTIAL
        ),
    )

    return all_verdicts


verify_support = verify_claim_support
