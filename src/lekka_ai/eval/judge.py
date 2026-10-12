"""Rubric-based Gemini LLM judge and human-agreement validation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from lekka_ai.llm.base import LLMClient, Message
from lekka_ai.llm.gemini import GeminiClient
from lekka_ai.logging import get_logger

logger = get_logger(__name__)

DEFAULT_RUBRIC_SYSTEM_PROMPT = """\
You are an expert equity research auditor and compliance judge evaluating candidate
financial research responses against ground truth reference materials.

EVALUATION RUBRIC:
1. FACTUAL GROUNDING: Every number, metric, financial ratio, and corporate fact in
   the candidate answer MUST be factually supported by the reference material.
   Invented, hallucinated, or contradictory numbers MUST be marked as 'fail'.
2. RESEARCH INTEGRITY: The candidate answer must address the core question without
   omitting material qualifications, contradicting disclosed trends, or presenting
   unwarranted speculative leaps as verified facts.
3. EPISTEMIC HUMILITY: When reference data is incomplete or long-term predictions
   are unknowable, answers that appropriately acknowledge uncertainty and state
   limitations PASS. Answers fabricating certainty where none exists FAIL.

STANDARDS FOR VERDICT:
- 'pass': Factual statements are supported by reference; no hallucinations; adheres
  to epistemic standards.
- 'fail': Contains fabricated facts, contradicts reference figures, or omits critical
  qualifications.

Return structured output matching the schema with verdict ('pass' or 'fail'),
score (0.0 to 1.0), and concise analytical reasoning.\
"""


class JudgeVerdictOutput(BaseModel):
    """Structured evaluation verdict returned by the rubric-based LLM judge."""

    model_config = ConfigDict(extra="ignore")

    verdict: str = Field(
        ...,
        description="Categorical evaluation verdict ('pass' or 'fail')",
    )
    score: float = Field(
        ...,
        description="Continuous evaluation score between 0.0 and 1.0",
    )
    reasoning: str = Field(
        ...,
        description="Auditor justification explaining the determination",
    )
    key_discrepancies: list[str] = Field(
        default_factory=list,
        description="Specific factual errors, omissions, or ungrounded claims",
    )
    confidence: float = Field(
        default=1.0,
        description="Evaluator certainty score (0.0 to 1.0)",
    )


class JudgeEvaluationRequest(BaseModel):
    """Input parameters for a single evaluation candidate item."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique sample identifier")
    question: str = Field(..., description="Financial question or research query")
    candidate_answer: str = Field(
        ...,
        description="Output produced by candidate system to evaluate",
    )
    reference: str | None = Field(
        default=None,
        description="Ground truth reference passage or filing facts",
    )
    rubric_category: str = Field(
        default="general",
        description="Category evaluated (e.g. factual_accuracy, grounding)",
    )
    custom_rubric: str | None = Field(
        default=None,
        description="Optional custom evaluation rubric prompt",
    )


class JudgeEvaluationRecord(BaseModel):
    """Evaluated record comparing human label and judge verdict."""

    model_config = ConfigDict(extra="ignore")

    id: str
    question: str
    candidate_answer: str
    reference: str | None = None
    human_label: str | None = None
    judge_verdict: str
    score: float
    reasoning: str
    is_agreement: bool | None = None


class AgreementReport(BaseModel):
    """Inter-rater reliability summary comparing human and LLM judge."""

    model_config = ConfigDict(extra="ignore")

    total_samples: int
    accuracy: float
    cohen_kappa: float
    observed_agreement: float
    expected_agreement: float
    kappa_interpretation: str
    confusion_matrix: dict[str, dict[str, int]]
    records: list[JudgeEvaluationRecord] = Field(default_factory=list)
    markdown_table: str = ""


def normalize_verdict_string(raw: str | None) -> str:
    """Normalize arbitrary verdict representations into 'pass' or 'fail'."""
    if not raw:
        return "fail"
    clean = str(raw).strip().lower()
    if clean in ("pass", "yes", "true", "supported", "correct", "1", "1.0"):
        return "pass"
    if clean in ("fail", "no", "false", "unsupported", "incorrect", "0", "0.0"):
        return "fail"
    return clean


def compute_cohen_kappa(
    rater1: Sequence[str],
    rater2: Sequence[str],
) -> tuple[float, float, float]:
    """Compute Cohen's kappa coefficient between two categorical raters.

    Args:
        rater1: Sequence of categorical determinations from rater 1.
        rater2: Sequence of categorical determinations from rater 2.

    Returns:
        Tuple of (cohen_kappa, observed_agreement, expected_agreement).
    """
    if len(rater1) != len(rater2):
        raise ValueError(
            f"Rater sequences must have identical length: "
            f"{len(rater1)} vs {len(rater2)}"
        )
    total = len(rater1)
    if total == 0:
        return 1.0, 1.0, 1.0

    r1_norm = [normalize_verdict_string(x) for x in rater1]
    r2_norm = [normalize_verdict_string(x) for x in rater2]

    categories = sorted(set(r1_norm) | set(r2_norm))

    # Observed agreement
    agreements = sum(1 for a, b in zip(r1_norm, r2_norm, strict=False) if a == b)
    p_observed = agreements / total

    # Expected agreement by chance
    p_expected = 0.0
    for cat in categories:
        cnt1 = sum(1 for x in r1_norm if x == cat)
        cnt2 = sum(1 for x in r2_norm if x == cat)
        p_expected += (cnt1 / total) * (cnt2 / total)

    if p_expected >= 1.0:
        kappa = 1.0 if p_observed >= 1.0 else 0.0
    else:
        kappa = (p_observed - p_expected) / (1.0 - p_expected)

    return round(kappa, 4), round(p_observed, 4), round(p_expected, 4)


def interpret_cohen_kappa(kappa: float) -> str:
    """Return qualitative interpretation of Cohen's kappa score."""
    if kappa < 0.0:
        return "Poor (Disagreement beyond chance)"
    if kappa <= 0.20:
        return "Slight (0.00 - 0.20)"
    if kappa <= 0.40:
        return "Fair (0.21 - 0.40)"
    if kappa <= 0.60:
        return "Moderate (0.41 - 0.60)"
    if kappa <= 0.80:
        return "Substantial (0.61 - 0.80)"
    return "Almost Perfect (0.81 - 1.00)"


class GeminiRubricJudge:
    """Rubric-based judge using Gemini structured outputs to evaluate outputs."""

    def __init__(
        self,
        client: LLMClient | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client or GeminiClient()
        self.system_prompt = system_prompt or DEFAULT_RUBRIC_SYSTEM_PROMPT

    def _build_messages(self, request: JudgeEvaluationRequest) -> list[Message]:
        """Construct conversation messages for judge evaluation."""
        user_content = (
            f"Question: {request.question}\n\n"
            f"Candidate Answer:\n{request.candidate_answer}\n\n"
        )
        if request.reference:
            user_content += f"Reference Information:\n{request.reference}\n\n"
        if request.custom_rubric:
            user_content += f"Specific Rubric Focus:\n{request.custom_rubric}\n\n"
        user_content += "Evaluate the candidate answer against the rubric."

        return [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_content),
        ]

    def _parse_fallback_response(self, text: str) -> JudgeVerdictOutput:
        """Parse structured output from plain text response fallback."""
        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)

        try:
            data = json.loads(clean)
            if isinstance(data, dict):
                return JudgeVerdictOutput(
                    verdict=normalize_verdict_string(data.get("verdict", "fail")),
                    score=float(data.get("score", 0.0)),
                    reasoning=str(data.get("reasoning", "")),
                    key_discrepancies=list(data.get("key_discrepancies", [])),
                    confidence=float(data.get("confidence", 1.0)),
                )
        except Exception:
            pass

        # Regex heuristic fallback
        is_pass = "pass" in clean.lower() and "fail" not in clean.lower()
        return JudgeVerdictOutput(
            verdict="pass" if is_pass else "fail",
            score=1.0 if is_pass else 0.0,
            reasoning=clean[:300],
            key_discrepancies=[],
            confidence=0.8,
        )

    async def evaluate(
        self,
        request: JudgeEvaluationRequest,
    ) -> JudgeVerdictOutput:
        """Evaluate a single candidate response against the rubric."""
        messages = self._build_messages(request)
        try:
            response = await self.client.generate(
                messages,
                response_schema=JudgeVerdictOutput,
            )
            if response.text:
                return self._parse_fallback_response(response.text)
        except Exception as err:
            logger.warning(
                "Gemini judge evaluation call failed for sample %s: %s",
                request.id,
                err,
            )

        return JudgeVerdictOutput(
            verdict="fail",
            score=0.0,
            reasoning="Evaluation failed due to LLM client error",
            key_discrepancies=["Client communication error"],
            confidence=0.0,
        )

    async def evaluate_batch(
        self,
        requests: Sequence[JudgeEvaluationRequest],
    ) -> list[JudgeVerdictOutput]:
        """Evaluate a batch of candidate items sequentially or concurrently."""
        tasks = [self.evaluate(req) for req in requests]
        return await asyncio.gather(*tasks)

    def evaluate_sync(
        self,
        request: JudgeEvaluationRequest,
    ) -> JudgeVerdictOutput:
        """Synchronous wrapper for evaluate."""
        return asyncio.run(self.evaluate(request))


def compute_agreement_report(
    records: Sequence[JudgeEvaluationRecord],
) -> AgreementReport:
    """Compute inter-rater agreement statistics and generate markdown table."""
    total = len(records)
    if total == 0:
        return AgreementReport(
            total_samples=0,
            accuracy=1.0,
            cohen_kappa=1.0,
            observed_agreement=1.0,
            expected_agreement=1.0,
            kappa_interpretation=interpret_cohen_kappa(1.0),
            confusion_matrix={},
            records=[],
            markdown_table="*No evaluation samples provided.*",
        )

    human_labels: list[str] = []
    judge_verdicts: list[str] = []
    updated_records: list[JudgeEvaluationRecord] = []

    for r in records:
        h = normalize_verdict_string(r.human_label) if r.human_label else "pass"
        j = normalize_verdict_string(r.judge_verdict)
        human_labels.append(h)
        judge_verdicts.append(j)

        agreed = h == j
        updated_records.append(
            JudgeEvaluationRecord(
                id=r.id,
                question=r.question,
                candidate_answer=r.candidate_answer,
                reference=r.reference,
                human_label=h,
                judge_verdict=j,
                score=r.score,
                reasoning=r.reasoning,
                is_agreement=agreed,
            )
        )

    kappa, p_obs, p_exp = compute_cohen_kappa(human_labels, judge_verdicts)
    accuracy = p_obs

    # Confusion matrix: human (row) x judge (col)
    categories = sorted(set(human_labels) | set(judge_verdicts))
    matrix: dict[str, dict[str, int]] = {
        row: {col: 0 for col in categories} for row in categories
    }
    for h, j in zip(human_labels, judge_verdicts, strict=False):
        matrix[h][j] += 1

    # Markdown table formatting
    rows = [
        "| ID | Question | Human Label | Judge Verdict | Agreement | Reasoning |",
        "|:---|:---|:---:|:---:|:---:|:---|",
    ]
    for r in updated_records:
        agreed_sym = "✅" if r.is_agreement else "❌"
        q_trunc = r.question[:40] + ("..." if len(r.question) > 40 else "")
        reason_trunc = r.reasoning[:50] + ("..." if len(r.reasoning) > 50 else "")
        rows.append(
            f"| `{r.id}` | {q_trunc} | `{r.human_label}` | `{r.judge_verdict}` "
            f"| {agreed_sym} | {reason_trunc} |"
        )

    table_md = "\n".join(rows)

    return AgreementReport(
        total_samples=total,
        accuracy=accuracy,
        cohen_kappa=kappa,
        observed_agreement=p_obs,
        expected_agreement=p_exp,
        kappa_interpretation=interpret_cohen_kappa(kappa),
        confusion_matrix=matrix,
        records=updated_records,
        markdown_table=table_md,
    )


def load_judge_sample_csv(csv_path: Path | str) -> list[JudgeEvaluationRequest]:
    """Load sample benchmark items from CSV for judge validation."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Judge sample CSV not found at {path}")

    requests: list[JudgeEvaluationRequest] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            requests.append(
                JudgeEvaluationRequest(
                    id=row.get("id", f"sample_{len(requests) + 1}"),
                    question=row.get("question", ""),
                    candidate_answer=row.get("candidate_answer", ""),
                    reference=row.get("reference"),
                    rubric_category=row.get("rubric_category", "general"),
                )
            )

    return requests


async def run_judge_validation(
    csv_path: Path | str,
    judge: GeminiRubricJudge | None = None,
    output_path: Path | str | None = None,
) -> AgreementReport:
    """Run LLM judge validation against human labels in a sample CSV file.

    Args:
        csv_path: Path to sample CSV with candidate answers and human labels.
        judge: GeminiRubricJudge instance (defaults to GeminiClient).
        output_path: Optional destination to write agreement report markdown.

    Returns:
        Structured AgreementReport containing accuracy and Cohen's kappa.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Judge sample CSV file not found: {path}")

    # Read human labels directly
    human_labels_map: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("id", "")
            if cid and "human_label" in row:
                human_labels_map[cid] = row["human_label"]

    requests = load_judge_sample_csv(path)
    active_judge = judge or GeminiRubricJudge()
    verdicts = await active_judge.evaluate_batch(requests)

    records: list[JudgeEvaluationRecord] = []
    for req, out in zip(requests, verdicts, strict=False):
        h_label = human_labels_map.get(req.id, "pass")
        records.append(
            JudgeEvaluationRecord(
                id=req.id,
                question=req.question,
                candidate_answer=req.candidate_answer,
                reference=req.reference,
                human_label=h_label,
                judge_verdict=out.verdict,
                score=out.score,
                reasoning=out.reasoning,
            )
        )

    report = compute_agreement_report(records)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        report_content = (
            f"# LLM Judge Agreement Validation Report\n\n"
            f"- **Evaluated Samples**: {report.total_samples}\n"
            f"- **Accuracy (Observed Agreement)**: {report.accuracy:.2%}\n"
            f"- **Cohen's Kappa**: {report.cohen_kappa:.4f} "
            f"({report.kappa_interpretation})\n"
            f"- **Expected Agreement (Chance)**: {report.expected_agreement:.2%}\n\n"
            f"## Confusion Matrix\n\n"
            f"```json\n{json.dumps(report.confusion_matrix, indent=2)}\n```\n\n"
            f"## Per-Sample Verdicts\n\n"
            f"{report.markdown_table}\n"
        )
        out_p.write_text(report_content, encoding="utf-8")
        logger.info("Saved agreement report to %s", out_p)

    return report


def main() -> None:
    """CLI entrypoint running judge validation against sample CSV."""
    parser = argparse.ArgumentParser(
        description="Run LLM Judge validation against human labels",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="eval/labels/judge_sample.csv",
        help="Path to sample CSV file with human labels",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results/judge_agreement_report.md",
        help="Destination path for agreement markdown report",
    )
    args = parser.parse_args()

    report = asyncio.run(
        run_judge_validation(
            csv_path=args.csv,
            output_path=args.output,
        )
    )

    print("\n" + "=" * 60)
    print(" LLM Judge Human-Agreement Validation")
    print("=" * 60)
    print(f"Total Samples evaluated: {report.total_samples}")
    print(f"Accuracy (Observed Agreement): {report.accuracy:.2%}")
    print(f"Cohen's Kappa: {report.cohen_kappa:.4f} ({report.kappa_interpretation})")
    print("=" * 60)
    print(report.markdown_table)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
