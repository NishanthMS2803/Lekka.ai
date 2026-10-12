"""Deterministic verification checks for financial calculations, types, and dates."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Sequence
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lekka_ai.logging import get_logger
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence

logger = get_logger(__name__)


class CheckStatus(StrEnum):
    """Categorical outcome status of a deterministic verification check."""

    PASSED = "passed"
    FAILED = "failed"


class CheckResult(BaseModel):
    """Structured result produced by an individual deterministic check."""

    model_config = ConfigDict(extra="ignore")

    check_name: str = Field(..., description="Identifier of the deterministic check")
    passed: bool = Field(..., description="Whether the check passed or failed")
    status: CheckStatus = Field(
        ...,
        description="Categorical status ('passed' or 'failed')",
    )
    message: str = Field(..., description="Human-readable explanation of outcome")
    claim_id: str | None = Field(
        default=None,
        description="Claim identifier evaluated, if applicable",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional debugging or diagnostic details",
    )


class ChecksReport(BaseModel):
    """Summary report aggregating outcomes of all deterministic checks."""

    model_config = ConfigDict(extra="ignore")

    all_passed: bool = Field(
        ...,
        description="True if all deterministic checks passed with zero failures",
    )
    passed_count: int = Field(default=0, description="Number of passed checks")
    failed_count: int = Field(default=0, description="Number of failed checks")
    results: list[CheckResult] = Field(
        default_factory=list,
        description="Full list of individual check results",
    )
    failures: list[CheckResult] = Field(
        default_factory=list,
        description="Subset of check results that failed",
    )


def _parse_iso_date(value: Any) -> date | None:
    """Parse string or datetime object into a datetime.date instance."""
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        clean = value.strip()
        if not clean:
            return None
        # ISO YYYY-MM-DD prefix
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", clean)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
        try:
            return datetime.fromisoformat(clean.replace("Z", "+00:00")).date()
        except ValueError:
            pass
    return None


def _extract_number_from_text(text: str) -> tuple[float | None, bool]:
    """Extract primary numeric value from text statement.

    Returns:
        (value, is_percentage) where is_percentage is True if followed by '%'.
    """
    clean = text.replace(",", "").strip()
    # Find numbers like $123.45, 29.8%, -31.8%, 0.224
    matches = list(re.finditer(r"(-?\d+(?:\.\d+)?)\s*(%?)", clean))
    if not matches:
        return None, False

    # Prefer the last or most prominent number in the statement
    for m in reversed(matches):
        num_str = m.group(1)
        pct_sign = m.group(2)
        try:
            val = float(num_str)
            return val, pct_sign == "%"
        except ValueError:
            continue
    return None, False


def _safe_eval_math(
    expression: str,
    variables: dict[str, Any] | None = None,
) -> float:
    """Safely evaluate mathematical expressions using Python AST.

    Restricted strictly to binary arithmetic, unary negation, math functions,
    and supplied numerical variables. Prevents arbitrary code execution.
    """
    clean_expr = expression.strip()
    # Strip optional assignments like 'result = ...'
    if "=" in clean_expr:
        clean_expr = clean_expr.split("=", 1)[1].strip()

    vars_dict = {k: float(v) for k, v in (variables or {}).items() if v is not None}

    allowed_operators: dict[type[ast.operator], Any] = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }

    allowed_unary_operators: dict[type[ast.unaryop], Any] = {
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: +a,
    }

    allowed_functions: dict[str, Any] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
    }

    parsed = ast.parse(clean_expr, mode="eval")

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type in math eval: {node.value!r}")
        elif isinstance(node, ast.Name):
            if node.id in vars_dict:
                return vars_dict[node.id]
            if node.id.lower() in vars_dict:
                return vars_dict[node.id.lower()]
            raise ValueError(f"Unknown variable in formula: '{node.id}'")
        elif isinstance(node, ast.BinOp):
            bin_op_type = type(node.op)
            if bin_op_type not in allowed_operators:
                raise ValueError(f"Unsupported operator: {bin_op_type}")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if bin_op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0.0:
                raise ZeroDivisionError("Division by zero in calculation")
            return float(allowed_operators[bin_op_type](left, right))
        elif isinstance(node, ast.UnaryOp):
            unary_op_type = type(node.op)
            if unary_op_type not in allowed_unary_operators:
                raise ValueError(f"Unsupported unary operator: {unary_op_type}")
            operand = _eval_node(node.operand)
            return float(allowed_unary_operators[unary_op_type](operand))
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only named function calls permitted in formula")
            func_name = node.func.id
            if func_name not in allowed_functions:
                raise ValueError(f"Disallowed function call in formula: '{func_name}'")
            args = [_eval_node(arg) for arg in node.args]
            return float(allowed_functions[func_name](*args))
        raise ValueError(f"Unsupported syntax tree element: {type(node).__name__}")

    return _eval_node(parsed)


# ===========================================================================
# Check (a): Calculation Claims Recomputation
# ===========================================================================


def check_calculation_recomputation(
    claim: Claim | dict[str, Any] | Any,
    *,
    tolerance: float = 0.02,
    inputs: dict[str, Any] | None = None,
    expected_value: float | None = None,
) -> CheckResult:
    """Deterministic Check (a): Recompute calculation claim and match within tolerance.

    Recomputes the mathematical calculation from stated inputs (via derived_by,
    evidence raw values, or explicit inputs) and validates whether the result
    matches the claimed expected value within relative or absolute tolerance.

    Args:
        claim: Claim or dictionary to verify.
        tolerance: Permitted difference margin (default 0.02, i.e. 2%).
        inputs: Optional dictionary of variable names and input numbers.
        expected_value: Optional expected number override.

    Returns:
        CheckResult indicating whether recomputation matched within tolerance.
    """
    claim_id: str | None = None
    claim_text: str = ""
    claim_type: str = "calculation"
    derived_by: str | None = None
    ev_items: list[Any] = []

    if isinstance(claim, dict):
        claim_id = claim.get("claim_id")
        claim_text = str(claim.get("text", ""))
        claim_type = str(claim.get("type", "calculation")).strip().lower()
        derived_by = claim.get("derived_by")
        ev_items = list(claim.get("evidence", []) or [])
        if expected_value is None and "expected_value" in claim:
            expected_value = float(claim["expected_value"])
    else:
        claim_id = getattr(claim, "claim_id", None)
        claim_text = str(getattr(claim, "text", ""))
        raw_type = getattr(claim, "type", "calculation")
        claim_type = (
            raw_type.value
            if hasattr(raw_type, "value")
            else str(raw_type).strip().lower()
        )
        derived_by = getattr(claim, "derived_by", None)
        ev_items = list(getattr(claim, "evidence", []) or [])
        if expected_value is None and hasattr(claim, "expected_value"):
            exp_attr = getattr(claim, "expected_value", None)
            if exp_attr is not None:
                expected_value = float(exp_attr)

    if claim_type not in ("calculation", ClaimType.CALCULATION.value):
        return CheckResult(
            check_name="calculation_recomputation",
            passed=True,
            status=CheckStatus.PASSED,
            claim_id=claim_id,
            message=f"Claim '{claim_id}' is not a calculation claim; check skipped.",
        )

    # 1. Gather variables from explicit inputs and evidence raw values
    combined_inputs: dict[str, Any] = {}
    if inputs:
        combined_inputs.update(inputs)

    formula_candidate = derived_by

    for ev in ev_items:
        raw_val = (
            ev.raw_value
            if isinstance(ev, Evidence)
            else (
                ev.get("raw_value")
                if isinstance(ev, dict)
                else getattr(ev, "raw_value", None)
            )
        )
        if isinstance(raw_val, dict):
            if "inputs" in raw_val and isinstance(raw_val["inputs"], dict):
                combined_inputs.update(raw_val["inputs"])
            else:
                combined_inputs.update(
                    {
                        k: v
                        for k, v in raw_val.items()
                        if isinstance(v, (int, float, str))
                    }
                )
            if not formula_candidate and "formula" in raw_val:
                formula_candidate = str(raw_val["formula"])
            if expected_value is None and "value" in raw_val:
                try:
                    expected_value = float(raw_val["value"])
                except (ValueError, TypeError):
                    pass

    # 2. Determine expected value if not yet specified
    is_pct_text = False
    if expected_value is None:
        extracted, is_pct_text = _extract_number_from_text(claim_text)
        expected_value = extracted

    if expected_value is None:
        return CheckResult(
            check_name="calculation_recomputation",
            passed=False,
            status=CheckStatus.FAILED,
            claim_id=claim_id,
            message=(
                f"Calculation claim '{claim_id}' lacks a determinable expected "
                "numeric value."
            ),
            details={"text": claim_text},
        )

    # 3. Evaluate formula via safe quant math evaluator
    if not formula_candidate:
        return CheckResult(
            check_name="calculation_recomputation",
            passed=False,
            status=CheckStatus.FAILED,
            claim_id=claim_id,
            message=(
                f"Calculation claim '{claim_id}' provides no 'derived_by' formula "
                "or calculation rule to recompute."
            ),
        )

    try:
        computed_val = _safe_eval_math(formula_candidate, combined_inputs)
    except Exception as exc:
        return CheckResult(
            check_name="calculation_recomputation",
            passed=False,
            status=CheckStatus.FAILED,
            claim_id=claim_id,
            message=(f"Calculation recomputation failed for claim '{claim_id}': {exc}"),
            details={"formula": formula_candidate, "inputs": combined_inputs},
        )

    # 4. Compare with tolerance
    # Support both raw decimal match (0.298 vs 0.298) and percentage match
    # (0.298 vs 29.8%)
    diff_direct = abs(computed_val - expected_value)
    diff_pct = abs((computed_val * 100.0) - expected_value)
    diff_dec = abs(computed_val - (expected_value / 100.0))

    rel_diff_direct = (
        diff_direct / max(abs(expected_value), 1e-9)
        if expected_value != 0.0
        else diff_direct
    )

    matches_direct = diff_direct <= tolerance or rel_diff_direct <= tolerance
    matches_pct = (diff_pct <= tolerance * 100.0) or (diff_dec <= tolerance)

    is_match = matches_direct or (is_pct_text and matches_pct) or matches_pct

    if is_match:
        return CheckResult(
            check_name="calculation_recomputation",
            passed=True,
            status=CheckStatus.PASSED,
            claim_id=claim_id,
            message=(
                f"Calculation claim '{claim_id}' successfully recomputed "
                f"({computed_val:.4f}) and matches expected ({expected_value}) "
                f"within tolerance {tolerance}."
            ),
            details={
                "computed": computed_val,
                "expected": expected_value,
                "formula": formula_candidate,
            },
        )

    return CheckResult(
        check_name="calculation_recomputation",
        passed=False,
        status=CheckStatus.FAILED,
        claim_id=claim_id,
        message=(
            f"Calculation claim '{claim_id}' recomputed value ({computed_val:.4f}) "
            f"deviates from stated value ({expected_value}) beyond tolerance "
            f"{tolerance}."
        ),
        details={
            "computed": computed_val,
            "expected": expected_value,
            "difference": diff_direct,
            "formula": formula_candidate,
        },
    )


# ===========================================================================
# Check (b): Interpretation Claims Have Existing Supporting Claim IDs
# ===========================================================================


def check_interpretation_supporting_ids(
    claim: Claim | dict[str, Any] | Any,
    existing_claim_ids: set[str] | Sequence[str],
) -> CheckResult:
    """Deterministic Check (b): Interpretation claims cite valid supporting claim IDs.

    Verifies that:
    1. Interpretation claims have a non-empty depends_on list.
    2. Every prerequisite claim ID in depends_on exists in the provided set of claims.

    Args:
        claim: Claim or dictionary to verify.
        existing_claim_ids: Set or list of known valid claim IDs.

    Returns:
        CheckResult indicating pass or failure.
    """
    claim_id: str | None = None
    claim_type: str = "interpretation"
    depends_on: list[str] = []

    if isinstance(claim, dict):
        claim_id = claim.get("claim_id")
        claim_type = str(claim.get("type", "interpretation")).strip().lower()
        depends_on = list(claim.get("depends_on", []) or [])
    else:
        claim_id = getattr(claim, "claim_id", None)
        raw_type = getattr(claim, "type", "interpretation")
        claim_type = (
            raw_type.value
            if hasattr(raw_type, "value")
            else str(raw_type).strip().lower()
        )
        depends_on = list(getattr(claim, "depends_on", []) or [])

    if claim_type not in ("interpretation", ClaimType.INTERPRETATION.value):
        return CheckResult(
            check_name="interpretation_supporting_ids",
            passed=True,
            status=CheckStatus.PASSED,
            claim_id=claim_id,
            message=(
                f"Claim '{claim_id}' is not an interpretation claim; check skipped."
            ),
        )

    if not depends_on:
        return CheckResult(
            check_name="interpretation_supporting_ids",
            passed=False,
            status=CheckStatus.FAILED,
            claim_id=claim_id,
            message=(
                f"Interpretation claim '{claim_id}' has empty 'depends_on' list "
                "and lacks required supporting claim IDs."
            ),
        )

    valid_set = set(existing_claim_ids)
    missing_ids = [pid for pid in depends_on if pid not in valid_set]

    if missing_ids:
        return CheckResult(
            check_name="interpretation_supporting_ids",
            passed=False,
            status=CheckStatus.FAILED,
            claim_id=claim_id,
            message=(
                f"Interpretation claim '{claim_id}' references non-existent "
                f"supporting claim IDs: {missing_ids}."
            ),
            details={"missing_ids": missing_ids, "depends_on": depends_on},
        )

    return CheckResult(
        check_name="interpretation_supporting_ids",
        passed=True,
        status=CheckStatus.PASSED,
        claim_id=claim_id,
        message=(
            f"Interpretation claim '{claim_id}' cites valid supporting claim IDs: "
            f"{depends_on}."
        ),
        details={"depends_on": depends_on},
    )


# ===========================================================================
# Check (c): No Evidence Date Exceeds As-Of Date
# ===========================================================================


def check_evidence_dates_cutoff(
    claim: Claim | dict[str, Any] | Any,
    as_of_date: str | date,
) -> CheckResult:
    """Deterministic Check (c): No evidence date exceeds point-in-time as_of_date.

    Validates that every cited Evidence object linked to the claim has an
    as_of_date or retrieved_at on or before the cutoff date, preventing lookahead bias.

    Args:
        claim: Claim or dictionary to evaluate.
        as_of_date: Cutoff cutoff point-in-time date.

    Returns:
        CheckResult indicating pass or failure.
    """
    cutoff = _parse_iso_date(as_of_date)
    if cutoff is None:
        raise ValueError(
            f"Invalid as_of_date '{as_of_date}'; must be valid ISO date string or date"
        )

    claim_id: str | None = None
    ev_items: list[Any] = []

    if isinstance(claim, dict):
        claim_id = claim.get("claim_id")
        ev_items = list(claim.get("evidence", []) or [])
    else:
        claim_id = getattr(claim, "claim_id", None)
        ev_items = list(getattr(claim, "evidence", []) or [])

    if not ev_items:
        return CheckResult(
            check_name="evidence_dates_cutoff",
            passed=True,
            status=CheckStatus.PASSED,
            claim_id=claim_id,
            message=f"Claim '{claim_id}' has no evidence dates to audit.",
        )

    future_dates: list[dict[str, Any]] = []

    for i, ev in enumerate(ev_items, start=1):
        ev_as_of: Any = None
        ev_retrieved: Any = None
        if isinstance(ev, Evidence):
            ev_as_of = ev.as_of_date
            ev_retrieved = ev.retrieved_at
        elif isinstance(ev, dict):
            ev_as_of = ev.get("as_of_date")
            ev_retrieved = ev.get("retrieved_at")
        else:
            ev_as_of = getattr(ev, "as_of_date", None)
            ev_retrieved = getattr(ev, "retrieved_at", None)

        parsed_ev_date = _parse_iso_date(ev_as_of)
        if parsed_ev_date and parsed_ev_date > cutoff:
            future_dates.append(
                {
                    "evidence_index": i,
                    "field": "as_of_date",
                    "date": parsed_ev_date.isoformat(),
                }
            )

        parsed_retrieved = _parse_iso_date(ev_retrieved)
        if parsed_retrieved and parsed_retrieved > cutoff:
            future_dates.append(
                {
                    "evidence_index": i,
                    "field": "retrieved_at",
                    "date": parsed_retrieved.isoformat(),
                }
            )

    if future_dates:
        return CheckResult(
            check_name="evidence_dates_cutoff",
            passed=False,
            status=CheckStatus.FAILED,
            claim_id=claim_id,
            message=(
                f"Claim '{claim_id}' cites evidence dated after the cutoff "
                f"as_of_date '{cutoff.isoformat()}' (lookahead bias detected)."
            ),
            details={"future_dates": future_dates, "cutoff": cutoff.isoformat()},
        )

    return CheckResult(
        check_name="evidence_dates_cutoff",
        passed=True,
        status=CheckStatus.PASSED,
        claim_id=claim_id,
        message=(
            f"All evidence dates for claim '{claim_id}' are on or before "
            f"as_of_date cutoff '{cutoff.isoformat()}'."
        ),
    )


# ===========================================================================
# Check (d): Background Claims Are Not the Sole Support of an Interpretation
# ===========================================================================


def check_background_not_sole_support(
    claim: Claim | dict[str, Any] | Any,
    claims_by_id: dict[str, Any],
) -> CheckResult:
    """Deterministic Check (d): Background claims cannot solely support interpretation.

    Ensures that an interpretation claim is grounded in at least one empirical
    fact or calculation claim, rather than resting solely on ungrounded
    general background statements.

    Args:
        claim: Claim or dictionary to evaluate.
        claims_by_id: Dictionary mapping claim_id to Claim or claim dict.

    Returns:
        CheckResult indicating pass or failure.
    """
    claim_id: str | None = None
    claim_type: str = "interpretation"
    depends_on: list[str] = []

    if isinstance(claim, dict):
        claim_id = claim.get("claim_id")
        claim_type = str(claim.get("type", "interpretation")).strip().lower()
        depends_on = list(claim.get("depends_on", []) or [])
    else:
        claim_id = getattr(claim, "claim_id", None)
        raw_type = getattr(claim, "type", "interpretation")
        claim_type = (
            raw_type.value
            if hasattr(raw_type, "value")
            else str(raw_type).strip().lower()
        )
        depends_on = list(getattr(claim, "depends_on", []) or [])

    if claim_type not in ("interpretation", ClaimType.INTERPRETATION.value):
        return CheckResult(
            check_name="background_not_sole_support",
            passed=True,
            status=CheckStatus.PASSED,
            claim_id=claim_id,
            message=(
                f"Claim '{claim_id}' is not an interpretation claim; check skipped."
            ),
        )

    if not depends_on:
        return CheckResult(
            check_name="background_not_sole_support",
            passed=False,
            status=CheckStatus.FAILED,
            claim_id=claim_id,
            message=(
                f"Interpretation claim '{claim_id}' has empty depends_on list "
                "and no supporting claims."
            ),
        )

    prereq_types: list[str] = []
    for pid in depends_on:
        if pid in claims_by_id:
            parent_claim = claims_by_id[pid]
            if isinstance(parent_claim, dict):
                p_type = str(parent_claim.get("type", "fact")).strip().lower()
            else:
                raw_pt = getattr(parent_claim, "type", "fact")
                p_type = (
                    raw_pt.value
                    if hasattr(raw_pt, "value")
                    else str(raw_pt).strip().lower()
                )
            prereq_types.append(p_type)

    if not prereq_types:
        return CheckResult(
            check_name="background_not_sole_support",
            passed=False,
            status=CheckStatus.FAILED,
            claim_id=claim_id,
            message=(
                f"Interpretation claim '{claim_id}' references supporting claim IDs "
                "that could not be resolved in the provided claim repository."
            ),
        )

    # Check if all resolved prerequisite claims are solely 'background'
    all_background = all(
        t in ("background", ClaimType.BACKGROUND.value) for t in prereq_types
    )

    if all_background:
        return CheckResult(
            check_name="background_not_sole_support",
            passed=False,
            status=CheckStatus.FAILED,
            claim_id=claim_id,
            message=(
                f"Interpretation claim '{claim_id}' is solely supported by "
                f"background claims ({depends_on}). It must be grounded in at least "
                "one empirical fact or calculation."
            ),
            details={"depends_on": depends_on, "prereq_types": prereq_types},
        )

    return CheckResult(
        check_name="background_not_sole_support",
        passed=True,
        status=CheckStatus.PASSED,
        claim_id=claim_id,
        message=(
            f"Interpretation claim '{claim_id}' is grounded in substantive "
            f"non-background evidence (prerequisite types: {set(prereq_types)})."
        ),
        details={"prereq_types": prereq_types},
    )


# ===========================================================================
# Comprehensive Deterministic Suite Runner
# ===========================================================================


def verify_deterministic_checks(
    claims: Sequence[Claim | dict[str, Any] | Any],
    *,
    as_of_date: str | date | None = None,
    tolerance: float = 0.02,
    existing_claims: Sequence[Claim | dict[str, Any] | Any] | None = None,
) -> ChecksReport:
    """Run all deterministic checks across a collection of financial claims.

    Audits:
    (a) Calculation claims recomputed from inputs match within tolerance.
    (b) Interpretation claims cite existing supporting claim IDs.
    (c) No evidence date exceeds as_of_date cutoff.
    (d) Background claims are not the sole support of an interpretation.

    Args:
        claims: Claims to audit.
        as_of_date: Cutoff date cutoff (if provided, audits evidence dates).
        tolerance: Numerical tolerance for calculation recomputation.
        existing_claims: Optional broader set of known claims in run state/ledger.

    Returns:
        ChecksReport summarizing all passes and failures.
    """
    all_claims_list = list(existing_claims or []) + list(claims)
    claims_by_id: dict[str, Any] = {}
    for c in all_claims_list:
        cid = c.get("claim_id") if isinstance(c, dict) else getattr(c, "claim_id", None)
        if cid:
            claims_by_id[str(cid)] = c

    known_ids = set(claims_by_id.keys())
    results: list[CheckResult] = []

    for c in claims:
        raw_type = c.get("type") if isinstance(c, dict) else getattr(c, "type", None)
        c_type = (
            str(raw_type.value)
            if raw_type is not None and hasattr(raw_type, "value")
            else str(raw_type or "").strip().lower()
        )

        # (a) Calculation check
        if c_type in ("calculation", ClaimType.CALCULATION.value):
            res_a = check_calculation_recomputation(c, tolerance=tolerance)
            results.append(res_a)

        # (b) Interpretation supporting IDs
        if c_type in ("interpretation", ClaimType.INTERPRETATION.value):
            res_b = check_interpretation_supporting_ids(c, existing_claim_ids=known_ids)
            results.append(res_b)

        # (c) Evidence dates cutoff
        if as_of_date is not None:
            res_c = check_evidence_dates_cutoff(c, as_of_date=as_of_date)
            results.append(res_c)

        # (d) Background not sole support
        if c_type in ("interpretation", ClaimType.INTERPRETATION.value):
            res_d = check_background_not_sole_support(c, claims_by_id=claims_by_id)
            results.append(res_d)

    failures = [r for r in results if not r.passed]
    passed_count = len(results) - len(failures)

    return ChecksReport(
        all_passed=len(failures) == 0,
        passed_count=passed_count,
        failed_count=len(failures),
        results=results,
        failures=failures,
    )
