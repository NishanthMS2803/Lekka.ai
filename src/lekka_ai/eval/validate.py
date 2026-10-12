"""Dataset schema validator script for evaluation benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.schemas import AdversarialExample, RiskRecallExample
from lekka_ai.logging import get_logger

logger = get_logger(__name__)


class DatasetValidationError(Exception):
    """Raised when one or more datasets fail validation checks."""


def validate_single_row(
    row: dict[str, Any],
    file_name: str,
    line_num: int,
) -> tuple[bool, str | None]:
    """Validate a single JSON row dictionary against the appropriate schema.

    Parameters:
        row: Raw dictionary parsed from a JSONL line.
        file_name: Base filename of the dataset (e.g. 'risk_recall_v0.jsonl').
        line_num: Line number in the file for error reporting.

    Returns:
        Tuple of (is_valid, error_message).
    """
    lower_name = file_name.lower()
    try:
        if "risk_recall" in lower_name:
            RiskRecallExample.model_validate(row)
        elif "adversarial" in lower_name:
            AdversarialExample.model_validate(row)
        else:
            EvalExample.model_validate(row)
        return True, None
    except ValidationError as err:
        err_details = "; ".join(f"{e['loc']}: {e['msg']}" for e in err.errors())
        msg = f"Line {line_num}: Validation failed -> {err_details}"
        return False, msg
    except Exception as exc:
        msg = f"Line {line_num}: Unexpected error -> {exc}"
        return False, msg


def validate_dataset_file(file_path: Path) -> tuple[bool, int, list[str]]:
    """Validate an entire JSONL dataset file.

    Parameters:
        file_path: Path to the target .jsonl file.

    Returns:
        Tuple of (is_valid, total_rows, errors_list).
    """
    errors: list[str] = []
    seen_ids: set[str] = set()
    total_rows = 0

    if not file_path.exists():
        return False, 0, [f"File does not exist: {file_path}"]

    if file_path.stat().st_size == 0:
        return False, 0, [f"File is completely empty: {file_path}"]

    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue

            total_rows += 1
            try:
                data = json.loads(clean)
            except json.JSONDecodeError as json_err:
                errors.append(f"Line {line_num}: Invalid JSON syntax ({json_err})")
                continue

            row_id = data.get("id")
            if not row_id:
                errors.append(f"Line {line_num}: Missing required 'id' attribute")
            elif row_id in seen_ids:
                errors.append(f"Line {line_num}: Duplicate ID '{row_id}' found")
            else:
                seen_ids.add(row_id)

            ok, err_msg = validate_single_row(data, file_path.name, line_num)
            if not ok and err_msg:
                errors.append(err_msg)

    if total_rows == 0:
        errors.append("Dataset contains 0 valid data rows.")

    is_valid = len(errors) == 0
    return is_valid, total_rows, errors


def validate_all_datasets(
    dataset_dirs: list[Path] | None = None,
) -> tuple[bool, dict[str, tuple[bool, int, list[str]]]]:
    """Validate all JSONL dataset files in specified directories.

    Parameters:
        dataset_dirs: List of directory paths to scan for .jsonl datasets.

    Returns:
        Tuple of (overall_success, results_mapping).
    """
    dirs = dataset_dirs or [
        Path("eval/datasets"),
        Path("src/lekka_ai/eval/datasets"),
    ]
    results: dict[str, tuple[bool, int, list[str]]] = {}
    overall_success = True

    for directory in dirs:
        if not directory.exists():
            continue
        for jsonl_file in sorted(directory.glob("*.jsonl")):
            rel_path = str(jsonl_file)
            is_valid, count, errors = validate_dataset_file(jsonl_file)
            results[rel_path] = (is_valid, count, errors)
            if not is_valid:
                overall_success = False

    return overall_success, results


def main() -> int:
    """CLI entrypoint for dataset schema validation."""
    parser = argparse.ArgumentParser(
        description="Validate benchmark dataset JSONL schemas and constraints."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Optional specific directory to validate datasets within",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional specific single .jsonl file to validate",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(" Lekka AI Benchmark Dataset Schema Validation")
    print("=" * 80)

    if args.file:
        target = Path(args.file)
        ok, count, errors = validate_dataset_file(target)
        results = {str(target): (ok, count, errors)}
        overall_ok = ok
    else:
        dirs = [Path(args.dir)] if args.dir else None
        overall_ok, results = validate_all_datasets(dirs)

    if not results:
        print("No .jsonl dataset files discovered.")
        return 1

    for path_str, (ok, count, errors) in results.items():
        status = "[PASS]" if ok else "[FAIL]"
        print(f"\n{status} {path_str} ({count} rows)")
        if not ok:
            for e in errors[:10]:
                print(f"       ERROR: {e}")
            if len(errors) > 10:
                print(f"       ... and {len(errors) - 10} more errors")

    print("\n" + "-" * 80)
    if overall_ok:
        total_files = len(results)
        total_rows = sum(c for _, c, _ in results.values())
        print(
            f"SUCCESS: All {total_files} dataset files ({total_rows} total rows) "
            "passed schema validation."
        )
        return 0

    print("FAILED: One or more datasets failed validation.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
