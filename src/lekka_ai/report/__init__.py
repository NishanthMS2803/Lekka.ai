"""Report generation package for synthesized equity research reports."""

from lekka_ai.report.confidence import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    ConfidenceBreakdown,
    calculate_confidence,
    generate_abstention_report,
    synthesize_report_with_confidence,
)
from lekka_ai.report.synthesize import (
    REQUIRED_SECTIONS,
    ReportValidationError,
    ReportValidationResult,
    SynthesizedReport,
    ValidationCheck,
    synthesize_report,
    validate_report,
)

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "ConfidenceBreakdown",
    "REQUIRED_SECTIONS",
    "ReportValidationError",
    "ReportValidationResult",
    "SynthesizedReport",
    "ValidationCheck",
    "calculate_confidence",
    "generate_abstention_report",
    "synthesize_report",
    "synthesize_report_with_confidence",
    "validate_report",
]
