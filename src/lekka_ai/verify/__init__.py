"""Verification package auditing claim grounding, consistency, and validity."""

from lekka_ai.verify.support import (
    BatchSupportResponse,
    ClaimSupportCheck,
    SupportVerdict,
    fetch_claim_evidence_passage,
    verify_claim_support,
    verify_support,
)

__all__ = [
    "BatchSupportResponse",
    "ClaimSupportCheck",
    "SupportVerdict",
    "fetch_claim_evidence_passage",
    "verify_claim_support",
    "verify_support",
]
